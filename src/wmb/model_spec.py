"""Per-model input spec: `workstreams/<workstream>/model.yaml`, with class metadata as the fallback.

    name: template_linear
    licence: MIT
    targets:
      RHOB: {required: [DTC], optional: [GR, NPHI]}

`load_model_spec(model)` always returns a `ModelSpec`; problems found in a model.yaml are listed in
`spec.errors` / `spec.warnings` (an invalid yaml falls back to the class metadata so callers keep working).
`eligible(spec, target, curves_present)` is the one rule for "can this model run on this well".
See docs/models.md.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path

from wmb.curves import CORE

REPO = Path(__file__).resolve().parents[2]
YAML_NAME = "model.yaml"
TOP_KEYS = {"name", "licence", "targets"}
TARGET_KEYS = {"required", "optional"}
_CURVE_OK = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class TargetSpec:
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    name: str
    licence: str | None = None
    targets: dict[str, TargetSpec] | None = None     # None = any target (class fallback without `targets`)
    default: TargetSpec = field(default_factory=TargetSpec)   # inputs used when `targets` is None
    source: str = "class"                            # "yaml" | "class"
    path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def supports(self, target: str) -> bool:
        return self.targets is None or target in self.targets

    def for_target(self, target: str) -> TargetSpec | None:
        """Inputs for one target (never listing the target itself); None when unsupported."""
        if not self.supports(target):
            return None
        t = self.default if self.targets is None else self.targets[target]
        return TargetSpec([c for c in t.required if c != target], [c for c in t.optional if c != target])


def _names(value, where: str, errors: list[str], warnings: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{where} must be a list of curve names")
        return []
    out: list[str] = []
    for v in value:
        if v == "DEPTH":
            errors.append(f"{where}: DEPTH is always present; do not list it")
        elif not _CURVE_OK.fullmatch(v):
            errors.append(f"{where}: '{v}' is not a canonical curve name (upper case, e.g. GR, RHOB, DTC)")
        elif v not in CORE:
            warnings.append(f"{where}: '{v}' is not a core curve ({', '.join(CORE)})")
        if v in out:
            errors.append(f"{where}: '{v}' is listed twice")
        else:
            out.append(v)
    return out


def parse_spec(data, path: Path | None = None) -> ModelSpec:
    """Validate a parsed model.yaml mapping. Never raises; check `spec.errors`."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(data, dict):
        return ModelSpec("", source="yaml", path=path, errors=["model.yaml must be a mapping with name, licence and targets"])
    for k in sorted(set(data) - TOP_KEYS, key=str):
        errors.append(f"unknown key '{k}' (allowed: {', '.join(sorted(TOP_KEYS))})")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name is required (the contender name)")
        name = ""
    licence = data.get("licence")
    if licence is None or licence == "":
        warnings.append("licence is empty: add the licence name, e.g. MIT or Apache-2.0")
        licence = None
    elif not isinstance(licence, str):
        errors.append("licence must be a string")
        licence = None
    raw = data.get("targets")
    targets: dict[str, TargetSpec] = {}
    if not isinstance(raw, dict) or not raw:
        errors.append("targets must be a non-empty mapping, e.g. targets: {RHOB: {required: [DTC]}}")
    else:
        for target, body in raw.items():
            where = f"targets.{target}"
            if not isinstance(target, str) or not _CURVE_OK.fullmatch(target):
                errors.append(f"{where}: not a canonical curve name")
                continue
            if target not in CORE:
                warnings.append(f"{where}: '{target}' is not a core curve ({', '.join(CORE)})")
            if body is None:
                body = {}
            if not isinstance(body, dict):
                errors.append(f"{where} must be a mapping with required / optional lists")
                continue
            for k in sorted(set(body) - TARGET_KEYS, key=str):
                errors.append(f"{where}: unknown key '{k}' (allowed: required, optional)")
            req = _names(body.get("required"), f"{where}.required", errors, warnings)
            opt = _names(body.get("optional"), f"{where}.optional", errors, warnings)
            if target in req + opt:
                errors.append(f"{where}: the target cannot be one of its own inputs")
            both = sorted(set(req) & set(opt))
            if both:
                errors.append(f"{where}: {', '.join(both)} listed as both required and optional")
            targets[target] = TargetSpec(req, opt)
    return ModelSpec(name, licence, targets, source="yaml", path=path, errors=errors, warnings=warnings)


def _model_cls(model):
    return model if inspect.isclass(model) else type(model)


def yaml_candidates(model) -> list[Path]:
    """`<name>.model.yaml` then `model.yaml` in the workstream folder, else beside the class file."""
    cls = _model_cls(model)
    folders: list[Path] = []
    ws = getattr(cls, "workstream", None)
    if ws and (REPO / "workstreams" / str(ws)).is_dir():
        folders.append(REPO / "workstreams" / str(ws))
    try:
        file = inspect.getfile(cls)
        if file and not file.startswith("<"):
            folders.append(Path(file).resolve().parent)
    except (TypeError, OSError):
        pass
    name = getattr(cls, "name", "")
    seen, out = set(), []
    for folder in folders:
        for fname in ([f"{name}.model.yaml"] if name else []) + [YAML_NAME]:
            p = folder / fname
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def class_spec(model) -> ModelSpec:
    """Spec from class metadata: targets, required_inputs / inputs, optional_inputs, licence_name."""
    cls = _model_cls(model)
    req = [c for c in (getattr(cls, "required_inputs", None) or getattr(cls, "inputs", None) or []) if c != "DEPTH"]
    opt = [c for c in (getattr(cls, "optional_inputs", None) or []) if c != "DEPTH" and c not in req]
    targets = getattr(cls, "targets", None)
    body = TargetSpec(req, opt)
    return ModelSpec(getattr(cls, "name", cls.__name__), getattr(cls, "licence_name", None),
                     None if targets is None else {t: TargetSpec(list(req), list(opt)) for t in targets},
                     default=body, source="class")


def load_model_spec(model) -> ModelSpec:
    """model.yaml for this model if present, else the class metadata. `model` is an instance or a class."""
    import yaml

    fallback = class_spec(model)
    for path in yaml_candidates(model):
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            fallback.errors.append(f"{path.name} could not be read: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
            fallback.path = path
            return fallback
        spec = parse_spec(data, path)
        if spec.name and spec.name != fallback.name:
            if path.name == YAML_NAME:   # shared by several models in one folder: describes another one
                continue
            spec.errors.append(f"name '{spec.name}' does not match the model name '{fallback.name}'")
        if spec.errors:   # keep the problems, but the model stays usable through its class metadata
            fallback.errors, fallback.warnings, fallback.path = spec.errors, spec.warnings, path
            return fallback
        return spec
    return fallback


def eligible(spec: ModelSpec, target: str, curves_present) -> tuple[bool, str]:
    """(ok, reason): the model supports the target and every required input exists in the well."""
    t = spec.for_target(target)
    if t is None:
        return False, f"does not support target {target}"
    present = set(curves_present)
    missing = [c for c in t.required if c not in present]
    if missing:
        return False, f"missing required curve{'s' if len(missing) > 1 else ''}: {', '.join(missing)}"
    return True, ""


def inputs_used(spec: ModelSpec, target: str, curves_present) -> dict[str, list[str]]:
    """The declared inputs that exist in the well: {"required": [...], "optional": [...]}."""
    t = spec.for_target(target) or TargetSpec()
    present = set(curves_present)
    return {"required": [c for c in t.required if c in present], "optional": [c for c in t.optional if c in present]}
