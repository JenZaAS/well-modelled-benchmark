"""Check that a model meets the benchmark contract on tiny synthetic tasks (no data needed).

    python -m benchmark.check_model template_linear
    python -m benchmark.check_model my_package.my_module:MyModel

Exit code 0 prints a short OK summary; any contract violation prints the error and exits 1.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import numpy as np

from benchmark import make_tasks as mt
from benchmark.registry import REPO, discover
from wmb import synthetic
from wmb.contender import Prediction
from wmb.contract import TARGET_CURVES, Task, validate_prediction
from wmb.model_spec import YAML_NAME, ModelSpec, load_model_spec, yaml_candidates

CHECK_SPEC = {"dataset": "synthetic", "seed": 2020, "gap_lengths_m": [10]}

LICENSE_NAMES = ("LICENSE.md", "LICENSE", "LICENSE.txt", "LICENCE.md", "LICENCE", "LICENCE.txt")


def _yaml_class_mismatch(model, spec: ModelSpec) -> list[str]:
    out = []
    declared = set(spec.targets or {})
    unsupported = sorted(t for t in declared if not model.supports(t))
    if unsupported:
        out.append(f"{spec.path.name} lists targets the class does not support: {', '.join(unsupported)}")
    undeclared = sorted(t for t in TARGET_CURVES if model.supports(t) and t not in declared)
    if undeclared:
        out.append(f"the class supports {', '.join(undeclared)} but {spec.path.name} has no entry for it")
    return out


def _class_folder(model) -> Path | None:
    """Directory of the model's wrapper class file."""
    cls = model if isinstance(model, type) else type(model)
    path = None
    try:
        path = inspect.getfile(cls)
    except TypeError:
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        path = getattr(mod, "__file__", None)
    if path and not str(path).startswith("<"):
        return Path(path).resolve().parent
    return None


def _rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def _licence_in(folder: Path | None) -> Path | None:
    if folder is None:
        return None
    for name in LICENSE_NAMES:
        p = folder / name
        if p.is_file():
            return p
    return None


def model_folder(model) -> Path | None:
    """Folder that should hold LICENSE.md: the wrapper / model.yaml directory, e.g. workstreams/<team>/models/<name>/."""
    return _class_folder(model)


def workstream_folder(model) -> Path | None:
    """Team folder workstreams/<workstream>, inferred from the class path when possible."""
    folder = _class_folder(model)
    if folder is not None:
        parts = folder.parts
        if "workstreams" in parts:
            i = parts.index("workstreams")
            if i + 1 < len(parts):
                return Path(*parts[: i + 2])
    ws = getattr(model, "workstream", None)
    if ws:
        candidate = REPO / "workstreams" / str(ws)
        if candidate.is_dir():
            return candidate
    return None


def licence_warnings(model) -> list[str]:
    """Licence-file notes (warnings only, never a hard fail). Prefer the model folder; the team folder is a soft fallback."""
    model_dir = model_folder(model)
    team_dir = workstream_folder(model)
    hint = "add one with the licence name, URL and a one-line summary of restrictions"
    if _licence_in(model_dir):
        return []
    same = model_dir is not None and team_dir is not None and model_dir.resolve() == team_dir.resolve()
    if not same and _licence_in(team_dir):
        where = _rel(model_dir) if model_dir is not None else "the model folder (next to the wrapper / model.yaml)"
        return [f"LICENSE.md is in {_rel(team_dir)}, not the model folder {where}; "
                f"move or copy it next to the model ({hint})"]
    if model_dir is not None:
        return [f"{_rel(model_dir)} has no LICENSE.md, LICENSE or LICENSE.txt; {hint}"]
    return [f"could not find the model folder; add LICENSE.md next to the wrapper / model.yaml; {hint}"]


def missing_licence_file(model) -> str | None:
    """First licence-file warning, or None if the model folder already has a licence file."""
    notes = licence_warnings(model)
    return notes[0] if notes else None



def resolve(name: str):
    """A registry name, or `module:Class` for a model outside workstreams/."""
    if ":" in name:
        mod, cls = name.split(":", 1)
        return getattr(importlib.import_module(mod), cls)
    found = discover()
    if name not in found:
        raise ValueError(f"unknown model '{name}'. Known: {', '.join(sorted(found))}")
    return found[name]


def _blanked(task: Task, fraction: float = 1.0) -> Task:
    """Copy of the task with the first `fraction` of the input values set to NaN."""
    frame = task.inputs.copy()
    rows = np.arange(len(frame)) < int(len(frame) * fraction)
    for c in frame.columns:
        if c != "DEPTH":
            frame.loc[rows, c] = np.nan
    return Task(task.well_id, task.depth, frame, task.target, task.predict_mask, task.scenario, task.gap_length_m,
                task.target_observed, task.dataset, task.split, dict(task.meta))


def check_contender(factory) -> tuple[str, list[str]]:
    """Run the checks; returns (summary, warnings). Raises on a contract violation."""
    warnings: list[str] = []
    model = factory()
    spec = load_model_spec(model)
    if spec.errors:
        where = spec.path.name if spec.path else YAML_NAME
        raise ValueError(f"{where} is invalid: " + "; ".join(spec.errors))
    warnings += [f"{spec.path.name}: {w}" for w in spec.warnings] if spec.path else []
    if spec.source == "yaml":
        warnings += _yaml_class_mismatch(model, spec)
    else:
        others = [p for p in yaml_candidates(model) if p.is_file()]
        warnings.append(
            f"{others[0].name} describes a different model; add a model.yaml (or {model.name}.model.yaml) for {model.name}; class metadata was used instead"
            if others else
            "no model.yaml: add workstreams/<workstream>/model.yaml (name, licence, targets with required/optional curves); "
            "class metadata (targets, required_inputs, optional_inputs, licence_name) was used instead")
        if not (getattr(model, "required_inputs", None) or getattr(model, "inputs", None)):
            warnings.append("required_inputs is empty: declare the curves your model needs (and optional_inputs)")
    warnings += licence_warnings(model)
    if not getattr(model, "licence_name", None):
        warnings.append("licence_name is empty: set licence_name, licence_url and licence_restrictions on the class")
    targets = [t for t in TARGET_CURVES if model.supports(t)]
    if not targets:
        raise ValueError(f"{model.name} supports none of {', '.join(TARGET_CURVES)}")
    wells = synthetic.make_wells(n_wells=4, n_samples=500, seed=1)
    train, test = wells[:3], wells[3:]
    n_checked = 0
    for target in targets:
        model = factory()
        model.fit(train, target)
        tasks, _ = mt.build_tasks({**CHECK_SPEC, "targets": [target]}, test)
        tasks += [_blanked(t) for t in tasks] + [_blanked(t, 0.3) for t in tasks]
        for task in tasks:
            pred = model.predict_task(task)
            if not isinstance(pred, Prediction):
                raise ValueError(f"{target}/{task.scenario_key}: predict must return a Prediction, got {type(pred).__name__}")
            try:
                ok = validate_prediction(task, pred)
            except ValueError as exc:
                raise ValueError(f"{target}/{task.scenario_key}: {exc}") from exc
            if not np.isfinite(ok.p50).any() and np.isfinite(task.inputs.drop(columns="DEPTH").to_numpy()).any():
                warnings.append(f"{target}/{task.scenario_key}: no finite p50 although inputs are present")
            n_checked += 1
        batch = model.predict_batch(tasks[:2])
        if len(batch) != 2:
            raise ValueError(f"predict_batch returned {len(batch)} predictions for 2 tasks")
    return (f"OK {model.name}: targets {', '.join(targets)}, {n_checked} predictions valid "
            "(missing_curve and gap, full and NaN inputs)"), warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a model against the benchmark contract")
    ap.add_argument("name", help="contender name from the registry, or module:Class")
    a = ap.parse_args(argv)
    try:
        summary, warnings = check_contender(resolve(a.name))
    except Exception as exc:
        print(f"FAILED {a.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for w in dict.fromkeys(warnings):
        print(f"warning: {w}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
