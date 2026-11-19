"""Standardised benchmark test definitions: one YAML file in benchmark/test_defs/ = one (dataset, target, scenario, wells) run.

    from benchmark import test_def
    tdef = test_def.load_def("rhob_missing")
    errors, warnings = test_def.check_def(tdef)

Schema and workflow: docs/tests.md. Files whose stem starts with "_" or is TEMPLATE are examples, not runnable tests.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from benchmark import make_tasks as mt
from benchmark.registry import REPO
from wmb import paths, splits
from wmb.contract import SCENARIO_GAP, SCENARIO_MISSING, TARGET_CURVES, Task, scenario_key

DEFS_DIR = REPO / "benchmark" / "test_defs"
TEMPLATE_NAME = "_template.yaml"
VERSION = 1
SCENARIO_TYPES = (SCENARIO_MISSING, SCENARIO_GAP)
WELL_SPLITS = ("test", "leaderboard", "list")
INPUTS_POLICIES = ("model",)
ALWAYS_KNOWN_DATASETS = ("synthetic", "force2020")
DEFAULT_METRICS = ("MAE", "RMSE", "R2", "cover_90", "cover_50", "interval_width_90", "has_uncertainty",
                   "load_s", "fit_s", "predict_s", "samples_per_s")
# every key the runner writes into scores.json (see wmb.scoring.regression_scores and benchmark.run)
KNOWN_METRICS = frozenset(DEFAULT_METRICS) | {
    "n", "n_samples", "cover_1sigma", "cover_2sigma", "sharpness_mean_sigma", "CRPS", "pinball_05", "pinball_50",
    "pinball_95", "predict_s_batch", "predict_s_per_well", "predict_s_per_1000"}
TOP_KEYS = {"version", "name", "description", "dataset", "target", "scenario", "wells", "inputs_policy", "metrics"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class TestDefError(ValueError):
    """A test definition is invalid or missing."""

    __test__ = False  # not a pytest class


@dataclass(frozen=True)
class TestDef:
    __test__ = False  # not a pytest class

    name: str
    description: str
    dataset: str
    target: str
    scenario_type: str = SCENARIO_MISSING
    gap_length_m: float | None = None
    n_gaps: int = 1
    seed: int | None = None
    wells_split: str = "test"
    wells_names: tuple[str, ...] = ()
    inputs_policy: str = "model"
    metrics: tuple[str, ...] = DEFAULT_METRICS
    version: int = VERSION
    path: Path | None = field(default=None, compare=False)

    @property
    def scenario_key(self) -> str:
        return scenario_key(self.scenario_type, self.gap_length_m)

    @property
    def is_default_selection(self) -> bool:
        return self.wells_split == "test" and self.n_gaps == 1

    def as_dict(self) -> dict:
        scenario = {"type": self.scenario_type, "gap_length_m": self.gap_length_m, "n_gaps": self.n_gaps, "seed": self.seed}
        return {"version": self.version, "name": self.name, "description": self.description, "dataset": self.dataset,
                "target": self.target, "scenario": scenario,
                "wells": {"split": self.wells_split, "names": list(self.wells_names)},
                "inputs_policy": self.inputs_policy, "metrics": list(self.metrics)}


def generated_name(dataset: str, target: str, scenario_key: str) -> str:
    """Stable name for runs without a def: synthetic_rhob_missing, synthetic_rhob_gap_50."""
    return f"{dataset}_{target.lower()}_{'missing' if scenario_key == SCENARIO_MISSING else scenario_key}"


def _positive_length(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return 0 < float(value) < float("inf")
    except OverflowError:
        return False


def _dataset_known(name: str) -> bool:
    if name in ALWAYS_KNOWN_DATASETS or name in paths.DATASETS or mt.has_spec(name):
        return True
    try:
        return paths.get_dataset_path(name).is_dir()
    except (Exception, SystemExit):
        return False


def validate_def(data, *, path=None, dataset_wells=None) -> list[str]:
    """Schema and static errors of a parsed YAML mapping. `dataset_wells`: well ids of the dataset, to check wells.names."""
    if not isinstance(data, dict):
        return ["the file must be a YAML mapping"]
    errs: list[str] = []
    extra = sorted(set(data) - TOP_KEYS, key=str)
    if extra:
        errs.append(f"unknown keys: {', '.join(map(str, extra))}")
    if type(data.get("version")) is not int or data["version"] != VERSION:
        errs.append(f"version must be {VERSION}, got {data.get('version')!r}")
    name = data.get("name")
    if not isinstance(name, str) or not name:
        errs.append("name is missing")
    elif path is not None and name != Path(path).stem:
        errs.append(f"name '{name}' must equal the file stem '{Path(path).stem}'")
    if not isinstance(data.get("description"), str) or not data["description"].strip():
        errs.append("description is missing")
    ds = data.get("dataset")
    if not isinstance(ds, str) or not ds:
        errs.append("dataset is missing")
    elif not _dataset_known(ds):
        errs.append(f"unknown dataset '{ds}' (use synthetic, force2020, or a dataset with a task spec or data folder)")
    if data.get("target") not in TARGET_CURVES:
        errs.append(f"unknown target {data.get('target')!r}; choose from {', '.join(TARGET_CURVES)}")

    sc = data.get("scenario")
    if not isinstance(sc, dict):
        errs.append("scenario must be a mapping with a type")
    else:
        stype, length, n_gaps, seed = sc.get("type"), sc.get("gap_length_m"), sc.get("n_gaps", 1), sc.get("seed")
        if stype not in SCENARIO_TYPES:
            errs.append(f"unknown scenario.type {stype!r}; choose from {', '.join(SCENARIO_TYPES)}")
        elif stype == SCENARIO_GAP and not _positive_length(length):
            errs.append("scenario.type gap needs a positive, finite gap_length_m")
        elif stype == SCENARIO_MISSING and length is not None:
            errs.append("scenario.gap_length_m must be omitted or null for type missing_curve")
        if not isinstance(n_gaps, int) or isinstance(n_gaps, bool) or n_gaps < 1:
            errs.append(f"scenario.n_gaps must be an integer >= 1, got {n_gaps!r}")
        elif stype == SCENARIO_MISSING and n_gaps != 1:
            errs.append("scenario.n_gaps only applies to type gap")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            errs.append(f"scenario.seed must be an integer, got {seed!r}")
        extra = sorted(set(sc) - {"type", "gap_length_m", "n_gaps", "seed"}, key=str)
        if extra:
            errs.append(f"unknown scenario keys: {', '.join(map(str, extra))}")

    wells = data.get("wells")
    if not isinstance(wells, dict):
        errs.append("wells must be a mapping with a split")
    else:
        split, names = wells.get("split"), wells.get("names", [])
        if split not in WELL_SPLITS:
            errs.append(f"unknown wells.split {split!r}; choose from {', '.join(WELL_SPLITS)}")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            errs.append("wells.names must be a list of well ids")
        elif split == "list" and not names:
            errs.append("wells.split list needs a non-empty wells.names")
        elif dataset_wells is not None:
            unknown = sorted(set(names) - set(dataset_wells))
            if unknown:
                errs.append(f"wells.names not in the dataset: {', '.join(unknown[:10])}")
        extra = sorted(set(wells) - {"split", "names"}, key=str)
        if extra:
            errs.append(f"unknown wells keys: {', '.join(map(str, extra))}")
    if data.get("inputs_policy") not in INPUTS_POLICIES:
        errs.append(f"unknown inputs_policy {data.get('inputs_policy')!r}; only {', '.join(INPUTS_POLICIES)} is supported")
    metrics = data.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errs.append("metrics must be a non-empty list")
    else:
        bad = sorted({str(m) for m in metrics if not isinstance(m, str) or m not in KNOWN_METRICS})
        if bad:
            errs.append(f"unknown metrics: {', '.join(bad)}")
    return errs


def _from_valid(data: dict, path: Path | None) -> TestDef:
    sc, wells = data["scenario"], data["wells"]
    length = sc.get("gap_length_m")
    return TestDef(name=data["name"], description=" ".join(data["description"].split()), dataset=data["dataset"],
                   target=data["target"], scenario_type=sc["type"], gap_length_m=None if length is None else float(length),
                   n_gaps=int(sc.get("n_gaps", 1)), seed=sc.get("seed"), wells_split=wells["split"],
                   wells_names=tuple(wells.get("names") or ()), inputs_policy=data["inputs_policy"],
                   metrics=tuple(data["metrics"]), version=data["version"], path=path)


def read_yaml(path: Path) -> object:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def is_runnable(path: Path) -> bool:
    return path.suffix == ".yaml" and not path.stem.startswith("_") and path.stem.upper() != "TEMPLATE"


def def_paths(defs_dir: Path | None = None) -> list[Path]:
    """Runnable def files, sorted by name."""
    d = Path(defs_dir) if defs_dir else DEFS_DIR
    return sorted((p for p in d.glob("*.yaml") if is_runnable(p)), key=lambda p: p.stem)


def load_def(name_or_path, defs_dir: Path | None = None) -> TestDef:
    """Load and validate one def by name (benchmark/test_defs/<name>.yaml) or by file path."""
    s = str(name_or_path)
    p = Path(s)
    if not (p.suffix == ".yaml" and p.is_file()):
        p = (Path(defs_dir) if defs_dir else DEFS_DIR) / f"{s}.yaml"
    if not p.is_file():
        known = ", ".join(q.stem for q in def_paths(defs_dir))
        raise TestDefError(f"unknown test '{s}'. Known: {known}")
    if not is_runnable(p) or p.stem == "all":
        raise TestDefError(f"{p.name} is not a runnable test definition")
    try:
        data = read_yaml(p)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise TestDefError(f"cannot read {p.name}: {exc}") from exc
    errs = validate_def(data, path=p)
    if errs:
        raise TestDefError(f"{p.name} is invalid: " + "; ".join(errs))
    return _from_valid(data, p)


def list_defs(defs_dir: Path | None = None) -> list[TestDef]:
    """List valid runnable defs; report invalid files without hiding their valid siblings."""
    defs = []
    for p in def_paths(defs_dir):
        try:
            defs.append(load_def(p))
        except TestDefError as exc:
            print(f"warning: skipping test {p.stem}: {exc}", file=sys.stderr)
    return defs


def match_shipped(dataset: str, target: str, scenario_key: str, defs_dir: Path | None = None) -> TestDef | None:
    """A runnable def with exactly this dataset, target and scenario, default wells (split test) and one gap."""
    for p in def_paths(defs_dir):
        try:
            t = load_def(p)
        except TestDefError:
            continue
        if (t.dataset, t.target, t.scenario_key) == (dataset, target, scenario_key) and t.is_default_selection:
            if t.seed is None or (mt.has_spec(dataset) and t.seed == mt.load_spec(dataset)["seed"]):
                return t
    return None


def generated_def(dataset: str, target: str, scenario: str, *, defs_dir: Path | None = None, **opts) -> TestDef:
    """Def for the old --dataset/--target/--scenario CLI. `scenario` is a scenario key (missing_curve, gap_50).

    A shipped def that matches exactly is returned as is; otherwise one is generated with a stable name.
    Options: seed, wells_split, wells_names, n_gaps.
    """
    if not opts:
        shipped = match_shipped(dataset, target, scenario, defs_dir)
        if shipped is not None:
            return shipped
    gap = scenario != SCENARIO_MISSING
    length = float(scenario.split("_", 1)[1]) if gap else None
    name = generated_name(dataset, target, scenario)
    return TestDef(name=name, description=f"Generated from --dataset {dataset} --target {target} --scenario {scenario}.",
                   dataset=dataset, target=target, scenario_type=SCENARIO_GAP if gap else SCENARIO_MISSING,
                   gap_length_m=length, n_gaps=opts.get("n_gaps", 1), seed=opts.get("seed"),
                   wells_split=opts.get("wells_split", "test"), wells_names=tuple(opts.get("wells_names", ())))


def effective_seed(tdef: TestDef) -> int:
    """Def seed, else the dataset task-spec seed, else 0 (the runner's default well-split seed)."""
    if tdef.seed is not None:
        return int(tdef.seed)
    return int(mt.load_spec(tdef.dataset)["seed"]) if mt.has_spec(tdef.dataset) else 0


def select_wells(tdef: TestDef, train: list, test: list) -> list:
    """The test wells this def scores. Train wells always stay the official train set."""
    if tdef.wells_split == "test":
        return list(test)
    if tdef.wells_split == "leaderboard":
        chosen = [w for w in test if w.split == "leaderboard"]
        if not chosen:
            raise ValueError(f"dataset {tdef.dataset} has no wells with split 'leaderboard'; use wells.split test or list")
        return chosen
    by = {w.name: w for w in test}
    train_names = {w.name for w in train}
    bad_train = sorted(n for n in tdef.wells_names if n in train_names)
    if bad_train:
        raise ValueError(f"wells.names are training wells, not test wells: {', '.join(bad_train[:10])}")
    unknown = sorted(n for n in tdef.wells_names if n not in by)
    if unknown:
        raise ValueError(f"wells.names not in the dataset: {', '.join(unknown[:10])}")
    return [by[n] for n in dict.fromkeys(tdef.wells_names)]


def place_gaps(depth, y, length_m: float, n_gaps: int, seed: int, well_id: str, target: str) -> list[tuple[int, int]]:
    """n non-overlapping gaps (start, end exclusive), each placed by splits.place_gap with its own derived seed.

    Gap 0 uses the same seed as the standard tasks, so n_gaps=1 reproduces them. [] if they do not all fit.
    """
    for attempt in range(1 if n_gaps == 1 else 32):
        remaining = np.asarray(y, float).copy()
        out = []
        for k in range(n_gaps):
            label = well_id if attempt == k == 0 else f"{well_id}#gap{k}#{attempt}"
            g = splits.place_gap(depth, remaining, length_m, splits.gap_seed(seed, label, target, length_m))
            if g is None:
                break
            out.append(g)
            remaining[max(0, g[0] - 1):g[1] + 1] = np.nan  # keep one sample between gaps
        if len(out) == n_gaps:
            return sorted(out)
        if not out:
            break  # no first gap is possible, regardless of seed
    return []


def _usable(tdef: TestDef, well) -> bool:
    if not well.has(tdef.target):
        return False
    if tdef.scenario_type == SCENARIO_MISSING:
        return True
    depth, y = np.asarray(well.depth, float), np.asarray(well.curves[tdef.target], float)
    return bool(place_gaps(depth, y, tdef.gap_length_m, tdef.n_gaps, effective_seed(tdef), well.name, tdef.target))


def _gap_tasks(tdef: TestDef, wells: list) -> list[Task]:
    tasks = []
    seed = effective_seed(tdef)
    for well in sorted(wells, key=lambda w: w.name):
        if not well.has(tdef.target):
            continue
        depth, y = np.asarray(well.depth, float), np.asarray(well.curves[tdef.target], float)
        gaps = place_gaps(depth, y, tdef.gap_length_m, tdef.n_gaps, seed, well.name, tdef.target)
        if not gaps:
            continue
        mask = np.zeros(len(depth), bool)
        for a, b in gaps:
            mask[a:b] = True
        observed = y.copy()
        observed[mask] = np.nan
        tasks.append(Task(well_id=well.name, depth=depth, inputs=mt._inputs_frame(well, tdef.target), target=tdef.target,
                          predict_mask=mask, scenario=SCENARIO_GAP, gap_length_m=float(tdef.gap_length_m),
                          target_observed=observed, dataset=tdef.dataset, split=well.split))
    return tasks


def build_def_tasks(tdef: TestDef, train: list, test: list, cache=None) -> list[Task]:
    """Tasks of one def. Defs that match the dataset task spec reuse its cached tasks; others are built in memory."""
    wells = select_wells(tdef, train, test)
    key = tdef.scenario_key
    spec = mt.load_spec(tdef.dataset) if mt.has_spec(tdef.dataset) else None
    if (spec is not None and tdef.n_gaps == 1 and (tdef.seed is None or tdef.seed == spec["seed"])
            and tdef.target in spec["targets"]
            and (tdef.scenario_type == SCENARIO_MISSING or tdef.gap_length_m in spec["gap_lengths_m"])):
        tasks = mt.load_tasks(tdef.dataset, cache, key, tdef.target, wells=(train, test))
        keep = {w.name for w in wells}
        return [t for t in tasks if t.well_id in keep]
    if tdef.scenario_type == SCENARIO_GAP:
        return _gap_tasks(tdef, wells)
    spec = {"dataset": tdef.dataset, "seed": effective_seed(tdef), "targets": [tdef.target],
            "gap_lengths_m": []}
    return mt.build_tasks(spec, wells)[0]


def check_def(tdef: TestDef, *, wells=None) -> tuple[list[str], list[str]]:
    """(errors, warnings). `wells` = (train, test) to skip loading; if the dataset cannot be loaded, data checks are skipped."""
    errors = validate_def(tdef.as_dict(), path=tdef.path)
    warnings: list[str] = []
    if errors:
        return errors, warnings
    try:
        train, test = wells if wells is not None else _load_wells(tdef)
    except (Exception, SystemExit) as exc:
        return errors, [f"skipped well-presence check: {exc}"]
    known = {w.name for w in train} | {w.name for w in test}
    unknown = sorted(set(tdef.wells_names) - known)
    if unknown:
        errors.append(f"wells.names not in the dataset: {', '.join(unknown[:10])}")
        return errors, warnings
    try:
        chosen = select_wells(tdef, train, test)
    except ValueError as exc:
        return [str(exc)], warnings
    n = sum(_usable(tdef, w) for w in chosen)
    what = "finite samples" if tdef.scenario_type == SCENARIO_MISSING else f"a placeable {tdef.gap_length_m:g} m gap"
    if n == 0:
        errors.append(f"no test well has {tdef.target} with {what}")
    elif n < 3:
        warnings.append(f"only {n} test well(s) have {tdef.target} with {what}; scores will be noisy")
    return errors, warnings


def _load_wells(tdef: TestDef):
    from benchmark.run import load_dataset

    return load_dataset(tdef.dataset, effective_seed(tdef))
