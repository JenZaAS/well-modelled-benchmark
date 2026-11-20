"""Frozen experiment artifacts: the manifest, its compatibility checks, and loading one variant for the station.

`<work_root>/black_granite/manifest.json` records, per variant: the weight revision, the tuned head file and its
sha256, the interval calibrator, and the identity of everything the artifacts were built from (config hash,
inference-code hash, input policy, A-D split hash, gap-mask hashes, training-data hash).

`load_variant(name, train_wells)`:
    * FORCE 2020 training wells that match the manifest  -> the frozen artifacts
    * FORCE 2020 training wells that do not match         -> error (rerun the pipeline; never a silent refit)
    * any other wells (synthetic contract checks)         -> a small local fit on those wells: the pretrained
                                                             weights, at most a few head updates, in-sample bands.
                                                             FORCE-trained heads are never loaded for it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from workstreams.black_physics.granite import calibrate as CAL
from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite import gaps as G

INPUT_POLICY = "target_only: RHOB flanks from task.target_observed; no covariates, calendar or frequency token"
INFERENCE_FILES = ("config.py", "gaps.py", "backends.py", "reconstruct.py", "calibrate.py")
NAME_TO_VARIANT = {v: k for k, v in C.VARIANT_IDS.items()}


class StaleArtifacts(RuntimeError):
    """The official wells are given, but the frozen artifacts do not match them or this code."""


def code_hash() -> str:
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in INFERENCE_FILES:
        h.update(f.encode())
        h.update((here / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path() -> Path:
    return C.work_dir() / "manifest.json"


def read_manifest() -> dict | None:
    p = manifest_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def eligible_names(train_wells, target: str = C.TARGET) -> list[str]:
    return sorted(w.name for w in train_wells if w.has(target))


def official_identity(train_wells, cfg: C.ExperimentConfig = C.CONFIG) -> dict:
    """Split, gap masks and data hash recomputed from the given training wells (deterministic)."""
    by = {w.name: w for w in train_wells}
    groups = G.split_wells([w.name for w in train_wells], cfg.split_sizes, cfg.seed)
    gaps = index_all(by, groups, cfg)
    return {"split_hash": G.split_hash(groups), "data_hash": G.wells_data_hash(train_wells),
            "gaps_hash": {g: G.gaps_hash(v) for g, v in gaps.items()}, "groups": groups, "gaps": gaps}


def index_all(by: dict, groups: dict, cfg: C.ExperimentConfig = C.CONFIG, length_m: float | None = None) -> dict:
    out = {}
    for g in G.GROUPS:
        rows = []
        for name in groups[g]:
            w = by[name]
            if not w.has(C.TARGET):
                continue
            rows += G.index_gaps(name, w.depth, w.curves[C.TARGET], group=g, n_max=cfg.gaps_per_well[g],
                                 overlap=(g == "A"), seed=cfg.seed, length_m=length_m or cfg.gap_length_m)
        out[g] = rows
    return out


def identity_record(ident: dict) -> dict:
    return {"config_hash": C.config_hash(), "code_hash": code_hash(), "input_policy": INPUT_POLICY,
            "ttm": f"{C.TTM_REPO}@{C.TTM_REVISION}", "tspulse": f"{C.TSPULSE_REPO}@{C.TSPULSE_REVISION}",
            "split_hash": ident["split_hash"], "data_hash": ident["data_hash"], "gaps_hash": ident["gaps_hash"]}


def check_compatible(manifest: dict, name: str, ident: dict) -> list[str]:
    want = identity_record(ident)
    got = manifest.get("identity", {})
    problems = [f"{k}: artifacts {got.get(k)!r} != current {v!r}" for k, v in want.items() if got.get(k) != v]
    var = manifest.get("variants", {}).get(name)
    if var is None:
        problems.append(f"variant {name} is not in the manifest")
    elif var.get("head"):
        p = C.work_dir() / var["head"]
        if not p.is_file():
            problems.append(f"head checkpoint {p} is missing")
        elif file_sha256(p) != var["head_sha256"]:
            problems.append(f"head checkpoint {p} changed (sha256 mismatch)")
    if var is not None and not var.get("calibrator"):
        problems.append(f"variant {name} has no interval calibrator yet (run the calibrate stage)")
    return problems


def load_head(rel: str) -> dict:
    import torch

    return torch.load(C.work_dir() / rel, map_location="cpu", weights_only=True)


def is_official(train_wells) -> bool:
    return bool(train_wells) and all(w.dataset == C.DATASET for w in train_wells)


def load_variant(name: str, train_wells, device=None, log=None) -> tuple:
    """(backend, calibrator, provenance dict) for one of the four Granite rows."""
    from workstreams.black_physics.granite.backends import make_backend

    arm, tuned = NAME_TO_VARIANT[name]
    if not is_official(train_wells):
        return local_fit(name, train_wells, device, log)
    manifest = read_manifest()
    if manifest is None:
        raise StaleArtifacts(f"no Granite manifest at {manifest_path()}; run "
                             "`python -m workstreams.black_physics.granite all` first")
    ident = official_identity(train_wells)
    problems = check_compatible(manifest, name, ident)
    if problems:
        raise StaleArtifacts("Granite artifacts do not match: " + "; ".join(problems) +
                             ". Rerun `python -m workstreams.black_physics.granite all`.")
    var = manifest["variants"][name]
    head = load_head(var["head"]) if var.get("head") else None
    backend = make_backend(arm, device, head)
    cal = CAL.IntervalCalibrator.from_dict(var["calibrator"])
    prov = {"source": "frozen", "manifest_frozen_at": manifest.get("frozen_at"), "head": var.get("head"),
            "head_sha256": var.get("head_sha256"), "calibrator": var["calibrator"], **identity_record(ident)}
    return backend, cal, prov


FT_NAME_TO_ARM = {"black_ttm_r3_ft": "ttm", "black_tspulse_ft": "tspulse"}   # stronger fine-tune (finetune.py)


def ft_manifest_path() -> Path:
    return C.work_dir() / "manifest_ft.json"


def load_ft_variant(name: str, train_wells, device=None, log=None) -> tuple:
    """(backend, calibrator, provenance) for a stronger fine-tune row; manifest written by `finetune_score`."""
    from workstreams.black_physics.granite.finetune import load_ft

    arm = FT_NAME_TO_ARM[name]
    if not is_official(train_wells):          # contract check: never read FORCE-trained files
        return local_fit(C.VARIANT_IDS[(arm, True)], train_wells, device, log)
    p = ft_manifest_path()
    if not p.is_file():
        raise StaleArtifacts(f"no fine-tune manifest at {p}; run the `finetune` and `finetune_score` stages")
    m = json.loads(p.read_text(encoding="utf-8"))
    ident = official_identity(train_wells)
    want = identity_record(ident)
    problems = [f"{k}: artifacts {m['identity'].get(k)!r} != current {v!r}" for k, v in want.items()
                if m["identity"].get(k) != v]
    var = m.get("variants", {}).get(name)
    if var is None:
        problems.append(f"variant {name} is not in the fine-tune manifest")
    else:
        ck = C.work_dir() / var["checkpoint"]
        if not ck.is_file():
            problems.append(f"checkpoint {ck} is missing")
        elif file_sha256(ck) != var["checkpoint_sha256"]:
            problems.append(f"checkpoint {ck} changed (sha256 mismatch)")
    if problems:
        raise StaleArtifacts("fine-tune artifacts do not match: " + "; ".join(problems) +
                             ". Rerun the `finetune` and `finetune_score` stages.")
    backend, meta = load_ft(arm, device, C.work_dir() / var["checkpoint"])
    cal = CAL.IntervalCalibrator.from_dict(var["calibrator"])
    prov = {"source": "frozen fine-tune", "manifest_frozen_at": m.get("frozen_at"), "checkpoint": var["checkpoint"],
            "checkpoint_sha256": var["checkpoint_sha256"], "scope": meta.get("scope"), "lr": meta.get("lr"),
            "calibrator": var["calibrator"], **want}
    return backend, cal, prov


def local_fit(name: str, train_wells, device=None, log=None, max_updates: int = 5, length_m: float = 10.0) -> tuple:
    """Contract-check fit on arbitrary (e.g. synthetic) wells. Never reads FORCE-trained files."""
    from workstreams.black_physics.granite import train as T
    from workstreams.black_physics.granite.backends import make_backend
    from workstreams.black_physics.granite.reconstruct import make_reconstructor

    arm, tuned = NAME_TO_VARIANT[name]
    cfg = C.CONFIG
    by = {w.name: w for w in train_wells if w.has(C.TARGET)}
    if not by:
        raise ValueError(f"{name}: no training well has {C.TARGET}")
    groups = G.local_split(by)
    small = {g: 2 for g in G.GROUPS}
    gaps = {}
    for g in G.GROUPS:
        gaps[g] = [x for n in groups[g] for x in G.index_gaps(n, by[n].depth, by[n].curves[C.TARGET], group=g,
                                                              n_max=small[g], overlap=False, seed=cfg.seed,
                                                              length_m=length_m)]
    backend = make_backend(arm, device)
    note = "pretrained weights"
    if tuned and gaps["A"]:
        ex = (T.ttm_examples if arm == "ttm" else T.tspulse_examples)(by, gaps["A"], cfg)
        if ex:
            import torch

            torch.manual_seed(cfg.seed)
            backend.set_trainable()
            params = [p for p in backend.model.parameters() if p.requires_grad]
            opt = T._optimizer(params, cfg)
            sampler = T.WellSampler(ex, cfg.seed)
            for _ in range(max_updates):
                T._update(backend, T.LOSSES[arm], sampler, opt, min(4, len(ex)), 1)
            backend.model.eval()
            note = f"pretrained weights + {max_updates} local head updates on {len(ex)} examples"
    rec = make_reconstructor(backend)
    tasks = [G.gap_task(by[g.well], g, length_m=length_m) for g in gaps["D"]]
    pieces = rec.run(tasks)
    records = [(t.well_id, ps[0], np.asarray(by[t.well_id].curves[C.TARGET], float)[ps[0].a:ps[0].b])
               for t, ps in zip(tasks, pieces) if ps]
    cal = CAL.fit(arm, records, source="local")
    prov = {"source": "local", "note": f"{note}; in-sample bands on {len(records)} gaps (contract check only)"}
    if log:
        log(f"{name}: local fit ({prov['note']})")
    return backend, cal, prov
