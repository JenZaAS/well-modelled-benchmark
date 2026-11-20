"""Saving and loading fitted models, so a new well can be predicted without retraining.

One joblib file per (log, variant): ``<model_dir>/<LOG>/<variant>.joblib``. Each file holds the whole fitted
``MLContender`` (feature builder with the stratigraphy order, preprocessing and the estimator), so a new well is
prepared exactly like the training wells. ``manifest.json`` records what the models were trained on and the library
versions (pickled models can misbehave under different scikit-learn / XGBoost versions).

Only load model files from people you trust: joblib files are pickles and can run code when loaded.
"""
from __future__ import annotations

import json
import time
from importlib import metadata
from pathlib import Path

import joblib

from wmb import paths

from .models import MLContender

OUT_NAME = "blue_ml_study"
MANIFEST = "manifest.json"
TRACKED_PACKAGES = ("scikit-learn", "xgboost", "numpy", "pandas")


def default_model_dir() -> Path:
    """Saved models are not results: they live in the work folder (``FH2026_WORK`` / ``work_root``), never in git."""
    return paths.work_root() / OUT_NAME / "models"


def model_path(model_dir: Path, target: str, variant: str) -> Path:
    """File of one fitted model."""
    return Path(model_dir) / target / f"{variant}.joblib"


def save_model(model_dir: Path, model: MLContender, target: str) -> Path:
    """Write a fitted contender and return its path."""
    from .models import variant_name
    path = model_path(model_dir, target, variant_name(model.kind, model.use_strat, model.context, model.trend))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(model_dir: Path, target: str, variant: str) -> MLContender:
    """Read one fitted contender (see the module note on trusting model files)."""
    path = model_path(model_dir, target, variant)
    if not path.exists():
        raise FileNotFoundError(f"no saved model for {target} / {variant} in {model_dir}")
    return joblib.load(path)


def list_models(model_dir: Path) -> dict[str, list[str]]:
    """Saved variants per log, e.g. ``{'RHOB': ['gbt', 'gbt_strat', ...]}``. Empty if the folder is missing."""
    root = Path(model_dir)
    if not root.is_dir():
        return {}
    found = {d.name: sorted(f.stem for f in d.glob("*.joblib")) for d in root.iterdir() if d.is_dir()}
    return {t: v for t, v in found.items() if v}


def installed_versions() -> dict[str, str]:
    """Versions of the libraries that decide whether a pickled model loads correctly."""
    out = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def write_manifest(model_dir: Path, info: dict) -> None:
    """Record what the models were trained on (merged with any earlier manifest)."""
    f = Path(model_dir) / MANIFEST
    old = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({**old, **info, "versions": installed_versions(),
                             "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2), encoding="utf-8")


def read_manifest(model_dir: Path) -> dict:
    """The manifest, or an empty dict if there is none."""
    f = Path(model_dir) / MANIFEST
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def version_warnings(manifest: dict) -> list[str]:
    """One message per library whose installed version differs from the one used for training."""
    now = installed_versions()
    return [f"{pkg}: models were trained with {old}, installed is {now.get(pkg)}. Predictions may differ or fail."
            for pkg, old in manifest.get("versions", {}).items() if now.get(pkg) != old]
