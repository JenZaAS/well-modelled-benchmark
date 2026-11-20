"""The results/ allowlist in .gitignore: scores are shared for every dataset, prediction files (which embed
well curves) only for openly licensed datasets. See results/README.md. No well data required."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SHARED = [
    "results/README.md",
    "results/m/force2020/DTC/scores.json",
    "results/m/force2020/DTC/gap_50/t/scores.json",
    "results/m/nlog_dts/DTS/gap_50/scores.json",
    "results/m/diskos/RHOB/missing_curve/scores.json",
    "results/m/force2020/DTC/gap_50/t/predictions.json",
    "results/m/force2020/DTC/gap_50/predictions.npz",
    "results/m/synthetic/RHOB/missing_curve/predictions.json",
    "results/m/synthetic/RHOB/gap_10/t/predictions.npz",
    "results/runs/20261120T000000_force2020.json",
    "results/summary/scoreboard.md",
    "results/summary/scoreboard.json",
]

LOCAL_ONLY = [
    "results/m/nlog_dts/DTS/gap_50/predictions.json",
    "results/m/kansas_kgs_dts/DTS/missing_curve/t/predictions.npz",
    "results/m/volve/RHOB/gap_10/predictions.json",
    "results/m/diskos/RHOB/gap_10/predictions.json",
    "results/m/some_new_dataset/RHOB/gap_10/predictions.json",
    "results/runs/20261120T000000_force2020.csv",
    "results/m/force2020/DTC/gap_50/failure.json",
    "results/blue_ml_study/metrics.csv",
    "results/blue_ml_study/predictions/x.parquet",
]


def _ignored(paths: list[str]) -> set[str]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("needs git and a git checkout")
    out = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *paths], cwd=REPO, capture_output=True, text=True,
    )
    assert out.returncode in (0, 1), out.stderr  # 1 means none of the paths is ignored
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def test_scores_and_open_dataset_predictions_are_shared():
    assert _ignored(SHARED) == set()


def test_other_predictions_and_local_files_stay_out_of_git():
    assert _ignored(LOCAL_ONLY) == set(LOCAL_ONLY)
