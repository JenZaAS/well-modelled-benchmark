"""Read the study results and shape them into tables for the comparison view (no Streamlit here)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from wmb.curves import CORE, LOG_SCALE

from .metrics import HIGHER_IS_BETTER, METRIC_COLUMNS
from .models import MODEL_KINDS, MODEL_LABELS, variant_name

FEATURE_FLAGS = [(False, False), (True, False), (False, True)]   # base, context, trend
VARIANTS = [variant_name(k, s, c, t) for k in MODEL_KINDS for c, t in FEATURE_FLAGS for s in (False, True)]
TABLE_COLUMNS = [c for c in METRIC_COLUMNS if c != "n"] + ["n"]


def variant_label(variant: str) -> str:
    """'rf_ctx_strat' -> 'Random forest + context + stratigraphy'."""
    kind, *flags = variant.split("_")
    return (MODEL_LABELS[kind] + (" + context" if "ctx" in flags else "") + (" + trend" if "trend" in flags else "")
            + (" + stratigraphy" if "strat" in flags else ""))


def available_variants(metrics: pd.DataFrame) -> list[str]:
    """Variants present in the results, in display order (older result files only have the first six)."""
    present = set(metrics["variant"])
    return [v for v in VARIANTS if v in present]


def load_metrics(out_dir: Path) -> pd.DataFrame:
    """``metrics.csv`` of a study run, with curves in the standard CORE order. Empty if the run hasn't started."""
    f = Path(out_dir) / "metrics.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    for flag in ("context", "trend"):   # results from before these feature sets existed
        if flag not in df:
            df[flag] = False
    df["target"] = pd.Categorical(df["target"], categories=CORE, ordered=True)
    return df.sort_values(["target", "variant"]).reset_index(drop=True)


def load_predictions(out_dir: Path, target: str) -> pd.DataFrame:
    """Predictions of all variants for one target: WELL, DEPTH_MD and one column per variant."""
    return pd.read_parquet(Path(out_dir) / "predictions" / f"{target}.parquet")


def _log_labels(index: pd.Index) -> list[str]:
    """Row labels: resistivity is scored in log10, so say so."""
    return [f"{t} (log10)" if t in LOG_SCALE else str(t) for t in index]


def variant_table(metrics: pd.DataFrame, variant: str) -> pd.DataFrame:
    """One row per log, one column per statistic, for a single model variant."""
    t = metrics[metrics["variant"] == variant].set_index("target")[TABLE_COLUMNS]
    t.index = _log_labels(t.index)
    return t.rename_axis("log")


def comparison_table(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per log, one column per model variant, for one statistic."""
    t = metrics.pivot(index="target", columns="variant", values=metric)
    t.index = _log_labels(t.index)
    return t[[v for v in VARIANTS if v in t.columns]].rename(columns=variant_label).rename_axis("log")


def best_variant_per_log(table: pd.DataFrame, metric: str) -> pd.Series:
    """Column label of the best model per row (R2/pearson highest, bias closest to 0, others lowest)."""
    if metric in HIGHER_IS_BETTER:
        return table.idxmax(axis=1)
    return (table.abs() if metric == "bias" else table).idxmin(axis=1)


def _effect_table(metrics: pd.DataFrame, metric: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Per log, ``metric`` of the second variant minus the first, for every pair whose results exist."""
    t = metrics.pivot(index="target", columns="variant", values=metric)
    t.index = _log_labels(t.index)
    out = {variant_label(base): t[other] - t[base] for base, other in pairs if {base, other} <= set(t.columns)}
    return pd.DataFrame(out).rename_axis("log")


def stratigraphy_effect(metrics: pd.DataFrame, metric: str = "R2") -> pd.DataFrame:
    """Change in ``metric`` when stratigraphy is added (with minus without), per log and model."""
    pairs = [(variant_name(k, False, c, t), variant_name(k, True, c, t)) for k in MODEL_KINDS
             for c, t in FEATURE_FLAGS]
    return _effect_table(metrics, metric, pairs)


def context_effect(metrics: pd.DataFrame, metric: str = "R2") -> pd.DataFrame:
    """Change in ``metric`` when context features are added (with minus without), per log and model."""
    pairs = [(variant_name(k, s), variant_name(k, s, True)) for k in MODEL_KINDS for s in (False, True)]
    return _effect_table(metrics, metric, pairs)


def trend_effect(metrics: pd.DataFrame, metric: str = "R2") -> pd.DataFrame:
    """Change in ``metric`` when trend features are added (with minus without), per log and model."""
    pairs = [(variant_name(k, s), variant_name(k, s, trend=True)) for k in MODEL_KINDS for s in (False, True)]
    return _effect_table(metrics, metric, pairs)
