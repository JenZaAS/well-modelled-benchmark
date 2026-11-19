"""Streamlit app: compare the ML models and explore predictions well by well.

    streamlit run workstreams/blue_boosted_trees/ml_study/app.py

Needs the study results first (see run_study.py) and the FORCE 2020 data configured in data_paths.local.yaml.
Only the 20 held-out wells are shown: the models never saw them during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "src", _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from wmb.curves import CORE, LOG_SCALE, UNITS  # noqa: E402
from workstreams.blue_boosted_trees.ml_study import report  # noqa: E402
from workstreams.blue_boosted_trees.ml_study.data import load_test_wells  # noqa: E402
from workstreams.blue_boosted_trees.ml_study.metrics import (  # noqa: E402
    HIGHER_IS_BETTER, evaluate, target_space, target_unit)
from workstreams.blue_boosted_trees.ml_study.run_study import default_out_dir  # noqa: E402

CURVE_COLOURS = {"GR": "#2e7d32", "RDEP": "#c62828", "RMED": "#ef6c00", "CALI": "#6a1b9a", "RHOB": "#1565c0",
                 "NPHI": "#00838f", "PEF": "#ad1457", "DTC": "#4e342e", "DTS": "#558b2f"}
METRIC_HELP = {
    "MAE": "mean absolute error (log units, lower is better)",
    "RMSE": "root mean squared error (lower is better; punishes big misses)",
    "R2": "share of the log's variance explained (1 = perfect, 0 = no better than the mean)",
    "bias": "mean error = mean(predicted - measured); positive = over-prediction",
    "MedAE": "median absolute error",
    "MAE_rel_std": "MAE divided by the spread (std) of the measured log; comparable across logs",
    "pearson_r": "correlation between predicted and measured",
}


@st.cache_resource(show_spinner="Loading the 20 held-out wells...")
def get_test_wells():
    return load_test_wells()


@st.cache_data(show_spinner=False)
def get_metrics(out_dir: str, mtime: float) -> pd.DataFrame:
    return report.load_metrics(Path(out_dir))


@st.cache_data(show_spinner=False)
def get_predictions(out_dir: str, target: str, mtime: float) -> pd.DataFrame:
    return report.load_predictions(Path(out_dir), target)


def highlight_best(table: pd.DataFrame, metric: str):
    """Style a comparison table: best model per row in bold green."""
    best = report.best_variant_per_log(table, metric)
    return table.style.format("{:.4f}" if metric != "n" else "{:.0f}").apply(
        lambda row: ["background-color:#c8e6c9;font-weight:bold" if c == best[row.name] else "" for c in row.index],
        axis=1)


def comparison_tab(metrics: pd.DataFrame) -> None:
    st.subheader("Model statistics per log")
    st.caption("Held-out FORCE 2020 wells (20), models trained on the 98 training wells. "
               "Each log is predicted from the other logs and depth. "
               "Errors are in each log's own unit; resistivity (RDEP, RMED) is scored in log10(ohm.m).")
    variant = st.selectbox("Model", report.VARIANTS, format_func=report.variant_label, index=1, key="cmp_variant")
    table = report.variant_table(metrics, variant)
    st.dataframe(table.style.format({**{c: "{:.4f}" for c in table.columns}, "n": "{:,.0f}"})
                 .background_gradient(subset=["R2"], cmap="RdYlGn", vmin=0, vmax=1), width="stretch")
    with st.expander("What do the columns mean?"):
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in METRIC_HELP.items()))

    st.subheader("Compare the models")
    metric = st.selectbox("Statistic", [c for c in report.TABLE_COLUMNS if c != "n"], index=2, key="cmp_metric",
                          format_func=lambda m: f"{m}: {METRIC_HELP[m]}")
    cmp = report.comparison_table(metrics, metric)
    st.dataframe(highlight_best(cmp, metric), width="stretch")
    st.caption("Green = best model for that log" + (" (closest to zero)." if metric == "bias" else "."))
    long = cmp.reset_index().melt("log", var_name="model", value_name=metric)
    fig = px.bar(long, x="log", y=metric, color="model", barmode="group", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Does stratigraphy help?")
    gain_metric = st.radio("Change in", ["R2", "MAE"], horizontal=True)
    effect = report.stratigraphy_effect(metrics, gain_metric)
    better = "positive" if gain_metric in HIGHER_IS_BETTER else "negative"
    st.caption(f"With stratigraphy minus without. A {better} number means stratigraphy improved the prediction.")
    st.dataframe(effect.style.format("{:+.4f}").background_gradient(
        cmap="RdYlGn" if gain_metric in HIGHER_IS_BETTER else "RdYlGn_r", axis=None), width="stretch")


def depth_tracks(well, target: str, pred: np.ndarray, variant_name: str) -> go.Figure:
    """Depth plot: stratigraphy strip, every other log with data, then predicted vs measured target."""
    inputs = [c for c in CORE if c != target and well.has(c)]
    titles = ["Group"] + [f"{c}<br>{UNITS[c]}" for c in inputs] + [f"{target} (pred. vs measured)<br>{UNITS[target]}"]
    fig = make_subplots(rows=1, cols=len(titles), shared_yaxes=True, subplot_titles=titles,
                        horizontal_spacing=0.012, column_widths=[0.5] + [1] * (len(titles) - 1))
    groups = well.meta["group"]
    edges = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    palette = px.colors.qualitative.Pastel
    names = list(dict.fromkeys(groups))
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(groups)]):
        top, base = well.depth[lo], well.depth[hi - 1]
        fig.add_hrect(y0=top, y1=base, fillcolor=palette[names.index(groups[lo]) % len(palette)], opacity=0.9,
                      line_width=0, row=1, col=1, exclude_empty_subplots=False)
        if base - top > 40:
            fig.add_annotation(x=0.5, y=(top + base) / 2, text=groups[lo].replace(" GP.", ""), showarrow=False,
                               textangle=-90, font_size=10, xref="x domain", yref="y", row=1, col=1)
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    for i, c in enumerate(inputs, start=2):
        fig.add_trace(go.Scatter(x=well.curves[c], y=well.depth, mode="lines", name=c,
                                 line=dict(color=CURVE_COLOURS[c], width=1)), row=1, col=i)
        if c in LOG_SCALE:
            fig.update_xaxes(type="log", row=1, col=i)
    last = len(titles)
    actual = well.curves.get(target)
    if actual is not None and np.isfinite(actual).any():
        fig.add_trace(go.Scatter(x=actual, y=well.depth, mode="lines", name=f"{target} measured",
                                 line=dict(color="black", width=1.3)), row=1, col=last)
    fig.add_trace(go.Scatter(x=pred, y=well.depth, mode="lines", name=f"{target} predicted ({variant_name})",
                             line=dict(color="#e53935", width=1.3)), row=1, col=last)
    if target in LOG_SCALE:
        fig.update_xaxes(type="log", row=1, col=last)
    fig.update_yaxes(autorange="reversed", title_text="Depth MD (m)", row=1, col=1)
    fig.update_layout(height=900, margin=dict(l=60, r=10, t=70, b=10), legend=dict(orientation="h", y=-0.01),
                      hovermode="y unified")
    return fig


def explorer_tab(out_dir: Path, metrics: pd.DataFrame) -> None:
    try:
        wells = get_test_wells()
    except FileNotFoundError as e:
        st.error(f"FORCE 2020 data not found: {e}. Set up data_paths.local.yaml (docs/SETUP_DATA_PATHS.md).")
        return
    by_name = {w.name: w for w in wells}
    targets = [t for t in CORE if t in set(metrics["target"].astype(str))]
    c1, c2, c3 = st.columns(3)
    well_name = c1.selectbox("Well", list(by_name), help="One of the 20 held-out wells")
    target = c2.selectbox("Log to predict", targets, index=targets.index("RHOB") if "RHOB" in targets else 0)
    sub = metrics[metrics["target"] == target]
    default = sub.loc[sub["R2"].idxmax(), "variant"]
    variant = c3.selectbox("Model", report.VARIANTS, format_func=report.variant_label,
                           index=report.VARIANTS.index(default), help="Default: best overall R2 for this log")
    well = by_name[well_name]

    f = out_dir / "predictions" / f"{target}.parquet"
    preds = get_predictions(str(out_dir), target, f.stat().st_mtime)
    p = preds[preds["WELL"] == well_name].sort_values("DEPTH_MD", kind="stable")
    if len(p) != len(well.depth) or not np.allclose(p["DEPTH_MD"].to_numpy(), well.depth):
        st.error("Saved predictions don't line up with this well's depth samples. Re-run run_study.py.")
        return
    actual = well.curves.get(target, np.full(len(well.depth), np.nan))
    has_actual = bool(np.isfinite(actual).any())
    scored = lambda v: evaluate(target_space(target, actual), target_space(target, p[v].to_numpy(float)))  # noqa: E731
    unit = target_unit(target)

    left, right = st.columns([4, 1.3])
    left.plotly_chart(depth_tracks(well, target, p[variant].to_numpy(float), report.variant_label(variant)),
                      width="stretch")
    with right:
        st.markdown(f"#### {target} in {well_name}")
        if not has_actual:
            st.info(f"No measured {target} in this well: showing the prediction only.")
        else:
            s = scored(variant)
            st.metric("Mean error (bias)", f"{s['bias']:+.4f} {unit}", help=METRIC_HELP["bias"])
            st.metric("MAE", f"{s['MAE']:.4f}", help=METRIC_HELP["MAE"])
            st.metric("RMSE", f"{s['RMSE']:.4f}", help=METRIC_HELP["RMSE"])
            st.metric("R²", f"{s['R2']:.3f}", help=METRIC_HELP["R2"])
            st.metric("Correlation r", f"{s['pearson_r']:.3f}", help=METRIC_HELP["pearson_r"])
            st.caption(f"Errors in {unit}. {s['n']:,} depth samples with both measured and predicted values.")
            rows = {report.variant_label(v): scored(v) for v in report.VARIANTS if v in p}
            cmp = pd.DataFrame(rows).T[["MAE", "R2", "bias"]]
            st.markdown("**All models, this well**")
            st.dataframe(cmp.style.format("{:.4f}"), width="stretch")


def main() -> None:
    st.set_page_config(page_title="Well-log prediction: blue-team ML study", layout="wide")
    st.title("Predicting well logs: gradient boosting vs linear regression vs random forest")
    out_dir = Path(st.sidebar.text_input("Results folder", str(default_out_dir())))
    f = out_dir / "metrics.csv"
    if not f.exists():
        st.warning(f"No results in {out_dir}. Run: python -m workstreams.blue_boosted_trees.ml_study.run_study")
        return
    metrics = get_metrics(str(out_dir), f.stat().st_mtime)
    st.sidebar.write(f"{metrics['target'].nunique()} logs, {metrics['variant'].nunique()} model variants")
    tab_cmp, tab_well = st.tabs(["Model comparison", "Well explorer"])
    with tab_cmp:
        comparison_tab(metrics)
    with tab_well:
        explorer_tab(out_dir, metrics)


main()
