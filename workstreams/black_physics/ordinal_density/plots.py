"""Figures: compact scoreboard, P(RHOB < 2.40) reliability, and the two pitch frames (before / after the reveal).

The pitch gap is chosen from observed context only: among test gaps with measured RHOB on both sides and no
abstention, the one with the median flank variability (lower median for an even count; ties by well name).
The marked depth is the gap's middle sample. Neither choice looks at the hidden values.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np

from workstreams.black_physics.ordinal_density import bins as B
from workstreams.black_physics.ordinal_density import cli as CL

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"red_tspulse_finetune": "#2a78d6", "red_timestral": "#eb6834"}      # categorical slots 1 and 2
MEASURED = "#e34948"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CREDIT = ("Data: FORCE 2020 well-log and lithofacies dataset (Bormann, Aursand, Dilib, Manral and Dischington 2020, "
          "Zenodo, doi:10.5281/zenodo.4351156; logs NLOD 2.0, labels CC BY 4.0). Credit FORCE and the Norwegian "
          "Offshore Directorate (Sodir).")
CONTEXT_M = 12.0


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt, LinearSegmentedColormap.from_list("prob", RAMP)


def _head_name(key: str) -> str:
    name, var = key.split("|")
    return f"{CL.HEAD_LABEL[name]}, {'temperature-calibrated' if var == 'cal' else 'raw'}"


# ------------------------------------------------------------------------------------------------ scoreboard
def scoreboard(d: Path, res: dict, plt) -> Path:
    panels = [(m, res["models"][m]["groups"]["test"]["heads"], res["models"][m]["selected"], CL.label(m) +
               "\nall 6,600 hidden test depths") for m in res["models"]]
    for m, z in res.get("matched_decima", {}).items():
        panels.append((m, z["groups"]["test"], None,
                       f"{CL.label(m)}, matched sample\n{z['n_states']['test']} test depths (8 per gap), with Decima"))
    order = list(CL.HEADS) + ["decima_small"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    xmax = 0.0
    for ax, (m, heads, sel, title) in zip(axes, panels):
        names = [h for h in order if f"{h}|raw" in heads]
        for i, h in enumerate(names):
            raw, cal = heads[f"{h}|raw"]["RPS"], heads[f"{h}|cal"]["RPS"]
            xmax = max(xmax, raw, cal)
            ax.scatter([raw], [i - 0.14], s=56, color=SERIES.get(m, INK2), zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([cal], [i + 0.14], s=56, facecolor=SURFACE, edgecolor=SERIES.get(m, INK2), linewidth=2, zorder=3)
            if sel and sel.startswith(h + "|"):
                v = heads[sel]["RPS"]
                ax.annotate("heatmap head", (v, i + (0.14 if sel.endswith("cal") else -0.14)), xytext=(8, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([CL.HEAD_LABEL[h] for h in order])
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Test RPS (lower is better)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    for ax in axes:
        ax.set_xlim(0, xmax * 1.25)
    fig.text(0.01, 0.01, "Filled: raw. Open: one temperature fitted on the 12 development (D) wells. The heatmap head "
             "was chosen by development RPS, not test.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = d / "fig_scoreboard.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ reliability
def reliability(d: Path, res: dict, plt) -> Path:
    fig = plt.figure(figsize=(9.6, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.08, wspace=0.25)
    for j, g in enumerate(("D", "test")):
        ax, axn = fig.add_subplot(gs[0, j]), fig.add_subplot(gs[1, j])
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
        width = 0.035
        for k, (m, mres) in enumerate(res["models"].items()):
            s = mres["groups"][g]["heads"][mres["selected"]]
            rel = s["reliability_below_2.40"]
            xs = [r["mean_p"] for r in rel if r["n"] >= 20]
            ys = [r["observed"] for r in rel if r["n"] >= 20]
            col = SERIES.get(m, INK2)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=3,
                    label=f"{CL.label(m)} (Brier {s['brier_below_2.40']:.3f})")
            cen = np.array([(r["lo"] + r["hi"]) / 2 for r in rel])
            axn.bar(cen + (k - 0.5) * width, [r["n"] for r in rel], width=width * 0.9, color=col)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticklabels([])
        ax.set_ylabel("Observed share below 2.40")
        ax.set_title(f"{'Development wells (D, 12)' if g == 'D' else 'Test wells (20)'}", loc="left", fontsize=10)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        axn.set_xlim(0, 1)
        axn.set_yscale("log")
        axn.set_ylabel("depths", fontsize=8)
        axn.set_xlabel("Predicted P(RHOB < 2.40 g/cm³)")
    fig.text(0.01, -0.06, "Heatmap head of each model. Points with fewer than 20 depths are not drawn; bars show "
             "every bin's depth count. "
             "Depths in a gap are strongly correlated, so counts overstate the evidence.", fontsize=7.5, color=INK2)
    p = d / "fig_reliability_below_2.40.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ pitch frames
def choose_gap(d: Path, models: list[str]) -> tuple[str, dict]:
    truth = dict(np.load(d / "truth.npz"))
    cands = []
    rows = [dict(np.load(d / f"prepared_{m}.npz")) for m in models]
    for gap in dict.fromkeys(truth["test|gap"]):
        sel = truth["test|gap"] == gap
        r0 = rows[0]
        two_sided = np.isfinite(r0["test|flank_up"][sel]).all() and np.isfinite(r0["test|flank_dn"][sel]).all()
        fv = float(r0["test|flank_var"][sel][0])
        full = all(np.isfinite(r["test|base"][sel]).all() for r in rows)
        if two_sided and full and np.isfinite(fv):
            cands.append((fv, str(truth["test|well"][sel][0]), str(gap)))
    cands.sort()
    fv, well, gap = cands[(len(cands) - 1) // 2]
    return gap, {"rule": "median flank variability among two-sided, fully predicted test gaps (lower median; ties "
                         "by well)", "eligible": len(cands), "flank_var": fv, "well": well, "gap_id": gap}


def pitch(d: Path, res: dict, models: list[str], plt, cmap) -> list[Path]:
    from workstreams.black_physics.granite import decide_any as DA

    gap, why = choose_gap(d, models)
    truth = dict(np.load(d / "truth.npz"))
    sel = truth["test|gap"] == gap
    groups, _ = DA.tasks_by_group()
    task = next(t for t in groups["test"] if t.meta["gap_id"] == gap)
    depth = np.asarray(task.depth, float)
    obs = np.asarray(task.target_observed, float)
    idx = truth["test|index"][sel]
    y = truth["test|y"][sel]
    zd = depth[idx]
    edges_y = np.r_[zd[0] - (zd[1] - zd[0]) / 2, (zd[:-1] + zd[1:]) / 2, zd[-1] + (zd[-1] - zd[-2]) / 2]
    mid = len(idx) // 2
    win = (zd[0] - CONTEXT_M, zd[-1] + CONTEXT_M)
    ctx = (depth >= win[0]) & (depth <= win[1])
    per = {}
    for m in models:
        mres = res["models"][m]
        key = mres["selected"]
        P = dict(np.load(d / f"probs_{m}.npz"))[f"test|{key}"][sel]
        tails = B.Tails.from_dict(json.loads((d / f"heads_{m}.json").read_text(encoding="utf-8"))["tails"])
        per[m] = {"p": P, "r": B.readouts(P, tails), "key": key, "s": mres["groups"]["test"]["heads"][key]}
    vmax = float(np.ceil(10 * max(np.quantile(v["p"], 0.995) for v in per.values())) / 10)
    xe = B.display_edges()
    xc = (xe[:-1] + xe[1:]) / 2
    paths = []
    for frame in (1, 2):
        fig = plt.figure(figsize=(12.8, 7.6))
        gs = fig.add_gridspec(2, len(models) + 1, height_ratios=(3.3, 1), width_ratios=[1] * len(models) + [0.04],
                              hspace=0.34, wspace=0.18, left=0.07, right=0.95, top=0.78, bottom=0.13)
        for j, m in enumerate(models):
            v = per[m]
            ax = fig.add_subplot(gs[0, j])
            mesh = ax.pcolormesh(xe, edges_y, v["p"], cmap=cmap, vmin=0, vmax=vmax, shading="flat", zorder=1)
            ax.plot(obs[ctx], depth[ctx], color=INK2, lw=1.4, zorder=3, label="measured RHOB outside the gap")
            ax.plot(v["r"]["median"], zd, color=INK, lw=1.6, ls=(0, (4, 2)), zorder=4,
                    label="median of the 15-bin forecast")
            if frame == 2:
                ax.plot(y, zd, color=SURFACE, lw=3.6, zorder=5)
                ax.plot(y, zd, color=MEASURED, lw=1.8, zorder=6, label="measured RHOB inside the gap (revealed)")
            ax.axhline(zd[mid], color=INK2, lw=0.9, ls=":", zorder=4)
            ax.axhspan(win[0], zd[0], color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.axhspan(zd[-1], win[1], color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.set_ylim(win[1], win[0])
            ax.set_xlim(xe[0], xe[-1])
            ax.set_xlabel("Bulk density RHOB (g/cm³)")
            if j == 0:
                ax.set_ylabel("Measured depth (m)")
            s = v["s"]
            ax.set_title(f"{CL.label(m)}: {CL.INFO[m]['inputs']}\nTest, 20 wells: RPS {s['RPS']:.4f}; 80 % interval "
                         f"covers {100 * s['cover80']:.0f} % (mean width {s['width80']:.2f} g/cm³)", loc="left",
                         fontsize=9.5)
            ax.text(xe[-1], zd[mid], " marked depth ", ha="right", va="bottom", fontsize=7.5, color=INK2, zorder=7)
            if j == 0:
                fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.065, 0.875), ncol=3,
                           fontsize=8.5, frameon=False)
            bx = fig.add_subplot(gs[1, j])
            pm = v["p"][mid]
            below = B.prob_below(pm[None])[0]
            cols = ["#256abf" if k <= B.EVENT_INDEX else "#9ec5f4" for k in range(B.N_BINS)]
            bx.bar(xc, pm, width=0.1 * 0.85, color=cols)
            bx.axvline(B.EVENT_EDGE, color=INK2, lw=1, ls="--")
            bx.set_xlim(xe[0], xe[-1])
            bx.set_ylim(0, max(0.05, pm.max() * 1.25))
            bx.set_ylabel("probability", fontsize=8)
            bx.set_title(f"At {zd[mid]:.1f} m: P(RHOB < 2.40) = {below:.2f}", loc="left", fontsize=9)
            if frame == 2:
                bx.axvline(y[mid], color=MEASURED, lw=2)
                bx.annotate(f"measured {y[mid]:.2f}", (y[mid], bx.get_ylim()[1]), xytext=(-5, -2),
                            textcoords="offset points", color=MEASURED, fontsize=8, va="top", ha="right")
            bx.grid(axis="y", color=GRID, lw=0.8)
            bx.set_axisbelow(True)
        cax = fig.add_subplot(gs[0, -1])
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label("probability of each 0.10 g/cm³ bin", fontsize=8)
        cb.outline.set_visible(False)
        head = ("1 of 2: two models fill the same hidden 50 m of density" if frame == 1 else
                "2 of 2: the measured curve revealed")
        fig.suptitle(f"{head}  (well {why['well']}, {zd[0]:.0f}–{zd[-1]:.0f} m)", x=0.07, ha="left", fontsize=13,
                     fontweight="bold")
        fig.text(0.07, 0.905, "Each row is one depth; the shading is the forecast probability of each density bin. "
                 "Bins: below 1.60, then 0.10 wide to 2.90, then 2.90 and above (open bins drawn 0.10 wide).",
                 fontsize=8.5, color=INK2)
        note = ("Heatmap heads chosen by development (D) RPS: "
                + "; ".join(f"{CL.label(m)}: {_head_name(per[m]['key'])}" for m in models)
                + f". Gap chosen without the hidden values: {why['rule']}. Bars: probabilities at the marked depth; "
                  "dark bars lie below 2.40 g/cm³.")
        fig.text(0.07, 0.03, textwrap.fill(note, 210), fontsize=7.5, color=INK2, va="bottom")
        fig.text(0.07, 0.012, CREDIT, fontsize=7, color=INK2)
        p = d / f"fig_pitch_frame{frame}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        paths.append(p)
    marked = {m: float(B.prob_below(per[m]["p"][mid][None])[0]) for m in models}
    info = {**why, "marked_depth_m": float(zd[mid]), "heads": {m: per[m]["key"] for m in models},
            "p_below_2.40_marked": marked, "measured_marked": float(y[mid])}
    (d / "pitch_gap.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return paths


def plot(test: str, models: list[str], run_id: str | None) -> Path:
    d, core = CL.run_dir(test, models, run_id)
    res = json.loads((d / "scores.json").read_text(encoding="utf-8"))
    plt, cmap = _mpl()
    out = [scoreboard(d, res, plt), reliability(d, res, plt), *pitch(d, res, models, plt, cmap)]
    CL.record(d, core, "plot", {"figures": [p.name for p in out]})
    return d
