"""Step 8: tables, paired bootstrap, figures and the 30-second text for the Granite 50 m density-gap experiment.

    python -m workstreams.black_physics.granite report

Reads only frozen artifacts (`final/`, the stage JSON files, the station `scores.json` rows) and writes:

    <work_root>/black_granite/report/REPORT.md            full local report (per-gap rows; well data, never committed)
    <work_root>/black_granite/report/results_table.json   replayable table: every number + the hashes it came from
    <work_root>/black_granite/report/figures/*.png        pitch figure variants and the supporting charts
    workstreams/black_physics/granite/RESULTS.md          aggregate-only summary (no well names, depths or values)

Nothing here refits anything; rerunning it on the same artifacts gives the same bytes except the timestamp.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from workstreams.black_physics.granite import config as C
from workstreams.black_physics.granite import decision as DEC

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRANITE = [C.VARIANT_IDS[k] for k in (("ttm", False), ("ttm", True), ("tspulse", False), ("tspulse", True))]
OTHERS = ["black_linear_gap", "black_gardner_fixed", "black_gardner_flank_resid"]
ROWS = GRANITE + OTHERS
UNAVAILABLE = {
    "red_timestral (with covariates)": "not run: no authorised Timestral-5M weights/environment on this laptop "
                                       "(Modified MIT revenue clause; needs organisation confirmation)",
    "black_timestral_target_only": "wrapper committed (WMB_BLACK_TIMESTRAL=1), not run for the same reason",
}
LABEL = {"black_ttm_r3_frozen": "TTM-R3 frozen", "black_ttm_r3_tuned": "TTM-R3 tuned",
         "black_tspulse_frozen": "TSPulse frozen", "black_tspulse_tuned": "TSPulse tuned",
         "black_linear_gap": "Linear interpolation", "black_gardner_fixed": "Gardner (fixed)",
         "black_gardner_flank_resid": "Gardner + flank residual"}
METHOD_LABEL = {"prevalence": "Prevalence", "width_only": "Width-only logistic", "numeric": "Numeric logistic",
                "numeric_gardner": "Numeric + Gardner", "numeric_cal": "Numeric (recal. D)",
                "numeric_gardner_cal": "Numeric + Gardner (recal. D)", "decima_raw": "Decima-small raw",
                "decima_cal": "Decima-small (recal. D)"}

# palette (dataviz reference instance, light mode)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                            "#008300", "#6250d6", "#e34948")
BLUE_LIGHT = "#b7d3f6"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
ROW_COLOR = {"black_ttm_r3_frozen": VIOLET, "black_ttm_r3_tuned": BLUE, "black_tspulse_frozen": MAGENTA,
             "black_tspulse_tuned": AQUA, "black_linear_gap": MUTED, "black_gardner_fixed": ORANGE,
             "black_gardner_flank_resid": YELLOW}


# ------------------------------------------------------------------------------------------------ loading
def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def station_scores(name: str) -> dict | None:
    from wmb.predict_cache import result_dir

    p = result_dir(REPO / "results", name, C.DATASET, C.TARGET, "gap_50", C.TEST_NAME) / "scores.json"
    return _load(p) if p.is_file() else None


class Inputs:
    def __init__(self):
        W = C.work_dir()
        self.W = W
        self.final = _load(W / "final" / "final.json")
        self.sidecar = _load(W / "final" / "decision_sidecar.json")
        self.npz = dict(np.load(W / "final" / "predictions.npz"))
        self.train = {a: _load(W / f"train_{a}.json") for a in C.ARMS if (W / f"train_{a}.json").is_file()}
        self.calibration = _load(W / "calibration.json")["variants"]
        self.selection = _load(W / "selection.json")
        self.decision = _load(W / "decision.json")
        self.manifest = _load(W / "manifest.json")
        self.prepared = _load(W / "prepared.json")
        self.dev_B = {n: _load(W / "dev" / f"B_{n}.json") for n in GRANITE if (W / "dev" / f"B_{n}.json").is_file()}
        self.dev_D = {n: _load(W / "dev" / f"D_{n}.json") for n in GRANITE if (W / "dev" / f"D_{n}.json").is_file()}
        self.dec_rows = {g: _load(W / "decision" / f"rows_{g}.json") for g in ("C", "D")}
        self.station = {n: station_scores(n) for n in ROWS}
        self.rows = self.final["rows"]
        self.keys = [r["key"] for r in self.rows[GRANITE[0]]]
        for n in ROWS:   # every compared row must be scored on the identical gap set
            if [r["key"] for r in self.rows[n]] != self.keys:
                raise ValueError(f"{n}: gap ids differ from {GRANITE[0]}")
        if sorted(s["gap_id"] for s in self.sidecar) != sorted(self.keys):
            raise ValueError("decision sidecar gap ids differ from the scored gaps")
        self.selected = self.final["selected"]

    def arr(self, name, key, kind):
        return self.npz.get(f"{name}|{key}|{kind}")

    def truth(self, key):
        return self.npz[f"truth|{key}"]

    def depth(self, key):
        return self.npz[f"depth|{key}"]


# ------------------------------------------------------------------------------------------------ statistics
def per_well(rows) -> dict[str, float]:
    by: dict = {}
    for r in rows:
        s = by.setdefault(r["well"], [0.0, 0])
        s[0] += r["abs_err_sum"]
        s[1] += r["n_pred"]
    return {w: (a / n if n else float("nan")) for w, (a, n) in by.items()}


def paired_bootstrap(a: dict, b: dict, n_boot: int, seed: int) -> dict:
    """Mean over wells of (a - b) per-well MAE, 95 % percentile CI from resampling wells with replacement."""
    wells = sorted(w for w in a if w in b and np.isfinite(a[w]) and np.isfinite(b[w]))
    d = np.array([a[w] - b[w] for w in wells])
    if not len(d):
        return {"n_wells": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_wells": len(d), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "ci95": [float(lo), float(hi)], "p_boot_lt0": float((boots < 0).mean()),
            "wells_better": int((d < 0).sum()), "wells_worse": int((d > 0).sum()), "n_boot": n_boot}


def coverage(I: Inputs, name: str, lo_k: str, hi_k: str) -> dict | None:
    hits, widths, by = [], [], {}
    for r in I.rows[name]:
        lo, hi = I.arr(name, r["key"], lo_k), I.arr(name, r["key"], hi_k)
        if lo is None or hi is None:
            return None
        y = I.truth(r["key"])
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        h = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
        hits.append(h)
        widths.append(hi[ok] - lo[ok])
        by.setdefault(r["well"], []).append(h)
    h = np.concatenate(hits)
    return {"pooled": float(h.mean()), "well_weighted": float(np.mean([np.concatenate(v).mean() for v in by.values()])),
            "mean_width": float(np.concatenate(widths).mean())}


def headline_method(I: Inputs) -> str:
    """Decision method for the headline: lowest D Brier among the fitted methods (dev data only, fixed rule)."""
    D = I.decision["dev_scores"]["D"]
    cands = [k for k in D if k != "prevalence" and D[k].get("brier") is not None]
    return min(cands, key=lambda k: (D[k]["brier"], k))


def representative(I: Inputs, which: str = "median") -> str:
    """Deterministic gap choice on the selected row. median: |MAE - median| smallest, ties by gap id."""
    rows = [r for r in I.rows[I.selected] if r["complete"]]
    if which == "best":
        return min(rows, key=lambda r: (r["mae"], r["key"]))["key"]
    if which == "worst":
        return max(rows, key=lambda r: (r["mae"], r["key"]))["key"]
    med = float(np.median([r["mae"] for r in rows]))
    return min(rows, key=lambda r: (abs(r["mae"] - med), r["key"]))["key"]


# ------------------------------------------------------------------------------------------------ tables
def row_table(I: Inputs) -> list[dict]:
    lin = per_well(I.rows["black_linear_gap"])
    out = []
    for name in ROWS:
        rows = I.rows[name]
        pw = per_well(rows)
        n = sum(r["n_pred"] for r in rows)
        d = np.array([pw[w] - lin[w] for w in pw if w in lin])
        st = I.station.get(name) or {}
        cal = I.calibration.get(name, {})
        b = I.selection["rows"].get(name, {}).get("summary", {}) if name in GRANITE else {}
        cov90, cov50 = coverage(I, name, "p05", "p95"), coverage(I, name, "p25", "p75")
        t = I.final["timing"].get(name, {})
        status: dict = {}
        for r in rows:
            for s in r["status"].split(","):
                status[s] = status.get(s, 0) + 1
        out.append({
            "row": name, "label": LABEL[name],
            "test_pooled_MAE": sum(r["abs_err_sum"] for r in rows) / n,
            "test_pooled_RMSE": float(np.sqrt(sum(r["sq_err_sum"] for r in rows) / n)),
            "test_mean_well_MAE": float(np.mean(list(pw.values()))),
            "test_median_well_MAE": float(np.median(list(pw.values()))),
            "station_MAE": st.get("MAE"), "station_R2": st.get("R2"),
            "station_check_abs_diff": None if st.get("MAE") is None else abs(st["MAE"] - sum(r["abs_err_sum"] for r in rows) / n),
            "median_well_change_vs_linear": float(np.median(d)) if name != "black_linear_gap" else 0.0,
            "wells_improved_vs_linear": int((d < 0).sum()) if name != "black_linear_gap" else None,
            "complete_gaps": sum(r["complete"] for r in rows), "gaps": len(rows),
            "fill_fraction": n / max(sum(r["n_measured"] for r in rows), 1),
            "status": status,
            "gaps_passing_tolerance": sum(1 for r in rows if r["label"] == 1),
            "cover90": cov90, "cover50": cov50,
            "B_mean_well_MAE": b.get("mean_well_MAE"), "D_mean_well_MAE": cal.get("summary", {}).get("mean_well_MAE"),
            "k50": cal.get("calibrator", {}).get("k50"), "k90": cal.get("calibrator", {}).get("k90"),
            "raw80_D": cal.get("raw_80"),
            "fit_s": t.get("fit_s"), "warm_inference_s_20_gaps": t.get("warm_inference_s"),
            "station_model_hash": st.get("model_hash"), "station_git": st.get("git_commit"),
        })
    return out


def bootstrap_table(I: Inputs) -> list[dict]:
    cfg = C.CONFIG
    pairs = [("black_ttm_r3_tuned", "black_ttm_r3_frozen"), ("black_tspulse_tuned", "black_tspulse_frozen"),
             *[(n, "black_linear_gap") for n in GRANITE + ["black_gardner_flank_resid"]],
             (I.selected, "black_gardner_flank_resid"), ("black_ttm_r3_tuned", "black_tspulse_tuned")]
    out = []
    for i, (a, b) in enumerate(pairs):
        res = paired_bootstrap(per_well(I.rows[a]), per_well(I.rows[b]), cfg.bootstrap, cfg.seed + i)
        out.append({"set": "test", "a": a, "b": b, **res})
    for i, (a, b) in enumerate(pairs[:2]):
        if a in I.dev_B and b in I.dev_B:
            res = paired_bootstrap(per_well(I.dev_B[a]), per_well(I.dev_B[b]), cfg.bootstrap, cfg.seed + 100 + i)
            out.append({"set": "B (selection wells)", "a": a, "b": b, **res})
    return out


def decision_table(I: Inputs) -> dict:
    out = {"test": {}, "C": I.decision["dev_scores"]["C"], "D": I.decision["dev_scores"]["D"]}
    curves = I.final["risk_coverage"]
    for k, m in I.final["decision_scores"].items():
        out["test"][k] = {**m, "risk_at_25pct": DEC.risk_at_coverage(curves[k], 0.25),
                          "risk_at_50pct": DEC.risk_at_coverage(curves[k], 0.50)}
    return out


def resources_table(I: Inputs) -> list[dict]:
    out = []
    for arm, t in I.train.items():
        chosen = next((r for r in t["measure"] if r["microbatch"] == t["microbatch"]), {})
        out.append({"arm": arm, "trainable": t["n_trainable"], "parameters": t["n_parameters"],
                    "examples": t["n_examples"], "wells": t["n_wells"], "device": t["device"],
                    "microbatch": t["microbatch"], "accumulation": t["accumulation"],
                    "s_per_update": chosen.get("s_per_update"), "updates_run": t["updates_run"],
                    "train_minutes": t["train_minutes"], "stop_reason": t["stop_reason"],
                    "peak_vram_reserved_mb": t.get("train_peak_reserved_mb"), "process_rss_mb": t.get("process_rss_mb"),
                    "step0_B": t["step0_metric"], "tuned_B": t["best_metric"], "tuned_update": t["best_update"],
                    "step0_won": t.get("step0_won"), "overfit_reload_exact": t["overfit_reload"]["reload_exact"],
                    "overfit_loss": [t["overfit_reload"]["first_loss"], t["overfit_reload"]["last_loss"]],
                    "head_max_abs_change": t["best_head_max_abs_change"], "head_sha256": t.get("head_sha256")})
    return out


def hashes(I: Inputs) -> dict:
    W = I.W
    ident = I.manifest["identity"]
    return {"config_hash": ident.get("config_hash"), "code_hash": ident.get("code_hash"),
            "split_hash": ident.get("split_hash"), "data_hash": ident.get("data_hash"),
            "gaps_hash": ident.get("gaps_hash"),
            "heads": {n: v.get("head_sha256") for n, v in I.manifest["variants"].items()},
            "weights": {n: v.get("weights") for n, v in I.manifest["variants"].items()},
            "predictions_npz_sha256": _sha(W / "final" / "predictions.npz"),
            "decision_sidecar_sha256": _sha(W / "final" / "decision_sidecar.json"),
            "final_json_sha256": _sha(W / "final" / "final.json"),
            "manifest_sha256": _sha(W / "manifest.json"),
            "test_def_sha256": _sha(REPO / "benchmark" / "test_defs" / f"{C.TEST_NAME}.yaml"),
            "manifest_frozen_at": I.manifest["frozen_at"], "git_at_freeze": I.manifest.get("git"),
            "git_at_final": I.final.get("git"), "decima": I.final.get("decima_meta", {}) and
            {k: I.final["decima_meta"].get(k) for k in ("model", "precision", "question", "ms_per_state")}}


# ------------------------------------------------------------------------------------------------ text
def thirty_seconds(I: Inputs, table: list[dict], boot: list[dict], dec: dict, method: str) -> str:
    r = {x["row"]: x for x in table}
    sel = r[I.selected]
    lin, gfr, gf = r["black_linear_gap"], r["black_gardner_flank_resid"], r["black_gardner_fixed"]
    best = min((x for x in table if x["row"] in GRANITE), key=lambda x: x["test_mean_well_MAE"])
    t = dec["test"][method]
    ttm = next(b for b in boot if b["set"] == "test" and b["a"] == "black_ttm_r3_tuned" and b["b"] == "black_ttm_r3_frozen")
    tsp = next(b for b in boot if b["set"] == "test" and b["a"] == "black_tspulse_tuned" and b["b"] == "black_tspulse_frozen")
    pct = lambda a, b: 100 * (b - a) / b   # noqa: E731
    secs = [60 * x["train_minutes"] for x in I.train.values()]
    ups = [x["updates_run"] for x in I.train.values()]
    minutes = (f"early-stopped after {min(ups)}–{max(ups)} updates, {max(secs):.0f} s or less on a laptop GPU"
               if secs else "not run")
    dz = dec["test"].get("decima_raw", {"accepted": 0, "brier": float("nan")})
    zs = [s["p_pass"].get("decima_raw", np.nan) for s in I.sidecar]
    pz = (float(np.nanmin(zs)), float(np.nanmax(zs))) if np.isfinite(zs).any() else (np.nan, np.nan)
    return (
        f"We hid one real 50 m stretch of bulk density in each of the 20 FORCE test wells and asked small IBM "
        f"Granite time-series models to fill it from the density log alone. The row picked in advance on held-out "
        f"training wells ({LABEL[I.selected]}) misses by {sel['test_mean_well_MAE']:.3f} g/cm³ on average, "
        f"{pct(sel['test_mean_well_MAE'], lin['test_mean_well_MAE']):.0f} % better than a straight line "
        f"({lin['test_mean_well_MAE']:.3f}) and {pct(sel['test_mean_well_MAE'], gf['test_mean_well_MAE']):.0f} % better "
        f"than Gardner from sonic ({gf['test_mean_well_MAE']:.3f}); it beats the line in "
        f"{sel['wells_improved_vs_linear']} of 20 wells. The best Granite row on test was {LABEL[best['row']]} "
        f"({best['test_mean_well_MAE']:.3f}). Head-only fine-tuning ({minutes}) moved little: TTM "
        f"{ttm['mean_diff']:+.4f} (95 % CI {ttm['ci95'][0]:+.4f} to {ttm['ci95'][1]:+.4f}), TSPulse "
        f"{tsp['mean_diff']:+.4f} g/cm³. Gardner corrected by the flank residual, a two-line physics baseline, "
        f"reaches {gfr['test_mean_well_MAE']:.3f}. For each gap a small calibrated rule ({METHOD_LABEL[method]}) "
        f"says whether the fill is within 0.05 g/cm³: at p ≥ 0.80 it accepted {t['accepted']} of 20 gaps with "
        f"{t['false_accepts']} false accepts and sent {t['review']} to review; Brier {t['brier']:.3f} vs "
        f"{dec['test']['prevalence']['brier']:.3f} for the base rate. A small verifier model (Decima-small) "
        f"reading the same facts as text accepted {dz['accepted']} gaps (p_pass {pz[0]:.2f}–{pz[1]:.2f}, Brier "
        f"{dz['brier']:.3f}): it barely separates good from bad gaps.")


# ------------------------------------------------------------------------------------------------ figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 1.6, "axes.axisbelow": True})
    return plt


def _context(I: Inputs, key: str, margin_m: float = 30.0):
    """Measured RHOB around the gap from the sealed test well (for drawing only)."""
    from workstreams.black_physics.granite.pipeline import official

    _, te = official()
    w = next(x for x in te if x.name == key.split("|")[0])
    a, b = (int(v) for v in key.split("|")[1].split(":"))
    d = np.asarray(w.depth, float)
    y = np.asarray(w.curves[C.TARGET], float)
    sel = (d >= d[a] - margin_m) & (d <= d[b - 1] + margin_m)
    return d[sel], y[sel], d[a], d[b - 1]


def _gap_panel(ax, I: Inputs, key: str, name: str, *, compact=False, show=("gardner", "linear"), margin_m=30.0):
    d, y, top, bot = _context(I, key, margin_m)
    gd, gt = I.depth(key), I.truth(key)
    ax.axhspan(top, bot, color="#f0efec", zorder=0, lw=0)
    outside = (d < top) | (d > bot)
    ax.plot(np.where(outside, y, np.nan), d, color=INK, lw=1.1, label="Measured (flanks)")
    ax.plot(gt, gd, color=INK, lw=1.1, ls=(0, (1.5, 1.5)), label="Hidden truth")
    lo, hi, p50 = I.arr(name, key, "p05"), I.arr(name, key, "p95"), I.arr(name, key, "p50")
    if lo is not None:
        ax.fill_betweenx(gd, lo, hi, color=BLUE_LIGHT, alpha=0.75, lw=0, label="90 % band")
    if "gardner" in show:
        g = I.arr("black_gardner_fixed", key, "p50")
        ax.plot(g, gd, color=ORANGE, lw=1.2, label="Gardner from sonic")
    if "linear" in show:
        ax.plot(I.arr("black_linear_gap", key, "p50"), gd, color=MUTED, lw=1.0, ls="--", label="Straight line")
    if "flank_resid" in show:
        ax.plot(I.arr("black_gardner_flank_resid", key, "p50"), gd, color=YELLOW, lw=1.4, label="Gardner + flank residual")
    if "ttm" in show and name != "black_ttm_r3_tuned":
        ax.plot(I.arr("black_ttm_r3_tuned", key, "p50"), gd, color=VIOLET, lw=1.2, label="TTM-R3 tuned")
    ax.plot(p50, gd, color=BLUE, lw=2.0, label=LABEL[name])
    ax.set_ylim(d.max(), d.min())
    ax.grid(axis="y", visible=False)
    if not compact:
        ax.set_ylabel("Measured depth (m)")
        ax.set_xlabel("Bulk density RHOB (g/cm³)")


def _p_vs_error(ax, I: Inputs, method: str, highlight: str | None = None):
    thr, tol = C.CONFIG.accept_threshold, C.CONFIG.tolerance_g_cm3
    p = np.array([s["p_pass"][method] for s in I.sidecar])
    e = np.array([s["mae"] for s in I.sidecar])
    acc = p >= thr
    ax.axvspan(thr, 1.0, color="#eef6ee", lw=0, zorder=0)
    ax.axhline(tol, color=INK2, lw=0.9, ls="--")
    ax.axvline(thr, color=INK2, lw=0.9, ls="--")
    ax.scatter(p[~acc], e[~acc], s=46, facecolor=SURF, edgecolor=INK2, lw=1.3, zorder=3, label="Sent to review")
    ax.scatter(p[acc], e[acc], s=46, color=GOOD, edgecolor=SURF, lw=1.5, zorder=3, label="Accepted")
    if highlight is not None:
        i = next(j for j, s in enumerate(I.sidecar) if s["gap_id"] == highlight)
        ax.scatter([p[i]], [e[i]], s=150, facecolor="none", edgecolor=BLUE, lw=2, zorder=4, label="Gap shown left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.12, float(np.nanmax(e)) * 1.08))
    ax.set_xlabel(f"Predicted chance the gap is within {tol:g} g/cm³ (p_pass)")
    ax.set_ylabel("Actual gap MAE (g/cm³)")
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.5, "accept\n" + f"at p ≥ {thr:.2f}", va="center", fontsize=8,
            color=INK2)
    ax.text(0.01, tol, f" tolerance {tol:g}", va="bottom", fontsize=8, color=INK2)
    fa = int(((p >= thr) & (e > tol)).sum())
    return int(acc.sum()), int((~acc).sum()), fa


def fig_pitch(I: Inputs, out: Path, method: str, which: str = "median") -> Path:
    plt = _mpl()
    key = representative(I, which)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    _gap_panel(a1, I, key, I.selected)
    sc = next(s for s in I.sidecar if s["gap_id"] == key)
    a1.set_title(f"One hidden 50 m density gap ({sc['well']})\n{LABEL[I.selected]}: MAE {sc['mae']:.3f} g/cm³, "
                 f"p_pass {sc['p_pass'][method]:.2f}", loc="left")
    a1.legend(loc="lower left", fontsize=7.5)
    acc, rev, fa = _p_vs_error(a2, I, method, key)
    a2.set_title(f"All 20 test gaps: {acc} accepted ({fa} false), {rev} to review\nrule: {METHOD_LABEL[method]}",
                 loc="left")
    a2.legend(loc="upper left", bbox_to_anchor=(0.14, 0.99), fontsize=8)
    fig.tight_layout()
    p = out / f"pitch_{which}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_gallery(I: Inputs, out: Path, method: str) -> Path:  # noqa: C901
    plt = _mpl()
    fig, axes = plt.subplots(4, 5, figsize=(14, 15), sharex=False)
    order = sorted(I.sidecar, key=lambda s: s["mae"])
    for ax, s in zip(axes.flat, order):
        _gap_panel(ax, I, s["gap_id"], I.selected, compact=True, show=("linear", "ttm"), margin_m=20)
        acc = s["p_pass"][method] >= C.CONFIG.accept_threshold
        ax.set_title(f"{s['well']}  MAE {s['mae']:.3f}\np_pass {s['p_pass'][method]:.2f} "
                     f"{'✓ accept' if acc else '· review'}", fontsize=8, loc="left",
                     color=GOOD if acc else INK2)
        ax.tick_params(labelsize=7)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(f"All 20 test gaps, sorted by error of the selected row ({LABEL[I.selected]}); "
                 f"decision rule: {METHOD_LABEL[method]}", x=0.01, ha="left", y=0.995, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    p = out / "gallery_test_gaps.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_leaderboard(I: Inputs, table: list[dict], boot: list[dict], out: Path) -> Path:
    plt = _mpl()
    lin = per_well(I.rows["black_linear_gap"])
    rows = sorted(table, key=lambda r: -r["test_mean_well_MAE"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    rng = np.random.default_rng(C.CONFIG.seed)
    for i, r in enumerate(rows):
        pw = np.array(list(per_well(I.rows[r["row"]]).values()))
        boots = pw[rng.integers(0, len(pw), (C.CONFIG.bootstrap, len(pw)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        col = BLUE if r["row"] == I.selected else ("#86b6ef" if r["row"] in GRANITE else ROW_COLOR[r["row"]])
        ax.barh(i, r["test_mean_well_MAE"], color=col, height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.2)
        ax.text(hi + 0.002, i, f"{r['test_mean_well_MAE']:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), [r["label"] + (" (selected on B)" if r["row"] == I.selected else "") for r in rows])
    ax.axvline(np.mean(list(lin.values())), color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Test mean per-well MAE (g/cm³), 20 wells, 95 % bootstrap CI over wells")
    ax.set_title("Filling a hidden 50 m density gap: lower is better", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = out / "leaderboard.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_paired(I: Inputs, boot: list[dict], out: Path) -> Path:
    from matplotlib.ticker import MaxNLocator

    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    lin = per_well(I.rows["black_linear_gap"])
    pairs = [("black_ttm_r3_tuned", "black_ttm_r3_frozen", "TTM-R3: tuned − frozen"),
             ("black_tspulse_tuned", "black_tspulse_frozen", "TSPulse: tuned − frozen"),
             (I.selected, "black_linear_gap", f"{LABEL[I.selected]} − straight line")]
    wells = sorted(lin, key=lambda w: lin[w])
    for ax, (a, b, title) in zip(axes, pairs):
        pa, pb = per_well(I.rows[a]), per_well(I.rows[b])
        d = np.array([pa[w] - pb[w] for w in wells])
        ax.axvline(0, color=AXIS, lw=1)
        ax.scatter(d, range(len(d)), s=40, color=np.where(d < 0, BLUE, RED), edgecolor=SURF, lw=1.2, zorder=3)
        res = next(x for x in boot if x["set"] == "test" and x["a"] == a and x["b"] == b)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.axvspan(*res["ci95"], color=GRID, alpha=0.7, lw=0)
        ax.axvline(res["mean_diff"], color=INK, lw=1.2)
        ax.set_title(f"{title}\nmean {res['mean_diff']:+.4f} [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]; "
                     f"better in {res['wells_better']}/{res['n_wells']}", loc="left", fontsize=9)
        ax.set_xlabel("Per-well MAE difference (g/cm³); left = better")
    axes[0].set_yticks(range(len(wells)), wells, fontsize=7)
    axes[0].set_ylabel("Test well (sorted by straight-line error)")
    fig.tight_layout()
    p = out / "paired_deltas.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_training(I: Inputs, out: Path) -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    for j, arm in enumerate(C.ARMS):
        t = I.train.get(arm)
        if t is None:
            continue
        h = t["history"]
        u = [x["update"] for x in h]
        axes[0, j].plot(u, [x["B_metric"] for x in h], color=BLUE, marker="o", ms=4)
        axes[0, j].axhline(t["step0_metric"], color=MUTED, ls="--", lw=0.9)
        axes[0, j].set_title(f"{'TTM-R3' if arm == 'ttm' else 'TSPulse'}: B mean well MAE (selection wells)", loc="left")
        axes[0, j].set_ylabel("g/cm³")
        lu = [x["update"] for x in h if x["train_loss"] is not None]
        axes[1, j].plot(lu, [x["train_loss"] for x in h if x["train_loss"] is not None], color=VIOLET, marker="o", ms=4)
        axes[1, j].set_title("Training loss (mean of the last 25 updates)", loc="left")
        axes[1, j].set_xlabel(f"Head update (stop: {t['stop_reason']}; {t['train_minutes']:.2f} min)")
    fig.tight_layout()
    p = out / "training_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_decisions(I: Inputs, dec: dict, out: Path) -> Path:
    plt = _mpl()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    show = ["width_only", "numeric_cal", "numeric_gardner_cal", "decima_cal"]
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    for k, c in zip(show, colors):
        cur = I.final["risk_coverage"].get(k)
        if not cur:
            continue
        a1.step([x["coverage"] for x in cur], [x["risk"] if x["risk"] is not None else np.nan for x in cur],
                where="post", color=c, label=METHOD_LABEL[k])
    prev = dec["test"]["prevalence"]["prevalence"]
    a1.axhline(1 - prev, color=MUTED, ls="--", lw=0.9, label="Accept everything")
    a1.set_xlabel("Coverage (share of test gaps accepted, most confident first)")
    a1.set_ylabel("Risk (share of accepted gaps with MAE > 0.05)")
    a1.set_title("Risk–coverage on the 20 test gaps", loc="left")
    a1.legend(fontsize=7.5)
    width = 0.2
    for i, (k, c) in enumerate(zip(show, colors)):
        bins = dec["test"][k]["reliability"]
        xs = np.arange(len(bins)) + (i - 1.5) * width
        a2.bar(xs, [b["pass_rate"] if b["pass_rate"] is not None else 0 for b in bins], width * 0.9, color=c,
               label=METHOD_LABEL[k])
        a2.scatter(xs, [b["mean_p"] if b["mean_p"] is not None else np.nan for b in bins], color=INK, s=12, zorder=3)
    a2.set_xticks(range(3), [b["bin"] for b in dec["test"]["width_only"]["reliability"]])
    a2.set_xlabel("Predicted p_pass bin (dots: mean predicted p)")
    a2.set_ylabel("Observed pass rate")
    a2.set_title("Reliability, 3 bins (test)", loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    p = out / "decisions_risk_reliability.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_coverage(I: Inputs, table: list[dict], out: Path) -> Path:
    plt = _mpl()
    rows = [r for r in table if r["cover90"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, r in enumerate(rows):
        ax.scatter(r["cover90"]["well_weighted"], i, color=BLUE, s=50, zorder=3, label="90 % band" if i == 0 else None)
        if r["cover50"]:
            ax.scatter(r["cover50"]["well_weighted"], i, color=ORANGE, s=50, zorder=3, label="50 % band" if i == 0 else None)
        ax.text(1.005, i, f"width90 {r['cover90']['mean_width']:.3f}", va="center", fontsize=7.5, color=INK2,
                transform=ax.get_yaxis_transform())
    for v in (0.5, 0.9):
        ax.axvline(v, color=MUTED, ls="--", lw=0.9)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Observed coverage on test (well-weighted); dashed = nominal")
    ax.set_title("Interval calibration (bands fitted on D wells)", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = out / "coverage.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_coefficients(I: Inputs, out: Path) -> Path:
    plt = _mpl()
    rule = I.decision["rules"]["numeric_gardner"]
    cols, beta = rule["columns"], rule["beta"]
    pairs = [(c, b) for c, b in zip(cols, beta) if c != "intercept"]
    pairs.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(range(len(pairs)), [b for _, b in pairs], color=[BLUE if b < 0 else RED for _, b in pairs], height=0.6)
    ax.set_yticks(range(len(pairs)), [c for c, _ in pairs])
    ax.axvline(0, color=AXIS)
    ax.set_xlabel("Standardized logistic coefficient (fitted on C); left = less likely to pass")
    ax.set_title("What the numeric + Gardner rule looks at", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = out / "decision_coefficients.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_extension(I: Inputs, out: Path) -> Path:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lin = {r["key"]: r["mae"] for r in I.rows["black_linear_gap"]}
    for name, col, lab in [("black_gardner_flank_resid", YELLOW, "Gardner + flank residual"),
                           (I.selected, BLUE, LABEL[I.selected])]:
        m = {r["key"]: r["mae"] for r in I.rows[name]}
        ax.scatter([lin[k] for k in lin], [m[k] for k in lin], s=40, color=col, edgecolor=SURF, lw=1.2, label=lab,
                   zorder=3)
    hi = max(max(lin.values()), 0.15) * 1.05
    ax.plot([0, hi], [0, hi], color=AXIS, lw=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Straight-line gap MAE (g/cm³)")
    ax.set_ylabel("Gap MAE (g/cm³)")
    ax.set_title("Per test gap: below the diagonal beats the line", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "extension_vs_linear.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ writing
def _f(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return "–" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def md_tables(I, table, boot, dec, res, method, aggregate_only: bool) -> str:
    L = []
    L.append("| Row | Test mean per-well MAE | Pooled MAE (station check) | RMSE | Median per-well Δ vs line | "
             "Wells better than line | Support | Gaps ≤ 0.05 | Cover 90 % (width) | Cover 50 % | B mean well MAE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        st = "" if r["station_MAE"] is None else f" ({r['station_MAE']:.4f})"
        c90 = "–" if not r["cover90"] else f"{r['cover90']['well_weighted']:.2f} ({r['cover90']['mean_width']:.3f})"
        c50 = "–" if not r["cover50"] else f"{r['cover50']['well_weighted']:.2f}"
        sup = f"{r['complete_gaps']}/{r['gaps']}"
        L.append(f"| {r['label']}{' ★' if r['row'] == I.selected else ''} | {_f(r['test_mean_well_MAE'])} | "
                 f"{_f(r['test_pooled_MAE'])}{st} | {_f(r['test_pooled_RMSE'])} | {_f(r['median_well_change_vs_linear'])} | "
                 f"{_f(r['wells_improved_vs_linear'])} | {sup} | {r['gaps_passing_tolerance']}/20 | {c90} | {c50} | "
                 f"{_f(r['B_mean_well_MAE'])} |")
    for k, why in UNAVAILABLE.items():
        L.append(f"| {k} | unavailable | | | | | | | | | {why} |")
    L.append("")
    L.append("★ selected before the test on B (lowest B mean well-level MAE). Test wells hold one gap each, so per-well "
             "MAE = gap MAE. Coverage is well-weighted over measured gap samples; bands are D-calibrated.")
    L.append("")
    L.append("**Paired bootstrap** (mean of per-well MAE differences a − b; 2,000 resamples of wells; negative = a better)")
    L.append("")
    L.append("| Set | a | b | Mean Δ | 95 % CI | P(Δ<0) | Wells a better / worse |")
    L.append("|---|---|---|---|---|---|---|")
    for b in boot:
        if b.get("n_wells"):
            L.append(f"| {b['set']} | {LABEL[b['a']]} | {LABEL[b['b']]} | {b['mean_diff']:+.4f} | "
                     f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | {b['p_boot_lt0']:.2f} | "
                     f"{b['wells_better']} / {b['wells_worse']} |")
    L.append("")
    L.append("**Decisions** on the selected reconstruction (accept when p_pass ≥ 0.80; label = gap MAE ≤ 0.05)")
    L.append("")
    L.append("| Method | Test Brier | Accepted | Review | False accepts | Risk @25 % cov. | Risk @50 % cov. | "
             "Reliability bins (n: mean p → pass rate) | D Brier | C Brier |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, m in dec["test"].items():
        rel = "; ".join(f"{b['n']}: {_f(b['mean_p'], 2)}→{_f(b['pass_rate'], 2)}" for b in m["reliability"])
        L.append(f"| {METHOD_LABEL.get(k, k)}{' ◆' if k == method else ''} | {_f(m['brier'], 3)} | {m['accepted']} | "
                 f"{m['review']} | {m['false_accepts']} | {_f(m['risk_at_25pct'], 2)} | {_f(m['risk_at_50pct'], 2)} | "
                 f"{rel} | {_f(dec['D'].get(k, {}).get('brier'), 3)} | {_f(dec['C'].get(k, {}).get('brier'), 3)} |")
    L.append("")
    L.append(f"◆ headline rule: lowest D Brier among fitted methods (dev data only). Pass rates: C "
             f"{I.decision['pass_rate']['C']:.2f}, D {I.decision['pass_rate']['D']:.2f}, test "
             f"{dec['test']['prevalence']['prevalence']:.2f}.")
    L.append("")
    L.append("**Training and resources** (head-only fine-tune on A wells, laptop GPU)")
    L.append("")
    L.append("| Arm | Trainable / total | Examples (wells) | Microbatch × accum | s/update | Updates | Minutes | Stop | "
             "Peak VRAM MB | RSS MB | B step 0 → tuned (update) | Overfit/reload exact |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        L.append(f"| {r['arm']} | {r['trainable']:,} / {r['parameters']:,} | {r['examples']} ({r['wells']}) | "
                 f"{r['microbatch']} × {r['accumulation']} | {_f(r['s_per_update'], 3)} | {r['updates_run']} | "
                 f"{r['train_minutes']:.2f} | {r['stop_reason']} | {_f(r['peak_vram_reserved_mb'], 0)} | "
                 f"{_f(r['process_rss_mb'], 0)} | {r['step0_B']:.5f} → {r['tuned_B']:.5f} ({r['tuned_update']}) | "
                 f"{r['overfit_reload_exact']} |")
    L.append("")
    L.append("| Row | k50 | k90 | Raw TTM 80 % coverage on D (well-weighted) | Fit s | Warm inference s (20 gaps) |")
    L.append("|---|---|---|---|---|---|")
    for r in table:
        if r["row"] in GRANITE:
            raw = r["raw80_D"]
            L.append(f"| {r['label']} | {_f(r['k50'], 3)} | {_f(r['k90'], 3)} | "
                     f"{'–' if not raw else format(raw["well_weighted"], ".3f")} | {_f(r['fit_s'], 2)} | "
                     f"{_f(r['warm_inference_s_20_gaps'], 3)} |")
    return "\n".join(L)


def build() -> dict:
    I = Inputs()
    out = I.W / "report"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    table = row_table(I)
    boot = bootstrap_table(I)
    dec = decision_table(I)
    res = resources_table(I)
    method = headline_method(I)
    text = thirty_seconds(I, table, boot, dec, method)
    H = hashes(I)
    figures = [fig_pitch(I, figs, method, w) for w in ("median", "best", "worst")]
    figures += [fig_gallery(I, figs, method), fig_leaderboard(I, table, boot, figs), fig_paired(I, boot, figs),
                fig_training(I, figs), fig_decisions(I, dec, figs), fig_coverage(I, table, figs),
                fig_coefficients(I, figs), fig_extension(I, figs)]
    replay = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "test": C.TEST_NAME, "selected": I.selected,
              "headline_decision": method, "rows": table, "bootstrap": boot, "decisions": dec, "resources": res,
              "unavailable": UNAVAILABLE, "hashes": H, "thirty_seconds": text,
              "replay": ["python -m workstreams.black_physics.granite all",
                         "WMB_BLACK_GRANITE=1 python -m benchmark.run --model black_ttm_r3_frozen,black_ttm_r3_tuned,"
                         "black_tspulse_frozen,black_tspulse_tuned,black_linear_gap,black_gardner_flank_resid,"
                         "black_gardner_fixed --test rhob_gap50 --predictions npz --rerun",
                         "python -m workstreams.black_physics.granite final",
                         "python -m workstreams.black_physics.granite report"]}
    (out / "results_table.json").write_text(json.dumps(replay, indent=1, default=float), encoding="utf-8")
    tables = md_tables(I, table, boot, dec, res, method, aggregate_only=False)
    gaps = ["| Gap | Status | " + " | ".join(LABEL[n] for n in ROWS) + f" | p_pass ({method}) | Decision |",
            "|---|---|" + "---|" * (len(ROWS) + 2)]
    thr = C.CONFIG.accept_threshold
    for s in sorted(I.sidecar, key=lambda s: s["gap_id"]):
        maes = [next(r["mae"] for r in I.rows[n] if r["key"] == s["gap_id"]) for n in ROWS]
        gaps.append(f"| {s['well']} {s['start_md']:.1f}–{s['end_md']:.1f} m | {s['status']} | "
                    + " | ".join(f"{m:.4f}" for m in maes)
                    + f" | {s['p_pass'][method]:.2f} | {'accept' if s['p_pass'][method] >= thr else 'review'} |")
    local = (f"# Granite 50 m density gap: local report\n\nGenerated {replay['generated_at']} from frozen artifacts "
             f"(manifest frozen {I.manifest['frozen_at']}). Contains per-well rows: keep local.\n\n"
             f"## 30-second result\n\n{text}\n\n## Tables\n\n{tables}\n\n## Per test gap (MAE, g/cm³)\n\n"
             + "\n".join(gaps) + "\n\n## Figures\n\n" + "\n".join(f"- `{p.name}`" for p in figures)
             + "\n\n## Hashes\n\n```json\n" + json.dumps(H, indent=1) + "\n```\n")
    (out / "REPORT.md").write_text(local, encoding="utf-8")
    agg = (f"# Granite 50 m density gap: results (aggregate only)\n\n"
           f"Written by `python -m workstreams.black_physics.granite report` from the frozen artifacts; no well names, "
           f"depths or curve values. Test `{C.TEST_NAME}`: one hidden 50 m RHOB gap in each of the 20 FORCE 2020 test "
           f"wells (6,600 samples). Code hash `{H['code_hash']}`, config `{H['config_hash']}`, split `{H['split_hash']}`, "
           f"data `{H['data_hash']}`; predictions sha256 `{(H['predictions_npz_sha256'] or '')[:16]}`, decision sidecar "
           f"`{(H['decision_sidecar_sha256'] or '')[:16]}` (local files).\n\n## 30-second result\n\n{text}\n\n"
           f"## Tables\n\n{tables}\n")
    (HERE / "RESULTS.md").write_text(agg, encoding="utf-8")
    from workstreams.black_physics.granite.pipeline import log

    log(f"report: {out / 'REPORT.md'}, {len(figures)} figures, RESULTS.md; headline decision {method}")
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(text)
    for p in figures:
        print(p)
    return replay


if __name__ == "__main__":
    build()
