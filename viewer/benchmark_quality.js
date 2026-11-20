/* Good / ok / poor rules, colours and pure helpers of the Benchmark tab (DOM-free, node-tested).
   The rules are documented in docs/benchmark_quality.md; change them here only. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbBenchmarkQuality = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const EPS = 1e-9; // boundaries are inclusive; absorbs float noise such as 0.93 - 0.90

  const COLORS = {
    good: { fg: "#157A54", bg: "#D4EFDF" },
    ok: { fg: "#A56B0C", bg: "#F8E6C0" },
    poor: { fg: "#C03A32", bg: "#F8D4D0" },
    muted: { fg: "#8A8F94", bg: "#E8EAEC" },
  };
  const WORDS = { good: "good", ok: "ok", poor: "poor", muted: "n/a" };

  const THRESHOLDS = {
    accuracy: { good: 0.10, ok: 0.30 },     // relative excess over the best model
    calibration: { good: 0.03, ok: 0.08 },  // |cover_90 - 0.90|
    sharpness: { good: 0.15, ok: 0.45 },    // relative excess over the best calibrated model
    speed: { realTime: 0.05, batch: 0.5 },  // seconds per well
  };
  const TARGET_COVERAGE = 0.90;
  const CAL_SCALE = { min: 0.6, max: 1.0 };
  const CAL_CLOSENESS_FULL_SCALE = 0.20; // mean deviation at which the All-mode closeness bar is empty
  const BASELINE_MODEL = "template_linear";

  const SPEED_AXIS = { minLog: -3, maxLog: 2, ticks: [0.001, 0.01, 0.1, 1, 10, 100], tickLabels: ["1 ms", "10 ms", "0.1 s", "1 s", "10 s", "100 s"] };
  const SPEED_ZONES = [
    { id: "good", name: "real-time", from: 0, to: THRESHOLDS.speed.realTime },
    { id: "ok", name: "batch-OK", from: THRESHOLDS.speed.realTime, to: THRESHOLDS.speed.batch },
    { id: "poor", name: "slow", from: THRESHOLDS.speed.batch, to: Infinity },
  ];

  const PROPERTIES = ["RHOB", "DTC", "DTS"];
  const PROPERTY_NAMES = { RHOB: "Density", DTC: "Compressional sonic", DTS: "Shear sonic" };
  const UNITS = { RHOB: "g/cm³", DTC: "µs/ft", DTS: "µs/ft" };
  const ALL_UNIT = "x baseline";

  function num(v) { return typeof v === "number" && Number.isFinite(v) ? v : null; }

  function unitOf(prop, all) { return all ? ALL_UNIT : UNITS[prop] || ""; }

  // ---------- classifiers ----------

  // value and best are errors (lower better); best is the best value in the same column / view.
  function classifyAccuracy(value, best) {
    if (num(value) == null || num(best) == null) return null;
    if (value <= best + EPS) return "good";
    const rel = best > 0 ? value / best - 1 : Infinity;
    if (rel <= THRESHOLDS.accuracy.good + EPS) return "good";
    return rel <= THRESHOLDS.accuracy.ok + EPS ? "ok" : "poor";
  }

  // dev = |cover_90 - 0.90| (or its mean over properties in All mode).
  function classifyCalibration(dev) {
    if (num(dev) == null) return null;
    if (dev <= THRESHOLDS.calibration.good + EPS) return "good";
    return dev <= THRESHOLDS.calibration.ok + EPS ? "ok" : "poor";
  }

  // bestCalibrated: narrowest width among models whose calibration is good or ok. Returns "muted" when it does not count.
  function classifySharpness(width, bestCalibrated, calClass) {
    if (num(width) == null || calClass == null) return null;
    if (calClass === "poor" || num(bestCalibrated) == null) return "muted";
    if (width <= bestCalibrated + EPS) return "good";
    const rel = bestCalibrated > 0 ? width / bestCalibrated - 1 : Infinity;
    if (rel <= THRESHOLDS.sharpness.good + EPS) return "good";
    return rel <= THRESHOLDS.sharpness.ok + EPS ? "ok" : "poor";
  }

  function speedZone(secondsPerWell) {
    if (num(secondsPerWell) == null) return null;
    if (secondsPerWell <= THRESHOLDS.speed.realTime + EPS) return "good";
    return secondsPerWell <= THRESHOLDS.speed.batch + EPS ? "ok" : "poor";
  }

  function speedZoneName(zoneId) {
    const z = SPEED_ZONES.find((x) => x.id === zoneId);
    return z ? z.name : "";
  }

  // Plain-language calibration word for one model.
  function calibrationWord(cover, calClass) {
    if (num(cover) == null) return "";
    if (calClass === "good") return "on target";
    return cover < TARGET_COVERAGE ? "overconfident" : "too cautious";
  }

  // All-mode words: only the distance is known, not its sign.
  function calibrationWordAll(calClass) {
    return calClass === "good" ? "on target" : calClass === "ok" ? "a bit off" : "off target";
  }

  // ---------- geometry helpers ----------

  // 0..1 position of a coverage on the 60-100% scale (clamped).
  function calScaleFrac(cover) {
    return Math.min(1, Math.max(0, (cover - CAL_SCALE.min) / (CAL_SCALE.max - CAL_SCALE.min)));
  }

  const CAL_TARGET_FRAC = calScaleFrac(TARGET_COVERAGE);

  // All-mode closeness bar: 1 = on target (0 off), 0 = CAL_CLOSENESS_FULL_SCALE or more off.
  function calClosenessFrac(dev) {
    return Math.min(1, Math.max(0, 1 - dev / CAL_CLOSENESS_FULL_SCALE));
  }

  // Log dial position 0..1: 0.001 s -> 0, 100 s -> 1.
  function speedFrac(t) {
    if (num(t) == null) return null;
    const lg = Math.log10(Math.max(t, 1e-12));
    return Math.min(1, Math.max(0, (lg - SPEED_AXIS.minLog) / (SPEED_AXIS.maxLog - SPEED_AXIS.minLog)));
  }

  function formatSeconds(t) {
    if (num(t) == null) return "–";
    if (t >= 1) return (t >= 10 ? t.toFixed(0) : t.toFixed(1)) + " s";
    const ms = t * 1000;
    return (ms >= 10 ? ms.toFixed(0) : ms >= 1 ? ms.toFixed(1) : ms.toFixed(2)) + " ms";
  }

  // Non-dominated points when both x and y are lower-better; returned sorted by x. Points need finite x and y.
  function paretoFront(points) {
    const pts = points.filter((p) => num(p.x) != null && num(p.y) != null)
      .sort((a, b) => a.x - b.x || a.y - b.y);
    const front = [];
    let bestY = Infinity;
    for (const p of pts) {
      if (p.y < bestY - EPS) { front.push(p); bestY = p.y; }
    }
    return front;
  }

  // ---------- red-to-green colour scale (scatter colour dimension) ----------

  function piecewise(v, stops) {
    if (v <= stops[0][0]) return stops[0][1];
    for (let k = 1; k < stops.length; k++) {
      const [x0, t0] = stops[k - 1], [x1, t1] = stops[k];
      if (v <= x1) return t0 + ((v - x0) / (x1 - x0)) * (t1 - t0);
    }
    return stops[stops.length - 1][1];
  }

  // Goodness 0 (worst) .. 1 (best) of one measure, consistent with the good / ok / poor boundaries:
  // the good boundary maps to 0.75, the ok boundary to 0.4 and twice the ok boundary (or beyond) to 0.
  // dim: "acc" | "width" (relative to best, the lowest value shown) | "dev" (|cover_90 - 0.90|) | "speed" (s per well).
  function goodness(dim, value, best) {
    if (num(value) == null) return null;
    const ladder = (good, ok) => [[0, 1], [good, 0.75], [ok, 0.4], [2 * ok, 0]];
    if (dim === "dev") return piecewise(value, ladder(THRESHOLDS.calibration.good, THRESHOLDS.calibration.ok));
    if (dim === "speed") {
      const lg = (t) => Math.log10(Math.max(t, 1e-12));
      const s = THRESHOLDS.speed;
      return piecewise(lg(value), [[lg(s.realTime / 10), 1], [lg(s.realTime), 0.75], [lg(s.batch), 0.4], [lg(s.batch * 10), 0]]);
    }
    const t = dim === "acc" ? THRESHOLDS.accuracy : dim === "width" ? THRESHOLDS.sharpness : null;
    if (!t || num(best) == null || best <= 0) return null;
    return piecewise(Math.max(0, value / best - 1), ladder(t.good, t.ok));
  }

  const RAMP = [[0, "#C03A32"], [0.5, "#E0A21B"], [1, "#2E9A5E"]];
  function rgb(hex) { return [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16)); }

  // Hex colour for a goodness 0..1: red (worse) -> amber -> green (better). null -> grey.
  function goodnessColor(t) {
    if (num(t) == null) return COLORS.muted.fg;
    const v = Math.min(1, Math.max(0, t));
    const k = v <= RAMP[1][0] ? 1 : 2;
    const [t0, c0] = RAMP[k - 1], [t1, c1] = RAMP[k];
    const f = (v - t0) / (t1 - t0), a = rgb(c0), b = rgb(c1);
    return "#" + a.map((x, i) => Math.round(x + f * (b[i] - x)).toString(16).padStart(2, "0")).join("");
  }

  // ---------- scoring a view ----------

  // items: [{id, acc, width, dev, speed}] (acc/width lower better; dev = distance from 90% coverage). Adds classes and ranks.
  function scoreItems(items) {
    const finite = (k) => items.map((i) => num(i[k])).filter((v) => v != null);
    const bestAcc = Math.min.apply(null, finite("acc"));
    const out = items.map((i) => {
      const calClass = classifyCalibration(i.dev);
      return Object.assign({}, i, { accClass: classifyAccuracy(i.acc, bestAcc), calClass, speedClass: speedZone(i.speed) });
    });
    const calibratedWidths = out.filter((i) => (i.calClass === "good" || i.calClass === "ok") && num(i.width) != null).map((i) => i.width);
    const bestWidth = calibratedWidths.length ? Math.min.apply(null, calibratedWidths) : null;
    for (const i of out) i.sharpClass = classifySharpness(i.width, bestWidth, i.calClass);
    const rank = (key) => {
      const order = out.filter((i) => num(i[key]) != null).sort((a, b) => a[key] - b[key]);
      order.forEach((i, n) => { i[key + "Rank"] = n + 1; });
    };
    rank("acc"); rank("width"); rank("speed");
    out.filter((i) => num(i.dev) != null).sort((a, b) => a.dev - b.dev).forEach((i, n) => { i.calRank = n + 1; });
    return out;
  }

  // ---------- All mode ----------

  function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : null; }

  // rows: scoreboard rows of one run. Per model: accuracy / sharpness relative to the baseline model, mean coverage
  // error and mean speed. A property missing for the baseline or the model is skipped in that mean.
  function aggregateAll(rows, baseline) {
    const base = baseline || BASELINE_MODEL;
    const baseRows = {};
    for (const r of rows) if (r.contender === base) baseRows[r.target] = r;
    const models = Array.from(new Set(rows.map((r) => r.contender)));
    return models.map((m) => {
      const mine = rows.filter((r) => r.contender === m);
      const rel = (key) => mean(mine.map((r) => {
        const b = baseRows[r.target];
        const v = num(r[key]), bv = b ? num(b[key]) : null;
        return v != null && bv != null && bv > 0 ? v / bv : null;
      }).filter((v) => v != null));
      const devs = mine.map((r) => (num(r.cover_90) == null ? null : Math.abs(r.cover_90 - TARGET_COVERAGE))).filter((v) => v != null);
      const speeds = mine.map((r) => num(r.predict_s_per_well)).filter((v) => v != null);
      return {
        id: m,
        acc: rel("MAE"),
        width: rel("interval_width_90"),
        dev: mean(devs),
        speed: mean(speeds),
        props: mine.map((r) => r.target),
      };
    });
  }

  // Keep the cluster linear; only an isolated maximum gets a compressed tail.
  function comparisonScale(values, domainMax = 0) {
    const sorted = values.filter((v) => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
    const peak = sorted[sorted.length - 1] || 0;
    const next = sorted[sorted.length - 2];
    const max = Math.max(domainMax, peak) || 1;
    const knee = peak > 4 * next ? next : null;
    function fraction(value) {
      if (!Number.isFinite(value) || value <= 0) return 0;
      if (knee === null) return Math.min(1, value / max);
      return value <= knee ? 0.8 * (value / knee) : Math.min(1, 0.8 + 0.2 * ((value - knee) / (max - knee)));
    }
    return { max, knee, fraction };
  }

  return {
    comparisonScale, EPS, COLORS, WORDS, THRESHOLDS, TARGET_COVERAGE, CAL_SCALE, CAL_TARGET_FRAC, BASELINE_MODEL,
    SPEED_AXIS, SPEED_ZONES, PROPERTIES, PROPERTY_NAMES, UNITS, ALL_UNIT,
    unitOf, classifyAccuracy, classifyCalibration, classifySharpness, speedZone, speedZoneName,
    calibrationWord, calibrationWordAll, calScaleFrac, calClosenessFrac, speedFrac, formatSeconds,
    paretoFront, scoreItems, aggregateAll, goodness, goodnessColor,
  };
});
