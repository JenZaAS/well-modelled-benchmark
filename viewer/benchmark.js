/* Tab registry + Benchmark tab for the live viewer. Pure helpers are DOM-free (node-tested); mount() needs a browser. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbBenchmark = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const Q = (typeof module === "object" && module.exports) ? require("./benchmark_quality.js") : globalThis.WmbBenchmarkQuality;

  // ---------- tabs ----------

  const BENCHMARK_VIEWS = ["summary-rhob", "summary-dtc", "summary-dts", "summary-all", "byproperty", "welllogs"];

  // "#benchmark" or "#benchmark/summary-rhob" -> "benchmark" when it is a known tab id, else fallback.
  function parseTabHash(hash, ids, fallback) {
    const h = String(hash || "").replace(/^#/, "");
    let id = h;
    try { id = decodeURIComponent(h); } catch (e) { /* keep raw */ }
    const base = id.split("/")[0];
    return ids.indexOf(base) >= 0 ? base : fallback;
  }

  function tabHash(id) { return "#" + encodeURIComponent(id); }

  // "#benchmark/summary-dtc" -> "summary-dtc"; null when the hash carries no valid Benchmark view.
  function parseBenchmarkView(hash) {
    const h = String(hash || "").replace(/^#/, "");
    const parts = h.split("/");
    return parts[0] === "benchmark" && BENCHMARK_VIEWS.indexOf(parts[1]) >= 0 ? parts[1] : null;
  }

  function benchmarkHash(view) { return "#benchmark/" + view; }

  // "summary-rhob" -> {mode: "summary", prop: "RHOB"}; "summary-all" -> prop "ALL";
  // "byproperty" / "welllogs" -> {mode, prop: null}.
  function viewState(view) {
    if (view === "byproperty") return { mode: "byproperty", prop: null };
    if (view === "welllogs") return { mode: "welllogs", prop: null };
    return { mode: "summary", prop: String(view || "").replace(/^summary-/, "").toUpperCase() };
  }

  function viewId(mode, prop) {
    if (mode === "byproperty") return "byproperty";
    if (mode === "welllogs") return "welllogs";
    return "summary-" + String(prop).toLowerCase();
  }

  // Default Summary property: RHOB when the run has it, else the first available one.
  function defaultProperty(props) {
    return props.indexOf("RHOB") >= 0 ? "RHOB" : props[0] || null;
  }

  // Registry of {id, title, pane, show(pane), disabled, tooltip}; renders buttons into opts.bar and toggles pane[hidden].
  // A disabled tab is shown greyed out, is not listed in ids() and can never become current.
  function createTabs(opts) {
    const list = [];
    let current = null;
    function render() {
      if (!opts.bar) return;
      opts.bar.innerHTML = "";
      for (const t of list) {
        const b = document.createElement("button");
        b.type = "button"; b.setAttribute("role", "tab"); b.textContent = t.title;
        b.setAttribute("aria-selected", String(t.id === current));
        if (t.disabled) {
          b.disabled = true;
          b.setAttribute("aria-disabled", "true");
          if (t.tooltip) b.title = t.tooltip;
        } else b.addEventListener("click", () => api.select(t.id));
        opts.bar.appendChild(b);
      }
    }
    const api = {
      register(tab) { list.push(tab); render(); return api; },
      ids() { return list.filter((t) => !t.disabled).map((t) => t.id); },
      current() { return current; },
      select(id) {
        const wanted = list.find((x) => x.id === id);
        if (wanted && wanted.disabled && current != null) return;
        const t = (wanted && !wanted.disabled && wanted) || list.find((x) => x.id === opts.defaultId && !x.disabled) || list.find((x) => !x.disabled);
        if (!t) return;
        current = t.id;
        for (const o of list) if (o.pane) o.pane.hidden = o.id !== t.id;
        render();
        if (t.show) t.show(t.pane);
        if (opts.onChange) opts.onChange(t.id);
      },
    };
    return api;
  }

  // ---------- scoreboard ----------

  const METRICS = [
    { key: "MAE", label: "MAE", dir: "low", fmt: 3,
      tip: "Mean absolute error: average absolute miss, in the units of the curve. Lower is better." },
    { key: "RMSE", label: "RMSE", dir: "low", fmt: 3,
      tip: "Root-mean-square error: like MAE but penalises large misses more. Lower is better." },
    { key: "R2", label: "R2", dir: "high", fmt: 3,
      tip: "Fraction of variance explained. 1 is perfect, 0 is no better than predicting the mean. Higher is better." },
    { key: "cover_90", label: "cover_90", dir: "target", ideal: 0.9, fmt: 2,
      tip: "Share of true values inside the 90% interval. 0.90 is ideally calibrated." },
    { key: "cover_50", label: "cover_50", dir: "target", ideal: 0.5, fmt: 2,
      tip: "Share of true values inside the 50% interval. 0.50 is ideal." },
    { key: "sharpness", label: "sharpness", dir: "low", fmt: 3, value: sharpnessOf,
      tip: "How tight the predicted interval or sigma is (smaller is sharper). Only meaningful if coverage is decent. Uses mean sigma, or the 90% interval width when sigma is missing." },
    { key: "speed", label: "speed (s)", dir: "low", fmt: 3, value: (r) => num(r.predict_s),
      tip: "Seconds to predict all test wells (predict_s); lower is faster. Hover a cell for samples per second." },
  ];

  function num(v) { return typeof v === "number" && Number.isFinite(v) ? v : null; }

  function sharpnessOf(r) {
    const s = num(r.sharpness_mean_sigma);
    return s != null ? s : num(r.interval_width_90);
  }

  function metricValue(row, m) { return m.value ? m.value(row) : num(row[m.key]); }

  // Distinct sorted non-empty values of one field.
  function uniq(rows, key) {
    return Array.from(new Set(rows.map((r) => r[key]).filter((v) => v != null && v !== ""))).sort();
  }

  // f: {dataset, scenario, target}; "" or missing = no filter.
  function filterRows(rows, f) {
    const x = f || {};
    return rows.filter((r) => ["dataset", "scenario", "target", "test"].every((k) => !x[k] || r[k] === x[k]));
  }

  // Bar length 0..1 (1 = best of the shown rows) for value v among vals. null when v is missing.
  function barFraction(v, vals, metric) {
    if (v == null) return null;
    const ok = vals.filter((x) => x != null);
    if (!ok.length) return null;
    const score = (x) => (metric.dir === "target" ? -Math.abs(x - metric.ideal) : metric.dir === "high" ? x : -x);
    const s = ok.map(score);
    const lo = Math.min.apply(null, s), hi = Math.max.apply(null, s);
    if (hi === lo) return 1;
    return (score(v) - lo) / (hi - lo);
  }

  // ---------- runs (one test = one dataset + scenario) ----------

  function datasetLabel(ds) {
    const m = /^force(\d{4})$/i.exec(ds || "");
    return m ? "FORCE " + m[1] : String(ds || "");
  }

  function scenarioLabel(sc) {
    const g = /^gap_(\d+(?:\.\d+)?)$/.exec(sc || "");
    if (g) return g[1] + " m gap (target visible around it)";
    if (sc === "missing_curve") return "whole well hidden";
    return String(sc || "").replace(/_/g, " ");
  }

  function orderedProps(targets) {
    const known = Q.PROPERTIES.filter((p) => targets.indexOf(p) >= 0);
    return known.concat(targets.filter((t) => known.indexOf(t) < 0).sort());
  }

  function parseTime(iso) {
    if (typeof iso !== "string") return null;
    const d = new Date(iso.replace(/([+-]\d{2})(\d{2})$/, "$1:$2"));
    return Number.isNaN(d.getTime()) ? null : d;
  }

  // Group scoreboard rows into runs by (dataset, scenario). Rows carry a per-property test name, so the test only splits a
  // group when the same model and property were scored under several tests.
  function groupRuns(rows) {
    const groups = new Map();
    for (const r of rows) {
      const k = r.dataset + "\u0000" + r.scenario;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(r);
    }
    const runs = [];
    groups.forEach((list) => {
      const tests = new Map();
      for (const r of list) {
        const ck = r.contender + "\u0000" + r.target;
        if (!tests.has(ck)) tests.set(ck, new Set());
        tests.get(ck).add(r.test || "");
      }
      const split = Array.from(tests.values()).some((s) => s.size > 1);
      const parts = new Map();
      for (const r of list) {
        const t = split ? r.test || "" : "";
        if (!parts.has(t)) parts.set(t, []);
        parts.get(t).push(r);
      }
      parts.forEach((rs, test) => {
        const times = rs.map((r) => parseTime(r.run_at)).filter(Boolean).map((d) => d.getTime());
        runs.push({
          key: [rs[0].dataset, rs[0].scenario, test].join("|"),
          dataset: rs[0].dataset, scenario: rs[0].scenario, test, rows: rs,
          props: orderedProps(uniq(rs, "target")), models: uniq(rs, "contender"),
          runAt: times.length ? Math.max.apply(null, times) : 0,
        });
      });
    });
    return runs.sort((a, b) => b.runAt - a.runAt || a.key.localeCompare(b.key));
  }

  // "RHOB · 50 m gaps · FORCE 2020"; several properties -> "All properties".
  function runTitle(run) {
    const p = run.props.length === 1 ? run.props[0] : "All properties";
    return [p, scenarioLabel(run.scenario), datasetLabel(run.dataset)].concat(run.test ? [run.test] : []).join(" · ");
  }

  function formatRunTime(iso) {
    const d = parseTime(iso);
    if (!d) return null;
    try {
      return new Intl.DateTimeFormat("en-GB", { timeZone: "Europe/Oslo", day: "numeric", month: "short", year: "numeric",
        hour: "2-digit", minute: "2-digit", hour12: false }).format(d) + " Europe/Oslo";
    } catch (e) { return null; }
  }

  function latestRow(rows) {
    return rows.slice().sort((a, b) => ((parseTime(b.run_at) || 0) - (parseTime(a.run_at) || 0)))[0] || null;
  }

  function hardwareLabel(rows) {
    const r = latestRow(rows.filter((x) => x.hardware || x.machine));
    if (!r) return null;
    const hw = r.hardware && typeof r.hardware === "object" ? r.hardware : {};
    let chip = hw.system || "";
    if (!chip && typeof r.machine === "string") chip = r.machine.split("/").pop().trim();
    if (chip === "Darwin") chip = "macOS";
    const detail = num(hw.cpu_count) != null ? hw.cpu_count + " threads" : "";
    return chip || detail ? { chip, detail } : null;
  }

  // Header facts of a run; every field is omitted (null) when the rows do not carry it.
  function runHeader(run, prop) {
    const props = prop && prop !== "ALL" ? [prop] : run.props;
    const rows = props.length === 1 ? run.rows.filter((r) => r.target === props[0]) : run.rows;
    const maxOf = (rs, keys) => {
      const v = rs.map((r) => keys.map((k) => num(r[k])).find((x) => x != null)).filter((x) => x != null);
      return v.length ? Math.max.apply(null, v) : null;
    };
    const wellsBy = {};
    for (const p of props) wellsBy[p] = maxOf(rows.filter((r) => r.target === p), ["n_test_wells"]);
    const known = props.filter((p) => wellsBy[p] != null);
    let wells = null;
    if (known.length) {
      const top = Math.max.apply(null, known.map((p) => wellsBy[p]));
      const fewer = known.filter((p) => wellsBy[p] < top).map((p) => `${wellsBy[p]} with ${p}`);
      wells = `${top} test wells` + (fewer.length ? ` (${fewer.join(", ")})` : "");
    }
    let samples = null;
    if (props.length === 1) {
      const s = maxOf(rows, ["n_samples", "n"]);
      if (s != null) samples = `${Math.round(s).toLocaleString("en-GB")} samples`;
    }
    const last = latestRow(run.rows);
    const full = last && typeof last.git_commit_full === "string" ? last.git_commit_full : "";
    const commit = last && typeof last.git_commit === "string" && last.git_commit ? last.git_commit : full.slice(0, 7);
    const header = { wells, samples, time: last ? formatRunTime(last.run_at) : null, commit: commit || null, hardware: hardwareLabel(run.rows) };
    if (run.rows.some((r) => r.stale === true)) header.stale = true;
    return header;
  }

  // Per-property wells and samples (by model-max) for the By property column headers.
  function propertyFacts(run, prop) {
    const rs = run.rows.filter((r) => r.target === prop);
    const pick = (keys) => {
      const v = rs.map((r) => keys.map((k) => num(r[k])).find((x) => x != null)).filter((x) => x != null);
      return v.length ? Math.max.apply(null, v) : null;
    };
    return { wells: pick(["n_test_wells"]), samples: pick(["n_samples", "n"]) };
  }

  // Scored items of one property of a run (one per model) or of the All aggregate.
  function propertyItems(run, prop) {
    if (prop === "ALL") {
      const total = run.props ? run.props.length : new Set(run.rows.map((r) => r.target)).size;
      const items = Q.aggregateAll(run.rows).filter((i) => i.props.length && i.acc != null);
      const full = items.filter((i) => i.props.length >= total);
      const partial = items.filter((i) => i.props.length < total);
      // Classes and ranks come from the full-property cohort only. Models scored on fewer properties stay visible,
      // are classified against that cohort's best, and take no rank.
      const scoredFull = Q.scoreItems(full);
      const bestAcc = Math.min.apply(null, full.map((i) => i.acc).filter((v) => v != null));
      const calWidths = scoredFull.filter((i) => (i.calClass === "good" || i.calClass === "ok") && i.width != null).map((i) => i.width);
      const bestWidth = calWidths.length ? Math.min.apply(null, calWidths) : null;
      const scoredPartial = partial.map((p) => {
        const calClass = Q.classifyCalibration(p.dev);
        return Object.assign({}, p, {
          accClass: Q.classifyAccuracy(p.acc, bestAcc), calClass, speedClass: Q.speedZone(p.speed),
          sharpClass: Q.classifySharpness(p.width, bestWidth, calClass), partial: { n: p.props.length, N: total },
        });
      });
      const byId = new Map(scoredFull.concat(scoredPartial).map((i) => [i.id, i]));
      return items.map((i) => byId.get(i.id));
    }
    const items = run.rows.filter((r) => r.target === prop).map((r) => ({
      id: r.contender, acc: num(r.MAE), width: num(r.interval_width_90), cover: num(r.cover_90),
      dev: num(r.cover_90) == null ? null : Math.abs(r.cover_90 - Q.TARGET_COVERAGE), speed: num(r.predict_s_per_well),
    }));
    return Q.scoreItems(items);
  }

  // Distinct colours per model (about 14); the baseline keeps the first colour.
  const MODEL_PALETTE = ["#2F6FAE", "#C9663A", "#7B5EA7", "#2E8B8B", "#B0892B", "#A34A6B", "#5C7F3A", "#6B6F76",
    "#D1495B", "#00798C", "#8F5B2C", "#3E4C9A", "#7A9E2E", "#C77DAA", "#1F8A5B", "#9C6B00"];
  function modelColor(name, all) {
    if (name === Q.BASELINE_MODEL) return MODEL_PALETTE[0];
    const sorted = all.filter((n) => n !== Q.BASELINE_MODEL).sort();
    return MODEL_PALETTE[1 + (sorted.indexOf(name) % (MODEL_PALETTE.length - 1))];
  }

  // ---------- prediction plot data ----------

  const Z90 = 1.6449;

  // {lo, hi} arrays (null where unknown) for the 90% band: p05/p95 if present, else p50 +/- 1.645 std.
  function bandArrays(w) {
    if (w.p05 && w.p95) {
      const lo = [], hi = [];
      for (let i = 0; i < w.p05.length; i++) {
        const valid = num(w.p05[i]) != null && num(w.p95[i]) != null;
        lo.push(valid ? w.p05[i] : null);
        hi.push(valid ? w.p95[i] : null);
      }
      return { lo, hi };
    }
    if (w.std && w.p50) {
      const f = (s, sign) => w.p50.map((p, i) => (num(p) != null && num(s[i]) != null ? p + sign * Z90 * s[i] : null));
      return { lo: f(w.std, -1), hi: f(w.std, 1) };
    }
    return null;
  }

  // Gap intervals [[top, bottom], ...] from the file, else rebuilt from nulls in `true` where p50 exists.
  function gapIntervals(w) {
    if (Array.isArray(w.gaps)) return w.gaps;
    const out = [];
    let start = null;
    for (let i = 0; i < w.depth.length; i++) {
      const miss = num(w.true && w.true[i]) == null && num(w.p50 && w.p50[i]) != null;
      if (miss && start == null) start = i;
      if (start != null && (!miss || i === w.depth.length - 1)) {
        out.push([w.depth[start], w.depth[miss ? i : i - 1]]);
        start = null;
      }
    }
    return out;
  }


  function contextDefs() {
    const ids = ["GR", "CALI", "RDEP", "RHOB", "NPHI", "DTC", "DTS", "PEF"];
    const style = (typeof WmbPlotStyle !== "undefined") ? WmbPlotStyle
      : (typeof require === "function" ? (function () { try { return require("./plot_style.js"); } catch (e) { return null; } })() : null);
    if (style) {
      return ids.map((id) => {
        const d = style.curveDef(id);
        return { id: id, title: id, unit: d.unit, color: d.color, xmin: d.xmin, xmax: d.xmax, log: !!d.log };
      });
    }
    return [
      { id: "GR", title: "GR", unit: "gAPI", color: "#1b7a3a", xmin: 0, xmax: 150 },
      { id: "CALI", title: "CALI", unit: "in", color: "#111", xmin: 6, xmax: 16 },
      { id: "RDEP", title: "RDEP", unit: "ohm.m", color: "#c0392b", xmin: 0.2, xmax: 2000, log: true },
      { id: "RHOB", title: "RHOB", unit: "g/cc", color: "#c0392b", xmin: 1.95, xmax: 2.95 },
      { id: "NPHI", title: "NPHI", unit: "v/v", color: "#1d4ed8", xmin: 0.45, xmax: -0.15 },
      { id: "DTC", title: "DTC", unit: "us/ft", color: "#111", xmin: 140, xmax: 40 },
      { id: "DTS", title: "DTS", unit: "us/ft", color: "#7c3aed", xmin: 240, xmax: 40 },
      { id: "PEF", title: "PEF", unit: "b/e", color: "#d97706", xmin: 0, xmax: 10 },
    ];
  }
  const CONTEXT_DEFS = contextDefs();
  const CTX_TRACK_W = 110;
  const PRED_TRACK_W = 150;

  // Up to `limit` (default 3) context tracks present in the well, excluding the predicted target.
  // `curves` is a packed {name: array} map or a list of curve names.
  function contextTracks(target, curves, limit) {
    const n = limit == null ? 3 : limit;
    let have;
    if (Array.isArray(curves)) have = new Set(curves);
    else {
      have = new Set();
      Object.keys(curves || {}).forEach((k) => {
        const a = curves[k];
        if (Array.isArray(a) && a.some((v) => typeof v === "number" && Number.isFinite(v))) have.add(k);
      });
    }
    return CONTEXT_DEFS.filter((d) => d.id !== target && have.has(d.id)).slice(0, n);
  }

  // ---------- view renderers (pure: data in, HTML string out) ----------

  const INFO_ALL = "\u201CAll\u201D compares models across properties without mixing units. " +
    "Each model\u2019s error is divided by the Linear template\u2019s error on that property, then averaged. " +
    "1.0 matches the baseline; lower is better. Sharpness is combined the same way. " +
    "Calibration is the average distance from 90% coverage. " +
    "Models scored on fewer properties are averaged over those only, shown as n/N, and are not ranked against models that have every property.";
  const INFO_READ = "Each cell scores one model on one property against the other models in its column. " +
    "Accuracy: average error, lower is better. Calibration: how close the 90% bands come to containing the truth 90% of the time (marker on the 60\u2013100% scale, dashed tick = 90%). " +
    "Sharpness: width of the 90% band, narrower is better, but it only counts when calibration is not poor. " +
    "Speed: time per well. Colours: green good, amber ok, red poor. Small grey numbers are the raw values.";
  const CAL_NOTE_ALL = "Full bar = on target";
  const SHARP_MUTED = "doesn\u2019t count \u2014 poorly calibrated";

  function chip(cls, text) { return `<span class="bm-chip ${cls || "muted"}">${esc(text)}</span>`; }
  function dot(color) { return `<i class="bm-dot" style="background:${color}"></i>`; }

  function withUnit(v, prop, all, width) {
    if (v == null) return "\u2013";
    const d = all ? 2 : prop === "RHOB" ? (width ? 3 : 4) : 1;
    return all ? `${v.toFixed(d)}\u00D7 baseline` : `${v.toFixed(d)} ${Q.unitOf(prop)}`.trim();
  }
  function modelName(id) { return id === Q.BASELINE_MODEL ? "Linear template" : id; }
  function itemName(i) { return modelName(i.id) + (i.partial ? ` (${i.partial.n}/${i.partial.N})` : ""); }
  function modelFamily(id) {
    if (id === "red_floor_train_mean") return "floor";
    if (id === Q.BASELINE_MODEL) return "baseline";
    if (/^black_/.test(id)) return "physics";
    if (/^blue_/.test(id)) return "GBT";
    if (/^(red|white|purple)_/.test(id)) return "foundation";
    return "";
  }
  function familyBadge(id) {
    const f = modelFamily(id);
    return f ? ` <small class="bm-fam na">${esc(f)}</small>` : "";
  }
  function ranked(rank) {
    return `<span class="bm-rk${rank === 1 ? " first" : ""}"${rank ? ` aria-label="Rank ${rank}"` : ""}>${rank || "\u2013"}</span>`;
  }
  function track(frac, cls) {
    return `<span class="bm-track"><b class="${cls || "muted"}" style="width:${(frac * 100).toFixed(1)}%"></b></span>`;
  }
  function magnitude(value, max) { return max > 0 ? Math.max(0.05, value / max) : 0; }
  function metricRow(i, ctx, rank, cls, value, visual, qualifier, note) {
    return `<div class="bm-r"><div class="bm-r-head">${ranked(rank)}<span class="bm-nm">${dot(ctx.colorOf(i.id))}${esc(itemName(i))}${familyBadge(i.id)}</span>` +
      `<small>${esc(value)}</small></div><div class="bm-r-track${qualifier != null ? " bm-cal-row" : ""}">` +
      `${visual}${chip(cls, Q.WORDS[cls || "muted"])}${qualifier != null ? `<small>${esc(qualifier)}</small>` : ""}</div>` +
      `${note ? `<div class="bm-cap-note">${esc(note)}</div>` : ""}</div>`;
  }

  function calTrack(cover, cls) {
    const pos = (Q.calScaleFrac(cover) * 100).toFixed(1);
    return `<span class="bm-cal" title="90% target">` +
      `<i class="bm-cal-tick" style="left:${(Q.CAL_TARGET_FRAC * 100).toFixed(1)}%"></i>` +
      `<i class="bm-cal-mk ${cls}" style="left:${pos}%"></i></span>`;
  }

  function cardAccuracy(items, ctx) {
    const rows = items.filter((i) => i.acc != null).sort((a, b) => a.acc - b.acc);
    const max = rows.length ? rows[rows.length - 1].acc : 0;
    const body = rows.map((i) => metricRow(i, ctx, i.accRank, i.accClass, withUnit(i.acc, ctx.prop, ctx.all),
      track(magnitude(i.acc, max), i.accClass))).join("");
    return card("Accuracy: average error", "\u2193 Lower is better", ctx.all ? "1.0\u00D7 = Linear template" : "shortest bar wins",
      body || '<p class="na">No accuracy scores.</p>');
  }

  function cardCalibration(items, ctx) {
    const rows = items.filter((i) => i.dev != null).sort((a, b) => a.dev - b.dev);
    let body = rows.map((i) => ctx.all
      ? metricRow(i, ctx, i.calRank, i.calClass, `${Math.round(i.dev * 100)} pt`, track(Q.calClosenessFrac(i.dev), i.calClass))
      : metricRow(i, ctx, i.calRank, i.calClass, `${(i.cover * 100).toFixed(0)}%`, calTrack(i.cover, i.calClass),
        Q.calibrationWord(i.cover, i.calClass))).join("");
    if (body) {
      body += ctx.all ? `<div class="bm-cap-note">${esc(CAL_NOTE_ALL)}</div>`
        : '<div class="bm-capscale"><span>60%</span><b>90% target</b><span>100%</span></div>';
    }
    return card("Calibration: do the 90% bands contain the truth 90% of the time?",
      ctx.all ? "\u2193 Lower is better" : "\u25CE Closest to 90% is better", ctx.all ? "mean distance from target" : "",
      body || '<p class="na">No calibration scores.</p>');
  }

  function cardSharpness(items, ctx) {
    const rows = items.filter((i) => i.width != null).sort((a, b) =>
      Number(!a.sharpClass || a.sharpClass === "muted") - Number(!b.sharpClass || b.sharpClass === "muted") || a.width - b.width);
    const max = Math.max(0, ...rows.map((i) => i.width));
    let rank = 0;
    const body = rows.map((i) => metricRow(i, ctx, i.sharpClass && i.sharpClass !== "muted" && !i.partial ? ++rank : null,
      i.sharpClass, withUnit(i.width, ctx.prop, ctx.all, true), track(magnitude(i.width, max), i.sharpClass),
      null, i.sharpClass === "muted" ? SHARP_MUTED : !i.sharpClass ? "No calibration score" : "")).join("");
    return card("Sharpness: width of the 90% uncertainty band", "\u2193 Narrower is better", "only if calibrated",
      body || '<p class="na">No interval widths.</p>');
  }

  // Semicircular log dial: 0.001-100 s per well, one needle per model.
  function dialSvg(items, ctx) {
    const cx = 140, cy = 113, R = 86, SW = 13;
    const pt = (f, r) => { const a = Math.PI * (1 - f); return [cx + r * Math.cos(a), cy - r * Math.sin(a)]; };
    const arc = (f0, f1, r) => {
      const [x0, y0] = pt(f0, r), [x1, y1] = pt(f1, r);
      return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
    };
    const edges = [0, Q.speedFrac(Q.THRESHOLDS.speed.realTime), Q.speedFrac(Q.THRESHOLDS.speed.batch), 1];
    const zones = Q.SPEED_ZONES.map((z, k) => `<path class="${z.id}" d="${arc(edges[k], edges[k + 1], R)}" fill="none" stroke-width="${SW}"/>`).join("");
    const ticks = Q.SPEED_AXIS.ticks.map((t, k) => {
      const f = Q.speedFrac(t), [x0, y0] = pt(f, R + SW / 2 + 1), [x1, y1] = pt(f, R + SW / 2 + 5), [tx, ty] = pt(f, R + SW / 2 + 15);
      return `<line x1="${x0.toFixed(1)}" y1="${y0.toFixed(1)}" x2="${x1.toFixed(1)}" y2="${y1.toFixed(1)}" stroke="#5c5c5c" stroke-width="1"/>` +
        `<text x="${tx.toFixed(1)}" y="${(ty + 3).toFixed(1)}" text-anchor="middle" font-size="10" fill="#5C6570">${Q.SPEED_AXIS.tickLabels[k]}</text>`;
    }).join("");
    const withSpeed = items.filter((i) => i.speed != null).sort((a, b) => a.speed - b.speed);
    const needles = withSpeed.slice().reverse().map((i, k) => {
      const [x, y] = pt(Q.speedFrac(i.speed), R - (k / Math.max(1, withSpeed.length - 1)) * 30);
      const c = ctx.colorOf(i.id);
      return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${c}" stroke-width="2.4" stroke-linecap="round"/>` +
        `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${c}" stroke="#fff"><title>${esc(modelName(i.id))}: ${esc(Q.formatSeconds(i.speed))} per well</title></circle>`;
    }).join("");
    const legend = withSpeed.map((i) => `<div class="bm-dl">${ranked(i.speedRank)}<span class="bm-nm">${dot(ctx.colorOf(i.id))}${esc(itemName(i))}${familyBadge(i.id)}</span>` +
      `${chip(i.speedClass, Q.speedZoneName(i.speedClass))}<small>${esc(Q.formatSeconds(i.speed))}</small></div>`).join("");
    const minorTicks = [-3, -2, -1, 0, 1].flatMap((exponent) => [2, 3, 4, 5, 6, 7, 8, 9].map((factor) => {
      const f = Q.speedFrac(factor * 10 ** exponent);
      const [x0, y0] = pt(f, R + SW / 2 + 1), [x1, y1] = pt(f, R + SW / 2 + 3);
      return `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" stroke="#B3B0A8"/>`;
    })).join("");
    const [fast, batch, slow] = Q.SPEED_ZONES;
    return `<svg class="bm-dial" viewBox="0 0 280 133" role="img" aria-label="Time per well, log scale from 0.001 to 100 seconds">${zones}${minorTicks}${ticks}${needles}` +
      `<circle cx="${cx}" cy="${cy}" r="6" fill="#1A1E23"/><text x="${cx}" y="128" text-anchor="middle" font-size="10" fill="#5C6570">seconds per well \u00B7 log scale</text></svg>` +
      `<div class="bm-zones">${chip(fast.id, `${fast.name} \u2264 ${fast.to} s`)}${chip(batch.id, `${batch.name} ${batch.from}\u2013${batch.to} s`)}${chip(slow.id, `${slow.name} > ${slow.from} s`)}</div>` +
      `<div class="bm-dials">${legend}</div>${ctx.hwKnown ? '<div class="bm-speed-note">Measured on this run\u2019s hardware.</div>' : ""}`;
  }

  function cardSpeed(items, ctx) {
    const any = items.some((i) => i.speed != null);
    return card("Speed: time per well", "\u2193 Lower is better", ctx.all ? "mean across properties" : "fast = real-time",
      any ? dialSvg(items, ctx) : '<p class="na">No timings.</p>');
  }

  function card(title, direction, detail, body) {
    return `<section class="bm-card"><div class="bm-card-head"><h3>${esc(title)}</h3><div class="bm-dir"><b>${esc(direction)}</b>${detail ? `<span>${esc(detail)}</span>` : ""}</div></div>${body}</section>`;
  }

  function niceTicks(max, n) {
    const raw = max / n, power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].find((v) => v * power >= raw) * power;
    return Array.from({ length: Math.floor(max / step) + 1 }, (_, k) => k * step);
  }
  function tickText(v) { return String(Number(v.toPrecision(4))); }

  // Relative scatter is centred on the Linear baseline at 1.0 (default domain 0..2).
  // If a model is worse than 2x, the axis grows past 2 but still starts at 0.
  function scatterExtent(pts, relative) {
    if (!relative) {
      const xmax = Math.max.apply(null, pts.map((p) => p.width).concat([0])) * 1.18 || 1;
      const ymax = Math.max.apply(null, pts.map((p) => p.acc).concat([0])) * 1.18 || 1;
      return { xmin: 0, xmax: xmax, ymin: 0, ymax: ymax };
    }
    const maxX = Math.max.apply(null, pts.map((p) => p.width).concat([1]));
    const maxY = Math.max.apply(null, pts.map((p) => p.acc).concat([1]));
    const hi = Math.max(2, maxX, maxY);
    const pad = hi > 2 ? hi * 1.05 : 2;
    return { xmin: 0, xmax: pad, ymin: 0, ymax: pad };
  }

  function scatterQuadrant(acc, width) {
    if (acc < 1 && width < 1) return "better";
    if (acc > 1 && width > 1) return "worse";
    return "mixed";
  }

  // Accuracy (y) vs sharpness (x); both lower is better, so the better corner is bottom-left.
  function scatterSvg(items, ctx) {
    const pts = items.filter((i) => i.acc != null && i.width != null);
    if (!pts.length) return '<p class="na">No models with both accuracy and interval width.</p>';
    const W = 850, H = 260, m = { l: 62, r: 22, t: 18, b: 39 };
    const ext = scatterExtent(pts, !!ctx.all);
    const xmax = ext.xmax, ymax = ext.ymax;
    const sx = (v) => m.l + (v / xmax) * (W - m.l - m.r);
    const sy = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    const unit = ctx.all ? "\u00D7 baseline" : Q.unitOf(ctx.prop);
    const measure = (label) => unit ? `${label} (${unit})` : label;
    const grid = niceTicks(xmax, 5).map((v) => `<line x1="${sx(v).toFixed(1)}" y1="${m.t}" x2="${sx(v).toFixed(1)}" y2="${H - m.b}" stroke="#eceeed"/>` +
      `<text x="${sx(v).toFixed(1)}" y="${H - m.b + 15}" text-anchor="middle" font-size="10" fill="#5c5c5c">${tickText(v)}</text>`).join("") +
      niceTicks(ymax, 4).map((v) => `<line x1="${m.l}" y1="${sy(v).toFixed(1)}" x2="${W - m.r}" y2="${sy(v).toFixed(1)}" stroke="#eceeed"/>` +
        `<text x="${m.l - 8}" y="${(sy(v) + 3).toFixed(1)}" text-anchor="end" font-size="10" fill="#5c5c5c">${tickText(v)}</text>`).join("");
    const g = Q.COLORS.good, r = Q.COLORS.poor;
    const x0 = sx(0), x1 = sx(1), x2 = sx(xmax), y0 = sy(0), y1 = sy(1), y2 = sy(ymax);
    const quadrants = ctx.all ? (
      `<rect x="${x0.toFixed(1)}" y="${y1.toFixed(1)}" width="${(x1 - x0).toFixed(1)}" height="${(y0 - y1).toFixed(1)}" fill="${g.fg}" opacity=".16"/>` +
      `<rect x="${x1.toFixed(1)}" y="${y2.toFixed(1)}" width="${(x2 - x1).toFixed(1)}" height="${(y1 - y2).toFixed(1)}" fill="${r.fg}" opacity=".12"/>` +
      `<rect x="${x0.toFixed(1)}" y="${y2.toFixed(1)}" width="${(x1 - x0).toFixed(1)}" height="${(y1 - y2).toFixed(1)}" fill="#c4a35a" opacity=".10"/>` +
      `<rect x="${x1.toFixed(1)}" y="${y1.toFixed(1)}" width="${(x2 - x1).toFixed(1)}" height="${(y0 - y1).toFixed(1)}" fill="#c4a35a" opacity=".10"/>` +
      `<text x="${((x0 + x1) / 2).toFixed(1)}" y="${((y0 + y1) / 2 - 4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="${g.fg}">better: more accurate and sharper</text>` +
      `<text x="${((x1 + x2) / 2).toFixed(1)}" y="${((y1 + y2) / 2 - 4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="${r.fg}">worse: less accurate and wider bands</text>` +
      `<text x="${((x0 + x1) / 2).toFixed(1)}" y="${((y1 + y2) / 2 - 4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="#7a6230">sharper bands, but less accurate</text>` +
      `<text x="${((x1 + x2) / 2).toFixed(1)}" y="${((y0 + y1) / 2 + 4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="#7a6230">more accurate, but wider bands</text>`
    ) : "";
    const base = ctx.all
      ? `<line x1="${x1.toFixed(1)}" y1="${m.t}" x2="${x1.toFixed(1)}" y2="${H - m.b}" stroke="#8a8f94" stroke-width="1"/>` +
        `<line x1="${m.l}" y1="${y1.toFixed(1)}" x2="${W - m.r}" y2="${y1.toFixed(1)}" stroke="#8a8f94" stroke-width="1"/>`
      : "";
    const front = Q.paretoFront(pts.filter((p) => !p.partial).map((p) => ({ x: p.width, y: p.acc })));
    const frontLine = front.length > 1
      ? `<polyline points="${front.map((p) => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ")}" fill="none" stroke="#4a5560" stroke-width="1.5" stroke-dasharray="6 4"/>` : "";
    const marks = pts.map((p) => {
      const x = sx(p.width), y = sy(p.acc), c = ctx.colorOf(p.id);
      const name = itemName(p), label = name.length > 23 ? name.slice(0, 22) + "\u2026" : name;
      const isLinear = ctx.all && (p.id === Q.BASELINE_MODEL || p.id === "template_linear");
      if (isLinear) {
        const right = x + 12 + label.length * 6 < W - m.r;
        return `<g><title>${esc(name)}: Linear reference (1.0)</title>` +
          `<circle cx="${x}" cy="${y}" r="3.5" fill="#8B929A" stroke="#fff" stroke-width="1"/>` +
          `<text x="${x + (right ? 10 : -10)}" y="${y + 3}" text-anchor="${right ? "start" : "end"}" font-size="10" font-weight="600" fill="#6b6f76">${esc(label)}</text></g>`;
      }
      const calOk = p.calClass === "good" || p.calClass === "ok";
      const ring = calOk
        ? `<circle cx="${x}" cy="${y}" r="14" fill="${g.fg}" opacity=".18"/><circle cx="${x}" cy="${y}" r="10.5" fill="none" stroke="${c}" stroke-width="2.5"/>`
        : `<circle cx="${x}" cy="${y}" r="10.5" fill="none" stroke="${p.calClass ? r.fg : "#8B929A"}" stroke-width="2" stroke-dasharray="3 2.5"/>`;
      const right = x + 17 + label.length * 7 < W - m.r;
      return `<g><title>${esc(name)}: ${p.calClass ? `calibration ${p.calClass}` : "no calibration score"}</title>${ring}` +
        `<circle cx="${x}" cy="${y}" r="6.5" fill="${c}" stroke="#fff" stroke-width="1.5"/>` +
        `<text x="${x + (right ? 17 : -17)}" y="${y + 4}" text-anchor="${right ? "start" : "end"}" font-size="12" font-weight="700" fill="#1A1E23">${esc(label)}</text></g>`;
    }).join("");
    const bx = m.l + 8, by = H - m.b - 28;
    const better = ctx.all ? "" : (`<g><rect x="${bx}" y="${by}" width="63" height="18" rx="9" fill="${g.fg}"/>` +
      `<text x="${bx + 31.5}" y="${by + 13}" text-anchor="middle" font-size="11" font-weight="700" fill="#fff">\u2199 Better</text></g>`);
    const xt = `${measure("Uncertainty width of the 90% band")} \u2014 lower is better \u2190`;
    const yt = `${measure("MAE")} \u2014 lower is better \u2193`;
    const wash = ctx.all ? "" : (
      `<defs><radialGradient id="bm-better" cx="0" cy="1" r="0.85"><stop offset="0" stop-color="${g.fg}" stop-opacity=".25"/><stop offset="1" stop-color="${g.fg}" stop-opacity="0"/></radialGradient></defs>` +
      `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="url(#bm-better)"/>`
    );
    return `<svg class="bm-scatter" viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy versus sharpness">` +
      `<rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="#fafbfa" stroke="#d5d5d3"/>` +
      `${wash}${quadrants}${grid}${base}${frontLine}${better}${marks}` +
      `<text x="${(m.l + (W - m.l - m.r) / 2).toFixed(1)}" y="${H - 10}" text-anchor="middle" font-size="11" fill="#333">${esc(xt)}</text>` +
      `<text transform="translate(14 ${(m.t + (H - m.t - m.b) / 2).toFixed(1)}) rotate(-90)" text-anchor="middle" font-size="11" fill="#333">${esc(yt)}</text></svg>`;
  }

  function scatterLegend() {
    const g = Q.COLORS.good.fg, r = Q.COLORS.poor.fg;
    return '<div class="bm-leg">' +
      `<span><svg width="30" height="22"><circle cx="15" cy="11" r="10" fill="${g}" opacity=".18"/><circle cx="15" cy="11" r="7" fill="none" stroke="#2F6FAE" stroke-width="2"/><circle cx="15" cy="11" r="3.5" fill="#2F6FAE"/></svg>Solid ring + green halo = calibration good or ok</span>` +
      `<span><svg width="30" height="22"><circle cx="15" cy="11" r="7" fill="none" stroke="${r}" stroke-width="2" stroke-dasharray="3 2.5"/><circle cx="15" cy="11" r="3.5" fill="#6b6f76"/></svg>Dashed ring = calibration off (red) or missing (grey)</span>` +
      '<span><svg width="30" height="22"><line x1="2" y1="11" x2="28" y2="11" stroke="#4a5560" stroke-width="1.5" stroke-dasharray="6 4"/></svg>Raw accuracy\u2013width Pareto front (ignores calibration)</span></div>';
  }

  // Models scored in other runs of the same dataset (on this property, or any for "ALL") but not in this run.
  // A physics model that never scores a property in any run is therefore not listed for it.
  function notScoredModels(run, runs, prop) {
    const all = prop === "ALL" || !prop;
    const inRun = (t) => !run.props || run.props.includes(t);
    const here = new Set(run.rows.filter((r) => all || r.target === prop).map((r) => r.contender));
    const out = new Set();
    for (const o of runs) {
      if (o === run || o.dataset !== run.dataset) continue;
      for (const r of o.rows) if ((all ? inRun(r.target) : r.target === prop) && !here.has(r.contender)) out.add(r.contender);
    }
    return Array.from(out).sort();
  }

  function summaryHtml(items, ctx) {
    const title = ctx.all ? "All properties \u2014 accuracy vs sharpness (relative to Linear)" : `${ctx.prop} \u2014 accuracy vs sharpness`;
    const models = items.map((i) => `<div class="bm-nm">${dot(ctx.colorOf(i.id))}${esc(itemName(i))}${familyBadge(i.id)}</div>`).join("") +
      (ctx.missing && ctx.missing.length
        ? `<div class="na bm-missing">Not scored in this test: ${ctx.missing.map((m) => esc(modelName(m))).join(", ")}</div>` : "");
    const nScatter = items.filter((i) => i.acc != null && i.width != null).length;
    const caption = "";
    return `<div class="bm-cards">${cardAccuracy(items, ctx)}${cardCalibration(items, ctx)}${cardSharpness(items, ctx)}${cardSpeed(items, ctx)}</div>` +
      `<div class="bm-tradeoff"><section class="bm-card bm-scatter-card"><h3>${esc(title)}</h3><span class="na">bottom-left is best \u00B7 ring = calibration</span>` +
      `${scatterSvg(items, ctx)}${caption}</section><aside class="bm-card"><h4>Models</h4><div class="bm-models">${models}</div>` +
      `<h4>Key</h4>${scatterLegend()}</aside></div>`;
  }

  function byPropertyCell(it, prop, colMax) {
    if (!it) return '<td class="bm-cell na">–</td>';
    const line = (label, cls, visual, value, rank, note) =>
      `<div class="bm-ln"><b>${label}</b>${rank ? ranked(rank) : "<span></span>"}${visual}` +
      `${chip(cls, Q.WORDS[cls || "muted"])}<small>${esc(value)}${note ? `<i>${esc(note)}</i>` : ""}</small></div>`;
    const acc = line("Accuracy", it.accClass, it.acc == null ? "<span></span>" : track(magnitude(it.acc, colMax.acc), it.accClass),
      withUnit(it.acc, prop), it.accRank);
    const cal = line("Calibration", it.calClass, it.cover == null ? "<span></span>" : calTrack(it.cover, it.calClass),
      it.cover == null ? "\u2013" : `${(it.cover * 100).toFixed(0)}%`, null, Q.calibrationWord(it.cover, it.calClass));
    const sharp = line("Sharpness", it.sharpClass, it.width == null ? "<span></span>" : track(magnitude(it.width, colMax.width), it.sharpClass),
      it.sharpClass === "muted" ? "\u2013" : withUnit(it.width, prop, false, true), null,
      it.sharpClass === "muted" ? "doesn\u2019t count" : it.width != null && !it.sharpClass ? "no calibration" : "");
    const speed = `<div class="bm-ln bm-ln-speed"><b>Speed</b>${chip(it.speedClass, it.speedClass ? Q.speedZoneName(it.speedClass) : Q.WORDS.muted)}` +
      `<small>${it.speed == null ? "\u2013" : `${esc(Q.formatSeconds(it.speed))} / well`}</small></div>`;
    return `<td class="bm-cell">${acc}${cal}${sharp}${speed}</td>`;
  }

  function byPropertyHtml(run, colorOf) {
    const perProp = {};
    for (const p of run.props) perProp[p] = propertyItems(run, p);
    const models = run.models.slice().sort((a, b) => (a === Q.BASELINE_MODEL ? -1 : b === Q.BASELINE_MODEL ? 1 : a.localeCompare(b)));
    const head = run.props.map((p) => {
      const f = propertyFacts(run, p);
      const facts = [Q.unitOf(p), f.wells != null ? `${f.wells} wells` : "", f.samples != null ? `${Math.round(f.samples).toLocaleString("en-GB")} samples` : ""].filter(Boolean).join(" \u00B7 ");
      return `<th scope="col"><b>${esc(p)}</b>${Q.PROPERTY_NAMES[p] ? ` <span class="na">\u00B7 ${esc(Q.PROPERTY_NAMES[p])}</span>` : ""}` +
        `${facts ? `<div class="na">${esc(facts)}</div>` : ""}</th>`;
    }).join("");
    const body = models.map((m) => `<tr><th class="bm-model" scope="row">${dot(colorOf(m))}${esc(modelName(m))}${familyBadge(m)}</th>` + run.props.map((p) => {
      const col = perProp[p];
      const max = (k) => Math.max.apply(null, col.map((i) => i[k]).filter((v) => v != null).concat([0]));
      return byPropertyCell(col.find((i) => i.id === m), p, { acc: max("acc"), width: max("width") });
    }).join("") + "</tr>").join("");
    return `<div class="bm-grid-wrap"><table class="bm-grid"><thead><tr><th scope="col">Model</th>${head}</tr></thead>` +
      `<tbody>${body}</tbody></table></div>`;
  }

  // ---------- browser UI ----------

  function plotStyleApi() {
    if (typeof WmbPlotStyle !== "undefined") return WmbPlotStyle;
    if (typeof require === "function") { try { return require("./plot_style.js"); } catch (e) { return null; } }
    return null;
  }
  const PS = plotStyleApi();
  const STD_SCALE = PS
    ? Object.fromEntries(["RHOB", "DTC", "DTS"].map((k) => [k, PS.standardScale(k)]))
    : { RHOB: [1.95, 2.95], DTC: [140, 40], DTS: [240, 40] };

  const EMPTY_HELP = "No benchmark results found yet. Generate some with:\n\n" +
    "python -m benchmark.run --model template_linear --dataset force2020 --scenario all --target all\n" +
    "python -m benchmark.scoreboard\n\n" +
    "then reload this tab. Results are read from the results/ folder (or WMB_RESULTS).";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function options(values, sel, any) {
    const o = any ? [`<option value="">${esc(any)}</option>`] : [];
    return o.concat(values.map((v) => `<option value="${esc(v)}"${v === sel ? " selected" : ""}>${esc(v)}</option>`)).join("");
  }

  async function getJson(url) {
    const r = await fetch(url);
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.error || r.statusText);
    }
    return r.json();
  }

  const mounted = new WeakMap();

  function wheelWindow(PU, win, ext, pixels, height, anchor, zoom, minSpan) {
    return zoom ? PU.zoomAt(win, ext, anchor, Math.exp(Math.max(-6, Math.min(6, pixels * 0.003))), minSpan)
      : PU.panBy(win, ext, pixels / height * (win.max - win.min), minSpan);
  }

  function mount(pane, PU) {
    if (mounted.has(pane)) {
      const show = mounted.get(pane);
      if (show) show();
      return;
    }
    mounted.set(pane, null);
    pane.innerHTML = '<div class="bm"><p class="na">Loading results…</p></div>';
    getJson("/api/benchmark/runs").then((j) => { mounted.set(pane, build(pane, PU, j.rows)); })
      .catch((e) => { pane.innerHTML = `<div class="bm"><p>Could not load results: ${esc(e.message)}</p></div>`; });
  }

  function build(pane, PU, rows) {
    if (!rows.length) {
      pane.innerHTML = `<div class="bm"><h2>Benchmark</h2><pre>${esc(EMPTY_HELP)}</pre></div>`;
      return null;
    }
    const runs = groupRuns(rows);
    const st = { sel: runs[0], mode: "summary", prop: null, model: "", target: "", well: "", scale: "auto", fillSource: "formation", data: null, token: 0 };
    pane.innerHTML = `<div class="bm">
      <div class="bm-top">
        <div>
          <div class="bm-title"><h2 id="bm-title"></h2>
            <button type="button" class="bm-i" id="bm-i-all" aria-label="What does All mean?" aria-controls="bm-tipbox" aria-expanded="false" hidden>\u2139</button>
          </div>
          <div class="bm-meta" id="bm-meta"></div>
          <div class="bm-hw" id="bm-hw"></div>
        </div>
        <div class="bm-ctl">
          <div class="bm-picker" id="bm-picker"></div>
          <button type="button" class="bm-i" id="bm-i-read" aria-label="How to read" aria-controls="bm-tipbox" aria-expanded="false" hidden>\u2139</button>
          <div class="bm-seg" id="bm-mode"><button type="button" data-mode="summary">Summary</button><button type="button" data-mode="byproperty">Per property</button><button type="button" data-mode="welllogs">Well logs</button></div>
        </div>
      </div>
      <label class="bm-run"${runs.length > 1 ? "" : " hidden"}>Run <select id="bm-run"></select></label>
      <div class="bm-tipbox" id="bm-tipbox" hidden></div>
      <div id="bm-view"></div>
      <div id="bm-logs-view" hidden>
        <h2 class="bm-sec">Well logs <span class="na">\u2014 true curve, prediction and 90% band for one well of the selected run</span></h2>
        <div class="bm-bar">
          <label>Model <select id="bm-model"></select></label>
          <label>Property <select id="bm-target"></select></label>
          <label>Well <select id="bm-well"></select></label>
          <label>X scale <select id="bm-scale"><option value="auto">Auto</option><option value="std">Standard</option></select></label>
          <label title="Shade zone bands behind all tracks. The TOPS strip and top lines always show.">Formation fill <select id="bm-fill"><option value="formation">Formation</option><option value="group">Group</option><option value="off">Off</option></select></label>
          <button type="button" id="bm-in">Zoom in</button>
          <button type="button" id="bm-out">Zoom out</button>
          <button type="button" id="bm-reset">Full depth</button>
        </div>
        <small>Scroll or drag to pan; pinch on a trackpad or Ctrl/⌘ + scroll to zoom at the pointer.</small>
        <div id="bm-note"></div>
        <div id="bm-example" class="bm-example" role="status">EXAMPLE DATA — lagged synthetic prediction, not a real template run</div>
        <div id="bm-plot-wrap"><canvas id="bm-canvas"></canvas></div>
      </div>
    </div>`;
    const $ = (id) => pane.querySelector("#" + id);

    const hasBaseline = () => st.sel.models.indexOf(Q.BASELINE_MODEL) >= 0;
    const availableViews = () => st.sel.props.map((p) => "summary-" + p.toLowerCase())
      .concat(hasBaseline() ? ["summary-all"] : []).concat(["byproperty", "welllogs"]);

    function currentView() { return viewId(st.mode, st.prop); }

    function writeHash() {
      if (pane.hidden || typeof history === "undefined" || !history.replaceState) return;
      const hash = benchmarkHash(currentView());
      if (location.hash === hash) return;
      const u = new URL(location.href);
      u.hash = hash;
      history.replaceState(null, "", u);
    }

    // Applies a view id; an unavailable one falls back to the default (RHOB, else the first property).
    function setView(view, writeBack) {
      const v = availableViews().indexOf(view) >= 0 ? view : "summary-" + String(defaultProperty(st.sel.props)).toLowerCase();
      const s = viewState(v);
      st.mode = s.mode;
      if (s.prop) st.prop = s.prop;
      else if (!st.prop) st.prop = defaultProperty(st.sel.props);
      if (writeBack !== false) writeHash();
      if (st.mode === "summary" && st.prop !== "ALL" && st.target !== st.prop && st.sel.props.indexOf(st.prop) >= 0) {
        st.target = st.prop;
        renderSelectors();
      }
      renderControls();
      const header = renderHeader();
      renderView(header);
    }

    function renderRunSelect() {
      $("bm-run").innerHTML = runs.map((r) => `<option value="${esc(r.key)}"${r === st.sel ? " selected" : ""}>${esc(runTitle(r))}</option>`).join("");
    }

    function renderHeader() {
      const by = st.mode === "byproperty";
      const label = by ? "Per property" : st.mode === "welllogs" ? "Well logs" : st.prop === "ALL" ? "All properties" : st.prop;
      $("bm-title").innerHTML = [label, scenarioLabel(st.sel.scenario), datasetLabel(st.sel.dataset), st.sel.test]
        .filter(Boolean).map(esc).join(' <span class="bm-sep">\u00B7</span> ');
      const h = runHeader(st.sel, by ? null : st.prop);
      const bits = [h.wells && `<strong>${esc(h.wells)}</strong>`, h.samples && `<span>${esc(h.samples)}</span>`,
        h.time && `<span>${esc(h.time)}</span>`, h.commit && `<span>commit <code>${esc(h.commit)}</code></span>`,
        h.stale && `<span class="na" title="The model code changed after this run; rerun to refresh the scores.">stale: model changed since this run</span>`].filter(Boolean);
      $("bm-meta").innerHTML = bits.join('<span class="bm-sep" aria-hidden="true">\u00B7</span>');
      $("bm-hw").innerHTML = h.hardware ? `${h.hardware.chip ? `<b>${esc(h.hardware.chip)}</b>` : ""}${esc(h.hardware.detail)}` : "";
      return h;
    }

    function renderControls() {
      const props = st.sel.props;
      const all = hasBaseline();
      $("bm-picker").hidden = st.mode !== "summary";
      const logs = $("bm-logs-view");
      if (logs) logs.hidden = st.mode !== "welllogs";
      $("bm-picker").innerHTML = '<span class="bm-picker-label">Property</span><div class="bm-seg">' +
        props.map((p) => `<button type="button" data-prop="${esc(p)}" aria-pressed="${st.prop === p}" title="${esc(Q.PROPERTY_NAMES[p] || p)}">${esc(p)}</button>`).join("") +
        `<button type="button" data-prop="ALL" aria-pressed="${st.prop === "ALL"}"${all ? "" : " disabled"} title="${all ? "All properties, relative to the Linear template" : "Needs the template_linear model in this run"}">All</button></div>`;
      pane.querySelectorAll("#bm-mode button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === st.mode));
      $("bm-i-all").hidden = st.mode !== "summary" || st.prop !== "ALL";
      $("bm-i-read").hidden = st.mode !== "byproperty";
      renderTip(null);
    }

    function renderTip(button) {
      const box = $("bm-tipbox"), all = $("bm-i-all"), read = $("bm-i-read");
      const open = button && button.getAttribute("aria-expanded") !== "true";
      all.setAttribute("aria-expanded", !!open && button === all);
      read.setAttribute("aria-expanded", !!open && button === read);
      box.hidden = !open;
      box.innerHTML = open ? `<strong>${button === all ? "What does All mean?" : "How to read this grid"}</strong>` +
        `<div>${esc(button === all ? INFO_ALL : INFO_READ)}</div>` : "";
    }

    function renderView(header) {
      const host = $("bm-view");
      const colorOf = (n) => modelColor(n, st.sel.models);
      if (st.mode === "welllogs") {
        host.innerHTML = "";
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(draw);
        else draw();
        return;
      }
      if (st.mode === "byproperty") { host.innerHTML = byPropertyHtml(st.sel, colorOf); return; }
      const all = st.prop === "ALL";
      const items = propertyItems(st.sel, st.prop);
      if (!items.length) { host.innerHTML = '<p class="na">No scores for this property.</p>'; return; }
      host.innerHTML = summaryHtml(items, { prop: st.prop, all, colorOf, hwKnown: !!header.hardware,
        missing: notScoredModels(st.sel, runs, st.prop) });
    }

    $("bm-run").addEventListener("change", (e) => {
      st.sel = runs.find((r) => r.key === e.target.value) || runs[0];
      st.model = ""; st.target = "";
      renderRunSelect();
      setView(currentView());
      renderSelectors();
    });
    $("bm-picker").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      if (b.dataset.prop) setView("summary-" + b.dataset.prop.toLowerCase());
    });
    $("bm-mode").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.mode === "byproperty" || b.dataset.mode === "welllogs") setView(b.dataset.mode);
      else setView("summary-" + String(st.prop || defaultProperty(st.sel.props)).toLowerCase());
    });
    for (const id of ["bm-i-all", "bm-i-read"]) {
      const button = $(id);
      button.addEventListener("click", () => renderTip(button));
      button.addEventListener("keydown", (e) => { if (e.key === "Escape") renderTip(null); });
    }

    // Per-well plot selectors cascade: model -> property over the selected run's rows.
    const pool = () => st.sel.rows;
    function renderSelectors() {
      const p = pool();
      const models = uniq(p, "contender");
      if (!models.includes(st.model)) st.model = models.includes(Q.BASELINE_MODEL) ? Q.BASELINE_MODEL : models[0] || "";
      $("bm-model").innerHTML = options(models, st.model);
      const targets = uniq(p.filter((r) => r.contender === st.model), "target");
      if (!targets.includes(st.target)) st.target = targets.includes(st.prop) ? st.prop : targets[0] || "";
      $("bm-target").innerHTML = options(targets, st.target);
      st.run = p.find((r) => r.contender === st.model && r.target === st.target) || null;
      loadWells();
    }

    function query(well) {
      const r = st.run;
      const test = r.prediction_test ?? r.test;
      return `contender=${encodeURIComponent(r.contender)}&dataset=${encodeURIComponent(r.dataset)}&target=${encodeURIComponent(r.target)}` +
        `&scenario=${encodeURIComponent(r.scenario)}&well=${encodeURIComponent(well)}` +
        (test ? `&test=${encodeURIComponent(test)}` : "");
    }

    // well=* lists the well keys stored in the run's predictions file.
    async function loadWells() {
      const token = ++st.token;
      const select = $("bm-well"), note = $("bm-note");
      select.disabled = true;
      select.innerHTML = "";
      st.data = null;
      draw();
      if (!st.run) { note.textContent = "No matching benchmark run."; return; }
      note.textContent = "Loading wells…";
      let wells;
      try {
        wells = (await getJson(`/api/benchmark/prediction?${query("*")}`)).wells;
        if (!Array.isArray(wells)) throw new Error("Invalid well list");
      } catch (e) {
        if (token === st.token) note.textContent = "Could not load wells: " + e.message;
        return;
      }
      if (token !== st.token) return;
      if (!wells.includes(st.well)) st.well = wells[0] || "";
      select.innerHTML = options(wells, st.well);
      select.disabled = !st.well;
      if (!st.well) { note.textContent = "No wells in this run."; return; }
      loadWell();
    }

    async function loadWell() {
      const token = ++st.token;
      st.data = null; st.context = null;
      const ex0 = $("bm-example"); if (ex0 && ex0.dataset) ex0.dataset.show = "";
      $("bm-note").textContent = "Loading well…";
      draw();
      try {
        const j = await getJson(`/api/benchmark/prediction?${query(st.well)}`);
        if (token !== st.token) return;
        st.data = j.well;
        st.win = null;
        st.context = null;
        st.zones = null;
        st.ext = PU.recordedExtent(j.well.depth);
        if (st.ext && st.ext.min === st.ext.max) st.ext = {min: st.ext.min - 0.5, max: st.ext.max + 0.5};
        st.step = PU.medianStep(j.well.depth);
        $("bm-note").textContent = `${st.run.contender} · ${st.run.dataset} · ${st.run.target} · ${st.run.scenario} · ${st.well}`;
        const runTime = formatRunTime(st.run.run_at);
        if (runTime) $("bm-note").textContent += " · " + runTime;
        if (st.run.stale) $("bm-note").textContent += " · stale: model changed since this run";
        const example = !!(j.well && (j.well.example || j.example)) || !!j.example;
        if (st.data) st.data.example = example;
        const ex = $("bm-example");
        if (ex && ex.dataset) {
          ex.dataset.show = example ? "1" : "";
          if (example) {
            ex.textContent = (j.well && j.well.example_label) || j.example_label
              || "EXAMPLE DATA — lagged synthetic prediction, not a real template run";
          }
        }
        if (j.well.context) {
          st.context = {d: j.well.depth, c: j.well.context, formation: j.well.formation || j.well.FORMATION,
            group: j.well.group || j.well.GROUP};
        }
        if (!st.context || (!st.context.formation && !st.context.group)) {
          try {
            const packed = await getJson("/api/well?dataset=" + encodeURIComponent(st.run.dataset) +
              "&well=" + encodeURIComponent(j.well.well_id || st.well));
            if (token !== st.token) return;
            if (!st.context) st.context = packed;
            st.zones = packed; // Zone samples may be on a different depth grid from embedded logs.
          } catch (e) {
            if (token !== st.token) return;
          }
        }
        st.zones = st.zones || st.context;
        if (!contextTracks(st.run.target, st.context && st.context.c).length) {
          $("bm-note").textContent += " · No measured context tracks available for this well.";
        }
        st.contextStep = st.context && st.context.d !== j.well.depth && st.context.d
          ? PU.medianStep(st.context.d) : st.step;
      } catch (e) {
        if (token !== st.token) return;
        $("bm-note").textContent = "Could not load this well: " + e.message;
      }
      draw();
    }

    $("bm-model").addEventListener("change", (e) => { st.model = e.target.value; renderSelectors(); });
    $("bm-target").addEventListener("change", (e) => { st.target = e.target.value; renderSelectors(); });
    $("bm-well").addEventListener("change", (e) => { st.well = e.target.value; loadWell(); });
    $("bm-scale").addEventListener("change", (e) => { st.scale = e.target.value; draw(); });
    $("bm-fill").addEventListener("change", (e) => { st.fillSource = e.target.value; draw(); });

    function draw() {
      if (pane.hidden) return;
      const cv = $("bm-canvas");
      if (!cv) return;
      const wrap = cv.parentElement;
      const W = wrap.clientWidth, H = wrap.clientHeight;
      if (W < 20 || H < 20) return;
      const dpr = window.devicePixelRatio || 1;
      cv.width = Math.max(1, Math.floor(W * dpr));
      cv.height = Math.max(1, Math.floor(H * dpr));
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const w = st.data;
      if (!w || !w.depth || !w.depth.length) return;

      if (!st.ext) return;
      if (!st.win) st.win = PU.resetWindow(st.ext);
      const dmin = st.win.min, dmax = st.win.max;
      const padTop = 44, padBot = 8, depthW = 52, gap = 6;
      const plotTop = padTop, plotH = H - padTop - padBot;
      if (plotH <= 2) return;
      const yOf = (d) => plotTop + (d - dmin) / (dmax - dmin) * plotH;
      st.geom = { plotTop, plotH };
      const ctxTracks = contextTracks(st.run.target, (st.context && st.context.c) || {}, 3);
      const maxGap = st.step * 2;
      const [i0, i1] = PU.visibleRange(w.depth, dmin, dmax);
      const rows = Math.max(1, Math.floor(plotH));

      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#ddd";
      ctx.beginPath(); ctx.moveTo(depthW, plotTop); ctx.lineTo(depthW, plotTop + plotH); ctx.stroke();
      ctx.fillStyle = "#444";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText("m", depthW - 6, 12);
      const ticks = PS ? PS.depthTicks(dmin, dmax) : (function () {
        const step = PU.niceStep(dmax - dmin), out = [];
        for (let d = Math.ceil(dmin / step) * step; d <= dmax; d += step) out.push(d);
        return out;
      }());
      for (const d of ticks) {
        const y = yOf(d);
        ctx.strokeStyle = "#eee";
        ctx.beginPath(); ctx.moveTo(depthW, y); ctx.lineTo(W, y); ctx.stroke();
        ctx.fillStyle = "#444";
        ctx.fillText(PU.formatTick(d), depthW - 6, y);
      }

      function shadeGaps(x, tw, alpha) {
        ctx.fillStyle = "rgba(210,140,0," + alpha + ")";
        for (const g of gapIntervals(w)) {
          const y0 = yOf(g[0]), y1 = yOf(g[1]);
          const top = Math.min(y0, y1), bot = Math.max(y0, y1);
          ctx.fillRect(x, Math.max(plotTop, top), tw, Math.max(1, Math.min(plotTop + plotH, bot) - Math.max(plotTop, top)));
        }
      }

      function strokeSeries(depths, arr, color, xmin, xmax, x, tw, log) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        const [start, end] = depths === w.depth ? [i0, i1] : PU.visibleRange(depths, dmin, dmax);
        const segs = PU.buildSegments(depths, arr, { log: !!log, i0: start, i1: end,
          maxGap: depths === w.depth ? maxGap : 2 * st.contextStep });
        for (const seg of segs) {
          const idx = PU.decimateRun(seg, depths, arr, dmin, dmax, rows);
          ctx.beginPath();
          let started = false;
          for (const i of idx) {
            const xx = log ? PU.xScaleLog(arr[i], xmin, xmax, x, tw) : PU.xScale(arr[i], xmin, xmax, x, tw);
            if (!Number.isFinite(xx)) { started = false; continue; }
            const yy = yOf(depths[i]);
            if (started) ctx.lineTo(xx, yy); else ctx.moveTo(xx, yy);
            started = true;
          }
          ctx.stroke();
        }
      }

      function header(title, x, tw, extras) {
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.font = "10px sans-serif";
        ctx.fillStyle = "#333";
        ctx.fillText(title, x + tw / 2, 4);
        (extras || []).forEach((t, i) => {
          ctx.fillStyle = t.color || "#333";
          ctx.fillText(t.text, x + tw / 2, 16 + i * 11);
        });
        ctx.textAlign = "left";
        ctx.fillStyle = "#666";
        ctx.fillText(PU.formatTick(extras && extras.xmin != null ? extras.xmin : 0), x + 2, padTop - 12);
        ctx.textAlign = "right";
        ctx.fillText(PU.formatTick(extras && extras.xmax != null ? extras.xmax : 1), x + tw - 2, padTop - 12);
      }

      let x = depthW + gap;
      const zones = st.zones;
      const intervals = (source) => PS && zones
        ? PS.formationIntervals(zones.d, zones.formation || zones.FORMATION, zones.group || zones.GROUP, source) : [];
      const tops = intervals("auto");
      const STRIP = tops.length ? 18 : 0;
      if (STRIP) {
        ctx.fillStyle = "#222";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.font = "10px sans-serif";
        ctx.fillText("TOPS", x + STRIP / 2, 4);
        ctx.strokeStyle = "#bbb";
        ctx.strokeRect(x + 0.5, plotTop + 0.5, STRIP - 1, plotH - 1);
        ctx.save();
        ctx.beginPath(); ctx.rect(x, plotTop, STRIP, plotH); ctx.clip();
        for (const iv of tops) {
          const y1 = yOf(iv.top), y2 = yOf(iv.base);
          ctx.fillStyle = iv.color;
          ctx.fillRect(x + 1, Math.min(y1, y2), STRIP - 2, Math.max(1, Math.abs(y2 - y1)));
        }
        ctx.restore();
        x += STRIP + gap;
      }
      const tracksLeft = x;
      const fills = intervals(st.fillSource);
      if (fills.length) {
        PS.fillFormationBands(ctx, fills, tracksLeft, ctxTracks.length * (CTX_TRACK_W + gap) + PRED_TRACK_W,
          yOf, plotTop, plotH, PS.FORMATION_FILL_ALPHA);
      }
      const autoScale = (series, std, log, name) => {
        const r = PU.autoValueRange(series, { std: std, log: !!log });
        if (!PS || !r) return r;
        const inc = PS.ensureIncreasingScale(name, r.xmin, r.xmax);
        return Object.assign({}, r, { xmin: inc[0], xmax: inc[1] });
      };
      ctxTracks.forEach((t) => {
        const tw = CTX_TRACK_W;
        const arr = (st.context.c && st.context.c[t.id]) || [];
        const depths = st.context.d || w.depth;
        const std = [t.xmin, t.xmax];
        const sc = st.scale === "std"
          ? { xmin: t.xmin, xmax: t.xmax }
          : autoScale(arr, std, t.log, t.id);
        const clip = PU.clipRect(x, plotTop, tw, plotH);
        ctx.save();
        ctx.beginPath(); ctx.rect(clip.x, clip.y, clip.w, clip.h); ctx.clip();
        shadeGaps(x, tw, 0.08);
        strokeSeries(depths, arr, t.color, sc.xmin, sc.xmax, x, tw, t.log);
        ctx.restore();
        ctx.strokeStyle = "#bbb";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, plotTop + 0.5, tw - 1, plotH - 1);
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.font = "10px sans-serif";
        ctx.fillStyle = t.color;
        ctx.fillText(t.title + (t.unit ? " " + t.unit : ""), x + tw / 2, 4);
        ctx.fillStyle = "#666";
        ctx.textAlign = "left";
        ctx.fillText(PU.formatTick(sc.xmin), x + 2, 18);
        ctx.textAlign = "right";
        ctx.fillText(PU.formatTick(sc.xmax), x + tw - 2, 18);
        x += tw + gap;
      });

      const tw = PRED_TRACK_W;
      const std = STD_SCALE[st.run.target] || [0, 1];
      // Range from true + p50 only so a wide 90% band cannot smear the track.
      const sc = st.scale === "std" && STD_SCALE[st.run.target]
        ? { xmin: std[0], xmax: std[1] }
        : autoScale([w.true, w.p50], std, false, st.run.target);
      const band = bandArrays(w);
      const clip = PU.clipRect(x, plotTop, tw, plotH);
      ctx.save();
      ctx.beginPath(); ctx.rect(clip.x, clip.y, clip.w, clip.h); ctx.clip();
      shadeGaps(x, tw, 0.16);
      if (band) {
        ctx.fillStyle = "rgba(43,108,163,0.22)";
        const segs = PU.buildSegments(w.depth, band.lo, { maxGap: maxGap, i0: i0, i1: i1 });
        for (const seg of segs) {
          const idx = PU.decimateRun(seg, w.depth, band.lo, dmin, dmax, rows)
            .filter((i) => num(band.hi[i]) != null && num(band.lo[i]) != null);
          if (idx.length < 2) continue;
          ctx.beginPath();
          idx.forEach((i, k) => {
            const xx = PU.xScale(band.lo[i], sc.xmin, sc.xmax, x, tw);
            if (k) ctx.lineTo(xx, yOf(w.depth[i])); else ctx.moveTo(xx, yOf(w.depth[i]));
          });
          for (let k = idx.length - 1; k >= 0; k--) {
            const i = idx[k];
            ctx.lineTo(PU.xScale(band.hi[i], sc.xmin, sc.xmax, x, tw), yOf(w.depth[i]));
          }
          ctx.closePath();
          ctx.fill();
        }
      }
      strokeSeries(w.depth, w.true, "#1f1f1f", sc.xmin, sc.xmax, x, tw, false);
      strokeSeries(w.depth, w.p50, "#c0392b", sc.xmin, sc.xmax, x, tw, false);
      ctx.restore();
      ctx.strokeStyle = "#bbb";
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, plotTop + 0.5, tw - 1, plotH - 1);
      ctx.textBaseline = "top";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "#333";
      ctx.fillText(st.run.target + "  true / p50", x + tw / 2, 4);
      ctx.fillStyle = "#1f1f1f";
      ctx.fillText("true", x + tw / 2 - 28, 16);
      ctx.fillStyle = "#c0392b";
      ctx.fillText("p50", x + tw / 2 + 8, 16);
      ctx.fillStyle = "rgba(43,108,163,0.85)";
      ctx.fillText("90%", x + tw / 2 + 34, 16);
      ctx.fillStyle = "#666";
      ctx.textAlign = "left";
      ctx.fillText(PU.formatTick(sc.xmin), x + 2, 28);
      ctx.textAlign = "right";
      ctx.fillText(PU.formatTick(sc.xmax), x + tw - 2, 28);
      if (w.example) {
        ctx.save();
        ctx.fillStyle = "rgba(124, 45, 18, 0.92)";
        ctx.font = "bold 16px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("EXAMPLE DATA — not a real template run", depthW + gap, plotTop + 8);
        ctx.restore();
      }
      if (tops.length && PS) {
        ctx.save();
        ctx.fillStyle = PS.TOP_LINE;
        ctx.font = "10px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        for (const iv of tops) {
          const y = yOf(iv.top);
          if (y < plotTop || y > plotTop + plotH) continue;
          PS.strokeTopLine(ctx, tracksLeft, x + tw, y);
          ctx.fillText(iv.name, tracksLeft + 2, y - 1);
        }
        ctx.restore();
      }
    }

    function depthAtEvent(e) {
      const rect = $("bm-canvas").getBoundingClientRect();
      const g = st.geom, win = st.win;
      if (!g || !win) return 0;
      return win.min + (e.clientY - rect.top - g.plotTop) / g.plotH * (win.max - win.min);
    }
    const cvEl = $("bm-canvas");
    cvEl.addEventListener("wheel", (e) => {
      if (!st.data || !st.win || !st.geom) return;
      e.preventDefault();
      const pixels = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? st.geom.plotH : 1);
      st.win = wheelWindow(PU, st.win, st.ext, pixels, st.geom.plotH,
        Math.max(st.win.min, Math.min(st.win.max, depthAtEvent(e))), e.ctrlKey || e.metaKey, st.step);
      draw();
    }, { passive: false });
    cvEl.addEventListener("pointerdown", (e) => {
      if (!st.data || !st.win || !st.geom || st.drag || e.button !== 0) return;
      st.drag = { id: e.pointerId, y: e.clientY, win: st.win };
      cvEl.setPointerCapture(e.pointerId);
      cvEl.classList.add("dragging");
    });
    cvEl.addEventListener("pointermove", (e) => {
      if (!st.drag || st.drag.id !== e.pointerId || !st.geom || !st.data) return;
      const dd = -(e.clientY - st.drag.y) / st.geom.plotH * (st.drag.win.max - st.drag.win.min);
      st.win = PU.panBy(st.drag.win, st.ext, dd, st.step);
      draw();
    });
    function endDrag(e) {
      if (!st.drag || (e && st.drag.id !== e.pointerId)) return;
      const id = st.drag.id;
      st.drag = null;
      cvEl.classList.remove("dragging");
      if (cvEl.hasPointerCapture(id)) cvEl.releasePointerCapture(id);
    }
    cvEl.addEventListener("pointerup", endDrag);
    cvEl.addEventListener("pointercancel", endDrag);
    cvEl.addEventListener("dblclick", () => { st.win = null; draw(); });
    for (const [id, factor] of [["bm-in", 0.5], ["bm-out", 2]]) {
      $(id).addEventListener("click", () => {
        if (!st.data || !st.win) return;
        st.win = PU.zoomAt(st.win, st.ext, (st.win.min + st.win.max) / 2, factor, st.step);
        draw();
      });
    }
    $("bm-reset").addEventListener("click", () => { st.win = null; draw(); });
    window.addEventListener("resize", draw);

    // Called each time the tab is shown: follows the URL hash (or sets the default view), then redraws.
    function show() {
      const fromHash = parseBenchmarkView(location.hash);
      if (fromHash && availableViews().indexOf(fromHash) >= 0) setView(fromHash, false);
      else setView(currentViewOrDefault());
      draw();
    }
    function currentViewOrDefault() {
      return st.prop ? currentView() : "summary-" + String(defaultProperty(st.sel.props)).toLowerCase();
    }

    renderRunSelect();
    st.prop = defaultProperty(st.sel.props);
    renderSelectors();
    show();
    return show;
  }

  return {
    parseTabHash, tabHash, parseBenchmarkView, benchmarkHash, viewState, viewId, defaultProperty, BENCHMARK_VIEWS,
    createTabs, METRICS, sharpnessOf, filterRows, barFraction, bandArrays, gapIntervals, contextTracks, wheelWindow, uniq, mount,
    groupRuns, runTitle, datasetLabel, scenarioLabel, runHeader, formatRunTime, hardwareLabel, propertyItems, notScoredModels, modelFamily,
    summaryHtml, byPropertyHtml, scatterSvg, scatterExtent, scatterQuadrant,
  };
});
