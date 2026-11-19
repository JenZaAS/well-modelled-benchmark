/* Tab registry + Benchmark tab for the live viewer. Pure helpers are DOM-free (node-tested); mount() needs a browser. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbBenchmark = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------- tabs ----------

  // "#benchmark" -> "benchmark" when it is a known tab id, else fallback.
  function parseTabHash(hash, ids, fallback) {
    const h = String(hash || "").replace(/^#/, "");
    let id = h;
    try { id = decodeURIComponent(h); } catch (e) { /* keep raw */ }
    return ids.indexOf(id) >= 0 ? id : fallback;
  }

  function tabHash(id) { return "#" + encodeURIComponent(id); }

  // Registry of {id, title, pane, show(pane)}; renders buttons into opts.bar and toggles pane[hidden].
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
        b.addEventListener("click", () => api.select(t.id));
        opts.bar.appendChild(b);
      }
    }
    const api = {
      register(tab) { list.push(tab); render(); return api; },
      ids() { return list.map((t) => t.id); },
      current() { return current; },
      select(id) {
        const t = list.find((x) => x.id === id) || list.find((x) => x.id === opts.defaultId) || list[0];
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


  const CONTEXT_DEFS = [
    { id: "GR", title: "GR", unit: "gAPI", color: "#1b7a3a", xmin: 0, xmax: 150 },
    { id: "CALI", title: "CALI", unit: "in", color: "#7a4a12", xmin: 6, xmax: 16 },
    { id: "RDEP", title: "RDEP", unit: "ohm.m", color: "#111", xmin: 0.2, xmax: 200, log: true },
    { id: "RHOB", title: "RHOB", unit: "g/cc", color: "#c0392b", xmin: 1.65, xmax: 2.85 },
    { id: "NPHI", title: "NPHI", unit: "v/v", color: "#1d4ed8", xmin: 0.45, xmax: -0.15 },
    { id: "DTC", title: "DTC", unit: "us/ft", color: "#111", xmin: 140, xmax: 40 },
    { id: "DTS", title: "DTS", unit: "us/ft", color: "#7c3aed", xmin: 240, xmax: 80 },
    { id: "PEF", title: "PEF", unit: "b/e", color: "#d97706", xmin: 0, xmax: 15 },
  ];
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

  // ---------- browser UI ----------

  function wheelWindow(PU, win, ext, pixels, height, anchor, zoom, minSpan) {
    return zoom ? PU.zoomAt(win, ext, anchor, Math.exp(Math.max(-6, Math.min(6, pixels * 0.003))), minSpan)
      : PU.panBy(win, ext, pixels / height * (win.max - win.min), minSpan);
  }

  const STD_SCALE = { RHOB: [1.65, 2.85], DTC: [140, 40], DTS: [240, 80] };
  const EMPTY_HELP = "No benchmark results found yet. Generate some with:\n\n" +
    "python -m benchmark.run --model template_linear --dataset force2020 --scenario all --target all\n" +
    "python -m benchmark.scoreboard\n\n" +
    "then reload this tab. Results are read from the results/ folder (or WMB_RESULTS).";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function fmtNum(v, d) { return v == null ? '<span class="na">–</span>' : v.toFixed(d); }

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

  function mount(pane, PU) {
    if (mounted.has(pane)) {
      const draw = mounted.get(pane);
      if (draw) draw();
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
      return;
    }
    const st = { filter: { dataset: "", scenario: "", target: "", test: "" }, model: "", scale: "auto", well: "", data: null, token: 0 };
    pane.innerHTML = `<div class="bm">
      <h2>Scoreboard</h2>
      <div class="bm-bar">
        <label>Dataset <select id="bm-f-dataset"></select></label>
        <label>Scenario <select id="bm-f-scenario"></select></label>
        <label>Target <select id="bm-f-target"></select></label>
        <label id="bm-f-test-wrap" hidden>Test <select id="bm-f-test"></select></label>
      </div>
      <div id="bm-table"></div>
      <h2>Predictions per well</h2>
      <div class="bm-bar">
        <label>Model <select id="bm-model"></select></label>
        <label>Target <select id="bm-target"></select></label>
        <label>Scenario <select id="bm-scenario"></select></label>
        <label>Well <select id="bm-well"></select></label>
        <label>X scale <select id="bm-scale"><option value="auto">Auto</option><option value="std">Standard</option></select></label>
        <button type="button" id="bm-in">Zoom in</button>
        <button type="button" id="bm-out">Zoom out</button>
        <button type="button" id="bm-reset">Full depth</button>
      </div>
      <small>Scroll or drag to pan; pinch on a trackpad or Ctrl/⌘ + scroll to zoom at the pointer.</small>
      <div id="bm-note"></div>
      <div id="bm-plot-wrap"><canvas id="bm-canvas"></canvas></div>
    </div>`;
    const $ = (id) => pane.querySelector("#" + id);

    function renderTable() {
      const shown = filterRows(rows, st.filter);
      const vals = METRICS.map((m) => shown.map((r) => metricValue(r, m)));
      const head = METRICS.map((m) => {
        const dir = m.dir === "high" ? "higher is better" : m.dir === "target" ? ("closer to " + m.ideal + " is better") : "lower is better";
        return `<th class="m-head" title="${esc(m.tip)}" data-dir="${esc(m.dir)}"><span class="m-name">${esc(m.label)}</span><span class="m-dir">${esc(dir)}</span></th>`;
      }).join("");
      const body = shown.map((r) => {
        const cells = METRICS.map((m, j) => {
          const v = metricValue(r, m);
          const f = barFraction(v, vals[j], m);
          const tip = m.key === "speed" && num(r.samples_per_s) != null ? ` title="${r.samples_per_s} samples/s"` : "";
          const bar = f == null ? "" : `<span class="bar" style="width:${Math.max(4, Math.round(f * 100))}%"></span>`;
          return `<td class="m num"${tip}>${fmtNum(v, m.fmt)}${bar}</td>`;
        }).join("");
        const ctx = [["dataset", r.dataset], ["test", r.test], ["target", r.target], ["scenario", r.scenario]]
          .filter(([k]) => !st.filter[k]).map(([, v]) => v).join(" / ");
        return `<tr><td title="${esc(r.dataset + " / " + r.target + " / " + r.scenario)}">${esc(r.contender)}${ctx ? ` <span class="na">${esc(ctx)}</span>` : ""}</td>${cells}</tr>`;
      }).join("");
      $("bm-table").innerHTML = shown.length
        ? `<table><thead><tr><th title="Contender (model) name">model</th>${head}</tr></thead><tbody>${body}</tbody></table>
           <div class="na" style="margin-top:4px">Bars are scaled within the rows shown: longer is better (for cover_90 / cover_50: closer to 0.90 / 0.50).</div>`
        : '<p class="na">No rows match these filters.</p>';
    }

    function renderFilters() {
      $("bm-f-dataset").innerHTML = options(uniq(rows, "dataset"), st.filter.dataset, "All");
      $("bm-f-scenario").innerHTML = options(uniq(rows, "scenario"), st.filter.scenario, "All");
      $("bm-f-target").innerHTML = options(uniq(rows, "target"), st.filter.target, "All");
      const tests = uniq(rows, "test");
      const wrap = $("bm-f-test-wrap");
      if (wrap) {
        wrap.hidden = !tests.length;
        $("bm-f-test").innerHTML = options(tests, st.filter.test, "All");
      }
    }
    for (const k of ["dataset", "scenario", "target", "test"]) {
      const el = $("bm-f-" + k);
      if (el) el.addEventListener("change", (e) => { st.filter[k] = e.target.value; renderTable(); });
    }

    // Plot selectors cascade: model -> target -> scenario over the available rows (dataset follows the filter when set).
    const pool = () => rows.filter((r) => (!st.filter.dataset || r.dataset === st.filter.dataset) && (!st.filter.test || r.test === st.filter.test));
    function renderSelectors() {
      const p = pool();
      const models = uniq(p, "contender");
      if (!models.includes(st.model)) st.model = models[0] || "";
      $("bm-model").innerHTML = options(models, st.model);
      const targets = uniq(p.filter((r) => r.contender === st.model), "target");
      if (!targets.includes(st.target)) st.target = targets[0] || "";
      $("bm-target").innerHTML = options(targets, st.target);
      const scen = uniq(p.filter((r) => r.contender === st.model && r.target === st.target), "scenario");
      if (!scen.includes(st.scenario)) st.scenario = scen.includes("missing_curve") ? "missing_curve" : scen[0] || "";
      $("bm-scenario").innerHTML = options(scen, st.scenario);
      st.run = p.find((r) => r.contender === st.model && r.target === st.target && r.scenario === st.scenario) || null;
      loadWells();
    }

    function query(well) {
      const r = st.run;
      return `contender=${encodeURIComponent(r.contender)}&dataset=${encodeURIComponent(r.dataset)}&target=${encodeURIComponent(r.target)}` +
        `&scenario=${encodeURIComponent(r.scenario)}&well=${encodeURIComponent(well)}`;
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
      $("bm-note").textContent = "Loading well…";
      draw();
      try {
        const j = await getJson(`/api/benchmark/prediction?${query(st.well)}`);
        if (token !== st.token) return;
        st.data = j.well;
        st.win = null;
        st.context = null;
        st.ext = PU.recordedExtent(j.well.depth);
        if (st.ext && st.ext.min === st.ext.max) st.ext = {min: st.ext.min - 0.5, max: st.ext.max + 0.5};
        st.step = PU.medianStep(j.well.depth);
        $("bm-note").textContent = `${st.run.contender} · ${st.run.dataset} · ${st.run.target} · ${st.run.scenario} · ${st.well}`;
        if (j.well.context) {
          st.context = {d: j.well.depth, c: j.well.context};
        } else try {
          const packed = await getJson("/api/well?dataset=" + encodeURIComponent(st.run.dataset) +
            "&well=" + encodeURIComponent(j.well.well_id || st.well));
          if (token !== st.token) return;
          st.context = packed;
        } catch (e) { if (token !== st.token) return; }
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
    $("bm-scenario").addEventListener("change", (e) => { st.scenario = e.target.value; renderSelectors(); });
    $("bm-well").addEventListener("change", (e) => { st.well = e.target.value; loadWell(); });
    $("bm-scale").addEventListener("change", (e) => { st.scale = e.target.value; draw(); });
    for (const k of ["dataset"]) $("bm-f-" + k).addEventListener("change", renderSelectors);


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
      const step = PU.niceStep(dmax - dmin);
      for (let d = Math.ceil(dmin / step) * step; d <= dmax; d += step) {
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
      ctxTracks.forEach((t) => {
        const tw = CTX_TRACK_W;
        const arr = (st.context.c && st.context.c[t.id]) || [];
        const depths = st.context.d || w.depth;
        const std = [t.xmin, t.xmax];
        const sc = st.scale === "std"
          ? { xmin: t.xmin, xmax: t.xmax }
          : PU.autoValueRange(arr, { std: std, log: !!t.log });
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
        : PU.autoValueRange([w.true, w.p50], { std: std });
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

    renderFilters();
    renderTable();
    renderSelectors();
    return draw;
  }

  return { parseTabHash, tabHash, createTabs, METRICS, sharpnessOf, filterRows, barFraction, bandArrays, gapIntervals, contextTracks, wheelWindow, uniq, mount };
});
