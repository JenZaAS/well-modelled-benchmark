/* Prediction tab: free-play prediction of one well. Pure helpers are DOM-free (node-tested); mount() needs a browser. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbPrediction = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function plotStyleApi() {
    if (typeof WmbPlotStyle !== "undefined") return WmbPlotStyle;
    if (typeof require === "function") {
      try { return require("./plot_style.js"); } catch (e) { return null; }
    }
    return null;
  }
  const PS = plotStyleApi();
  const TARGETS = (PS && PS.TARGETS) ? PS.TARGETS.slice() : ["RHOB", "DTC", "DTS"];
  const DEFAULT_DISPLAY = ["GR", "CALI", "RDEP", "RMED", "RHOB", "NPHI", "DTC", "DTS", "PEF"];
  const SHARED_TRACKS = [["RDEP", "RMED", "RSHA"], ["GR", "CALI", "BS"], ["RHOB", "NPHI"], ["DTC", "DTS"], ["PHIT", "PHIE"], ["VSH", "SW"]];
  const LOG_CURVES = ["RDEP", "RMED", "RSHA"];
  const STD_SCALE = PS
    ? Object.fromEntries(Object.keys(PS.CURVES).map((k) => [k, [PS.CURVES[k].xmin, PS.CURVES[k].xmax]]))
    : {
      GR: [0, 150], CALI: [6, 16], RDEP: [0.2, 2000], RMED: [0.2, 2000], RHOB: [1.95, 2.95], NPHI: [0.45, -0.15],
      DTC: [140, 40], DTS: [240, 40], PEF: [0, 10],
    };
  const COLORS = PS
    ? Object.fromEntries(Object.keys(PS.CURVES).map((k) => [k, PS.CURVES[k].color]))
    : { GR: "#1b7a3a", CALI: "#111", RDEP: "#c0392b", RMED: "#1d4ed8", RHOB: "#c0392b", NPHI: "#1d4ed8",
      DTC: "#111", DTS: "#7c3aed", PEF: "#d97706" };
  const MODEL_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"];
  const RANK_METRICS = [
    { key: "MAE", label: "MAE (lower is better)", dir: "low" },
    { key: "RMSE", label: "RMSE (lower is better)", dir: "low" },
    { key: "R2", label: "R2 (higher is better)", dir: "high" },
    { key: "cover_90", label: "cover_90 (closest to 0.90)", dir: "ideal", ideal: 0.9 },
  ];
  const RUN_HINT = "python -m benchmark.run --model template_linear --dataset force2020 --scenario all --target all";
  const Z90 = 1.6449;

  function num(v) { return typeof v === "number" && Number.isFinite(v) ? v : null; }

  // ---------- display tracks ----------

  // The default display curves that exist in the well, in default order.
  function defaultDisplayCurves(available) {
    if (PS) return PS.defaultDisplayNames(available);
    const have = new Set(available || []);
    return DEFAULT_DISPLAY.filter((c) => have.has(c));
  }

  // Available curves that are not part of the default set: the "More logs" choices (display only).
  function moreLogs(available) {
    if (PS) return PS.moreLogNames(available);
    return Array.from(new Set(available || [])).filter((c) => c !== "DEPTH" && !DEFAULT_DISPLAY.includes(c)).sort();
  }

  // Defaults plus the user's extra curves that exist in the well.
  function displayCurves(available, extra) {
    const defaults = defaultDisplayCurves(available);
    const already = new Set(defaults);
    const have = new Set(available || []);
    const more = (extra || []).filter((c) => have.has(c) && !already.has(c));
    return defaults.concat(Array.from(new Set(more)));
  }

  // Curve names -> track descriptors [{id, curves, log}] (RDEP/RMED share a log track).
  function displayTracks(curves) {
    if (PS) {
      return PS.tracksFromNames(curves).map((t) => ({
        id: t.id, title: t.title, curves: t.curves.map((c) => c.k), log: !!t.log, style: t, noData: !!t.noData,
      }));
    }
    const left = new Set(curves);
    const tracks = [];
    for (const c of curves) {
      if (!left.has(c)) continue;
      const group = SHARED_TRACKS.find((g) => g.includes(c));
      const members = group ? group.filter((x) => left.has(x)) : [c];
      members.forEach((x) => left.delete(x));
      tracks.push({ id: members.join("/"), curves: members, log: members.every((x) => LOG_CURVES.includes(x)) });
    }
    return tracks;
  }

  // ---------- eligibility ----------

  // {enabled, status, text} for one entry of /api/predict/models.
  function eligibilityState(m) {
    if (m.eligible) return { enabled: true, status: "ok", text: "ready" };
    const reason = m.reason || "not available";
    let status = "unavailable";
    if (m.error) status = "error";
    else if (/does not support/.test(reason)) status = "unsupported";
    else if (/^opt-in/.test(reason)) status = "optin";
    else if (/missing required/.test(reason)) status = "missing";
    else if (/^needs /.test(reason)) status = "error";
    else if (/select a well/.test(reason)) status = "pending";
    return { enabled: false, status, text: reason };
  }

  // "DTC (required) · GR, NPHI (optional)" for {required, optional}; "no declared inputs" when empty.
  function usedInputsText(u) {
    const parts = [];
    if (u && u.required && u.required.length) parts.push(u.required.join(", ") + " (required)");
    if (u && u.optional && u.optional.length) parts.push(u.optional.join(", ") + " (optional)");
    return parts.length ? parts.join(" · ") : "no declared inputs";
  }


  // Dataset id(s) for a well or a scoreboard row: dataset_key, dataset, and the prefix of "key::well".
  function datasetIds(obj) {
    if (obj == null) return [];
    if (typeof obj === "string") return obj ? [obj] : [];
    const fromId = typeof obj.id === "string" && obj.id.includes("::") ? obj.id.split("::")[0] : null;
    return Array.from(new Set([obj.dataset_key, obj.dataset, obj.datasetKey, fromId].filter((x) => typeof x === "string" && x)));
  }
  function sameDataset(row, ref) {
    const ids = new Set(datasetIds(ref));
    return datasetIds(row).some((id) => ids.has(id));
  }

  // ---------- best model ----------

  function metricDef(key) { return RANK_METRICS.find((m) => m.key === key) || RANK_METRICS[0]; }

  function runTime(row) {
    if (typeof row.run_at !== "string" || !row.run_at) return -Infinity;
    const stamp = row.run_at;
    const t = Date.parse(/(?:Z|[+-]\d{2}:?\d{2})$/i.test(stamp) || stamp.length <= 10 ? stamp : stamp + "Z");
    return Number.isFinite(t) ? t : -Infinity;
  }

  // Rows of one (dataset, target, scenario) ranked best first. One row per model (latest run_at wins);
  // rows without a finite metric and models outside `eligible` (a Set or array, when given) are skipped.
  function rankModels(rows, o) {
    const m = metricDef(o.metric);
    const ok = o.eligible ? new Set(o.eligible) : null;
    const latest = new Map();
    for (const r of rows || []) {
      if (!sameDataset(r, o.well || o) || r.target !== o.target || (r.scenario || "missing_curve") !== (o.scenario || "missing_curve")) continue;
      if (typeof r.contender !== "string" || (ok && !ok.has(r.contender))) continue;
      const old = latest.get(r.contender);
      if (!old || runTime(r) >= runTime(old)) latest.set(r.contender, r);
    }
    const key = (r) => (m.dir === "low" ? r[m.key] : m.dir === "high" ? -r[m.key] : Math.abs(r[m.key] - m.ideal));
    const sorted = Array.from(latest.values()).filter((r) => num(r[m.key]) != null)
      .sort((a, b) => key(a) - key(b) || (a.contender < b.contender ? -1 : 1));
    return sorted.map((r, i) => ({ model: r.contender, rank: i + 1, n: sorted.length, metric: m.key, value: r[m.key],
      run_at: r.run_at || null, scenario: r.scenario || "missing_curve" }));
  }

  function pickBest(rows, o) { return rankModels(rows, o)[0] || null; }

  // Scenarios that have scoreboard rows for this dataset and target (missing_curve first).
  function rankScenarios(rows, dataset, target) {
    const ids = new Set(datasetIds(dataset));
    const s = Array.from(new Set((rows || []).filter((r) => ids.has(r.dataset) && r.target === target).map((r) => r.scenario || "missing_curve")));
    return s.sort((a, b) => (a === "missing_curve" ? -1 : b === "missing_curve" ? 1 : a < b ? -1 : 1));
  }

  function bestText(b) {
    if (!b) return "";
    const v = b.metric === "cover_90" ? b.value.toFixed(2) : b.value.toPrecision(3);
    const when = b.run_at ? ` (run ${String(b.run_at).replace("T", " ")})` : "";
    return `${b.model}: rank ${b.rank} of ${b.n} by ${b.metric} ${v}, scenario ${b.scenario}${when}`;
  }

  // Models to send: the manual picks, plus the best one when "Best model" is on. Manual picks are never dropped.
  function mergeSelection(manual, best, useBest) {
    const out = Array.from(manual || []);
    if (useBest && best && !out.includes(best)) out.push(best);
    return out;
  }

  // ---------- plot data ----------

  // {lo, hi} for the 90% band of one model result (p05/p95, else p50 +/- 1.645 std), or null.
  function bandOf(m) {
    if (m.p05 && m.p95) return { lo: m.p05, hi: m.p95 };
    if (m.p50 && m.std) {
      const f = (s) => m.p50.map((p, i) => (num(p) != null && num(m.std[i]) != null ? p + s * Z90 * m.std[i] : null));
      return { lo: f(-1), hi: f(1) };
    }
    return null;
  }

  function modelColor(i) { return MODEL_COLORS[i] || `hsl(${(i * 137.508) % 360}, 70%, 38%)`; }

  function scoreText(m) {
    const f = (v, d) => (num(v) == null ? "–" : v.toFixed(d));
    return m.mae == null && m.r2 == null ? "" : `MAE ${f(m.mae, 3)} · R2 ${f(m.r2, 2)}`;
  }


  const DEFAULT_AUTO = (PS && PS.AUTO_DEFAULT) ? PS.AUTO_DEFAULT : new Set(["CALI", "PEF"]);

  function filtersApi() {
    if (typeof WmbWellFilters !== "undefined") return WmbWellFilters;
    if (typeof require === "function") {
      try { return require("./well_filters.js"); } catch (e) { return null; }
    }
    return null;
  }
  const WF = filtersApi();

  function wellHasCurve(w, curve) { return WF ? WF.wellHasCurve(w, curve) : true; }
  function filterWells(wells, f) { return WF ? WF.filterWells(wells, f) : (wells || []); }

  // Same x-scale rule as the Viewer: mixed default (Auto for CALI/PEF), else all Auto or all Standard.
  function useAutoScale(curve, xMode) {
    if (PS) return PS.useAutoScale(curve, xMode);
    return xMode === "auto" || (xMode == null && DEFAULT_AUTO.has(curve));
  }

  const HEADER_LINE = 14;
  // One row per curve so twin-track scales (GR/CALI, RHOB/NPHI) do not overlap.
  function trackHeaderRows(names, firstY) {
    const y0 = firstY == null ? 12 : firstY;
    return (names || []).map((k, i) => ({
      k: k,
      title: k,
      y: y0 + i * HEADER_LINE,
      color: (PS && PS.colorOf(k)) || COLORS[k] || "#222",
    }));
  }
  function trackHeaderHeight(tracks) {
    const n = Math.max(1, Math.max.apply(null, [1].concat((tracks || []).map((t) => {
      if (Array.isArray(t)) return t.length;
      if (t && Array.isArray(t.curves)) return Math.max(1, t.curves.length);
      return 1;
    }))));
    return 8 + HEADER_LINE * n;
  }

  function targetScale(curve, series, xMode, PU) {
    const std = (PS && PS.standardScale(curve)) || STD_SCALE[curve] || [0, 1];
    const log = PS ? PS.isLogCurve(curve) : LOG_CURVES.indexOf(curve) >= 0;
    if (useAutoScale(curve, xMode)) {
      const r = PU.autoValueRange(series, { std, log });
      if (!PS || !r) return r;
      const inc = PS.ensureIncreasingScale(curve, r.xmin, r.xmax);
      return Object.assign({}, r, { xmin: inc[0], xmax: inc[1] });
    }
    return { xmin: std[0], xmax: std[1] };
  }
  // Prefer /api/predict "true"; fall back to the well's display logs so the black line can appear even with no result.
  function truthForPlot(result, logs, target) {
    if (result && Array.isArray(result.true) && Array.isArray(result.depth)) {
      return { depth: result.depth, values: result.true, source: "result" };
    }
    const vals = logs && logs.c && logs.c[target];
    if (logs && Array.isArray(logs.d) && Array.isArray(vals) && vals.some((v) => num(v) != null)) {
      return { depth: logs.d, values: vals, source: "logs" };
    }
    return null;
  }
  function modelListState(m, result) {
    const failed = result && (result.models || []).find((x) => x.name === m.name && x.ok === false && x.error);
    if (failed) return { enabled: false, status: "error", text: failed.error };
    return eligibilityState(m);
  }

  // Sections of the model list, in display order. Every model lands in exactly one; unknown states go last.
  const MODEL_GROUPS = [
    { key: "ok", statuses: ["ok"], label: (t) => `Ready for ${t}` },
    { key: "pending", statuses: ["pending"], label: () => "Select a well to check" },
    { key: "missing", statuses: ["missing"], label: () => "Missing inputs in this well" },
    { key: "optin", statuses: ["optin"], label: () => "Opt-in: set flag, restart viewer" },
    { key: "unsupported", statuses: ["unsupported"], label: (t) => `Other targets (not ${t})` },
    { key: "error", statuses: ["error", "unavailable"], label: () => "Unavailable" },
  ];

  // [{key, label, items: [{m, s}]}] for the non-empty sections; s is modelListState(m, result).
  function groupModels(models, result, target) {
    const items = (models || []).map((m) => ({ m, s: modelListState(m, result) }));
    const last = MODEL_GROUPS[MODEL_GROUPS.length - 1];
    const home = (s) => MODEL_GROUPS.find((g) => g.statuses.includes(s.status)) || last;
    return MODEL_GROUPS.map((g) => ({ key: g.key, label: g.label(target || ""), items: items.filter((x) => home(x.s) === g) }))
      .filter((g) => g.items.length);
  }

  // Second line of a model row; the section heading already says why a model is off, so only the detail is shown.
  function modelNote(m, s) {
    if (s.enabled || s.status === "pending") return usedInputsText(m.inputs_used || m.inputs);
    if (s.status === "optin") return m.opt_in || s.text;
    if (s.status === "unsupported") return "";
    return s.text;
  }

  // Widths that fill the plot area: display tracks grow from 100 to 160 px, the prediction track takes the rest
  // (260 to 900 px). Below the minimum the plot keeps 100/260 and scrolls sideways.
  const TRACK_W = [100, 160], PRED_W = [260, 900];
  function plotWidths(avail, nTracks) {
    const n = Math.max(0, Math.floor(nTracks) || 0), room = Math.max(0, num(avail) || 0);
    const grow = Math.max(1, room / (n * TRACK_W[0] + PRED_W[0]));
    const tw = Math.min(TRACK_W[1], TRACK_W[0] * grow);
    const pw = Math.min(PRED_W[1], Math.max(PRED_W[0], room - n * tw));
    return { TW: Math.floor(tw), PW: Math.floor(pw) };
  }

  // ---------- browser UI ----------

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  async function getJson(url, init) {
    const r = await fetch(url, init);
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  }

  const mounted = new WeakMap();

  // ctx: {PU, DATA, L}. First show builds the pane; later shows redraw and resize the minimap.
  function mount(pane, ctx) {
    if (mounted.has(pane)) { mounted.get(pane)(); return; }
    mounted.set(pane, build(pane, ctx));
    mounted.get(pane)();
  }

  function build(pane, ctx) {
    const PU = ctx.PU, L = ctx.L, DATA = ctx.DATA;
    const wells = DATA.wells || [];
    const st = { well: null, target: "RHOB", models: [], manual: new Set(), useBest: false, metric: "MAE", scenario: "missing_curve",
      rows: [], extra: new Set(), logs: null, result: null, curves: [], notes: [], token: 0, wellToken: 0, busy: false,
      modelsReady: false, modelsLoading: true, modelError: "", logError: "", rowsError: "", dropped: [],
      xMode: null, fillSource: "formation", datasets: new Set(), splits: new Set(), hasTargetOn: true, hasLog: "", folded: new Set() };
    const dsChoices = WF ? WF.datasetChoices(wells) : [];
    const splitOpts = WF ? WF.splitChoices(wells) : [];
    pane.innerHTML = `<div class="pd">
      <div class="pd-left">
      <div class="pd-side">
        <div class="pd-mini"><div id="pd-map"></div><div id="pd-well" class="pd-label">No well selected</div></div>
        <div class="pd-filters" id="pd-filters">
          <label>Dataset <select id="pd-dataset"><option value="">All</option>${dsChoices.map((d) =>
            `<option value="${esc(d.id)}">${esc(d.label)}</option>`).join("")}</select></label>
          <div class="pd-splits">${splitOpts.map((s) =>
            `<label title="${esc(WF ? WF.splitTip(s) : s)}"><input type="checkbox" data-split="${esc(s)}" checked> ${esc(s)}</label>`).join(" ")}</div>
          <label><input type="checkbox" id="pd-has-target" checked> has <span id="pd-has-target-lab">RHOB</span></label>
          <label>has log <select id="pd-has-log"><option value="">(any)</option>${(WF ? WF.CORE_LOGS : []).map((c) =>
            `<option>${esc(c)}</option>`).join("")}</select></label>
        </div>
        <label>Well <select id="pd-well-select"></select></label>
        <label>Target <select id="pd-target">${TARGETS.map((t) => `<option>${t}</option>`).join("")}</select></label>
        <h3>More logs <span class="na">(display only)</span></h3>
        <div id="pd-more" class="pd-more"></div>
      </div>
      <div class="pd-modelcol">
        <h3>Models <span id="pd-models-count" class="na"></span></h3>
        <div id="pd-models" class="pd-models"></div>
        <div class="pd-best">
          <label><input type="checkbox" id="pd-best"> Best model</label>
          <label>Rank by <select id="pd-metric">${RANK_METRICS.map((m) => `<option value="${m.key}">${esc(m.label)}</option>`).join("")}</select></label>
          <label>Scenario <select id="pd-scenario"></select></label>
          <div id="pd-best-note" class="na"></div>
        </div>
        <div class="pd-actions">
          <button id="pd-run" type="button">Predict (use cache)</button>
          <button id="pd-rerun" type="button" title="Recompute and replace the cached prediction">Rerun</button>
          <span id="pd-cache" class="na"></span>
        </div>
      </div>
      </div>
      <div class="pd-main">
        <div class="pd-head">
          <div id="pd-note" class="pd-note"></div>
          <div class="range" id="pd-xscale" title="Until you choose a mode for all tracks, CALI and PEF use Auto and the other tracks use Standard. The prediction track shares the target display track's scale.">
            <button type="button" id="pd-xscale-auto" aria-pressed="false">Auto</button>
            <button type="button" id="pd-xscale-std" aria-pressed="false">Standard</button>
          </div>
          <label id="pd-fill-ctl" title="Shade zone bands behind all tracks. The TOPS strip and top lines always show.">Formation fill
            <select id="pd-fill"><option value="formation">Formation</option><option value="group">Group</option><option value="off">Off</option></select></label>
        </div>
        <div id="pd-legend" class="pd-legend"></div>
        <div id="pd-wrap"><canvas id="pd-canvas"></canvas></div>
      </div>
    </div>`;
    const $ = (id) => pane.querySelector("#" + id);

    // ----- minimap (a second Leaflet instance; the Viewer map is untouched) -----
    let map = null, markers = [];
    function initMap() {
      if (map || !L) return;
      const el = $("pd-map");
      if (!el) return;
      map = L.map(el, { zoomControl: false, attributionControl: false });
      let tilesOk = false;
      const coast = L.geoJSON(DATA.coast || { type: "FeatureCollection", features: [] }, {
        style: { color: "#5b6b63", weight: 1.2, opacity: 0.9, fill: false },
        interactive: false,
      });
      function showCoastFallback() {
        if (!tilesOk && map && !map.hasLayer(coast)) coast.addTo(map);
      }
      if (typeof L.tileLayer === "function") {
        const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "&copy; OpenStreetMap",
          subdomains: "abc",
          maxZoom: 12,
        });
        tiles.on("load", () => { tilesOk = true; if (map.hasLayer(coast)) map.removeLayer(coast); });
        tiles.on("tileerror", showCoastFallback);
        tiles.addTo(map);
        setTimeout(showCoastFallback, 1500);
      } else {
        coast.addTo(map);
      }
      const placed = wells.filter((w) => w.lat != null && w.lon != null);
      for (const w of placed) {
        const m = L.circleMarker([w.lat, w.lon], { radius: 4, weight: 1, color: "#333", fillColor: "#2b6ca3", fillOpacity: 0.8 });
        const label = document.createElement("span");
        label.textContent = w.name;
        m.bindTooltip(label, { direction: "top" });
        m.on("click", () => selectWell(w));
        m.addTo(map);
        markers.push({ w, m });
      }
      if (placed.length) map.fitBounds(L.latLngBounds(placed.map((w) => [w.lat, w.lon])).pad(0.1), { maxZoom: 8 });
      else map.setView([59.5, 3.2], 5);
    }
    function layoutMap() {
      try {
        initMap();
        if (map) { map.invalidateSize(); styleMarkers(); }
      } catch (e) { /* tab may not have been laid out yet, or Leaflet may fail in tests */ }
    }
    function styleMarkers() {
      const vis = new Set(visibleWells());
      for (const { w, m } of markers) {
        const show = vis.has(w);
        const el = m.getElement && m.getElement();
        if (el) el.style.display = show ? "" : "none";
        if (typeof m.setStyle === "function") {
          const on = w === st.well;
          m.setStyle({ radius: on ? 7 : 4, fillColor: on ? "#c0392b" : "#2b6ca3", opacity: show ? 1 : 0, fillOpacity: show ? 0.8 : 0 });
          if (on && show) m.bringToFront();
        }
      }
    }

    function filterOpts() {
      return { datasets: st.datasets, splits: st.splits, hasTarget: st.target, hasTargetOn: st.hasTargetOn, hasLog: st.hasLog };
    }
    function visibleWells() { return filterWells(wells, filterOpts()); }

    function renderWellSelect() {
      const vis = visibleWells();
      const sel = $("pd-well-select");
      if (!sel) return;
      const keep = st.well && vis.includes(st.well);
      sel.innerHTML = `<option value="">Select a well</option>` + vis.map((w) =>
        `<option value="${wells.indexOf(w)}">${esc(w.name)} · ${esc(w.dataset)}</option>`).join("");
      sel.value = keep ? String(wells.indexOf(st.well)) : "";
      if (st.well && !keep) selectWell(null);
    }

    function syncHasTargetLabel() {
      const lab = $("pd-has-target-lab");
      if (lab) lab.textContent = st.target;
    }

    function applyFilters() {
      syncHasTargetLabel();
      renderWellSelect();
      try { styleMarkers(); } catch (e) { /* map may not have a size yet */ }
    }

    // ----- state changes -----
    const datasetOf = (w) => w.dataset_key || w.dataset;

    async function selectWell(w) {
      st.well = w; st.logs = null; st.logError = ""; st.extra = new Set();
      $("pd-well").textContent = w ? `${w.name} · ${w.dataset}` : "No well selected";
      $("pd-well-select").value = w ? String(wells.indexOf(w)) : "";
      try { styleMarkers(); } catch (e) { /* map may not have a size yet */ }
      const token = ++st.wellToken;
      changeContext();
      if (w) loadLogs(w, token);
    }

    async function loadLogs(w, token) {
      try {
        let logs = w.c && !w.logs_pending ? w : null;
        if (!logs) logs = await getJson(`/api/well?dataset=${encodeURIComponent(datasetOf(w))}&well=${encodeURIComponent(w.name)}`);
        if (token !== st.wellToken) return;
        st.logs = { d: logs.d || [], c: logs.c || {}, formation: logs.formation || logs.FORMATION, group: logs.group || logs.GROUP };
        renderMore();
      } catch (e) {
        if (token !== st.wellToken) return;
        st.logError = "Could not load logs for display: " + e.message;
      }
      renderNote();
      renderLegend();
      draw();
    }

    function availableCurves() {
      const fromLogs = Object.keys((st.logs && st.logs.c) || {}).filter((k) => (st.logs.c[k] || []).some((v) => num(v) != null));
      return Array.from(new Set(fromLogs.concat(st.curves)));
    }

    function changeContext() {
      st.result = null; st.notes = []; st.dropped = []; st.models = []; st.curves = [];
      st.modelsReady = false; st.modelsLoading = true; st.modelError = "";
      renderModels(); renderBest(); renderMore(); renderNote(); renderLegend(); draw();
      refreshModels(++st.token);
    }

    async function refreshModels(token) {
      const q = st.well
        ? `dataset=${encodeURIComponent(datasetOf(st.well))}&well=${encodeURIComponent(st.well.name)}&target=${st.target}`
        : `target=${st.target}`;
      try {
        const j = await getJson("/api/predict/models?" + q);
        if (token !== st.token) return;
        st.models = j.models || [];
        st.curves = j.curves || st.curves;
        st.modelsReady = true;
      } catch (e) {
        if (token !== st.token) return;
        st.modelError = "Could not list models: " + e.message;
      }
      st.modelsLoading = false;
      const dropped = [];
      for (const name of st.modelsReady && st.well ? st.manual : []) {
        const m = st.models.find((x) => x.name === name);
        if (!m || !m.eligible) { st.manual.delete(name); dropped.push(`${name} (${m ? m.reason : "not registered"})`); }
      }
      st.dropped = dropped;
      renderModels(); renderBest(); renderMore(); renderNote();
    }

    // ----- panels -----
    function renderModels() {
      const groups = groupModels(st.models, st.result, st.target);
      const ready = groups.find((g) => g.key === "ok");
      $("pd-models-count").textContent = st.models.length ? `${ready ? ready.items.length : 0} of ${st.models.length} ready` : "";
      $("pd-models").innerHTML = st.models.length ? groups.map((g) => {
        const rows = g.items.map(({ m, s }) => {
          const note = modelNote(m, s);
          const noteHtml = note ? `<span class="${s.enabled ? "na" : "why " + s.status}">${esc(note)}</span>` : "";
          return `<label class="pd-model${s.enabled ? "" : " off"}" title="${esc(s.enabled ? note : s.text)}">` +
            `<input type="checkbox" data-model="${esc(m.name)}"${s.enabled ? "" : " disabled"}${st.manual.has(m.name) ? " checked" : ""}>` +
            `<span class="pd-mtext"><span class="pd-mname">${esc(m.name)}</span>${noteHtml}</span></label>`;
        }).join("");
        return `<details class="pd-group" data-group="${g.key}"${st.folded.has(g.key) ? "" : " open"}>` +
          `<summary>${esc(g.label)} <span class="na">${g.items.length}</span></summary>${rows}</details>`;
      }).join("") : `<span class="na">${st.modelsLoading ? "Checking models…" : st.modelError ? "Model list unavailable." : "No models registered."}</span>`;
      for (const el of pane.querySelectorAll("input[data-model]")) {
        el.addEventListener("change", () => { el.checked ? st.manual.add(el.dataset.model) : st.manual.delete(el.dataset.model); renderBest(); });
      }
      for (const el of pane.querySelectorAll("details[data-group]")) {
        el.addEventListener("toggle", () => { el.open ? st.folded.delete(el.dataset.group) : st.folded.add(el.dataset.group); });
      }
    }

    function eligibleNames() { return st.models.filter((m) => m.eligible).map((m) => m.name); }

    function currentBest() {
      if (!st.well) return null;
      const opts = { well: st.well, dataset: datasetOf(st.well), target: st.target, scenario: st.scenario, metric: st.metric };
      if (st.modelsReady) opts.eligible = eligibleNames();
      return pickBest(st.rows, opts);
    }

    function renderBest() {
      const scen = new Set(["missing_curve", ...(st.well ? rankScenarios(st.rows, st.well, st.target) : [])]);
      if (!scen.has(st.scenario)) st.scenario = "missing_curve";
      $("pd-scenario").innerHTML = Array.from(scen, (s) => `<option${s === st.scenario ? " selected" : ""}>${esc(s)}</option>`).join("");
      const best = currentBest();
      const box = $("pd-best");
      box.disabled = !best;
      box.checked = Boolean(best && st.useBest);
      $("pd-best-note").innerHTML = best ? esc(bestText(best))
        : !st.well ? "Select a well first." : st.rowsError ? esc(st.rowsError)
          : st.modelsLoading ? "Checking model eligibility…" : !st.modelsReady ? "Model eligibility unavailable."
            : `No matching benchmark results for an eligible model. Run:<br><code>${esc(RUN_HINT)}</code>`;
      $("pd-run").disabled = st.busy || !st.well || !st.modelsReady;
      $("pd-rerun").disabled = $("pd-run").disabled;
    }

    function renderMore() {
      const more = moreLogs(availableCurves());
      $("pd-more").innerHTML = more.length ? more.map((c) =>
        `<label><input type="checkbox" data-more="${esc(c)}"${st.extra.has(c) ? " checked" : ""}> ${esc(c)}</label>`).join(" ")
        : '<span class="na">none</span>';
      for (const el of pane.querySelectorAll("input[data-more]")) {
        el.addEventListener("change", () => { el.checked ? st.extra.add(el.dataset.more) : st.extra.delete(el.dataset.more); draw(); });
      }
    }

    function renderNote() {
      const bits = st.notes.concat([st.logError, st.modelError].filter(Boolean));
      if (st.dropped.length) bits.push("Deselected (no longer eligible): " + st.dropped.join(", "));
      if (st.well && st.models.length && !eligibleNames().length) bits.push("No eligible model for this well and target.");
      if (!st.well) bits.push("Pick a well on the minimap.");
      $("pd-note").innerHTML = bits.map(esc).join("<br>");
      $("pd-cache").textContent = cacheBadge(st.result);
    }

    function renderLegend() {
      const r = st.result;
      const truth = truthForPlot(r, st.logs, st.target);
      const items = truth
        ? [`<span><i style="background:#111"></i>true</span>`]
        : (st.well ? [`<span><i style="background:#111"></i>true (not in this well)</span>`] : []);
      if (!r) { $("pd-legend").innerHTML = items.join(""); return; }
      r.models.forEach((m, i) => {
        if (m.ok) {
          items.push(`<span title="shaded: 90% band"><i style="background:${modelColor(i)}"></i><b>${esc(m.name)}</b> ${esc(scoreText(m))}<br>` +
            `<span class="na">inputs: ${esc(usedInputsText(m.inputs_used))} · ${m.runtime_s}s</span></span>`);
        } else {
          items.push(`<span class="err"><i style="background:#999"></i><b>${esc(m.name)}</b>: ${esc(m.error)}</span>`);
        }
      });
      $("pd-legend").innerHTML = items.join("");
    }

    // ----- events -----
    $("pd-well-select").addEventListener("change", (e) => selectWell(e.target.value === "" ? null : wells[e.target.value]));
    $("pd-target").addEventListener("change", (e) => {
      st.target = e.target.value; applyFilters(); changeContext();
    });
    $("pd-dataset").addEventListener("change", (e) => {
      st.datasets = e.target.value ? new Set([e.target.value]) : new Set();
      applyFilters();
    });
    for (const inp of pane.querySelectorAll("input[data-split]")) {
      inp.addEventListener("change", () => {
        const checked = [...pane.querySelectorAll("input[data-split]:checked")].map((i) => i.dataset.split);
        st.splits = (checked.length === splitOpts.length) ? new Set() : new Set(checked);
        applyFilters();
      });
    }
    $("pd-has-target").addEventListener("change", (e) => { st.hasTargetOn = e.target.checked; applyFilters(); });
    $("pd-has-log").addEventListener("change", (e) => { st.hasLog = e.target.value; applyFilters(); });
    function syncXButtons() {
      const auto = $("pd-xscale-auto"), std = $("pd-xscale-std");
      if (auto && auto.setAttribute) auto.setAttribute("aria-pressed", String(st.xMode === "auto"));
      if (std && std.setAttribute) std.setAttribute("aria-pressed", String(st.xMode === "standard"));
    }
    $("pd-xscale-auto").addEventListener("click", () => { st.xMode = "auto"; syncXButtons(); draw(); });
    $("pd-xscale-std").addEventListener("click", () => { st.xMode = "standard"; syncXButtons(); draw(); });
    syncXButtons();
    $("pd-fill").value = st.fillSource;
    $("pd-fill").addEventListener("change", (e) => { st.fillSource = e.target.value; draw(); });
    applyFilters();
    $("pd-best").addEventListener("change", (e) => { st.useBest = e.target.checked; });
    $("pd-metric").addEventListener("change", (e) => { st.metric = e.target.value; renderBest(); });
    $("pd-scenario").addEventListener("change", (e) => { st.scenario = e.target.value; renderBest(); });
    $("pd-run").addEventListener("click", () => run(false));
    $("pd-rerun").addEventListener("click", () => (
      confirm("Recompute from scratch? This can take minutes for foundation models.") ? run(true) : undefined));

    function cacheBadge(j) {
      const ms = (j && j.models || []).filter((m) => m.ok);
      const hit = ms.filter((m) => m.cached);
      if (!hit.length) return "";
      const at = hit.map((m) => m.cached_at).filter(Boolean).sort().pop() || "";
      return at ? `cached, ${at}` : "cached";
    }

    async function run(force) {
      if (!st.well || st.busy || !st.modelsReady) return;
      const best = st.useBest ? currentBest() : null;
      const models = mergeSelection(Array.from(st.manual), best && best.model, st.useBest);
      if (!models.length) { st.notes = ["Select at least one eligible model (or turn on Best model)."]; renderNote(); return; }
      st.busy = true; st.notes = ["Predicting…"]; renderNote(); $("pd-run").disabled = true; $("pd-rerun").disabled = true;
      const token = st.token;
      try {
        const j = await getJson("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset: datasetOf(st.well), well: st.well.name, target: st.target, models, force: force === true }) });
        if (token !== st.token) return;
        st.result = j; st.curves = j.curves || st.curves; st.notes = [];
        if (best) st.notes.push("Best model: " + bestText(best));
      } catch (e) {
        if (token === st.token) st.notes = ["Prediction failed: " + e.message];
      } finally {
        st.busy = false; renderBest(); $("pd-rerun").disabled = !st.well || !st.modelsReady;
      }
      renderModels(); renderMore(); renderNote(); renderLegend(); draw();
    }

    let rowsToken = 0;
    async function refreshRows() {
      const token = ++rowsToken;
      try {
        const j = await getJson("/api/benchmark/runs");
        if (token !== rowsToken) return;
        st.rows = j.rows || []; st.rowsError = "";
      } catch (e) {
        if (token !== rowsToken) return;
        st.rows = []; st.rowsError = "Could not load benchmark results: " + e.message;
      }
      renderBest();
    }
    refreshModels(st.token);
    renderBest();

    // ----- plot -----
    function stdScale(c) { return STD_SCALE[c] || null; }

    function draw() {
      if (pane.hidden) return;
      const wrap = $("pd-wrap"), cv = $("pd-canvas");
      const box = wrap.getBoundingClientRect();
      const r = st.result;
      const disp = st.logs ? displayTracks(displayCurves(availableCurves(), Array.from(st.extra))) : [];
      const tops = (PS && st.logs) ? PS.formationIntervals(st.logs.d, st.logs.formation, st.logs.group) : [];
      const fills = (PS && st.logs && st.fillSource !== "off") ? PS.formationIntervals(st.logs.d, st.logs.formation, st.logs.group, st.fillSource) : [];
      const L0 = 58, B = 8, STRIP = tops.length ? 18 : 0, GAP = 10, RIGHT = 16;
      const left = L0 + (STRIP ? STRIP + 4 : 0);
      const { TW, PW } = plotWidths(box.width - left - GAP - RIGHT, disp.length);
      const T = Math.max(36, trackHeaderHeight(disp));
      const width = Math.max(box.width, left + disp.length * TW + (r || st.well ? GAP + PW : 0) + RIGHT);
      const dpr = window.devicePixelRatio || 1;
      cv.style.width = width + "px"; cv.style.height = box.height + "px";
      cv.width = Math.max(1, Math.floor(width * dpr)); cv.height = Math.max(1, Math.floor(box.height * dpr));
      const ctx2 = cv.getContext("2d");
      ctx2.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx2.clearRect(0, 0, width, box.height);
      if (!st.well) return;
      const ph = box.height - T - B;
      const c = st.logs ? st.logs.c : {};
      const okModels = r ? r.models.map((m, i) => ({ m, i })).filter((x) => x.m.ok) : [];
      const preds = r ? [r.true].concat(okModels.flatMap((x) => [x.m.p50]).concat(okModels.map((x) => (bandOf(x.m) || {}).lo), okModels.map((x) => (bandOf(x.m) || {}).hi))).filter(Boolean) : [];
      const wins = [];
      if (st.logs && st.logs.d.length) wins.push(PU.defaultWindow(st.logs.d, disp.flatMap((t) => t.curves.map((k) => c[k] || []))));
      if (r && r.depth.length) wins.push(PU.defaultWindow(r.depth, preds));
      if (!wins.length) return;
      const win = { min: Math.min(...wins.map((w) => w.min)), max: Math.max(...wins.map((w) => w.max)) };
      const Y = (d) => T + ((d - win.min) / (win.max - win.min)) * ph;
      ctx2.font = "11px sans-serif";
      // depth axis
      ctx2.fillStyle = "#333"; ctx2.textAlign = "right";
      const ticks = PS ? PS.depthTicks(win.min, win.max) : (function () {
        const step = PU.niceStep(win.max - win.min), out = [];
        for (let d = Math.ceil(win.min / step) * step; d <= win.max; d += step) out.push(d);
        return out;
      }());
      for (const d of ticks) {
        ctx2.strokeStyle = "#eee"; ctx2.beginPath(); ctx2.moveTo(L0, Y(d)); ctx2.lineTo(width, Y(d)); ctx2.stroke();
        ctx2.fillStyle = "#333"; ctx2.fillText(PU.formatTick(d), L0 - 6, Y(d) + 4);
      }

      const frame = (x, w, title) => {
        ctx2.strokeStyle = "#999"; ctx2.lineWidth = 1; ctx2.strokeRect(x, T, w, ph);
        if (title) {
          ctx2.fillStyle = "#222"; ctx2.textAlign = "center"; ctx2.fillText(title, x + w / 2, 12);
        }
      };
      function drawScaleHeader(x, w, names, labelFor) {
        const rows = trackHeaderRows(names);
        rows.forEach((row) => {
          const key = row.k;
          const sc = key ? scaleOf(key) : null;
          const label = labelFor ? labelFor(key, row) : row.title;
          ctx2.font = "10px sans-serif";
          ctx2.fillStyle = row.color;
          if (sc) {
            ctx2.textAlign = "left";
            ctx2.fillText(PU.formatTick(sc.xmin), x + 2, row.y);
            ctx2.textAlign = "right";
            ctx2.fillText(PU.formatTick(sc.xmax), x + w - 2, row.y);
          }
          ctx2.textAlign = "center";
          ctx2.fillText(label, x + w / 2, row.y);
        });
        ctx2.font = "11px sans-serif";
      }
      const line = (depth, arr, X, color, lw, log) => {
        ctx2.strokeStyle = color; ctx2.lineWidth = lw;
        for (const seg of PU.buildSegments(depth, arr, { log })) {
          ctx2.beginPath();
          seg.forEach((i, k) => { const xx = X(arr[i]), yy = Y(depth[i]); k ? ctx2.lineTo(xx, yy) : ctx2.moveTo(xx, yy); });
          ctx2.stroke();
        }
      };
      function scaleOf(k) { return targetScale(k, [c[k] || []], st.xMode, PU); }
      function Xof(k, x0) {
        const sc = scaleOf(k);
        const log = tlog(k);
        return (v) => (log ? PU.xScaleLog(v, sc.xmin, sc.xmax, x0, TW) : PU.xScale(v, sc.xmin, sc.xmax, x0, TW));
      }
      function tlog(k) { return PS ? PS.isLogCurve(k) : LOG_CURVES.indexOf(k) >= 0; }
      let x = L0;
      if (STRIP) {
        ctx2.fillStyle = "#222"; ctx2.textAlign = "center"; ctx2.fillText("TOPS", x + STRIP / 2, 14);
        ctx2.strokeStyle = "#999"; ctx2.strokeRect(x, T, STRIP, ph);
        ctx2.save(); ctx2.beginPath(); ctx2.rect(x, T, STRIP, ph); ctx2.clip();
        for (const iv of tops) {
          const y1 = Y(iv.top), y2 = Y(iv.base);
          ctx2.fillStyle = iv.color;
          ctx2.fillRect(x + 1, Math.min(y1, y2), STRIP - 2, Math.max(1, Math.abs(y2 - y1)));
        }
        ctx2.restore();
        x += STRIP + 4;
      }
      const tracksLeft = x;
      if (fills.length && disp.length) {
        PS.fillFormationBands(ctx2, fills, tracksLeft, disp.length * TW + GAP + PW, Y, T, ph, PS.FORMATION_FILL_ALPHA);
      }
      for (const t of disp) {
        frame(x, TW, t.curves.length ? "" : (t.title || t.id));
        if (!t.curves.length) {
          ctx2.fillStyle = "#777"; ctx2.textAlign = "center";
          ctx2.fillText("no data", x + TW / 2, T + ph / 2);
          x += TW; continue;
        }
        drawScaleHeader(x, TW, t.curves);
        ctx2.save(); ctx2.beginPath(); ctx2.rect(x, T, TW, ph); ctx2.clip();
        if (PS && t.style && t.style.crossFill && t.curves.length >= 2) {
          const kr = t.curves.find((k) => (PS.canonName(k) || k) === "RHOB");
          const kn = t.curves.find((k) => (PS.canonName(k) || k) === "NPHI");
          if (kr && kn) {
            const Xr = Xof(kr, x), Xn = Xof(kn, x), sr = scaleOf(kr), sn = scaleOf(kn);
            const depth = st.logs.d, rv = c[kr] || [], nv = c[kn] || [], step = PU.medianStep(depth);
            for (let i = 0; i < depth.length - 1; i++) {
              if (!PS.fillPairOk(depth, i, step, [rv, nv], [false, false])) continue;
              const kind = PS.densityNeutronFill(rv[i], nv[i], [sr.xmin, sr.xmax], [sn.xmin, sn.xmax]);
              if (!kind) continue;
              const x1r = Xr(rv[i]), x1n = Xn(nv[i]), x2r = Xr(rv[i + 1]), x2n = Xn(nv[i + 1]);
              const y1 = Y(depth[i]), y2 = Y(depth[i + 1]);
              if (![x1r, x1n, x2r, x2n, y1, y2].every(Number.isFinite)) continue;
              ctx2.fillStyle = kind === "yellow" ? PS.CROSS_YELLOW : PS.CROSS_GREY;
              ctx2.beginPath();
              ctx2.moveTo(x1r, y1); ctx2.lineTo(x1n, y1); ctx2.lineTo(x2n, y2); ctx2.lineTo(x2r, y2);
              ctx2.closePath(); ctx2.fill();
            }
          }
        }
        t.curves.forEach((k) => {
          const def = PS && PS.curveDef(k);
          if (def && def.fillFrom != null && def.fillColor) {
            const X = Xof(k, x), arr = c[k] || [], depth = st.logs.d, x0 = X(def.fillFrom), step = PU.medianStep(depth);
            if (Number.isFinite(x0)) {
              ctx2.fillStyle = def.fillColor;
              for (let i = 0; i < depth.length - 1; i++) {
                if (!PS.fillPairOk(depth, i, step, [arr], [tlog(k)])) continue;
                const x1 = X(arr[i]), x2 = X(arr[i + 1]), y1 = Y(depth[i]), y2 = Y(depth[i + 1]);
                if (![x1, x2, y1, y2].every(Number.isFinite)) continue;
                ctx2.beginPath();
                ctx2.moveTo(x0, y1); ctx2.lineTo(x1, y1); ctx2.lineTo(x2, y2); ctx2.lineTo(x0, y2);
                ctx2.closePath(); ctx2.fill();
              }
            }
          }
        });
        t.curves.forEach((k) => {
          const def = PS && PS.curveDef(k);
          ctx2.setLineDash((def && def.dash) || []);
          line(st.logs.d, c[k] || [], Xof(k, x), (PS && PS.colorOf(k)) || COLORS[k] || "#444", 1, tlog(k));
          ctx2.setLineDash([]);
        });
        ctx2.restore();
        x += TW;
      }
      const tracksRight = x;
      // prediction track — always, when the well has the target; same x-scale as the target display track.
      // (A previous auto-range on predictions alone produced e.g. DTS 260-120 vs the display track's standard 240-40,
      // so the black true line did not line up and could look like a lone grey trace.)
      x += GAP;
      frame(x, PW, "");
      const truth = truthForPlot(r, st.logs, st.target);
      const targetSeries = (st.logs && st.logs.c && st.logs.c[st.target]) || (truth && truth.values) || [];
      const sc = targetScale(st.target, [targetSeries], st.xMode, PU);
      drawScaleHeader(x, PW, [st.target], () => `${st.target} (prediction)`);
      const X = (v) => PU.xScale(v, sc.xmin, sc.xmax, x, PW);
      function drawTopsAcross(right) {
        if (!tops.length || !PS) return;
        ctx2.save();
        ctx2.fillStyle = PS.TOP_LINE;
        ctx2.font = "10px sans-serif"; ctx2.textAlign = "left"; ctx2.textBaseline = "bottom";
        for (const iv of tops) {
          const y = Y(iv.top);
          if (y < T || y > T + ph) continue;
          PS.strokeTopLine(ctx2, tracksLeft, right, y);
          ctx2.fillText(iv.name, tracksLeft + 2, y - 1);
        }
        ctx2.restore();
      }
      if (!r && !truth) {
        ctx2.fillStyle = "#777"; ctx2.textAlign = "center";
        ctx2.fillText("Select models and press Predict", x + PW / 2, T + 30);
        drawTopsAcross(x + PW);
        return;
      }
      ctx2.save(); ctx2.beginPath(); ctx2.rect(x, T, PW, ph); ctx2.clip();
      ctx2.globalAlpha = 0.2;
      for (const { m, i } of okModels) {
        const band = bandOf(m);
        if (!band) continue;
        ctx2.fillStyle = modelColor(i);
        const pairedLo = band.lo.map((v, k) => num(band.hi[k]) == null ? null : v);
        for (const seg of PU.buildSegments(r.depth, pairedLo)) {
          if (seg.length < 2) continue;
          ctx2.beginPath();
          seg.forEach((k, j) => (j ? ctx2.lineTo(X(band.lo[k]), Y(r.depth[k])) : ctx2.moveTo(X(band.lo[k]), Y(r.depth[k]))));
          for (let j = seg.length - 1; j >= 0; j--) ctx2.lineTo(X(band.hi[seg[j]]), Y(r.depth[seg[j]]));
          ctx2.closePath(); ctx2.fill();
        }
      }
      ctx2.globalAlpha = 1;   // after every band: inside the loop it left later bands opaque and, with none, a grey truth line
      if (truth) line(truth.depth, truth.values, X, "#111", 1.2);
      for (const { m, i } of okModels) line(r.depth, m.p50, X, modelColor(i), 1.2);
      ctx2.restore();
      if (!r) {
        ctx2.fillStyle = "#777"; ctx2.textAlign = "center";
        ctx2.fillText("Select models and press Predict", x + PW / 2, T + 18);
      }
      drawTopsAcross(x + PW);
    }
    window.addEventListener("resize", () => { layoutMap(); draw(); });

    return function show() {
      // Load the scoreboard even if the minimap constructor throws (zero-size pane, missing Leaflet).
      refreshRows();
      layoutMap();
      if (typeof requestAnimationFrame === "function") requestAnimationFrame(layoutMap);
      setTimeout(layoutMap, 0);
      setTimeout(layoutMap, 80);
      renderLegend();
      draw();
    };
  }

  return {
    TARGETS, DEFAULT_DISPLAY, RANK_METRICS, RUN_HINT, DEFAULT_AUTO, defaultDisplayCurves, moreLogs, displayCurves, displayTracks,
    eligibilityState, usedInputsText, groupModels, modelNote, plotWidths, datasetIds, sameDataset, rankModels, pickBest, rankScenarios, bestText, mergeSelection, bandOf, scoreText, modelColor,
    wellHasCurve, filterWells, useAutoScale, targetScale, trackHeaderRows, trackHeaderHeight, truthForPlot, modelListState, mount,
  };
});
