/* Prediction tab: free-play prediction of one well. Pure helpers are DOM-free (node-tested); mount() needs a browser. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbPrediction = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TARGETS = ["RHOB", "DTC", "DTS"];
  const DEFAULT_DISPLAY = ["GR", "CALI", "RDEP", "RMED", "RHOB", "NPHI", "DTC", "DTS", "PEF"];
  const SHARED_TRACKS = [["RDEP", "RMED"]];   // drawn on one (log) track
  const LOG_CURVES = ["RDEP", "RMED"];
  const STD_SCALE = {
    GR: [0, 150], CALI: [6, 16], RDEP: [0.2, 200], RMED: [0.2, 200], RHOB: [1.65, 2.85], NPHI: [0.45, -0.15],
    DTC: [140, 40], DTS: [240, 80], PEF: [0, 15],
  };
  const COLORS = { GR: "#1b7a3a", CALI: "#7a4a12", RDEP: "#111", RMED: "#c0392b", RHOB: "#c0392b", NPHI: "#1d4ed8",
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
    const have = new Set(available || []);
    return DEFAULT_DISPLAY.filter((c) => have.has(c));
  }

  // Available curves that are not part of the default set: the "More logs" choices (display only).
  function moreLogs(available) {
    return Array.from(new Set(available || [])).filter((c) => c !== "DEPTH" && !DEFAULT_DISPLAY.includes(c)).sort();
  }

  // Defaults plus the user's extra curves that exist in the well.
  function displayCurves(available, extra) {
    const have = new Set(available || []);
    const more = (extra || []).filter((c) => have.has(c) && !DEFAULT_DISPLAY.includes(c));
    return defaultDisplayCurves(available).concat(Array.from(new Set(more)));
  }

  // Curve names -> track descriptors [{id, curves, log}] (RDEP/RMED share a log track).
  function displayTracks(curves) {
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
    else if (/missing required/.test(reason)) status = "missing";
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
      modelsReady: false, modelsLoading: true, modelError: "", logError: "", rowsError: "", dropped: [] };
    pane.innerHTML = `<div class="pd">
      <div class="pd-side">
        <div class="pd-mini"><div id="pd-map"></div><div id="pd-well" class="pd-label">No well selected</div></div>
        <label>Well <select id="pd-well-select"><option value="">Select a well</option>${wells.map((w, i) =>
          `<option value="${i}">${esc(w.name)} · ${esc(w.dataset)}</option>`).join("")}</select></label>
        <label>Target <select id="pd-target">${TARGETS.map((t) => `<option>${t}</option>`).join("")}</select></label>
        <h3>Models</h3>
        <div id="pd-models" class="pd-models"></div>
        <div class="pd-best">
          <label><input type="checkbox" id="pd-best"> Best model</label>
          <label>Rank by <select id="pd-metric">${RANK_METRICS.map((m) => `<option value="${m.key}">${esc(m.label)}</option>`).join("")}</select></label>
          <label>Scenario <select id="pd-scenario"></select></label>
          <div id="pd-best-note" class="na"></div>
        </div>
        <button id="pd-run" type="button">Predict</button>
        <h3>More logs <span class="na">(display only)</span></h3>
        <div id="pd-more" class="pd-more"></div>
      </div>
      <div class="pd-main">
        <div id="pd-note" class="pd-note"></div>
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
      for (const { w, m } of markers) {
        const on = w === st.well;
        m.setStyle({ radius: on ? 7 : 4, fillColor: on ? "#c0392b" : "#2b6ca3" });
        if (on) m.bringToFront();
      }
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
        let logs = w.c && !w.logs_pending ? { d: w.d, c: w.c } : null;
        if (!logs) logs = await getJson(`/api/well?dataset=${encodeURIComponent(datasetOf(w))}&well=${encodeURIComponent(w.name)}`);
        if (token !== st.wellToken) return;
        st.logs = { d: logs.d || [], c: logs.c || {} };
        renderMore();
      } catch (e) {
        if (token !== st.wellToken) return;
        st.logError = "Could not load logs for display: " + e.message;
      }
      renderNote();
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
      $("pd-models").innerHTML = st.models.length ? st.models.map((m) => {
        const s = eligibilityState(m);
        const used = m.eligible ? usedInputsText(m.inputs_used || m.inputs) : "";
        const note = s.enabled ? `<span class="na">${esc(used)}</span>` : `<span class="why ${s.status}">${esc(s.text)}</span>`;
        return `<label class="pd-model${s.enabled ? "" : " off"}"><input type="checkbox" data-model="${esc(m.name)}"${s.enabled ? "" : " disabled"}${st.manual.has(m.name) ? " checked" : ""}> ${esc(m.name)}<br>${note}</label>`;
      }).join("") : `<span class="na">${st.modelsLoading ? "Checking models…" : st.modelError ? "Model list unavailable." : "No models registered."}</span>`;
      for (const el of pane.querySelectorAll("input[data-model]")) {
        el.addEventListener("change", () => { el.checked ? st.manual.add(el.dataset.model) : st.manual.delete(el.dataset.model); renderBest(); });
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
    }

    function renderLegend() {
      const r = st.result;
      if (!r) { $("pd-legend").innerHTML = ""; return; }
      const items = [`<span><i style="background:#1f1f1f"></i>true${r.true ? "" : " (not in this well)"}</span>`];
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
      st.target = e.target.value; changeContext();
    });
    $("pd-best").addEventListener("change", (e) => { st.useBest = e.target.checked; });
    $("pd-metric").addEventListener("change", (e) => { st.metric = e.target.value; renderBest(); });
    $("pd-scenario").addEventListener("change", (e) => { st.scenario = e.target.value; renderBest(); });
    $("pd-run").addEventListener("click", run);

    async function run() {
      if (!st.well || st.busy || !st.modelsReady) return;
      const best = st.useBest ? currentBest() : null;
      const models = mergeSelection(Array.from(st.manual), best && best.model, st.useBest);
      if (!models.length) { st.notes = ["Select at least one eligible model (or turn on Best model)."]; renderNote(); return; }
      st.busy = true; st.notes = ["Predicting…"]; renderNote(); $("pd-run").disabled = true;
      const token = st.token;
      try {
        const j = await getJson("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset: datasetOf(st.well), well: st.well.name, target: st.target, models }) });
        if (token !== st.token) return;
        st.result = j; st.curves = j.curves || st.curves; st.notes = [];
        if (best) st.notes.push("Best model: " + bestText(best));
      } catch (e) {
        if (token === st.token) st.notes = ["Prediction failed: " + e.message];
      } finally {
        st.busy = false; renderBest();
      }
      renderMore(); renderNote(); renderLegend(); draw();
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
      const TW = 92, PW = 260, L0 = 58, T = 44, B = 8;
      const width = Math.max(box.width, L0 + disp.length * TW + (r || st.well ? PW : 0) + 16);
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
      const step = PU.niceStep(win.max - win.min);
      for (let d = Math.ceil(win.min / step) * step; d <= win.max; d += step) ctx2.fillText(PU.formatTick(d), L0 - 6, Y(d) + 4);

      const frame = (x, w, title) => {
        ctx2.strokeStyle = "#999"; ctx2.lineWidth = 1; ctx2.strokeRect(x, T, w, ph);
        ctx2.fillStyle = "#222"; ctx2.textAlign = "center"; ctx2.fillText(title, x + w / 2, 14);
      };
      const line = (depth, arr, X, color, lw, log) => {
        ctx2.strokeStyle = color; ctx2.lineWidth = lw;
        for (const seg of PU.buildSegments(depth, arr, { log })) {
          ctx2.beginPath();
          seg.forEach((i, k) => { const xx = X(arr[i]), yy = Y(depth[i]); k ? ctx2.lineTo(xx, yy) : ctx2.moveTo(xx, yy); });
          ctx2.stroke();
        }
      };
      let x = L0;
      for (const t of disp) {
        frame(x, TW, t.id);
        const series = t.curves.map((k) => c[k] || []);
        const first = stdScale(t.curves[0]);
        const sc = first ? { xmin: first[0], xmax: first[1] } : PU.autoValueRange(series, { std: [0, 1] });
        const X = (v) => (t.log ? PU.xScaleLog(v, sc.xmin, sc.xmax, x, TW) : PU.xScale(v, sc.xmin, sc.xmax, x, TW));
        ctx2.save(); ctx2.beginPath(); ctx2.rect(x, T, TW, ph); ctx2.clip();
        t.curves.forEach((k) => line(st.logs.d, c[k] || [], X, COLORS[k] || "#444", 1, t.log));
        ctx2.restore();
        ctx2.fillStyle = "#555"; ctx2.textAlign = "left"; ctx2.fillText(PU.formatTick(sc.xmin), x + 2, 28);
        ctx2.textAlign = "right"; ctx2.fillText(PU.formatTick(sc.xmax), x + TW - 2, 28);
        x += TW;
      }
      // prediction track
      x += 10;
      frame(x, PW, `${st.target} (prediction)`);
      if (!r) { ctx2.fillStyle = "#777"; ctx2.textAlign = "center"; ctx2.fillText("Select models and press Predict", x + PW / 2, T + 30); return; }
      const std = STD_SCALE[st.target] || [0, 1];
      const sc = PU.autoValueRange(preds, { std });
      const X = (v) => PU.xScale(v, sc.xmin, sc.xmax, x, PW);
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
        ctx2.globalAlpha = 1;
      }
      if (r.true) line(r.depth, r.true, X, "#1f1f1f", 1.2);
      for (const { m, i } of okModels) line(r.depth, m.p50, X, modelColor(i), 1.2);
      ctx2.restore();
      ctx2.fillStyle = "#555"; ctx2.textAlign = "left"; ctx2.fillText(PU.formatTick(sc.xmin), x + 2, 28);
      ctx2.textAlign = "right"; ctx2.fillText(PU.formatTick(sc.xmax), x + PW - 2, 28);
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
    TARGETS, DEFAULT_DISPLAY, RANK_METRICS, RUN_HINT, defaultDisplayCurves, moreLogs, displayCurves, displayTracks,
    eligibilityState, usedInputsText, datasetIds, sameDataset, rankModels, pickBest, rankScenarios, bestText, mergeSelection, bandOf, scoreText, modelColor, mount,
  };
});
