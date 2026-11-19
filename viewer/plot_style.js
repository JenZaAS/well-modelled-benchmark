/* Shared log-plot style for Viewer, Prediction, and the Benchmark well-log plot.
 * DigTech / Erling conventions (scales, colours, layout only — no DigTech code).
 * Display-only extras (PHI/VSH/SW/BS/RSHA, GROUP/FORMATION) are never targets and never computed.
 * DOM-free; node-tested.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbPlotStyle = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const NAVY = "#001f5b";
  const CROSS_YELLOW = "#f5d76e";
  const CROSS_GREY = "#b0b0b0";
  const VSH_FILL = "#d0a079";
  const SHALE_FILL = "#c5d8b8";

  // Standard (fixed) scales and colours. Auto mode may replace xmin/xmax per curve.
  const CURVES = {
    GR:   { k: "GR",   unit: "gAPI",  color: "#1b7a3a", xmin: 0,    xmax: 150,  log: false },
    CALI: { k: "CALI", unit: "in",    color: "#111",    xmin: 6,    xmax: 16,   log: false, autoDefault: true },
    BS:   { k: "BS",   unit: "in",    color: "#7c3aed", xmin: 6,    xmax: 16,   log: false, dash: [6, 3, 2, 3], autoDefault: true },
    RDEP: { k: "RDEP", unit: "ohm.m", color: "#c0392b", xmin: 0.2,  xmax: 2000, log: true },
    RMED: { k: "RMED", unit: "ohm.m", color: "#1d4ed8", xmin: 0.2,  xmax: 2000, log: true },
    RSHA: { k: "RSHA", unit: "ohm.m", color: "#2ca02c", xmin: 0.2,  xmax: 2000, log: true },
    RHOB: { k: "RHOB", unit: "g/cc",  color: "#c0392b", xmin: 1.95, xmax: 2.95, log: false },
    NPHI: { k: "NPHI", unit: "v/v",   color: "#1d4ed8", xmin: 0.45, xmax: -0.15,log: false },
    DTC:  { k: "DTC",  unit: "us/ft", color: "#111",    xmin: 140,  xmax: 40,   log: false },
    DTS:  { k: "DTS",  unit: "us/ft", color: "#7c3aed", xmin: 240,  xmax: 40,   log: false },
    PEF:  { k: "PEF",  unit: "b/e",   color: "#d97706", xmin: 0,    xmax: 10,   log: false, autoDefault: true },
    PHIT: { k: "PHIT", unit: "v/v",   color: "#111",    xmin: 0,    xmax: 0.5,  log: false, displayOnly: true },
    PHIE: { k: "PHIE", unit: "v/v",   color: "#1d4ed8", xmin: 0,    xmax: 0.5,  log: false, displayOnly: true },
    VSH:  { k: "VSH",  unit: "v/v",   color: "#8b5a2b", xmin: 0,    xmax: 1,    log: false, displayOnly: true, fillFrom: 0, fillColor: "#cd853f" },
    SW:   { k: "SW",   unit: "v/v",   color: "#1e90ff", xmin: 0,    xmax: 1,    log: false, displayOnly: true },
  };

  // First listed alias is preferred when several are present. NPHI is not a porosity alias.
  const ALIASES = {
    GR: ["GR"],
    CALI: ["CALI"],
    BS: ["BS", "BIT", "BITSIZE"],
    RDEP: ["RDEP"],
    RMED: ["RMED"],
    RSHA: ["RSHA", "RXO", "MSFL"],
    RHOB: ["RHOB"],
    NPHI: ["NPHI"],
    DTC: ["DTC"],
    DTS: ["DTS"],
    PEF: ["PEF"],
    PHIE: ["PHIE", "PHIF", "LFP_PHIE", "LFP_PHIEC", "LFP_PHIEWC", "PHI"],
    PHIT: ["PHIT", "LFP_PHIT", "PHITC", "LFP_PHITC", "PHITWC", "LFP_PHITWC"],
    VSH: ["VSH", "VSHALE", "VCL", "LFP_VSH", "VSHGR", "LFP_VSHGR", "VSHDN", "LFP_VSHDN"],
    SW: ["SW", "SWT", "SWE", "LFP_SWT", "LFP_SWE", "LFP_SWTC", "LFP_SWEC"],
  };

  const ALIAS_TO_CANON = (function () {
    const out = {};
    Object.keys(ALIASES).forEach((canon) => {
      ALIASES[canon].forEach((a) => { out[String(a).toUpperCase()] = canon; });
      out[canon] = canon;
    });
    return out;
  })();

  const TARGETS = ["RHOB", "DTC", "DTS"];
  const AUTO_DEFAULT = new Set(["CALI", "BS", "PEF"]);
  const FORMATION_KEYS = ["FORMATION", "GROUP"];
  const TOP_LINE = NAVY;
  const FORMATION_PALETTE = [
    "#c9d6e3", "#e6d5b8", "#d4e6c3", "#e4c7c7", "#d7cde8",
    "#cfe6e2", "#f0dcc0", "#d9dde6", "#e8d0c0", "#c8ddd0",
  ];

  // Optional GR shale fill to the right of this API cutoff; null disables the fill.
  const GR_SHALE_CUTOFF = 75;

  const CORE_TRACKS = [
    { id: "gr",  title: "GR / CALI", members: ["GR", "CALI", "BS"], required: true },
    { id: "res", title: "Resistivity", members: ["RDEP", "RMED", "RSHA"], log: true, required: true },
    { id: "dn",  title: "RHOB / NPHI", members: ["RHOB", "NPHI"], required: true, crossFill: true },
    { id: "son", title: "DTC / DTS", members: ["DTC", "DTS"], required: true },
    { id: "pef", title: "PEF", members: ["PEF"], required: true },
  ];
  const EXTRA_TRACKS = [
    { id: "phi", title: "Porosity", members: ["PHIT", "PHIE"], required: false },
    { id: "vsh", title: "VSH / SW", members: ["VSH", "SW"], required: false },
  ];

  function canonName(name) {
    if (name == null || name === "") return null;
    return ALIAS_TO_CANON[String(name).toUpperCase()] || null;
  }

  function curveDef(name) {
    const canon = canonName(name) || String(name || "").toUpperCase();
    const d = CURVES[canon];
    if (!d) return null;
    return Object.assign({}, d, { k: name || d.k, canon: canon });
  }

  function standardScale(name) {
    const d = curveDef(name);
    return d ? [d.xmin, d.xmax] : null;
  }

  function isLogCurve(name) {
    const d = curveDef(name);
    return !!(d && d.log);
  }

  function isAutoDefault(name) {
    const canon = canonName(name) || String(name || "").toUpperCase();
    return AUTO_DEFAULT.has(canon);
  }

  function useAutoScale(name, xMode) {
    return xMode === "auto" || (xMode == null && isAutoDefault(name));
  }

  function colorOf(name) {
    const d = curveDef(name);
    return (d && d.color) || "#444";
  }

  // Pick the preferred alias present in vailable (array or object of curve names).
  function pickPresent(canon, available) {
    const have = available instanceof Set ? available : new Set(
      (Array.isArray(available) ? available : Object.keys(available || {})).map((k) => String(k))
    );
    const upper = new Map();
    have.forEach((k) => upper.set(String(k).toUpperCase(), k));
    for (const alias of (ALIASES[canon] || [canon])) {
      if (upper.has(alias.toUpperCase())) return upper.get(alias.toUpperCase());
    }
    return null;
  }

  function depthGridStep(span) {
    const s = Number(span);
    if (!Number.isFinite(s) || s <= 0) return 25;
    return s <= 600 ? 25 : 100;
  }

  function depthTicks(min, max) {
    const lo = Number(min), hi = Number(max);
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [];
    const step = depthGridStep(hi - lo);
    const start = Math.ceil(lo / step) * step;
    const out = [];
    for (let d = start; d <= hi + 1e-9; d = Math.round((d + step) * 1e6) / 1e6) out.push(d);
    return out;
  }

  function frac(v, xmin, xmax) {
    if (v == null || !Number.isFinite(v) || xmax === xmin) return null;
    return (v - xmin) / (xmax - xmin);
  }

  // yellow: density porosity > neutron (RHOB left of NPHI); grey: neutron > density porosity.
  function densityNeutronFill(rhob, nphi, rhobScale, nphiScale) {
    const rs = rhobScale || [CURVES.RHOB.xmin, CURVES.RHOB.xmax];
    const ns = nphiScale || [CURVES.NPHI.xmin, CURVES.NPHI.xmax];
    const xr = frac(rhob, rs[0], rs[1]);
    const xn = frac(nphi, ns[0], ns[1]);
    if (xr == null || xn == null) return null;
    if (xr < xn) return "yellow";
    if (xn < xr) return "grey";
    return null;
  }

  function hashName(name) {
    let h = 0;
    const s = String(name || "");
    for (let i = 0; i < s.length; i++) h = (h * 33 + s.charCodeAt(i)) >>> 0;
    return h;
  }

  function formationColor(name) {
    return FORMATION_PALETTE[hashName(name) % FORMATION_PALETTE.length];
  }

  function _labelAt(i, formation, group) {
    const raw = (formation && formation[i] != null && String(formation[i]).trim() !== "")
      ? formation[i]
      : (group && group[i] != null ? group[i] : null);
    if (raw == null) return null;
    const s = String(raw).trim();
    if (!s || s.toLowerCase() === "nan" || s === "None") return null;
    return s;
  }

  // Intervals from FORCE FORMATION, else GROUP. Skips null runs. depth is metres, down.
  function formationIntervals(depth, formation, group) {
    const d = depth || [];
    const n = d.length;
    const out = [];
    let i = 0;
    while (i < n) {
      const name = _labelAt(i, formation, group);
      let j = i + 1;
      while (j < n && _labelAt(j, formation, group) === name) j++;
      if (name && Number.isFinite(d[i])) {
        const last = Number.isFinite(d[j - 1]) ? d[j - 1] : d[i];
        out.push({ name: name, top: d[i], base: last, color: formationColor(name) });
      }
      i = j;
    }
    return out;
  }

  function tracksFromNames(available, opts) {
    opts = opts || {};
    const includeEmptyCore = !!opts.includeEmptyCore;
    const names = Array.isArray(available) ? available : Object.keys(available || {});
    const tracks = [];
    const used = new Set();
    const templates = CORE_TRACKS.concat(EXTRA_TRACKS);
    for (const tmpl of templates) {
      const curves = [];
      for (const canon of tmpl.members) {
        const present = pickPresent(canon, names);
        if (!present) continue;
        used.add(String(present).toUpperCase());
        used.add(canon);
        const d = curveDef(present) || CURVES[canon];
        curves.push(Object.assign({}, d, { k: present, canon: canon }));
      }
      if (!curves.length && !(tmpl.required && includeEmptyCore)) continue;
      tracks.push({
        id: tmpl.id,
        title: tmpl.title,
        log: !!tmpl.log,
        required: !!tmpl.required,
        crossFill: !!tmpl.crossFill,
        noData: curves.length === 0,
        curves: curves,
      });
    }
    return tracks;
  }

  function defaultDisplayNames(available) {
    const tracks = tracksFromNames(available, { includeEmptyCore: false });
    const out = [];
    tracks.forEach((t) => t.curves.forEach((c) => out.push(c.k)));
    return out;
  }

  function moreLogNames(available) {
    const used = new Set(defaultDisplayNames(available).map((k) => String(k).toUpperCase()));
    Object.keys(ALIAS_TO_CANON).forEach((a) => {
      if (used.has(ALIAS_TO_CANON[a])) used.add(a);
    });
    return (available || []).filter((k) => k && k !== "DEPTH" && !used.has(String(k).toUpperCase())).sort();
  }

  // Standard-scale tooltip used by the Viewer X-scale control.
  const STANDARD_HELP = "Fixed scales: GR 0-150 gAPI, resistivity 0.2-2000 log, RHOB 1.95-2.95, NPHI 0.45 to -0.15, DTC 140-40, DTS 240-40, CALI 6-16, PEF 0-10, porosity 0-0.5, VSH 0-1, SW 0-1.";

  return {
    CURVES, ALIASES, TARGETS, AUTO_DEFAULT, FORMATION_KEYS, FORMATION_PALETTE,
    NAVY, TOP_LINE, CROSS_YELLOW, CROSS_GREY, VSH_FILL, SHALE_FILL, GR_SHALE_CUTOFF,
    CORE_TRACKS, EXTRA_TRACKS, STANDARD_HELP,
    canonName, curveDef, standardScale, isLogCurve, isAutoDefault, useAutoScale, colorOf,
    pickPresent, depthGridStep, depthTicks, densityNeutronFill, formationColor,
    formationIntervals, tracksFromNames, defaultDisplayNames, moreLogNames,
  };
});
