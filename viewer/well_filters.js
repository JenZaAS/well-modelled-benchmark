/* Shared well filters for the Viewer and Prediction tabs. DOM-free; node-tested. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WmbWellFilters = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const SPLIT_TIPS = {
    train: "Wells with lithology labels used to train in the FORCE 2020 competition",
    leaderboard: "FORCE 2020 public test wells (scored on the open leaderboard during the competition)",
    hidden: "FORCE 2020 final blind test wells",
  };
  const CORE_LOGS = ["GR", "CALI", "RDEP", "RMED", "RHOB", "NPHI", "DTC", "DTS", "PEF"];

  function unique(arr) { return [...new Set(arr.filter(Boolean))]; }

  function splitTip(s) { return SPLIT_TIPS[String(s).toLowerCase()] || ("Wells in split: " + s); }

  function datasetId(w) { return (w && (w.dataset_key || w.dataset)) || ""; }

  function wellHasCurve(w, curve) {
    if (!curve) return true;
    if (!w) return false;
    if (curve === "DTS" && typeof w.has_dts === "boolean") return !!w.has_dts;
    const curves = w.curves;
    if (!curves || typeof curves !== "object" || !Object.keys(curves).length) return true;
    const v = curves[curve];
    return v != null && Number(v) > 0;
  }

  // f.datasets / f.splits: Set or array of values; empty or omitted = all.
  // f.hasTarget: curve name to require when f.hasTargetOn is true (default true if hasTarget is set).
  // f.hasLog: extra required curve, or empty.
  function filterWells(wells, f) {
    f = f || {};
    const splits = f.splits instanceof Set ? f.splits : new Set(f.splits || []);
    const datasets = f.datasets instanceof Set ? f.datasets : new Set(f.datasets || []);
    const requireTarget = f.hasTarget && (f.hasTargetOn !== false);
    const hasLog = f.hasLog || "";
    return (wells || []).filter((w) => {
      if (splits.size && !splits.has(w.split || "")) return false;
      if (datasets.size) {
        const ids = [w.dataset, w.dataset_key].filter(Boolean);
        if (!ids.some((id) => datasets.has(id))) return false;
      }
      if (requireTarget && !wellHasCurve(w, f.hasTarget)) return false;
      if (hasLog && !wellHasCurve(w, hasLog)) return false;
      return true;
    });
  }

  function datasetChoices(wells) {
    const seen = new Map();
    for (const w of wells || []) {
      const key = datasetId(w);
      if (key && !seen.has(key)) seen.set(key, w.dataset || key);
    }
    return [...seen.entries()].map(([id, label]) => ({ id, label }));
  }

  function splitChoices(wells) {
    return unique((wells || []).map((w) => w.split || ""));
  }

  return { SPLIT_TIPS, CORE_LOGS, unique, splitTip, datasetId, wellHasCurve, filterWells, datasetChoices, splitChoices };
});
