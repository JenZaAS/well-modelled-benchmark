/* Pure helpers for the well-map log plot (no DOM). Loaded by wellmap.html and by node tests. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PlotUtils = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // True when a sample cannot be drawn (null, undefined, NaN, Infinity, or <= 0 on a log track).
  function isGap(v, log) {
    if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return true;
    return !!log && v <= 0;
  }

  // Median positive depth step; duplicate depths do not contribute a sampling interval.
  function medianStep(d, fallback) {
    const fb = fallback == null ? 0.15 : fallback;
    if (!d || d.length < 2) return fb;
    const diffs = [];
    for (let i = 1; i < d.length; i++) {
      const s = Math.abs(d[i] - d[i - 1]);
      if (Number.isFinite(s) && s > 0) diffs.push(s);
    }
    if (!diffs.length) return fb;
    diffs.sort((a, b) => a - b);
    const mid = Math.floor(diffs.length / 2);
    return diffs.length % 2 ? diffs[mid] : (diffs[mid - 1] + diffs[mid]) / 2;
  }

  // FORCE CSV and Volve LAS depths are sorted ascending by load_well_frame; pad the visible range by one sample per side.
  function visibleRange(depths, d0, d1) {
    const n = depths.length;
    let lo = 0, hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (depths[m] < d0) lo = m + 1; else hi = m; }
    const start = Math.max(0, lo - 1);
    lo = 0; hi = n;
    while (lo < hi) { const m = (lo + hi) >> 1; if (depths[m] <= d1) lo = m + 1; else hi = m; }
    return [start, Math.min(n, lo + 1)];
  }

  // Split samples [i0, i1) into runs of contiguous valid indices; runs break at gaps and depth jumps > maxGap from the last valid depth.
  function buildSegments(depths, values, opts) {
    const o = opts || {};
    const i0 = o.i0 == null ? 0 : o.i0;
    const i1 = o.i1 == null ? depths.length : Math.min(o.i1, depths.length);
    const maxGap = o.maxGap == null ? Infinity : o.maxGap;
    const segs = [];
    let cur = null;
    let lastValidD = null;
    for (let i = i0; i < i1; i++) {
      const d = depths[i];
      if (isGap(values[i], o.log) || !Number.isFinite(d)) { cur = null; continue; }
      if (cur && lastValidD != null && Math.abs(d - lastValidD) > maxGap) cur = null;
      if (!cur) { cur = []; segs.push(cur); }
      cur.push(i);
      lastValidD = d;
    }
    return segs;
  }

  // Reduce one run of indices to first/min/max/last per pixel row so spikes survive; small runs pass through.
  function decimateRun(run, depths, values, d0, d1, rows) {
    if (run.length <= 4 * rows || d1 <= d0 || rows < 1) return run.slice();
    const out = [];
    let bucket = null, first = 0, last = 0, mn = 0, mx = 0;
    const flush = () => {
      if (bucket === null) return;
      const set = new Set([first, mn, mx, last]);
      out.push(...[...set].sort((a, b) => a - b));
    };
    for (const i of run) {
      const b = Math.floor((depths[i] - d0) / (d1 - d0) * rows);
      if (b !== bucket) {
        flush();
        bucket = b; first = last = mn = mx = i;
      } else {
        last = i;
        if (values[i] < values[mn]) mn = i;
        if (values[i] > values[mx]) mx = i;
      }
    }
    flush();
    return out;
  }

  // Linear x position of v on a scale xmin..xmax (xmin may exceed xmax for reversed scales); not clamped.
  function xScale(v, xmin, xmax, x, w) {
    if (xmax === xmin) return x + w / 2;
    return x + (v - xmin) / (xmax - xmin) * w;
  }

  // Log10 x position; null for invalid samples.
  function xScaleLog(v, xmin, xmax, x, w) {
    if (isGap(v, true)) return null;
    const a = Math.log10(xmin), b = Math.log10(xmax);
    return x + (Math.log10(v) - a) / (b - a) * w;
  }

  // Header tick values for a track curve: log gets 0.2/2/20/200, linear gets min/mid/max in scale order.
  function tickValues(curve, log) {
    if (log) return [0.2, 2, 20, 200];
    return [curve.xmin, (curve.xmin + curve.xmax) / 2, curve.xmax];
  }

  // Compact tick label (avoids float noise such as 0.30000000000000004).
  function formatTick(v) {
    if (Math.abs(v) < 1e-9) return "0";
    return String(parseFloat(v.toPrecision(4)));
  }

  // Plot rectangle used for clipping curves: inside the track border, below the header.
  function clipRect(x, plotTop, w, plotH) {
    return { x: x + 1, y: plotTop + 1, w: Math.max(0, w - 2), h: Math.max(0, plotH - 2) };
  }

  // Round-number depth tick spacing giving roughly six ticks over span.
  function niceStep(span) {
    if (!(span > 0)) return 50;
    const raw = span / 6;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    if (n < 1.5) return mag;
    if (n < 3.5) return 2 * mag;
    if (n < 7.5) return 5 * mag;
    return 10 * mag;
  }

  // Smallest allowed window: a few sample steps, at least 2 m, never more than the full extent.
  function minSpanFor(step, ext) {
    return Math.min(ext.max - ext.min, Math.max(2, 5 * (step || 0)));
  }

  // Shift/shrink a window to lie inside ext with span >= minSpan.
  function clampWindow(win, ext, minSpan) {
    const full = ext.max - ext.min;
    let span = Math.min(full, Math.max(minSpan, win.max - win.min));
    let lo = win.min;
    if (lo < ext.min) lo = ext.min;
    if (lo + span > ext.max) lo = ext.max - span;
    return { min: lo, max: lo + span };
  }

  // Zoom the window about a depth: factor < 1 zooms in, > 1 zooms out; the cursor depth keeps its screen fraction.
  function zoomAt(win, ext, cursorDepth, factor, minSpan) {
    const span = win.max - win.min;
    const frac = span > 0 ? (cursorDepth - win.min) / span : 0.5;
    const ns = Math.min(ext.max - ext.min, Math.max(minSpan, span * factor));
    const lo = cursorDepth - frac * ns;
    return clampWindow({ min: lo, max: lo + ns }, ext, minSpan);
  }

  // Pan the window by dd depth units, clamped to ext.
  function panBy(win, ext, dd, minSpan) {
    return clampWindow({ min: win.min + dd, max: win.max + dd }, ext, minSpan);
  }

  // Full-extent window.
  function resetWindow(ext) {
    return { min: ext.min, max: ext.max };
  }

  // Parse min/max text inputs into a clamped window; null when invalid or min >= max.
  function parseRange(minStr, maxStr, ext, minSpan) {
    if (String(minStr).trim() === "" || String(maxStr).trim() === "") return null;
    const a = Number(minStr), b = Number(maxStr);
    if (!Number.isFinite(a) || !Number.isFinite(b) || a >= b) return null;
    const lo = Math.max(ext.min, a), hi = Math.min(ext.max, b);
    if (hi <= lo) return null;
    return clampWindow({ min: lo, max: hi }, ext, minSpan);
  }

  return {
    isGap, medianStep, visibleRange, buildSegments, decimateRun, xScale, xScaleLog,
    tickValues, formatTick, clipRect, niceStep, minSpanFor, clampWindow, zoomAt, panBy,
    resetWindow, parseRange,
  };
});
