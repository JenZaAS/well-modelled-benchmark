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

  // Header tick values for a track curve: log gets the standard 0.2/2/20/200 or the decades inside its
  // limits; linear gets min/mid/max in scale order.
  function tickValues(curve, log) {
    if (log) {
      const lo = Math.min(curve.xmin, curve.xmax), hi = Math.max(curve.xmin, curve.xmax);
      if (lo === 0.2 && hi === 200) return [0.2, 2, 20, 200];
      const a = Math.ceil(Math.log10(lo)), b = Math.floor(Math.log10(hi));
      const stride = Math.max(1, Math.ceil((b - a + 1) / 6));
      const out = [];
      for (let e = a; e <= b; e += stride) {
        const tick = Math.pow(10, e);
        if (tick >= lo && tick <= hi) out.push(tick);
      }
      return out.length ? out : [lo, hi];
    }
    return [curve.xmin, (curve.xmin + curve.xmax) / 2, curve.xmax];
  }

  function finiteValues(arr, positiveOnly) {
    const out = [];
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (typeof v === "number" && Number.isFinite(v) && (!positiveOnly || v > 0)) out.push(v);
    }
    return out;
  }

  // Linear-interpolated percentile (p in 0..100) of an ascending array.
  function percentile(sorted, p) {
    if (!sorted.length) return NaN;
    const pos = (sorted.length - 1) * p / 100;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  // Round-number limit spacing giving roughly eight ticks over span.
  function niceLimitStep(span) {
    const raw = span / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 2.25 ? 2 : n < 3.75 ? 2.5 : n < 7.5 ? 5 : 10) * mag;
  }

  function cleanNum(v) { return parseFloat(v.toPrecision(12)); }

  // Median +/- k * 1.4826 * MAD, to crop far outliers that P1-P99 still includes
  // when a large bad cluster sits well away from the median (e.g. PEF spikes of 50-300).
  function madFence(sorted, k) {
    const kk = k == null ? 3 : k;
    const med = percentile(sorted, 50);
    const dev = sorted.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
    const mad = percentile(dev, 50);
    const scale = 1.4826 * mad;
    if (!(scale > 0)) return null;
    return { lo: med - kk * scale, hi: med + kk * scale };
  }

  // Robust P1/P99 of finite (log: positive) samples, intersected with a MAD fence; null when fewer than minCount.
  function robustBounds(arr, opts) {
    const o = opts || {};
    const v = finiteValues(arr || [], !!o.log).sort((a, b) => a - b);
    if (v.length < (o.minCount == null ? 10 : o.minCount)) return null;
    let lo = percentile(v, 1), hi = percentile(v, 99);
    // On short curves, interpolation with a single extreme can overwhelm both the padding and snap.
    if (v.length >= 3) { lo = Math.max(lo, v[1]); hi = Math.min(hi, v[v.length - 2]); }
    const fence = madFence(v, o.madK);
    if (fence) {
      const flo = Math.max(lo, fence.lo), fhi = Math.min(hi, fence.hi);
      if (fhi > flo) { lo = flo; hi = fhi; }
    }
    return { lo, hi };
  }

  // Pad bounds by 5% and snap to nice limits (decades for log); returns ascending {lo, hi} or null.
  function snapBounds(b, opts) {
    const o = opts || {};
    const pad = o.pad == null ? 0.05 : o.pad;
    if (o.log) {
      let a = Math.log10(b.lo), c = Math.log10(b.hi);
      if (c - a < 1e-9) { a -= 0.5; c += 0.5; }
      const p = (c - a) * pad;
      let fa = Math.floor(a - p), fc = Math.ceil(c + p);
      if (fc <= fa) fc = fa + 1;
      return { lo: Math.pow(10, fa), hi: Math.pow(10, fc) };
    }
    let lo = b.lo, hi = b.hi;
    if (!(hi - lo > 1e-12 * Math.max(1, Math.abs(lo)))) {
      const h = (Math.abs(lo) || 1) * 0.1;
      lo -= h; hi += h;
    }
    const p = (hi - lo) * pad;
    lo -= p; hi += p;
    const step = niceLimitStep(hi - lo);
    const r = { lo: cleanNum(Math.floor(lo / step + 1e-9) * step), hi: cleanNum(Math.ceil(hi / step - 1e-9) * step) };
    return r.hi > r.lo ? r : null;
  }

  // Automatic scale from one array or a list of arrays (overlay union, snapped once). Returns
  // {xmin, xmax, auto} in the standard scale's direction, or the standard scale with auto=false.
  // opts: std [xmin, xmax] (required), log, pad, minCount.
  function autoValueRange(series, opts) {
    const o = opts || {};
    const std = o.std;
    const fallback = { xmin: std[0], xmax: std[1], auto: false };
    const list = Array.isArray(series) && series.some(Array.isArray) ? series : [series || []];
    let lo = Infinity, hi = -Infinity;
    for (const s of list) {
      const b = robustBounds(s, o);
      if (b) { lo = Math.min(lo, b.lo); hi = Math.max(hi, b.hi); }
    }
    if (!(lo <= hi)) return fallback;
    const r = snapBounds({ lo, hi }, o);
    if (!r || !Number.isFinite(r.lo) || !Number.isFinite(r.hi) || (o.log && !(r.lo > 0))) return fallback;
    if (!o.log) {
      const data = [];
      for (const s of list) for (const val of finiteValues(s, false)) data.push(val);
      const stdMin = Math.min(std[0], std[1]);
      if (data.length && data.every((val) => val >= 0) && stdMin >= 0 && r.lo < 0) r.lo = 0;
    }
    if (!(r.hi > r.lo)) return fallback;
    return std[0] > std[1] ? { xmin: r.hi, xmax: r.lo, auto: true } : { xmin: r.lo, xmax: r.hi, auto: true };
  }

  // Count of finite samples (log: also positive) outside the scale; full array, not a window.
  function countOutside(arr, xmin, xmax, log) {
    const lo = Math.min(xmin, xmax), hi = Math.max(xmin, xmax);
    let n = 0;
    for (let i = 0; i < (arr || []).length; i++) {
      const v = arr[i];
      if (typeof v !== "number" || !Number.isFinite(v) || (log && v <= 0)) continue;
      if (v < lo || v > hi) n++;
    }
    return n;
  }

  // Compact tick label (avoids float noise such as 0.30000000000000004).
  function formatTick(v) {
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

  // Min/max of finite depth samples (the recorded index), or null if none.
  function recordedExtent(depths) {
    let lo = Infinity, hi = -Infinity;
    if (!depths) return null;
    for (const d of depths) {
      if (Number.isFinite(d)) { if (d < lo) lo = d; if (d > hi) hi = d; }
    }
    if (!Number.isFinite(lo)) return null;
    return { min: lo, max: hi > lo ? hi : lo };
  }

  // Min/max depth where at least one aligned series has a finite value, or null.
  function dataBearingExtent(depths, seriesList) {
    if (!depths || !depths.length) return null;
    const series = seriesList || [];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < depths.length; i++) {
      const d = depths[i];
      if (!Number.isFinite(d)) continue;
      let ok = false;
      for (const s of series) {
        if (s && i < s.length && typeof s[i] === 'number' && Number.isFinite(s[i])) { ok = true; break; }
      }
      if (!ok) continue;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    if (!Number.isFinite(lo)) return null;
    return { min: lo, max: hi > lo ? hi : lo };
  }

  // Expand ext by frac of its span (at least minPad), clamped to full when given.
  function padExtent(ext, full, frac, minPad) {
    if (!ext) return null;
    const f = frac == null ? 0.02 : frac;
    const pad = Math.max(minPad == null ? 2 : minPad, (ext.max - ext.min) * f);
    let lo = ext.min - pad, hi = ext.max + pad;
    if (hi <= lo) hi = lo + Math.max(pad, 1);
    if (full) {
      if (lo < full.min) lo = full.min;
      if (hi > full.max) hi = full.max;
      if (hi <= lo) return { min: full.min, max: full.max > full.min ? full.max : full.min + 1 };
    }
    return { min: lo, max: hi };
  }

  // Default / Reset window: padded data-bearing interval, else the recorded index.
  function defaultWindow(depths, seriesList) {
    const full = recordedExtent(depths);
    if (!full) return { min: 0, max: 1 };
    const data = dataBearingExtent(depths, seriesList) || full;
    return padExtent(data, full) || { min: full.min, max: full.max > full.min ? full.max : full.min + 1 };
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
    tickValues, formatTick, percentile, autoValueRange, countOutside, clipRect, niceStep, minSpanFor, clampWindow, zoomAt, panBy,
    resetWindow, parseRange, recordedExtent, dataBearingExtent, padExtent, defaultWindow,
  };
});
