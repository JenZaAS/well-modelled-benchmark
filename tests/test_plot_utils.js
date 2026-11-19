const test = require("node:test");
const assert = require("node:assert/strict");
const P = require("../viewer/plot_utils.js");

const ext = { min: 0, max: 100 };

test("isGap handles null, NaN, and non-positive log values", () => {
  assert.ok(P.isGap(null) && P.isGap(undefined) && P.isGap(NaN) && P.isGap(Infinity));
  assert.ok(!P.isGap(0));
  assert.ok(P.isGap(0, true) && P.isGap(-1, true) && !P.isGap(1, true));
});

test("buildSegments breaks at nulls and at depth jumps from the last valid sample", () => {
  const d = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  const v = [1, 1, null, 1, 1, null, null, 1, 1];
  assert.deepEqual(P.buildSegments(d, v, { maxGap: 2 }), [[0, 1], [3, 4], [7, 8]]);
  // valid samples separated by a 4-step hole (3 nulls) must not join, even though each null step is small
  const v2 = [1, null, null, null, 1];
  assert.deepEqual(P.buildSegments([0, 1, 2, 3, 4], v2, { maxGap: 2 }), [[0], [4]]);
  // one null (jump of 2 = limit) is still a break per the null rule
  assert.deepEqual(P.buildSegments([0, 1, 2], [1, null, 1], { maxGap: 2 }), [[0], [2]]);
});

test("buildSegments breaks on depth gaps without nulls and on log <= 0", () => {
  assert.deepEqual(P.buildSegments([0, 1, 2, 10, 11], [1, 1, 1, 1, 1], { maxGap: 2 }), [[0, 1, 2], [3, 4]]);
  assert.deepEqual(P.buildSegments([0, 1, 2], [1, 0, 1], { log: true, maxGap: 2 }), [[0], [2]]);
});

test("off-scale values are kept (not dropped as gaps)", () => {
  assert.deepEqual(P.buildSegments([0, 1, 2], [5, 333, 5], { maxGap: 2 }), [[0, 1, 2]]);
});

test("medianStep and visibleRange", () => {
  assert.equal(P.medianStep([0, 2, 4, 6, 100]), 2);
  assert.equal(P.medianStep([1]), 0.15);
  const d = [0, 10, 20, 30, 40, 50];
  assert.deepEqual(P.visibleRange(d, 15, 35), [1, 5]);
  assert.deepEqual(P.visibleRange(d, -5, 100), [0, 6]);
});

test("gap cutoff uses the median positive step rather than the upper middle step", () => {
  const depths = [0, 1, 2, 6, 6, 11];
  const step = P.medianStep(depths);
  assert.equal(step, 2.5);
  assert.deepEqual(
    P.buildSegments(depths, [1, 1, 1, 1, 1, 1], { maxGap: 2 * step }),
    [[0, 1, 2, 3, 4, 5]],
  );
  depths[5] = 12;
  assert.deepEqual(
    P.buildSegments(depths, [1, 1, 1, 1, 1, 1], { maxGap: 2 * P.medianStep(depths) }),
    [[0, 1, 2, 3, 4], [5]],
  );
});

test("xScale supports reversed scales and log; log rejects non-positive", () => {
  assert.equal(P.xScale(0.45, 0.45, -0.15, 0, 100), 0);
  assert.equal(P.xScale(-0.15, 0.45, -0.15, 0, 100), 100);
  assert.equal(P.xScale(100, 140, 40, 10, 50), 30);
  assert.equal(P.xScale(20, 0, 10, 0, 100), 200);
  assert.ok(Math.abs(P.xScaleLog(2, 0.2, 200, 0, 300) - 100) < 1e-9);
  assert.equal(P.xScaleLog(0, 0.2, 200, 0, 300), null);
  assert.equal(P.xScaleLog(Infinity, 0.2, 200, 0, 300), null);
});

test("tickValues and formatTick", () => {
  assert.deepEqual(P.tickValues({ xmin: 140, xmax: 40 }, false), [140, 90, 40]);
  assert.deepEqual(P.tickValues({ xmin: 0.45, xmax: -0.15 }, false).map(P.formatTick), ["0.45", "0.15", "-0.15"]);
  assert.deepEqual(P.tickValues({ xmin: 0.2, xmax: 200 }, true), [0.2, 2, 20, 200]);
  assert.deepEqual(P.tickValues({ xmin: 0.8, xmax: 12 }, true), [1, 10]);
  assert.deepEqual(P.tickValues({ xmin: 2, xmax: 4 }, true), [2, 4]);
  assert.deepEqual(P.tickValues({ xmin: 1e-11, xmax: 1e-10 }, true).map(P.formatTick), ["1e-11", "1e-10"]);
  assert.equal(P.formatTick(0.30000000000000004), "0.3");
});

test("clipRect stays inside the track border", () => {
  assert.deepEqual(P.clipRect(10, 50, 100, 200), { x: 11, y: 51, w: 98, h: 198 });
});

test("decimateRun keeps spikes and passes small runs through", () => {
  const run = Array.from({ length: 1000 }, (_, i) => i);
  const d = run.map((i) => i / 10);
  const v = run.map(() => 1);
  v[503] = 99;
  const out = P.decimateRun(run, d, v, 0, 100, 10);
  assert.ok(out.length < 100 && out.includes(503));
  assert.deepEqual(out, [...out].sort((a, b) => a - b));
  assert.deepEqual(P.decimateRun([0, 1, 2], [0, 1, 2], [1, 2, 3], 0, 2, 100), [0, 1, 2]);
});

test("zoomAt keeps the cursor depth fixed and clamps", () => {
  const w = P.zoomAt({ min: 0, max: 100 }, ext, 25, 0.5, 5);
  assert.deepEqual(w, { min: 12.5, max: 62.5 });
  assert.deepEqual(P.zoomAt({ min: 0, max: 100 }, ext, 50, 5, 5), { min: 0, max: 100 });
  const tiny = P.zoomAt({ min: 40, max: 50 }, ext, 45, 0.0001, 5);
  assert.equal(tiny.max - tiny.min, 5);
});

test("panBy clamps to extent; resetWindow restores it", () => {
  assert.deepEqual(P.panBy({ min: 10, max: 30 }, ext, -50, 5), { min: 0, max: 20 });
  assert.deepEqual(P.panBy({ min: 10, max: 30 }, ext, 500, 5), { min: 80, max: 100 });
  assert.deepEqual(P.resetWindow(ext), { min: 0, max: 100 });
});

test("parseRange ignores invalid input and clamps valid input", () => {
  assert.equal(P.parseRange("abc", "10", ext, 5), null);
  assert.equal(P.parseRange("", "10", ext, 5), null);
  assert.equal(P.parseRange("50", "40", ext, 5), null);
  assert.deepEqual(P.parseRange("-20", "40", ext, 5), { min: 0, max: 40 });
  assert.deepEqual(P.parseRange("10", "500", ext, 5), { min: 10, max: 100 });
  assert.deepEqual(P.parseRange("10", "11", ext, 5), { min: 10, max: 15 });
});

test("minSpanFor uses a few steps, at least 2 m, never beyond the extent", () => {
  assert.equal(P.minSpanFor(0.15, ext), 2);
  assert.equal(P.minSpanFor(2, ext), 10);
  assert.equal(P.minSpanFor(2, { min: 0, max: 3 }), 3);
});

test("recordedExtent and dataBearingExtent ignore empty header depths", () => {
  const d = [0, 1, 2, 7, 8, 9, 10];
  const gr = [null, null, null, 50, 51, 52, 53];
  const rhob = [null, null, null, null, 2.2, 2.3, 2.4];
  assert.deepEqual(P.recordedExtent(d), { min: 0, max: 10 });
  assert.deepEqual(P.dataBearingExtent(d, [gr, rhob]), { min: 7, max: 10 });
  assert.equal(P.dataBearingExtent(d, [gr.map(() => null)]), null);
  assert.equal(P.dataBearingExtent([], []), null);
  assert.deepEqual(P.recordedExtent([5, 5, 5]), { min: 5, max: 5 });
});

test("padExtent adds a small margin and stays inside the recorded range", () => {
  const full = { min: 0, max: 100 };
  assert.deepEqual(P.padExtent({ min: 20, max: 80 }, full, 0.1, 2), { min: 14, max: 86 });
  assert.deepEqual(P.padExtent({ min: 0, max: 10 }, full, 0.5, 2), { min: 0, max: 15 });
  assert.deepEqual(P.padExtent({ min: 90, max: 100 }, full, 0.5, 2), { min: 85, max: 100 });
});

test("defaultWindow prefers padded data-bearing depths over the full index", () => {
  const d = [];
  const v = [];
  for (let i = 0; i <= 100; i++) { d.push(i); v.push(i < 70 ? null : 1); }
  const win = P.defaultWindow(d, [v]);
  // 70-100 padded by max(2, 0.02*30)=2, clamped to 0-100 -> 68-100
  assert.deepEqual(win, { min: 68, max: 100 });
  assert.deepEqual(P.defaultWindow(d, [v.map(() => null)]), { min: 0, max: 100 });
  assert.deepEqual(P.defaultWindow([], []), { min: 0, max: 1 });
});

const cluster = Array.from({ length: 100 }, (_, i) => 6 + (i / 99) * 14);

test("autoValueRange ignores a far outlier and snaps to nice limits", () => {
  const r = P.autoValueRange([...cluster, 1000, null, NaN, Infinity], { std: [6, 16] });
  assert.equal(r.auto, true);
  assert.ok(r.xmin <= 6 && r.xmin >= 4 && r.xmax >= 20 && r.xmax <= 24, JSON.stringify(r));
  assert.equal(r.xmin % 2, 0);
});

test("CALI spanning 6.8 to 28 inches keeps a tight auto scale even with a 1000-inch spike", () => {
  for (const length of [30, 200]) {
    const values = Array.from({ length }, (_, i) => 6.8 + (i / (length - 1)) * (28 - 6.8));
    values.push(1000);
    const r = P.autoValueRange(values, { std: [6, 16] });
    assert.deepEqual(r, { xmin: 5, xmax: 30, auto: true });
    assert.equal(P.countOutside(values, r.xmin, r.xmax), 1);
  }
});

test("autoValueRange pads about 5% before snapping", () => {
  const v = [...Array.from({ length: 50 }, () => 0), ...Array.from({ length: 50 }, () => 100)];
  const r = P.autoValueRange(v, { std: [0, 10] });
  assert.deepEqual([r.xmin, r.xmax], [0, 110]); // non-negative data: do not extend below 0
});

test("autoValueRange falls back on sparse data and expands constant curves", () => {
  assert.deepEqual(P.autoValueRange([1, 2, 3], { std: [6, 16] }), { xmin: 6, xmax: 16, auto: false });
  assert.equal(P.autoValueRange([null, NaN], { std: [6, 16] }).auto, false);
  const c = P.autoValueRange(new Array(20).fill(5), { std: [6, 16] });
  assert.ok(c.auto && c.xmin < 5 && c.xmax > 5);
});

test("autoValueRange keeps reversed direction", () => {
  for (const std of [[0.45, -0.15], [140, 40], [240, 80]]) {
    const v = Array.from({ length: 50 }, (_, i) => std[1] + (std[0] - std[1]) * (0.25 + i / 100));
    const r = P.autoValueRange(v, { std });
    assert.ok(r.auto && r.xmin >= v.at(-1) && r.xmax <= v[0]);
  }
});

test("log autoValueRange snaps to decades, ignores non-positive, tick values follow", () => {
  const v = [0, -1, ...Array.from({ length: 50 }, (_, i) => 0.5 + i * 0.1)];
  const r = P.autoValueRange(v, { std: [0.2, 200], log: true });
  assert.deepEqual([r.xmin, r.xmax], [0.1, 10]);
  assert.deepEqual(P.tickValues(r, true), [0.1, 1, 10]);
  assert.deepEqual(P.tickValues({ xmin: 0.2, xmax: 200 }, true), [0.2, 2, 20, 200]);
});

test("overlay log series share one union range", () => {
  const a = Array.from({ length: 30 }, (_, i) => 1 + i * 0.1);
  const b = Array.from({ length: 30 }, (_, i) => 50 + i);
  const r = P.autoValueRange([a, b], { std: [0.2, 200], log: true });
  assert.deepEqual([r.xmin, r.xmax], [0.1, 100]);
});

test("countOutside counts finite samples beyond the scale", () => {
  assert.equal(P.countOutside([5, 10, 17, null, NaN, 30], 6, 16), 3);
  assert.equal(P.countOutside([0.1, 5, -1, 300], 0.2, 200, true), 2);
  assert.equal(P.countOutside([0, 1], 1, 0), 0);
});

test("autoValueRange crops a far cluster that P1-P99 would still include", () => {
  // Bulk around 3-8, plus a large bad cluster at 80-300 (PEF-like).
  const good = Array.from({ length: 120 }, (_, i) => 3 + (i % 50) / 10);
  const bad = Array.from({ length: 60 }, (_, i) => 80 + i * 3);
  const r = P.autoValueRange(good.concat(bad), { std: [0, 15] });
  assert.equal(r.auto, true);
  assert.ok(r.xmin >= 0 && r.xmin <= 3, r.xmin);
  assert.ok(r.xmax > 8 && r.xmax <= 25, r.xmax);
});
