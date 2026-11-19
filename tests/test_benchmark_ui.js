const test = require("node:test");
const assert = require("node:assert/strict");
const B = require("../viewer/benchmark.js");
const PU = require("../viewer/plot_utils.js");

test("benchmark depth zoom anchors at pointer and scroll reaches both ends", () => {
  const ext = { min: 100, max: 4100 };
  let win = B.wheelWindow(PU, ext, ext, -200, 400, 1100, true, 0.15);
  assert.ok(Math.abs((1100 - win.min) / (win.max - win.min) - 0.25) < 1e-10);
  for (let i = 0; i < 10; i++) win = B.wheelWindow(PU, win, ext, -2000, 400, 1100, true, 0.15);
  assert.ok(Math.abs(win.max - win.min - 0.15) < 1e-9);
  assert.equal(B.wheelWindow(PU, win, ext, 1e10, 400, 1100, false, 0.15).max, ext.max);
  assert.equal(B.wheelWindow(PU, win, ext, -1e10, 400, 1100, false, 0.15).min, ext.min);
});

const IDS = ["viewer", "benchmark"];

test("parseTabHash accepts only registered tab ids", () => {
  assert.equal(B.parseTabHash("#benchmark", IDS, "viewer"), "benchmark");
  assert.equal(B.parseTabHash("viewer", IDS, null), "viewer");
  assert.equal(B.parseTabHash("#well=ABC", IDS, "viewer"), "viewer");
  assert.equal(B.parseTabHash("", IDS, null), null);
  assert.equal(B.tabHash("benchmark"), "#benchmark");
});

test("createTabs toggles panes, reports changes and falls back to the default", () => {
  const panes = { viewer: { hidden: false }, benchmark: { hidden: true } };
  const seen = [];
  const t = B.createTabs({ bar: null, defaultId: "viewer", onChange: (id) => seen.push(id) });
  t.register({ id: "viewer", title: "Viewer", pane: panes.viewer });
  t.register({ id: "benchmark", title: "Benchmark", pane: panes.benchmark, show: () => seen.push("shown") });
  assert.deepEqual(t.ids(), IDS);
  t.select("benchmark");
  assert.deepEqual([panes.viewer.hidden, panes.benchmark.hidden], [true, false]);
  t.select("nope");
  assert.equal(t.current(), "viewer");
  assert.deepEqual(seen, ["shown", "benchmark", "viewer"]);
});

test("barFraction: lower-better, higher-better and distance-from-ideal", () => {
  const low = { dir: "low" }, high = { dir: "high" }, cov = { dir: "target", ideal: 0.9 };
  assert.equal(B.barFraction(1, [1, 3], low), 1);
  assert.equal(B.barFraction(3, [1, 3], low), 0);
  assert.equal(B.barFraction(0.9, [0.5, 0.9], high), 1);
  assert.equal(B.barFraction(0.91, [0.91, 0.5, 0.99], cov), 1);
  assert.ok(B.barFraction(0.99, [0.91, 0.5, 0.99], cov) < 1);
  assert.equal(B.barFraction(0.5, [0.91, 0.5, 0.99], cov), 0);
  assert.equal(B.barFraction(2, [2], low), 1);
  assert.equal(B.barFraction(null, [1, 2], low), null);
});

test("filterRows and sharpness fallback", () => {
  const rows = [
    { dataset: "a", scenario: "missing_curve", target: "RHOB", sharpness_mean_sigma: 0.1, interval_width_90: 0.5 },
    { dataset: "a", scenario: "gap_50", target: "DTC", interval_width_90: 4 },
    { dataset: "b", scenario: "gap_50", target: "DTC" },
  ];
  assert.equal(B.filterRows(rows, { dataset: "a", scenario: "", target: "DTC" }).length, 1);
  assert.equal(B.filterRows(rows, {}).length, 3);
  assert.equal(B.sharpnessOf(rows[0]), 0.1);
  assert.equal(B.sharpnessOf(rows[1]), 4);
  assert.equal(B.sharpnessOf(rows[2]), null);
});

test("bandArrays uses p05/p95 else p50 +/- 1.645 std", () => {
  const w = { p05: [1], p95: [3], p50: [2] };
  assert.deepEqual(B.bandArrays(w), { lo: [1], hi: [3] });
  assert.deepEqual(B.bandArrays({ p05: [1, 2, null], p95: [3, null, 5] }),
    { lo: [1, null, null], hi: [3, null, null] });
  const b = B.bandArrays({ p50: [2, null], std: [1, 1] });
  assert.ok(Math.abs(b.lo[0] - (2 - 1.6449)) < 1e-9 && Math.abs(b.hi[0] - (2 + 1.6449)) < 1e-9);
  assert.equal(b.lo[1], null);
  assert.equal(B.bandArrays({ p50: [2] }), null);
});

test("gapIntervals uses stored gaps, else derives them from nulls in the truth", () => {
  assert.deepEqual(B.gapIntervals({ depth: [0, 1], gaps: [[0, 1]] }), [[0, 1]]);
  const w = { depth: [0, 1, 2, 3, 4], true: [1, null, null, 1, null], p50: [1, 1, 1, 1, 1] };
  assert.deepEqual(B.gapIntervals(w), [[1, 2], [4, 4]]);
});

test("Benchmark mounts once and ignores stale well lists after switching runs", async (t) => {
  const pending = [], elements = new Map(), resize = [];
  t.mock.method(globalThis, "fetch", (url) => new Promise((resolve) => {
    pending.push({ url, respond: (body) => resolve({ ok: true, json: async () => body }) });
  }));
  globalThis.window = { addEventListener: (event, listener) => { if (event === "resize") resize.push(listener); } };
  t.after(() => { delete globalThis.window; });
  const pane = {
    hidden: true,
    querySelector(id) {
      if (!elements.has(id)) elements.set(id, { listeners: [], addEventListener(_, cb) { this.listeners.push(cb); } });
      return elements.get(id);
    },
  };
  const settle = () => new Promise((resolve) => setImmediate(resolve));
  B.mount(pane, PU);
  B.mount(pane, PU);
  assert.equal(pending.length, 1);
  pending[0].respond({ rows: ["m1", "m2"].map((contender) => ({
    contender, dataset: "ds", target: "RHOB", scenario: "missing_curve",
  })) });
  await settle();
  assert.equal(resize.length, 1);
  assert.match(pending[1].url, /contender=m1/);

  elements.get("#bm-model").listeners[0]({ target: { value: "m2" } });
  assert.match(pending[2].url, /contender=m2/);
  pending[2].respond({ wells: ["new-well"] });
  await settle();
  assert.match(pending[3].url, /contender=m2.*well=new-well/);
  pending[1].respond({ wells: ["stale-well"] });
  await settle();
  assert.equal(pending.length, 4);
  assert.match(elements.get("#bm-well").innerHTML, /new-well/);

  pane.innerHTML = "existing pane";
  B.mount(pane, {});
  assert.equal(pane.innerHTML, "existing pane");
  assert.equal(pending.length, 4);
  assert.equal(resize.length, 1);
  pending[3].respond({ well: { well_id: 'new-well', depth: [1, 2], context: {GR: [3, 4], RDEP: [1, 2]} } });
  await settle();
  assert.equal(pending.length, 4, 'embedded logs do not require a separate well lookup');
  assert.doesNotMatch(elements.get('#bm-note').textContent, /unavailable|could not/i);
});

test("contextTracks picks up to 3 present logs excluding the target", () => {
  const curves = { GR: [1], CALI: [8], RHOB: [2.2], NPHI: [0.2], DTC: [80], PEF: [5] };
  assert.deepEqual(B.contextTracks("DTC", curves).map((d) => d.id), ["GR", "CALI", "RHOB"]);
  assert.deepEqual(B.contextTracks("RHOB", ["GR", "NPHI", "DTC"]).map((d) => d.id), ["GR", "NPHI", "DTC"]);
  assert.equal(B.contextTracks("GR", { GR: [1] }).length, 0);
  assert.equal(B.contextTracks("DTC", { GR: [null, null], RHOB: [2] }).map((d) => d.id).join(), "RHOB");
});

test("filterRows tolerates an optional test field and ignores it when unset", () => {
  const rows = [
    { dataset: "a", scenario: "missing_curve", target: "RHOB", test: "official", MAE: 1 },
    { dataset: "a", scenario: "missing_curve", target: "RHOB", test: "custom", MAE: 2 },
    { dataset: "a", scenario: "missing_curve", target: "RHOB", MAE: 3 },
  ];
  assert.equal(B.filterRows(rows, {}).length, 3);
  assert.deepEqual(B.filterRows(rows, { test: "official" }).map((r) => r.MAE), [1]);
  assert.equal(B.filterRows(rows, { dataset: "a", test: "" }).length, 3);
});
