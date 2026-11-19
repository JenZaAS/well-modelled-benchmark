const test = require("node:test");
const assert = require("node:assert/strict");
const P = require("../viewer/prediction.js");
const B = require("../viewer/benchmark.js");

test("#prediction is a registered tab hash", () => {
  const ids = ["viewer", "benchmark", "prediction"];
  assert.equal(B.parseTabHash("#prediction", ids, "viewer"), "prediction");
  assert.equal(B.tabHash("prediction"), "#prediction");
  const t = B.createTabs({ bar: null, defaultId: "viewer" });
  for (const id of ids) t.register({ id, title: id, pane: { hidden: true } });
  t.select("prediction");
  assert.equal(t.current(), "prediction");
});

test("default display curves follow the well; More logs lists only the rest", () => {
  const avail = ["DEPTH", "PEF", "GR", "RHOB", "DTC", "LITH_X", "BS", "NPHI", "RDEP"];
  assert.deepEqual(P.defaultDisplayCurves(avail), ["GR", "RDEP", "RHOB", "NPHI", "DTC", "PEF"]);
  assert.deepEqual(P.moreLogs(avail), ["BS", "LITH_X"]);
  assert.deepEqual(P.displayCurves(avail, ["BS", "GR", "NOT_THERE"]), ["GR", "RDEP", "RHOB", "NPHI", "DTC", "PEF", "BS"]);
  assert.deepEqual(P.displayCurves([], ["BS"]), []);
});

test("RDEP and RMED share one log track; other curves get their own", () => {
  const tracks = P.displayTracks(["GR", "RDEP", "RMED", "RHOB"]);
  assert.deepEqual(tracks.map((t) => t.id), ["GR", "RDEP/RMED", "RHOB"]);
  assert.deepEqual(tracks.map((t) => t.log), [false, true, false]);
  assert.deepEqual(P.displayTracks(["RMED"]).map((t) => t.curves), [["RMED"]]);
});

test("eligibilityState explains why a model is greyed out", () => {
  assert.deepEqual(P.eligibilityState({ eligible: true }), { enabled: true, status: "ok", text: "ready" });
  const s = (reason, error) => P.eligibilityState({ eligible: false, reason, error });
  assert.equal(s("does not support target DTS").status, "unsupported");
  assert.equal(s("missing required curve: DTC").status, "missing");
  assert.equal(s("import failed: ModuleNotFoundError: sklearn", true).status, "error");
  assert.equal(s("select a well").status, "pending");
  assert.equal(s("missing required curve: DTC").enabled, false);
  assert.equal(P.usedInputsText({ required: ["DTC"], optional: ["GR", "NPHI"] }), "DTC (required) · GR, NPHI (optional)");
  assert.equal(P.usedInputsText({ required: [], optional: [] }), "no declared inputs");
});

const row = (contender, extra) => Object.assign({ contender, dataset: "ds", target: "RHOB", scenario: "missing_curve", run_at: "2026-01-01T00:00:00" }, extra);
const ROWS = [
  row("a", { MAE: 0.3, RMSE: 0.1, R2: 0.5, cover_90: 0.6 }),
  row("b", { MAE: 0.1, RMSE: 0.3, R2: 0.9, cover_90: 0.97 }),
  row("c", { MAE: 0.2, RMSE: 0.2, R2: 0.7, cover_90: 0.89 }),
  row("x", { MAE: 0.01, target: "DTC" }),
  row("g", { MAE: 0.001, scenario: "gap_50" }),
  row("n", { MAE: null }),
];
const O = { dataset: "ds", target: "RHOB", scenario: "missing_curve" };

test("pickBest ranks by metric, honouring direction and 0.90 coverage", () => {
  const order = (metric) => P.rankModels(ROWS, Object.assign({ metric }, O)).map((r) => r.model).join("");
  assert.equal(order("MAE"), "bca");
  assert.equal(order("RMSE"), "acb");
  assert.equal(order("R2"), "bca");
  assert.equal(order("cover_90"), "cba");
  const best = P.pickBest(ROWS, Object.assign({ metric: "MAE" }, O));
  assert.deepEqual([best.model, best.rank, best.n, best.value, best.run_at], ["b", 1, 3, 0.1, "2026-01-01T00:00:00"]);
});

test("pickBest skips ineligible models, uses the latest run and returns null without results", () => {
  assert.equal(P.pickBest(ROWS, Object.assign({ metric: "MAE", eligible: ["a", "c"] }, O)).model, "c");
  assert.equal(P.pickBest(ROWS, Object.assign({ metric: "MAE", eligible: [] }, O)), null);
  assert.equal(P.pickBest([], Object.assign({ metric: "MAE" }, O)), null);
  assert.equal(P.pickBest(ROWS, { dataset: "ds", target: "RHOB", scenario: "gap_50", metric: "MAE" }).model, "g");
  const rows = ROWS.concat([row("a", { MAE: 0.05, run_at: "2026-02-01T00:00:00" }), row("a", { MAE: 0.001, run_at: "2025-01-01T00:00:00" })]);
  const best = P.pickBest(rows, Object.assign({ metric: "MAE" }, O));
  assert.deepEqual([best.model, best.value], ["a", 0.05]);
  assert.deepEqual(P.rankScenarios(ROWS, "ds", "RHOB"), ["missing_curve", "gap_50"]);
  assert.match(P.bestText(best), /^a: rank 1 of 3 by MAE 0\.0500, scenario missing_curve \(run 2026-02-01 00:00:00\)$/);
});

test("mergeSelection unions best with manual picks and never drops a manual pick", () => {
  assert.deepEqual(P.mergeSelection(["a", "b"], "c", true), ["a", "b", "c"]);
  assert.deepEqual(P.mergeSelection(["a", "b"], "a", true), ["a", "b"]);
  assert.deepEqual(P.mergeSelection(["a"], "c", false), ["a"]);
  assert.deepEqual(P.mergeSelection([], null, true), []);
});

test("latest run wins by instant even when its metric is missing", () => {
  const rows = ROWS.concat([
    row("b", { MAE: 0.01, run_at: "2026-02-01T01:00:00+0200" }),
    row("b", { MAE: null, run_at: "2026-01-31T23:30:00Z" }),
  ]);
  assert.equal(P.pickBest(rows, { ...O, metric: "MAE" }).model, "c");
});

test("changing target keeps in-flight display logs and ignores stale model errors", async (t) => {
  const elements = new Map(), pending = new Map();
  const pane = {
    hidden: true,
    querySelector(id) {
      if (!elements.has(id)) elements.set(id, { addEventListener(event, fn) { this[event] = fn; } });
      return elements.get(id);
    },
    querySelectorAll: () => [],
  };
  t.mock.method(globalThis, "fetch", (url) => new Promise((resolve, reject) => {
    pending.set(url, { resolve: (body) => resolve({ ok: true, json: async () => body }), reject });
  }));
  const oldWindow = globalThis.window;
  globalThis.window = { addEventListener() {} };
  t.after(() => { if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow; });
  P.mount(pane, { DATA: { wells: [{ name: "W", dataset: "ds" }] } });
  pane.querySelector("#pd-well-select").change({ target: { value: "0" } });
  pane.querySelector("#pd-target").change({ target: { value: "DTC" } });
  assert.equal(pane.querySelector("#pd-run").disabled, true);
  pending.get("/api/predict/models?dataset=ds&well=W&target=DTC").resolve({ models: [{ name: "current", eligible: true }] });
  pending.get("/api/predict/models?dataset=ds&well=W&target=RHOB").reject(new Error("stale"));
  pending.get("/api/well?dataset=ds&well=W").resolve({ d: [1], c: { EXTRA: [2] } });
  pending.get("/api/predict/models?target=RHOB").resolve({ models: [] });
  pending.get("/api/benchmark/runs").resolve({ rows: [row("current", { target: "DTC", scenario: "gap_50", MAE: 0.1 })] });
  await new Promise(setImmediate);
  assert.match(pane.querySelector("#pd-more").innerHTML, /EXTRA/);
  assert.match(pane.querySelector("#pd-models").innerHTML, /current/);
  assert.doesNotMatch(pane.querySelector("#pd-note").innerHTML, /stale/);
  assert.equal(pane.querySelector("#pd-run").disabled, false);
  assert.match(pane.querySelector("#pd-scenario").innerHTML, /<option selected>missing_curve/);
  assert.equal(pane.querySelector("#pd-best").disabled, true);
});

test("bandOf prefers p05/p95, else p50 +/- 1.645 std", () => {
  assert.deepEqual(P.bandOf({ p05: [1], p95: [3], p50: [2] }), { lo: [1], hi: [3] });
  const b = P.bandOf({ p50: [2, null], std: [1, 1] });
  assert.ok(Math.abs(b.lo[0] - (2 - 1.6449)) < 1e-9 && b.hi[1] === null);
  assert.equal(P.bandOf({ p50: [1] }), null);
  assert.equal(P.scoreText({ mae: 0.0237, r2: 0.23 }), "MAE 0.024 · R2 0.23");
  assert.equal(P.scoreText({ mae: null, r2: null }), "");
});

test("sameDataset matches dataset_key, dataset, or id prefix", () => {
  const well = { dataset: "FORCE 2020", dataset_key: "force2020", id: "force2020::15/9-14" };
  assert.equal(P.sameDataset({ dataset: "force2020" }, well), true);
  assert.equal(P.sameDataset({ dataset: "FORCE 2020" }, well), true);
  assert.equal(P.sameDataset({ dataset: "volve" }, well), false);
  assert.deepEqual(P.datasetIds(well).sort(), ["FORCE 2020", "force2020"]);
  assert.deepEqual(P.datasetIds("ds"), ["ds"]);
});

test("pickBest finds template_linear on force2020 RHOB missing_curve via the well's dataset_key", () => {
  const rows = [
    { contender: "template_linear", dataset: "force2020", target: "RHOB", scenario: "missing_curve", MAE: 0.099, run_at: "2026-11-19T12:50:01+0100" },
    { contender: "template_linear", dataset: "force2020", target: "DTC", scenario: "missing_curve", MAE: 10, run_at: "2026-11-19T12:50:01+0100" },
  ];
  const well = { name: "15/9-14", dataset: "FORCE 2020", dataset_key: "force2020" };
  const best = P.pickBest(rows, { well, dataset: well.dataset_key, target: "RHOB", scenario: "missing_curve", metric: "MAE", eligible: ["template_linear"] });
  assert.equal(best.model, "template_linear");
  assert.equal(best.value, 0.099);
  assert.deepEqual(P.rankScenarios(rows, well, "RHOB"), ["missing_curve"]);
});

function installFetch(t, pending) {
  t.mock.method(globalThis, "fetch", (url) => new Promise((resolve, reject) => {
    pending.set(url, { resolve: (body) => resolve({ ok: true, json: async () => body }), reject });
  }));
  const oldWindow = globalThis.window;
  globalThis.window = { addEventListener() {} };
  t.after(() => { if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow; });
}

function fakePane() {
  const elements = new Map();
  return {
    hidden: true,
    querySelector(id) {
      if (!elements.has(id)) elements.set(id, { addEventListener(event, fn) { this[event] = fn; } });
      return elements.get(id);
    },
    querySelectorAll: () => [],
  };
}

test("Best model is enabled from scoreboard rows even if the minimap constructor throws", async (t) => {
  const pending = new Map();
  const pane = fakePane();
  installFetch(t, pending);
  const L = { map() { throw new Error("zero-size map pane"); } };
  const well = { name: "15/9-14", dataset: "force2020", dataset_key: "force2020", lat: 58.3, lon: 1.7 };
  P.mount(pane, { DATA: { wells: [well], coast: { type: "FeatureCollection", features: [] } }, L });
  pane.querySelector("#pd-well-select").change({ target: { value: "0" } });
  const resolve = (part, body) => {
    const key = Array.from(pending.keys()).find((k) => k.includes(part));
    assert.ok(key, "missing fetch containing " + part + "; have " + Array.from(pending.keys()).join(", "));
    pending.get(key).resolve(body);
  };
  resolve("well=15%2F9-14&target=RHOB", {
    models: [{ name: "template_linear", eligible: true, inputs_used: { required: ["DTC"], optional: ["GR", "NPHI"] } }],
  });
  resolve("/api/predict/models?target=RHOB", { models: [] });
  resolve("/api/well?", { d: [1], c: { GR: [1], DTC: [80], RHOB: [2] } });
  resolve("/api/benchmark/runs", { rows: [
    { contender: "template_linear", dataset: "force2020", target: "RHOB", scenario: "missing_curve", MAE: 0.099, run_at: "2026-11-19T12:50:01+0100" },
  ] });
  await new Promise(setImmediate);
  assert.equal(pane.querySelector("#pd-best").disabled, false);
  assert.match(pane.querySelector("#pd-best-note").innerHTML, /template_linear/);
  assert.match(pane.querySelector("#pd-best-note").innerHTML, /rank 1/);
});
