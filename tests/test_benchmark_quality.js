const test = require("node:test");
const assert = require("node:assert/strict");
const B = require("../viewer/benchmark.js");
const Q = require("../viewer/benchmark_quality.js");

const IDS = ["viewer", "benchmark", "prediction"];

test("comparison scale only compresses an isolated maximum", () => {
  for (const values of [[], [NaN, null, Infinity, 0], [1], [1, 4], [1, 9, 9]]) {
    const scale = Q.comparisonScale(values);
    assert.equal(scale.knee, null);
    assert.equal(scale.fraction(0), 0);
  }
  assert.equal(Q.comparisonScale([1, 2, 4]).fraction(2), 0.5);
  const scale = Q.comparisonScale([100, 4, 5, 6]);
  assert.equal(scale.knee, 6);
  assert.equal(scale.fraction(6), 0.8);
  assert.equal(scale.fraction(100), 1);
  const mapped = [0, 4, 5, 6, 7, 100].map(scale.fraction);
  assert.ok(mapped.slice(1).every((v, i) => v > mapped[i]));
});

test("comparison views disclose compressed axes without changing scores", () => {
  const rows = [4, 5, 6, 100].map((MAE, i) => ({ contender: `model${i}`, target: "DTC", MAE,
    interval_width_90: i + 1, cover_90: 0.9 }));
  const run = { rows, props: ["DTC"], models: rows.map((r) => r.contender) };
  const colorOf = () => "#000";
  const html = B.summaryHtml(B.propertyItems(run, "DTC"), { prop: "DTC", colorOf });
  assert.match(html, /Error: scale compressed above 6/);
  assert.match(html, /MAE \(vertical\): scale compressed above 6/);
  assert.doesNotMatch(html, /Width.*?: scale compressed above/);
  assert.match(html, /MAE 100, width 4/);
  assert.match(B.byPropertyHtml(run, colorOf), /Error: scale compressed above 6/);
});

test("tab hash: sub-views resolve to the tab, view helper parses them", () => {
  assert.equal(B.parseTabHash("#benchmark/summary-rhob", IDS, "viewer"), "benchmark");
  assert.equal(B.parseBenchmarkView("#benchmark/summary-rhob"), "summary-rhob");
  assert.equal(B.parseBenchmarkView("#benchmark/byproperty"), "byproperty");
  assert.equal(B.parseBenchmarkView("#benchmark/welllogs"), "welllogs");
  assert.equal(B.parseBenchmarkView("#benchmark"), null);
  assert.equal(B.parseBenchmarkView("#benchmark/nope"), null);
  assert.equal(B.parseBenchmarkView("#prediction/summary-rhob"), null);
  assert.equal(B.parseTabHash("#prediction", IDS, "viewer"), "prediction");
  assert.equal(B.parseTabHash("#leaderboard", IDS, null), null);
  assert.deepEqual(B.viewState("summary-dtc"), { mode: "summary", prop: "DTC" });
  assert.deepEqual(B.viewState("byproperty"), { mode: "byproperty", prop: null });
  assert.deepEqual(B.viewState("welllogs"), { mode: "welllogs", prop: null });
  assert.equal(B.viewId("welllogs"), "welllogs");
  assert.equal(B.benchmarkHash("welllogs"), "#benchmark/welllogs");
  assert.equal(B.viewId("summary", "ALL"), "summary-all");
  assert.equal(B.benchmarkHash("summary-dts"), "#benchmark/summary-dts");
  assert.equal(B.defaultProperty(["DTC", "RHOB"]), "RHOB");
  assert.equal(B.defaultProperty(["DTC", "DTS"]), "DTC");
});

test("a disabled tab never becomes current", () => {
  const t = B.createTabs({ bar: null, defaultId: "viewer" });
  t.register({ id: "viewer", title: "Viewer", pane: { hidden: false } });
  t.register({ id: "benchmark", title: "Benchmark", pane: { hidden: true } });
  t.register({ id: "leaderboard", title: "Leaderboard", disabled: true, tooltip: "Coming later" });
  assert.deepEqual(t.ids(), ["viewer", "benchmark"]);
  t.select("benchmark");
  t.select("leaderboard");
  assert.equal(t.current(), "benchmark");
});

test("accuracy: good within 10%, ok within 30%, poor beyond", () => {
  assert.equal(Q.classifyAccuracy(1.1, 1), "good");
  assert.equal(Q.classifyAccuracy(1.11, 1), "ok");
  assert.equal(Q.classifyAccuracy(1.3, 1), "ok");
  assert.equal(Q.classifyAccuracy(1.31, 1), "poor");
});

test("calibration: 3 points inclusive is good, 8 ok, beyond poor", () => {
  const c = (cover) => Q.classifyCalibration(Math.abs(cover - 0.9));
  assert.equal(c(0.87), "good");
  assert.equal(c(0.93), "good");
  assert.equal(c(0.9), "good");
  assert.equal(c(0.82), "ok");
  assert.equal(c(0.98), "ok");
  assert.equal(c(0.81), "poor");
  assert.equal(c(0.99), "poor");
});

test("sharpness is judged against the best calibrated model and muted when poorly calibrated", () => {
  assert.equal(Q.classifySharpness(1.15, 1, "good"), "good");
  assert.equal(Q.classifySharpness(1.4, 1, "ok"), "ok");
  assert.equal(Q.classifySharpness(1.5, 1, "good"), "poor");
  assert.equal(Q.classifySharpness(0.5, 1, "poor"), "muted");
  const items = Q.scoreItems([
    { id: "a", acc: 1, width: 1, dev: 0.01, speed: 0.01 },
    { id: "b", acc: 1, width: 0.2, dev: 0.2, speed: 0.01 },
  ]);
  assert.equal(items[1].sharpClass, "muted");
  assert.equal(items[0].sharpClass, "good");
});

test("speed zones: real-time <= 0.05 s, batch-OK <= 0.5 s, else slow", () => {
  assert.equal(Q.speedZone(0.05), "good");
  assert.equal(Q.speedZone(0.0501), "ok");
  assert.equal(Q.speedZone(0.5), "ok");
  assert.equal(Q.speedZone(0.51), "poor");
});

test("All mode: relative to template_linear, skipping properties missing for either side", () => {
  const row = (contender, target, MAE, w, cover, s) => ({ contender, target, MAE, interval_width_90: w, cover_90: cover, predict_s_per_well: s });
  const rows = [
    row("template_linear", "RHOB", 0.04, 0.2, 0.9, 0.001), row("template_linear", "DTC", 10, 40, 0.9, 0.001),
    row("m", "RHOB", 0.02, 0.1, 0.95, 0.1), row("m", "DTC", 5, 20, 0.8, 0.3), row("m", "DTS", 7, 30, 0.9, 0.2),
  ];
  const m = Q.aggregateAll(rows).find((x) => x.id === "m");
  assert.ok(Math.abs(m.acc - 0.5) < 1e-12);
  assert.ok(Math.abs(m.width - 0.5) < 1e-12);
  assert.ok(Math.abs(m.dev - (0.05 + 0.1 + 0) / 3) < 1e-12);
  assert.ok(Math.abs(m.speed - 0.2) < 1e-12);
  const base = Q.aggregateAll(rows).find((x) => x.id === "template_linear");
  assert.equal(base.acc, 1);
  // baseline lacks DTS, so a DTS-only model has nothing to compare
  const only = Q.aggregateAll([row("template_linear", "RHOB", 1, 1, 0.9, 1), row("z", "DTS", 3, 3, 0.9, 1)]).find((x) => x.id === "z");
  assert.equal(only.acc, null);
});

test("Pareto front keeps non-dominated points", () => {
  const f = Q.paretoFront([{ id: "a", x: 1, y: 5 }, { id: "b", x: 2, y: 3 }, { id: "c", x: 3, y: 4 }, { id: "d", x: 4, y: 1 }]);
  assert.deepEqual(f.map((p) => p.id), ["a", "b", "d"]);
});

test("speed dial log map", () => {
  assert.equal(Q.speedFrac(0.001), 0);
  assert.equal(Q.speedFrac(100), 1);
  assert.ok(Q.speedFrac(0.007) > Q.speedFrac(0.001));
  assert.ok(Math.abs(Q.speedFrac(1) - 0.6) < 1e-12);
});

test("calibration scales: 90% sits on the target tick, closeness bar full at 0 off", () => {
  assert.ok(Math.abs(Q.calScaleFrac(0.9) - Q.CAL_TARGET_FRAC) < 1e-12);
  assert.ok(Math.abs(Q.CAL_TARGET_FRAC - 0.75) < 1e-12);
  assert.equal(Q.calScaleFrac(0.5), 0);
  assert.equal(Q.calClosenessFrac(0), 1);
  assert.equal(Q.calClosenessFrac(0.05), 0.75);
  assert.equal(Q.calClosenessFrac(0.20), 0);
});

test("runs group by dataset + scenario and are titled by property", () => {
  const r = (contender, target, extra) => Object.assign({ contender, dataset: "force2020", scenario: "gap_50", target, test: target.toLowerCase() + "_gap50" }, extra);
  const runs = B.groupRuns([r("a", "RHOB"), r("a", "DTS"), r("a", "RHOB", { scenario: "missing_curve" })]);
  assert.equal(runs.length, 2);
  const gap = runs.find((x) => x.scenario === "gap_50");
  assert.deepEqual(gap.props, ["RHOB", "DTS"]);
  assert.equal(B.runTitle(gap), "All properties · 50 m gap (target visible around it) · FORCE 2020");
  assert.equal(B.runTitle(runs.find((x) => x.scenario === "missing_curve")), "RHOB · whole well hidden · FORCE 2020");
  const split = B.groupRuns([r("a", "RHOB", { test: "x" }), r("a", "RHOB", { test: "y" })]);
  assert.equal(split.length, 2);
});

test("All ranks only models with every property; partial models show n/N", () => {
  const row = (contender, target, mae) => ({ contender, target, MAE: mae, interval_width_90: mae, cover_90: 0.9, predict_s_per_well: 1 });
  const rows = [
    row("template_linear", "RHOB", 1), row("template_linear", "DTS", 1),
    row("blue_gbt", "RHOB", 0.5), row("blue_gbt", "DTS", 0.6),
    row("black_gardner", "RHOB", 0.1),
  ];
  const run = { rows, props: ["RHOB", "DTS"] };
  const items = B.propertyItems(run, "ALL");
  const gardner = items.find((i) => i.id === "black_gardner");
  assert.deepEqual(gardner.partial, { n: 1, N: 2 });
  assert.equal(gardner.accRank, undefined);
  assert.equal(items.find((i) => i.id === "blue_gbt").accRank, 1);
  assert.equal(items.find((i) => i.id === "template_linear").accRank, 2);
  const html = B.summaryHtml(items, { prop: "ALL", all: true, colorOf: () => "#000" });
  assert.match(html, /black_gardner \(1\/2\)/);
  assert.doesNotMatch(html, /blue_gbt \(/);
  const sharp = html.split("Sharpness: width")[1].split("Speed")[0];
  const sharpRow = (name) => sharp.split('<div class="bm-r">').find((s) => s.includes(name));
  assert.match(sharpRow("blue_gbt"), /aria-label="Rank 1"/);
  assert.doesNotMatch(sharpRow("black_gardner"), /aria-label="Rank/);
  assert.notEqual(items.find((i) => i.id === "blue_gbt").accClass, "poor");
  assert.notEqual(items.find((i) => i.id === "blue_gbt").sharpClass, "poor");
});

test("All Pareto front is drawn from full-property models only", () => {
  const row = (contender, target, mae, w) => ({ contender, target, MAE: mae, interval_width_90: w, cover_90: 0.9, predict_s_per_well: 1 });
  const rows = [
    row("template_linear", "RHOB", 1, 1), row("template_linear", "DTS", 1, 1),
    row("blue_gbt", "RHOB", 0.5, 1.2), row("blue_gbt", "DTS", 0.6, 1.2),
    row("black_gardner", "RHOB", 0.1, 0.1),
  ];
  const items = B.propertyItems({ rows, props: ["RHOB", "DTS"] }, "ALL");
  const html = B.summaryHtml(items, { prop: "ALL", all: true, colorOf: () => "#000" });
  assert.match(html, /black_gardner \(1\/2\)/);
  const line = html.match(/<polyline points="([^"]*)"/);
  assert.ok(line, "front line drawn from the two full models");
  assert.equal(line[1].split(" ").length, 2);
});

test("summary lists models scored in other runs of the dataset but missing here", () => {
  const r = (contender, target, scenario) => ({ contender, target, scenario, dataset: "force2020", test: "t_" + scenario, MAE: 1 });
  const runs = B.groupRuns([
    r("a", "RHOB", "gap_50"), r("red_timesfm25", "RHOB", "gap_50"), r("black_castagna", "DTS", "gap_50"),
    r("a", "RHOB", "missing_curve"),
  ]);
  const whole = runs.find((x) => x.scenario === "missing_curve");
  assert.deepEqual(B.notScoredModels(whole, runs, "RHOB"), ["red_timesfm25"]);
  const html = B.summaryHtml(B.propertyItems(whole, "RHOB"), { prop: "RHOB", colorOf: () => "#000", missing: ["red_timesfm25"] });
  assert.match(html, /Not scored in this test: red_timesfm25/);
  assert.equal(B.modelFamily("red_floor_train_mean"), "floor");
  assert.equal(B.modelFamily("white_ttm"), "foundation");
  assert.equal(B.modelFamily("red_ttm"), "foundation");
  assert.equal(B.modelFamily("red_moment"), "foundation");
  assert.equal(B.modelFamily("black_castagna"), "physics");
});

test("All 'not scored' stays inside the run's properties", () => {
  const r = (contender, target, scenario) => ({ contender, target, scenario, dataset: "force2020", test: "t_" + scenario + target, MAE: 1 });
  const runs = B.groupRuns([
    r("a", "RHOB", "gap_50"), r("red_timesfm25", "RHOB", "gap_50"), r("black_castagna", "DTS", "gap_50"),
    r("a", "RHOB", "missing_curve"),
  ]);
  const whole = runs.find((x) => x.scenario === "missing_curve");
  assert.deepEqual(whole.props, ["RHOB"]);
  assert.deepEqual(B.notScoredModels(whole, runs, "ALL"), ["red_timesfm25"]);
  assert.deepEqual(B.notScoredModels(whole, runs, "RHOB"), ["red_timesfm25"]);
});

test("run header omits missing fields and summarises uneven well counts", () => {
  const base = { contender: "m", dataset: "d", scenario: "gap_50" };
  const bare = B.runHeader({ rows: [Object.assign({ target: "RHOB" }, base)], props: ["RHOB"] });
  assert.deepEqual([bare.wells, bare.samples, bare.time, bare.commit, bare.hardware], [null, null, null, null, null]);
  const rows = [
    Object.assign({ target: "RHOB", n_test_wells: 20, n: 1500, run_at: "2026-11-19T14:00:00+0100", git_commit_full: "abcdef0123456", hardware: { system: "Windows", cpu_count: 16 } }, base),
    Object.assign({ target: "DTS", n_test_wells: 14 }, base),
  ];
  const h = B.runHeader({ rows, props: ["RHOB", "DTS"] });
  assert.equal(h.wells, "20 test wells (14 with DTS)");
  assert.equal(h.commit, "abcdef0");
  assert.deepEqual(h.hardware, { chip: "Windows", detail: "16 threads" });
  assert.match(h.time, /19 Nov 2026, 14:00 Europe\/Oslo/);
  assert.equal(B.runHeader({ rows, props: ["RHOB", "DTS"] }, "RHOB").samples, "1,500 samples");
  const dts = B.runHeader({ rows, props: ["RHOB", "DTS"] }, "DTS");
  assert.deepEqual([dts.wells, dts.samples, dts.time], ["14 test wells", null, h.time]);
});

test("summary switches calibration visuals and keeps a single speed dial", () => {
  const rows = [
    { contender: "template_linear", target: "RHOB", MAE: 0.1, interval_width_90: 0.4, cover_90: 0.93, predict_s_per_well: 0.01 },
    { contender: "m", target: "RHOB", MAE: 0.2, interval_width_90: 0.2, cover_90: 0.76, predict_s_per_well: 0.6 },
  ];
  const ctx = { prop: "RHOB", colorOf: () => "#2F6FAE" };
  const single = B.summaryHtml(B.propertyItems({ rows }, "RHOB"), ctx);
  assert.match(single, /bm-cal-tick" style="left:75\.0%/);
  assert.match(single, /bm-cal-mk good/);
  assert.match(single, /0\.1000 g\/cm³/);
  assert.match(single, /poorly calibrated/);
  const all = B.summaryHtml(B.propertyItems({ rows }, "ALL"), { ...ctx, all: true });
  assert.doesNotMatch(all, /bm-cal-tick/);
  assert.equal((all.match(/>3 pt</g) || []).length, 1);
  assert.match(all, /Full bar = on target/);
  assert.doesNotMatch(all, /Linear baseline/);
  assert.doesNotMatch(all, /baseline = 1\.0/);
  assert.doesNotMatch(all, /better than Linear/);
  assert.match(all, /better: more accurate and sharper/);
  assert.match(all, /worse: less accurate and wider bands/);
  assert.doesNotMatch(all, /Linear template is the reference/);
  assert.equal((all.match(/class="bm-dial"/g) || []).length, 1);
  assert.equal((all.match(/<title>[^<]+ per well<\/title>/g) || []).length, 2);
});

test("relative scatter is centred on the Linear baseline (0..2) with quadrant labels", () => {
  const ext = B.scatterExtent([{ acc: 1, width: 1 }], true);
  assert.equal(ext.xmin, 0);
  assert.equal(ext.xmax, 2);
  assert.equal(ext.ymax, 2);
  const wide = B.scatterExtent([{ acc: 3, width: 0.5 }], true);
  assert.ok(wide.xmax >= 3 && wide.ymax >= 3);
  assert.equal(B.scatterQuadrant(0.7, 0.8), "better");
  assert.equal(B.scatterQuadrant(1.2, 1.1), "worse");
  assert.equal(B.scatterQuadrant(0.7, 1.2), "mixed");
  assert.equal(B.scatterQuadrant(1.2, 0.8), "mixed");
  assert.equal(B.scatterQuadrant(1, 1), "mixed");
  const only = B.summaryHtml(
    [{ id: "template_linear", acc: 1, width: 1, calClass: "ok" }],
    { prop: "ALL", all: true, colorOf: () => "#2F6FAE" },
  );
  assert.doesNotMatch(only, /Linear template is the reference/);
  assert.doesNotMatch(only, /Linear baseline/);
  assert.doesNotMatch(only, /better than Linear/);
  assert.match(only, /better: more accurate and sharper/);
  assert.match(only, /worse: less accurate and wider bands/);
  assert.match(only, /sharper bands, but less accurate/);
  assert.match(only, /more accurate, but wider bands/);
  assert.match(only, /r="3.5"/);
  const svg = B.scatterSvg(
    [{ id: "template_linear", acc: 1, width: 1, calClass: "ok" }],
    { all: true, colorOf: () => "#2F6FAE" },
  );
  assert.match(svg, /viewBox="0 0 850 260"/);
});
