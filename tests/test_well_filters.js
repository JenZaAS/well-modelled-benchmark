const test = require("node:test");
const assert = require("node:assert/strict");
const F = require("../viewer/well_filters.js");

const wells = [
  { name: "16/5-3", dataset: "force2020", dataset_key: "force2020", split: "leaderboard", has_dts: true, curves: { DTS: 80, GR: 90, RHOB: 70 } },
  { name: "15/9-14", dataset: "force2020", dataset_key: "force2020", split: "train", has_dts: false, curves: { GR: 90, RHOB: 80 } },
  { name: "15/9-19A", dataset: "volve", dataset_key: "volve", split: "", has_dts: true, curves: { DTS: 50, GR: 40 } },
];

test("wellHasCurve uses has_dts or a positive curve completeness", () => {
  assert.equal(F.wellHasCurve(wells[0], "DTS"), true);
  assert.equal(F.wellHasCurve(wells[1], "DTS"), false);
  assert.equal(F.wellHasCurve(wells[1], "GR"), true);
  assert.equal(F.wellHasCurve(wells[1], "NPHI"), false);
  assert.equal(F.wellHasCurve(wells[0], ""), true);
});

test("filterWells: dataset All vs one dataset, splits, has target, has log", () => {
  assert.equal(F.filterWells(wells, {}).length, 3);
  assert.deepEqual(F.filterWells(wells, { datasets: ["volve"] }).map((w) => w.name), ["15/9-19A"]);
  assert.deepEqual(F.filterWells(wells, { datasets: ["force2020"], splits: ["train"] }).map((w) => w.name), ["15/9-14"]);
  assert.deepEqual(F.filterWells(wells, { hasTarget: "DTS", hasTargetOn: true }).map((w) => w.name).sort(), ["15/9-19A", "16/5-3"]);
  assert.deepEqual(F.filterWells(wells, { hasTarget: "DTS", hasTargetOn: false }).length, 3);
  assert.deepEqual(F.filterWells(wells, { hasLog: "RHOB" }).map((w) => w.name).sort(), ["15/9-14", "16/5-3"]);
});

test("datasetChoices lists All-able ids from dataset_key", () => {
  const ids = F.datasetChoices(wells).map((d) => d.id).sort();
  assert.deepEqual(ids, ["force2020", "volve"]);
});

test("has-target keeps wells whose curve completeness is unknown", () => {
  const bare = [{ name: "X", dataset: "force2020", dataset_key: "force2020", split: "train" }];
  assert.equal(F.filterWells(bare, { hasTarget: "DTS", hasTargetOn: true }).length, 1);
});
