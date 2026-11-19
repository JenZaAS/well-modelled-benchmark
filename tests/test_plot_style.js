"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const S = require("../viewer/plot_style.js");

test("Standard scales follow DigTech conventions", () => {
  assert.deepEqual(S.standardScale("GR"), [0, 150]);
  assert.deepEqual(S.standardScale("CALI"), [6, 16]);
  assert.deepEqual(S.standardScale("RDEP"), [0.2, 2000]);
  assert.deepEqual(S.standardScale("RMED"), [0.2, 2000]);
  assert.deepEqual(S.standardScale("RSHA"), [0.2, 2000]);
  assert.deepEqual(S.standardScale("RHOB"), [1.95, 2.95]);
  assert.deepEqual(S.standardScale("NPHI"), [0.45, -0.15]);
  assert.deepEqual(S.standardScale("DTC"), [140, 40]);
  assert.deepEqual(S.standardScale("DTS"), [240, 40]);
  assert.deepEqual(S.standardScale("PEF"), [0, 10]);
  assert.deepEqual(S.standardScale("PHIE"), [0, 0.5]);
  assert.deepEqual(S.standardScale("PHIT"), [0, 0.5]);
  assert.deepEqual(S.standardScale("VSH"), [0, 1]);
  assert.deepEqual(S.standardScale("SW"), [0, 1]);
});

test("aliases map Volve mnemonics onto display curves, not NPHI", () => {
  assert.equal(S.canonName("LFP_PHIE"), "PHIE");
  assert.equal(S.canonName("PHIF"), "PHIE");
  assert.equal(S.canonName("lfp_vsh"), "VSH");
  assert.equal(S.canonName("VSHALE"), "VSH");
  assert.equal(S.canonName("SWT"), "SW");
  assert.equal(S.canonName("NPHI"), "NPHI");
  assert.equal(S.canonName("PHI"), "PHIE");
  assert.equal(S.pickPresent("PHIE", ["NPHI", "PHIF", "GR"]), "PHIF");
  assert.equal(S.pickPresent("PHIE", ["NPHI", "GR"]), null);
});

test("prediction targets stay RHOB/DTC/DTS and exclude display-only curves", () => {
  assert.deepEqual(S.TARGETS, ["RHOB", "DTC", "DTS"]);
  assert.ok(S.CURVES.PHIE.displayOnly && S.CURVES.VSH.displayOnly && S.CURVES.SW.displayOnly);
  assert.ok(!S.TARGETS.includes("PHIE") && !S.TARGETS.includes("VSH") && !S.TARGETS.includes("SW"));
});

test("mixed Auto default is CALI (and PEF/BS), not GR or RHOB", () => {
  assert.equal(S.useAutoScale("CALI", null), true);
  assert.equal(S.useAutoScale("PEF", null), true);
  assert.equal(S.useAutoScale("GR", null), false);
  assert.equal(S.useAutoScale("RHOB", null), false);
  assert.equal(S.useAutoScale("GR", "auto"), true);
  assert.equal(S.useAutoScale("CALI", "standard"), false);
});

test("depth grid is 25 m up to 600 m, else 100 m", () => {
  assert.equal(S.depthGridStep(100), 25);
  assert.equal(S.depthGridStep(600), 25);
  assert.equal(S.depthGridStep(600.1), 100);
  assert.deepEqual(S.depthTicks(1490, 1610), [1500, 1525, 1550, 1575, 1600]);
  assert.deepEqual(S.depthTicks(1000, 2000), [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]);
});

test("tracks skip missing curves; optional petro tracks appear only when present", () => {
  const core = S.tracksFromNames(["GR", "RDEP", "RHOB", "NPHI", "DTC"]);
  assert.deepEqual(core.map((t) => t.id), ["gr", "res", "dn", "son"]);
  assert.deepEqual(core.find((t) => t.id === "gr").curves.map((c) => c.k), ["GR"]);
  assert.deepEqual(core.find((t) => t.id === "res").curves.map((c) => c.k), ["RDEP"]);
  const emptyCore = S.tracksFromNames(["GR"], { includeEmptyCore: true });
  assert.ok(emptyCore.find((t) => t.id === "pef").noData);
  const petro = S.tracksFromNames(["GR", "PHIF", "PHIT", "VSH", "SW"]);
  assert.ok(petro.find((t) => t.id === "phi"));
  assert.deepEqual(petro.find((t) => t.id === "phi").curves.map((c) => c.canon), ["PHIT", "PHIE"]);
  assert.equal(petro.find((t) => t.id === "phi").curves.find((c) => c.canon === "PHIE").k, "PHIF");
  assert.deepEqual(petro.find((t) => t.id === "vsh").curves.map((c) => c.canon), ["VSH", "SW"]);
});

test("density-neutron crossover: yellow when density porosity > neutron, grey otherwise", () => {
  // RHOB left (light) of NPHI → yellow; NPHI left (high neutron) of RHOB → grey.
  assert.equal(S.densityNeutronFill(2.05, 0.10), "yellow");
  assert.equal(S.densityNeutronFill(2.70, 0.40), "grey");
  assert.equal(S.densityNeutronFill(null, 0.20), null);
});

test("fills are not bridged across gaps or depth jumps", () => {
  const depth = [100, 100.5, 101, 105, 105.5];
  const v = [1, 2, null, 3, 4];
  assert.equal(S.fillPairOk(depth, 0, 0.5, [v], [false]), true);
  assert.equal(S.fillPairOk(depth, 1, 0.5, [v], [false]), false);
  assert.equal(S.fillPairOk(depth, 2, 0.5, [[1, 2, 3, 4, 5]], [false]), false);
  assert.equal(S.fillPairOk(depth, 3, 0.5, [v], [false]), true);
  assert.equal(S.fillPairOk(depth, 0, 0.5, [[-1, 2]], [true]), false);
});

test("formation fill source does not fall back to GROUP", () => {
  const depth = [100, 110, 120];
  const formation = [null, "TY R", "TY R"];
  const group = ["NORDLAND GP", "HORDALAND GP", "HORDALAND GP"];
  const fill = S.formationIntervals(depth, formation, group, "formation");
  assert.equal(fill.length, 1);
  assert.equal(fill[0].name, "TY R");
  assert.equal(fill[0].top, 110);
  const byGroup = S.formationIntervals(depth, formation, group, "group");
  assert.equal(byGroup[0].name, "NORDLAND GP");
  assert.equal(byGroup[1].name, "HORDALAND GP");
  assert.deepEqual(S.formationIntervals(depth, formation, group, "off"), []);
});

test("formation intervals prefer FORMATION over GROUP and colour the strip", () => {
  const depth = [100, 110, 120, 130, 140];
  const formation = [null, "HORDALAND GP", "HORDALAND GP", "TY R", "TY R"];
  const group = ["NORDLAND GP", "NORDLAND GP", "HORDALAND GP", "HORDALAND GP", "ROGALAND GP"];
  const iv = S.formationIntervals(depth, formation, group);
  assert.equal(iv.length, 3);
  assert.equal(iv[0].name, "NORDLAND GP");
  assert.equal(iv[0].top, 100);
  assert.equal(iv[1].name, "HORDALAND GP");
  assert.equal(iv[1].top, 110);
  assert.equal(iv[2].name, "TY R");
  assert.equal(iv[2].top, 130);
  assert.equal(iv[2].base, 140);
  assert.equal(S.formationColor("HORDALAND GP"), S.formationColor("HORDALAND GP"));
  assert.notEqual(S.formationColor("HORDALAND GP"), S.formationColor("TY R"));
});

test("curve colours: deep red, medium blue, shallow green; PHIT black, PHIE blue, SW dodgerblue", () => {
  assert.equal(S.colorOf("RDEP"), "#c0392b");
  assert.equal(S.colorOf("RMED"), "#1d4ed8");
  assert.equal(S.colorOf("RSHA"), "#2ca02c");
  assert.equal(S.colorOf("PHIT"), "#111");
  assert.equal(S.colorOf("PHIE"), "#1d4ed8");
  assert.equal(S.colorOf("SW"), "#1e90ff");
  assert.equal(S.colorOf("CALI"), "#111");
});
