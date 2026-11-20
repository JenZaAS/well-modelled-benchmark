// Optional live-viewer smoke test: WMB_VIEWER_URL, playwright-core, and a Chromium executable.
// Score and well endpoints are intercepted so this never depends on local well data.
const test = require("node:test");
const assert = require("node:assert/strict");

test("benchmark highlights, family selection, and model detail navigation", { skip: !process.env.WMB_VIEWER_URL }, async (t) => {
  const browser = await require("playwright-core").chromium.launch({ executablePath: process.env.WMB_CHROMIUM });
  t.after(() => browser.close());
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const models = ["template_linear", "purple_tabum_v11", "purple_kumo_tabular", "red_chronos2"];
  const rows = models.map((contender, i) => ({ contender, dataset: "fixture", target: "RHOB", scenario: "gap_50",
    MAE: i + 1, interval_width_90: i + 1, cover_90: 0.9, predict_s_per_well: 1 + i }));
  await page.route("**/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = path === "/api/benchmark/runs" ? { rows } : path === "/api/map" ?
      { live: true, datasets: [], wells: [], unplaced: [], lithology: {}, coast: { type: "FeatureCollection", features: [] } } : { wells: [] };
    return route.fulfill({ json });
  });
  await page.goto(process.env.WMB_VIEWER_URL + "/#benchmark/summary-all");
  const family = page.getByLabel("Model type");
  assert.equal(await family.inputValue(), "");
  await family.selectOption("Tabular foundation");
  const same = page.locator('#bm-view [data-model-id="purple_tabum_v11"]');
  const name = same.locator("a.bm-model-link").first();
  await name.hover();
  const count = await same.count();
  assert.ok(count >= 7, "accuracy, calibration, sharpness, speed label and needle, scatter and legend");
  assert.equal(await page.locator("#bm-view .bm-picked").count(), count);
  assert.ok(await page.locator("#bm-view .bm-subdued").count() > 0);
  await family.hover();
  assert.equal(await page.locator("#bm-view .bm-picked").count(), 0);
  await name.focus();
  assert.equal(await page.locator("#bm-view .bm-picked").count(), count);
  await name.press("Enter");
  await page.getByRole("heading", { name: "tabUM v1.1" }).waitFor();
  assert.equal(await page.locator("#bm-detail li").count(), 3);
  const diagram = page.getByRole("img", { name: "tabUM v1.1: prediction workflow" });
  await diagram.waitFor();
  await page.setViewportSize({ width: 375, height: 812 });
  const bounds = await diagram.boundingBox();
  assert.ok(bounds && bounds.width > 0 && bounds.x >= 0 && bounds.x + bounds.width <= 375);
  await page.setViewportSize({ width: 1280, height: 800 });
  assert.ok(page.url().endsWith("#benchmark/model/purple_tabum_v11"));
  await page.goBack();
  await family.waitFor();
  assert.equal(await family.inputValue(), "Tabular foundation");
  assert.ok(page.url().endsWith("#benchmark/summary-all"));
  await page.getByRole("button", { name: "Per property" }).click();
  await page.getByRole("link", { name: "Kumo-Tabular", exact: true }).click();
  await page.reload();
  await page.getByRole("heading", { name: "Kumo-Tabular" }).waitFor();
  await page.getByRole("link", { name: "Back to benchmark" }).click();
  await family.waitFor();
  assert.deepEqual(errors, []);
});