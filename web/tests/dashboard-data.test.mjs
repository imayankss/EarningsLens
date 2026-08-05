import assert from "node:assert/strict";
import test from "node:test";

import { validateDashboardData, validateFile } from "../scripts/validate-dashboard-data.mjs";

test("checked dashboard artifact satisfies the deployment contract", async () => {
  const data = await validateFile("public/data/dashboard.json");
  assert.equal(data.dataset.kind, "demonstration");
  assert.equal(data.dataset.observationCount, 1);
  assert.equal(data.analyses[0].transcript.id, "AAPL_20201029");
  assert.equal(data.modeling.modelsTrained, 0);
});

test("validator rejects fabricated or invalid probability values", () => {
  const errors = validateDashboardData({
    schemaVersion: 1,
    dataset: {},
    analyses: [{ transcript: { id: "example" }, finbert: { positiveProbability: 1.2 } }],
  });
  assert.ok(errors.some((message) => message.includes("positiveProbability")));
});
