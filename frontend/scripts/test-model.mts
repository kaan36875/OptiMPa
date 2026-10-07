/**
 * Checks the browser model (app/lib/model.ts) against the Python model.
 * Test vectors come from ml_pipeline/export_web_model.py.
 *
 *   npm run test:model
 */
import { readFileSync } from "node:fs";
import { explain, expectedValue, predict, type Mix, type ModelData } from "../app/lib/model.ts";

const read = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const model: ModelData = read("../app/lib/model.json");
const vectors: { expected_value: number; cases: { input: Mix; prediction: number; shap: number[] }[] } =
  read("./model-test-vectors.json");

const TOL_PRED = 1e-3; // MPa — Python predicts in float32
const TOL_SHAP = 1e-3;

let maxPred = 0;
let maxShap = 0;
let maxSum = 0;
let failures = 0;

for (const [i, c] of vectors.cases.entries()) {
  const p = predict(model, c.input);
  const e = explain(model, c.input);
  const dPred = Math.abs(p - c.prediction);
  const dShap = Math.max(...model.features.map((f, j) => Math.abs(e.shap[f] - c.shap[j])));
  const dSum = Math.abs(e.baseValue + Object.values(e.shap).reduce((a, b) => a + b, 0) - e.prediction);
  maxPred = Math.max(maxPred, dPred);
  maxShap = Math.max(maxShap, dShap);
  maxSum = Math.max(maxSum, dSum);
  if (dPred > TOL_PRED || dShap > TOL_SHAP) {
    failures++;
    if (failures <= 5) console.log(`case ${i}: |dPred|=${dPred.toExponential(2)} |dShap|=${dShap.toExponential(2)}`, c.input);
  }
}

const dBase = Math.abs(expectedValue(model) - vectors.expected_value);
console.log(`cases: ${vectors.cases.length}`);
console.log(`max |prediction - python|     : ${maxPred.toExponential(2)} MPa`);
console.log(`max |SHAP - python|           : ${maxShap.toExponential(2)}`);
console.log(`|base value - python|         : ${dBase.toExponential(2)}`);
console.log(`max |base + sum(SHAP) - pred| : ${maxSum.toExponential(2)}`);

if (failures || dBase > TOL_SHAP) {
  console.error(`FAILED (${failures} cases outside tolerance)`);
  process.exit(1);
}
console.log("OK");
