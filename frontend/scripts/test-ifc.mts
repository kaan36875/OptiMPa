/**
 * Checks the browser IFC reader (app/lib/ifc.ts) on bim/examples/sample_building.ifc.
 * Expected values are the ones listed in bim/examples/make_sample_ifc.py.
 *
 *   npm run test:ifc [-- other.ifc ...]   (extra files are only printed)
 */
import { readFileSync } from "node:fs";
import { IfcAPI } from "web-ifc";
import { analyseIfc } from "../app/lib/ifc.ts";

const api = new IfcAPI();
await api.Init(undefined, true);

const EXPECTED: Record<string, [number, string | null]> = {
  "Ground slab": [20.0, "C30/37"],
  "Column 1": [0.48, "C35/45"],
  "Column 2": [0.48, "C35/45"],
  "Column 3": [0.48, "C35/45"],
  "Column 4": [0.48, "C35/45"],
  "External wall": [6.0, "C30/37"],
  "Strip footing": [7.2, "C25/30"],
  "Internal wall": [2.58, null],
};

function run(path: string) {
  const result = analyseIfc(api, new Uint8Array(readFileSync(path)));
  console.log(`\n${path} (${result.schema})`);
  for (const e of result.elements) {
    console.log(`  ${e.ifcClass.padEnd(12)} ${e.name.padEnd(16)} ${e.volume.toFixed(4).padStart(9)} m3  ${String(e.strengthClass).padEnd(7)} ${e.volumeSource} ${e.note}`);
  }
  for (const r of result.review) console.log(`  review: ${r.ifcClass} '${r.name}': ${r.issue}`);
  return result;
}

const sample = new URL("../../bim/examples/sample_building.ifc", import.meta.url);
const result = run(decodeURIComponent(sample.pathname).replace(/^\/([A-Za-z]:)/, "$1"));

const errors: string[] = [];
const seen = new Set<string>();
for (const e of result.elements) {
  seen.add(e.name);
  const exp = EXPECTED[e.name];
  if (!exp) errors.push(`unexpected element ${e.name}`);
  else {
    if (Math.abs(e.volume - exp[0]) > 1e-3) errors.push(`${e.name}: volume ${e.volume} != ${exp[0]}`);
    if (e.strengthClass !== exp[1]) errors.push(`${e.name}: class ${e.strengthClass} != ${exp[1]}`);
  }
}
for (const name of Object.keys(EXPECTED)) if (!seen.has(name)) errors.push(`missing ${name}`);
if (result.review.length !== 1 || result.review[0].name !== "Stair (placeholder)") errors.push("review list should hold only the stair");

for (const extra of process.argv.slice(2)) run(extra);

if (errors.length) {
  console.error("\nFAILED\n  " + errors.join("\n  "));
  process.exit(1);
}
console.log("\nOK");
