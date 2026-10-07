import type { Mix } from "./model";

export const DEFAULT_MIX: Mix = {
  cement: 350, slag: 0, fly_ash: 0, water: 175, superplasticizer: 6,
  coarse_agg: 1040, fine_agg: 755, age: 28,
};

/** Example mixes per class — same as bim/mixes.example.json. */
export const CLASS_MIXES: Record<string, Mix> = {
  "C25/30": { cement: 260, slag: 0, fly_ash: 80, water: 175, superplasticizer: 4, coarse_agg: 1040, fine_agg: 790, age: 28 },
  "C30/37": { cement: 330, slag: 0, fly_ash: 0, water: 165, superplasticizer: 6, coarse_agg: 1040, fine_agg: 780, age: 28 },
  "C35/45": { cement: 380, slag: 0, fly_ash: 0, water: 155, superplasticizer: 8, coarse_agg: 1030, fine_agg: 760, age: 28 },
};

export const SLIDERS = [
  { key: "cement",           label: "Cement",             unit: "kg/m³", min: 100, max: 700,  step: 5,   note: "Primary binder — drives early strength" },
  { key: "slag",             label: "Blast Furnace Slag", unit: "kg/m³", min: 0,   max: 360,  step: 5,   note: "Latent hydraulic binder — boosts long-term strength" },
  { key: "fly_ash",          label: "Fly Ash",            unit: "kg/m³", min: 0,   max: 200,  step: 5,   note: "Pozzolanic filler — reduces heat of hydration" },
  { key: "water",            label: "Water",              unit: "kg/m³", min: 120, max: 250,  step: 1,   note: "Lower w/c ratio → denser paste → higher strength" },
  { key: "superplasticizer", label: "Superplasticizer",   unit: "kg/m³", min: 0,   max: 32,   step: 0.5, note: "Maintains workability at low water content" },
  { key: "coarse_agg",       label: "Coarse Aggregate",   unit: "kg/m³", min: 800, max: 1150, step: 5,   note: "Structural skeleton — crushed stone or gravel" },
  { key: "fine_agg",         label: "Fine Aggregate",     unit: "kg/m³", min: 550, max: 1000, step: 5,   note: "Sand — fills voids between coarse particles" },
  { key: "age",              label: "Curing Age",         unit: "days",  min: 1,   max: 365,  step: 1,   note: "28 days = standard reference per EN 206" },
] as const satisfies readonly { key: keyof Mix; label: string; unit: string; min: number; max: number; step: number; note: string }[];

export const FEATURE_LABELS: Record<string, string> = {
  cement: "Cement",
  slag: "Blast Furnace Slag",
  fly_ash: "Fly Ash",
  water: "Water",
  superplasticizer: "Superplasticizer",
  coarse_agg: "Coarse Aggregate",
  fine_agg: "Fine Aggregate",
  age: "Curing Age",
  wc_ratio: "Water/Cement Ratio",
  binder_total: "Total Binder",
  wb_ratio: "Water/Binder Ratio",
  fine_coarse_ratio: "Fine/Coarse Agg. Ratio",
  slag_cement_ratio: "Slag/Cement Ratio",
  fly_ash_cement_ratio: "Fly Ash/Cement Ratio",
};

// The mix set on the predictor page is remembered per browser, so the BIM page
// can use it. Storage may be unavailable (private mode) — then defaults apply.
const STORAGE_KEY = "optimpa.mix";

export function loadMix(): Mix {
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "null");
    if (saved && typeof saved === "object") return { ...DEFAULT_MIX, ...saved };
  } catch {
    /* fall through */
  }
  return DEFAULT_MIX;
}

export function saveMix(mix: Mix) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(mix));
  } catch {
    /* not critical */
  }
}
