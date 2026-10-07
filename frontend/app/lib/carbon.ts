/**
 * Embodied carbon of a concrete mix, A1–A3 (cradle to gate).
 * Same factors and method as bim/carbon.py — keep the two in sync.
 *
 * Approximate generic values, mostly ICE database v3.0 (Circular Ecology,
 * 2019); superplasticizer from the EFCA admixture EPD. Good for comparing
 * mixes; for reporting, use EPD values of the actual products.
 */

import type { Mix } from "./model";

// kg CO2e per kg of material
export const EMISSION_FACTORS = {
  cement: 0.912,
  slag: 0.083,
  fly_ash: 0.004,
  water: 0.000344,
  superplasticizer: 1.88,
  coarse_agg: 0.0075,
  fine_agg: 0.0075,
} as const;

export type CarbonMaterial = keyof typeof EMISSION_FACTORS;
export const CARBON_MATERIALS = Object.keys(EMISSION_FACTORS) as CarbonMaterial[];

export interface MixCarbon {
  total: number; // kg CO2e per m³
  byMaterial: Record<CarbonMaterial, number>;
}

export function mixCarbon(mix: Mix): MixCarbon {
  const byMaterial = Object.fromEntries(
    CARBON_MATERIALS.map((k) => [k, mix[k] * EMISSION_FACTORS[k]]),
  ) as Record<CarbonMaterial, number>;
  const total = CARBON_MATERIALS.reduce((s, k) => s + byMaterial[k], 0);
  return { total, byMaterial };
}

/**
 * Reference mix for comparison, like the EcoConcrete "base case": the same mix
 * with all binder as Portland cement (slag and fly ash replaced 1:1 by mass),
 * so binder content and w/b ratio stay the same.
 */
export function referenceMix(mix: Mix): Mix {
  return { ...mix, cement: mix.cement + mix.slag + mix.fly_ash, slag: 0, fly_ash: 0 };
}
