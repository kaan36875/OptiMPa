/**
 * OptiMPa model in the browser.
 *
 * The XGBoost trees are exported by ml_pipeline/export_web_model.py. This file
 * evaluates them (prediction) and computes exact SHAP values with the same
 * path-dependent TreeSHAP algorithm the Python `shap` package uses, so the web
 * page gives the same numbers as the FastAPI service. `npm run test:model`
 * checks this against Python outputs.
 */

export const RAW_FEATURES = [
  "cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age",
] as const;

export type RawFeature = (typeof RAW_FEATURES)[number];
export type Mix = Record<RawFeature, number>;

interface Tree {
  l: number[]; // left child ("x < threshold"), -1 for leaves
  r: number[]; // right child
  f: number[]; // feature index
  v: number[]; // threshold, or leaf value for leaves
  c: number[]; // cover (sum of hessians) — used by TreeSHAP
}

export interface ModelData {
  features: string[];
  base_score: number;
  trees: Tree[];
  data_ranges: Record<string, [number, number]>;
  metrics: {
    xgboost: Record<string, number>;
    rf_baseline: Record<string, number>;
    dataset: { rows: number; mix_designs: number };
    split: { train_samples: number; test_samples: number; cv_folds: number };
    best_params: Record<string, number>;
  };
}

/** The 14 model inputs, in training order (must match engineer_features in Python). */
export function featureVector(mix: Mix): number[] {
  const binder = mix.cement + mix.slag + mix.fly_ash;
  return [
    mix.cement, mix.slag, mix.fly_ash, mix.water, mix.superplasticizer,
    mix.coarse_agg, mix.fine_agg, mix.age,
    mix.water / (mix.cement + 1e-6),
    binder,
    mix.water / (binder + 1e-6),
    mix.fine_agg / (mix.coarse_agg + 1e-6),
    mix.slag / (mix.cement + 1e-6),
    mix.fly_ash / (mix.cement + 1e-6),
  ];
}

// XGBoost stores thresholds and compares features as float32. The JSON holds
// decimal approximations, so both sides are rounded to float32 before comparing
// (otherwise a feature exactly equal to a threshold can take the wrong branch).
const toF32 = (x: number[]) => x.map(Math.fround);

const f32Trees = new WeakMap<ModelData, Tree[]>();

function trees(model: ModelData): Tree[] {
  let t = f32Trees.get(model);
  if (!t) {
    t = model.trees.map((tree) => ({ ...tree, v: Array.from(new Float32Array(tree.v)) }));
    f32Trees.set(model, t);
  }
  return t;
}

function leafIndex(tree: Tree, x: number[]): number {
  let node = 0;
  while (tree.l[node] !== -1) {
    node = x[tree.f[node]] < tree.v[node] ? tree.l[node] : tree.r[node];
  }
  return node;
}

export function predict(model: ModelData, mix: Mix): number {
  const x = toF32(featureVector(mix));
  let sum = model.base_score;
  for (const tree of trees(model)) sum += tree.v[leafIndex(tree, x)];
  return sum;
}

// ─── TreeSHAP (Lundberg et al. 2020, Algorithm 2) ────────────────────────────

interface PathElement { d: number; z: number; o: number; w: number }

function extendPath(m: PathElement[], pz: number, po: number, pi: number) {
  const l = m.length;
  m.push({ d: pi, z: pz, o: po, w: l === 0 ? 1 : 0 });
  for (let i = l - 1; i >= 0; i--) {
    m[i + 1].w += (po * m[i].w * (i + 1)) / (l + 1);
    m[i].w = (pz * m[i].w * (l - i)) / (l + 1);
  }
}

function unwindPath(m: PathElement[], index: number) {
  const l = m.length - 1;
  const { o, z } = m[index];
  let next = m[l].w;
  for (let j = l - 1; j >= 0; j--) {
    if (o !== 0) {
      const tmp = m[j].w;
      m[j].w = (next * (l + 1)) / ((j + 1) * o);
      next = tmp - (m[j].w * z * (l - j)) / (l + 1);
    } else {
      m[j].w = (m[j].w * (l + 1)) / (z * (l - j));
    }
  }
  for (let j = index; j < l; j++) {
    m[j].d = m[j + 1].d;
    m[j].z = m[j + 1].z;
    m[j].o = m[j + 1].o;
  }
  m.pop();
}

function unwoundPathSum(m: PathElement[], index: number): number {
  const l = m.length - 1;
  const { o, z } = m[index];
  let total = 0;
  if (o !== 0) {
    let next = m[l].w;
    for (let j = l - 1; j >= 0; j--) {
      const tmp = next / ((j + 1) * o);
      total += tmp;
      next = m[j].w - tmp * z * (l - j);
    }
  } else {
    for (let j = l - 1; j >= 0; j--) total += m[j].w / ((l - j) * z);
  }
  return total * (l + 1);
}

function treeShap(tree: Tree, x: number[], phi: number[]) {
  const recurse = (node: number, parent: PathElement[], pz: number, po: number, pi: number) => {
    const m = parent.map((e) => ({ ...e }));
    extendPath(m, pz, po, pi);

    if (tree.l[node] === -1) {
      for (let i = 1; i < m.length; i++) {
        phi[m[i].d] += unwoundPathSum(m, i) * (m[i].o - m[i].z) * tree.v[node];
      }
      return;
    }

    const feature = tree.f[node];
    const goesLeft = x[feature] < tree.v[node];
    const hot = goesLeft ? tree.l[node] : tree.r[node];
    const cold = goesLeft ? tree.r[node] : tree.l[node];
    let iz = 1;
    let io = 1;
    const k = m.findIndex((e) => e.d === feature);
    if (k !== -1) {
      iz = m[k].z;
      io = m[k].o;
      unwindPath(m, k);
    }
    recurse(hot, m, (tree.c[hot] / tree.c[node]) * iz, io, feature);
    recurse(cold, m, (tree.c[cold] / tree.c[node]) * iz, 0, feature);
  };
  recurse(0, [], 1, 1, -1);
}

const expectedCache = new WeakMap<ModelData, number>();

/** SHAP base value: base_score + cover-weighted mean leaf value of every tree. */
export function expectedValue(model: ModelData): number {
  const cached = expectedCache.get(model);
  if (cached !== undefined) return cached;
  const nodeMean = (t: Tree, n: number): number => {
    if (t.l[n] === -1) return t.v[n];
    const [a, b] = [t.l[n], t.r[n]];
    return (t.c[a] * nodeMean(t, a) + t.c[b] * nodeMean(t, b)) / (t.c[a] + t.c[b]);
  };
  const value = trees(model).reduce((sum, t) => sum + nodeMean(t, 0), model.base_score);
  expectedCache.set(model, value);
  return value;
}

export interface Explanation {
  prediction: number;
  baseValue: number;
  shap: Record<string, number>; // per model feature, sums to prediction − baseValue
}

export function explain(model: ModelData, mix: Mix): Explanation {
  const x = toF32(featureVector(mix));
  const phi = new Array<number>(model.features.length).fill(0);
  let prediction = model.base_score;
  for (const tree of trees(model)) {
    prediction += tree.v[leafIndex(tree, x)];
    treeShap(tree, x, phi);
  }
  return {
    prediction,
    baseValue: expectedValue(model),
    shap: Object.fromEntries(model.features.map((f, i) => [f, phi[i]])),
  };
}

// ─── EN 206 class (same rule as api/main.py) ─────────────────────────────────

const EN206: [number, string][] = [
  [8, "C8/10"], [12, "C12/15"], [16, "C16/20"], [20, "C20/25"], [25, "C25/30"],
  [30, "C30/37"], [35, "C35/45"], [40, "C40/50"], [45, "C45/55"], [50, "C50/60"],
  [55, "C55/67"], [60, "C60/75"], [70, "C70/85"], [80, "C80/95"], [90, "C90/105"],
  [100, "C100/115"],
];

/** Highest EN 206 class met, with fck = predicted mean − 8 MPa (EN 1992-1-1). */
export function en206Grade(mpa: number): string {
  const fck = mpa - 8;
  let grade = "< C8/10";
  for (const [threshold, label] of EN206) if (fck >= threshold) grade = label;
  return grade;
}

/** Inputs outside the range of the training data (the model is extrapolating there). */
export function outOfRange(model: ModelData, mix: Mix): RawFeature[] {
  return RAW_FEATURES.filter((k) => {
    const [lo, hi] = model.data_ranges[k];
    return mix[k] < lo || mix[k] > hi;
  });
}
