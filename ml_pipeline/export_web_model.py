"""
OptiMPa — export the trained model for the web app

The website runs the model in the browser (no Python server), so the XGBoost
trees are written to a compact JSON file that frontend/app/lib/model.ts reads.

Run after train_xgboost.py:  python ml_pipeline/export_web_model.py
Writes:
  frontend/app/lib/model.json            trees + metadata (metrics, data ranges)
  frontend/scripts/model-test-vectors.json  Python predictions and SHAP values,
                                            used by `npm run test:model`
  frontend/public/reports/*.png          plots for the model page
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from concrete_data import load_data
from train_xgboost import FEATURES, RAW_FEATURES, engineer_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = ROOT / "api" / "model.pkl"
FRONTEND = ROOT / "frontend"
METRICS = HERE / "example_reports" / "metrics.json"


def f32(x: float) -> float:
    """Shortest decimal that maps back to the same float32 (what XGBoost stores)."""
    return float(f"{np.float32(x):.9g}")


def export_trees(booster) -> tuple[float, list]:
    raw = json.loads(booster.save_raw("json").decode())["learner"]
    base_score = float(raw["learner_model_param"]["base_score"])
    trees = []
    for t in raw["gradient_booster"]["model"]["trees"]:
        # Leaves have left_children == -1; their split_conditions hold the leaf value
        trees.append({
            "l": t["left_children"],
            "r": t["right_children"],
            "f": t["split_indices"],
            "v": [f32(v) for v in t["split_conditions"]],
            "c": [f32(c) for c in t["sum_hessian"]],
        })
    return base_score, trees


def test_vectors(model, n_random: int = 300, seed: int = 0) -> dict:
    """Random mixes across the input ranges + real dataset rows, with Python outputs."""
    rng = np.random.default_rng(seed)
    data = load_data()
    lo = {"cement": 100, "slag": 0, "fly_ash": 0, "water": 120, "superplasticizer": 0,
          "coarse_agg": 800, "fine_agg": 550, "age": 1}
    hi = {"cement": 700, "slag": 360, "fly_ash": 200, "water": 250, "superplasticizer": 32,
          "coarse_agg": 1150, "fine_agg": 1000, "age": 365}
    rows = [{k: round(float(rng.uniform(lo[k], hi[k])), 1) for k in RAW_FEATURES} for _ in range(n_random)]
    for r in rows:
        r["age"] = int(round(r["age"]))
    rows += data[RAW_FEATURES].sample(100, random_state=seed).to_dict("records")

    X = engineer_features(pd.DataFrame(rows))[FEATURES]
    explanation = shap.TreeExplainer(model)(X)
    return {
        "expected_value": float(explanation.base_values[0]),
        "cases": [
            {"input": {k: float(r[k]) for k in RAW_FEATURES},
             "prediction": float(p),
             "shap": [float(s) for s in sv]}
            for r, p, sv in zip(rows, model.predict(X), explanation.values)
        ],
    }


def main():
    model = joblib.load(MODEL_PATH)
    booster = model.get_booster()
    assert booster.feature_names == FEATURES, booster.feature_names
    base_score, trees = export_trees(booster)

    data = load_data()
    meta = {
        "features": FEATURES,
        "base_score": base_score,
        "metrics": json.loads(METRICS.read_text(encoding="utf-8")),
        "data_ranges": {k: [float(data[k].min()), float(data[k].max())] for k in RAW_FEATURES + ["strength"]},
        "trees": trees,
    }
    out = FRONTEND / "app" / "lib" / "model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
    print(f"[OK] {len(trees)} trees -> {out} ({out.stat().st_size / 1024:.0f} KB)")

    vectors = FRONTEND / "scripts" / "model-test-vectors.json"
    vectors.parent.mkdir(parents=True, exist_ok=True)
    vectors.write_text(json.dumps(test_vectors(model)), encoding="utf-8")
    print(f"[OK] test vectors -> {vectors}")

    reports = FRONTEND / "public" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for name in ("model_evaluation.png", "shap_summary.png"):
        shutil.copy(HERE / "example_reports" / name, reports / name)
    print(f"[OK] plots -> {reports}")


if __name__ == "__main__":
    main()
