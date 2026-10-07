"""
OptiMPa — Concrete Compressive Strength Predictor
Phase 1 Upgrade: XGBoost + Optuna + SHAP Pipeline

Dataset: UCI Concrete Compressive Strength (ID: 165)

Run from anywhere:  python ml_pipeline/train_xgboost.py
Outputs:
  - api/model.pkl                      (production model, trained on all data)
  - ml_pipeline/reports/*.png          (evaluation + SHAP plots)
  - ml_pipeline/reports/metrics.json   (CV, holdout and baseline metrics)
"""

import json
import os
from pathlib import Path

import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed to save figures
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from concrete_data import load_data

# Suppress Optuna logs to keep stdout clean unless warning/error
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS = 50
N_SPLITS = 5

HERE = Path(__file__).resolve().parent
REPORTS_DIR = HERE / "reports"
MODEL_PATH = HERE.parent / "api" / "model.pkl"

RAW_FEATURES = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
FEATURES = RAW_FEATURES + [
    "wc_ratio", "binder_total", "wb_ratio", "fine_coarse_ratio", "slag_cement_ratio", "fly_ash_cement_ratio"
]
TARGET = "strength"

# ─────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Incorporate civil engineering domain knowledge.
    - Water/Cement ratio (Abrams' Law)
    - Water/Binder ratio (Binder = Cement + Slag + Fly Ash)
    - Fine/Coarse Aggregate ratio
    - Slag & Fly Ash relative binder proportions

    NOTE: api/main.py (build_feature_vector) must apply the exact same formulas.
    """
    df = df.copy()

    # Su/Cimento Orani (Abrams Yasasi - Mukavemetin birincil surucusu)
    df["wc_ratio"] = df["water"] / (df["cement"] + 1e-6)

    # Toplam Baglayici Miktari (Cimento + Curuf + Ucucu Kul)
    df["binder_total"] = df["cement"] + df["slag"] + df["fly_ash"]

    # Su/Baglayici Orani
    df["wb_ratio"] = df["water"] / (df["binder_total"] + 1e-6)

    # Ince/Kaba Agrega Orani
    df["fine_coarse_ratio"] = df["fine_agg"] / (df["coarse_agg"] + 1e-6)

    # Curuf ve Ucucu Kulun Cimentoya Oranlari
    df["slag_cement_ratio"] = df["slag"] / (df["cement"] + 1e-6)
    df["fly_ash_cement_ratio"] = df["fly_ash"] / (df["cement"] + 1e-6)

    return df

# ─────────────────────────────────────────────
# 2. DATA SPLITTING & GROUPING (Leak-free Validation)
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """
    Group by concrete mix design (excluding curing age & strength) to prevent
    data leakage where observations of the same mix layout appear in both sets.
    """
    df = df.copy()

    # Group unique mix proportions
    mix_cols = RAW_FEATURES[:-1]  # everything except age
    df["mix_id"] = df.groupby(mix_cols).ngroup()

    # Apply feature engineering
    df = engineer_features(df)

    # Split unique mix designs into 80% train and 20% test
    unique_mixes = df["mix_id"].unique()
    train_mixes, test_mixes = train_test_split(unique_mixes, test_size=0.20, random_state=SEED)

    train_df = df[df["mix_id"].isin(train_mixes)].reset_index(drop=True)
    test_df = df[df["mix_id"].isin(test_mixes)].reset_index(drop=True)

    print(f"  [->] Total Samples: {len(df)} | Total Mix Designs: {len(unique_mixes)}")
    print(f"  [->] Training Mix Designs (for CV & Tuning): {len(train_mixes)}")
    print(f"  [->] Test Mix Designs (held out completely): {len(test_mixes)}")
    print(f"  [->] Train Samples: {train_df.shape[0]} | Test Samples: {test_df.shape[0]}\n")

    return train_df, test_df

# ─────────────────────────────────────────────
# 3. OPTUNA HYPERPARAMETER TUNING (GroupKFold CV)
# ─────────────────────────────────────────────

def group_cv_scores(make_model, train_df: pd.DataFrame, features: list) -> tuple[float, float]:
    """Mean RMSE and R2 over a GroupKFold split of the training mixes."""
    X, y, groups = train_df[features], train_df[TARGET], train_df["mix_id"]
    rmses, r2s = [], []
    for train_idx, val_idx in GroupKFold(n_splits=N_SPLITS).split(X, y, groups=groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
        r2s.append(r2_score(y.iloc[val_idx], preds))
    return float(np.mean(rmses)), float(np.mean(r2s))


def tune_xgboost(train_df: pd.DataFrame) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna and 5-fold GroupKFold.
    Target metric: Root Mean Squared Error (RMSE).
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        rmse, _ = group_cv_scores(lambda: make_xgb(params), train_df, FEATURES)
        return rmse

    print(f"  [->] Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials, {N_SPLITS}-fold GroupKFold)...")
    # Seeded sampler → the same search path (and model) on every run
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print("  [OK] Optimization finished.")
    print(f"      Best trial CV RMSE: {study.best_value:.4f} MPa")
    print(f"      Best parameters: {study.best_params}\n")

    return study.best_params


def make_xgb(params: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=0)


def make_rf_baseline() -> RandomForestRegressor:
    """Phase-1 baseline: Random Forest on the 8 raw features only."""
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def holdout_scores(model, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list) -> dict:
    """Fit on train mixes, score on the unseen test mixes."""
    model.fit(train_df[features], train_df[TARGET])
    train_preds = model.predict(train_df[features])
    test_preds = model.predict(test_df[features])
    return {
        "train_r2": float(r2_score(train_df[TARGET], train_preds)),
        "train_rmse": float(np.sqrt(mean_squared_error(train_df[TARGET], train_preds))),
        "test_r2": float(r2_score(test_df[TARGET], test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(test_df[TARGET], test_preds))),
        "test_preds": test_preds,
    }


def evaluate_models(best_params: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Compare tuned XGBoost against the RF baseline under the SAME leak-free
    protocol: GroupKFold CV on training mixes + a held-out set of unseen mixes.
    """
    xgb_cv_rmse, xgb_cv_r2 = group_cv_scores(lambda: make_xgb(best_params), train_df, FEATURES)
    rf_cv_rmse, rf_cv_r2 = group_cv_scores(make_rf_baseline, train_df, RAW_FEATURES)
    xgb_hold = holdout_scores(make_xgb(best_params), train_df, test_df, FEATURES)
    rf_hold = holdout_scores(make_rf_baseline(), train_df, test_df, RAW_FEATURES)

    print("=" * 66)
    print("  MODEL EVALUATION RESULTS (LEAK-FREE, GROUPED BY MIX DESIGN)")
    print("=" * 66)
    print(f"  {'':28s}{'CV R2':>9s}{'CV RMSE':>10s}{'Test R2':>9s}{'Test RMSE':>10s}")
    for name, cv_r2, cv_rmse, hold in [
        ("RF baseline (raw)", rf_cv_r2, rf_cv_rmse, rf_hold),
        ("XGBoost (engineered)", xgb_cv_r2, xgb_cv_rmse, xgb_hold),
    ]:
        print(f"  {name:28s}{cv_r2:9.4f}{cv_rmse:10.2f}{hold['test_r2']:9.4f}{hold['test_rmse']:10.2f}")
    print(f"  XGBoost train R2 / RMSE   : {xgb_hold['train_r2']:.4f} / {xgb_hold['train_rmse']:.2f} MPa")
    print("=" * 66 + "\n")

    # Save Predicted vs Actual Plot
    y_test, test_preds = test_df[TARGET], xgb_hold["test_preds"]
    # Scoped style: a global plt.style.use would also darken the SHAP plot,
    # whose feature labels are drawn in dark grey
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(y_test, test_preds, alpha=0.6, color="#4F86C6", edgecolors="white", linewidth=0.4, s=50)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5, label="Perfect Prediction")
        ax.set_xlabel("Actual Strength (MPa)")
        ax.set_ylabel("Predicted Strength (MPa)")
        ax.set_title(
            f"XGBoost Test Set (unseen mixes) | R2 = {xgb_hold['test_r2']:.4f} | RMSE = {xgb_hold['test_rmse']:.2f} MPa"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "model_evaluation.png", dpi=150)
        plt.close()
    print(f"  [OK] Performance dashboard saved -> {REPORTS_DIR / 'model_evaluation.png'}\n")

    strip = lambda h: {k: round(v, 4) for k, v in h.items() if k != "test_preds"}
    return {
        "xgboost": {"cv_r2": round(xgb_cv_r2, 4), "cv_rmse": round(xgb_cv_rmse, 4), **strip(xgb_hold)},
        "rf_baseline": {"cv_r2": round(rf_cv_r2, 4), "cv_rmse": round(rf_cv_rmse, 4), **strip(rf_hold)},
    }

# ─────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def run_shap_explanations(model, df: pd.DataFrame, features: list):
    """
    Generate and save SHAP summary plots.
    """
    print("  [->] Computing SHAP values...")
    X = df[features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("XGBoost SHAP Feature Importance Summary", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] SHAP summary plot saved -> {REPORTS_DIR / 'shap_summary.png'}\n")

# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n[1/5] Loading UCI Concrete Compressive Strength dataset...")
    df = load_data()  # 1030 rows, 25 exact duplicates dropped → 1005

    print("[2/5] Preparing data & separating mix designs (no leakage)...")
    train_df, test_df = prepare_data(df)

    print("[3/5] Tuning XGBoost hyperparameters with Optuna...")
    best_params = tune_xgboost(train_df)

    # Evaluate on our splits (re-fitting on train only, alongside the RF baseline)
    metrics = evaluate_models(best_params, train_df, test_df)

    print("[4/5] Training final production model on full dataset...")
    full_df = engineer_features(df)
    final_model = make_xgb(best_params)
    final_model.fit(full_df[FEATURES], full_df[TARGET])
    print("  [OK] Final model training complete.")

    # Explain production model
    print("[5/5] Running SHAP Explainability on production model...")
    run_shap_explanations(final_model, full_df, FEATURES)

    metrics.update({
        "dataset": {"rows": len(df), "mix_designs": int(df.groupby(RAW_FEATURES[:-1]).ngroups)},
        "split": {"train_samples": len(train_df), "test_samples": len(test_df), "cv_folds": N_SPLITS},
        "best_params": best_params,
    })
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Metrics saved -> {REPORTS_DIR / 'metrics.json'}")

    # Export model to API path
    print(f"  [->] Exporting production model to {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"  [OK] Model exported successfully ({size_kb:.1f} KB)\n")

    print("=" * 60)
    print("  OptiMPa Pipeline Upgrade Complete. Ready for API Server.")
    print("=" * 60 + "\n")
