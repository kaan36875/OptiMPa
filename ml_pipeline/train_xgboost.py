"""
OptiMPa — Concrete Compressive Strength Predictor
Phase 1 Upgrade: XGBoost + Optuna + SHAP Pipeline

Dataset: UCI Concrete Compressive Strength (ID: 165)
"""

import os
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from concrete_data import load_data

# Suppress Optuna logs to keep stdout clean unless warning/error
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    mix_cols = ["cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg"]
    df["mix_id"] = df.groupby(mix_cols).ngroup()
    
    # Apply feature engineering
    df = engineer_features(df)
    
    FEATURES = [
        "cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age",
        "wc_ratio", "binder_total", "wb_ratio", "fine_coarse_ratio", "slag_cement_ratio", "fly_ash_cement_ratio"
    ]
    TARGET = "strength"
    
    # Split unique mix designs into 80% train and 20% test
    unique_mixes = df["mix_id"].unique()
    train_mixes, test_mixes = train_test_split(unique_mixes, test_size=0.20, random_state=42)
    
    train_df = df[df["mix_id"].isin(train_mixes)].reset_index(drop=True)
    test_df = df[df["mix_id"].isin(test_mixes)].reset_index(drop=True)
    
    print(f"  [->] Total Mix Designs: {len(unique_mixes)}")
    print(f"  [->] Training Mix Designs (for CV & Tuning): {len(train_mixes)}")
    print(f"  [->] Test Mix Designs (held out completely): {len(test_mixes)}")
    print(f"  [->] Train Samples: {train_df.shape[0]} | Test Samples: {test_df.shape[0]}\n")
    
    return train_df, test_df, FEATURES, TARGET

# ─────────────────────────────────────────────
# 3. OPTUNA HYPERPARAMETER TUNING (GroupKFold CV)
# ─────────────────────────────────────────────

def tune_xgboost(train_df: pd.DataFrame, features: list, target: str) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna and 5-fold GroupKFold.
    Target metric: Root Mean Squared Error (RMSE).
    """
    X_train = train_df[features]
    y_train = train_df[target]
    groups = train_df["mix_id"]
    
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
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        }
        
        gkf = GroupKFold(n_splits=5)
        rmse_scores = []
        
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)
            
        return np.mean(rmse_scores)
    
    print("  [->] Starting Optuna Hyperparameter Optimization (5-fold GroupKFold)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    print("  [OK] Optimization finished.")
    print(f"      Best trial CV RMSE: {study.best_value:.4f} MPa")
    print(f"      Best parameters: {study.best_params}\n")
    
    return study.best_params

# ─────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, target: str):
    """
    Evaluate on train and unseen test mix designs.
    """
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    print("=" * 60)
    print("  XGBOOST MODEL EVALUATION RESULTS (LEAK-FREE)")
    print("=" * 60)
    print(f"  Train R2           : {train_r2:.4f}")
    print(f"  Test  R2 (Unseen)  : {test_r2:.4f}")
    print(f"  Train RMSE         : {train_rmse:.4f} MPa")
    print(f"  Test  RMSE (Unseen): {test_rmse:.4f} MPa")
    print("=" * 60 + "\n")
    
    # Save Predicted vs Actual Plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, test_preds, alpha=0.6, color="#4F86C6", edgecolors="white", linewidth=0.4, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Strength (MPa)")
    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title(f"XGBoost Test Set Performance | R2 = {test_r2:.4f} | RMSE = {test_rmse:.2f} MPa")
    ax.legend()
    plt.tight_layout()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/model_evaluation.png", dpi=150)
    plt.close()
    print("  [OK] Performance dashboard saved -> reports/model_evaluation.png\n")

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
    plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] SHAP summary plot saved -> reports/shap_summary.png\n")

# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_PATH = "../api/model.pkl"
    
    print("\n[1/5] Loading UCI Concrete Compressive Strength dataset...")
    df = load_data()
    
    print("[2/5] Preparing data & separating mix designs (no leakage)...")
    train_df, test_df, features, target = prepare_data(df)
    
    print("[3/5] Tuning XGBoost hyperparameters with Optuna...")
    best_params = tune_xgboost(train_df, features, target)
    
    print("[4/5] Training final production model on full dataset...")
    # Engineer features on the full dataset
    full_df = engineer_features(df)
    X_full = full_df[features]
    y_full = full_df[target]
    
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(X_full, y_full)
    print("  [OK] Final model training complete.")
    
    # Evaluate on our splits (re-fitting on train only for evaluation visualization)
    eval_model = xgb.XGBRegressor(**best_params, random_state=42)
    eval_model.fit(train_df[features], train_df[target])
    evaluate_model(eval_model, train_df, test_df, features, target)
    
    # Explain production model
    print("[5/5] Running SHAP Explainability on production model...")
    run_shap_explanations(final_model, full_df, features)
    
    # Export model to API path
    print(f"  [->] Exporting production model to {MODEL_PATH}...")
    joblib.dump(final_model, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"  [OK] Model exported successfully ({size_kb:.1f} KB)\n")
    
    print("=" * 60)
    print("  OptiMPa Pipeline Upgrade Complete. Ready for API Server.")
    print("=" * 60 + "\n")
