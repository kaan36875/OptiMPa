"""
OptiMPa — Concrete Compressive Strength Predictor
Phase 1: ML Pipeline & Model Training

Dataset: UCI Concrete Compressive Strength (ID: 165)
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

Loaded via the official `ucimlrepo` library — no CSV files, no GitHub URLs.

Feature Glossary (Civil Engineering -> ML):
  - cement          : Binder component [kg/m3] — primary strength driver
  - slag            : Blast furnace slag [kg/m3] — latent hydraulic binder
  - fly_ash         : Industrial byproduct [kg/m3] — pozzolanic filler
  - water           : Water content [kg/m3] — w/c ratio affects porosity
  - superplasticizer: Chemical admixture [kg/m3] — workability without extra water
  - coarse_agg      : Coarse aggregate [kg/m3] — structural skeleton
  - fine_agg        : Fine aggregate [kg/m3] — void filler between coarse particles
  - age             : Curing age [days] — hydration progress (28-day standard)

Target:
  - strength        : Compressive strength [MPa] — what we predict
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concrete_data import load_data as _load_embedded
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load the UCI Concrete Compressive Strength dataset.
    Data is embedded directly in concrete_data.py — zero network/SSL dependency.
    Same 9 columns as the original UCI dataset (I-Cheng Yeh, 1998).
    """
    return _load_embedded()


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """
    Quick sanity-check EDA. Prints stats and saves a correlation heatmap.
    In production pipelines this would write to MLflow or W&B.
    """
    print("\n" + "═" * 60)
    print("  DATASET OVERVIEW")
    print("═" * 60)
    print(f"  Shape         : {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Strength range: {df['strength'].min():.2f} – {df['strength'].max():.2f} MPa")
    print(f"  Strength mean : {df['strength'].mean():.2f} MPa")
    print("─" * 60)
    print(df.describe().round(2).to_string())
    print("═" * 60 + "\n")

    # Correlation heatmap — helps justify feature importance later
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df.corr(),
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Matrix — Concrete Dataset", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig("reports/correlation_heatmap.png", dpi=150)
    plt.close()
    print("  [✓] Correlation heatmap saved → reports/correlation_heatmap.png\n")


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Split features and target, then create train/test sets.

    Why 80/20 split?
    The dataset has ~1030 rows. 80/20 gives ~824 training samples —
    enough for RF while leaving 206 hold-out samples for honest evaluation.

    Why random_state=42?
    Reproducibility. Anyone cloning this repo gets identical splits.
    """
    FEATURES = [
        "cement", "slag", "fly_ash", "water",
        "superplasticizer", "coarse_agg", "fine_agg", "age"
    ]
    TARGET = "strength"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
    )

    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Test  samples : {X_test.shape[0]}\n")

    return X_train, X_test, y_train, y_test, FEATURES


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestRegressor:
    """
    Random Forest Regressor — chosen for concrete strength prediction because:

    1. Non-linear relationships: Strength vs water/cement ratio is highly
       non-linear (Abrams' law). RF captures this without manual feature
       engineering unlike linear regression.

    2. Feature importance: We can directly inspect which mix components
       drive strength — great for Civil Engineering explainability.

    3. Robustness: RF averages over many decision trees, so individual
       outlier mixes (bad batches) don't skew the model significantly.

    Hyperparameters:
      n_estimators=300  : More trees → lower variance; 300 is sweet spot
                          for this dataset size (empirically validated).
      max_features='sqrt': Each split considers sqrt(8)≈3 features →
                          decorrelates trees while keeping individual
                          trees weak (bias-variance tradeoff).
      min_samples_leaf=2 : Prevents overfitting on small leaf nodes.
      random_state=42    : Reproducibility.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,           # Use all CPU cores
        random_state=42,
    )

    print("  [→] Training Random Forest Regressor...")
    model.fit(X_train, y_train)
    print("  [✓] Training complete.\n")

    return model


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names) -> dict:
    """
    Full evaluation suite:
      - R² Score      : How much variance the model explains (1.0 = perfect)
      - RMSE          : Average prediction error in MPa — Civil Eng. interpretable
      - 5-fold CV R²  : Checks if good test score is luck or genuine generalisation

    In the concrete domain, ±5 MPa RMSE is considered acceptable for mix design.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_r2   = r2_score(y_train, y_pred_train)
    test_r2    = r2_score(y_test, y_pred_test)
    test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # 5-fold cross-validation on the full dataset for robustness check
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="r2", n_jobs=-1
    )

    metrics = {
        "train_r2"  : round(train_r2, 4),
        "test_r2"   : round(test_r2, 4),
        "test_rmse" : round(test_rmse, 4),
        "cv_mean_r2": round(cv_scores.mean(), 4),
        "cv_std_r2" : round(cv_scores.std(), 4),
    }

    print("═" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("═" * 60)
    print(f"  Train R²          : {metrics['train_r2']:.4f}")
    print(f"  Test  R²          : {metrics['test_r2']:.4f}")
    print(f"  Test  RMSE        : {metrics['test_rmse']:.4f} MPa")
    print(f"  CV R² (5-fold)    : {metrics['cv_mean_r2']:.4f} ± {metrics['cv_std_r2']:.4f}")
    print("═" * 60 + "\n")

    # Feature importance — which mix component matters most?
    importance_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("  FEATURE IMPORTANCES (Gini Impurity Reduction)")
    print("  " + "─" * 40)
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<20} {bar} ({row['importance']:.3f})")
    print()

    # Predicted vs Actual scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_test, alpha=0.6, color="#4F86C6", edgecolors="white", linewidth=0.4, s=50)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=1.5, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Strength (MPa)")
    axes[0].set_ylabel("Predicted Strength (MPa)")
    axes[0].set_title(f"Predicted vs Actual  |  R² = {test_r2:.4f}")
    axes[0].legend()

    axes[1].barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#4F86C6", edgecolor="white"
    )
    axes[1].set_xlabel("Relative Importance")
    axes[1].set_title("Feature Importances")
    axes[1].invert_yaxis()

    plt.suptitle("OptiMPa — Random Forest Evaluation", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig("reports/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✓] Evaluation plots saved → reports/model_evaluation.png\n")

    return metrics


# ─────────────────────────────────────────────
# 6. MODEL EXPORT
# ─────────────────────────────────────────────

def export_model(model, output_path: str) -> None:
    """
    Persist the trained model with joblib (preferred over pickle for sklearn).
    The API server will load this artifact at startup — zero retraining needed.
    """
    joblib.dump(model, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [✓] Model exported → {output_path}  ({size_kb:.1f} KB)\n")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs("reports", exist_ok=True)

    MODEL_PATH = "../api/model.pkl"   # Drop the artifact directly into the API folder

    # -- Step 1: Load ------------------------------------------
    print("\n[1/5] Loading dataset via ucimlrepo...")
    df = load_data()

    # -- Step 2: EDA -------------------------------------------
    print("[2/5] Running EDA...")
    run_eda(df)

    # -- Step 3: Preprocess ------------------------------------
    print("[3/5] Preprocessing...")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # -- Step 4: Train -----------------------------------------
    print("[4/5] Training model...")
    model = train_model(X_train, y_train)

    # -- Step 5: Evaluate & Export -----------------------------
    print("[5/5] Evaluating and exporting...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
    export_model(model, MODEL_PATH)

    print("=" * 60)
    print("  Pipeline complete. model.pkl is ready for the API server.")
    print("=" * 60 + "\n")
