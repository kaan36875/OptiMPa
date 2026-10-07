"""
OptiMPa — Phase 2: FastAPI Prediction Server
=============================================

Architecture:
  Next.js Frontend  →  POST /predict  →  FastAPI  →  model.pkl  →  MPa result
                                              ↑
                                    Pydantic validates the
                                    8 concrete mix features

Design decisions:
  - Model is loaded ONCE at startup via lifespan context manager (FastAPI best
    practice). Avoids per-request disk I/O overhead.
  - Pydantic v2 Field validators enforce realistic Civil Engineering ranges so
    the model is never asked to extrapolate beyond its training domain.
  - CORS is wide-open for localhost ports so the Next.js dev server (3000) and
    any other local tool can call the API freely during development.
  - A /health endpoint lets the frontend detect whether the API is alive before
    showing the UI.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING — lifespan context (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────────────────

# Global model store — populated once at startup, reused for every request.
_model_store: dict = {}

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Exact column order expected by the XGBoost model (see ml_pipeline/train_xgboost.py)
FEATURES_ORDER = [
    "cement", "slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age",
    "wc_ratio", "binder_total", "wb_ratio", "fine_coarse_ratio", "slag_cement_ratio", "fly_ash_cement_ratio"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the XGBoost model and initialize SHAP TreeExplainer at startup.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run ml_pipeline/train_xgboost.py first to generate the model artifact."
        )
    model = joblib.load(MODEL_PATH)
    # Fail fast if the artifact was trained on a different feature layout
    trained_features = list(model.get_booster().feature_names or [])
    if trained_features != FEATURES_ORDER:
        raise RuntimeError(
            f"model.pkl feature mismatch.\n  expected: {FEATURES_ORDER}\n  model   : {trained_features}"
        )
    _model_store["model"] = model
    _model_store["explainer"] = shap.TreeExplainer(model)
    print(f"[OK] Model and SHAP Explainer loaded from {MODEL_PATH}")
    yield
    # Cleanup on shutdown
    _model_store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2. APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OptiMPa — Concrete Strength Prediction API",
    description=(
        "Predicts concrete compressive strength (MPa) from mix design parameters "
        "using an XGBoost Regressor trained on the UCI Concrete dataset, with "
        "local SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# Allows the Next.js dev server (port 3000) and any localhost origin to POST.
# In production, replace ["*"] with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,       # False when allow_origins=["*"]
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ConcreteFeatures(BaseModel):
    """
    The 8 input features that define a concrete mix design.
    Ranges are validated against the UCI dataset's min/max values
    plus a small buffer — prevents the model from extrapolating wildly.

    Civil Engineering context:
      All quantities are in kg per cubic meter of fresh concrete (kg/m³).
      Age is in days; 28 days is the standard reference strength per EN 206.
    """

    cement: float = Field(
        ...,
        ge=100.0, le=700.0,
        description="Cement content [kg/m³]. Primary binder.",
        examples=[350.0],
    )
    slag: float = Field(
        ...,
        ge=0.0, le=400.0,
        description="Blast furnace slag [kg/m³]. Latent hydraulic binder.",
        examples=[0.0],
    )
    fly_ash: float = Field(
        ...,
        ge=0.0, le=250.0,
        description="Fly ash [kg/m³]. Pozzolanic supplementary material.",
        examples=[0.0],
    )
    water: float = Field(
        ...,
        ge=120.0, le=250.0,
        description="Water content [kg/m³]. Controls w/c ratio and porosity.",
        examples=[180.0],
    )
    superplasticizer: float = Field(
        ...,
        ge=0.0, le=35.0,
        description="Superplasticizer [kg/m³]. Reduces water demand.",
        examples=[6.0],
    )
    coarse_agg: float = Field(
        ...,
        ge=800.0, le=1200.0,
        description="Coarse aggregate [kg/m³]. Structural skeleton of the mix.",
        examples=[1040.0],
    )
    fine_agg: float = Field(
        ...,
        ge=550.0, le=1050.0,
        description="Fine aggregate [kg/m³]. Fills voids between coarse particles.",
        examples=[755.0],
    )
    age: int = Field(
        ...,
        ge=1, le=365,
        description="Curing age [days]. 28d = standard reference per EN 206.",
        examples=[28],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "cement": 350.0,
                "slag": 0.0,
                "fly_ash": 0.0,
                "water": 175.0,
                "superplasticizer": 6.0,
                "coarse_agg": 1040.0,
                "fine_agg": 780.0,
                "age": 28,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    API response schema.
    strength_mpa  : The model's predicted compressive strength.
    strength_grade: Highest EN 206 class the mix satisfies (C8/10 … C100/115).
    input_summary : Echo of validated inputs (useful for frontend display).
    """
    strength_mpa: float = Field(..., description="Predicted compressive strength [MPa]")
    strength_grade: str  = Field(..., description="Estimated EN 206 strength class")
    input_summary: dict  = Field(..., description="Validated input features")


class ExplanationResponse(BaseModel):
    """
    API response schema for SHAP local explainability.
    """
    predicted_strength: float = Field(..., description="Predicted compressive strength [MPa]")
    strength_grade: str = Field(..., description="Estimated EN 206 strength class")
    base_value: float = Field(..., description="SHAP base value (mean prediction)")
    shap_values: dict[str, float] = Field(..., description="SHAP value for each feature contribution")
    engineered_features: dict[str, float] = Field(..., description="Values of engineered features")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER: EN 206 GRADE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# EN 206 strength classes: (fck,cyl [MPa], label), ascending
_EN206_GRADES = [
    (8,   "C8/10"),
    (12,  "C12/15"),
    (16,  "C16/20"),
    (20,  "C20/25"),
    (25,  "C25/30"),
    (30,  "C30/37"),
    (35,  "C35/45"),
    (40,  "C40/50"),
    (45,  "C45/55"),
    (50,  "C50/60"),
    (55,  "C55/67"),
    (60,  "C60/75"),
    (70,  "C70/85"),
    (80,  "C80/95"),
    (90,  "C90/105"),
    (100, "C100/115"),
]

# EN 1992-1-1 Table 3.1: fck = fcm − 8 MPa. The model predicts a mean
# (cylinder) strength, so the margin is removed before classification.
_FCM_FCK_MARGIN = 8.0


def _to_en206_grade(mpa: float) -> str:
    """
    Map a predicted mean cylinder strength (fcm) to the highest EN 206 class
    whose characteristic strength fck it satisfies.
    """
    fck = mpa - _FCM_FCK_MARGIN
    grade = "< C8/10"
    for threshold, label in _EN206_GRADES:
        if fck >= threshold:
            grade = label
    return grade


def build_feature_vector(features: "ConcreteFeatures") -> pd.DataFrame:
    """
    Assemble the 8 raw inputs into a (1, 14) DataFrame with the physical
    features engineered exactly as in ml_pipeline/train_xgboost.py.
    """
    fv = pd.DataFrame([features.model_dump()])
    fv["wc_ratio"] = fv["water"] / (fv["cement"] + 1e-6)
    fv["binder_total"] = fv["cement"] + fv["slag"] + fv["fly_ash"]
    fv["wb_ratio"] = fv["water"] / (fv["binder_total"] + 1e-6)
    fv["fine_coarse_ratio"] = fv["fine_agg"] / (fv["coarse_agg"] + 1e-6)
    fv["slag_cement_ratio"] = fv["slag"] / (fv["cement"] + 1e-6)
    fv["fly_ash_cement_ratio"] = fv["fly_ash"] / (fv["cement"] + 1e-6)
    return fv[FEATURES_ORDER]


def predict_strength(fv: pd.DataFrame) -> float:
    """Model prediction clamped to a physically meaningful range (>= 0 MPa)."""
    raw_prediction = float(_model_store["model"].predict(fv)[0])
    return round(max(raw_prediction, 0.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """
    Lightweight liveness probe. The Next.js frontend calls this on mount
    to detect whether the FastAPI server is running before enabling the UI.
    """
    return HealthResponse(
        status="ok",
        model_loaded="model" in _model_store,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: ConcreteFeatures):
    """
    Core prediction endpoint.

    Flow:
      1. Pydantic validates the 8 input features (type + range).
      2. Features are engineered and ordered as in training (build_feature_vector).
      3. The XGBoost model predicts the compressive strength.
      4. The result is mapped to an EN 206 concrete grade.
      5. Response is returned as JSON.
    """
    if "model" not in _model_store:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    strength_mpa = predict_strength(build_feature_vector(features))

    return PredictionResponse(
        strength_mpa=strength_mpa,
        strength_grade=_to_en206_grade(strength_mpa),
        input_summary=features.model_dump(),
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain(features: ConcreteFeatures):
    """
    Generates local SHAP explanations for a specific mix design.
    Also returns the prediction and grade, so a client needs only this call.
    """
    if "model" not in _model_store or "explainer" not in _model_store:
        raise HTTPException(status_code=503, detail="Model/Explainer not loaded yet.")

    feature_vector = build_feature_vector(features)
    strength_mpa = predict_strength(feature_vector)

    # Compute SHAP Values
    explainer = _model_store["explainer"]
    shap_results = explainer(feature_vector)

    # shap_results.base_values is an array of shape (1,) or float
    base_value = float(shap_results.base_values[0])
    
    # Extract contributions
    shap_contribs = {
        feat: float(shap_results.values[0, i])
        for i, feat in enumerate(FEATURES_ORDER)
    }

    # Extract engineered features for output summary
    eng_feats = {
        "wc_ratio": round(float(feature_vector["wc_ratio"].iloc[0]), 3),
        "binder_total": round(float(feature_vector["binder_total"].iloc[0]), 2),
        "wb_ratio": round(float(feature_vector["wb_ratio"].iloc[0]), 3),
        "fine_coarse_ratio": round(float(feature_vector["fine_coarse_ratio"].iloc[0]), 3),
    }

    return ExplanationResponse(
        predicted_strength=strength_mpa,
        strength_grade=_to_en206_grade(strength_mpa),
        base_value=round(base_value, 2),
        shap_values={k: round(v, 4) for k, v in shap_contribs.items()},
        engineered_features=eng_feats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # Auto-reload on file changes during development
        log_level="info",
    )
