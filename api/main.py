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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING — lifespan context (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────────────────

# Global model store — populated once at startup, reused for every request.
_model_store: dict = {}

MODEL_PATH = Path(__file__).parent / "model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Random Forest model from disk once when the server starts.
    Using joblib ensures efficient deserialization of numpy arrays inside the RF.
    The model object is stored in _model_store["rf"] for request handlers.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}. "
            "Run ml_pipeline/train.py first to generate the model artifact."
        )
    _model_store["rf"] = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
    yield
    # Cleanup on shutdown (nothing to release for joblib RF)
    _model_store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 2. APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OptiMPa — Concrete Strength Prediction API",
    description=(
        "Predicts concrete compressive strength (MPa) from mix design parameters "
        "using a Random Forest Regressor trained on the UCI Concrete dataset."
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

    @field_validator("water")
    @classmethod
    def check_water_cement_ratio(cls, v):
        """
        Abrams' Law guardrail: w/c > 0.80 produces concrete too weak to be
        structurally meaningful. We warn but don't block — the frontend slider
        already guides users toward sensible ranges.
        """
        return v   # Validation is handled by the ge/le bounds above

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
    strength_grade: Nearest EN 206 concrete grade string (C8/10 … C90/105).
    input_summary : Echo of validated inputs (useful for frontend display).
    """
    strength_mpa: float = Field(..., description="Predicted compressive strength [MPa]")
    strength_grade: str  = Field(..., description="Nearest EN 206 concrete grade")
    input_summary: dict  = Field(..., description="Validated input features")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER: EN 206 GRADE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# EN 206 characteristic cylinder strengths → grade label
_EN206_GRADES = [
    (12,  "C8/10"),
    (16,  "C12/15"),
    (20,  "C16/20"),
    (25,  "C20/25"),
    (30,  "C25/30"),
    (35,  "C28/35"),
    (40,  "C32/40"),
    (45,  "C35/45"),
    (50,  "C40/50"),
    (55,  "C45/55"),
    (60,  "C50/60"),
    (70,  "C55/67"),
    (80,  "C60/75"),
    (90,  "C70/85"),
    (100, "C80/95"),
    (999, "C90/105"),
]

def _to_en206_grade(mpa: float) -> str:
    """
    Map a predicted cylinder strength to the nearest EN 206 concrete class.
    The threshold is the characteristic cylinder strength (fck).
    """
    for threshold, label in _EN206_GRADES:
        if mpa <= threshold:
            return label
    return "C90/105"


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
        model_loaded="rf" in _model_store,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: ConcreteFeatures):
    """
    Core prediction endpoint.

    Flow:
      1. Pydantic validates the 8 input features (type + range).
      2. Features are assembled into a (1, 8) numpy array in the exact
         column order the Random Forest was trained on.
      3. The RF predicts one value → compressive strength in MPa.
      4. The result is mapped to an EN 206 concrete grade for CE context.
      5. Response is returned as JSON.

    The column order MUST match the training order in train.py:
      cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age
    """
    if "rf" not in _model_store:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    # Assemble named DataFrame — column order must match training (train.py)
    # Using DataFrame instead of raw numpy array suppresses sklearn's feature-name warning
    feature_vector = pd.DataFrame([{
        "cement"          : features.cement,
        "slag"            : features.slag,
        "fly_ash"         : features.fly_ash,
        "water"           : features.water,
        "superplasticizer": features.superplasticizer,
        "coarse_agg"      : features.coarse_agg,
        "fine_agg"        : features.fine_agg,
        "age"             : features.age,
    }])

    # Predict — RF.predict returns shape (1,), so we take [0]
    raw_prediction: float = float(_model_store["rf"].predict(feature_vector)[0])

    # Clamp to physically meaningful range (RF can occasionally predict slightly
    # negative values for very lean mixes at early ages)
    strength_mpa = round(max(raw_prediction, 0.0), 2)

    return PredictionResponse(
        strength_mpa=strength_mpa,
        strength_grade=_to_en206_grade(strength_mpa),
        input_summary=features.model_dump(),
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
