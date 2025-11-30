"""
FastAPI service for real-time customer churn prediction.

Usage:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health        - Health check
    POST /predict       - Single customer churn prediction
    POST /predict/batch - Batch predictions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_META_PATH = ARTIFACT_DIR / "model_report.json"

# Load model and metadata at startup
try:
    with open(MODEL_META_PATH, "r", encoding="utf-8") as fh:
        MODEL_META = json.load(fh)
    model_path = Path(MODEL_META["model_path"])
    MODEL = joblib.load(model_path)
    logger.info(f"Loaded model: {MODEL_META['best_model']} from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed. Ensure artifacts exist.") from e

# Feature configuration (must match training pipeline)
FEATURE_SCHEMA = MODEL_META.get("features", {})

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn scoring for subscription customers",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    """Single customer feature vector."""

    Age: int = Field(..., ge=18, le=100, description="Customer age in years")
    Gender: str = Field(..., description="Gender (Male/Female)")
    Tenure: int = Field(..., ge=0, description="Months as customer")
    Usage_Frequency: int = Field(..., ge=0, description="Monthly usage count")
    Support_Calls: int = Field(..., ge=0, description="Number of support tickets")
    Payment_Delay: int = Field(..., ge=0, description="Days of payment delay")
    Subscription_Type: str = Field(..., description="Basic/Standard/Premium")
    Contract_Length: str = Field(..., description="Monthly/Quarterly/Annual")
    Total_Spend: float = Field(..., ge=0, description="Total revenue from customer")
    Last_Interaction: int = Field(..., ge=0, le=30, description="Days since last contact")

    # Engineered features (computed or provided)
    TenureBucket: str = Field(
        None, description="Tenure segment: <1y, 1-2y, 2-4y, 4-5y, 5y+"
    )
    HighSupport: int = Field(None, ge=0, le=1, description="Above-median support flag")
    RecentInteraction: int = Field(
        None, ge=0, le=1, description="Recent interaction flag"
    )
    UsagePerTenure: float = Field(None, ge=0, description="Usage intensity ratio")
    SpendPerMonth: float = Field(None, ge=0, description="Normalized monthly spend")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Female",
                "Tenure": 18,
                "Usage_Frequency": 12,
                "Support_Calls": 5,
                "Payment_Delay": 10,
                "Subscription_Type": "Standard",
                "Contract_Length": "Monthly",
                "Total_Spend": 540.0,
                "Last_Interaction": 8,
                "TenureBucket": "1-2y",
                "HighSupport": 1,
                "RecentInteraction": 1,
                "UsagePerTenure": 0.67,
                "SpendPerMonth": 30.0,
            }
        }


class BatchInput(BaseModel):
    """Batch prediction input."""

    customers: list[CustomerInput]


class PredictionOutput(BaseModel):
    """Churn prediction response."""

    customer_id: str | None = None
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_risk: str = Field(
        ..., description="Risk tier: Low/Medium/High/Critical"
    )
    recommendation: str = Field(..., description="Suggested retention action")
    timestamp: str


def compute_engineered_features(data: dict[str, Any]) -> dict[str, Any]:
    """Compute engineered features if not provided."""
    if data.get("TenureBucket") is None:
        tenure = data["Tenure"]
        if tenure < 12:
            data["TenureBucket"] = "<1y"
        elif tenure < 24:
            data["TenureBucket"] = "1-2y"
        elif tenure < 48:
            data["TenureBucket"] = "2-4y"
        elif tenure < 60:
            data["TenureBucket"] = "4-5y"
        else:
            data["TenureBucket"] = "5y+"

    if data.get("UsagePerTenure") is None:
        data["UsagePerTenure"] = (
            data["Usage_Frequency"] / max(data["Tenure"], 1)
        )

    if data.get("SpendPerMonth") is None:
        contract_map = {"Monthly": 1, "Quarterly": 3, "Annual": 12}
        months = contract_map.get(data["Contract_Length"], 1)
        data["SpendPerMonth"] = data["Total_Spend"] / months

    # Placeholder for median-based flags (use training medians in production)
    if data.get("HighSupport") is None:
        data["HighSupport"] = 1 if data["Support_Calls"] > 5 else 0

    if data.get("RecentInteraction") is None:
        data["RecentInteraction"] = 1 if data["Last_Interaction"] <= 15 else 0

    return data


def get_risk_tier(probability: float) -> str:
    """Map probability to risk tier."""
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.5:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"


def get_recommendation(risk_tier: str) -> str:
    """Return action recommendation by risk tier."""
    recommendations = {
        "Critical": "Immediate outreach + personalized retention offer (50% discount)",
        "High": "Customer success call + priority support + usage report",
        "Medium": "Automated engagement email + in-app nudge",
        "Low": "Standard nurture campaign",
    }
    return recommendations.get(risk_tier, "Monitor")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_META["best_model"],
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(customer: CustomerInput) -> PredictionOutput:
    """Predict churn for a single customer."""
    try:
        # Convert to dict and compute missing features
        data = customer.model_dump()
        data = compute_engineered_features(data)

        # Create DataFrame with correct column names
        df = pd.DataFrame([data])

        # Predict
        probability = float(MODEL.predict_proba(df)[0][1])
        risk_tier = get_risk_tier(probability)
        recommendation = get_recommendation(risk_tier)

        logger.info(
            f"Prediction: prob={probability:.3f}, risk={risk_tier}"
        )

        return PredictionOutput(
            churn_probability=probability,
            churn_risk=risk_tier,
            recommendation=recommendation,
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") from e


@app.post("/predict/batch")
def predict_batch(batch: BatchInput) -> dict[str, Any]:
    """Predict churn for multiple customers."""
    try:
        results = []
        for idx, customer in enumerate(batch.customers):
            data = customer.model_dump()
            data = compute_engineered_features(data)
            df = pd.DataFrame([data])
            probability = float(MODEL.predict_proba(df)[0][1])
            risk_tier = get_risk_tier(probability)
            results.append(
                {
                    "index": idx,
                    "churn_probability": probability,
                    "churn_risk": risk_tier,
                    "recommendation": get_recommendation(risk_tier),
                }
            )

        logger.info(f"Batch prediction completed: {len(results)} customers")
        return {
            "predictions": results,
            "timestamp": datetime.utcnow().isoformat(),
            "model": MODEL_META["best_model"],
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {e}"
        ) from e


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    """Return model metadata."""
    return {
        "model_name": MODEL_META["best_model"],
        "metrics": MODEL_META.get("metrics", {}),
        "trained_at": MODEL_META.get("generated_at"),
        "features": FEATURE_SCHEMA,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

