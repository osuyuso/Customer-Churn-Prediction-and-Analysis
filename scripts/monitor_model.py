"""
Automated monitoring script for churn prediction model.

Checks:
- Data drift (KS-test on feature distributions)
- Model performance (precision, recall, F1 on labeled feedback)
- Prediction volume and distribution

Run:
    python scripts/monitor_model.py

Outputs:
    - artifacts/monitoring_report_YYYYMMDD.json
    - Alerts to stdout if drift/degradation detected
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, precision_recall_fscore_support

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_META_PATH = ARTIFACT_DIR / "model_report.json"
TRAINING_DATA_PATH = ARTIFACT_DIR / "cleaned_customer_churn.csv"

# Thresholds
KS_DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test
F1_DEGRADATION_THRESHOLD = 0.05  # absolute drop in F1 that triggers alert


def load_training_data() -> pd.DataFrame:
    """Load the original training dataset."""
    return pd.read_csv(TRAINING_DATA_PATH)


def load_production_data() -> pd.DataFrame:
    """
    Load recent production predictions with ground truth labels.
    
    In production, this would query a database of:
    - prediction timestamp
    - customer features
    - predicted churn probability
    - actual churn outcome (ground truth, collected after prediction)
    
    For demo purposes, we simulate by sampling from training data.
    """
    # PLACEHOLDER: Replace with actual production data query
    df = load_training_data()
    # Simulate production drift by sampling last 1000 records
    return df.tail(1000).copy()


def detect_drift(
    train_df: pd.DataFrame, prod_df: pd.DataFrame, numeric_cols: list[str]
) -> dict[str, dict[str, float]]:
    """
    Run KS-test on numeric features to detect distribution drift.
    
    Returns:
        Dict mapping feature -> {statistic, p_value, drift_detected}
    """
    drift_results = {}
    for col in numeric_cols:
        if col in train_df.columns and col in prod_df.columns:
            train_vals = train_df[col].dropna()
            prod_vals = prod_df[col].dropna()
            statistic, p_value = ks_2samp(train_vals, prod_vals)
            drift_results[col] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < KS_DRIFT_THRESHOLD),
            }
    return drift_results


def evaluate_performance(
    model: object, prod_df: pd.DataFrame, target_col: str = "Churn"
) -> dict[str, float]:
    """
    Evaluate model on production data with ground truth labels.
    
    Returns:
        Dict with precision, recall, f1, and degradation flag.
    """
    if target_col not in prod_df.columns:
        return {"error": "No ground truth labels available"}

    X_prod = prod_df.drop(columns=[target_col])
    y_true = prod_df[target_col]

    try:
        y_pred = model.predict(X_prod)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(report["1"]["support"]),
            "accuracy": float(report["accuracy"]),
        }
    except Exception as e:
        return {"error": str(e)}


def check_prediction_distribution(prod_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze distribution of predictions in production.
    
    Returns:
        Stats on churn probability distribution and volume.
    """
    if "Churn" not in prod_df.columns:
        return {"error": "No churn predictions found"}

    churn_rate = prod_df["Churn"].mean()
    return {
        "total_predictions": len(prod_df),
        "churn_rate": float(churn_rate),
        "churn_count": int(prod_df["Churn"].sum()),
        "retained_count": int((prod_df["Churn"] == 0).sum()),
    }


def generate_alerts(
    drift_results: dict, perf_metrics: dict, baseline_f1: float
) -> list[str]:
    """Generate human-readable alerts based on monitoring results."""
    alerts = []

    # Drift alerts
    drifted_features = [
        feat for feat, res in drift_results.items() if res.get("drift_detected")
    ]
    if drifted_features:
        alerts.append(
            f"[!] DATA DRIFT DETECTED in {len(drifted_features)} feature(s): "
            f"{', '.join(drifted_features[:5])}"
        )

    # Performance degradation
    if "f1" in perf_metrics:
        current_f1 = perf_metrics["f1"]
        f1_drop = baseline_f1 - current_f1
        if f1_drop > F1_DEGRADATION_THRESHOLD:
            alerts.append(
                f"[!] PERFORMANCE DEGRADATION: F1 dropped from {baseline_f1:.3f} "
                f"to {current_f1:.3f} (Delta={f1_drop:.3f})"
            )

    if not alerts:
        alerts.append("[OK] All monitoring checks passed. No issues detected.")

    return alerts


def main() -> None:
    """Run monitoring pipeline."""
    print("=" * 60)
    print("CHURN MODEL MONITORING REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load metadata and model
    with open(MODEL_META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = joblib.load(meta["model_path"])
    baseline_f1 = meta["metrics"]["f1"]

    print(f"\nModel: {meta['best_model']}")
    print(f"Baseline F1: {baseline_f1:.3f}")
    print(f"Trained: {meta['generated_at']}")

    # Load data
    train_df = load_training_data()
    prod_df = load_production_data()

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Production samples: {len(prod_df)}")

    # Drift detection
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Churn"]
    drift_results = detect_drift(train_df, prod_df, numeric_cols)

    print("\n" + "=" * 60)
    print("DRIFT DETECTION RESULTS")
    print("=" * 60)
    drifted = [f for f, r in drift_results.items() if r["drift_detected"]]
    print(f"Features with drift: {len(drifted)} / {len(drift_results)}")
    for feat in drifted[:5]:
        res = drift_results[feat]
        print(f"  - {feat}: KS={res['ks_statistic']:.4f}, p={res['p_value']:.4f}")

    # Performance evaluation
    perf_metrics = evaluate_performance(model, prod_df)

    print("\n" + "=" * 60)
    print("PERFORMANCE EVALUATION")
    print("=" * 60)
    if "error" in perf_metrics:
        print(f"WARNING: {perf_metrics['error']}")
    else:
        print(f"Precision: {perf_metrics['precision']:.3f}")
        print(f"Recall:    {perf_metrics['recall']:.3f}")
        print(f"F1-Score:  {perf_metrics['f1']:.3f}")
        print(f"Support:   {perf_metrics['support']}")

    # Prediction distribution
    pred_stats = check_prediction_distribution(prod_df)

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)
    if "error" in pred_stats:
        print(f"WARNING: {pred_stats['error']}")
    else:
        print(f"Total predictions: {pred_stats['total_predictions']}")
        print(f"Churn rate: {pred_stats['churn_rate']:.2%}")
        print(f"Churned: {pred_stats['churn_count']}")
        print(f"Retained: {pred_stats['retained_count']}")

    # Generate alerts
    alerts = generate_alerts(drift_results, perf_metrics, baseline_f1)

    print("\n" + "=" * 60)
    print("ALERTS & RECOMMENDATIONS")
    print("=" * 60)
    for alert in alerts:
        print(alert)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": meta["best_model"],
        "baseline_f1": baseline_f1,
        "drift_detection": drift_results,
        "performance": perf_metrics,
        "prediction_stats": pred_stats,
        "alerts": alerts,
    }

    report_path = (
        ARTIFACT_DIR / f"monitoring_report_{datetime.now().strftime('%Y%m%d')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nReport saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

