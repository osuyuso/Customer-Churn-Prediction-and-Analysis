"""
Builds the full churn pipeline:
- loads raw dataset
- cleans + engineers features
- trains and tunes multiple models
- persists best model + metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401 (kept for future statistical logging)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import shuffle

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "customer_churn_dataset-testing-master.csv"
ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

TENURE_BINS = [0, 12, 24, 48, 60, 100]
TENURE_LABELS = ["<1y", "1-2y", "2-4y", "4-5y", "5y+"]
CONTRACT_MAP = {"Monthly": 1, "Quarterly": 3, "Annual": 12}


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop_duplicates().dropna().reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | list]]:
    engineered = df.copy()
    engineered["TenureBucket"] = pd.cut(
        engineered["Tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )

    support_median = float(engineered["Support Calls"].median())
    engineered["HighSupport"] = (engineered["Support Calls"] > support_median).astype(
        int
    )

    last_interaction_median = float(engineered["Last Interaction"].median())
    engineered["RecentInteraction"] = (
        engineered["Last Interaction"] <= last_interaction_median
    ).astype(int)

    usage_per_tenure = engineered["Usage Frequency"] / engineered["Tenure"].replace(
        0, np.nan
    )
    usage_per_tenure_median = float(usage_per_tenure.median())
    engineered["UsagePerTenure"] = usage_per_tenure.fillna(usage_per_tenure_median)

    engineered["ContractMonths"] = engineered["Contract Length"].map(CONTRACT_MAP)
    spend_per_month = engineered["Total Spend"] / engineered["ContractMonths"]
    spend_per_month_median = float(spend_per_month.median())
    engineered["SpendPerMonth"] = spend_per_month.fillna(spend_per_month_median)

    engineered = engineered.drop(columns=["CustomerID", "ContractMonths"])
    config = {
        "tenure_bins": TENURE_BINS,
        "tenure_labels": TENURE_LABELS,
        "contract_map": CONTRACT_MAP,
        "support_calls_median": support_median,
        "last_interaction_median": last_interaction_median,
        "usage_per_tenure_median": usage_per_tenure_median,
        "spend_per_month_median": spend_per_month_median,
    }
    return engineered, config


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced"),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def hyperparams() -> dict[str, dict[str, list[int | float | str | None]]]:
    return {
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "Random Forest": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 15, 25],
            "clf__min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 4],
        },
    }


def main() -> None:
    df = load_and_clean(DATA_PATH)
    df.to_csv(ARTIFACT_DIR / "cleaned_customer_churn.csv", index=False)

    df, feature_config = engineer_features(df)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X, y = shuffle(X, y, random_state=42)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models(preprocessor)
    search_spaces = hyperparams()

    results: list[dict] = []
    best_name = None
    best_model = None
    best_f1 = -np.inf

    for name, pipeline in models.items():
        grid = GridSearchCV(
            pipeline,
            search_spaces[name],
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        estimator = grid.best_estimator_
        preds = estimator.predict(X_test)
        probs = estimator.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary"
        )
        roc_auc = roc_auc_score(y_test, probs)
        results.append(
            {
                "model": name,
                "best_params": grid.best_params_,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_model = estimator
            best_name = name

    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACT_DIR / "model_results.csv", index=False)

    if best_model is None or best_name is None:
        raise RuntimeError("No best model identified.")

    model_path = ARTIFACT_DIR / f"best_model_{best_name.replace(' ', '_').lower()}.joblib"
    joblib.dump(best_model, model_path)

    config_path = ARTIFACT_DIR / "feature_config.json"
    with open(config_path, "w", encoding="utf-8") as cfg:
        json.dump(feature_config, cfg, indent=2)

    metadata = {
        "best_model": best_name,
        "metrics": results_df.loc[results_df["model"] == best_name].to_dict(
            orient="records"
        )[0],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "feature_config_path": str(config_path),
    }
    with open(ARTIFACT_DIR / "model_report.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"Artifacts saved to {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()

