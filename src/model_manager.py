import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import CV_FOLDS, CV_SAMPLE_SIZE, DATA_PATH, METRICS_DIR, MODEL_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.data_loader import load_dataset
from src.data_validation import encode_target, validate_dataset
from src.evaluate_models import evaluate_predictions
from src.preprocessing import build_pipeline, feature_frame
from src.train_logistic import create_model as create_logistic
from src.train_random_forest import create_model as create_random_forest
from src.train_xgboost import create_model as create_xgboost

logger = logging.getLogger(__name__)


def model_registry() -> dict:
    return {
        "Logistic Regression": create_logistic,
        "Random Forest": create_random_forest,
        "XGBoost": create_xgboost,
    }


def split_dataset(frame: pd.DataFrame):
    features = feature_frame(frame)
    target = encode_target(frame[TARGET_COLUMN])
    return train_test_split(features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target)


def train_all_models(data_path: str | Path | None = None) -> dict:
    frame = load_dataset(data_path or DATA_PATH)
    validation = validate_dataset(frame)
    if not validation["valid"]:
        failures = [check["detail"] for check in validation["checks"] if check["status"] == "fail"]
        raise ValueError("Dataset validation failed: " + "; ".join(failures))

    X_train, X_test, y_train, y_test = split_dataset(frame)
    cv_count = min(CV_SAMPLE_SIZE, len(X_train))
    cv_frame = X_train.sample(n=cv_count, random_state=RANDOM_STATE)
    cv_target = y_train.loc[cv_frame.index]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {metric: metric for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    parameter_grids = {
        "Logistic Regression": {"classifier__C": [0.5, 1.0, 2.0]},
        "Random Forest": {"classifier__max_features": ["sqrt", 0.8]},
        "XGBoost": {"classifier__max_depth": [4, 5]},
    }
    registry = model_registry()
    cv_results = {}
    best_parameters = {}
    metrics = {}
    prediction_frame = pd.DataFrame({"actual": y_test.to_numpy()})

    for name, factory in registry.items():
        logger.info("Cross-validating %s on %s training rows", name, cv_count)
        cv_pipeline = build_pipeline(cv_frame, factory())
        search = GridSearchCV(cv_pipeline, parameter_grids[name], cv=cv, scoring=scoring,
                      refit="f1", n_jobs=1, return_train_score=False)
        search.fit(cv_frame, cv_target)
        best_index = search.best_index_
        cv_results[name] = {metric: float(search.cv_results_[f"mean_test_{metric}"][best_index])
                    for metric in scoring}
        best_parameters[name] = search.best_params_
        pipeline = build_pipeline(X_train, factory())
        pipeline.set_params(**search.best_params_)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        result = evaluate_predictions(y_test, probabilities)
        result["cv_mean_f1"] = cv_results[name]["f1"]
        metrics[name] = result
        prediction_frame[name] = probabilities
        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path)
        logger.info("Saved %s to %s", name, model_path)

    selected = max(registry, key=lambda name: (cv_results[name]["f1"], cv_results[name]["roc_auc"]))
    metadata = {
        "selected_model": selected,
        "selection_rule": "Highest mean 3-fold training-only CV F1; CV ROC-AUC breaks exact F1 ties.",
        "cv_folds": CV_FOLDS,
        "cv_sample_rows": cv_count,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": list(X_train.columns),
        "validation": validation,
        "cv_metrics": cv_results,
        "best_parameters": best_parameters,
        "test_metrics": metrics,
        "source": str(data_path or DATA_PATH),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(metrics, orient="index").drop(columns="confusion_matrix").to_csv(
        METRICS_DIR / "model_metrics.csv", index_label="model",
    )
    prediction_frame.to_csv(METRICS_DIR / "test_predictions.csv", index=False)
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    logger.info("Selected %s by training-only CV F1", selected)
    return metadata


def load_models() -> tuple[dict, dict]:
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Trained models are missing. Run `python main.py train` first.")
    with metadata_path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    models = {}
    for name in model_registry():
        path = MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Saved model is missing: {path}")
        try:
            models[name] = joblib.load(path)
        except Exception as error:
            raise RuntimeError(f"Could not load saved {name} model: {error}") from error
    return models, metadata


def saved_metrics() -> pd.DataFrame:
    path = METRICS_DIR / "model_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError("Metrics are not yet computed. Run `python main.py train` first.")
    return pd.read_csv(path, index_col="model")