import json

import pandas as pd
from sklearn.base import clone

from src.config import CONTEXT_FEATURES, METRICS_DIR
from src.evaluate_models import evaluate_predictions
from src.model_manager import load_models, split_dataset
from src.preprocessing import build_pipeline


def run_ablation(frame: pd.DataFrame, selected_model: str | None = None) -> dict:
    models, metadata = load_models()
    chosen = selected_model or metadata["selected_model"]
    if chosen not in models:
        raise ValueError(f"Unknown model: {chosen}")
    X_train, X_test, y_train, y_test = split_dataset(frame)
    context_features = [column for column in CONTEXT_FEATURES if column in X_train.columns]
    if not context_features:
        raise ValueError("No context-only features were found in the dataset.")
    estimator = models[chosen].named_steps["classifier"]
    context_pipeline = build_pipeline(X_train[context_features], clone(estimator))
    context_pipeline.fit(X_train[context_features], y_train)
    context_probability = context_pipeline.predict_proba(X_test[context_features])[:, 1]
    full_probability = models[chosen].predict_proba(X_test)[:, 1]
    full_metrics = evaluate_predictions(y_test, full_probability)
    context_metrics = evaluate_predictions(y_test, context_probability)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    result = {
        "model": chosen,
        "full_features": {metric: full_metrics[metric] for metric in metric_names},
        "context_only": {metric: context_metrics[metric] for metric in metric_names},
        "difference_full_minus_context": {
            metric: full_metrics[metric] - context_metrics[metric] for metric in metric_names
        },
        "interpretation": "Predictive feature ablation only; not a causal experiment.",
    }
    with (METRICS_DIR / "ablation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    pd.DataFrame({
        "metric": metric_names,
        "Context-only": [context_metrics[metric] for metric in metric_names],
        "Full features": [full_metrics[metric] for metric in metric_names],
        "Difference": [result["difference_full_minus_context"][metric] for metric in metric_names],
    }).to_csv(METRICS_DIR / "ablation_metrics.csv", index=False)
    return result