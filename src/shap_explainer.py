import numpy as np
import pandas as pd


def _positive_class_values(values) -> np.ndarray:
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def explain_rows(pipeline, rows: pd.DataFrame, background: pd.DataFrame | None = None) -> dict:
    """Compute class-1 SHAP contributions for a fitted sklearn pipeline."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError("SHAP is not installed. Install dependencies with `pip install -r requirements.txt`.") from error

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(rows)
    feature_names = preprocessor.get_feature_names_out()
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    transformed_frame = pd.DataFrame(dense, columns=feature_names, index=rows.index)
    if estimator.__class__.__name__ in {"RandomForestClassifier", "XGBClassifier", "XGBRFClassifier"}:
        explainer = shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(transformed)
        base_value = explainer.expected_value
    else:
        background_rows = (background if background is not None else rows).head(100)
        background_transformed = preprocessor.transform(background_rows)
        explainer = shap.LinearExplainer(estimator, background_transformed)
        raw_values = explainer.shap_values(transformed)
        base_value = explainer.expected_value

    values = _positive_class_values(raw_values)
    base_array = np.asarray(base_value)
    base_value = float(base_array.ravel()[-1]) if base_array.ndim else float(base_array)
    contribution_frame = pd.DataFrame(values, columns=feature_names, index=rows.index)
    return {
        "values": contribution_frame,
        "transformed": transformed_frame,
        "base_value": base_value,
        "output_scale": "model output; SHAP values are not probabilities",
    }


def global_importance(explanation: dict) -> pd.DataFrame:
    return (explanation["values"].abs().mean().sort_values(ascending=False)
            .rename("mean_abs_shap").rename_axis("feature").reset_index())


def local_contributions(explanation: dict, position: int = 0) -> pd.DataFrame:
    values = explanation["values"].iloc[position]
    features = explanation["transformed"].iloc[position]
    result = pd.DataFrame({"feature": values.index, "value": features.to_numpy(), "shap_value": values.to_numpy()})
    return result.sort_values("shap_value", key=lambda series: series.abs(), ascending=False)