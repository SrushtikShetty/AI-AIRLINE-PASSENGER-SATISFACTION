import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ablation_study import run_ablation
from src.config import (
    CONTEXT_FEATURES, ID_COLUMN, METRICS_DIR, RATING_FEATURES, SERVICE_FEATURES,
    RATING_RANGES, TARGET_COLUMN, TARGET_LABELS,
)
from src.data_loader import load_dataset
from src.data_validation import validate_dataset
from src.eda import numeric_correlation, satisfaction_by_category, target_distribution
from src.model_manager import load_models, saved_metrics
from src.passenger_report import build_passenger_report
from src.preprocessing import feature_frame
from src.scenario_analysis import run_scenario
from src.shap_explainer import explain_rows, global_importance, local_contributions

st.set_page_config(page_title="Airline Satisfaction Research", page_icon="✈", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.4rem; max-width: 1500px;}
  [data-testid="stMetric"] {background: #f1f6f5; border-left: 3px solid #117c72; padding: 12px 14px;}
  .research-note {border-left: 3px solid #d28b28; padding: 8px 12px; background: #fff8e8;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading passenger dataset...")
def cached_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Loading trained models...")
def cached_models():
    return load_models()


@st.cache_data(show_spinner="Computing SHAP explanations...")
def cached_explanation(model_name, rows, background):
    models, _ = cached_models()
    return explain_rows(models[model_name], rows, background)


def predict_all(models, row):
    result = {}
    for name, model in models.items():
        probability = float(model.predict_proba(row)[:, 1][0])
        result[name] = {
            "probability": probability,
            "label": "Satisfied" if probability >= 0.5 else "Neutral or Dissatisfied",
        }
    return result


def clean_features(row):
    return feature_frame(row).reset_index(drop=True)


def model_feature_importance(model):
    estimator = model.named_steps["classifier"]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False).head(12))


def metric_columns(metrics):
    cols = st.columns(5)
    for col, key, label in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                               ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
        col.metric(label, f"{metrics[key]:.3f}")


def render_dashboard(frame, validation, models, metadata, metrics):
    st.title("Explainable Airline Passenger Satisfaction")
    st.caption("Multi-model prediction, SHAP interpretation, and controlled service-scenario analysis")
    selected = metadata["selected_model"]
    selected_metrics = metadata["test_metrics"][selected]
    cols = st.columns(5)
    cols[0].metric("Passengers", f"{len(frame):,}")
    cols[1].metric("Predictive features", len(feature_frame(frame).columns))
    cols[2].metric("Trained models", len(models))
    cols[3].metric("Selected model", selected)
    cols[4].metric("Holdout F1", f"{selected_metrics['f1']:.3f}")
    st.caption(f"Validation: {'VALID' if validation['valid'] else 'ISSUES FOUND'} · "
               f"{metadata['train_rows']:,} train / {metadata['test_rows']:,} test · "
               "final selection is based on training-only cross-validation.")

    left, right = st.columns(2)
    distribution = target_distribution(frame)
    left.plotly_chart(px.bar(distribution, x=TARGET_COLUMN, y="passengers", color=TARGET_COLUMN,
                             title="Observed satisfaction distribution", text_auto=True), use_container_width=True)
    compare = metrics.reset_index().melt(id_vars="model", value_vars=["accuracy", "f1", "roc_auc"],
                                         var_name="metric", value_name="score")
    right.plotly_chart(px.bar(compare, x="model", y="score", color="metric", barmode="group",
                              title="Holdout model comparison", range_y=[0, 1]), use_container_width=True)

    left, right = st.columns(2)
    importance = model_feature_importance(models[selected])
    left.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                             orientation="h", title=f"Top model features · {selected}"), use_container_width=True)
    predictions_path = METRICS_DIR / "test_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        passenger_probs = predictions[[name for name in models]].head(120).mean().rename_axis("model").reset_index(name="mean satisfied probability")
        right.plotly_chart(px.bar(passenger_probs, x="model", y="mean satisfied probability",
                                  title="Mean holdout satisfied probability", range_y=[0, 1]), use_container_width=True)
    st.subheader("System architecture")
    st.code("""CSV → Validation → Leakage-safe preprocessing → 3-fold CV model selection
                                           ↓
                             Logistic Regression | Random Forest | XGBoost
                                           ↓
                Holdout evaluation → SHAP explanations → Scenario analysis → Report""", language="text")


def render_explorer(frame):
    st.title("Dataset Explorer")
    columns = st.columns(4)
    columns[0].metric("Rows", f"{len(frame):,}")
    columns[1].metric("Columns", len(frame.columns))
    columns[2].metric("Missing cells", f"{int(frame.isna().sum().sum()):,}")
    columns[3].metric("Duplicate rows", f"{int(frame.duplicated().sum()):,}")
    with st.expander("Schema and data quality", expanded=True):
        schema = pd.DataFrame({"column": frame.columns, "dtype": frame.dtypes.astype(str).values,
                               "missing": frame.isna().sum().values,
                               "unique": frame.nunique(dropna=False).values})
        st.dataframe(schema, use_container_width=True, hide_index=True)
    with st.expander("Numerical summary"):
        st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)
    with st.expander("Categorical values"):
        categorical = frame.select_dtypes(include=["object", "str", "category"]).columns
        values = pd.DataFrame({
            "column": categorical,
            "observed values": [", ".join(map(str, frame[column].dropna().unique())) for column in categorical],
        })
        st.dataframe(values, use_container_width=True, hide_index=True)
    search = st.text_input("Search values in preview", placeholder="e.g. Business, Satisfied")
    preview = frame
    if search:
        mask = frame.astype("string").apply(lambda column: column.str.contains(search, case=False, na=False)).any(axis=1)
        preview = frame.loc[mask]
    st.caption(f"Showing {min(1000, len(preview)):,} of {len(preview):,} matching rows")
    st.dataframe(preview.head(1000), use_container_width=True, hide_index=True)


def render_eda(frame):
    st.title("Exploratory Data Analysis")
    sample = frame.sample(min(len(frame), 18000), random_state=42)
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(sample, x=TARGET_COLUMN, color=TARGET_COLUMN,
                                   title="Satisfaction class counts", text_auto=True), use_container_width=True)
    categorical = st.selectbox("Compare satisfaction by", ["Customer Type", "Type of Travel", "Class"])
    grouped = satisfaction_by_category(frame, categorical).melt(id_vars=categorical,
                 var_name="Satisfaction", value_name="share")
    right.plotly_chart(px.bar(grouped, x=categorical, y="share", color="Satisfaction", barmode="group",
                              title=f"Satisfaction share by {categorical}", range_y=[0, 1]), use_container_width=True)
    selected_service = st.selectbox("Service rating", SERVICE_FEATURES, index=SERVICE_FEATURES.index("Online Boarding"))
    left, right = st.columns(2)
    left.plotly_chart(px.box(sample, x=TARGET_COLUMN, y=selected_service, color=TARGET_COLUMN,
                             title=f"{selected_service} by satisfaction"), use_container_width=True)
    right.plotly_chart(px.box(sample, x=TARGET_COLUMN, y="Flight Distance", color=TARGET_COLUMN,
                              title="Flight distance by satisfaction"), use_container_width=True)
    delay = st.selectbox("Delay measure", ["Departure Delay", "Arrival Delay"])
    st.plotly_chart(px.box(sample, x=TARGET_COLUMN, y=delay, color=TARGET_COLUMN,
                           title=f"{delay} by satisfaction", points=False), use_container_width=True)
    st.plotly_chart(px.histogram(sample, x=selected_service, color=TARGET_COLUMN, barmode="group",
                                 title=f"{selected_service} rating distribution", nbins=6), use_container_width=True)
    corr = numeric_correlation(frame)
    encoded_target = frame[TARGET_COLUMN].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    corr[TARGET_COLUMN] = frame.select_dtypes(include="number").drop(columns=[ID_COLUMN], errors="ignore").corrwith(encoded_target)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Numeric feature correlation"), use_container_width=True)


def render_prediction(frame, models, metadata, metrics):
    st.title("Passenger Prediction")
    st.caption("Enter profile, flight context, and observed-scale service ratings. Passenger ID is excluded.")
    choices = {}
    for column in ("Gender", "Customer Type", "Type of Travel", "Class"):
        choices[column] = sorted(frame[column].dropna().unique().tolist())
    with st.form("passenger_form"):
        st.subheader("Passenger and flight profile")
        cols = st.columns(3)
        row = {}
        for index, column in enumerate(CONTEXT_FEATURES):
            col = cols[index % 3]
            if column in choices:
                row[column] = col.selectbox(column, choices[column], key=f"input_{column}")
            else:
                low, high = int(frame[column].min()), int(frame[column].max())
                row[column] = col.number_input(column, min_value=low, max_value=high,
                    value=int(frame[column].median()), step=1, key=f"input_{column}")
        st.subheader("Service ratings")
        cols = st.columns(3)
        for index, column in enumerate(SERVICE_FEATURES):
            minimum, maximum = RATING_RANGES[column]
            row[column] = cols[index % 3].slider(column, min_value=minimum, max_value=maximum,
                                                  value=int(frame[column].median()), key=f"input_{column}")
        selected = st.selectbox("Final model for this prediction", list(models),
                                index=list(models).index(metadata["selected_model"]))
        submitted = st.form_submit_button("Run all models", type="primary")
    if submitted:
        passenger = pd.DataFrame([row], columns=metadata["features"])
        results = predict_all(models, passenger)
        st.session_state["passenger"] = passenger
        st.session_state["prediction_results"] = results
        st.session_state["prediction_model"] = selected
    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]
        selected = st.session_state.get("prediction_model", metadata["selected_model"])
        st.subheader("Multi-model results")
        cols = st.columns(len(results))
        for col, (name, result) in zip(cols, results.items()):
            col.metric(name, result["label"], f"{result['probability']:.2%} satisfied probability")
        final = results[selected]
        st.success(f"Final model: {selected} · {final['label']} · Satisfied probability {final['probability']:.2%}")
        st.caption("Probabilities are model-estimated outputs; they are not automatically calibrated confidence scores.")


def render_comparison(models, metadata, metrics):
    st.title("Model Comparison")
    st.caption("Metrics below are computed on the stratified holdout set, not used for model selection.")
    display = metrics[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_mean_f1"]].rename(
        columns={"roc_auc": "ROC-AUC", "cv_mean_f1": "CV mean F1"})
    st.dataframe(display.style.format("{:.4f}"), use_container_width=True)
    metric = st.selectbox("Metric chart", ["accuracy", "precision", "recall", "f1", "roc_auc"])
    st.plotly_chart(px.bar(metrics.reset_index(), x="model", y=metric, color="model", range_y=[0, 1],
                           title=f"Holdout {metric.replace('_', ' ').upper()}"), use_container_width=True)
    predictions_path = METRICS_DIR / "test_predictions.csv"
    if not predictions_path.exists():
        st.info("Test predictions are not yet computed.")
        return
    pred = pd.read_csv(predictions_path)
    chosen = st.selectbox("Inspect model", list(models), key="comparison_model")
    y_true = pred["actual"].to_numpy()
    probability = pred[chosen].to_numpy()
    from sklearn.metrics import confusion_matrix, roc_curve
    cm = confusion_matrix(y_true, probability >= 0.5, labels=[0, 1])
    left, right = st.columns(2)
    left.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted 0", "Predicted 1"],
                               y=["Actual 0", "Actual 1"], color_continuous_scale="Blues",
                               title=f"{chosen} confusion matrix"), use_container_width=True)
    false_positive, true_positive, _ = roc_curve(y_true, probability)
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=false_positive, y=true_positive, mode="lines", name=chosen))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No-skill reference", line={"dash": "dash"}))
    roc.update_layout(title=f"{chosen} ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    right.plotly_chart(roc, use_container_width=True)
    st.info(f"Default final model from training-only CV: {metadata['selected_model']}. {metadata['selection_rule']}")


def render_shap(frame, models, metadata):
    st.title("SHAP Explainability")
    selected = st.selectbox("Model to explain", list(models),
                            index=list(models).index(st.session_state.get("prediction_model", metadata["selected_model"])))
    st.caption("SHAP values describe contributions on the explainer's model-output scale, not probability points.")
    tabs = st.tabs(["Global explanation", "Passenger explanation"])
    with tabs[0]:
        sample = feature_frame(frame).sample(min(300, len(frame)), random_state=7).reset_index(drop=True)
        background = feature_frame(frame).sample(min(100, len(frame)), random_state=17).reset_index(drop=True)
        try:
            explanation = cached_explanation(selected, sample, background)
            importance = global_importance(explanation).head(20)
            st.plotly_chart(px.bar(importance.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                                   orientation="h", title="Mean absolute SHAP value"), use_container_width=True)
            summary_features = importance.head(12)["feature"].tolist()
            summary = explanation["values"][summary_features].stack().rename("SHAP value").reset_index()
            summary.columns = ["passenger", "feature", "SHAP value"]
            summary["feature value"] = explanation["transformed"][summary_features].stack().to_numpy()
            st.plotly_chart(px.scatter(summary, x="SHAP value", y="feature", color="feature value",
                                       color_continuous_scale="RdBu_r", title="SHAP summary plot",
                                       render_mode="webgl"), use_container_width=True)
            st.dataframe(importance, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Global SHAP calculation failed: {error}")
    with tabs[1]:
        if "passenger" in st.session_state:
            use_latest = st.checkbox("Use latest passenger prediction", value=True)
        else:
            use_latest = False
        if use_latest:
            passenger = st.session_state["passenger"]
        else:
            row_index = st.number_input("Dataset row", min_value=0, max_value=len(frame) - 1, value=0, step=1)
            passenger = feature_frame(frame.iloc[[int(row_index)]]).reset_index(drop=True)
        result = predict_all(models, passenger)[selected]
        st.write(f"Prediction: **{result['label']}** · satisfied probability: **{result['probability']:.2%}**")
        background = feature_frame(frame).sample(min(100, len(frame)), random_state=17).reset_index(drop=True)
        try:
            explanation = cached_explanation(selected, passenger, background)
            contributions = local_contributions(explanation).head(16)
            st.plotly_chart(go.Figure(go.Waterfall(
                orientation="h", y=contributions["feature"], x=contributions["shap_value"],
                measure=["relative"] * len(contributions), increasing={"marker": {"color": "#168276"}},
                decreasing={"marker": {"color": "#c45443"}},
            )).update_layout(title="Largest passenger-specific SHAP contributions",
                             xaxis_title="SHAP contribution (model-output scale)"), use_container_width=True)
            positive = contributions[contributions["shap_value"] > 0].head(6)
            negative = contributions[contributions["shap_value"] < 0].head(6)
            left, right = st.columns(2)
            left.markdown("**Contributions toward satisfied output**")
            left.dataframe(positive, hide_index=True, use_container_width=True)
            right.markdown("**Contributions away from satisfied output**")
            right.dataframe(negative, hide_index=True, use_container_width=True)
        except Exception as error:
            st.error(f"Passenger SHAP calculation failed: {error}")


def render_scenarios(frame, models, metadata):
    st.title("Service Scenario Analysis")
    st.markdown('<div class="research-note">Controlled model sensitivity only. This does not establish that changing a real airline service causes the same change in satisfaction.</div>', unsafe_allow_html=True)
    if "passenger" in st.session_state:
        use_latest = st.checkbox("Use latest passenger prediction", value=True)
    else:
        use_latest = False
    if use_latest:
        passenger = st.session_state["passenger"].copy()
    else:
        row_index = st.number_input("Dataset passenger row", 0, len(frame) - 1, 0, key="scenario_row")
        passenger = feature_frame(frame.iloc[[int(row_index)]]).reset_index(drop=True)
    selected = st.selectbox("Scenario model", list(models), index=list(models).index(metadata["selected_model"]))
    baseline = float(models[selected].predict_proba(passenger)[:, 1][0])
    st.metric("Current satisfied probability", f"{baseline:.2%}")
    scenarios = []
    default_features = ["Online Boarding", "In-flight Wifi Service", "Seat Comfort"]
    for index in range(3):
        left, middle, right = st.columns([2, 1, 1])
        default_index = RATING_FEATURES.index(default_features[index])
        feature = left.selectbox(f"Service feature {index + 1}", RATING_FEATURES,
                                 index=default_index, key=f"scenario_feature_{index}")
        current = int(passenger.iloc[0][feature])
        minimum, maximum = RATING_RANGES[feature]
        middle.caption(f"Current: {current}")
        scenario_default = current + 1 if current < maximum else current - 1
        value = right.slider(f"Scenario value {index + 1}", minimum, maximum, scenario_default,
                             key=f"scenario_value_{index}")
        scenarios.append((feature, value))
    if st.button("Run scenario comparison", type="primary"):
        output = []
        for label, (feature, value) in zip(["Scenario A", "Scenario B", "Scenario C"], scenarios):
            result = run_scenario(models[selected], passenger, feature, value)
            result["scenario"] = label
            output.append(result)
        st.session_state["scenario_results"] = output
        st.session_state["scenario_signature"] = (selected, tuple(scenarios), tuple(passenger.iloc[0].items()))
        st.session_state["report_scenario"] = output[0]
    signature = (selected, tuple(scenarios), tuple(passenger.iloc[0].items()))
    if st.session_state.get("scenario_signature") == signature:
        output = st.session_state["scenario_results"]
        chart = pd.DataFrame([{"Scenario": "Current", "Satisfied probability": baseline}] + [
            {"Scenario": item["scenario"], "Satisfied probability": item["scenario_probability"]} for item in output
        ])
        st.plotly_chart(px.bar(chart, x="Scenario", y="Satisfied probability", range_y=[0, 1],
                               title="Baseline and one-feature-at-a-time model outputs"), use_container_width=True)
        table = pd.DataFrame([{
            "scenario": item["scenario"], "feature": item["feature"],
            "change": f"{item['current_value']} → {item['scenario_value']}",
            "scenario probability": item["scenario_probability"],
            "change (percentage points)": item["change_percentage_points"],
        } for item in output])
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_ablation(frame):
    st.title("Feature Ablation Study")
    st.caption("Context-only versus full features; predictive comparison, not a causal experiment.")
    path = METRICS_DIR / "ablation_metrics.json"
    if not path.exists():
        st.info("NOT YET COMPUTED. Run `python main.py ablation` after training.")
        return
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        rows.append({"metric": metric, "Context-only": result["context_only"][metric],
                     "Full features": result["full_features"][metric],
                     "Difference": result["difference_full_minus_context"][metric]})
    comparison = pd.DataFrame(rows)
    st.write(f"Selected estimator: **{result['model']}**")
    st.dataframe(comparison.style.format({column: "{:.4f}" for column in comparison.columns[1:]}),
                 use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(comparison.melt(id_vars="metric", var_name="feature set", value_name="score"),
                           x="metric", y="score", color="feature set", barmode="group", range_y=[0, 1],
                           title="Heldout ablation comparison"), use_container_width=True)
    st.info(result["interpretation"])


def render_report(frame, models, metadata, metrics, validation):
    st.title("Final Integrated Passenger Report")
    if "passenger" in st.session_state:
        use_latest = st.checkbox("Use latest passenger prediction", value=True)
    else:
        use_latest = False
    if use_latest:
        passenger = st.session_state["passenger"].copy()
    else:
        row_index = st.number_input("Passenger dataset row", min_value=0, max_value=len(frame) - 1,
                                    value=0, step=1, key="report_row")
        passenger = feature_frame(frame.iloc[[int(row_index)]]).reset_index(drop=True)
    results = predict_all(models, passenger)
    selected = st.session_state.get("prediction_model", metadata["selected_model"])
    final = results[selected]
    explanation_lines = []
    drivers = []
    try:
        background = feature_frame(frame).sample(min(100, len(frame)), random_state=17).reset_index(drop=True)
        explanation = cached_explanation(selected, passenger, background)
        contributions = local_contributions(explanation)
        global_rows = feature_frame(frame).sample(min(300, len(frame)), random_state=7).reset_index(drop=True)
        global_explanation = cached_explanation(selected, global_rows, background)
        drivers = [f"{row.feature}: mean |SHAP|={row.mean_abs_shap:.5f}" for row in
               global_importance(global_explanation).head(5).itertuples()]
        explanation_lines = [f"{row.feature}: SHAP {row.shap_value:+.5f} (encoded value {row.value:g})"
                             for row in contributions.head(6).itertuples()]
    except Exception as error:
        explanation_lines = [f"SHAP unavailable: {error}"]
    validation_items = {
        "Rows": f"{validation['rows']:,}", "Columns": validation["columns"],
        "Missing values": validation["missing_values"], "Duplicate rows": validation["duplicate_rows"],
    }
    for check in validation["checks"]:
        validation_items[check["name"]] = f"{check['status'].upper()}: {check['detail']}"
    profile = {column: passenger.iloc[0][column] for column in passenger.columns}
    report_data = {
        "profile": profile,
        "validation": {"status": "VALID" if validation["valid"] else "ISSUES FOUND", "items": validation_items},
        "predictions": results,
        "final": {"model": selected, **final, "selection_rule": metadata["selection_rule"]},
        "drivers": drivers,
        "explanation": explanation_lines,
        "scenario": st.session_state.get("report_scenario"),
        "metrics": metadata["test_metrics"],
        "summary": f"{selected} predicts {final['label']} with a model-estimated satisfied probability of {final['probability']:.2%}. This is predictive, not causal evidence.",
    }
    report = build_passenger_report(report_data)
    st.text_area("Generated report", report, height=620)
    st.download_button("Download report as TXT", report, file_name="airline_passenger_report.txt", mime="text/plain")


def main():
    st.sidebar.title("Passenger Satisfaction")
    page = st.sidebar.radio("Navigate", [
        "Dashboard", "Dataset Explorer", "EDA / Analysis", "Passenger Prediction",
        "Model Comparison", "SHAP Explainability", "Scenario Analysis",
        "Feature Ablation", "Final Report",
    ])
    try:
        frame = cached_dataset()
        validation = validate_dataset(frame)
        models, metadata = cached_models()
        metrics = saved_metrics()
    except Exception as error:
        st.error(f"Application data or trained artifacts could not be loaded: {error}")
        st.info("Install requirements, then run `python main.py train` and `python main.py ablation`.")
        st.stop()
    if page == "Dashboard":
        render_dashboard(frame, validation, models, metadata, metrics)
    elif page == "Dataset Explorer":
        render_explorer(frame)
    elif page == "EDA / Analysis":
        render_eda(frame)
    elif page == "Passenger Prediction":
        render_prediction(frame, models, metadata, metrics)
    elif page == "Model Comparison":
        render_comparison(models, metadata, metrics)
    elif page == "SHAP Explainability":
        render_shap(frame, models, metadata)
    elif page == "Scenario Analysis":
        render_scenarios(frame, models, metadata)
    elif page == "Feature Ablation":
        render_ablation(frame)
    else:
        render_report(frame, models, metadata, metrics, validation)


if __name__ == "__main__":
    main()