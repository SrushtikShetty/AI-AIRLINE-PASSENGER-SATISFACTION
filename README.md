# Explainable Airline Passenger Satisfaction

A research-oriented, multi-model binary classification application for predicting whether an airline passenger is **Satisfied** or **Neutral / Dissatisfied**. The system uses the supplied cleaned CSV as its data source and presents observed dataset checks, holdout evaluation, SHAP explanations, feature ablation, and controlled service-scenario sensitivity in a Streamlit application.

Predictions and probabilities are model estimates. SHAP describes model-output contributions. Ablation and scenario analysis are predictive/sensitivity analyses and do not establish causal effects.

## Problem and Objectives

- Validate the supplied schema and data-quality constraints before training.
- Compare Logistic Regression, Random Forest, and XGBoost on identical stratified partitions and preprocessing.
- Select a default model using training-only cross-validation F1, with CV ROC-AUC as a tie-breaker.
- Explain global and passenger-specific model behavior with SHAP.
- Compare context-only versus full-feature prediction and simulate controlled one-feature changes.
- Demonstrate the complete workflow through a multi-page Streamlit application and CLI.

## Dataset

Place `airline_passenger_satisfaction_cleaned.csv` in `data/`. The configured dataset includes passenger profile, flight context, 14 service-rating fields, `ID`, and the `Satisfaction` target. In the supplied file, most service ratings range 0–5 and Baggage Handling ranges 1–5; validation and UI inputs honor those feature-specific bounds. `ID` is validated but excluded from model features. Target mapping is explicit: `Satisfied` → 1 and `Neutral or Dissatisfied` → 0; unknown labels fail validation.

## Architecture

```text
CSV → validation → train-only preprocessing → 3-fold CV model selection
                                  ↓
            Logistic Regression | Random Forest | XGBoost
                                  ↓
       holdout evaluation → SHAP → scenarios / ablation → report
                                  ↑
                  Streamlit UI and Python CLI
```

Preprocessing is built as a scikit-learn `ColumnTransformer` inside each `Pipeline`: numeric values are median-imputed and scaled; categorical values are most-frequent-imputed and one-hot encoded with unknown categories ignored. The pipeline is fitted only on training folds/data. The 80/20 stratified test partition is held out from model selection. CV uses three folds on a deterministic, stratified sample of up to 24,000 training rows to keep this demonstration practical. Small per-model grids tune logistic regularization strength, Random Forest feature subsampling, and XGBoost depth with CV F1; full training then refits each chosen configuration on all training rows.

## Models and Evaluation

- **Logistic Regression**: scaled linear baseline with balanced class weights.
- **Random Forest**: 160-tree ensemble with balanced subsampling.
- **XGBoost**: CPU histogram tree booster with restrained fixed hyperparameters.

Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and test probabilities are computed at runtime. Results are written to `reports/metrics/`; model pipelines and the selection metadata are written to `models/`. No metric is hard-coded. The final default is whichever model wins the documented CV selection rule on the current run.

## Explainability and Experiments

SHAP TreeExplainer is used for tree estimators and LinearExplainer for Logistic Regression. Global mean absolute SHAP values and local passenger contributions are generated from the fitted pipeline's transformed features. SHAP is computed on demand in the UI and can be exported by the CLI. Scenario analysis changes one service rating while keeping the remaining passenger fields fixed; this is model sensitivity, not a causal estimate. Ablation compares context-only and all features for the selected estimator on the same holdout split.

## Installation and Usage

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py validate
python main.py eda
python main.py train
streamlit run app.py
```

The `train` command performs model CV/fitting, writes holdout metrics and predictions, selects the default model, and runs the ablation experiment. CLI commands also include:

```powershell
python main.py evaluate
python main.py explain
python main.py scenario
python main.py predict --row-index 0
python main.py ablation
```

The app contains Dashboard, Dataset Explorer, EDA / Analysis, Passenger Prediction, Model Comparison, SHAP Explainability, Scenario Analysis, Feature Ablation, and Final Report sections. It loads saved artifacts and does not retrain when UI controls change.

## Tests

```powershell
pytest -q
```

Tests cover schema/target validation, preprocessing leakage boundaries, model fit/save/load/probability behavior, and scenario constraints. The larger end-to-end training run is performed separately through `python main.py train`.

## Project Structure

```text
app.py
main.py
data/airline_passenger_satisfaction_cleaned.csv
src/
  config.py, data_loader.py, data_validation.py, preprocessing.py
  model_manager.py, evaluate_models.py, ablation_study.py
  eda.py, shap_explainer.py, scenario_analysis.py, passenger_report.py
models/                 generated model pipelines and metadata
reports/metrics/        generated metrics, holdout predictions, experiments
reports/figures/        generated EDA charts
reports/shap/           generated SHAP artifacts
tests/                  pytest coverage
```

## Limitations and Future Work

The training-only CV selection sample is capped for runtime. Probabilities are not calibrated. The dataset is observational and may contain population, temporal, or sampling bias. SHAP is an explanation of model behavior, not a causal account. Service scenarios are controlled input perturbations and should not be interpreted as expected effects of airline interventions. Future work may include calibration, nested CV, subgroup/fairness evaluation, temporal validation, model monitoring, and additional estimators.