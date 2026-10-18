import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.ablation_study import run_ablation
from src.config import FIGURES_DIR, METRICS_DIR, SHAP_DIR, TARGET_COLUMN, TARGET_MAPPING
from src.data_loader import load_dataset
from src.data_validation import validate_dataset
from src.model_manager import load_models, saved_metrics, train_all_models
from src.preprocessing import feature_frame
from src.scenario_analysis import run_scenario
from src.shap_explainer import explain_rows, global_importance
from src.utils import configure_logging


def command_eda(frame):
    sample = frame.sample(min(18000, len(frame)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    frame[TARGET_COLUMN].value_counts().plot.bar(ax=axes[0], color=["#167d78", "#df9a39"])
    axes[0].set_title("Observed satisfaction distribution")
    axes[0].set_ylabel("Passengers")
    pd.crosstab(sample["Class"], sample[TARGET_COLUMN], normalize="index").plot.bar(
        ax=axes[1], color=["#df9a39", "#167d78"])
    axes[1].set_title("Satisfaction share by class")
    axes[1].set_ylabel("Share")
    fig.tight_layout()
    output = FIGURES_DIR / "eda_overview.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"EDA figure saved: {output}")


def command_train(frame):
    metadata = train_all_models()
    ablation = run_ablation(frame, metadata["selected_model"])
    print(json.dumps({
        "selected_model": metadata["selected_model"],
        "selection_rule": metadata["selection_rule"],
        "cv_metrics": metadata["cv_metrics"],
        "test_metrics": metadata["test_metrics"],
        "ablation": ablation,
    }, indent=2))


def command_explain(frame):
    models, metadata = load_models()
    selected = metadata["selected_model"]
    rows = feature_frame(frame).sample(min(300, len(frame)), random_state=42).reset_index(drop=True)
    background = feature_frame(frame).sample(min(100, len(frame)), random_state=7).reset_index(drop=True)
    explanation = explain_rows(models[selected], rows, background)
    importance = global_importance(explanation)
    importance.to_csv(SHAP_DIR / "global_shap_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    top = importance.head(20).sort_values("mean_abs_shap")
    ax.barh(top["feature"], top["mean_abs_shap"], color="#167d78")
    ax.set_title(f"Mean absolute SHAP values · {selected}")
    ax.set_xlabel("Mean absolute SHAP value (model-output scale)")
    fig.tight_layout()
    fig.savefig(SHAP_DIR / "global_shap_importance.png", dpi=160)
    plt.close(fig)
    print(importance.head(15).to_string(index=False))


def command_scenario(frame):
    models, metadata = load_models()
    selected = metadata["selected_model"]
    passenger = feature_frame(frame.iloc[[0]]).reset_index(drop=True)
    feature = "Online Boarding"
    current = int(passenger.iloc[0][feature])
    result = run_scenario(models[selected], passenger, feature, 5 if current < 5 else 4)
    path = METRICS_DIR / "example_scenario.json"
    with path.open("w", encoding="utf-8") as output:
        json.dump({"model": selected, **result}, output, indent=2)
    print(json.dumps({"model": selected, **result}, indent=2))


def command_predict(frame, row_index):
    if row_index < 0 or row_index >= len(frame):
        raise ValueError(f"Row index must be between 0 and {len(frame) - 1}.")
    models, metadata = load_models()
    passenger = feature_frame(frame.iloc[[row_index]]).reset_index(drop=True)
    predictions = {}
    for name, model in models.items():
        probability = float(model.predict_proba(passenger)[:, 1][0])
        predictions[name] = {
            "satisfied_probability": probability,
            "prediction": "Satisfied" if probability >= 0.5 else "Neutral or Dissatisfied",
        }
    print(json.dumps({"row_index": row_index, "selected_model": metadata["selected_model"],
                      "models": predictions}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Airline passenger satisfaction research workflows")
    parser.add_argument("command", choices=["validate", "eda", "train", "evaluate", "explain", "scenario", "predict", "ablation"])
    parser.add_argument("--data", help="Path to an alternate dataset CSV")
    parser.add_argument("--row-index", type=int, default=0, help="Passenger row index for CLI prediction")
    args = parser.parse_args()
    configure_logging()
    frame = load_dataset(args.data)

    if args.command == "validate":
        print(json.dumps(validate_dataset(frame), indent=2))
    elif args.command == "eda":
        command_eda(frame)
    elif args.command == "train":
        command_train(frame)
    elif args.command == "evaluate":
        print(saved_metrics().to_string(float_format=lambda value: f"{value:.4f}"))
    elif args.command == "explain":
        command_explain(frame)
    elif args.command == "scenario":
        command_scenario(frame)
    elif args.command == "predict":
        command_predict(frame, args.row_index)
    elif args.command == "ablation":
        _, metadata = load_models()
        print(json.dumps(run_ablation(frame, metadata["selected_model"]), indent=2))


if __name__ == "__main__":
    main()