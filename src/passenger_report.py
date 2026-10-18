from datetime import datetime


def build_passenger_report(data: dict) -> str:
    lines = [
        "EXPLAINABLE AIRLINE PASSENGER SATISFACTION REPORT",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "1. PASSENGER & SERVICE PROFILE",
    ]
    lines.extend(f"{key}: {value}" for key, value in data["profile"].items())
    lines.extend(["", "2. DATA QUALITY & VALIDATION", f"Status: {data['validation']['status']}"])
    lines.extend(f"{name}: {value}" for name, value in data["validation"]["items"].items())
    lines.extend(["", "3. MULTI-MODEL PREDICTION"])
    for name, result in data["predictions"].items():
        lines.append(f"{name}: {result['label']} (satisfied probability: {result['probability']:.2%})")
    final = data["final"]
    lines.extend([
        "", "4. FINAL MODEL DECISION", f"Model: {final['model']}",
        f"Prediction: {final['label']}", f"Satisfied probability: {final['probability']:.2%}",
        f"Selection rule: {final['selection_rule']}", "", "5. TOP SATISFACTION DRIVERS",
    ])
    lines.extend(data["drivers"] or ["NOT YET COMPUTED"])
    lines.extend(["", "6. PASSENGER-SPECIFIC EXPLANATION"])
    lines.extend(data["explanation"] or ["NOT YET COMPUTED"])
    scenario = data.get("scenario")
    lines.extend(["", "7. SERVICE SCENARIO ANALYSIS"])
    if scenario:
        lines.extend([
            f"{scenario['feature']}: {scenario['current_value']} -> {scenario['scenario_value']}",
            f"Satisfied probability: {scenario['current_probability']:.2%} -> {scenario['scenario_probability']:.2%}",
            f"Change: {scenario['change_percentage_points']:+.2f} percentage points",
            scenario["interpretation"],
        ])
    else:
        lines.append("NOT YET COMPUTED")
    lines.extend(["", "8. MODEL PERFORMANCE SUMMARY"])
    for name, metrics in data["metrics"].items():
        lines.append(f"{name}: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    lines.extend([
        "", "9. EXECUTIVE SUMMARY", data["summary"], "",
        "Research note: predictions are model estimates. Service scenarios are controlled model sensitivity analyses, not causal evidence.",
    ])
    return "\n".join(lines)