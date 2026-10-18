import pandas as pd

from src.config import RATING_FEATURES, RATING_RANGES


def run_scenario(model, passenger: pd.DataFrame, feature: str, value: int) -> dict:
    if feature not in RATING_FEATURES or feature not in passenger.columns:
        raise ValueError(f"Unsupported service feature: {feature}")
    minimum, maximum = RATING_RANGES[feature]
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{feature} must be between {minimum} and {maximum}, matching its observed dataset range.")
    if len(passenger) != 1:
        raise ValueError("Scenario analysis expects exactly one passenger row.")
    current_value = int(passenger.iloc[0][feature])
    current_probability = float(model.predict_proba(passenger)[:, 1][0])
    changed = passenger.copy()
    changed.loc[:, feature] = int(value)
    scenario_probability = float(model.predict_proba(changed)[:, 1][0])
    return {
        "feature": feature,
        "current_value": current_value,
        "scenario_value": int(value),
        "current_probability": current_probability,
        "scenario_probability": scenario_probability,
        "change_percentage_points": (scenario_probability - current_probability) * 100,
        "interpretation": "Controlled model sensitivity analysis; not evidence of a real-world causal effect.",
    }