import numpy as np
import pytest

from src.preprocessing import build_pipeline, feature_frame
from src.scenario_analysis import run_scenario
from sklearn.linear_model import LogisticRegression
from tests.helpers import sample_frame


def test_scenario_changes_only_selected_service_feature():
    frame = sample_frame()
    X = feature_frame(frame)
    y = frame["Satisfaction"].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    model = build_pipeline(X, LogisticRegression(max_iter=300)).fit(X, y)
    passenger = X.iloc[[0]].copy()
    result = run_scenario(model, passenger, "Online Boarding", 5)
    assert result["current_value"] == passenger.iloc[0]["Online Boarding"]
    assert 0 <= result["current_probability"] <= 1
    assert 0 <= result["scenario_probability"] <= 1
    assert result["scenario_value"] == 5
    assert np.isclose(result["change_percentage_points"],
                      (result["scenario_probability"] - result["current_probability"]) * 100)


def test_scenario_rejects_invalid_rating():
    frame = sample_frame()
    X = feature_frame(frame)
    y = frame["Satisfaction"].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    model = build_pipeline(X, LogisticRegression(max_iter=300)).fit(X, y)
    with pytest.raises(ValueError, match="between 0 and 5"):
        run_scenario(model, X.iloc[[0]], "Online Boarding", 8)


def test_scenario_respects_baggage_observed_minimum():
    frame = sample_frame()
    X = feature_frame(frame)
    y = frame["Satisfaction"].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    model = build_pipeline(X, LogisticRegression(max_iter=300)).fit(X, y)
    with pytest.raises(ValueError, match="between 1 and 5"):
        run_scenario(model, X.iloc[[0]], "Baggage Handling", 0)