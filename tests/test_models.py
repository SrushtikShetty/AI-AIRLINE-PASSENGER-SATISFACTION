import joblib
from sklearn.base import clone

from src.model_manager import model_registry
from src.preprocessing import build_pipeline, feature_frame
from tests.helpers import sample_frame


def test_registered_models_fit_predict_and_round_trip(tmp_path):
    frame = sample_frame(60)
    X = feature_frame(frame)
    y = frame["Satisfaction"].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    for name, factory in model_registry().items():
        estimator = factory()
        if hasattr(estimator, "n_estimators"):
            estimator.set_params(n_estimators=8)
        pipeline = build_pipeline(X, estimator)
        pipeline.fit(X.iloc[:48], y.iloc[:48])
        assert pipeline.predict(X.iloc[48:]).shape == (12,)
        assert pipeline.predict_proba(X.iloc[48:]).shape == (12, 2)
        path = tmp_path / f"{name}.joblib"
        joblib.dump(pipeline, path)
        restored = joblib.load(path)
        assert restored.predict_proba(X.iloc[48:]).shape == (12, 2)