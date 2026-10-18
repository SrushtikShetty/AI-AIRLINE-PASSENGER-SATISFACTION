from src.preprocessing import build_pipeline, feature_frame
from sklearn.linear_model import LogisticRegression

from tests.helpers import sample_frame


def test_pipeline_drops_id_and_predicts_probabilities():
    frame = sample_frame()
    X = feature_frame(frame)
    y = frame["Satisfaction"].map({"Neutral or Dissatisfied": 0, "Satisfied": 1})
    assert "ID" not in X.columns
    pipeline = build_pipeline(X, LogisticRegression(max_iter=300))
    pipeline.fit(X.iloc[:30], y.iloc[:30])
    probability = pipeline.predict_proba(X.iloc[30:])
    assert probability.shape == (10, 2)
    assert ((probability >= 0) & (probability <= 1)).all()