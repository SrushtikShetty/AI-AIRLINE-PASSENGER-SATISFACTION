from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


def create_model():
    return RandomForestClassifier(
        n_estimators=160, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )