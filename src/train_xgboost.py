from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def create_model():
    return XGBClassifier(
        n_estimators=180, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        n_jobs=4, random_state=RANDOM_STATE, verbosity=0,
    )