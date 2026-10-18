from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def create_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)