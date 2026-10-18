from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, TARGET_COLUMN


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = [column for column in (ID_COLUMN, TARGET_COLUMN) if column in frame.columns]
    return frame.drop(columns=excluded).copy()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
                             remainder="drop")


def build_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(features)), ("classifier", estimator)])