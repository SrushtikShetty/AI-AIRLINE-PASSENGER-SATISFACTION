import pandas as pd

from src.config import TARGET_COLUMN


def target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[TARGET_COLUMN].value_counts().rename_axis(TARGET_COLUMN).reset_index(name="passengers")


def satisfaction_by_category(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.crosstab(frame[column], frame[TARGET_COLUMN], normalize="index").reset_index()


def satisfaction_by_numeric_bin(frame: pd.DataFrame, column: str, bins: int = 12) -> pd.DataFrame:
    grouped = frame.assign(_bin=pd.cut(frame[column], bins=bins, duplicates="drop"))
    return pd.crosstab(grouped["_bin"], grouped[TARGET_COLUMN], normalize="index").reset_index()


def service_distribution(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].melt(var_name="service", value_name="rating")


def numeric_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number").drop(columns=["ID"], errors="ignore")
    return numeric.corr()