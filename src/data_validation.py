from typing import Any

import pandas as pd

from src.config import EXPECTED_FEATURES, ID_COLUMN, RATING_FEATURES, RATING_RANGES, TARGET_COLUMN, TARGET_MAPPING


def _check(name: str, passed: bool, detail: str, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "status": "pass" if passed else ("warning" if severity == "warning" else "fail"), "detail": detail}


def validate_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return structured schema and data-quality checks for the source frame."""
    required = set(EXPECTED_FEATURES + [TARGET_COLUMN, ID_COLUMN])
    missing_columns = sorted(required - set(frame.columns))
    checks = [
        _check("Required columns", not missing_columns, "Missing: " + (", ".join(missing_columns) or "none")),
        _check("Rows available", not frame.empty, f"{len(frame):,} rows"),
    ]
    if missing_columns:
        return {"valid": False, "rows": len(frame), "columns": len(frame.columns),
                "missing_values": int(frame.isna().sum().sum()), "duplicate_rows": int(frame.duplicated().sum()),
                "checks": checks}

    missing_count = int(frame.isna().sum().sum())
    checks.append(_check("Missing values", missing_count == 0, f"{missing_count:,} missing cells", "warning"))
    duplicate_count = int(frame.duplicated().sum())
    checks.append(_check("Duplicate rows", duplicate_count == 0, f"{duplicate_count:,} duplicate rows", "warning"))
    id_nulls = int(frame[ID_COLUMN].isna().sum())
    id_unique = frame[ID_COLUMN].nunique(dropna=True) == len(frame)
    checks.append(_check("ID uniqueness", id_nulls == 0 and id_unique,
                         f"{frame[ID_COLUMN].nunique(dropna=True):,} unique IDs; {id_nulls} null IDs"))

    observed_targets = set(frame[TARGET_COLUMN].dropna().astype(str).unique())
    invalid_targets = sorted(observed_targets - set(TARGET_MAPPING))
    checks.append(_check("Target labels", not invalid_targets and frame[TARGET_COLUMN].notna().all(),
                         f"Observed: {sorted(observed_targets)}; invalid: {invalid_targets}"))

    expected_categories = {
        "Gender": {"Male", "Female"}, "Customer Type": {"First-time", "Returning"},
        "Type of Travel": {"Business", "Personal"}, "Class": {"Business", "Economy", "Economy Plus"},
    }
    invalid_categories = {}
    for column, allowed in expected_categories.items():
        observed = set(frame[column].dropna().astype(str).unique())
        invalid = sorted(observed - allowed)
        if invalid:
            invalid_categories[column] = invalid
    checks.append(_check("Categorical values", not invalid_categories,
                         f"Invalid values: {invalid_categories or 'none'}"))

    invalid_ratings = {}
    for column in RATING_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        minimum, maximum = RATING_RANGES[column]
        count = int((values.isna() & frame[column].notna()).sum() +
                    (values.notna() & ~values.between(minimum, maximum)).sum())
        if count:
            invalid_ratings[column] = count
    checks.append(_check("Service rating ranges", not invalid_ratings,
                         f"Expected feature-specific observed ranges; invalid: {invalid_ratings or 'none'}"))

    numeric_issues = {}
    plausibility = {
        "Age": ~pd.to_numeric(frame["Age"], errors="coerce").between(1, 120),
        "Flight Distance": pd.to_numeric(frame["Flight Distance"], errors="coerce") < 0,
        "Departure Delay": pd.to_numeric(frame["Departure Delay"], errors="coerce") < 0,
        "Arrival Delay": pd.to_numeric(frame["Arrival Delay"], errors="coerce") < 0,
    }
    for column, invalid in plausibility.items():
        count = int(invalid.fillna(True).sum())
        if count:
            numeric_issues[column] = count
    checks.append(_check("Numeric plausibility", not numeric_issues,
                         f"Invalid values: {numeric_issues or 'none'}"))
    return {"valid": all(check["status"] != "fail" for check in checks), "rows": len(frame),
            "columns": len(frame.columns), "missing_values": missing_count,
            "duplicate_rows": duplicate_count, "checks": checks}


def encode_target(values: pd.Series) -> pd.Series:
    observed = set(values.dropna().astype(str).unique())
    unknown = observed - set(TARGET_MAPPING)
    if unknown or values.isna().any():
        raise ValueError(f"Unknown or missing target labels: {sorted(unknown)}")
    return values.map(TARGET_MAPPING).astype("int8")