"""
Shared logic used by train.py, app.py and the tests:
data validation, feature engineering and the sklearn pipeline builder.

`add_features` lives at module level so the saved joblib pipeline can be
unpickled by app.py (pickle stores a reference to features.add_features).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "math_score"
CATEGORICAL = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
NUMERIC = ["reading_score", "writing_score"]
FEATURES = CATEGORICAL + NUMERIC
ENGINEERED = ["rw_avg", "rw_diff", "edu_ord"]

EDU_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]
ALLOWED = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": EDU_ORDER,
    "lunch": ["standard", "free/reduced"],
    "test_preparation_course": ["none", "completed"],
}
SCORE_RANGE = (0, 100)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the training data and return a cleaned copy. Raises ValueError."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[FEATURES + [TARGET]].copy()
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].isna() | ~df[col].between(*SCORE_RANGE)
        if bad.any():
            raise ValueError(f"{col}: {int(bad.sum())} values missing or outside {SCORE_RANGE}")
    for col, allowed in ALLOWED.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            raise ValueError(f"{col}: unexpected values {sorted(df.loc[bad, col].unique())}")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        print(f"[validate] dropping {n_dup} duplicate rows")
        df = df.drop_duplicates()
    return df.reset_index(drop=True)


def validate_input(row: dict) -> dict:
    """Validate one prediction request (from the web form or the API)."""
    clean = {}
    for col, allowed in ALLOWED.items():
        value = row.get(col)
        if value not in allowed:
            raise ValueError(f"Invalid value for {col}: {value!r}")
        clean[col] = value
    for col in NUMERIC:
        try:
            value = float(row.get(col))
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number")
        if not np.isfinite(value) or not SCORE_RANGE[0] <= value <= SCORE_RANGE[1]:
            raise ValueError(f"{col} must be between {SCORE_RANGE[0]} and {SCORE_RANGE[1]}")
        clean[col] = value
    return clean


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Engineered features:
    rw_avg   - mean of reading and writing (they correlate 0.95, averaging reduces noise)
    rw_diff  - reading minus writing (verbal-skill balance)
    edu_ord  - parental education as an ordered 0-5 number (unknown -> NaN -> imputed)
    """
    X = X.copy()
    X["rw_avg"] = (X["reading_score"] + X["writing_score"]) / 2
    X["rw_diff"] = X["reading_score"] - X["writing_score"]
    X["edu_ord"] = X["parental_level_of_education"].map({e: i for i, e in enumerate(EDU_ORDER)})
    return X


def build_pipeline(model, use_fe: bool = False) -> Pipeline:
    numeric_cols = NUMERIC + (ENGINEERED if use_fe else [])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, CATEGORICAL),
    ])
    steps = []
    if use_fe:
        steps.append(("features", FunctionTransformer(add_features, validate=False)))
    steps += [("preprocessor", preprocessor), ("model", model)]
    return Pipeline(steps)
