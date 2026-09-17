"""
preprocessing.py
==================
Phase 11 — Preprocessing pipeline for the Early Sepsis Detection project.

CRITICAL LEAKAGE GUARD (read before anything else):
The patient-level dataset (from Phase 8) carries several METADATA columns
that were needed to CONSTRUCT the leakage-safe feature window, but which
themselves leak the label almost perfectly if used as model inputs:

  - `onset_hour`   : NaN for every negative patient, a real number for
                      every positive patient. "Is this NaN?" alone would
                      let a model predict the label with near-100% accuracy
                      — this is not a real clinical signal, it's an
                      artifact of how the dataset was constructed.
  - `usable_hours` / `n_hours_used` : always 6 for negative patients, but
                      1-5 for positive patients whose sepsis onset happened
                      early (see Phase 7). Strongly correlated with the
                      label for the same structural reason.

These three columns (plus identifiers `patient_id`, `source_set`, and the
`label` column itself) are explicitly excluded from the feature matrix by
`get_feature_columns()` below. They remain in the saved patient-level
parquet for auditing/debugging, but must NEVER reach `X` in any model.

Fit/leakage rule: the imputer and scaler in this pipeline are fit ONLY on
the training split (Phase 10) — never on validation or test data.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

logger = config.get_logger(__name__)


# ------------------------------------------------------------------
# Column classification
# ------------------------------------------------------------------
# Metadata / leakage-risk columns that must NEVER be used as model features.
NON_FEATURE_COLS = [
    config.PATIENT_ID_COL,
    config.SOURCE_SET_COL,
    "label",
    "onset_hour",      # LEAKAGE: NaN <=> negative, non-null <=> positive
    "usable_hours",    # LEAKAGE: correlated with label via window construction
    "n_hours_used",    # LEAKAGE: same as usable_hours in this pipeline
]

# Categorical columns: encoded as small integers/sentinels (not truly
# ordinal), so they are one-hot encoded rather than scaled. Unit1_known/
# Unit2_known are already clean binary flags and are treated as numeric
# pass-through (no encoding needed for a 0/1 flag).
CATEGORICAL_COLS = ["Gender", "Unit1", "Unit2"]


def get_feature_columns(patient_df: pd.DataFrame) -> list[str]:
    """Return all columns EXCEPT the identifier/label/leakage-risk metadata columns."""
    return [c for c in patient_df.columns if c not in NON_FEATURE_COLS]


def get_column_groups(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split feature_cols into (numeric_cols, categorical_cols) for the ColumnTransformer."""
    categorical = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]
    return numeric, categorical


# ------------------------------------------------------------------
# Pipeline construction
# ------------------------------------------------------------------
def build_preprocessing_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build (but do not fit) a scikit-learn ColumnTransformer:
      - numeric columns: median imputation + standard scaling
      - categorical columns: most-frequent imputation + one-hot encoding

    Median imputation (not mean) is used for numeric clinical features
    since many are right-skewed (e.g. Lactate, Bilirubin) — the median is
    more robust to the extreme values documented in Phase 4's EDA.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])
    return preprocessor


def fit_preprocessing_pipeline(
    patient_df: pd.DataFrame, train_patient_ids: np.ndarray
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Fit the preprocessing pipeline using ONLY rows whose patient_id is in
    train_patient_ids. Returns the fitted pipeline plus the numeric/
    categorical column lists it was fit on (needed to reconstruct output
    column names later, since ColumnTransformer output is a plain array).
    """
    feature_cols = get_feature_columns(patient_df)
    numeric_cols, categorical_cols = get_column_groups(feature_cols)

    train_df = patient_df[patient_df[config.PATIENT_ID_COL].isin(train_patient_ids)]
    logger.info("Fitting preprocessing pipeline on %d training patients "
                "(%d numeric + %d categorical features).",
                len(train_df), len(numeric_cols), len(categorical_cols))

    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    preprocessor.fit(train_df[feature_cols])

    return preprocessor, numeric_cols, categorical_cols


def transform_split(
    preprocessor: ColumnTransformer,
    patient_df: pd.DataFrame,
    patient_ids: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Transform a given split (identified by patient_ids) using an ALREADY-
    FITTED preprocessor. Returns (X, y, output_feature_names).
    """
    sub = patient_df[patient_df[config.PATIENT_ID_COL].isin(patient_ids)].copy()
    feature_cols = numeric_cols + categorical_cols

    X = preprocessor.transform(sub[feature_cols])
    y = sub["label"].to_numpy()

    # Use the ColumnTransformer's own get_feature_names_out(), which
    # correctly delegates through each inner Pipeline (imputer -> scaler /
    # imputer -> one-hot) regardless of how many categories or columns
    # actually survived fitting. Manually reconstructing names via
    # named_transformers_["categorical"]...get_feature_names_out(categorical_cols)
    # breaks if SimpleImputer silently drops an all-missing column during
    # fit (a real edge case: a categorical column with zero non-missing
    # values in a given training slice) — this is more robust.
    output_names = list(preprocessor.get_feature_names_out())
    # Strip the ColumnTransformer's "numeric__"/"categorical__" prefixes
    # for readability, matching this project's existing naming convention.
    output_names = [name.split("__", 1)[-1] for name in output_names]

    return X, y, output_names


def save_pipeline(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> None:
    payload = {"preprocessor": preprocessor, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}
    joblib.dump(payload, config.MODELS_DIR / "preprocessing_pipeline.pkl")
    logger.info("Saved preprocessing pipeline to %s", config.MODELS_DIR / "preprocessing_pipeline.pkl")


def load_pipeline() -> dict:
    return joblib.load(config.MODELS_DIR / "preprocessing_pipeline.pkl")


if __name__ == "__main__":
    patient_df = pd.read_parquet(config.PROCESSED_PATIENT_LEVEL_PARQUET)
    split_lookup = pd.read_parquet(config.PROCESSED_DIR / "patient_split_assignment.parquet")

    train_ids = split_lookup.loc[split_lookup["split"] == "train", config.PATIENT_ID_COL].to_numpy()
    val_ids = split_lookup.loc[split_lookup["split"] == "val", config.PATIENT_ID_COL].to_numpy()

    preprocessor, numeric_cols, categorical_cols = fit_preprocessing_pipeline(patient_df, train_ids)
    X_train, y_train, feat_names = transform_split(preprocessor, patient_df, train_ids, numeric_cols, categorical_cols)
    X_val, y_val, _ = transform_split(preprocessor, patient_df, val_ids, numeric_cols, categorical_cols)

    print("X_train:", X_train.shape, "X_val:", X_val.shape, "n_feature_names:", len(feat_names))
    save_pipeline(preprocessor, numeric_cols, categorical_cols)
