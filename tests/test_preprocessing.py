"""
tests/test_preprocessing.py
=============================
Phase 26 — Unit tests for src/preprocessing.py and src/split.py.

Run with: pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src import config
from src import preprocessing as pp
from src import split as sp


@pytest.fixture
def toy_patient_df():
    """A small synthetic patient-level feature table, including the
    leakage-risk metadata columns that must always be excluded."""
    np.random.seed(0)
    n = 60
    df = pd.DataFrame({
        config.PATIENT_ID_COL: [f"p{i}" for i in range(n)],
        config.SOURCE_SET_COL: ["A"] * n,
        "label": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "HR_mean": np.random.normal(85, 10, n),
        "Lactate_count": np.random.poisson(1, n).astype(float),
        "Gender": np.random.choice([0, 1], n).astype(float),
        "Unit1": np.random.choice([0, 1, -1], n).astype(float),
        "Unit2": np.random.choice([0, 1, -1], n).astype(float),
        "onset_hour": np.nan,
        "usable_hours": 6.0,
        "n_hours_used": 6.0,
    })
    # Introduce missing values to exercise the imputer
    df.loc[df.sample(10, random_state=1).index, "HR_mean"] = np.nan
    # Make onset_hour non-null exactly for positives (realistic leakage pattern)
    df.loc[df["label"] == 1, "onset_hour"] = 5.0
    df.loc[df["label"] == 1, "usable_hours"] = 3.0
    return df


class TestFeatureColumnSelection:
    def test_leakage_columns_excluded(self, toy_patient_df):
        feature_cols = pp.get_feature_columns(toy_patient_df)
        for leak_col in ["onset_hour", "usable_hours", "n_hours_used",
                          "label", config.PATIENT_ID_COL, config.SOURCE_SET_COL]:
            assert leak_col not in feature_cols

    def test_categorical_and_numeric_split_correctly(self, toy_patient_df):
        feature_cols = pp.get_feature_columns(toy_patient_df)
        numeric_cols, categorical_cols = pp.get_column_groups(feature_cols)
        assert "Gender" in categorical_cols
        assert "Unit1" in categorical_cols
        assert "HR_mean" in numeric_cols
        assert "Lactate_count" in numeric_cols


class TestPreprocessingPipeline:
    def test_fit_only_on_training_patients(self, toy_patient_df):
        train_ids = toy_patient_df[config.PATIENT_ID_COL].to_numpy()[:40]
        preprocessor, numeric_cols, categorical_cols = pp.fit_preprocessing_pipeline(
            toy_patient_df, train_ids
        )
        assert preprocessor is not None
        assert len(numeric_cols) > 0
        assert len(categorical_cols) > 0

    def test_no_nan_after_transform(self, toy_patient_df):
        train_ids = toy_patient_df[config.PATIENT_ID_COL].to_numpy()[:40]
        val_ids = toy_patient_df[config.PATIENT_ID_COL].to_numpy()[40:]

        preprocessor, numeric_cols, categorical_cols = pp.fit_preprocessing_pipeline(
            toy_patient_df, train_ids
        )
        X_train, y_train, _ = pp.transform_split(preprocessor, toy_patient_df, train_ids, numeric_cols, categorical_cols)
        X_val, y_val, _ = pp.transform_split(preprocessor, toy_patient_df, val_ids, numeric_cols, categorical_cols)

        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()

    def test_output_row_count_matches_input(self, toy_patient_df):
        train_ids = toy_patient_df[config.PATIENT_ID_COL].to_numpy()[:40]
        preprocessor, numeric_cols, categorical_cols = pp.fit_preprocessing_pipeline(
            toy_patient_df, train_ids
        )
        X_train, y_train, feat_names = pp.transform_split(
            preprocessor, toy_patient_df, train_ids, numeric_cols, categorical_cols
        )
        assert X_train.shape[0] == 40
        assert len(y_train) == 40
        assert X_train.shape[1] == len(feat_names)

    def test_feature_names_match_column_count(self, toy_patient_df):
        train_ids = toy_patient_df[config.PATIENT_ID_COL].to_numpy()
        preprocessor, numeric_cols, categorical_cols = pp.fit_preprocessing_pipeline(
            toy_patient_df, train_ids
        )
        X, y, feat_names = pp.transform_split(preprocessor, toy_patient_df, train_ids, numeric_cols, categorical_cols)
        assert X.shape[1] == len(feat_names), (
            "Feature name count must exactly match X's column count — a "
            "mismatch here previously caused a ValueError (see preprocessing.py fix)."
        )


class TestPatientLevelSplit:
    def test_no_patient_overlap_between_splits(self, toy_patient_df):
        splits = sp.patient_level_split(toy_patient_df, seed=42)
        overlaps = sp.verify_no_overlap(splits)
        assert overlaps["train_val"] == 0
        assert overlaps["train_test"] == 0
        assert overlaps["val_test"] == 0

    def test_all_patients_assigned_exactly_once(self, toy_patient_df):
        splits = sp.patient_level_split(toy_patient_df, seed=42)
        total_assigned = sum(len(v) for v in splits.values())
        assert total_assigned == len(toy_patient_df)

    def test_split_is_reproducible_with_same_seed(self, toy_patient_df):
        splits_a = sp.patient_level_split(toy_patient_df, seed=99)
        splits_b = sp.patient_level_split(toy_patient_df, seed=99)
        assert sorted(splits_a["train"]) == sorted(splits_b["train"])
        assert sorted(splits_a["val"]) == sorted(splits_b["val"])
        assert sorted(splits_a["test"]) == sorted(splits_b["test"])

    def test_different_seeds_produce_different_splits(self, toy_patient_df):
        splits_a = sp.patient_level_split(toy_patient_df, seed=1)
        splits_b = sp.patient_level_split(toy_patient_df, seed=2)
        assert sorted(splits_a["train"]) != sorted(splits_b["train"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
