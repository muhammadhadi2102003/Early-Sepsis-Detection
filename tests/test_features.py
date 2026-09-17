"""
tests/test_features.py
========================
Phase 26 — Unit tests for src/feature_engineering.py and src/target.py.

Run with: pytest tests/test_features.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src import config
from src import target as tgt
from src import feature_engineering as fe


@pytest.fixture
def toy_raw_df():
    """A tiny, hand-built raw long-format dataframe covering key edge cases:
    a negative patient, a positive patient with early onset (short window),
    and a positive patient whose onset_hour <= 1 (should be excluded)."""
    rows = []
    # Negative patient: full 6-hour window
    for h in range(1, 7):
        rows.append({"patient_id": "neg1", "source_set": "A", "ICULOS": h,
                     "HR": 80 + h, "SBP": 120, "DBP": 80, "SaO2": 95, "FiO2": 0.21,
                     "Temp": 37.0, "Lactate": np.nan, "SepsisLabel": 0})
    # Positive patient: onset at hour 5 -> usable_hours = min(6, 4) = 4
    for h in range(1, 9):
        rows.append({"patient_id": "pos_early", "source_set": "A", "ICULOS": h,
                     "HR": 90 + h, "SBP": 110, "DBP": 70, "SaO2": 94, "FiO2": 0.25,
                     "Temp": 37.5, "Lactate": 2.5 if h == 2 else np.nan,
                     "SepsisLabel": 1 if h >= 5 else 0})
    # Positive patient: onset at hour 1 -> should be EXCLUDED entirely
    for h in range(1, 4):
        rows.append({"patient_id": "pos_too_early", "source_set": "B", "ICULOS": h,
                     "HR": 100, "SBP": 100, "DBP": 60, "SaO2": 90, "FiO2": 0.4,
                     "Temp": 38.0, "Lactate": np.nan, "SepsisLabel": 1})

    df = pd.DataFrame(rows)
    for c in config.ALL_CLINICAL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["Age", "Gender", "Unit1", "Unit2", "Unit1_known", "Unit2_known", "HospAdmTime"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


class TestTargetConstruction:
    def test_onset_hour_computed_correctly(self, toy_raw_df):
        onset = tgt.get_onset_hour(toy_raw_df)
        assert onset["pos_early"] == 5
        assert onset["pos_too_early"] == 1
        assert "neg1" not in onset.index  # negative patients have no onset

    def test_label_monotonicity_holds(self, toy_raw_df):
        bad = tgt.verify_label_monotonicity(toy_raw_df)
        assert len(bad) == 0

    def test_window_table_excludes_onset_at_hour_1(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        row = table[table["patient_id"] == "pos_too_early"].iloc[0]
        assert row["included"] == False
        assert "onset_hour" in row["exclusion_reason"]

    def test_window_table_truncates_early_onset_correctly(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        row = table[table["patient_id"] == "pos_early"].iloc[0]
        assert row["included"] == True
        assert row["usable_hours"] == 4  # min(6, onset_hour-1) = min(6, 4)

    def test_negative_patient_gets_full_window(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        row = table[table["patient_id"] == "neg1"].iloc[0]
        assert row["included"] == True
        assert row["usable_hours"] == 6

    def test_no_leakage_after_slicing(self, toy_raw_df):
        """The core leakage guarantee: no included patient's sliced data
        should contain any hour at or after their onset hour."""
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)

        onset = tgt.get_onset_hour(toy_raw_df)
        for pid in sliced["patient_id"].unique():
            patient_sliced = sliced[sliced["patient_id"] == pid]
            if pid in onset.index:
                assert patient_sliced["ICULOS"].max() < onset[pid], (
                    f"LEAKAGE: {pid} has a sliced hour >= its onset hour!"
                )

    def test_excluded_patient_absent_from_sliced_data(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        assert "pos_too_early" not in sliced["patient_id"].unique()


class TestFeatureEngineering:
    def test_output_shape_one_row_per_included_patient(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        patient_df = fe.build_patient_level_features(sliced, table)
        # 2 patients included (neg1, pos_early); pos_too_early excluded
        assert len(patient_df) == 2
        assert set(patient_df["patient_id"]) == {"neg1", "pos_early"}

    def test_count_reflects_only_real_measurements(self, toy_raw_df):
        """HR is measured every hour in this fixture, so HR_count should
        equal the number of usable hours (never inflated by LOCF)."""
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        patient_df = fe.build_patient_level_features(sliced, table)

        neg1_row = patient_df[patient_df["patient_id"] == "neg1"].iloc[0]
        assert neg1_row["HR_count"] == 6

        pos_row = patient_df[patient_df["patient_id"] == "pos_early"].iloc[0]
        assert pos_row["HR_count"] == 4  # only 4 usable hours

    def test_missing_ratio_is_correct(self, toy_raw_df):
        """pos_early has Lactate measured once (hour 2) within its 4 usable
        hours -> missing_ratio should be 3/4 = 0.75, not near-zero (which
        would indicate a LOCF-contamination bug — see Phase 8 design note)."""
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        patient_df = fe.build_patient_level_features(sliced, table)

        pos_row = patient_df[patient_df["patient_id"] == "pos_early"].iloc[0]
        assert pos_row["Lactate_count"] == 1
        assert pytest.approx(pos_row["Lactate_missing_ratio"], abs=1e-9) == 0.75

    def test_derived_shock_index_computed(self, toy_raw_df):
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        patient_df = fe.build_patient_level_features(sliced, table)

        neg1_row = patient_df[patient_df["patient_id"] == "neg1"].iloc[0]
        # ShockIndex = HR/SBP; HR ranges 81-86, SBP constant 120
        assert neg1_row["ShockIndex_mean"] > 0
        assert neg1_row["ShockIndex_mean"] < 1.0

    def test_no_leakage_columns_in_final_output_when_using_preprocessing(self, toy_raw_df):
        from src import preprocessing as pp
        table = tgt.build_patient_level_window_labels(toy_raw_df, window_hours=6)
        sliced = tgt.slice_features_to_window(toy_raw_df, table)
        patient_df = fe.build_patient_level_features(sliced, table)

        feature_cols = pp.get_feature_columns(patient_df)
        for leak_col in ["onset_hour", "usable_hours", "n_hours_used", "label"]:
            assert leak_col not in feature_cols, f"LEAKAGE: {leak_col} found in feature columns!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
