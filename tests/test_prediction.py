"""
tests/test_prediction.py
==========================
Phase 26 — Unit tests for src/predict.py (SepsisPredictor / predict_sepsis).

Uses a small, fast, real (not mocked) trained pipeline — built fresh in a
temp directory for each test session — so these tests exercise the ACTUAL
feature-engineering -> preprocessing -> model code path, not a stub.

Run with: pytest tests/test_prediction.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import joblib
import lightgbm as lgb

from src import config
from src import feature_engineering as fe
from src import preprocessing as pp
from src import target as tgt
from src.predict import SepsisPredictor


@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    """
    Build a small, real, end-to-end trained pipeline (feature engineering,
    preprocessing, and a tiny LightGBM model) and save it to a temp
    directory, then load it via SepsisPredictor — exactly the same code
    path a real deployment uses, just with a fast/tiny model.
    """
    models_dir = tmp_path_factory.mktemp("models")

    np.random.seed(0)
    rows = []
    for pid in range(150):
        n_hours = np.random.randint(2, 7)
        label = np.random.choice([0, 1], p=[0.85, 0.15])
        for h in range(1, n_hours + 1):
            rows.append({
                "patient_id": f"p{pid}", "source_set": "A", "ICULOS": h,
                "HR": 85 + np.random.randn() * 8, "O2Sat": 96 + np.random.randn() * 2,
                "Temp": 37 + np.random.randn() * 0.4, "SBP": 115 + np.random.randn() * 10,
                "DBP": 70 + np.random.randn() * 6, "MAP": 85 + np.random.randn() * 6,
                "Resp": 18 + np.random.randn() * 3,
                "WBC": np.nan if np.random.rand() < 0.9 else 11.0,
                "Lactate": np.nan if np.random.rand() < 0.95 else 2.5,
                "Age": 60, "Gender": pid % 2, "Unit1": 1.0,
                "Unit2": np.nan if np.random.rand() < 0.4 else float(pid % 2),
                "HospAdmTime": -5, "SepsisLabel": label,
            })
    raw_df = pd.DataFrame(rows)
    for c in config.ALL_CLINICAL_FEATURES:
        if c not in raw_df.columns:
            raw_df[c] = np.nan

    window_table = tgt.build_patient_level_window_labels(raw_df, config.PRIMARY_PREDICTION_WINDOW_HOURS)
    sliced = tgt.slice_features_to_window(raw_df, window_table)
    patient_df = fe.build_patient_level_features(sliced, window_table)

    train_ids = patient_df[config.PATIENT_ID_COL].to_numpy()
    preprocessor, numeric_cols, categorical_cols = pp.fit_preprocessing_pipeline(patient_df, train_ids)
    X, y, feat_names = pp.transform_split(preprocessor, patient_df, train_ids, numeric_cols, categorical_cols)
    model = lgb.LGBMClassifier(n_estimators=15, verbose=-1).fit(X, y)

    joblib.dump({"preprocessor": preprocessor, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
                models_dir / "preprocessing_pipeline.pkl")
    joblib.dump(model, models_dir / "best_model.pkl")
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump({"model_name": "test_model", "decision_threshold": 0.3, "feature_list": feat_names}, f)

    return SepsisPredictor(models_dir=models_dir)


def make_valid_patient(n_hours=6):
    return pd.DataFrame({
        "ICULOS": list(range(1, n_hours + 1)),
        "HR": [90] * n_hours, "O2Sat": [95] * n_hours, "Temp": [37.2] * n_hours,
        "SBP": [110] * n_hours, "DBP": [70] * n_hours, "MAP": [83] * n_hours,
        "Resp": [19] * n_hours, "Age": [65] * n_hours, "Gender": [1] * n_hours,
    })


class TestInputValidation:
    def test_rejects_non_dataframe_input(self, trained_predictor):
        with pytest.raises(ValueError, match="DataFrame"):
            trained_predictor.predict({"HR": 90})

    def test_rejects_empty_dataframe(self, trained_predictor):
        with pytest.raises(ValueError, match="empty"):
            trained_predictor.predict(pd.DataFrame())

    def test_rejects_missing_iculos_column(self, trained_predictor):
        df = pd.DataFrame({"HR": [90, 91], "Age": [65, 65], "Gender": [1, 1]})
        with pytest.raises(ValueError, match="ICULOS"):
            trained_predictor.predict(df)

    def test_rejects_missing_demographic_columns(self, trained_predictor):
        df = pd.DataFrame({"ICULOS": [1, 2], "HR": [90, 91]})
        with pytest.raises(ValueError, match="demographic"):
            trained_predictor.predict(df)


class TestPredictionOutput:
    def test_output_has_all_expected_keys(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient())
        for key in ["probability", "prediction", "risk_category", "threshold_used",
                    "n_hours_provided", "data_completeness_warning", "warning_message"]:
            assert key in result

    def test_probability_in_valid_range(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient())
        assert 0.0 <= result["probability"] <= 1.0

    def test_prediction_is_binary(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient())
        assert result["prediction"] in (0, 1)

    def test_prediction_matches_threshold_logic(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient())
        expected = int(result["probability"] >= result["threshold_used"])
        assert result["prediction"] == expected

    def test_risk_category_is_valid(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient())
        assert result["risk_category"] in ("Low Risk", "Medium Risk", "High Risk")

    def test_missing_lab_values_handled_gracefully(self, trained_predictor):
        """A patient with NO labs at all (only vitals) should not crash —
        missing labs must be imputed, not cause a KeyError/ValueError."""
        df = make_valid_patient()
        result = trained_predictor.predict(df)  # WBC/Lactate never provided at all
        assert result["probability"] is not None


class TestDataCompletenessWarning:
    def test_full_window_has_no_warning(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient(n_hours=config.PRIMARY_PREDICTION_WINDOW_HOURS))
        assert result["n_hours_provided"] == config.PRIMARY_PREDICTION_WINDOW_HOURS
        assert result["data_completeness_warning"] is False
        assert result["warning_message"] is None

    def test_short_window_triggers_warning(self, trained_predictor):
        result = trained_predictor.predict(make_valid_patient(n_hours=2))
        assert result["n_hours_provided"] == 2
        assert result["data_completeness_warning"] is True
        assert result["warning_message"] is not None
        assert "2 hour" in result["warning_message"]

    def test_extra_hours_beyond_window_are_truncated_not_rejected(self, trained_predictor):
        """More than PRIMARY_PREDICTION_WINDOW_HOURS of data should be
        truncated (with a warning logged), not raise an error."""
        df = make_valid_patient(n_hours=10)
        result = trained_predictor.predict(df)
        assert result["n_hours_provided"] == config.PRIMARY_PREDICTION_WINDOW_HOURS


class TestConsistency:
    def test_same_input_gives_same_output(self, trained_predictor):
        """Determinism check: identical input must give identical output —
        no hidden randomness in the prediction path."""
        df = make_valid_patient()
        result1 = trained_predictor.predict(df)
        result2 = trained_predictor.predict(df)
        assert result1["probability"] == result2["probability"]
        assert result1["prediction"] == result2["prediction"]

    def test_worse_vitals_do_not_crash_and_stay_in_range(self, trained_predictor):
        """Sanity check with clinically extreme (but not invalid) vitals —
        must still produce a valid probability, whatever its value."""
        df = make_valid_patient()
        df["HR"] = [130, 135, 140, 145, 150, 155]
        df["MAP"] = [55, 52, 50, 48, 45, 42]
        result = trained_predictor.predict(df)
        assert 0.0 <= result["probability"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
