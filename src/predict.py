"""
predict.py
===========
Phase 24 — Reusable prediction pipeline for the Early Sepsis Detection
project.

Provides `predict_sepsis(patient_data)`, which takes a new patient's
hourly clinical records (a small dataframe, one row per ICU hour, in the
official PhysioNet schema) and returns a risk probability, binary
prediction, and risk category — using the EXACT SAME preprocessing,
feature engineering, model, and threshold established in Phases 8-23.

This module intentionally re-derives features the same way
src/feature_engineering.py does, rather than requiring the caller to
already have an engineered feature vector — the input contract here is
"raw-ish" hourly clinical data, matching what a real deployment would
receive from an EHR feed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import config
from . import feature_engineering as fe
from . import preprocessing as pp

logger = config.get_logger(__name__)


class SepsisPredictor:
    """
    Loads the final model, preprocessing pipeline, and metadata once, and
    exposes `predict(patient_data)` for repeated use — avoids reloading
    large artifacts from disk on every call.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or config.MODELS_DIR

        metadata_path = self.models_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"model_metadata.json not found at {metadata_path}. "
                "Run notebook 21_final_model_selection.ipynb first."
            )
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.model = joblib.load(self.models_dir / "best_model.pkl")
        pipeline_payload = joblib.load(self.models_dir / "preprocessing_pipeline.pkl")
        self.preprocessor = pipeline_payload["preprocessor"]
        self.numeric_cols = pipeline_payload["numeric_cols"]
        self.categorical_cols = pipeline_payload["categorical_cols"]
        self.threshold = self.metadata["decision_threshold"]
        self.expected_features = self.metadata["feature_list"]

        logger.info("SepsisPredictor loaded: model=%s, threshold=%s, n_features=%d",
                    self.metadata["model_name"], self.threshold, len(self.expected_features))

    # ------------------------------------------------------------------
    # Step 1 — Input validation
    # ------------------------------------------------------------------
    def _validate_input(self, patient_data: pd.DataFrame) -> None:
        """
        Confirm the input looks like valid hourly patient data BEFORE any
        processing. Raises ValueError with a clear message on failure —
        never silently proceeds with malformed input.
        """
        if not isinstance(patient_data, pd.DataFrame):
            raise ValueError("patient_data must be a pandas DataFrame (one row per ICU hour).")

        if len(patient_data) == 0:
            raise ValueError("patient_data is empty — at least one hourly row is required.")

        if "ICULOS" not in patient_data.columns:
            raise ValueError("patient_data must include an 'ICULOS' column (hour index).")

        missing_required = set(["Age", "Gender"]) - set(patient_data.columns)
        if missing_required:
            raise ValueError(f"patient_data is missing required demographic columns: {missing_required}")

        n_hours = len(patient_data)
        if n_hours > config.PRIMARY_PREDICTION_WINDOW_HOURS:
            logger.warning(
                "patient_data has %d hours, more than the model's trained window "
                "(%dh) — only the first %d hours will be used.",
                n_hours, config.PRIMARY_PREDICTION_WINDOW_HOURS, config.PRIMARY_PREDICTION_WINDOW_HOURS,
            )

    # ------------------------------------------------------------------
    # Step 2-4 — Preprocess, feature-engineer, and vectorize a single patient
    # ------------------------------------------------------------------
    def _build_feature_row(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same derived-feature and temporal-aggregation logic as
        src/feature_engineering.py to a single patient's raw hourly rows,
        truncated to the primary prediction window.
        """
        df = patient_data.copy().sort_values("ICULOS")
        df = df[df["ICULOS"] <= config.PRIMARY_PREDICTION_WINDOW_HOURS]

        if config.PATIENT_ID_COL not in df.columns:
            df[config.PATIENT_ID_COL] = "new_patient"

        # Ensure every expected raw clinical column exists (as NaN if absent)
        # so feature_engineering's aggregation logic doesn't KeyError on a
        # column the caller simply never provided (e.g. a rarely-measured lab).
        for col in config.ALL_CLINICAL_FEATURES + ["SBP", "DBP", "HR", "SaO2", "FiO2", "Temp"]:
            if col not in df.columns:
                df[col] = np.nan
        for col in ["Unit1", "Unit2", "HospAdmTime"]:
            if col not in df.columns:
                df[col] = np.nan

        # Unit1_known/Unit2_known (Phase 3's encoding) — derive if not already present.
        if "Unit1_known" not in df.columns:
            df["Unit1_known"] = df["Unit1"].notna().astype("int8")
            df["Unit1"] = df["Unit1"].fillna(-1)
        if "Unit2_known" not in df.columns:
            df["Unit2_known"] = df["Unit2"].notna().astype("int8")
            df["Unit2"] = df["Unit2"].fillna(-1)

        enriched = fe.add_derived_clinical_features(df)
        all_cols = fe.ALL_CLINICAL_FEATURES = config.ALL_CLINICAL_FEATURES + fe.DERIVED_FEATURES
        all_cols = [c for c in all_cols if c in enriched.columns]

        temporal_feats = fe.compute_temporal_features(enriched, all_cols)
        static_feats = fe.extract_static_features(enriched)

        patient_row = temporal_feats.merge(static_feats, on=config.PATIENT_ID_COL, how="left")
        return patient_row

    def _align_to_expected_features(self, patient_row: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the engineered row has EXACTLY the columns the preprocessing
        pipeline expects, in the numeric_cols + categorical_cols order —
        missing engineered features (e.g. a lab never measured for this
        patient) become NaN, to be handled by the pipeline's imputer,
        exactly as in training.
        """
        expected_cols = self.numeric_cols + self.categorical_cols
        for col in expected_cols:
            if col not in patient_row.columns:
                patient_row[col] = np.nan
        return patient_row[expected_cols]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def predict(self, patient_data: pd.DataFrame) -> dict:
        """
        Score a single patient's early ICU hourly data.

        Parameters
        ----------
        patient_data : pd.DataFrame
            One row per ICU hour, containing at minimum 'ICULOS', 'Age',
            'Gender', and as many of the official PhysioNet clinical
            columns as are available. Missing columns are treated as
            "never measured" (NaN), consistent with training.

        Returns
        -------
        dict with keys: probability, prediction, risk_category,
        threshold_used, n_hours_provided, data_completeness_warning
        """
        self._validate_input(patient_data)

        n_hours_provided = int((patient_data["ICULOS"] <= config.PRIMARY_PREDICTION_WINDOW_HOURS).sum())

        patient_row = self._build_feature_row(patient_data)
        aligned = self._align_to_expected_features(patient_row)

        X = self.preprocessor.transform(aligned)
        probability = float(self.model.predict_proba(X)[:, 1][0])
        prediction = int(probability >= self.threshold)

        if probability < 0.05:
            risk_category = "Low Risk"
        elif probability < self.threshold:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"

        # ------------------------------------------------------------
        # KNOWN LIMITATION, discovered during Phase 24 testing, documented
        # in model_metadata.json's "known_limitations":
        #
        # During training (Phase 7), every SEPSIS-NEGATIVE patient's
        # feature window covered the full config.PRIMARY_PREDICTION_WINDOW_HOURS
        # (6h), while SEPSIS-POSITIVE patients' windows were truncated to
        # end strictly before their sepsis onset — which could be much
        # shorter (as little as 1 hour) if onset happened early. This means
        # "having fewer than 6 real hours of data in the window" was, by
        # construction, a pattern that ONLY ever occurred for positive
        # (septic) training patients — never for negatives.
        #
        # Consequently, any engineered feature sensitive to how many real
        # measurements exist (nearly all "_count" features, and to a
        # lesser extent "_missing_ratio"/"_std") can act as an indirect
        # proxy for "this patient's window was truncated due to early
        # onset" — i.e. an indirect leakage channel that survives even
        # though onset_hour/usable_hours themselves were correctly
        # excluded as direct features (Phase 11).
        #
        # Practical impact: a genuinely new patient who simply has not
        # yet accumulated a full 6 hours of ICU data (regardless of their
        # true future outcome) may receive an artificially inflated risk
        # score, because short real-hour counts were disproportionately
        # associated with the positive class during training.
        #
        # Mitigation implemented here (rather than reworking Phase 7-19,
        # which would require re-deriving features/retuning/recalibrating
        # every model at substantial computational cost): flag this
        # explicitly in the output so any consumer (e.g. the Streamlit
        # app, Phase 25) can display an appropriate caveat instead of
        # presenting the risk score as equally reliable regardless of
        # how much real history was available.
        # ------------------------------------------------------------
        data_completeness_warning = n_hours_provided < config.PRIMARY_PREDICTION_WINDOW_HOURS
        warning_message = (
            f"This patient has only {n_hours_provided} hour(s) of data (model trained "
            f"on windows up to {config.PRIMARY_PREDICTION_WINDOW_HOURS}h). During training, "
            "windows shorter than the full 6 hours occurred almost exclusively for patients "
            "who went on to develop sepsis (due to how the training data was constructed), "
            "so the risk score for early/incomplete-history patients like this one may be "
            "inflated and should be interpreted with caution — see model_metadata.json's "
            "'known_limitations' for details."
        ) if data_completeness_warning else None

        return {
            "probability": round(probability, 4),
            "prediction": prediction,
            "risk_category": risk_category,
            "threshold_used": self.threshold,
            "n_hours_provided": n_hours_provided,
            "data_completeness_warning": data_completeness_warning,
            "warning_message": warning_message,
        }


# ------------------------------------------------------------------
# Convenience function-style API (per the project spec's example)
# ------------------------------------------------------------------
_predictor_singleton: Optional[SepsisPredictor] = None


def predict_sepsis(patient_data: pd.DataFrame) -> dict:
    """
    Convenience wrapper matching the project spec's requested signature:

        predict_sepsis(patient_data) -> {"probability": ..., "prediction": ..., "risk_category": ...}

    Lazily initializes a module-level SepsisPredictor so repeated calls
    (e.g. from the Streamlit app) don't reload the model from disk each time.
    """
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = SepsisPredictor()
    return _predictor_singleton.predict(patient_data)


if __name__ == "__main__":
    # Minimal smoke test with a synthetic patient (6 hours, mostly missing labs)
    example = pd.DataFrame({
        "ICULOS": [1, 2, 3, 4, 5, 6],
        "HR": [88, 90, 95, 98, 102, 105],
        "O2Sat": [96, 95, 94, 93, 92, 91],
        "Temp": [37.2, 37.4, np.nan, 37.8, np.nan, 38.1],
        "SBP": [110, 108, 105, 100, 98, 95],
        "DBP": [70, 68, 66, 65, 63, 60],
        "MAP": [83, 81, 79, 77, 75, 72],
        "Resp": [18, 19, 20, 21, 22, 23],
        "WBC": [np.nan, np.nan, 14.2, np.nan, np.nan, np.nan],
        "Lactate": [np.nan, np.nan, np.nan, 2.8, np.nan, np.nan],
        "Age": [67] * 6,
        "Gender": [1] * 6,
        "Unit1": [1] * 6,
        "Unit2": [0] * 6,
        "HospAdmTime": [-10] * 6,
    })
    result = predict_sepsis(example)
    print("Prediction result:", result)
