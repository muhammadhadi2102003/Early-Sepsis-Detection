"""
utils.py
=========
Phase 25 — Helper functions for the Streamlit dashboard.

Keeps app/streamlit_app.py focused on layout/UX; this module handles
loading the model, building input widgets' underlying dataframe, and
formatting SHAP-based explanations for display.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src import ...` when run as `streamlit run app/streamlit_app.py`
# from the project root.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import shap

from src import config
from src.predict import SepsisPredictor


def load_predictor() -> SepsisPredictor:
    """Cached-friendly loader — Streamlit's own @st.cache_resource wraps this in the app."""
    return SepsisPredictor()


def load_model_metadata() -> dict:
    with open(config.MODELS_DIR / "model_metadata.json") as f:
        return json.load(f)


def build_patient_dataframe(hourly_inputs: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-hour input dicts (one per ICU hour the user filled
    in) into the wide-format hourly dataframe predict_sepsis() expects.
    """
    df = pd.DataFrame(hourly_inputs)
    df["ICULOS"] = range(1, len(df) + 1)
    return df


def get_top_shap_contributions(
    predictor: SepsisPredictor, patient_data: pd.DataFrame, top_n: int = 8
) -> pd.DataFrame:
    """
    Compute SHAP values for a single patient's engineered feature vector
    (reusing the same LightGBM base model SHAP was validated on in
    Phase 20), returning the top_n features by |SHAP value| for display.

    Note: SHAP explains the raw (pre-calibration) LightGBM model, exactly
    as in Phase 20 — calibration only rescales the final probability
    monotonically and does not change which features drove the ranking.
    """
    patient_row = predictor._build_feature_row(patient_data)
    aligned = predictor._align_to_expected_features(patient_row)
    X = predictor.preprocessor.transform(aligned)

    # The underlying LightGBM model may be wrapped multiple layers deep:
    #   CalibratedClassifierCV -> calibrated_classifiers_[0].estimator -> FrozenEstimator -> actual LightGBM model
    # (scikit-learn >=1.6's calibration API wraps the fitted base estimator
    # in FrozenEstimator; see src/preprocessing.py / Phase 19 for context).
    # Repeatedly unwrap via `.estimator` until we reach the real tree model
    # (which has no further `.estimator` attribute of its own).
    base_model = predictor.model
    if hasattr(base_model, "calibrated_classifiers_"):
        base_model = base_model.calibrated_classifiers_[0].estimator
    while hasattr(base_model, "estimator"):
        base_model = base_model.estimator

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Use the preprocessor's own output feature names (post one-hot
    # expansion) — guaranteed to match X's actual column count, avoiding
    # any manual reconstruction/alignment mismatch.
    feature_names = list(predictor.preprocessor.get_feature_names_out())
    feature_names = [name.split("__", 1)[-1] for name in feature_names]

    if len(feature_names) != shap_values.shape[1]:
        # Defensive fallback — should not normally trigger.
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0],
        "feature_value": X_dense[0],
    })
    contrib["abs_shap"] = contrib["shap_value"].abs()
    return contrib.sort_values("abs_shap", ascending=False).head(top_n)


RISK_COLORS = {
    "Low Risk": "#2ca02c",
    "Medium Risk": "#ff9800",
    "High Risk": "#d32f2f",
}
