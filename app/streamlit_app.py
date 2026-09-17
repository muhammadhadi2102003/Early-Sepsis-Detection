"""
streamlit_app.py
==================
Phase 25 — Early Sepsis Detection dashboard.

Run with:
    streamlit run app/streamlit_app.py

Loads the final model + preprocessing pipeline + metadata from Phase 23
(models/best_model.pkl, models/preprocessing_pipeline.pkl,
models/model_metadata.json) via src/predict.py — no retraining happens here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from utils import (
    load_predictor,
    load_model_metadata,
    build_patient_dataframe,
    get_top_shap_contributions,
    RISK_COLORS,
)

st.set_page_config(page_title="Early Sepsis Detection", page_icon="🩺", layout="wide")


# ------------------------------------------------------------------
# Cached resource loading (model/metadata loaded once per session)
# ------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    return load_predictor()


@st.cache_resource
def get_metadata():
    return load_model_metadata()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🩺 Early Sepsis Detection")
st.caption(
    "Research/educational prototype — predicts short-term sepsis risk from a "
    "patient's first hours of ICU clinical data (PhysioNet/CinC 2019 Challenge dataset)."
)

try:
    predictor = get_predictor()
    metadata = get_metadata()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(
        f"Model artifacts not found: {e}\n\n"
        "Run notebooks 09 (preprocessing) and 21 (final model selection) first, "
        "so `models/best_model.pkl`, `models/preprocessing_pipeline.pkl`, and "
        "`models/model_metadata.json` exist."
    )

if model_loaded:
    # ------------------------------------------------------------------
    # Patient Information (inputs)
    # ------------------------------------------------------------------
    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=65)
        gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    with col2:
        unit1 = st.selectbox("ICU Unit type (Unit1)", options=[("Unknown", -1), ("Type 0", 0), ("Type 1", 1)],
                              format_func=lambda x: x[0])[1]
        hosp_adm_time = st.number_input("Hours from hospital admission to ICU admission (negative = admitted to hospital first)",
                                         value=-5.0, step=1.0)
    with col3:
        n_hours = st.slider(f"Hours of ICU data available (model trained on up to {config.PRIMARY_PREDICTION_WINDOW_HOURS}h)",
                             min_value=1, max_value=config.PRIMARY_PREDICTION_WINDOW_HOURS,
                             value=config.PRIMARY_PREDICTION_WINDOW_HOURS)

    st.subheader("Hourly vitals and labs")
    st.caption("Leave a cell blank (NaN) if that value was not measured at that hour — "
               "this matches how the model was trained (missingness itself is informative).")

    default_rows = pd.DataFrame({
        "HR": [88.0] * n_hours, "O2Sat": [96.0] * n_hours, "Temp": [37.0] * n_hours,
        "SBP": [115.0] * n_hours, "DBP": [70.0] * n_hours, "MAP": [85.0] * n_hours,
        "Resp": [18.0] * n_hours, "WBC": [np.nan] * n_hours, "Lactate": [np.nan] * n_hours,
    })
    edited = st.data_editor(default_rows, num_rows="fixed", width='stretch', key="hourly_editor")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    if st.button("Predict Sepsis Risk", type="primary"):
        hourly_dicts = edited.to_dict(orient="records")
        for d in hourly_dicts:
            d["Age"] = age
            d["Gender"] = gender
            d["Unit1"] = unit1
            d["HospAdmTime"] = hosp_adm_time

        patient_df = build_patient_dataframe(hourly_dicts)

        with st.spinner("Scoring patient..."):
            result = predictor.predict(patient_df)

        st.header("Prediction")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Sepsis Probability", f"{result['probability']*100:.1f}%")
        col_b.metric("Prediction", "Sepsis Risk Flagged" if result["prediction"] == 1 else "No Flag")
        col_c.metric("Decision Threshold", f"{result['threshold_used']}")

        # ------------------------------------------------------------
        # Risk Indicator
        # ------------------------------------------------------------
        risk_category = result["risk_category"]
        color = RISK_COLORS.get(risk_category, "#888888")
        st.markdown(
            f"<div style='padding:1rem;border-radius:0.5rem;background-color:{color};"
            f"color:white;text-align:center;font-size:1.5rem;font-weight:bold;'>"
            f"{risk_category.upper()}</div>",
            unsafe_allow_html=True,
        )

        # Data-completeness warning (Phase 24 finding — see model_metadata.json)
        if result.get("data_completeness_warning"):
            st.warning(f"⚠️ {result['warning_message']}")

        # ------------------------------------------------------------
        # Explainability
        # ------------------------------------------------------------
        st.header("Explainability — Top Contributing Features")
        st.caption(
            "SHAP values show this model's associations/contributions for THIS "
            "patient — not medical causality. Positive values push the "
            "prediction toward higher sepsis risk; negative values push it lower."
        )
        try:
            with st.spinner("Computing SHAP explanation..."):
                contrib = get_top_shap_contributions(predictor, patient_df, top_n=8)
            st.bar_chart(contrib.set_index("feature")["shap_value"])
            st.dataframe(contrib[["feature", "feature_value", "shap_value"]], width='stretch')
        except Exception as e:
            st.info(f"SHAP explanation unavailable for this input: {e}")

    # ------------------------------------------------------------------
    # Model Information
    # ------------------------------------------------------------------
    st.header("Model Information")
    col_x, col_y, col_z = st.columns(3)
    col_x.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
    col_y.write(f"**Decision threshold:** {metadata.get('decision_threshold', 'N/A')}")
    val_metrics = metadata.get("validation_metrics", {})
    col_z.write(f"**Validation ROC-AUC:** {val_metrics.get('roc_auc', 'N/A'):.3f}" if val_metrics.get("roc_auc") else "**Validation ROC-AUC:** N/A")

    with st.expander("Full validation & test metrics"):
        st.json({
            "validation_metrics": metadata.get("validation_metrics", {}),
            "test_metrics": metadata.get("test_metrics", {}),
        })

    with st.expander("Known limitations (please read)"):
        for name, text in metadata.get("known_limitations", {}).items():
            st.markdown(f"**{name}:** {text}")

# ------------------------------------------------------------------
# Disclaimer
# ------------------------------------------------------------------
st.divider()
st.error(
    "**Disclaimer:** This application is a research/educational prototype and "
    "is not intended for clinical diagnosis or treatment decisions."
)
