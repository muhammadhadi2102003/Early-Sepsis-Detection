"""
feature_engineering.py
========================
Phase 8 — Patient-level temporal feature engineering.

CRITICAL DESIGN DECISION (documented, not incidental):
Statistical aggregates (mean/median/min/max/std/slope/count/missing_ratio)
are computed from the RAW (pre-LOCF) hourly values — i.e. TRUE measurements
only — never from the Phase-3 forward-filled ("LOCF") data.

Why: LOCF duplicates a single true measurement across every subsequent hour
until the next reading. Computing mean/std on LOCF-filled data would:
  (a) artificially shrink standard deviation (many rows are literally the
      same repeated number),
  (b) artificially inflate "count" and shrink "missing_ratio" — destroying
      exactly the signal Phase 5 found to be predictive (e.g. Lactate was
      measured ~2.3x more often in septic patients; that signal only
      exists in the RAW measurement pattern, not after it's smeared across
      every hour by LOCF).

"first"/"last" features intentionally DO use the most-recent-known-value
concept (conceptually like a single LOCF step) since "the patient's last
known state" is itself a legitimate, distinct feature type — not a
duplicated series used for variance/count statistics.

All feature construction here operates on a dataframe ALREADY SLICED to a
leakage-safe window by src/target.py's slice_features_to_window() — this
module does not know or care which prediction window was used; it simply
aggregates whatever hourly rows it is given per patient.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)


# ------------------------------------------------------------------
# Derived, row-level clinical features (computed BEFORE aggregation, so
# they get the same mean/min/max/std/last/etc. treatment as any raw variable)
# ------------------------------------------------------------------
# What: Shock Index (HR/SBP), Pulse Pressure (SBP-DBP), a SaO2/FiO2 ratio
#       (non-invasive proxy conceptually related to the P/F ratio used in
#       ARDS/sepsis-related respiratory assessment), and a binary
#       temperature-abnormality flag (SIRS-consistent thresholds).
# Why: These combine two raw signals into a single value with known
#       clinical relevance to hemodynamic/respiratory instability, rather
#       than requiring the model to learn the interaction from scratch.
# Why NOT more exotic derived features: each one added here has an
#       explicit physiological rationale in the comment beside it — no
#       feature is added "because it might help" without a stated reason.
def add_derived_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Shock Index = HR / SBP. Elevated (>0.9) is a recognized marker of
    # hemodynamic compromise/occult shock, more sensitive than HR or SBP alone.
    out["ShockIndex"] = out["HR"] / out["SBP"].replace(0, np.nan)

    # Pulse pressure = SBP - DBP. A narrowed pulse pressure can indicate
    # reduced stroke volume / poor perfusion, relevant to septic shock.
    out["PulsePressure"] = out["SBP"] - out["DBP"]

    # SaO2/FiO2 ratio: a non-invasive analogue of the PaO2/FiO2 ("P/F") ratio
    # used to assess the severity of oxygenation impairment (e.g. in
    # sepsis-associated ARDS). Only meaningful where FiO2 > 0.
    out["SaO2_FiO2_ratio"] = out["SaO2"] / out["FiO2"].replace(0, np.nan)

    # Temperature abnormality flag: fever (>38.3C) or hypothermia (<36.0C)
    # are both part of SIRS criteria historically associated with sepsis
    # screening.
    out["TempAbnormal"] = ((out["Temp"] > 38.3) | (out["Temp"] < 36.0)).astype("float32")
    out.loc[out["Temp"].isna(), "TempAbnormal"] = np.nan  # preserve missingness, don't fabricate a flag

    return out


DERIVED_FEATURES = ["ShockIndex", "PulsePressure", "SaO2_FiO2_ratio", "TempAbnormal"]


# ------------------------------------------------------------------
# Per-variable temporal aggregation
# ------------------------------------------------------------------
def _slope(values: np.ndarray, hours: np.ndarray) -> float:
    """Linear-regression slope of `values` against `hours`; NaN if <2 valid points."""
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    try:
        coeffs = np.polyfit(hours[mask], values[mask], deg=1)
        return float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_temporal_features(raw_window_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Compute, per patient, the standard feature set for each column in
    `columns`, using TRUE (non-imputed) measurements only:

        <col>_mean, _median, _min, _max, _std, _first, _last,
        _change (last - first), _abs_change, _count, _missing_ratio, _slope

    `raw_window_df` must already be sliced to the leakage-safe window
    (see src/target.py) and must be the RAW (pre-LOCF) hourly dataframe.
    """
    rows = []
    grouped = raw_window_df.sort_values([config.PATIENT_ID_COL, "ICULOS"]).groupby(
        config.PATIENT_ID_COL, sort=False
    )

    for pid, g in grouped:
        n_hours = len(g)
        hours = g["ICULOS"].to_numpy(dtype="float64")
        feat_row = {config.PATIENT_ID_COL: pid}

        for col in columns:
            values = g[col].to_numpy(dtype="float64")
            non_null = values[~np.isnan(values)]

            feat_row[f"{col}_mean"] = float(np.mean(non_null)) if len(non_null) else np.nan
            feat_row[f"{col}_median"] = float(np.median(non_null)) if len(non_null) else np.nan
            feat_row[f"{col}_min"] = float(np.min(non_null)) if len(non_null) else np.nan
            feat_row[f"{col}_max"] = float(np.max(non_null)) if len(non_null) else np.nan
            feat_row[f"{col}_std"] = float(np.std(non_null, ddof=1)) if len(non_null) > 1 else np.nan

            if len(non_null):
                first_idx = np.argmax(~np.isnan(values))
                last_idx = len(values) - 1 - np.argmax(~np.isnan(values[::-1]))
                feat_row[f"{col}_first"] = float(values[first_idx])
                feat_row[f"{col}_last"] = float(values[last_idx])
                feat_row[f"{col}_change"] = float(values[last_idx] - values[first_idx])
                feat_row[f"{col}_abs_change"] = abs(feat_row[f"{col}_change"])
            else:
                feat_row[f"{col}_first"] = np.nan
                feat_row[f"{col}_last"] = np.nan
                feat_row[f"{col}_change"] = np.nan
                feat_row[f"{col}_abs_change"] = np.nan

            feat_row[f"{col}_count"] = int(len(non_null))
            feat_row[f"{col}_missing_ratio"] = 1.0 - (len(non_null) / n_hours) if n_hours else np.nan
            feat_row[f"{col}_slope"] = _slope(values, hours)

        feat_row["n_hours_used"] = n_hours
        rows.append(feat_row)

    result = pd.DataFrame(rows)
    logger.info("Computed temporal features for %d patients x %d source columns (%d output columns).",
                len(result), len(columns), result.shape[1])
    return result


# ------------------------------------------------------------------
# Demographic / static features (constant per patient — take first value)
# ------------------------------------------------------------------
STATIC_FEATURES = ["Age", "Gender", "Unit1", "Unit2", "Unit1_known", "Unit2_known", "HospAdmTime"]


def extract_static_features(raw_window_df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in STATIC_FEATURES if c in raw_window_df.columns]
    static_df = raw_window_df.groupby(config.PATIENT_ID_COL, sort=False)[available].first().reset_index()
    return static_df


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------
def build_patient_level_features(
    raw_window_df: pd.DataFrame,
    window_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full Phase 8 pipeline: add derived clinical features, compute temporal
    aggregates for all clinical variables + derived features, attach static
    demographics, and merge in the patient-level label from `window_table`
    (produced by src/target.py — already leakage-safe for this window).
    """
    enriched = add_derived_clinical_features(raw_window_df)

    all_temporal_cols = config.ALL_CLINICAL_FEATURES + DERIVED_FEATURES
    all_temporal_cols = [c for c in all_temporal_cols if c in enriched.columns]

    temporal_feats = compute_temporal_features(enriched, all_temporal_cols)
    static_feats = extract_static_features(enriched)

    patient_df = temporal_feats.merge(static_feats, on=config.PATIENT_ID_COL, how="left")
    patient_df = patient_df.merge(
        window_table[[config.PATIENT_ID_COL, config.SOURCE_SET_COL, "label", "onset_hour", "usable_hours"]],
        on=config.PATIENT_ID_COL, how="left",
    )

    logger.info("Final patient-level feature table: %d patients x %d columns.",
                patient_df.shape[0], patient_df.shape[1])
    return patient_df


if __name__ == "__main__":
    from . import target as tgt

    raw_df = pd.read_parquet(config.INTERIM_LONG_PARQUET)
    window_table = tgt.build_patient_level_window_labels(raw_df, config.PRIMARY_PREDICTION_WINDOW_HOURS)
    sliced = tgt.slice_features_to_window(raw_df, window_table)
    patient_df = build_patient_level_features(sliced, window_table)
    print(patient_df.shape)
