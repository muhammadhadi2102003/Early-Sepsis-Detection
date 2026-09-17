"""
target.py
==========
Phase 7 — Target Construction for the Early Sepsis Detection project.

CRITICAL PURPOSE: define a leakage-safe patient-level prediction task from
the official, already-hourly-labeled `SepsisLabel` column.

Background (official PhysioNet/CinC 2019 Challenge framing, described here
for orientation — the specifics below are verified empirically against the
actual downloaded data by 06_target_construction.ipynb, not assumed):
The Challenge's SepsisLabel is already constructed to reward EARLY
prediction: for septic patients it is defined relative to an estimated
sepsis onset time using clinical criteria (suspected infection + SOFA
score change), and is 1 starting several hours before that onset (the
"6-hour-early" framing referenced throughout this project), then remains
1 for the rest of the encoded stay. This is why we cannot just use "the
whole stay" as a feature window for positive patients without checking
exactly where the label turns on.

This module implements ONLY the mechanics of building a safe, patient-level
feature window; it makes no unverified claims about the exact official
onset formula. All behavior described in code comments here is checked
against the real, loaded data by the corresponding notebook before being
relied upon.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)


# ------------------------------------------------------------------
# Step 1 — Empirical verification of label behavior (run BEFORE trusting
# any assumption about "label turns on once and stays on")
# ------------------------------------------------------------------
def verify_label_monotonicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every patient, check whether SepsisLabel ever goes from 1 back to 0
    (i.e. is NOT simply "off, then on, then on forever").

    Returns a dataframe of any patients that violate this pattern, so the
    assumption is proven against the real data rather than assumed from
    the literature.
    """
    def has_1_to_0_transition(labels: pd.Series) -> bool:
        arr = labels.to_numpy()
        return bool(np.any((arr[:-1] == 1) & (arr[1:] == 0)))

    violations = (
        df.sort_values([config.PATIENT_ID_COL, "ICULOS"])
        .groupby(config.PATIENT_ID_COL)[config.TARGET_COL]
        .apply(has_1_to_0_transition)
    )
    bad_patients = violations[violations].index.tolist()
    if bad_patients:
        logger.warning(
            "%d patients have a SepsisLabel 1->0 transition — the "
            "'label turns on and stays on' assumption does NOT hold for "
            "them. Inspect these patients individually before proceeding.",
            len(bad_patients),
        )
    else:
        logger.info("Verified: no patient has a SepsisLabel 1->0 transition "
                    "(monotonic non-decreasing label confirmed for all %d patients).",
                    df[config.PATIENT_ID_COL].nunique())
    return df[df[config.PATIENT_ID_COL].isin(bad_patients)]


def get_onset_hour(df: pd.DataFrame) -> pd.Series:
    """
    For each sepsis-positive patient, return the first ICULOS hour at which
    SepsisLabel == 1 (their "onset hour" as encoded in this dataset).
    Sepsis-negative patients are excluded (no onset hour exists).
    """
    positive_rows = df[df[config.TARGET_COL] == 1]
    onset_hour = positive_rows.groupby(config.PATIENT_ID_COL)["ICULOS"].min()
    onset_hour.name = "onset_hour"
    return onset_hour


# ------------------------------------------------------------------
# Step 2 — Leakage-safe prediction-window construction
# ------------------------------------------------------------------
# Design decision (documented, not fabricated):
#   For a prediction window of W hours (e.g. W=6, 12, 24 — Phase 9's
#   PREDICTION_WINDOWS_HOURS), we build patient-level features using ONLY
#   hourly rows STRICTLY BEFORE the patient's onset hour (if positive) or
#   the first W hours of the stay (if negative).
#
#   - Sepsis-negative patient: feature window = rows with ICULOS <= W.
#   - Sepsis-positive patient: feature window = rows with ICULOS <= min(W, onset_hour - 1).
#     This guarantees we NEVER include the onset hour itself or any hour
#     at/after it — the exact leakage this phase exists to prevent.
#   - A positive patient whose onset_hour <= 1 has NO valid pre-onset
#     window (onset_hour - 1 <= 0): they are EXCLUDED from this task and
#     counted/reported explicitly, never silently dropped.
#
# Why not just use "all data before the fixed calendar hour W" for positive
# patients regardless of onset? Because if onset happens at, say, hour 3,
# and W=6, using hours 1-6 would include hours 4,5,6 which are AT/AFTER
# onset (label=1) — using those hours' vitals/labs to "predict" sepsis at
# a point where sepsis has already technically started/been flagged is
# leakage relative to the stated "early prediction" goal.
def build_patient_level_window_labels(
    df: pd.DataFrame, window_hours: int
) -> pd.DataFrame:
    """
    Return one row per patient (patient_id, source_set, label, window_hours,
    included, exclusion_reason) describing how much pre-onset history is
    available for each patient at this prediction window.

    `label` is the patient-level target for this task: 1 if the patient
    became septic at ANY point during their stay (matching Phase 2/4's
    patient-level prevalence definition), 0 otherwise. `included` tells
    the feature-engineering step (Phase 8) whether this patient is usable
    for THIS SPECIFIC window without leakage.

    IMPORTANT (discovered empirically, not assumed): a subset of patients'
    raw files do not begin at ICULOS==1 — their earliest recorded hour can
    be much later (observed up to ICULOS==304 in this dataset). For such a
    patient, even though onset_hour math says a window is "usable," there
    may be ZERO actual rows with ICULOS <= usable_hours, since their file
    simply has no data that early. This function explicitly detects and
    excludes those patients with a clear reason, rather than letting them
    silently disappear during slicing (which produced an unexplained gap
    of 595 patients before this fix).
    """
    patient_label = df.groupby(config.PATIENT_ID_COL)[config.TARGET_COL].max()
    onset_hour = get_onset_hour(df)
    source_set = df.groupby(config.PATIENT_ID_COL)[config.SOURCE_SET_COL].first()
    min_iculos = df.groupby(config.PATIENT_ID_COL)["ICULOS"].min()

    rows = []
    for pid, label in patient_label.items():
        first_hour = min_iculos.get(pid, np.nan)

        if label == 1:
            oh = onset_hour.get(pid, np.nan)
            usable_hours = min(window_hours, oh - 1) if pd.notna(oh) else 0
            included = usable_hours >= 1
            reason = "" if included else f"onset_hour={oh} <= 1, no pre-onset data available"
        else:
            usable_hours = window_hours
            included = True
            reason = ""

        # Second, independent exclusion check: does this patient's file
        # actually contain any row within [first_hour, usable_hours]? If
        # their earliest recorded hour is already beyond the usable window,
        # there is no real data to build features from, regardless of the
        # onset-based math above.
        if included and pd.notna(first_hour) and first_hour > usable_hours:
            included = False
            reason = (
                f"first recorded ICULOS={int(first_hour)} exceeds usable window "
                f"({usable_hours}h) — no data available in this patient's file "
                f"for the required early hours"
            )

        rows.append({
            config.PATIENT_ID_COL: pid,
            config.SOURCE_SET_COL: source_set.get(pid),
            "label": int(label),
            "onset_hour": onset_hour.get(pid, np.nan),
            "first_recorded_iculos": first_hour,
            "window_hours_requested": window_hours,
            "usable_hours": usable_hours,
            "included": included,
            "exclusion_reason": reason,
        })

    result = pd.DataFrame(rows)
    n_excluded = int((~result["included"]).sum())
    n_excluded_onset = int((result["exclusion_reason"].str.startswith("onset_hour", na=False)).sum())
    n_excluded_late_start = int((result["exclusion_reason"].str.startswith("first recorded", na=False)).sum())
    if n_excluded:
        logger.warning(
            "Window=%dh: excluding %d/%d total patients (%d due to onset_hour<=1, "
            "%d due to no data within the usable window despite a later onset).",
            window_hours, n_excluded, len(result), n_excluded_onset, n_excluded_late_start,
        )
    return result


def slice_features_to_window(df: pd.DataFrame, window_table: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw/cleaned long-format hourly dataframe and a window table
    from build_patient_level_window_labels(), return only the rows that
    fall within each patient's leakage-safe usable window.

    Excluded patients (included == False) are dropped from the returned
    dataframe entirely — they should NOT appear in any split for this
    specific prediction-window experiment.
    """
    usable = window_table[window_table["included"]].set_index(config.PATIENT_ID_COL)["usable_hours"]
    df = df[df[config.PATIENT_ID_COL].isin(usable.index)].copy()
    df["_usable_hours"] = df[config.PATIENT_ID_COL].map(usable)
    df = df[df["ICULOS"] <= df["_usable_hours"]]
    return df.drop(columns=["_usable_hours"])


if __name__ == "__main__":
    df = pd.read_parquet(config.INTERIM_DIR / "sepsis_hourly_cleaned.parquet")
    bad = verify_label_monotonicity(df)
    print(f"Patients with 1->0 label transitions: {bad[config.PATIENT_ID_COL].nunique() if len(bad) else 0}")

    for w in config.PREDICTION_WINDOWS_HOURS:
        table = build_patient_level_window_labels(df, w)
        print(f"Window={w}h: {table['included'].sum()} / {len(table)} patients usable "
              f"({(~table['included']).sum()} excluded).")
