"""
sequence_builder.py
=====================
Phase 16 — Build padded hourly sequences for the LSTM/GRU deep-learning
model, as a distinct input representation from Phase 8's aggregated
patient-level features.

Uses the LOCF-filled (Phase 3) hourly data, sliced to the SAME leakage-safe
primary window as every other model in this project (src/target.py),
so the deep-learning model sees exactly the same allowed information as
the tree-based models — no more, no less.

Padding: sequences shorter than the primary window length (patients whose
usable_hours < window_hours, per Phase 7) are padded with a sentinel value
far outside any real clinical range, and a Keras Masking layer is used so
the model ignores padded timesteps entirely rather than learning from them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)

MASK_VALUE = -999.0

# A focused set of variables for the sequence model — the most complete
# vitals (Phase 2 missingness) plus two key sepsis-relevant labs, kept
# small so the LSTM has a learnable, low-noise input at this dataset size
# (~27k training patients, only ~1.7k positive) rather than 34 sparse
# variables that would mostly be LOCF-duplicated placeholders.
SEQUENCE_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "WBC", "Lactate"]


def build_sequences(
    df_sliced: pd.DataFrame,
    patient_ids: np.ndarray,
    seq_vars: list[str] = SEQUENCE_VARS,
    max_len: int = config.PRIMARY_PREDICTION_WINDOW_HOURS,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a padded (n_patients, max_len, n_features) array from the
    LOCF-filled, window-sliced hourly dataframe.

    Patients with fewer than max_len rows (see Phase 7's usable_hours) are
    padded at the END with MASK_VALUE. Patients not present in df_sliced
    at all get an all-masked sequence (should not normally happen if
    `patient_ids` was derived correctly, but handled defensively).
    """
    n_patients = len(patient_ids)
    n_features = len(seq_vars)
    X = np.full((n_patients, max_len, n_features), MASK_VALUE, dtype="float32")

    grouped = df_sliced.sort_values([config.PATIENT_ID_COL, "ICULOS"]).groupby(config.PATIENT_ID_COL)
    pid_to_idx = {pid: i for i, pid in enumerate(patient_ids)}

    n_missing_patient = 0
    for pid, g in grouped:
        if pid not in pid_to_idx:
            continue
        i = pid_to_idx[pid]
        values = g[seq_vars].to_numpy(dtype="float32")
        t = min(len(values), max_len)
        X[i, :t, :] = values[:t, :]

    for pid in patient_ids:
        if pid not in grouped.groups:
            n_missing_patient += 1

    if n_missing_patient:
        logger.warning("%d patients had no rows in df_sliced and are fully masked/padded.", n_missing_patient)

    return X, seq_vars


def fit_sequence_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std using ONLY real (non-masked) timesteps
    from the TRAINING sequences — the leakage-safe equivalent of Phase 11's
    StandardScaler, adapted for 3D sequence data.

    CRITICAL FIX: uses np.nanmean/np.nanstd, not np.mean/np.std. Sparse
    variables (e.g. Lactate, WBC) frequently have NO real measurement for
    many patients within a short 6-hour window (see Phase 5's measurement-
    frequency findings), so `real_values` legitimately contains NaN. Plain
    np.mean/np.std over an array containing ANY NaN returns NaN for that
    entire column (not just the NaN entries) — this NaN then propagates
    through scaling to EVERY patient's value for that feature, silently
    poisoning the whole column and preventing the model from training at
    all (observed empirically: loss frozen at exactly ln(2) every epoch).
    """
    mask = ~np.all(X_train == MASK_VALUE, axis=-1)  # (n_patients, max_len) True where real data
    real_values = X_train[mask]  # (n_real_timesteps, n_features)

    with np.errstate(invalid="ignore"):
        mean = np.nanmean(real_values, axis=0)
        std = np.nanstd(real_values, axis=0)

    # A feature with ZERO real (non-NaN) measurements across the entire
    # training set would still yield NaN mean/std even with nanmean/nanstd
    # (nanmean of an all-NaN slice is NaN by definition) — guard explicitly
    # rather than let it silently reappear as a bug.
    if np.isnan(mean).any():
        bad_idx = np.where(np.isnan(mean))[0]
        logger.warning(
            "Feature indices %s have ZERO real measurements anywhere in the "
            "training set — their scaler mean/std default to 0/1 (i.e. no "
            "scaling applied; they will be constant-filled downstream).",
            bad_idx.tolist(),
        )
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.where(np.isnan(std), 1.0, std)

    std[std == 0] = 1.0  # avoid divide-by-zero for any constant feature
    return mean, std


def apply_sequence_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale real timesteps, leave masked (padded) timesteps at MASK_VALUE so Masking still works."""
    X_scaled = X.copy()
    mask = ~np.all(X == MASK_VALUE, axis=-1)
    X_scaled[mask] = (X[mask] - mean) / std
    return X_scaled


def fill_remaining_nans(X: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    After scaling, any NaN in a REAL (non-masked) timestep means that
    variable was never measured up to that hour even after LOCF (Phase 3) —
    fill with `fill_value` (0.0 in the SCALED space == the training mean),
    consistent in spirit with Phase 11's median imputation, but leave
    masked timesteps as MASK_VALUE so the Masking layer still works.
    """
    mask_real = ~np.all(X == MASK_VALUE, axis=-1, keepdims=True)
    nan_mask = np.isnan(X) & mask_real
    X_filled = np.where(nan_mask, fill_value, X)
    return X_filled
