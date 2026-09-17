"""
split.py
=========
Phase 10 — Leakage-safe, patient-level train/validation/test split.

CRITICAL RULE: a patient's hourly rows must never be split across sets —
splitting is done at the PATIENT level (using patient_id as the group key),
never at the row level. This module also explicitly verifies the three
pairwise intersections are empty, per the project's mandatory leakage checks.

Stratification: the split is stratified by the patient-level `label`
column so that the ~6.26% positive prevalence (observed in Phase 8) is
approximately preserved in all three sets — an unstratified random split
on such an imbalanced target risks a validation or test set with too few
positive patients to evaluate reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)


def patient_level_split(
    patient_df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    test_frac: float = config.TEST_FRAC,
    seed: int = config.RANDOM_SEED,
    label_col: str = "label",
) -> dict[str, np.ndarray]:
    """
    Split patient_df's patient_id values into train/val/test, stratified by
    label_col, using a reproducible seed.

    Returns a dict: {"train": array of patient_ids, "val": ..., "test": ...}
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9, "Fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for label_value, group in patient_df.groupby(label_col):
        pids = group[config.PATIENT_ID_COL].to_numpy().copy()
        rng.shuffle(pids)

        n = len(pids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Remainder goes to test to avoid rounding-induced patient loss/duplication
        n_test = n - n_train - n_val

        splits["train"].extend(pids[:n_train])
        splits["val"].extend(pids[n_train:n_train + n_val])
        splits["test"].extend(pids[n_train + n_val:])

        logger.info("Label=%s: %d patients -> train=%d, val=%d, test=%d",
                    label_value, n, n_train, n_val, n_test)

    result = {k: np.array(v) for k, v in splits.items()}
    return result


def verify_no_overlap(splits: dict[str, np.ndarray]) -> dict[str, int]:
    """
    Explicitly verify (not assume) that train/val/test patient sets are
    pairwise disjoint. Returns the size of each pairwise intersection —
    all must be 0 before proceeding to Phase 11.
    """
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    overlaps = {
        "train_val": len(train_set & val_set),
        "train_test": len(train_set & test_set),
        "val_test": len(val_set & test_set),
    }

    for pair, n in overlaps.items():
        if n > 0:
            logger.error("LEAKAGE DETECTED: %d overlapping patients between %s!", n, pair)
        else:
            logger.info("Verified: %s intersection is empty.", pair)

    total = len(train_set) + len(val_set) + len(test_set)
    total_unique = len(train_set | val_set | test_set)
    if total != total_unique:
        logger.error("Patient count mismatch: sum of set sizes (%d) != union size (%d) — "
                     "a patient appears in more than one split.", total, total_unique)

    return overlaps


def summarize_split(patient_df: pd.DataFrame, splits: dict[str, np.ndarray], label_col: str = "label") -> pd.DataFrame:
    """Report size and class balance of each split, for the README/report tables."""
    rows = []
    for split_name, pids in splits.items():
        sub = patient_df[patient_df[config.PATIENT_ID_COL].isin(pids)]
        rows.append({
            "split": split_name,
            "n_patients": len(sub),
            "n_positive": int((sub[label_col] == 1).sum()),
            "n_negative": int((sub[label_col] == 0).sum()),
            "positive_pct": round((sub[label_col] == 1).mean() * 100, 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    patient_df = pd.read_parquet(config.PROCESSED_PATIENT_LEVEL_PARQUET)
    splits = patient_level_split(patient_df)
    overlaps = verify_no_overlap(splits)
    print(overlaps)
    print(summarize_split(patient_df, splits))
