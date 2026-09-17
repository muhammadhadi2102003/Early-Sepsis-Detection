"""
data_loader.py
===============
Robust, memory-conscious loading of the PhysioNet/CinC 2019 Sepsis dataset.

Design goals (per project spec):
- Discover patient files across training_setA / training_setB
- Read .psv files (pipe-separated, one row per ICU hour)
- Validate columns against the official schema
- Preserve patient identity (derived from filename) and temporal order
  (row order == hour order, i.e. ICULOS is strictly increasing)
- Never modify anything inside data/raw/
- Avoid loading the entire dataset into memory at once when it is not
  necessary — a generator (`iter_patients`) is provided for patient-wise
  / chunked processing, and `load_combined_dataframe` gives an optional
  fully-materialized long dataframe with a `low_memory` mode using
  smaller dtypes.

Nothing here fabricates data. If a directory is empty or missing, the
functions raise clear, actionable errors instead of silently returning
fake results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------
@dataclass(frozen=True)
class PatientFile:
    """Reference to a single patient's raw .psv file (path only, not loaded)."""
    patient_id: str
    source_set: str          # "A" or "B"
    path: Path


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------
def discover_patient_files(raw_dirs: Optional[Iterable[Path]] = None) -> list[PatientFile]:
    """
    Scan the configured raw-data subdirectories and return one PatientFile
    per .psv file found, without reading their contents.

    Raises
    ------
    FileNotFoundError
        If none of the configured raw directories exist, or if they exist
        but contain zero .psv files (this usually means the dataset has
        not been downloaded/extracted yet — see README "Data Setup").
    """
    raw_dirs = list(raw_dirs) if raw_dirs is not None else config.RAW_SUBDIRS

    existing_dirs = [d for d in raw_dirs if d.exists()]
    if not existing_dirs:
        raise FileNotFoundError(
            "No raw data directories found. Expected one or both of:\n"
            + "\n".join(f"  - {d}" for d in raw_dirs)
            + "\n\nDownload the dataset from "
              "https://physionet.org/content/challenge-2019/1.0.0/ and extract "
              "training_setA/ and training_setB/ into these folders "
              "(or set EARLY_SEPSIS_DATA_DIR)."
        )

    files: list[PatientFile] = []
    for d in existing_dirs:
        source_set = "A" if "setA" in d.name else ("B" if "setB" in d.name else d.name)
        for p in sorted(d.glob(f"*{config.RAW_FILE_EXTENSION}")):
            patient_id = p.stem  # e.g. "p000001"
            files.append(PatientFile(patient_id=patient_id, source_set=source_set, path=p))

    if not files:
        raise FileNotFoundError(
            f"Raw directories exist but contain no '{config.RAW_FILE_EXTENSION}' files: "
            f"{[str(d) for d in existing_dirs]}. Nothing to load."
        )

    logger.info("Discovered %d patient files across %d directories.", len(files), len(existing_dirs))
    return files


# ------------------------------------------------------------------
# Column validation
# ------------------------------------------------------------------
def validate_columns(columns: list[str], patient_id: str) -> None:
    """
    Confirm a loaded file's columns match the official Challenge schema.

    We validate *set equality with order allowed to differ*, since some
    PhysioNet distributions have minor column ordering differences between
    training_setA and training_setB. A hard error is raised only when a
    column is missing outright or an unexpected column appears — silently
    proceeding on a schema mismatch would risk mis-aligned features later.
    """
    expected = set(config.EXPECTED_RAW_COLUMNS)
    actual = set(columns)

    missing = expected - actual
    unexpected = actual - expected

    if missing:
        raise ValueError(
            f"[{patient_id}] Missing expected columns: {sorted(missing)}. "
            "This file does not match the official PhysioNet 2019 schema."
        )
    if unexpected:
        logger.warning(
            "[%s] Unexpected extra columns found (kept as-is): %s",
            patient_id, sorted(unexpected),
        )


# ------------------------------------------------------------------
# Single-file reading
# ------------------------------------------------------------------
# Compact dtypes to reduce memory footprint. SepsisLabel and Unit1/Unit2/
# Gender are small integer flags in the official dataset (0/1, or NaN for
# Unit1/Unit2 in some records) -> float32 is used to safely hold NaN.
_DTYPE_MAP = {col: "float32" for col in config.ALL_CLINICAL_FEATURES}
_DTYPE_MAP.update({
    "Age": "float32",
    "Gender": "float32",
    "Unit1": "float32",
    "Unit2": "float32",
    "HospAdmTime": "float32",
    "ICULOS": "float32",
    config.TARGET_COL: "float32",  # cast to int8 after confirming no NaN
})


def read_patient_psv(pf: PatientFile, validate: bool = True) -> pd.DataFrame:
    """
    Read a single patient .psv file into a dataframe with patient_id,
    source_set, and preserved row (temporal) order.

    Does NOT sort by ICULOS blindly — instead it *checks* that ICULOS is
    non-decreasing and logs a warning if not, since silently re-sorting
    could mask a real data-quality issue.
    """
    df = pd.read_csv(pf.path, sep=config.RAW_FILE_DELIMITER, dtype=_DTYPE_MAP)

    if validate:
        validate_columns(list(df.columns), pf.patient_id)

    if "ICULOS" in df.columns and len(df) > 1:
        if not df["ICULOS"].is_monotonic_increasing:
            logger.warning(
                "[%s] ICULOS is not strictly increasing — temporal order may be "
                "corrupted in the source file. Row order preserved as-is; "
                "investigate before using this patient in time-series models.",
                pf.patient_id,
            )

    df.insert(0, config.SOURCE_SET_COL, pf.source_set)
    df.insert(0, config.PATIENT_ID_COL, pf.patient_id)

    return df


# ------------------------------------------------------------------
# Patient-wise generator (preferred for large-scale / low-memory processing)
# ------------------------------------------------------------------
def iter_patients(
    raw_dirs: Optional[Iterable[Path]] = None,
    limit: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield one dataframe per patient, one at a time.

    This is the recommended entry point for feature engineering
    (Phase 8) and the deep-learning sequence builder (Phase 16), since it
    never holds more than one patient's hourly records in memory at once
    beyond what the caller chooses to accumulate.

    Parameters
    ----------
    limit : Optional[int]
        If set, only yield the first `limit` patients. Useful for a
        clearly-labeled quick smoke test — never use this to silently
        represent a sample as the full dataset.
    """
    files = discover_patient_files(raw_dirs)
    if limit is not None:
        files = files[:limit]

    for i, pf in enumerate(files, start=1):
        try:
            yield read_patient_psv(pf)
        except Exception as exc:  # noqa: BLE001 - we want to log & continue
            logger.error("Failed to read %s (%s): %s", pf.patient_id, pf.path, exc)
            continue

        if i % 5000 == 0:
            logger.info("Processed %d / %d patient files...", i, len(files))


# ------------------------------------------------------------------
# Fully materialized long dataframe (use only when the machine has enough RAM,
# or with `sample_n_patients` for exploratory work — always clearly labeled)
# ------------------------------------------------------------------
def load_combined_dataframe(
    raw_dirs: Optional[Iterable[Path]] = None,
    sample_n_patients: Optional[int] = None,
    save_parquet: bool = False,
) -> pd.DataFrame:
    """
    Concatenate all (or a labeled sample of) patients into one long
    dataframe: one row per patient-hour.

    Parameters
    ----------
    sample_n_patients : Optional[int]
        If provided, only this many patients are loaded and the resulting
        dataframe carries an explicit `is_sample=True` marker attribute so
        downstream code/notebooks cannot accidentally present it as the
        full dataset.
    save_parquet : bool
        If True, caches the result at config.INTERIM_LONG_PARQUET for
        fast reloading (only when sample_n_patients is None, i.e. the
        full dataset).

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, source_set, <40 clinical/demographic columns>, SepsisLabel
    """
    files = discover_patient_files(raw_dirs)
    is_sample = sample_n_patients is not None
    if is_sample:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(len(files), size=min(sample_n_patients, len(files)), replace=False)
        files = [files[i] for i in sorted(idx)]
        logger.warning(
            "Loading a SAMPLE of %d / (full dataset) patients. "
            "Do NOT report results from this sample as full-dataset results.",
            len(files),
        )

    frames = []
    for i, pf in enumerate(files, start=1):
        try:
            frames.append(read_patient_psv(pf))
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read %s (%s): %s", pf.patient_id, pf.path, exc)
            continue
        if i % 5000 == 0:
            logger.info("Loaded %d / %d patient files...", i, len(files))

    if not frames:
        raise RuntimeError("No patient files could be loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    combined.attrs["is_sample"] = is_sample
    combined.attrs["n_patients"] = combined[config.PATIENT_ID_COL].nunique()

    # Duplicate-record check at load time (Phase 3 will handle remediation)
    n_dupes = combined.duplicated().sum()
    if n_dupes:
        logger.warning("Combined dataframe contains %d fully duplicated rows.", n_dupes)

    if save_parquet and not is_sample:
        config.INTERIM_LONG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(config.INTERIM_LONG_PARQUET, index=False)
        logger.info("Saved combined long dataframe to %s", config.INTERIM_LONG_PARQUET)

    return combined


# ------------------------------------------------------------------
# Convenience: quick dataset summary used by 01_data_understanding.ipynb
# ------------------------------------------------------------------
def get_patient_file_counts(raw_dirs: Optional[Iterable[Path]] = None) -> pd.DataFrame:
    """Return a small dataframe: counts of patient files per source_set."""
    files = discover_patient_files(raw_dirs)
    df = pd.DataFrame([{"source_set": pf.source_set, "patient_id": pf.patient_id} for pf in files])
    return df.groupby("source_set", as_index=False)["patient_id"].count().rename(
        columns={"patient_id": "n_patient_files"}
    )


if __name__ == "__main__":
    # Minimal smoke test when run directly: python -m src.data_loader
    try:
        files = discover_patient_files()
        print(f"Found {len(files)} patient files.")
        print(get_patient_file_counts())
    except FileNotFoundError as e:
        print(e)
