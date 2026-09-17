"""
config.py
=========
Central configuration for the Early Sepsis Detection project.

Every path, seed, and constant used across notebooks and src/ modules
is defined here so the project is reproducible and easy to reconfigure
on a different machine.

Nothing in this file should require you to edit any other file when you
move the dataset to a new location — just change DATA_DIR (or set the
EARLY_SEPSIS_DATA_DIR environment variable) below.
"""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------------
# This resolves to the Early-Sepsis-Detection/ folder regardless of
# which notebook or script imports config.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# DATA PATHS
# ------------------------------------------------------------------
# You can override this without touching any code, e.g.:
#   export EARLY_SEPSIS_DATA_DIR=/mnt/d/datasets/physionet_sepsis/raw
DATA_DIR = Path(os.environ.get("EARLY_SEPSIS_DATA_DIR", PROJECT_ROOT / "data" / "raw"))

# The official PhysioNet/CinC 2019 release ships two sub-cohorts.
# Place the extracted .psv files here exactly as PhysioNet distributes them:
#   data/raw/training_setA/p000001.psv, p000002.psv, ...
#   data/raw/training_setB/p100001.psv, p100002.psv, ...
TRAINING_SET_A_DIR = DATA_DIR / "training_setA"
TRAINING_SET_B_DIR = DATA_DIR / "training_setB"
RAW_SUBDIRS = [TRAINING_SET_A_DIR, TRAINING_SET_B_DIR]

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

for _d in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# FILE NAMING
# ------------------------------------------------------------------
RAW_FILE_EXTENSION = ".psv"          # PhysioNet raw format: pipe-separated values
RAW_FILE_DELIMITER = "|"

# Cached, patient-wise-assembled long dataframe (hour-level rows)
INTERIM_LONG_PARQUET = INTERIM_DIR / "sepsis_hourly_long.parquet"

# Final patient-level engineered feature table used for modeling
PROCESSED_PATIENT_LEVEL_PARQUET = PROCESSED_DIR / "patient_level_features.parquet"

# ------------------------------------------------------------------
# OFFICIAL PHYSIONET/CinC 2019 COLUMN SCHEMA
# ------------------------------------------------------------------
# Source: https://physionet.org/content/challenge-2019/1.0.0/
# 40 clinical variables + demographics + ICULOS + SepsisLabel = 41 columns.
VITAL_SIGNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LABORATORY_VALUES = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

TARGET_COL = "SepsisLabel"

# Patient identifier is NOT a column in the raw .psv files — it is derived
# from the filename (e.g. "p000001.psv" -> patient_id "p000001"). This is
# added explicitly by data_loader.py and must be treated as a group key,
# never as a predictive feature.
PATIENT_ID_COL = "patient_id"
SOURCE_SET_COL = "source_set"       # "A" or "B", useful for stratified checks

ALL_CLINICAL_FEATURES = VITAL_SIGNS + LABORATORY_VALUES
EXPECTED_RAW_COLUMNS = VITAL_SIGNS + LABORATORY_VALUES + DEMOGRAPHICS + [TARGET_COL]

# ------------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# PATIENT-LEVEL SPLIT RATIOS (Phase 10)
# ------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

# ------------------------------------------------------------------
# THRESHOLD GRID FOR OPTIMIZATION (Phase 18)
# ------------------------------------------------------------------
THRESHOLD_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# ------------------------------------------------------------------
# PREDICTION WINDOWS TO COMPARE (Phase 9)
# ------------------------------------------------------------------
# Number of ICU hours of history used to build patient-level features
# for the "early prediction" experiments. The PRIMARY experiment is
# defined explicitly in notebook 07 / train.py and must not be mixed
# with these exploratory windows without labeling.
PREDICTION_WINDOWS_HOURS = [6, 12, 24]
PRIMARY_PREDICTION_WINDOW_HOURS = 6   # matches the official Challenge framing

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
LOG_LEVEL = os.environ.get("EARLY_SEPSIS_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str):
    """Return a configured module-level logger (call once per module)."""
    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
    return logger
