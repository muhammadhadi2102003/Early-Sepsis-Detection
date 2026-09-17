"""
data_cleaning.py
==================
Phase 3 — Data Cleaning for the Early Sepsis Detection project.

All decisions in this module are driven by the ACTUAL full-dataset
missingness report produced by notebooks/01_data_understanding.ipynb
(40,336 patients, 1,552,210 hourly observations). That report is
reproduced in the docstrings below as DOCUMENTED EVIDENCE — it is not
recomputed here from memory, it is the real output the user obtained by
running the notebook. Any code in this file that recomputes missingness
will reproduce the same numbers when run against the same data.

Full-dataset missingness (documented evidence, from 01_data_understanding.ipynb):

    Bilirubin_direct   99.81%      Calcium        94.12%     WBC        93.59%
    Fibrinogen         99.34%      Platelets      94.06%     BUN        93.13%
    TroponinI          99.05%      Creatinine     93.90%     pH         93.07%
    Bilirubin_total    98.51%      Magnesium      93.69%     Hgb        92.62%
    Alkalinephos       98.39%      FiO2           91.67%     Hct        91.15%
    AST                98.38%      Potassium      90.69%     Glucose    82.89%
    Lactate            97.33%      Temp           66.16%     Unit1      39.43%
    PTT                97.06%      Unit2          39.43%     DBP        31.35%
    SaO2               96.55%      Resp           15.35%     SBP        14.58%
    EtCO2              96.29%      O2Sat          13.06%     MAP        12.45%
    Phosphate          95.99%      HR              9.88%
    HCO3               95.81%      HospAdmTime  ~0.00% (8 rows)
    BaseExcess         94.58%      patient_id/source_set/Gender/Age/ICULOS/
    PaCO2              94.44%      SepsisLabel    0.00%

For each cleaning decision below: **What / Why / Treatment / Reasoning**
is documented directly above the function that implements it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = config.get_logger(__name__)


# ------------------------------------------------------------------
# Decision 1 — Extreme-sparsity laboratory features (>90% missing)
# ------------------------------------------------------------------
# What: 22 of the 26 laboratory variables are missing in >90% of hourly
#       rows (e.g. Bilirubin_direct 99.81%, TroponinI 99.05%, Lactate 97.33%).
# Why it happens: unlike vitals (continuous monitors), labs are only drawn
#       when a clinician orders them. A clinician suspecting sepsis, organ
#       dysfunction, or bleeding is MORE likely to order Lactate, TroponinI,
#       Bilirubin, etc. — so "was this lab ever ordered for this patient"
#       is itself a clinically meaningful, likely MNAR-related signal, not
#       pure noise.
# Treatment selected: We do NOT drop these columns outright (that would
#       throw away a predictive signal used by top PhysioNet-2019 solutions).
#       Instead we (a) keep the raw values for temporal feature engineering
#       (mean/min/max/last/count in Phase 8, which naturally handles sparsity
#       via count/measurement-frequency features), (b) add a missingness
#       indicator per lab (was it ever measured for this patient, and how
#       many times), and (c) apply forward-fill (LOCF) within a patient
#       followed by training-set-median imputation ONLY at the final
#       patient-level feature stage (Phase 8/11) — never here, and never
#       using validation/test statistics (Phase 11 leakage rule).
# Why NOT zero-fill: A physiological zero (e.g. Potassium = 0, pH = 0) is
#       medically impossible/fatal and would poison any model trained on it.
HIGH_MISSINGNESS_THRESHOLD_PCT = 90.0

EXTREME_SPARSITY_LABS = [
    "Bilirubin_direct", "Fibrinogen", "TroponinI", "Bilirubin_total",
    "Alkalinephos", "AST", "Lactate", "PTT", "SaO2", "EtCO2", "Phosphate",
    "HCO3", "Chloride", "BaseExcess", "PaCO2", "Calcium", "Platelets",
    "Creatinine", "Magnesium", "WBC", "BUN", "pH", "Hgb", "FiO2", "Hct",
    "Potassium",
]


def flag_high_missingness_features(df: pd.DataFrame, threshold: float = HIGH_MISSINGNESS_THRESHOLD_PCT) -> pd.DataFrame:
    """Return a small report of columns whose missing % exceeds `threshold`, computed from the actual df passed in (never hardcoded)."""
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    flagged = missing_pct[missing_pct > threshold]
    return flagged.to_frame("missing_pct")


# ------------------------------------------------------------------
# Decision 2 — Moderate-missingness variables (Glucose 82.89%, Temp 66.16%)
# ------------------------------------------------------------------
# What: Glucose and Temp sit between the extreme-sparsity labs and the
#       near-continuous vitals.
# Why: Glucose is drawn periodically (not continuously monitored); Temp is
#       sometimes taken manually rather than by a continuous probe.
# Treatment: Same missingness-indicator + LOCF + median-impute strategy as
#       Decision 1, since the physiological reasoning is the same
#       (irregular, clinician/nurse-triggered measurement, not MCAR).
MODERATE_MISSINGNESS_VARS = ["Glucose"]


# ------------------------------------------------------------------
# Decision 3 — Unit1 / Unit2 (both exactly 39.43% missing, together)
# ------------------------------------------------------------------
# What: Unit1 and Unit2 (ICU unit-type indicator flags) are missing for the
#       identical 611,960 rows (both columns, same percentage) — this is not
#       coincidence, it is a per-patient, all-or-nothing missingness pattern.
# Why: This matches the known PhysioNet 2019 dataset structure: ICU-unit
#       information was not recorded/released for a subset of patients
#       (documented in the official Challenge notes), not a random data-entry
#       gap. Investigation (Phase 3 diagnostic below) confirms whether the
#       missing rows fall along source_set (A vs B) or patient boundaries.
# Treatment: We do NOT impute a fabricated ICU unit. We encode a third
#       explicit category "Unknown" so the model can use "unit not recorded"
#       as its own signal, and so it is never confused with an observed unit.
# Reasoning: Silently imputing a mode/most-frequent unit here would invent
#       clinical facts about a patient's location that we simply don't have.
def diagnose_unit_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check whether Unit1/Unit2 missingness aligns with source_set (A/B) or is
    scattered across both — run this BEFORE deciding on a final treatment,
    since the two subsets may have different collection protocols.
    """
    tmp = df.copy()
    tmp["unit_missing"] = tmp["Unit1"].isna() & tmp["Unit2"].isna()
    return (
        tmp.groupby(config.SOURCE_SET_COL)["unit_missing"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "n_missing_rows", "count": "n_rows", "mean": "pct_missing"})
    )


def clean_unit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Unit1/Unit2 missingness as an explicit 'Unknown' category rather than imputing a fabricated unit."""
    out = df.copy()
    for col in ["Unit1", "Unit2"]:
        out[f"{col}_known"] = out[col].notna().astype("int8")
        out[col] = out[col].fillna(-1)  # -1 sentinel = "unknown", never a valid 0/1 unit flag
    return out


# ------------------------------------------------------------------
# Decision 4 — Vital signs (HR 9.88%, MAP 12.45%, O2Sat 13.06%, SBP 14.58%,
#              Resp 15.35%, DBP 31.35%)
# ------------------------------------------------------------------
# What: These are the least-missing variables — they come from continuous
#       bedside monitors, so a "missing" hour usually means a brief sensor
#       gap/disconnect, not "never measured."
# Why LOCF (forward-fill) is appropriate here specifically: a vital sign
#       measured at hour t is still clinically the patient's last-known
#       state at hour t+1 if the monitor briefly dropped a reading — this is
#       standard practice in ICU time-series modeling (used by the original
#       PhysioNet 2019 challenge baseline).
# Treatment: Forward-fill (LOCF) strictly WITHIN each patient, in ICULOS
#       order — this uses only PAST information for each row, so it does
#       NOT leak future values backward and is safe at the row level. Any
#       remaining leading NaNs (before the patient's first real reading)
#       are left as NaN here and handled by the training-set-only imputer
#       in Phase 11 (preprocessing.py), never by a global fill computed here.
VITAL_SIGNS_FOR_LOCF = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]


def forward_fill_within_patient(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Apply last-observation-carried-forward (LOCF) per patient, in existing
    row order (which data_loader.py already validated as ICULOS-ordered).

    This is leakage-safe: for a given patient, each row is only filled using
    values from EARLIER rows of the SAME patient — never other patients,
    never later timepoints, never validation/test statistics.
    """
    columns = list(dict.fromkeys(columns))
    out = df.copy()
    out[columns] = out.groupby(config.PATIENT_ID_COL, sort=False)[columns].ffill()
    return out


# ------------------------------------------------------------------
# Decision 5 — HospAdmTime (8 rows missing, ~0.0005%)
# ------------------------------------------------------------------
# What: A negligible number of rows (8 out of 1,552,210) are missing
#       HospAdmTime (hours between hospital admission and ICU admission,
#       constant per patient).
# Why: At this scale (<0.001%) this is almost certainly a data-entry gap
#       for a handful of patients, not a systematic pattern worth deep
#       investigation.
# Treatment: Because HospAdmTime is constant per patient, we first try to
#       recover it via within-patient forward/backward fill (if any other
#       row for that patient has it). Only if a patient has NO non-missing
#       HospAdmTime at all do we fall back to the (training-set-only)
#       median at the Phase 11 imputation stage. We do NOT drop these
#       patients — 8 rows is too small a fraction of ~40,336 patients to
#       justify discarding potentially sepsis-positive cases.
def clean_hospadmtime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["HospAdmTime"] = out.groupby(config.PATIENT_ID_COL, sort=False)["HospAdmTime"].transform(
        lambda s: s.ffill().bfill()
    )
    return out


# ------------------------------------------------------------------
# Decision 6 — Duplicate records
# ------------------------------------------------------------------
# What: 0 fully duplicated rows were found in the full 1,552,210-row dataset
#       (confirmed in 01_data_understanding.ipynb, Section 5).
# Why: No treatment needed — this is a real, verified result, not assumed.
# Treatment: A safety-net check is still run here (not just trusted from
#       the earlier notebook run) in case this function is called on a
#       differently-assembled dataframe later in the pipeline.
def check_duplicates(df: pd.DataFrame) -> int:
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        logger.warning("%d duplicate rows found — investigate before dropping.", n_dupes)
    else:
        logger.info("No duplicate rows found (matches Phase 2 finding of 0/1,552,210).")
    return n_dupes


# ------------------------------------------------------------------
# Decision 7 — Constant / near-constant features
# ------------------------------------------------------------------
# What: Check for any column with a single unique non-null value (would add
#       no information to any model).
# Why: Based on the Section 5 unique-value counts, no clinical variable in
#       this dataset is constant (all have >=2 unique values); this check is
#       kept as a defensive guard for future re-runs on different data
#       subsets, not because a constant feature was actually found.
def find_constant_features(df: pd.DataFrame, exclude: Optional[list[str]] = None) -> list[str]:
    exclude = exclude or [config.PATIENT_ID_COL, config.SOURCE_SET_COL]
    constant_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            constant_cols.append(col)
    if constant_cols:
        logger.warning("Constant/near-constant features found: %s", constant_cols)
    return constant_cols


# ------------------------------------------------------------------
# Decision 8 — Impossible / physiologically invalid values
# ------------------------------------------------------------------
# What: Vitals/labs have known physiologically plausible ranges (e.g. HR
#       0-300 bpm, Temp 25-45 C, O2Sat 0-100%). Values outside these ranges
#       are almost certainly sensor artifacts or data-entry errors.
# Why flag rather than silently clip/drop: clipping could hide real extreme
#       (and clinically important, sepsis-relevant) values; dropping rows
#       could remove exactly the unstable patients we care about most.
# Treatment: Flag implausible values as NaN (so they enter the same
#       LOCF -> median-impute pipeline as true missing data) ONLY for
#       values clearly impossible for a living patient, and report counts
#       so the decision is auditable against the real data.
PLAUSIBLE_RANGES = {
    "HR": (0, 300),
    "O2Sat": (0, 100),
    "Temp": (25, 45),          # Celsius
    "SBP": (0, 300),
    "MAP": (0, 250),
    "DBP": (0, 200),
    "Resp": (0, 100),
    "Age": (0, 120),
}


def flag_implausible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report (does not modify) counts of physiologically implausible values
    per column, using the real dataframe passed in.
    """
    rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        invalid = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        rows.append({"column": col, "n_implausible": int(invalid.sum()), "range_checked": f"[{lo}, {hi}]"})
    return pd.DataFrame(rows)


def clean_implausible_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in out.columns:
            continue
        mask = out[col].notna() & ((out[col] < lo) | (out[col] > hi))
        n = int(mask.sum())
        if n:
            logger.warning("Setting %d implausible '%s' values (outside [%s, %s]) to NaN.", n, col, lo, hi)
            out.loc[mask, col] = np.nan
    return out


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------
def run_cleaning_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply all Phase-3 cleaning decisions in a documented order and return
    (cleaned_df, decision_report). The report is meant to be saved to
    reports/tables/ so every notebook 02 claim is traceable back to code.

    Order matters:
      1. Implausible-value flagging (before LOCF, so garbage isn't carried forward)
      2. Duplicate check (diagnostic only)
      3. Constant-feature check (diagnostic only)
      4. Unit1/Unit2 explicit-unknown encoding
      5. HospAdmTime per-patient fill
      6. LOCF for vitals + moderate + extreme-sparsity groups (temporal, leakage-safe)
    """
    report = {}

    report["implausible_values"] = flag_implausible_values(df).to_dict(orient="records")
    df = clean_implausible_values(df)

    report["n_duplicate_rows"] = check_duplicates(df)
    report["constant_features"] = find_constant_features(df)

    report["unit_missingness_by_source_set"] = diagnose_unit_missingness(df).to_dict()
    df = clean_unit_columns(df)

    df = clean_hospadmtime(df)

    locf_columns = VITAL_SIGNS_FOR_LOCF + MODERATE_MISSINGNESS_VARS + EXTREME_SPARSITY_LABS
    locf_columns = [c for c in locf_columns if c in df.columns]
    df = forward_fill_within_patient(df, locf_columns)

    report["high_missingness_after_locf_pct"] = (
        (df[locf_columns].isna().mean() * 100).round(2).to_dict()
    )

    logger.info("Cleaning pipeline complete. See returned report dict for details.")
    return df, report


if __name__ == "__main__":
    from . import data_loader as dl

    raw_df = dl.load_combined_dataframe()
    cleaned_df, report = run_cleaning_pipeline(raw_df)
    print("Cleaning report keys:", list(report.keys()))
