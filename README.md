# Early Sepsis Detection Using Clinical Time-Series Data

> Research/educational ML prototype. **Not a clinical diagnostic tool.**

## Overview
An end-to-end machine learning system that predicts an ICU patient's short-term
risk of developing sepsis using the first 6 hours of hourly clinical
observations (vitals + labs), built on the PhysioNet/CinC 2019 Challenge
dataset. The project covers the full pipeline: data understanding, cleaning,
EDA, leakage-safe target construction, feature engineering, model training
and tuning (classical ML + deep learning), calibration, explainability,
error/subgroup analysis, and a Streamlit demo dashboard.

## Problem Statement
Sepsis is a life-threatening response to infection where early intervention
materially improves outcomes. This project asks: using only the clinical
data available in a patient's first 6 ICU hours, can a model flag elevated
sepsis risk before it becomes clinically obvious?

## Motivation
This is a portfolio/learning project demonstrating a complete, rigorous ML
workflow on real, imperfect clinical time-series data — with a particular
focus on avoiding target leakage, honest evaluation, and transparent
reporting of a model's actual limitations (several of which were discovered
and documented during this project, not glossed over).

## Dataset
**Source:** PhysioNet/CinC Challenge 2019 — https://physionet.org/content/challenge-2019/1.0.0/

**Actual full-dataset statistics** (`notebooks/01_data_understanding.ipynb`):

| Metric | Value |
|---|---|
| Patient files loaded | 40,336 |
| Total hourly observations | 1,552,210 |
| Duplicate rows | 0 |
| Sepsis-positive patients | 2,932 |
| Sepsis-negative patients | 37,404 |
| Patient-level sepsis prevalence | 7.269% |
| ICU length of stay: min / median / max | 8 / 39 / 336 hours |

**Missingness:** ranges from 0% (`Age`, `Gender`, `ICULOS`, `SepsisLabel`) to
99.81% (`Bilirubin_direct`). Rare specialty labs (Troponin, Lactate,
Bilirubin) are >90% missing because they're only drawn when clinically
indicated — this missingness pattern turned out to be predictive in its own
right (see Feature Engineering below).

### Data setup
Download from the PhysioNet link above and place the extracted files as:
```
data/raw/training_setA/p000001.psv ...
data/raw/training_setB/p100001.psv ...
```

## Methodology & Data Pipeline
```
Raw .psv files (src/data_loader.py)
  -> Cleaning: LOCF within-patient, implausible-value flagging, Unit1/Unit2
     "Unknown" encoding (src/data_cleaning.py)
  -> Target construction: leakage-safe 6h prediction window, verified
     patient-by-patient (src/target.py)
  -> Feature engineering: 34 clinical vars x 12 stats + 4 derived features,
     computed from RAW (pre-LOCF) data to preserve measurement-frequency
     signal (src/feature_engineering.py)
  -> Patient-level train/val/test split, stratified, zero overlap
     (src/split.py)
  -> Preprocessing: median imputation + scaling (numeric), one-hot
     (categorical), fit on TRAIN ONLY (src/preprocessing.py)
  -> Models -> Calibration -> Threshold -> Final model (src/predict.py)
```

## Feature Engineering
For each of 34 clinical variables (8 vitals + 26 labs) plus 4 derived
features (ShockIndex, PulsePressure, SaO2/FiO2 ratio, TempAbnormal):
`mean, median, min, max, std, first, last, change, abs_change, count,
missing_ratio, slope` — 464 features total after preprocessing.

**Key design decision:** aggregates are computed from RAW (pre-LOCF)
measurements, not the LOCF-filled data, because LOCF duplicating a single
reading across many hours would artificially shrink variance and inflate
measurement counts — destroying the real signal that "how often a lab was
drawn" carries (confirmed in Phase 5: Lactate was drawn ~2.3x more often in
septic patients).

**Critical exclusions:** `onset_hour`, `usable_hours`, `n_hours_used` are
retained for audit purposes but explicitly excluded from the feature matrix
— they are near-perfectly correlated with the label by construction
(`corr(label, onset_hour_is_null) = -1.0`, verified empirically).

## Models
Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM,
CatBoost (all with class-weighting, no SMOTE), and an LSTM on raw hourly
sequences (Masking -> LSTM(64) -> Dropout -> Dense -> Sigmoid).

## Evaluation
Accuracy is not used as a ranking metric (a trivial "always negative"
classifier scores 93.7% while catching zero septic patients — see Phase 12).
Primary metric: **PR-AUC** (more sensitive to minority-class performance
than ROC-AUC at ~6% prevalence); recall, specificity, and Brier score
(calibration) are reported alongside.

## Results

**Model comparison (validation set, default/tuned hyperparameters):**

| Model | ROC-AUC | PR-AUC | Recall | Specificity |
|---|---|---|---|---|
| **LightGBM (tuned)** | **0.761** | **0.240** | 49.1% | 85.1% |
| CatBoost (tuned) | 0.739 | 0.219 | 35.2% | 91.6% |
| LightGBM (default) | 0.711 | 0.207 | 29.8% | 92.9% |
| XGBoost | 0.701 | 0.194 | 19.8% | 95.3% |
| CatBoost (default) | 0.723 | 0.190 | 36.6% | 87.9% |
| Random Forest | 0.703 | 0.180 | 15.2% | 96.4% |
| Logistic Regression | 0.705 | 0.164 | 58.5% | 74.1% |
| Decision Tree | 0.673 | 0.149 | 57.7% | 71.3% |
| LSTM | 0.528 | 0.068 | 43.4% | 61.4% |

Deep learning underperformed tree-based models here — plausibly due to the
limited training size (~1,724 positive patients) and short (\u22646h) sequences,
a regime where tree ensembles on engineered features are known to compete
well against sequence models.

**Final model: LightGBM (tuned) + isotonic calibration, threshold = 0.1**

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.7664 | 0.7533 |
| PR-AUC | 0.2307 | 0.2232 |
| Recall (sensitivity) | 57.2% | 52.4% |
| Specificity | 81.0% | 81.6% |
| Accuracy | 79.5% | 79.8% |
| Brier score | 0.0527 | 0.0533 |

**A note on accuracy:** it is reported above for familiarity, but is
deliberately NOT used anywhere in this project to select or rank models
(see Phase 12). At ~6% positive prevalence, a trivial "always predict
negative" classifier scores **93.7% accuracy** — higher than this model's
79.8% — while catching zero septic patients. This model's lower accuracy
is the direct, expected cost of catching 52% of positives instead of 0%;
PR-AUC and recall are the metrics that actually reflect its clinical
usefulness. Always present accuracy alongside this trivial baseline, never
in isolation.

Validation and test metrics are closely matched throughout tuning/threshold
selection, indicating the pipeline generalizes rather than overfitting to
the validation set. Isotonic calibration reduced Brier score by ~57%
(0.125 -> 0.053) versus the uncalibrated model.

## Explainability
SHAP (`notebooks/18_shap_explainability.ipynb`) on the underlying LightGBM
model showed measurement-frequency features (`Lactate_count`, `FiO2_count`,
`SBP_count`, `*_missing_ratio`) among the top predictors — consistent with
Phase 5's finding that clinicians order more tests when suspecting sepsis.
`HospAdmTime` was unexpectedly the single most important feature; this is
noted as an open question rather than a fully explained mechanism.

## Error & Subgroup Analysis
- **False negatives** are not "close misses": their predicted probabilities
  (mean 0.047, max 0.088) sit far below the 0.1 threshold, and their vitals
  looked closer to normal (lower HR/Resp, less Lactate testing) than true
  positives — these appear to be genuinely subtler presentations.
- **False positives clinically resemble true positives** far more than true
  negatives (near-identical HR/Resp/ShockIndex/Lactate_count to TPs) —
  suggesting many false alarms are "near-miss" at-risk patients rather than
  arbitrary model errors.
- **Subgroup disparities found** (validation set): source_set B (41.0%
  recall vs. 67.0% for source_set A), patients aged 80+ (50.0% recall, 0.133
  PR-AUC vs. ~0.25 for other age groups), and a ~14-point recall gap by
  gender and a ~20-point gap by ICU unit. Root cause not established from
  this dataset alone (see Limitations).

## Dashboard
```bash
streamlit run app/streamlit_app.py
```
Interactive UI: patient inputs, live risk score, SHAP explanation for that
specific patient, and a data-completeness warning (see Limitations) when
fewer than 6 hours of data are provided.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run notebooks `01` through `21` in order (each documents what to verify
before proceeding), then:
```bash
pytest tests/ -v                       # 37 unit tests
streamlit run app/streamlit_app.py     # interactive dashboard
```

## Project Structure
```
data/{raw,interim,processed}/  notebooks/  src/  models/  reports/  app/  tests/
```
See repository tree for full layout.

## Limitations

1. **Indirect window-length leakage (critical, discovered in Phase 24):**
   every sepsis-NEGATIVE training patient's window covered the full 6
   hours, while POSITIVE patients' windows were truncated before their
   onset (as short as 1 hour) — so "fewer real hours of data" occurred,
   by construction, almost exclusively in positive training patients.
   Engineered `_count` features can act as an indirect proxy for this,
   meaning a genuinely new patient who simply hasn't yet accumulated 6
   hours of ICU data may receive an inflated risk score. Mitigated (not
   fully fixed) via an explicit `data_completeness_warning` in
   `src/predict.py` and the Streamlit app. A full fix would require
   re-deriving negative patients' window truncation and re-running
   Phases 8-19.
2. **Subgroup disparities** (age 80+, one gender, one ICU unit, and
   source_set B) show meaningfully lower recall — documented in Phase 22,
   not resolved. Root cause (data collection artifact vs. true population
   difference vs. sample-size noise) cannot be established from this
   dataset alone.
3. **Excluded patients:** 965 patients (370 with onset\u22641h, 595 whose raw
   file didn't start recording within the usable window) receive no
   prediction under this task definition.
4. Not clinically validated; not evaluated prospectively; no regulatory
   review. **Must not be used for real patient care.**

## Ethical Considerations
- SHAP explanations describe model associations, not medical causality.
- Similar subgroup metrics do not, by themselves, establish fairness.
- Any real-world deployment would require prospective validation,
  clinician-in-the-loop review, and resolution of the limitations above.

## Future Work
- Phase 9 (deferred): compare 6h/12h/24h prediction windows using the same
  leakage-safe methodology.
- Re-derive negative-patient window truncation to remove the indirect
  leakage channel described above.
- Investigate `HospAdmTime`'s outsized SHAP importance.
- Prospective/external validation on a separate hospital system.

## References
- Reyna, M. et al. "Early Prediction of Sepsis From Clinical Data: The
  PhysioNet/Computing in Cardiology Challenge 2019." Critical Care
  Medicine, 2020.
- PhysioNet/CinC Challenge 2019 dataset:
  https://physionet.org/content/challenge-2019/1.0.0/
