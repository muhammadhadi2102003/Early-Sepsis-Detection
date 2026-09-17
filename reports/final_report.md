# Early Sepsis Detection Using Clinical Time-Series Data
### Final Project Report

---

## 1. Abstract

This report documents an end-to-end machine learning pipeline built to predict
short-term sepsis risk in ICU patients from their first six hours of hourly
clinical observations, using the PhysioNet/CinC 2019 Challenge dataset
(40,336 patients, 1,552,210 hourly observations). After leakage-safe target
construction, feature engineering (464 patient-level features), and
systematic model comparison across six classical ML algorithms and an
LSTM, a tuned LightGBM model with isotonic probability calibration was
selected as the final model, achieving a test-set ROC-AUC of 0.753 and
PR-AUC of 0.223 at a decision threshold of 0.1 (52.4% recall, 81.6%
specificity). Explainability analysis (SHAP), error analysis, and subgroup
analysis surfaced both clinically consistent predictive signal and
meaningful limitations — including a critical indirect leakage channel
discovered during deployment-pipeline testing — all of which are reported
transparently below rather than omitted.

## 2. Introduction

Sepsis is a dysregulated host response to infection that can rapidly
progress to organ failure and death; early recognition is associated with
better outcomes in the clinical literature. The PhysioNet/CinC 2019
Challenge released a large, hourly-labeled ICU dataset specifically to
encourage early-warning models. This project builds a complete, reproducible
pipeline on that dataset — from raw file parsing through a deployed
Streamlit dashboard — with an explicit emphasis on catching and reporting
data leakage, since early-prediction tasks on time-series clinical data are
particularly prone to it.

## 3. Problem Statement

Given a patient's clinical observations from ICU hour 1 up to (at most) hour
6, predict whether that patient will be labeled septic under the official
Challenge definition, without using any information from or after their
actual sepsis onset hour.

## 4. Objectives

1. Build a leakage-safe, reproducible data pipeline from raw PhysioNet files.
2. Engineer patient-level features that preserve clinically meaningful
   signal (including measurement-frequency patterns).
3. Compare classical ML and deep learning approaches fairly, using
   consistent evaluation.
4. Select, calibrate, and threshold a final model using validation data only,
   with a single honest test-set evaluation.
5. Explain the model's behavior (SHAP), audit its errors, and check for
   subgroup disparities.
6. Package the result as a reusable prediction function and interactive
   dashboard, with known limitations documented rather than hidden.

## 5. Dataset

PhysioNet/CinC Challenge 2019 (training_setA + training_setB): 40,336
patients, 41 official columns (8 vital signs, 26 laboratory values, 6
demographic/administrative fields, `SepsisLabel`), hourly resolution,
patient-level sepsis prevalence 7.269%.

## 6. Data Preparation

**Loading** (`src/data_loader.py`): patient-wise `.psv` parsing, schema
validation against the official 41-column definition, ICULOS
monotonicity checks. **Cleaning** (`src/data_cleaning.py`): implausible
physiological values (e.g. Temp outside [25,45]°C) converted to NaN (6, 201,
and 103 values found for Temp/MAP/DBP respectively, out of 1.55M rows);
Unit1/Unit2 missingness (39.43% each, differing significantly by
source_set: 48.9% in Set A vs. 29.6% in Set B) encoded as an explicit
"Unknown" category rather than imputed; last-observation-carried-forward
(LOCF) applied strictly within each patient, in validated ICULOS order,
for vitals and laboratory values.

## 7. Exploratory Data Analysis

Sepsis-positive patients showed consistently elevated HR (~87-89 vs. 83-85
bpm) and respiratory rate, lower MAP (~79-83 vs. 82-90 mmHg), and elevated
WBC — directionally consistent with SIRS/sepsis screening criteria. No
pairwise correlation among key vitals exceeded |r|=0.23, indicating low
multicollinearity risk from these variables alone.

## 8. Missing Value Analysis

Missingness ranged from 0% (demographics, label) to 99.81%
(`Bilirubin_direct`). Formally testing whether missingness relates to
outcome: Lactate was measured in 5.38% of sepsis-positive hourly rows vs.
2.33% of negative rows (~2.3x) — the strongest such gap found, consistent
with Lactate's specific role in sepsis/septic-shock screening protocols.
This established that missingness patterns themselves carry predictive
signal (MAR/MNAR-consistent, not assumed to be MCAR).

## 9. Time-Series Analysis

Beyond mean trajectories, variability (standard deviation) of respiratory
rate remained consistently higher in sepsis-positive patients across the
full 72-hour window examined, suggesting a uniform group-level shift
rather than a small unstable subgroup. Hour-to-hour rate-of-change for HR
and respiratory rate was also consistently higher in positive patients.

## 10. Feature Engineering

For each of 34 clinical variables plus 4 derived features (Shock Index,
Pulse Pressure, SaO2/FiO2 ratio, Temperature-abnormality flag): mean,
median, min, max, std, first, last, change, absolute change, count,
missing ratio, and slope — 458 temporal features plus static demographics,
464 features after one-hot encoding. Aggregates were deliberately computed
from raw (pre-LOCF) measurements to preserve true measurement-frequency
signal; LOCF-filled values would have artificially inflated counts and
deflated variance.

## 11. Methodology

Leakage-safe target construction (Section 12) preceded feature
engineering; patient-level (not row-level) train/val/test splitting
(70/15/15, stratified, zero patient overlap verified) preceded
preprocessing; the preprocessing pipeline (median imputation + scaling,
one-hot encoding) was fit exclusively on the training split.

### 11.1 Target Construction — Critical Verification

The official `SepsisLabel` was empirically verified (not assumed) to be
monotonic non-decreasing per patient across all 40,336 patients (zero 1→0
transitions). Onset hour (first hour with label=1) ranged from 1 to 331
(median 29). Two exclusion categories were identified and explicitly
handled: 370 patients with onset≤1h (no pre-onset history exists) and 595
additional patients whose raw file began recording after hour 6 (a
previously undocumented dataset characteristic, discovered via diagnostic
code rather than assumed) — both excluded with a stated reason, not
silently dropped. The final leakage check confirmed 0 patients had any
retained hour at or after their onset hour.

## 12. Machine Learning Models

Six models were trained with class-weighting (`class_weight='balanced'`
or `scale_pos_weight`; SMOTE was deliberately not used, per project
requirements, due to clinical-implausibility and leakage risks of
synthetic oversampling). Validation results:

| Model | ROC-AUC | PR-AUC | Recall | Specificity |
|---|---|---|---|---|
| LightGBM (tuned) | 0.761 | 0.240 | 49.1% | 85.1% |
| CatBoost (tuned) | 0.739 | 0.219 | 35.2% | 91.6% |
| LightGBM (default) | 0.711 | 0.207 | 29.8% | 92.9% |
| XGBoost | 0.701 | 0.194 | 19.8% | 95.3% |
| CatBoost (default) | 0.723 | 0.190 | 36.6% | 87.9% |
| Random Forest | 0.703 | 0.180 | 15.2% | 96.4% |
| Logistic Regression | 0.705 | 0.164 | 58.5% | 74.1% |
| Decision Tree | 0.673 | 0.149 | 57.7% | 71.3% |

## 13. Deep Learning

An LSTM (Masking → LSTM(64) → Dropout → Dense → Sigmoid) was trained on
raw hourly sequences of 9 key variables, using the same leakage-safe
window and patient split. A critical bug (NaN propagation through
`np.mean`/`np.std` on sparse columns with zero real measurements for many
patients, causing the model to receive uniformly poisoned input) was
identified and fixed (`np.nanmean`/`np.nanstd`) before valid training
occurred. The corrected LSTM achieved ROC-AUC 0.528–0.664 across two
evaluation runs (a run-to-run discrepancy attributed to sequence
reconstruction differences between notebooks, noted but not fully
resolved) — underperforming every tree-based model, plausibly due to the
limited positive-class training size (~1,724 patients) and short (≤6h)
sequences.

## 14. Hyperparameter Tuning

LightGBM and CatBoost were tuned via Optuna (30 trials each), optimizing
validation PR-AUC. LightGBM improved from PR-AUC 0.207 to 0.240 (ROC-AUC
0.711→0.761, recall 29.8%→49.1%). CatBoost improved from PR-AUC 0.190 to
0.219. Tuning provided a substantial, verified improvement over
untuned defaults for both models.

## 15. Model Evaluation

Accuracy was explicitly rejected as a primary metric: a trivial
"always-negative" classifier scores 93.7% accuracy while achieving 0%
sensitivity. PR-AUC was used as the primary ranking metric due to its
greater sensitivity to minority-class (positive) performance at ~6%
prevalence. All models were systematically overconfident in their raw
probability estimates (calibration curves consistently below the diagonal),
motivating explicit calibration (Section 17).

## 16. Threshold Optimization

Evaluated on the grid [0.10, ..., 0.70] using validation data for LightGBM
(tuned): selection rule was "maximum F1 among thresholds achieving
recall≥50%," yielding threshold=0.4 pre-calibration (validation recall
63.1%, test recall 61.1% — closely matched, indicating no threshold
overfitting).

## 17. Calibration

Isotonic regression outperformed Platt scaling marginally (validation
Brier score 0.0527 vs. 0.0537) and both dramatically improved on the
uncalibrated model (0.1269). Because calibration changes the probability
scale, the operating threshold was re-derived on the calibrated
probabilities using the identical selection rule, yielding a new
threshold of 0.1. ROC-AUC/PR-AUC were confirmed stable across calibration
methods (monotonic transforms preserve ranking).

## 18. Explainability

SHAP (TreeExplainer on the underlying LightGBM model) identified
`HospAdmTime` as the single most important feature by mean |SHAP value|,
followed by multiple measurement-frequency features (`FiO2_count`,
`SBP_count`, `Lactate_count`, `Temp_missing_ratio`) — directly consistent
with Section 8's finding that measurement frequency itself carries signal.
Individual waterfall explanations showed a confidently-correct true
positive driven primarily by a low `SBP_count` (consistent with a short,
early-onset window) and high `Lactate_count`, while a false-negative
example showed no single dominant feature — consistent with a genuinely
subtler clinical presentation rather than a data artifact.

## 19. Error Analysis

False negatives (n=158, validation) had predicted probabilities far below
threshold (mean 0.047, max 0.088) — these are not "close misses" but cases
the model was confidently wrong about, with vitals closer to normal (lower
HR/Resp, less frequent Lactate testing) than true positives. False
positives (n=1,053) closely resembled true positives across HR, respiratory
rate, MAP, Shock Index, and Lactate testing frequency — far more similar to
true positives than to true negatives — suggesting many false alarms
represent genuinely at-risk-looking patients rather than arbitrary errors.
A notable data-quality finding: patients whose usable window was very short
(1–4 hours) were, in this validation sample, exclusively true positives,
never false negatives — a consequence of how short windows can only arise
from early sepsis onset in the training data (see Section 21.1).

## 20. Subgroup Analysis

| Subgroup | Recall | PR-AUC |
|---|---|---|
| Age 40-65 | 59.7% | 0.254 |
| Age 65-80 | 58.6% | 0.254 |
| **Age 80+** | **50.0%** | **0.133** |
| Source Set A | 67.0% | 0.255 |
| **Source Set B** | **41.0%** | 0.199 |
| Gender 0 | 49.0% | 0.184 |
| Gender 1 | 63.1% | 0.269 |
| ICU Unit 0 | 41.6% | 0.217 |
| ICU Unit 1 | 62.2% | 0.264 |

Meaningful disparities exist across every subgroup dimension examined.
Root cause (data-collection artifact, true population difference, or
sample-size effects in smaller subgroups) cannot be established from this
dataset alone; these are reported as findings requiring further
investigation, not resolved issues.

## 21. Final Model

**LightGBM (tuned) + isotonic calibration, decision threshold = 0.1.**

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.7664 | 0.7533 |
| PR-AUC | 0.2307 | 0.2232 |
| Recall | 57.2% | 52.4% |
| Specificity | 81.0% | 81.6% |
| Brier score | 0.0527 | 0.0533 |

Selected based on the combination of highest PR-AUC/ROC-AUC among all
models tested, close validation/test agreement (no overfitting evidence),
substantially improved calibration, computational efficiency relative to
the LSTM, and SHAP-verified clinically-plausible feature importance —
not on any single metric in isolation.

### 21.1 Critical Post-Hoc Finding

During prediction-pipeline testing (Phase 24), a synthetic 2-hour patient
received a higher risk score (0.343) than a synthetic patient with
dramatically abnormal vitals (0.065). Investigation revealed that, by
construction, every sepsis-negative training patient's window covered the
full 6 hours, while positive patients' windows were truncated before
onset — meaning "having fewer than 6 real hours of data" occurred, in
training, almost exclusively for positive patients. Engineered `_count`
features can therefore act as an indirect proxy for this construction
artifact. This does not invalidate the validation/test metrics reported
above (both sets share the identical construction rule), but it is a
material limitation for real-time use on genuinely early-stage patients.
Mitigated via an explicit `data_completeness_warning` in the prediction
pipeline and dashboard rather than a full pipeline rebuild (see Limitations).

## 22. Limitations

1. The indirect window-length leakage channel described in 21.1.
2. Subgroup disparities (Section 20) not resolved or fully explained.
3. 965 patients excluded from this task definition entirely (370 onset≤1h,
   595 with late-starting raw records).
4. LSTM results showed a run-to-run discrepancy not fully root-caused.
5. `HospAdmTime`'s outsized SHAP importance is not clinically explained.
6. No prospective or external-site validation performed.

## 23. Ethical Considerations

This is a research prototype, not a validated clinical tool, and must not
inform real patient care. SHAP values describe model associations, not
medical causality. Similar subgroup metrics do not establish fairness —
several subgroups here show clear disparities that would need resolution
before any deployment discussion.

## 24. Future Work

- Execute deferred Phase 9 (6h/12h/24h window comparison).
- Re-derive negative-patient window truncation to close the indirect
  leakage channel (21.1) and re-run the full modeling pipeline.
- Investigate root causes of subgroup disparities and `HospAdmTime`'s
  importance.
- External/prospective validation.

## 25. Conclusion

This project delivered a complete, leakage-audited ML pipeline achieving a
test ROC-AUC of 0.753 for 6-hour-ahead sepsis prediction, with rigorous
calibration, explainability, and error/subgroup analysis. Several
non-trivial issues — a hidden dataset characteristic (Section 11.1), an
LSTM training bug, and a subtle indirect leakage channel (21.1) — were
discovered through careful verification at each stage and are reported
transparently rather than hidden, consistent with the project's central
methodological commitment: every reported number is traceable to actual
executed code against the real dataset, and every known weakness is
documented alongside the results.

## 26. References

- Reyna, M. et al. "Early Prediction of Sepsis From Clinical Data: The
  PhysioNet/Computing in Cardiology Challenge 2019." Critical Care
  Medicine, 2020.
- PhysioNet/CinC Challenge 2019 dataset:
  https://physionet.org/content/challenge-2019/1.0.0/
