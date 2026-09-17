"""
evaluate.py
============
Phase 13-17 — Shared evaluation utilities used by every model in this
project, so metrics are computed identically (same definitions, same
threshold handling) regardless of which model produced the predictions.

Metrics computed: ROC-AUC, PR-AUC, precision, recall (sensitivity),
specificity, F1, Brier score, confusion matrix — never accuracy alone
(see Phase 12's justification).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config

logger = config.get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute the full metric suite at a given decision threshold.
    `y_prob` must be predicted probabilities (not hard labels) so ROC-AUC/
    PR-AUC/Brier score (threshold-independent) can be computed alongside
    the threshold-dependent metrics (precision/recall/F1/specificity).

    NOTE on `accuracy`: included for completeness/familiarity, but this
    project deliberately does NOT use it as a ranking or selection metric
    (see Phase 12) — with ~6% positive prevalence, a trivial "always
    predict negative" classifier scores ~93.7% accuracy while catching
    zero true positives. Always report `accuracy` alongside that trivial
    baseline (see `trivial_baseline_accuracy` below) and alongside
    recall/PR-AUC, never in isolation.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    trivial_baseline_accuracy = (tn + fp) / (tp + tn + fp + fn)  # = 1 - prevalence

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "trivial_baseline_accuracy": trivial_baseline_accuracy,
        "brier_score": brier_score_loss(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    print(f"--- {model_name} (threshold={metrics['threshold']}) ---")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity): {metrics['recall_sensitivity']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    if "accuracy" in metrics:
        print(f"  Accuracy:           {metrics['accuracy']:.4f}  "
              f"(trivial always-negative baseline: {metrics['trivial_baseline_accuracy']:.4f} "
              f"— accuracy alone is NOT a meaningful comparison at this prevalence, see Phase 12)")
    print(f"  Brier score:        {metrics['brier_score']:.4f}")
    print(f"  Confusion matrix:   TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model", ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model", ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"{model_name} (PR-AUC={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Chance (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model", n_bins: int = 10, ax=None):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    return ax


def plot_confusion_matrix(metrics: dict, model_name: str = "Model", ax=None):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    ax.set_title(f"{model_name} Confusion Matrix (threshold={metrics['threshold']})")
    return ax


def save_model_results(model_name: str, metrics: dict, results_path=None) -> None:
    """Append a model's metrics as one row to a running results CSV — the
    single source of truth for Phase 17's model comparison table."""
    results_path = results_path or (config.TABLES_DIR / "model_comparison.csv")
    row = {"model": model_name, **metrics}
    if results_path.exists():
        existing = pd.read_csv(results_path)
        existing = existing[existing["model"] != model_name]  # replace if re-run
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
    combined.to_csv(results_path, index=False)
    logger.info("Saved/updated results for '%s' in %s", model_name, results_path)
