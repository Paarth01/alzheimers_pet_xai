"""
utils/metrics.py
─────────────────────────────────────────────────────────────────────
Evaluation metrics for 4-class AD classification.

Includes:
  - Accuracy, Macro F1, Per-class AUC-ROC
  - Cohen's Kappa
  - Confusion matrix with class labels
  - Per-class sensitivity (recall) and specificity
  - Summary table printer
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    cohen_kappa_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


CLASS_NAMES = ["CN", "EMCI", "MCI", "AD"]


def compute_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: np.ndarray,
                         class_names: List[str] = None) -> Dict:
    """
    Compute all classification metrics for the paper results table.

    Args:
        y_true:      Ground truth labels (N,)
        y_pred:      Predicted labels (N,)
        y_prob:      Predicted probabilities (N, num_classes)
        class_names: List of class names

    Returns:
        Dict with all metrics
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)
    labels = list(range(num_classes))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro",
                             labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                labels=labels, zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None,
                             labels=labels, zero_division=0)
    for i, cls in enumerate(class_names):
        metrics[f"f1_{cls}"] = per_class_f1[i]

    # Per-class AUC-ROC (one-vs-rest)
    if y_prob is not None and y_prob.shape[1] == num_classes:
        try:
            macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                       average="macro")
            metrics["macro_auc"] = macro_auc

            for i, cls in enumerate(class_names):
                binary_true = (y_true == i).astype(int)
                auc = roc_auc_score(binary_true, y_prob[:, i])
                metrics[f"auc_{cls}"] = auc
        except ValueError as e:
            print(f"[Metrics] AUC skipped: {e}")
            metrics["macro_auc"] = float("nan")

    # Per-class sensitivity and specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics[f"sensitivity_{cls}"] = sensitivity
        metrics[f"specificity_{cls}"] = specificity

    metrics["confusion_matrix"] = cm
    return metrics


def print_metrics_table(metrics: Dict, class_names: List[str] = None):
    """Pretty-print all metrics as a table."""
    if class_names is None:
        class_names = CLASS_NAMES

    print("\n" + "═"*60)
    print("EVALUATION RESULTS")
    print("═"*60)
    print(f"  Accuracy:       {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa:  {metrics['cohen_kappa']:.4f}")
    print(f"  Macro AUC-ROC:  {metrics.get('macro_auc', float('nan')):.4f}")
    print()
    print(f"  {'Class':<8} {'F1':>8} {'AUC':>8} {'Sensitivity':>13} {'Specificity':>13}")
    print("  " + "-"*52)
    for cls in class_names:
        f1  = metrics.get(f"f1_{cls}", float("nan"))
        auc = metrics.get(f"auc_{cls}", float("nan"))
        sen = metrics.get(f"sensitivity_{cls}", float("nan"))
        spe = metrics.get(f"specificity_{cls}", float("nan"))
        print(f"  {cls:<8} {f1:>8.4f} {auc:>8.4f} {sen:>13.4f} {spe:>13.4f}")
    print("═"*60 + "\n")


def plot_confusion_matrix(cm: np.ndarray,
                           class_names: List[str],
                           save_path: str,
                           title: str = "Confusion Matrix"):
    """
    Plot and save a normalized confusion matrix heatmap.

    Args:
        cm:          Confusion matrix from sklearn
        class_names: List of class names
        save_path:   File path to save the figure
        title:       Figure title
    """
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, ttl in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        [f"{title} (Counts)", f"{title} (Normalized)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(ttl, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", labelpad=10)
        ax.set_ylabel("True Label", labelpad=10)

    plt.suptitle("ResNet50 + CBAM — 4-Class AD Staging", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] Confusion matrix saved to {save_path}")


def plot_roc_curves(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     class_names: List[str],
                     save_path: str):
    """
    Plot per-class ROC curves (one-vs-rest) on a single figure.

    Args:
        y_true:      Ground truth labels (N,)
        y_prob:      Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path:   File path to save the figure
    """
    from sklearn.metrics import roc_curve, auc

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        binary_true = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — AD Stage Classification (One-vs-Rest)",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Metrics] ROC curves saved to {save_path}")
