"""
evaluate.py
─────────────────────────────────────────────────────────────────────
Model evaluation with subject-level aggregation.

Metrics computed:
  - Overall accuracy
  - Macro F1-score
  - Per-class AUC-ROC (one-vs-rest)
  - Confusion matrix
  - Cohen's Kappa
  - Classification report (precision, recall, F1 per class)

Subject-level aggregation:
  All slices for a subject → model → softmax probabilities
  → average across slices → argmax → subject prediction
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from tqdm import tqdm


CLASS_NAMES = ["CN", "EMCI", "MCI", "AD"]


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                    device: torch.device) -> Dict:
    """Load model weights from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Evaluate] Loaded checkpoint: {checkpoint_path}")
    print(f"  → Epoch: {ckpt.get('epoch', '?')} | "
          f"Val F1: {ckpt.get('val_f1', '?'):.4f} | "
          f"Val Acc: {ckpt.get('val_acc', '?'):.4f}")
    return ckpt


@torch.no_grad()
def predict_subjects(model: nn.Module, loader: DataLoader,
                     device: torch.device,
                     num_classes: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subject-level prediction via slice probability averaging.

    Args:
        model:       Model in eval mode
        loader:      Volume-level DataLoader (batch_size=1)
        device:      Torch device
        num_classes: Number of classes

    Returns:
        all_preds:   (N,) predicted class indices
        all_labels:  (N,) true class indices
        all_probs:   (N, num_classes) averaged softmax probabilities
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for slices, label, subject_id in tqdm(loader, desc="Predicting"):
        # slices: (1, n_slices, 3, H, W)
        slices = slices.squeeze(0).to(device)   # (n_slices, 3, H, W)
        label_val = label.item()

        # Forward pass on all slices
        logits = model(slices)                  # (n_slices, num_classes)
        probs  = torch.softmax(logits, dim=1)   # (n_slices, num_classes)

        # Subject-level: mean probability
        subject_prob = probs.mean(dim=0).cpu().numpy()   # (num_classes,)
        pred = subject_prob.argmax()

        all_preds.append(pred)
        all_labels.append(label_val)
        all_probs.append(subject_prob)

    return (np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


def compute_metrics(preds: np.ndarray, labels: np.ndarray,
                    probs: np.ndarray,
                    class_names: List[str] = None) -> Dict[str, float]:
    """
    Compute full evaluation metrics.

    Args:
        preds:       (N,) predicted class indices
        labels:      (N,) true class indices
        probs:       (N, num_classes) predicted probabilities
        class_names: List of class name strings

    Returns:
        metrics dict
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro",
                         labels=list(range(num_classes)), zero_division=0)
    kappa = cohen_kappa_score(labels, preds)

    # Per-class AUC-ROC (one-vs-rest)
    per_class_auc = {}
    for cls_idx, cls_name in enumerate(class_names):
        binary_labels = (labels == cls_idx).astype(int)
        cls_probs = probs[:, cls_idx]
        if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
            per_class_auc[f"auc_{cls_name}"] = roc_auc_score(binary_labels, cls_probs)
        else:
            per_class_auc[f"auc_{cls_name}"] = float("nan")

    mean_auc = np.nanmean(list(per_class_auc.values()))

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa":    kappa,
        "mean_auc": mean_auc,
        **per_class_auc
    }

    return metrics


def print_metrics(metrics: Dict[str, float],
                  preds: np.ndarray, labels: np.ndarray,
                  class_names: List[str] = None,
                  split_name: str = "Test"):
    """Pretty-print evaluation results."""
    if class_names is None:
        class_names = CLASS_NAMES

    print("\n" + "═"*60)
    print(f"  EVALUATION RESULTS — {split_name.upper()} SET")
    print("═"*60)
    print(f"  Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"  Cohen Kappa: {metrics['kappa']:.4f}")
    print(f"  Mean AUC:    {metrics['mean_auc']:.4f}")
    print()
    for cls in class_names:
        auc_val = metrics.get(f"auc_{cls}", float("nan"))
        print(f"  AUC-ROC [{cls}]:  {auc_val:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(labels, preds, target_names=class_names,
                                 zero_division=0))
    print("═"*60)


def plot_confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                           class_names: List[str] = None,
                           save_path: str = "outputs/confusion_matrix.png",
                           title: str = "Confusion Matrix"):
    """
    Plot and save a normalized confusion matrix.

    Args:
        preds:       (N,) predicted classes
        labels:      (N,) true classes
        class_names: List of class names
        save_path:   Output path
        title:       Figure title
    """
    if class_names is None:
        class_names = CLASS_NAMES

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, linewidths=0.5)
    ax1.set_xlabel("Predicted", fontsize=12)
    ax1.set_ylabel("True", fontsize=12)
    ax1.set_title("Counts", fontsize=13, fontweight="bold")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, linewidths=0.5, vmin=0, vmax=1)
    ax2.set_xlabel("Predicted", fontsize=12)
    ax2.set_ylabel("True", fontsize=12)
    ax2.set_title("Normalized (row)", fontsize=13, fontweight="bold")

    plt.suptitle(f"{title}\nResNet50+CBAM | FDG-PET | ADNI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved: {save_path}")


def evaluate_model(model: nn.Module,
                   loaders: Dict[str, DataLoader],
                   device: torch.device,
                   checkpoint_path: Optional[str] = None,
                   output_dir: str = "outputs",
                   class_names: List[str] = None) -> Dict[str, Dict]:
    """
    Full evaluation on test and OASIS-3 (external validation) sets.

    Args:
        model:           ResNet50CBAM model
        loaders:         Dict with "test" and optionally "oasis" loaders
        device:          Torch device
        checkpoint_path: Path to .pth checkpoint (optional, loads if given)
        output_dir:      Directory for output plots
        class_names:     List of class names

    Returns:
        results: dict with split_name → metrics dict
    """
    if class_names is None:
        class_names = CLASS_NAMES

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, device)

    results = {}

    for split_name, loader in loaders.items():
        if split_name == "train":
            continue  # Skip training set evaluation

        print(f"\n[Evaluate] Running on {split_name} split...")
        preds, labels, probs = predict_subjects(model, loader, device)

        metrics = compute_metrics(preds, labels, probs, class_names)
        print_metrics(metrics, preds, labels, class_names, split_name)

        # Confusion matrix
        cm_path = f"{output_dir}/confusion_matrix_{split_name}.png"
        plot_confusion_matrix(preds, labels, class_names, cm_path,
                               title=f"Confusion Matrix — {split_name}")

        results[split_name] = {
            "metrics": metrics,
            "preds":   preds,
            "labels":  labels,
            "probs":   probs,
        }

    return results
