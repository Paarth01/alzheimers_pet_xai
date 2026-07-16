"""
utils/visualize.py
─────────────────────────────────────────────────────────────────────
Visualization utilities:
  - Brain region activation matrix (Grad-CAM per stage)
  - Insertion/Deletion AUC curves
  - SHAP summary plot per class
  - Training loss/accuracy curves
  - Side-by-side Grad-CAM vs SHAP comparison
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ── Color palette (consistent across all figures) ────────────────
STAGE_COLORS = {
    "CN":   "#2196F3",   # Blue
    "EMCI": "#4CAF50",   # Green
    "MCI":  "#FF9800",   # Orange
    "AD":   "#F44336",   # Red
}
REGION_COLORS = ["#7C3AED", "#059669", "#DC2626", "#D97706"]


def plot_brain_region_matrix(activation_matrix: Dict[str, Dict[str, float]],
                              class_names: List[str],
                              save_path: str,
                              title: str = "Grad-CAM Brain Region Activation per AD Stage"):
    """
    Plot a heatmap of mean Grad-CAM activation per brain region per AD stage.
    This is the clinical validation figure for the paper (Figure 6).

    Expected pattern:
      - PCC high activation from EMCI onward
      - Hippocampus high from MCI onward
      - Temporal-Parietal high in AD stage

    Args:
        activation_matrix: {region_name: {class_name: mean_activation}}
        class_names:       ["CN", "EMCI", "MCI", "AD"]
        save_path:         Output file path
        title:             Figure title
    """
    regions = list(activation_matrix.keys())
    data = np.array([
        [activation_matrix[region][cls] for cls in class_names]
        for region in regions
    ])  # (n_regions, n_classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # ── Heatmap ───────────────────────────────────────────────────
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=regions,
        linewidths=0.5, ax=ax1,
        cbar_kws={"label": "Mean Grad-CAM Activation", "shrink": 0.8},
        vmin=0, vmax=data.max()
    )
    ax1.set_title(title, fontweight="bold", pad=12)
    ax1.set_xlabel("Disease Stage", fontsize=12)
    ax1.set_ylabel("Brain Region", fontsize=12)
    ax1.tick_params(axis="x", rotation=0)

    # ── Line plot (progression per region) ───────────────────────
    x = np.arange(len(class_names))
    for i, (region, color) in enumerate(zip(regions, REGION_COLORS)):
        vals = [activation_matrix[region][cls] for cls in class_names]
        ax2.plot(x, vals, marker="o", color=color, linewidth=2.5,
                  markersize=8, label=region)

    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, fontsize=11)
    ax2.set_ylabel("Mean Activation", fontsize=12)
    ax2.set_title("Progression Across Stages", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Add clinical annotation
    ax2.annotate("Expected:\nPCC earliest", xy=(1, data[0, 1]),
                  xytext=(1.5, data[0, 1] + 0.05),
                  fontsize=8, color=REGION_COLORS[0],
                  arrowprops=dict(arrowstyle="->", color=REGION_COLORS[0]))

    plt.suptitle("Clinical Brain Region Validation — Grad-CAM Activation",
                  fontsize=13, y=1.02)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Brain region matrix saved to {save_path}")


def plot_faithfulness_curves(insertion_curves: Dict[str, np.ndarray],
                               deletion_curves: Dict[str, np.ndarray],
                               insertion_aucs: Dict[str, float],
                               deletion_aucs: Dict[str, float],
                               save_path: str):
    """
    Plot Insertion and Deletion AUC curves for Grad-CAM vs SHAP.
    This is the quantitative XAI comparison figure (Figure 5).

    Args:
        insertion_curves: {"GradCAM": (n_steps,), "SHAP": (n_steps,)}
        deletion_curves:  {"GradCAM": (n_steps,), "SHAP": (n_steps,)}
        insertion_aucs:   {"GradCAM": float, "SHAP": float}
        deletion_aucs:    {"GradCAM": float, "SHAP": float}
        save_path:        Output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"GradCAM": "#2196F3", "SHAP": "#FF5722"}
    x = np.linspace(0, 1, next(iter(insertion_curves.values())).shape[0])

    # ── Insertion ─────────────────────────────────────────────────
    for method, curve in insertion_curves.items():
        auc = insertion_aucs[method]
        ax1.plot(x, curve, color=colors[method], linewidth=2.5,
                  label=f"{method} (AUC={auc:.3f})")
        ax1.fill_between(x, 0, curve, alpha=0.1, color=colors[method])

    ax1.set_xlabel("Fraction of Pixels Revealed", fontsize=12)
    ax1.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax1.set_title("Insertion AUC ↑\n(Higher = more faithful)",
                   fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    winner_ins = max(insertion_aucs, key=insertion_aucs.get)
    ax1.annotate(f"Winner: {winner_ins}", xy=(0.5, 0.05),
                  xycoords="axes fraction", ha="center",
                  fontsize=11, color=colors[winner_ins], fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="white", edgecolor=colors[winner_ins]))

    # ── Deletion ──────────────────────────────────────────────────
    for method, curve in deletion_curves.items():
        auc = deletion_aucs[method]
        ax2.plot(x, curve, color=colors[method], linewidth=2.5,
                  label=f"{method} (AUC={auc:.3f})")
        ax2.fill_between(x, 0, curve, alpha=0.1, color=colors[method])

    ax2.set_xlabel("Fraction of Pixels Removed", fontsize=12)
    ax2.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax2.set_title("Deletion AUC ↓\n(Lower = more faithful)",
                   fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    winner_del = min(deletion_aucs, key=deletion_aucs.get)
    ax2.annotate(f"Winner: {winner_del}", xy=(0.5, 0.05),
                  xycoords="axes fraction", ha="center",
                  fontsize=11, color=colors[winner_del], fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="white", edgecolor=colors[winner_del]))

    plt.suptitle("Quantitative XAI Faithfulness: Grad-CAM vs SHAP on FDG-PET",
                  fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Faithfulness curves saved to {save_path}")


def plot_gradcam_vs_shap(image_tensor,
                          gradcam_heatmap: np.ndarray,
                          shap_values: np.ndarray,
                          true_label: str,
                          pred_label: str,
                          save_path: str):
    """
    Side-by-side comparison: PET slice | Grad-CAM | SHAP | Overlay.
    Used for qualitative comparison in the paper.

    Args:
        image_tensor:    (3, H, W) normalized tensor
        gradcam_heatmap: (H, W) Grad-CAM in [0, 1]
        shap_values:     (H, W) SHAP positive contributions in [0, 1]
        true_label:      Ground truth class name
        pred_label:      Predicted class name
        save_path:       Output file path
    """
    import torch

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image_tensor.cpu() * std + mean).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    correct = true_label == pred_label
    color = "green" if correct else "red"

    # 1. Original PET
    axes[0].imshow(img[:, :, 0], cmap="hot")
    axes[0].set_title("FDG-PET Slice", fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM
    im1 = axes[1].imshow(gradcam_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM", fontweight="bold", color="#2196F3")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. SHAP
    shap_display = np.clip(shap_values, 0, None)
    if shap_display.max() > 0:
        shap_display = shap_display / shap_display.max()
    im2 = axes[2].imshow(shap_display, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("SHAP Values (+)", fontweight="bold", color="#FF5722")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Overlay (Grad-CAM on PET)
    import cv2
    heatmap_uint8 = (gradcam_heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_uint8 = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, colored, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Grad-CAM Overlay", fontweight="bold")
    axes[3].axis("off")

    status = "✓ Correct" if correct else "✗ Incorrect"
    fig.suptitle(
        f"True: {true_label}  |  Predicted: {pred_label}  |  {status}",
        fontsize=13, fontweight="bold", color=color, y=1.02
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """
    Plot training and validation loss/accuracy/F1 curves.

    Args:
        history: {"train_loss": [...], "val_loss": [...],
                   "train_acc": [...], "val_acc": [...],
                   "val_f1": [...]}
        save_path: Output file path
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3,
                  label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=3,
                  label="Val", linewidth=2)
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=3,
                  label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=3,
                  label="Val", linewidth=2)
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])

    # Macro F1
    if "val_f1" in history:
        axes[2].plot(epochs, history["val_f1"], "g-o", markersize=3,
                      label="Val Macro F1", linewidth=2)
        axes[2].set_title("Validation Macro F1", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro F1 Score")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        axes[2].set_ylim([0, 1])

        # Mark phase boundary
        if "phase_boundary" in history:
            pb = history["phase_boundary"]
            for ax in axes:
                ax.axvline(x=pb, color="purple", linestyle="--",
                            alpha=0.7, linewidth=1.5, label="P1→P2")

    plt.suptitle("Training Curves — ResNet50+CBAM on ADNI FDG-PET",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Training curves saved to {save_path}")
