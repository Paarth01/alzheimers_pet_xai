"""
xai/shap_explainer.py
─────────────────────────────────────────────────────────────────────
SHAP DeepExplainer for ResNet50+CBAM on FDG-PET slices.

Reference: Lundberg & Lee, NeurIPS 2017
           "A Unified Approach to Interpreting Model Predictions"

DeepExplainer approximates Shapley values for deep networks using
a set of background reference samples. Each pixel receives a
SHAP value indicating its marginal contribution to the prediction.

Positive SHAP → pushes prediction toward predicted class
Negative SHAP → pushes prediction away from predicted class
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[SHAP] shap not installed. Run: pip install shap")


class SHAPExplainer:
    """
    SHAP DeepExplainer wrapper for FDG-PET classification.

    Args:
        model:       ResNet50CBAM model (eval mode)
        background:  Background reference tensor (N_bg, 3, H, W)
                     Randomly sampled from training set
        device:      Torch device
    """

    def __init__(self, model: nn.Module,
                 background: torch.Tensor,
                 device: torch.device = torch.device("cpu")):
        if not SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap>=0.44.0")

        self.model = model.eval()
        self.device = device
        self.background = background.to(device)

        # Wrap model for SHAP compatibility (output softmax probs)
        self._model_wrapper = self._build_wrapper()

        print(f"[SHAP] Initializing DeepExplainer with "
              f"{background.shape[0]} background samples...")
        self.explainer = shap.DeepExplainer(
            self._model_wrapper,
            self.background
        )
        print("[SHAP] DeepExplainer ready.")

    def _build_wrapper(self) -> nn.Module:
        """Wrap model to output probabilities (SHAP prefers probability outputs)."""
        class SoftmaxWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return torch.softmax(self.model(x), dim=1)

        return SoftmaxWrapper(self.model).to(self.device)

    def explain(self, images: torch.Tensor,
                target_class: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for a batch of images.

        Args:
            images:       (B, 3, H, W) tensor
            target_class: Class index to explain. None = explain all classes.

        Returns:
            shap_values: numpy array
                - If target_class is int: (B, 3, H, W) — values for that class
                - If None: list of (B, 3, H, W) arrays, one per class
        """
        images = images.to(self.device)

        with torch.no_grad():
            shap_values = self.explainer.shap_values(images)

        # shap_values is list of (B, 3, H, W) arrays, one per class
        if target_class is not None:
            return shap_values[target_class]  # (B, 3, H, W)
        return shap_values  # list of arrays

    def mean_shap_per_class(self, loader,
                             n_per_class: int = 100,
                             class_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Compute mean absolute SHAP values per AD stage.
        Averages across RGB channels to produce a spatial importance map.

        Args:
            loader:       DataLoader (volume-level)
            n_per_class:  Max samples per class
            class_names:  ["CN", "EMCI", "MCI", "AD"]

        Returns:
            dict: class_name → mean SHAP map (H, W), values in [0, 1]
        """
        if class_names is None:
            class_names = ["CN", "EMCI", "MCI", "AD"]

        # Collect samples per class
        class_samples: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(class_names))}

        for slices, label, _ in loader:
            cls = label.item()
            if len(class_samples[cls]) < n_per_class:
                # Take middle slice as representative
                mid = slices.shape[1] // 2
                class_samples[cls].append(slices[0, mid])  # (3, H, W)
            if all(len(v) >= n_per_class for v in class_samples.values()):
                break

        mean_maps = {}
        for cls_idx, cls_name in enumerate(class_names):
            samples = class_samples[cls_idx]
            if not samples:
                continue

            batch = torch.stack(samples, dim=0)  # (N, 3, H, W)
            print(f"[SHAP] Computing for {cls_name} ({len(samples)} samples)...")

            shap_vals = self.explain(batch, target_class=cls_idx)  # (N, 3, H, W)

            # Absolute values, mean over RGB channels → spatial map
            mean_map = np.abs(shap_vals).mean(axis=(0, 1))  # (H, W)

            # Normalize to [0, 1]
            if mean_map.max() > 1e-8:
                mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min())

            mean_maps[cls_name] = mean_map

        return mean_maps

    @staticmethod
    def plot_shap_comparison(mean_shap_maps: Dict[str, np.ndarray],
                              mean_gradcam_maps: Optional[Dict[str, np.ndarray]],
                              class_names: List[str],
                              save_path: str):
        """
        Side-by-side comparison figure: SHAP vs Grad-CAM per AD stage.
        This is the key XAI comparison figure for the paper.

        Args:
            mean_shap_maps:    dict cls_name → (H, W) SHAP map
            mean_gradcam_maps: dict cls_name → (H, W) Grad-CAM map (or None)
            class_names:       ["CN", "EMCI", "MCI", "AD"]
            save_path:         Output file path
        """
        n_classes = len(class_names)
        n_cols = 3 if mean_gradcam_maps else 2  # SHAP | Grad-CAM | Difference

        fig, axes = plt.subplots(n_classes, n_cols,
                                  figsize=(n_cols * 4, n_classes * 3.5))
        if n_classes == 1:
            axes = axes[np.newaxis, :]

        stage_colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]
        col_titles = ["SHAP (mean |values|)", "Grad-CAM", "Difference (SHAP - GradCAM)"]

        for cls_idx, cls_name in enumerate(class_names):
            shap_map = mean_shap_maps.get(cls_name, np.zeros((224, 224)))

            # SHAP map
            im = axes[cls_idx, 0].imshow(shap_map, cmap="RdBu_r", vmin=0, vmax=1)
            axes[cls_idx, 0].set_title(col_titles[0] if cls_idx == 0 else "",
                                        fontsize=10)
            axes[cls_idx, 0].set_ylabel(cls_name, fontsize=13, fontweight="bold",
                                         color=stage_colors[cls_idx])
            axes[cls_idx, 0].axis("off")
            plt.colorbar(im, ax=axes[cls_idx, 0], fraction=0.046, pad=0.04)

            if mean_gradcam_maps and n_cols >= 2:
                gcam_map = mean_gradcam_maps.get(cls_name, np.zeros_like(shap_map))

                im2 = axes[cls_idx, 1].imshow(gcam_map, cmap="jet", vmin=0, vmax=1)
                axes[cls_idx, 1].set_title(col_titles[1] if cls_idx == 0 else "",
                                            fontsize=10)
                axes[cls_idx, 1].axis("off")
                plt.colorbar(im2, ax=axes[cls_idx, 1], fraction=0.046, pad=0.04)

                if n_cols == 3:
                    diff = shap_map - gcam_map
                    im3 = axes[cls_idx, 2].imshow(diff, cmap="coolwarm",
                                                   vmin=-0.5, vmax=0.5)
                    axes[cls_idx, 2].set_title(col_titles[2] if cls_idx == 0 else "",
                                               fontsize=10)
                    axes[cls_idx, 2].axis("off")
                    plt.colorbar(im3, ax=axes[cls_idx, 2], fraction=0.046, pad=0.04)

        plt.suptitle("XAI Comparison: SHAP vs Grad-CAM per AD Stage\n"
                     "(FDG-PET, ResNet50+CBAM, ADNI dataset)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Comparison figure saved: {save_path}")

    @staticmethod
    def plot_individual_shap(image: torch.Tensor,
                              shap_values: np.ndarray,
                              pred_class: int,
                              true_class: int,
                              class_names: List[str],
                              save_path: str):
        """
        Plot SHAP attribution for a single PET slice showing
        positive (red) and negative (blue) contributions.

        Args:
            image:       (3, H, W) tensor
            shap_values: (num_classes, 3, H, W) SHAP values
            pred_class:  Predicted class index
            true_class:  True class index
            class_names: ["CN", "EMCI", "MCI", "AD"]
            save_path:   Output path
        """
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes + 1,
                                  figsize=((n_classes + 1) * 3.5, 3.5))

        # Original image
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(std * img + mean, 0, 1)

        axes[0].imshow(img)
        axes[0].set_title(f"PET Slice\nTrue: {class_names[true_class]}\n"
                           f"Pred: {class_names[pred_class]}", fontsize=9)
        axes[0].axis("off")

        # SHAP values per class
        vmax = max(np.abs(shap_values[c]).max() for c in range(n_classes))
        stage_colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

        for cls_idx in range(n_classes):
            vals = shap_values[cls_idx].mean(axis=0)  # (H, W) avg over RGB
            im = axes[cls_idx + 1].imshow(
                vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            border_color = stage_colors[cls_idx]
            for spine in axes[cls_idx + 1].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            axes[cls_idx + 1].set_title(
                f"SHAP: {class_names[cls_idx]}"
                + (" ★" if cls_idx == pred_class else ""),
                fontsize=9, color=border_color, fontweight="bold"
            )
            axes[cls_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[cls_idx + 1], fraction=0.046, pad=0.04)

        plt.suptitle("SHAP Attribution Map — FDG-PET AD Classification",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
