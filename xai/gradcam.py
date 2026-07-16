"""
xai/gradcam.py
─────────────────────────────────────────────────────────────────────
Grad-CAM implementation for ResNet50+CBAM on FDG-PET slices.

Reference: Selvaraju et al., ICCV 2017
           "Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization"

Usage:
    cam = GradCAMGenerator(model, target_layer="layer4")
    heatmaps = cam.generate(images, target_class=None)  # None = predicted class
    cam.visualize(image, heatmap, save_path="out.png")
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import cv2


class GradCAMGenerator:
    """
    Grad-CAM heatmap generator for the ResNet50+CBAM model.

    Hooks into the target convolutional layer to capture:
      - Forward activations (feature maps)
      - Backward gradients (gradient of class score w.r.t. feature maps)

    Args:
        model:        ResNet50CBAM instance
        target_layer: Layer name to hook (default: "layer4")
        device:       Torch device
    """

    def __init__(self, model: nn.Module,
                 target_layer: str = "layer4",
                 device: torch.device = torch.device("cpu")):
        self.model = model
        self.device = device
        self.model.eval()

        # Storage for hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._hooks = []

        # Register hooks on target layer
        self._register_hooks(target_layer)

    def _register_hooks(self, layer_name: str):
        """Attach forward and backward hooks to the target layer."""
        # Get target layer module
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            available = [n for n, _ in self.model.named_modules() if n]
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available layers: {available[:20]}..."
            )

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all registered hooks (call when done)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(self, images: torch.Tensor,
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmaps for a batch of images.

        Args:
            images:       Input tensor (B, 3, H, W)
            target_class: Class index for gradient computation.
                          None = use predicted class per image.

        Returns:
            heatmaps: numpy array (B, H, W) with values in [0, 1]
        """
        self.model.eval()
        images = images.to(self.device).requires_grad_(False)

        # Forward pass
        logits = self.model(images)  # (B, num_classes)
        probs  = torch.softmax(logits, dim=1)

        batch_size = images.shape[0]
        heatmaps = []

        for i in range(batch_size):
            # Select target class
            cls = target_class if target_class is not None else logits[i].argmax().item()

            # Zero gradients
            self.model.zero_grad()

            # Compute gradient of class score w.r.t. feature map
            score = logits[i, cls]
            score.backward(retain_graph=(i < batch_size - 1))

            # ── Grad-CAM computation ──────────────────────────────
            # Gradients: (B, C, H, W) — take slice for image i
            grads = self._gradients[i]         # (C, H', W')
            acts  = self._activations[i]       # (C, H', W')

            # Importance weights: global average pool of gradients
            weights = grads.mean(dim=(1, 2))   # (C,)

            # Weighted combination of feature maps
            cam = (weights[:, None, None] * acts).sum(dim=0)  # (H', W')

            # ReLU: keep only positive contributions
            cam = torch.relu(cam)

            # Normalize to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)

            heatmaps.append(cam.cpu().numpy())

        return np.stack(heatmaps, axis=0)  # (B, H', W')

    def generate_all_classes(self, image: torch.Tensor,
                              class_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for ALL classes for a single image.
        Useful for visualizing how each class "sees" the same PET slice.

        Args:
            image:        Single image tensor (3, H, W)
            class_names:  List of class names ["CN", "EMCI", "MCI", "AD"]

        Returns:
            dict mapping class_name → heatmap (H', W')
        """
        image_batch = image.unsqueeze(0)  # (1, 3, H, W)
        results = {}
        for cls_idx, cls_name in enumerate(class_names):
            heatmap = self.generate(image_batch, target_class=cls_idx)
            results[cls_name] = heatmap[0]  # (H', W')
        return results

    @staticmethod
    def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                        alpha: float = 0.4,
                        colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original PET slice image.

        Args:
            image:    Original image (H, W, 3) in uint8 [0, 255]
            heatmap:  Grad-CAM heatmap (H', W') in [0, 1]
            alpha:    Transparency of heatmap overlay (0=no overlay, 1=full)
            colormap: OpenCV colormap (default JET)

        Returns:
            Overlay image (H, W, 3) in uint8
        """
        # Resize heatmap to original image size
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (1 - alpha) * image.astype(np.float32) + \
                   alpha * heatmap_colored.astype(np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert normalized image tensor (3, H, W) back to
        displayable numpy array (H, W, 3) in uint8.
        Reverses ImageNet normalization.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        img = std * img + mean                          # Denormalize
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def visualize_grid(self, images: torch.Tensor, heatmaps: np.ndarray,
                       labels: List[int], preds: List[int],
                       class_names: List[str], save_path: str,
                       n_cols: int = 4):
        """
        Plot a grid of original | heatmap | overlay for a batch of images.
        Annotates each with true label and predicted class.

        Args:
            images:      (B, 3, H, W) tensor
            heatmaps:    (B, H', W') numpy array
            labels:      True class indices
            preds:       Predicted class indices
            class_names: ["CN", "EMCI", "MCI", "AD"]
            save_path:   Output file path
            n_cols:      Columns per row (each showing orig | heatmap | overlay)
        """
        B = images.shape[0]
        n_rows = (B + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols * 3,
                                  figsize=(n_cols * 9, n_rows * 3.5))
        axes = np.array(axes).reshape(n_rows, n_cols * 3)

        # Color code: correct=green, wrong=red
        COLORS = {True: "#27ae60", False: "#e74c3c"}

        for i in range(B):
            row = i // n_cols
            col_base = (i % n_cols) * 3

            img_np = self.tensor_to_numpy_image(images[i])
            hm = cv2.resize(heatmaps[i], (img_np.shape[1], img_np.shape[0]))
            overlay = self.overlay_heatmap(img_np, hm)

            correct = labels[i] == preds[i]
            color = COLORS[correct]
            title = (f"True: {class_names[labels[i]]}\n"
                     f"Pred: {class_names[preds[i]]}")

            for j, (display, ttl) in enumerate([
                (img_np, "Original PET"),
                ((hm * 255).astype(np.uint8), "Grad-CAM"),
                (overlay, "Overlay")
            ]):
                ax = axes[row, col_base + j]
                if j == 1:
                    ax.imshow(display, cmap="jet", vmin=0, vmax=255)
                else:
                    ax.imshow(display)
                ax.axis("off")
                if j == 1:
                    ax.set_title(title, fontsize=9,
                                 color=color, fontweight="bold")
                else:
                    ax.set_title(ttl, fontsize=8, color="#555555")

        # Hide unused subplots
        for i in range(B, n_rows * n_cols):
            for j in range(3):
                row = i // n_cols
                col_base = (i % n_cols) * 3
                axes[row, col_base + j].axis("off")

        plt.suptitle("Grad-CAM Heatmaps — FDG-PET Alzheimer's Classification",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Grad-CAM] Grid saved: {save_path}")

    def stage_comparison_figure(self, samples_per_class: Dict[int, torch.Tensor],
                                 class_names: List[str], save_path: str):
        """
        Generate a 4-row figure (one per AD stage) showing how Grad-CAM
        activations change across CN → EMCI → MCI → AD.
        This is the key figure for the paper.

        Args:
            samples_per_class: Dict mapping class_idx → tensor (N, 3, H, W)
            class_names:       ["CN", "EMCI", "MCI", "AD"]
            save_path:         Output path
        """
        n_classes = len(class_names)
        n_samples = max(v.shape[0] for v in samples_per_class.values())

        fig, axes = plt.subplots(n_classes, n_samples + 1,
                                  figsize=((n_samples + 1) * 3.5, n_classes * 3.5))

        stage_colors = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

        for cls_idx, cls_name in enumerate(class_names):
            images = samples_per_class.get(cls_idx)
            if images is None:
                continue

            heatmaps = self.generate(images.to(self.device), target_class=cls_idx)

            # Mean heatmap for this class (first column)
            mean_hm = heatmaps.mean(axis=0)
            ax = axes[cls_idx, 0]
            ax.imshow(mean_hm, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Mean Grad-CAM\n{cls_name}", fontsize=10,
                         color=stage_colors[cls_idx], fontweight="bold")
            ax.axis("off")
            ax.set_ylabel(cls_name, fontsize=12, fontweight="bold",
                          color=stage_colors[cls_idx])

            # Individual sample overlays
            for s_idx in range(min(n_samples, images.shape[0])):
                img_np = self.tensor_to_numpy_image(images[s_idx])
                hm = cv2.resize(heatmaps[s_idx],
                                (img_np.shape[1], img_np.shape[0]))
                overlay = self.overlay_heatmap(img_np, hm, alpha=0.45)
                ax = axes[cls_idx, s_idx + 1]
                ax.imshow(overlay)
                ax.axis("off")
                if cls_idx == 0:
                    ax.set_title(f"Sample {s_idx+1}", fontsize=9, color="#555")

        plt.suptitle("Grad-CAM Stage Progression: CN → EMCI → MCI → AD\n"
                     "(FDG-PET, ResNet50+CBAM)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[Grad-CAM] Stage comparison saved: {save_path}")
