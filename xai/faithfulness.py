"""
xai/faithfulness.py
─────────────────────────────────────────────────────────────────────
Quantitative XAI faithfulness evaluation via Insertion and Deletion AUC.

Reference: Petsiuk et al., BMVC 2018
           "RISE: Randomized Input Sampling for Explanation of
            Black-box Models"

Insertion AUC  (↑ better):
    Start with blurred/baseline image. Progressively reveal pixels
    in order of importance. Track class probability. A faithful
    explanation causes probability to rise quickly. AUC = faithfulness.

Deletion AUC   (↓ better):
    Start with original image. Progressively remove pixels
    in order of importance (replace with mean). Track class probability.
    A faithful explanation causes probability to drop quickly. AUC = faithfulness.

Usage:
    evaluator = FaithfulnessEvaluator(model, device)
    results = evaluator.evaluate(images, gradcam_maps, shap_maps, labels)
    evaluator.plot_curves(results, save_path="outputs/faithfulness.png")
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import scipy.ndimage as ndimage


class FaithfulnessEvaluator:
    """
    Computes Insertion and Deletion AUC for comparing XAI methods.

    Args:
        model:      ResNet50CBAM (eval mode)
        device:     Torch device
        n_steps:    Number of pixel insertion/deletion steps (default 50)
        batch_size: Batch size for forward passes during evaluation
    """

    def __init__(self, model: nn.Module,
                 device: torch.device = torch.device("cpu"),
                 n_steps: int = 50,
                 batch_size: int = 16):
        self.model = model.eval()
        self.device = device
        self.n_steps = n_steps
        self.batch_size = batch_size

    @staticmethod
    def _get_baseline(image: torch.Tensor, mode: str = "blur") -> torch.Tensor:
        """
        Generate baseline image for insertion metric.

        Args:
            image: (3, H, W) tensor
            mode:  "blur" (Gaussian blur) | "mean" (mean pixel) | "zero"

        Returns:
            Baseline tensor (3, H, W)
        """
        if mode == "blur":
            img_np = image.cpu().numpy()
            blurred = ndimage.gaussian_filter(img_np, sigma=(0, 10, 10))
            return torch.FloatTensor(blurred)
        elif mode == "mean":
            mean_val = image.mean().item()
            return torch.full_like(image, mean_val)
        else:  # zero
            return torch.zeros_like(image)

    @staticmethod
    def _rank_pixels(heatmap: np.ndarray) -> np.ndarray:
        """
        Rank pixel positions by importance (highest first).

        Args:
            heatmap: (H, W) importance map in [0, 1]

        Returns:
            Flat indices sorted by descending importance
        """
        return np.argsort(heatmap.flatten())[::-1]

    @torch.no_grad()
    def _get_prob(self, images: torch.Tensor,
                  target_class: int) -> np.ndarray:
        """
        Get class probability for a batch of images.

        Args:
            images:       (B, 3, H, W) tensor
            target_class: Class index

        Returns:
            Probabilities array (B,)
        """
        probs_list = []
        for i in range(0, images.shape[0], self.batch_size):
            batch = images[i:i+self.batch_size].to(self.device)
            logits = self.model(batch)
            probs  = torch.softmax(logits, dim=1)[:, target_class]
            probs_list.append(probs.cpu().numpy())
        return np.concatenate(probs_list)

    def deletion_curve(self, image: torch.Tensor,
                        heatmap: np.ndarray,
                        target_class: int) -> np.ndarray:
        """
        Compute deletion curve for a single image.
        Progressively removes most important pixels.

        Args:
            image:        (3, H, W) original image
            heatmap:      (H, W) importance map
            target_class: Class to track

        Returns:
            probs_curve: (n_steps+1,) class probabilities at each deletion step
        """
        H, W = image.shape[1], image.shape[2]
        n_pixels = H * W
        step_size = n_pixels // self.n_steps

        # Resize heatmap to image size if needed
        if heatmap.shape != (H, W):
            from PIL import Image as PILImage
            hm_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
            hm_pil = hm_pil.resize((W, H), PILImage.BILINEAR)
            heatmap = np.array(hm_pil).astype(np.float32) / 255.0

        pixel_order = self._rank_pixels(heatmap)  # Most → least important
        mean_val    = image.mean().item()

        imgs_sequence = []
        current = image.clone().cpu()

        for step in range(self.n_steps + 1):
            imgs_sequence.append(current.clone())
            # Remove next batch of pixels
            n_remove = min(step_size, n_pixels - step * step_size)
            start = step * step_size
            end   = start + n_remove
            if start >= n_pixels:
                break
            flat_idx = pixel_order[start:end]
            row_idx, col_idx = np.unravel_index(flat_idx, (H, W))
            current[:, row_idx, col_idx] = mean_val

        imgs_tensor = torch.stack(imgs_sequence)  # (n_steps+1, 3, H, W)
        probs = self._get_prob(imgs_tensor, target_class)
        return probs

    def insertion_curve(self, image: torch.Tensor,
                         heatmap: np.ndarray,
                         target_class: int,
                         baseline_mode: str = "blur") -> np.ndarray:
        """
        Compute insertion curve for a single image.
        Progressively reveals most important pixels from baseline.

        Returns:
            probs_curve: (n_steps+1,) class probabilities at each insertion step
        """
        H, W = image.shape[1], image.shape[2]
        n_pixels = H * W
        step_size = n_pixels // self.n_steps

        if heatmap.shape != (H, W):
            from PIL import Image as PILImage
            hm_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
            hm_pil = hm_pil.resize((W, H), PILImage.BILINEAR)
            heatmap = np.array(hm_pil).astype(np.float32) / 255.0

        pixel_order = self._rank_pixels(heatmap)
        baseline    = self._get_baseline(image, mode=baseline_mode).cpu()

        imgs_sequence = []
        current = baseline.clone()

        for step in range(self.n_steps + 1):
            imgs_sequence.append(current.clone())
            n_reveal = min(step_size, n_pixels - step * step_size)
            start = step * step_size
            end   = start + n_reveal
            if start >= n_pixels:
                break
            flat_idx = pixel_order[start:end]
            row_idx, col_idx = np.unravel_index(flat_idx, (H, W))
            current[:, row_idx, col_idx] = image[:, row_idx, col_idx]

        imgs_tensor = torch.stack(imgs_sequence)
        probs = self._get_prob(imgs_tensor, target_class)
        return probs

    @staticmethod
    def auc(curve: np.ndarray) -> float:
        """Compute area under curve using trapezoidal rule, normalized to [0,1]."""
        x = np.linspace(0, 1, len(curve))
        return float(np.trapz(curve, x))

    def evaluate(self, images: torch.Tensor,
                  gradcam_maps: np.ndarray,
                  shap_maps: np.ndarray,
                  labels: List[int],
                  n_samples: int = 200) -> Dict[str, Dict[str, float]]:
        """
        Run full faithfulness evaluation for both Grad-CAM and SHAP.

        Args:
            images:       (N, 3, H, W) test images
            gradcam_maps: (N, H', W') Grad-CAM heatmaps
            shap_maps:    (N, H', W') SHAP mean-abs maps
            labels:       True class indices (N,)
            n_samples:    Max samples to evaluate (subset for speed)

        Returns:
            results: {
                "gradcam": {"insertion_auc": float, "deletion_auc": float,
                            "insertion_curves": [...], "deletion_curves": [...]},
                "shap":    {"insertion_auc": float, "deletion_auc": float, ...}
            }
        """
        n = min(n_samples, len(images))
        indices = np.random.choice(len(images), n, replace=False)

        results = {
            "gradcam": {"insertion_curves": [], "deletion_curves": [],
                        "insertion_auc": 0., "deletion_auc": 0.},
            "shap":    {"insertion_curves": [], "deletion_curves": [],
                        "insertion_auc": 0., "deletion_auc": 0.},
        }

        for xai_name, maps in [("gradcam", gradcam_maps), ("shap", shap_maps)]:
            ins_aucs, del_aucs = [], []

            for idx in tqdm(indices, desc=f"Faithfulness [{xai_name}]"):
                image = images[idx]
                hmap  = maps[idx]
                cls   = labels[idx]

                # Insertion
                ins_curve = self.insertion_curve(image, hmap, cls)
                ins_aucs.append(self.auc(ins_curve))
                results[xai_name]["insertion_curves"].append(ins_curve)

                # Deletion
                del_curve = self.deletion_curve(image, hmap, cls)
                del_aucs.append(self.auc(del_curve))
                results[xai_name]["deletion_curves"].append(del_curve)

            results[xai_name]["insertion_auc"] = float(np.mean(ins_aucs))
            results[xai_name]["deletion_auc"]  = float(np.mean(del_aucs))

            print(f"[{xai_name.upper()}] "
                  f"Insertion AUC: {results[xai_name]['insertion_auc']:.4f} | "
                  f"Deletion AUC:  {results[xai_name]['deletion_auc']:.4f}")

        # Summary
        print("\n" + "─"*50)
        print("FAITHFULNESS SUMMARY")
        print("─"*50)
        for method in ["gradcam", "shap"]:
            ins = results[method]["insertion_auc"]
            dlt = results[method]["deletion_auc"]
            print(f"  {method.upper():10s} | Insertion AUC: {ins:.4f} | "
                  f"Deletion AUC: {dlt:.4f}")
        print("─"*50)

        winner_ins = "Grad-CAM" if (results["gradcam"]["insertion_auc"] >
                                     results["shap"]["insertion_auc"]) else "SHAP"
        winner_del = "Grad-CAM" if (results["gradcam"]["deletion_auc"] <
                                     results["shap"]["deletion_auc"]) else "SHAP"
        print(f"  Better Insertion AUC: {winner_ins}")
        print(f"  Better Deletion AUC:  {winner_del}")

        return results

    def plot_curves(self, results: Dict, save_path: str,
                    class_names: List[str] = None):
        """
        Plot mean Insertion and Deletion curves for both XAI methods.
        This is the key quantitative XAI figure for the paper.

        Args:
            results:   Output of evaluate()
            save_path: Output file path
        """
        x = np.linspace(0, 100, self.n_steps + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = {"gradcam": "#e74c3c", "shap": "#3498db"}
        labels = {"gradcam": "Grad-CAM", "shap": "SHAP"}

        for method in ["gradcam", "shap"]:
            curves = np.array(results[method]["insertion_curves"])
            mean_c = curves.mean(axis=0)
            std_c  = curves.std(axis=0)
            ins_auc = results[method]["insertion_auc"]
            c = colors[method]

            ax1.plot(x, mean_c, color=c,
                     label=f"{labels[method]} (AUC={ins_auc:.3f})", linewidth=2)
            ax1.fill_between(x, mean_c - std_c, mean_c + std_c,
                              alpha=0.15, color=c)

        ax1.set_xlabel("% Pixels Inserted", fontsize=12)
        ax1.set_ylabel("Mean Class Probability", fontsize=12)
        ax1.set_title("Insertion AUC (↑ Better)", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1)

        for method in ["gradcam", "shap"]:
            curves = np.array(results[method]["deletion_curves"])
            mean_c = curves.mean(axis=0)
            std_c  = curves.std(axis=0)
            del_auc = results[method]["deletion_auc"]
            c = colors[method]

            ax2.plot(x, mean_c, color=c,
                     label=f"{labels[method]} (AUC={del_auc:.3f})", linewidth=2)
            ax2.fill_between(x, mean_c - std_c, mean_c + std_c,
                              alpha=0.15, color=c)

        ax2.set_xlabel("% Pixels Deleted", fontsize=12)
        ax2.set_ylabel("Mean Class Probability", fontsize=12)
        ax2.set_title("Deletion AUC (↓ Better)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1)

        plt.suptitle("Quantitative XAI Faithfulness Evaluation\n"
                     "ResNet50+CBAM — FDG-PET Alzheimer's Classification (ADNI)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[Faithfulness] Curves saved: {save_path}")
