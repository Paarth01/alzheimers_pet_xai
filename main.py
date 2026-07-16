"""
main.py
─────────────────────────────────────────────────────────────────────
Entry point for the Alzheimer's Disease FDG-PET Classification project.

Modes:
  --mode train     → Run two-phase training
  --mode evaluate  → Load checkpoint and evaluate on test/OASIS sets
  --mode xai       → Generate Grad-CAM, SHAP, and faithfulness outputs
  --mode all       → Train → Evaluate → XAI (full pipeline)

Usage:
  python main.py --mode train
  python main.py --mode evaluate --checkpoint checkpoints/best_model.pth
  python main.py --mode xai --checkpoint checkpoints/best_model.pth
  python main.py --mode all
─────────────────────────────────────────────────────────────────────
"""

import argparse
import random
import torch
import numpy as np
from pathlib import Path

from config import data_cfg, model_cfg, train_cfg, xai_cfg
from data.dataset import build_dataloaders, compute_class_weights
from models.resnet_cbam import build_model
from train import train
from evaluate import evaluate_model
from xai.gradcam import GradCAMGenerator
from xai.shap_explainer import SHAPExplainer
from xai.faithfulness import FaithfulnessEvaluator


def set_seed(seed: int = 42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Reproducibility seed set: {seed}")


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (no GPU detected)")
    return device


def run_training(device: torch.device):
    """Phase 1 + Phase 2 training."""
    print("\n" + "█"*60)
    print("  MODE: TRAINING")
    print("█"*60)

    # Build dataloaders
    loaders = build_dataloaders(
        adni_csv=data_cfg.adni_csv,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        batch_size=train_cfg.batch_size,
        n_slices=data_cfg.n_slices_per_volume,
        target_size=data_cfg.image_size,
        num_workers=train_cfg.num_workers,
        random_seed=data_cfg.random_seed,
        oasis_csv=data_cfg.oasis_csv,
    )

    # Class weights for imbalance handling
    class_weights = compute_class_weights(data_cfg.adni_csv)
    print(f"[Weights] Class weights: {class_weights.tolist()}")

    # Build model
    model = build_model(
        num_classes=model_cfg.num_classes,
        pretrained=model_cfg.pretrained,
        use_cbam=model_cfg.use_cbam,
        dropout_rate=model_cfg.dropout_rate,
        device=str(device),
    )

    # Train
    best_ckpt = train(
        model=model,
        loaders=loaders,
        class_weights=class_weights,
        device=device,
        checkpoint_dir=data_cfg.checkpoint_dir,
        epochs_phase1=train_cfg.epochs_phase1,
        epochs_phase2=train_cfg.epochs_phase2,
        lr_head=train_cfg.lr_head,
        lr_backbone=train_cfg.lr_backbone,
        lr_finetune=train_cfg.lr_finetune,
        weight_decay=train_cfg.weight_decay,
        label_smoothing=train_cfg.label_smoothing,
        patience=train_cfg.patience,
        use_wandb=train_cfg.use_wandb,
        wandb_project=train_cfg.wandb_project,
        wandb_run_name=train_cfg.wandb_run_name,
        class_names=data_cfg.class_names,
    )

    return best_ckpt


def run_evaluation(checkpoint_path: str, device: torch.device):
    """Evaluate model on test + OASIS-3 sets."""
    print("\n" + "█"*60)
    print("  MODE: EVALUATION")
    print("█"*60)

    loaders = build_dataloaders(
        adni_csv=data_cfg.adni_csv,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        batch_size=train_cfg.batch_size,
        n_slices=data_cfg.n_slices_per_volume,
        target_size=data_cfg.image_size,
        num_workers=train_cfg.num_workers,
        oasis_csv=data_cfg.oasis_csv,
    )

    model = build_model(
        num_classes=model_cfg.num_classes,
        pretrained=False,   # Weights loaded from checkpoint
        use_cbam=model_cfg.use_cbam,
        device=str(device),
    )

    results = evaluate_model(
        model=model,
        loaders=loaders,
        device=device,
        checkpoint_path=checkpoint_path,
        output_dir=data_cfg.output_dir,
        class_names=data_cfg.class_names,
    )

    return results


def run_xai(checkpoint_path: str, device: torch.device):
    """Generate Grad-CAM, SHAP, and faithfulness evaluation outputs."""
    print("\n" + "█"*60)
    print("  MODE: XAI ANALYSIS")
    print("█"*60)

    loaders = build_dataloaders(
        adni_csv=data_cfg.adni_csv,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        batch_size=train_cfg.batch_size,
        n_slices=data_cfg.n_slices_per_volume,
        target_size=data_cfg.image_size,
        num_workers=train_cfg.num_workers,
    )

    model = build_model(
        num_classes=model_cfg.num_classes,
        pretrained=False,
        use_cbam=model_cfg.use_cbam,
        device=str(device),
    )

    # Load checkpoint
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[XAI] Model loaded from {checkpoint_path}")

    # ─────────────────────────────────────────────────
    # 1. GRAD-CAM
    # ─────────────────────────────────────────────────
    print("\n[XAI] Step 1/3 — Generating Grad-CAM heatmaps...")
    cam = GradCAMGenerator(
        model=model,
        target_layer=xai_cfg.gradcam_target_layer,
        device=device
    )

    # Collect representative samples per class
    samples_per_class = {}
    n_per_class = 5
    class_counts = {i: 0 for i in range(model_cfg.num_classes)}

    test_loader = loaders["test"]
    test_images_all, test_labels_all, test_cams_all = [], [], []

    for slices, label, _ in test_loader:
        cls = label.item()
        mid_slice = slices[0, slices.shape[1]//2].unsqueeze(0)  # (1, 3, H, W)

        # Grad-CAM for this slice
        hmap = cam.generate(mid_slice.to(device), target_class=cls)

        test_images_all.append(mid_slice.squeeze(0))
        test_labels_all.append(cls)
        test_cams_all.append(hmap[0])

        if class_counts[cls] < n_per_class:
            if cls not in samples_per_class:
                samples_per_class[cls] = []
            samples_per_class[cls].append(mid_slice.squeeze(0))
            class_counts[cls] += 1

    # Convert to tensors/arrays
    for cls in samples_per_class:
        samples_per_class[cls] = torch.stack(samples_per_class[cls])

    test_images_tensor = torch.stack(test_images_all)
    test_cams_array    = np.stack(test_cams_all)

    # Stage comparison figure (key paper figure)
    cam.stage_comparison_figure(
        samples_per_class=samples_per_class,
        class_names=data_cfg.class_names,
        save_path=f"{xai_cfg.heatmap_dir}/stage_comparison_gradcam.png"
    )
    cam.remove_hooks()

    # ─────────────────────────────────────────────────
    # 2. SHAP
    # ─────────────────────────────────────────────────
    print("\n[XAI] Step 2/3 — Generating SHAP attributions...")

    # Collect background samples from training set
    bg_samples = []
    for slices, _, _ in loaders["train"]:
        if isinstance(slices, torch.Tensor) and slices.ndim == 4:
            bg_samples.append(slices[:1])  # Take 1 sample
        if len(bg_samples) >= xai_cfg.shap_n_background:
            break
    background = torch.cat(bg_samples[:xai_cfg.shap_n_background], dim=0)

    shap_exp = SHAPExplainer(model=model, background=background, device=device)

    # Mean SHAP maps per class
    mean_shap_maps = shap_exp.mean_shap_per_class(
        loader=loaders["test"],
        n_per_class=xai_cfg.shap_n_test,
        class_names=data_cfg.class_names,
    )

    # Rebuild mean Grad-CAM maps per class for comparison
    mean_gcam_maps = {}
    for cls_idx, cls_name in enumerate(data_cfg.class_names):
        cls_mask = np.array(test_labels_all) == cls_idx
        if cls_mask.sum() > 0:
            mean_map = test_cams_array[cls_mask].mean(axis=0)
            if mean_map.max() > 1e-8:
                mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min())
            mean_gcam_maps[cls_name] = mean_map

    # SHAP vs Grad-CAM comparison figure
    SHAPExplainer.plot_shap_comparison(
        mean_shap_maps=mean_shap_maps,
        mean_gradcam_maps=mean_gcam_maps,
        class_names=data_cfg.class_names,
        save_path=f"{xai_cfg.shap_dir}/shap_vs_gradcam_comparison.png"
    )

    # Compute SHAP maps for all test images (for faithfulness eval)
    print("[XAI] Computing per-image SHAP maps for faithfulness evaluation...")
    test_shaps_all = []
    for i, (img, label) in enumerate(zip(test_images_all, test_labels_all)):
        if i >= xai_cfg.faithfulness_n_samples:
            break
        sv = shap_exp.explain(img.unsqueeze(0).to(device), target_class=label)
        shap_spatial = np.abs(sv).mean(axis=1)[0]  # (H, W)
        test_shaps_all.append(shap_spatial)
    test_shaps_array = np.stack(test_shaps_all)

    # ─────────────────────────────────────────────────
    # 3. FAITHFULNESS (Insertion / Deletion AUC)
    # ─────────────────────────────────────────────────
    print("\n[XAI] Step 3/3 — Computing faithfulness metrics...")
    faith_eval = FaithfulnessEvaluator(
        model=model,
        device=device,
        n_steps=xai_cfg.faithfulness_n_steps,
        batch_size=xai_cfg.faithfulness_batch_size,
    )

    n_faith = min(xai_cfg.faithfulness_n_samples, len(test_images_all))

    faith_results = faith_eval.evaluate(
        images=test_images_tensor[:n_faith],
        gradcam_maps=test_cams_array[:n_faith],
        shap_maps=test_shaps_array[:n_faith],
        labels=test_labels_all[:n_faith],
        n_samples=n_faith,
    )

    faith_eval.plot_curves(
        results=faith_results,
        save_path=f"{xai_cfg.faithfulness_dir}/insertion_deletion_curves.png",
        class_names=data_cfg.class_names,
    )

    print("\n[XAI] All outputs generated:")
    print(f"  Grad-CAM: {xai_cfg.heatmap_dir}/")
    print(f"  SHAP:     {xai_cfg.shap_dir}/")
    print(f"  Faithfulness: {xai_cfg.faithfulness_dir}/")

    return faith_results


def main():
    parser = argparse.ArgumentParser(
        description="Alzheimer's FDG-PET Classification with Attention + XAI"
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "evaluate", "xai", "all"],
        help="Execution mode"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pth) for evaluate/xai modes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device()

    print("\n" + "═"*60)
    print("  Alzheimer's Disease Classification — FDG-PET")
    print("  CBAM-ResNet50 + Grad-CAM + SHAP | ADNI + OASIS-3")
    print("═"*60)
    print(f"  Mode:   {args.mode}")
    print(f"  Device: {device}")
    print(f"  CBAM:   {'enabled' if model_cfg.use_cbam else 'disabled (ablation)'}")
    print("═"*60)

    checkpoint_path = args.checkpoint

    if args.mode in ("train", "all"):
        checkpoint_path = run_training(device)

    if args.mode in ("evaluate", "all"):
        if checkpoint_path is None:
            checkpoint_path = f"{data_cfg.checkpoint_dir}/best_model.pth"
        run_evaluation(checkpoint_path, device)

    if args.mode in ("xai", "all"):
        if checkpoint_path is None:
            checkpoint_path = f"{data_cfg.checkpoint_dir}/best_model.pth"
        run_xai(checkpoint_path, device)

    print("\n[Done] All tasks completed.")


if __name__ == "__main__":
    main()
