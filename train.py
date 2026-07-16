"""
train.py
─────────────────────────────────────────────────────────────────────
Two-phase training loop:
  Phase 1 (epochs 1-15): Freeze backbone, train classifier + CBAM only
  Phase 2 (epochs 16-30): Unfreeze all, fine-tune end-to-end at lower LR

Features:
  - Weighted CrossEntropyLoss for class imbalance
  - Label smoothing
  - CosineAnnealing scheduler
  - Early stopping on val macro F1
  - Weights & Biases logging
  - Automatic best model checkpoint saving
─────────────────────────────────────────────────────────────────────
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """
    Stop training when monitored metric stops improving.

    Args:
        patience:  Epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode:      "max" for metrics like F1/accuracy, "min" for loss
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta
                    if self.mode == "max"
                    else score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_interval: int = 10) -> Dict[str, float]:
    """
    Single epoch training pass.

    Returns:
        Dict with keys: loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc":  f"{correct / total:.4f}"
            })

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             class_names: list) -> Dict[str, float]:
    """
    Validation pass with subject-level aggregation.
    Each subject's slices are averaged to get subject-level probabilities.

    Returns:
        Dict with keys: loss, accuracy, macro_f1
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    for slices, label, subject_id in tqdm(loader, desc="[Val]", leave=False):
        # slices: (1, n_slices, 3, H, W) from volume dataset
        slices = slices.squeeze(0).to(device)   # (n_slices, 3, H, W)
        label_val = label.item()

        # Forward pass on all slices
        logits = model(slices)                  # (n_slices, num_classes)
        probs = torch.softmax(logits, dim=1)    # (n_slices, num_classes)

        # Subject-level: average probabilities across slices
        subject_prob = probs.mean(dim=0)        # (num_classes,)
        pred = subject_prob.argmax().item()

        # Loss on first slice (approximate)
        label_tensor = torch.tensor([label_val], device=device)
        loss = criterion(logits[:1], label_tensor)
        total_loss += loss.item()

        all_preds.append(pred)
        all_labels.append(label_val)

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        labels=list(range(len(class_names))), zero_division=0)

    return {"loss": avg_loss, "accuracy": float(accuracy), "macro_f1": float(macro_f1)}


def train(model: nn.Module,
          loaders: Dict[str, DataLoader],
          class_weights: torch.Tensor,
          device: torch.device,
          checkpoint_dir: str = "checkpoints",
          epochs_phase1: int = 15,
          epochs_phase2: int = 15,
          lr_head: float = 1e-3,
          lr_backbone: float = 1e-4,
          lr_finetune: float = 1e-5,
          weight_decay: float = 1e-4,
          label_smoothing: float = 0.1,
          patience: int = 7,
          use_wandb: bool = False,
          wandb_project: str = "alzheimers-pet-xai",
          wandb_run_name: str = "resnet50_cbam",
          class_names: list = None) -> str:
    """
    Full two-phase training.

    Returns:
        Path to best model checkpoint
    """
    if class_names is None:
        class_names = ["CN", "EMCI", "MCI", "AD"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")

    # ── W&B initialization ────────────────────────────────────────
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=wandb_project, name=wandb_run_name,
                   config={"epochs_p1": epochs_phase1,
                           "epochs_p2": epochs_phase2,
                           "lr_head": lr_head,
                           "lr_finetune": lr_finetune})

    # ── Loss ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing
    )

    # ── Early stopping tracker ────────────────────────────────────
    early_stopper = EarlyStopping(patience=patience, mode="max")
    best_val_f1 = 0.0

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Freeze backbone, train head + CBAM only
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("PHASE 1: Training classifier head + CBAM (backbone frozen)")
    print("═"*60)
    model.freeze_backbone()

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_head, weight_decay=weight_decay
    )
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=epochs_phase1, eta_min=1e-6)

    for epoch in range(1, epochs_phase1 + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer_p1, device, epoch
        )
        val_metrics = validate(model, loaders["val"], criterion, device, class_names)
        scheduler_p1.step()

        elapsed = time.time() - t0
        print(f"[P1 Ep {epoch:02d}/{epochs_phase1}] "
              f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | {elapsed:.1f}s")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "phase": 1, "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/acc": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
                "lr": scheduler_p1.get_last_lr()[0]
            })

        # Save best
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p1.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": val_metrics["accuracy"],
                "class_names": class_names,
            }, best_ckpt_path)
            print(f"  ✓ New best val F1: {best_val_f1:.4f} — checkpoint saved")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Unfreeze all, fine-tune end-to-end
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═"*60)
    print("PHASE 2: Full fine-tuning (all layers, lower LR)")
    print("═"*60)
    model.unfreeze_all()

    # Reload best phase-1 weights before phase 2
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded best Phase 1 checkpoint (val F1: {ckpt['val_f1']:.4f})")

    # Differential LRs: backbone lower than head
    param_groups = model.get_parameter_groups(
        lr_backbone=lr_finetune, lr_head=lr_finetune * 5
    )
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=epochs_phase2, eta_min=1e-7)

    early_stopper = EarlyStopping(patience=patience, mode="max")

    for epoch in range(1, epochs_phase2 + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer_p2, device,
            epochs_phase1 + epoch
        )
        val_metrics = validate(model, loaders["val"], criterion, device, class_names)
        scheduler_p2.step()

        elapsed = time.time() - t0
        print(f"[P2 Ep {epoch:02d}/{epochs_phase2}] "
              f"Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | {elapsed:.1f}s")

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "phase": 2,
                "epoch": epochs_phase1 + epoch,
                "train/loss": train_metrics["loss"],
                "train/acc": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/acc": val_metrics["accuracy"],
                "val/macro_f1": val_metrics["macro_f1"],
            })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epochs_phase1 + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "val_f1": best_val_f1,
                "val_acc": val_metrics["accuracy"],
                "class_names": class_names,
            }, best_ckpt_path)
            print(f"  ✓ New best val F1: {best_val_f1:.4f} — checkpoint saved")

        if early_stopper.step(val_metrics["macro_f1"]):
            print(f"\n  Early stopping triggered at epoch {epochs_phase1 + epoch}")
            break

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path
