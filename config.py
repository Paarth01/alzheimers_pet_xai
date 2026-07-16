"""
config.py
─────────────────────────────────────────────────────────
Central configuration for all hyperparameters, paths, and
training settings. Edit this file before each experiment.
─────────────────────────────────────────────────────────
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    # ── Paths ────────────────────────────────────────────
    # DataConfig class — update these paths
    adni_root: str = "C:/data/ADNI/NIfTI"          # Root dir for ADNI NIfTI scans
    oasis_root: str = "/data/OASIS3/FDG_PET"       # Root dir for OASIS-3 NIfTI scans
    adni_csv:  str = "C:/data/ADNI/adni_labels_final.csv"   # CSV: subject_id, scan_path, label
    oasis_csv: str = "/data/OASIS3/oasis_labels.csv"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "outputs"

    # ── Classes ──────────────────────────────────────────
    # 0=CN, 1=EMCI, 2=MCI, 3=AD
    class_names: List[str] = field(default_factory=lambda: ["CN", "EMCI", "MCI", "AD"])
    num_classes: int = 4

    # ── Splits ───────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # ── Preprocessing ────────────────────────────────────
    image_size: Tuple[int, int] = (224, 224)   # Resize target
    n_slices_per_volume: int = 50              # Informative slices selected per subject
    slice_axis: int = 1                        # 0=sagittal, 1=coronal, 2=axial
    intensity_threshold: float = 0.1          # Min mean intensity for slice selection
    normalize: bool = True                    # Z-score normalization per slice


@dataclass
class ModelConfig:
    backbone: str = "resnet50"           # timm model name
    pretrained: bool = True              # ImageNet pretrained weights
    use_cbam: bool = True                # Toggle CBAM attention
    cbam_reduction_ratio: int = 16       # Channel attention reduction ratio
    cbam_kernel_size: int = 7            # Spatial attention conv kernel size
    cbam_insert_after: str = "layer3"   # ResNet layer after which CBAM is inserted
    dropout_rate: float = 0.5           # Classifier head dropout
    num_classes: int = 4


@dataclass
class TrainConfig:
    # ── Core ─────────────────────────────────────────────
    epochs_phase1: int = 15              # Phase 1: head only
    epochs_phase2: int = 15              # Phase 2: full fine-tune
    batch_size: int = 32
    num_workers: int = 4

    # ── Learning rates ───────────────────────────────────
    lr_head: float = 1e-3               # Phase 1 classifier head LR
    lr_backbone: float = 1e-4           # Phase 1 backbone LR (frozen mostly)
    lr_finetune: float = 1e-5           # Phase 2 all-layers LR

    # ── Regularization ───────────────────────────────────
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1        # Helps with class boundary uncertainty

    # ── Scheduler ────────────────────────────────────────
    scheduler: str = "cosine"           # "cosine" | "step" | "plateau"
    warmup_epochs: int = 3

    # ── Early stopping ───────────────────────────────────
    patience: int = 7
    min_delta: float = 1e-4

    # ── Logging ──────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "alzheimers-pet-xai"
    wandb_run_name: str = "resnet50_cbam_fdgpet_2"
    log_interval: int = 10             # Log every N batches

    # ── Device ───────────────────────────────────────────
    device: str = "cuda"               # "cuda" | "cpu" | "mps"


@dataclass
class XAIConfig:
    # ── Grad-CAM ─────────────────────────────────────────
    gradcam_target_layer: str = "layer4"    # Last conv layer for Grad-CAM
    gradcam_n_samples: int = 200            # Samples for visualization

    # ── SHAP ─────────────────────────────────────────────
    shap_n_background: int = 200            # Background samples for DeepExplainer
    shap_n_test: int = 100                  # Test samples per class for SHAP

    # ── Faithfulness (Insertion / Deletion AUC) ──────────
    faithfulness_n_samples: int = 200       # Samples per method
    faithfulness_n_steps: int = 50          # Steps in pixel insertion/deletion curve
    faithfulness_batch_size: int = 16       # Batch size for faithfulness eval

    # ── Brain region mapping ─────────────────────────────
    # MNI-space region names mapped from AAL atlas (approximate slice ranges)
    brain_regions: dict = field(default_factory=lambda: {
        "PCC": {"description": "Posterior Cingulate Cortex", "mni_z_range": (-10, 20)},
        "Hippocampus": {"description": "Hippocampus", "mni_z_range": (-30, -5)},
        "Temporal": {"description": "Temporal-Parietal Cortex", "mni_z_range": (-20, 10)},
        "Frontal": {"description": "Frontal Cortex", "mni_z_range": (30, 70)},
    })

    # ── Output paths ─────────────────────────────────────
    heatmap_dir: str = "outputs/heatmaps"
    shap_dir: str = "outputs/shap"
    faithfulness_dir: str = "outputs/faithfulness"


# ── Instantiate configs ───────────────────────────────────
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()
xai_cfg = XAIConfig()

# ── Create output directories ─────────────────────────────
for d in [data_cfg.checkpoint_dir, data_cfg.log_dir, data_cfg.output_dir,
          xai_cfg.heatmap_dir, xai_cfg.shap_dir, xai_cfg.faithfulness_dir]:
    os.makedirs(d, exist_ok=True)
