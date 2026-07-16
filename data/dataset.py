"""
data/dataset.py
─────────────────────────────────────────────────────────────────────
PyTorch Dataset classes for ADNI and OASIS-3 FDG-PET data.

Expected CSV format (adni_labels.csv / oasis_labels.csv):
    subject_id, scan_path,       label
    002_S_0413, /data/ADNI/...   CN
    002_S_0816, /data/ADNI/...   AD
    ...

Label mapping:  CN=0, EMCI=1, MCI=2, AD=3
─────────────────────────────────────────────────────────────────────
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, List
from data.preprocessing import preprocess_volume


# ── Label encoding ────────────────────────────────────────────────
LABEL_MAP = {"CN": 0, "EMCI": 1, "MCI": 2, "AD": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class ADNISliceDataset(Dataset):
    """
    Slice-level dataset for FDG-PET classification.

    Each 3D NIfTI volume is decomposed into N 2D slices.
    Each slice is treated as an independent sample at training time
    (subject-level aggregation is done at inference via majority vote
    or probability averaging — see evaluate.py).

    Args:
        df:           DataFrame with columns [scan_path, label]
        split:        "train" | "val" | "test"
        n_slices:     Slices to extract per volume
        target_size:  Spatial resize target (H, W)
        axis:         Slicing axis (1 = coronal, recommended for PET)
        augment:      Apply training augmentations
        cache:        Cache preprocessed tensors in memory (uses RAM)
    """

    def __init__(self, df: pd.DataFrame, split: str = "train",
                 n_slices: int = 50, target_size: Tuple[int, int] = (224, 224),
                 axis: int = 1, augment: bool = False, cache: bool = False):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.n_slices = n_slices
        self.target_size = target_size
        self.axis = axis
        self.augment = augment
        self.cache = cache
        self._cache: Dict[int, torch.Tensor] = {}

        # Build flat slice index: [(volume_idx, slice_idx), ...]
        self.index: List[Tuple[int, int]] = []
        for vol_idx in range(len(self.df)):
            for sl_idx in range(self.n_slices):
                self.index.append((vol_idx, sl_idx))

    def __len__(self) -> int:
        return len(self.index)

    def _load_volume(self, vol_idx: int) -> torch.Tensor:
        """Load and cache a volume's slices (n_slices, 3, H, W)."""
        if self.cache and vol_idx in self._cache:
            return self._cache[vol_idx]

        row = self.df.iloc[vol_idx]
        slices = preprocess_volume(
            nifti_path=row["scan_path"],
            axis=self.axis,
            n_slices=self.n_slices,
            target_size=self.target_size,
        )  # (n_slices, 3, H, W)

        if self.cache:
            self._cache[vol_idx] = slices

        return slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        vol_idx, sl_idx = self.index[idx]
        row = self.df.iloc[vol_idx]

        slices = self._load_volume(vol_idx)
        image = slices[sl_idx]   # (3, H, W)

        label = LABEL_MAP[row["label"]]
        return image, label, vol_idx   # vol_idx enables subject-level aggregation


class ADNIVolumeDataset(Dataset):
    """
    Volume-level dataset — returns all slices for a subject as a batch.
    Used at inference for subject-level probability aggregation.

    Returns:
        slices: (n_slices, 3, H, W)
        label:  int
        subject_id: str
    """

    def __init__(self, df: pd.DataFrame, n_slices: int = 50,
                 target_size: Tuple[int, int] = (224, 224), axis: int = 1):
        self.df = df.reset_index(drop=True)
        self.n_slices = n_slices
        self.target_size = target_size
        self.axis = axis

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slices = preprocess_volume(
            nifti_path=row["scan_path"],
            axis=self.axis,
            n_slices=self.n_slices,
            target_size=self.target_size,
        )  # (n_slices, 3, H, W)
        label = LABEL_MAP[row["label"]]
        subject_id = row.get("subject_id", str(idx))
        return slices, label, subject_id


def build_dataloaders(
    adni_csv: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    n_slices: int = 50,
    target_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    random_seed: int = 42,
    oasis_csv: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from ADNI CSV.
    Optionally includes OASIS-3 as a separate external test loader.

    Class imbalance is handled via WeightedRandomSampler for training.

    Returns:
        dict with keys: "train", "val", "test", and optionally "oasis"
    """

    df = pd.read_csv(adni_csv)

    # ── Stratified split ──────────────────────────────────────────
    test_ratio_adjusted = 1 - train_ratio - val_ratio
    df_train, df_temp = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df["label"], random_state=random_seed
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=test_ratio_adjusted / (test_ratio_adjusted + val_ratio),
        stratify=df_temp["label"], random_state=random_seed
    )

    # ── Datasets ─────────────────────────────────────────────────
    train_ds = ADNISliceDataset(df_train, split="train", n_slices=n_slices,
                                 target_size=target_size, augment=True)
    val_ds   = ADNIVolumeDataset(df_val, n_slices=n_slices, target_size=target_size)
    test_ds  = ADNIVolumeDataset(df_test, n_slices=n_slices, target_size=target_size)

    # ── Weighted sampler for class imbalance ─────────────────────
    labels = [LABEL_MAP[l] for l in df_train["label"]]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
    # Each volume contributes n_slices samples
    sample_weights_expanded = []
    for w in sample_weights:
        sample_weights_expanded.extend([w] * n_slices)

    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights_expanded),
        num_samples=len(sample_weights_expanded),
        replacement=True
    )

    # ── DataLoaders ───────────────────────────────────────────────
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }

    # ── Optional OASIS-3 external validation ─────────────────────
    if oasis_csv and os.path.exists(oasis_csv):
        df_oasis = pd.read_csv(oasis_csv)
        oasis_ds = ADNIVolumeDataset(df_oasis, n_slices=n_slices,
                                      target_size=target_size)
        loaders["oasis"] = DataLoader(oasis_ds, batch_size=1, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

    print(f"[Dataset] Train volumes: {len(df_train)} | "
          f"Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"[Dataset] Train slices: {len(train_ds)} | "
          f"Batch size: {batch_size}")
    print(f"[Dataset] Class distribution (train): "
          f"{dict(zip(['CN','EMCI','MCI','AD'], class_counts.tolist()))}")

    return loaders


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """
    Compute class weights for weighted CrossEntropyLoss.
    Weights = inverse of class frequency, normalized.

    Returns:
        Tensor of shape (num_classes,) on CPU
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].map(LABEL_MAP).value_counts().sort_index()
    weights = 1.0 / counts.values.astype(np.float32)
    weights = weights / weights.sum() * len(weights)   # Normalize
    return torch.FloatTensor(weights)
