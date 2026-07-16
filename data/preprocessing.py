"""
data/preprocessing.py
─────────────────────────────────────────────────────────────────────
FDG-PET preprocessing pipeline:
  1. Load NIfTI volume (nibabel)
  2. Skull stripping (optional mask)
  3. Z-score normalization
  4. Informative slice selection
  5. Resize to target dimensions
  6. Convert grayscale → 3-channel (for ImageNet backbone)

NOTE: MNI-space normalization is expected to be done BEFORE this
      script using SPM12 or FSL. Input NIfTI files should already
      be in MNI space and SUVR-normalized.
─────────────────────────────────────────────────────────────────────
"""

import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple, Optional


# ── Constants ─────────────────────────────────────────────────────
SLICE_AXES = {0: "sagittal", 1: "coronal", 2: "axial"}


def load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file and return as float32 numpy array (H x W x D)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data


def zscore_normalize(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize a 3D PET volume.
    Operates on non-zero voxels only to avoid background distortion.
    """
    mask = volume > 0
    if mask.sum() == 0:
        return volume
    mean = volume[mask].mean()
    std  = volume[mask].std()
    volume = (volume - mean) / (std + eps)
    volume[~mask] = 0.0
    return volume


def clip_intensity(volume: np.ndarray, percentile_low: float = 1.0,
                   percentile_high: float = 99.0) -> np.ndarray:
    """
    Clip extreme intensity outliers using percentile-based windowing.
    Reduces influence of scanner artifacts on normalization.
    """
    low  = np.percentile(volume[volume > 0], percentile_low)
    high = np.percentile(volume[volume > 0], percentile_high)
    return np.clip(volume, low, high)


def select_informative_slices(volume: np.ndarray, axis: int = 1,
                               n_slices: int = 50,
                               threshold: float = 0.1) -> List[np.ndarray]:
    """
    Select the N most informative 2D slices from a 3D PET volume.
    Informativeness is measured by mean intensity across the slice
    (after normalization), filtering out background/skull slices.

    Args:
        volume:    3D PET volume (H x W x D)
        axis:      Axis along which to slice (0=sag, 1=cor, 2=axial)
        n_slices:  Number of slices to select
        threshold: Minimum mean intensity for a slice to qualify

    Returns:
        List of 2D numpy arrays (selected slices)
    """
    n_total = volume.shape[axis]
    slices, scores = [], []

    for i in range(n_total):
        sl = np.take(volume, i, axis=axis)
        score = sl.mean()
        if score > threshold:
            slices.append((i, sl, score))

    if not slices:
        # Fallback: take middle n_slices
        mid = n_total // 2
        start = max(0, mid - n_slices // 2)
        slices = [(i, np.take(volume, i, axis=axis), 0.0)
                  for i in range(start, start + n_slices)]

    # Sort by informativeness and pick top N evenly spaced
    slices.sort(key=lambda x: x[2], reverse=True)
    selected = slices[:n_slices]
    # Re-sort by slice index for spatial ordering
    selected.sort(key=lambda x: x[0])

    return [s[1] for s in selected]


def slice_to_tensor(sl: np.ndarray,
                    target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Convert a 2D PET slice to a normalized 3-channel tensor.

    Steps:
      1. Normalize slice to [0, 255] range
      2. Convert to PIL Image
      3. Resize to target_size
      4. Convert to 3-channel RGB (repeat grayscale)
      5. Apply ImageNet normalization

    Returns:
        Tensor of shape (3, H, W)
    """
    # Min-max scale to [0, 255]
    sl_min, sl_max = sl.min(), sl.max()
    if sl_max - sl_min > 0:
        sl_scaled = ((sl - sl_min) / (sl_max - sl_min) * 255).astype(np.uint8)
    else:
        sl_scaled = np.zeros_like(sl, dtype=np.uint8)

    # PIL conversion
    pil_img = Image.fromarray(sl_scaled, mode="L")  # Grayscale
    pil_img = pil_img.resize(target_size, Image.BILINEAR)

    # Convert to RGB by repeating channels
    pil_rgb = Image.merge("RGB", [pil_img, pil_img, pil_img])

    # ImageNet normalization (standard for pretrained backbones)
    transform = transforms.Compose([
        transforms.ToTensor(),                         # [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],               # ImageNet mean
            std=[0.229, 0.224, 0.225]                 # ImageNet std
        )
    ])

    return transform(pil_rgb)  # (3, H, W)


def preprocess_volume(nifti_path: str,
                       axis: int = 1,
                       n_slices: int = 50,
                       target_size: Tuple[int, int] = (224, 224),
                       intensity_threshold: float = 0.1) -> torch.Tensor:
    """
    Full preprocessing pipeline for a single FDG-PET NIfTI volume.

    Returns:
        Tensor of shape (n_slices, 3, H, W)
    """
    # Step 1: Load
    volume = load_nifti(nifti_path)

    # Step 2: Clip outliers
    volume = clip_intensity(volume)

    # Step 3: Z-score normalize
    volume = zscore_normalize(volume)

    # Step 4: Select informative slices
    slices = select_informative_slices(
        volume, axis=axis, n_slices=n_slices, threshold=intensity_threshold
    )

    # Step 5: Convert each slice to tensor
    tensors = [slice_to_tensor(sl, target_size) for sl in slices]

    # Stack: (n_slices, 3, H, W)
    return torch.stack(tensors, dim=0)


# ── Augmentation transforms ───────────────────────────────────────

def get_train_transform(image_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Training augmentation.
    NOTE: Augmentations are conservative for PET scans to avoid
          introducing metabolic artifacts. No color jitter (grayscale
          converted to RGB — color changes are meaningless).
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomCrop(image_size, padding=10),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform() -> transforms.Compose:
    """Validation/test transform — normalization only."""
    return transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
