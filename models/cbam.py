"""
models/cbam.py
─────────────────────────────────────────────────────────────────────
Convolutional Block Attention Module (CBAM)
Reference: Woo et al., ECCV 2018 — "CBAM: Convolutional Block
           Attention Module"

Architecture:
    Input F  →  Channel Attention  →  F'
             →  Spatial Attention  →  F''  (output)

Channel Attention:
    - Global average pool + Global max pool → shared MLP → sigmoid
    - Recalibrates WHICH feature channels to emphasize
    - Helps model focus on metabolically relevant feature maps

Spatial Attention:
    - Channel-wise avg + max pooling → concatenate → 7×7 conv → sigmoid
    - Recalibrates WHERE in the spatial map to focus
    - Helps model localize hypometabolic PET regions
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module.

    For feature map F ∈ R^(B × C × H × W):
      1. Squeeze spatial dimensions via AvgPool and MaxPool → (B, C, 1, 1)
      2. Excite through shared MLP: FC(C → C/r) → ReLU → FC(C/r → C)
      3. Merge: sigmoid(avg_out + max_out) → channel weights ∈ [0, 1]

    Args:
        in_channels:      Number of input channels C
        reduction_ratio:  Bottleneck reduction ratio r (default 16)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction_ratio, 1)

        # Shared MLP applied to both pooled descriptors
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_out = self.shared_mlp(self.avg_pool(x))    # (B, C)
        max_out = self.shared_mlp(self.max_pool(x))    # (B, C)

        # Merge and reshape to (B, C, 1, 1)
        attention = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    For channel-refined feature F' ∈ R^(B × C × H × W):
      1. Channel-wise AvgPool and MaxPool → (B, 1, H, W) each
      2. Concatenate → (B, 2, H, W)
      3. 7×7 conv → sigmoid → spatial weights ∈ [0, 1]^(H×W)

    Args:
        kernel_size: Conv kernel size (7 as per original paper)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=kernel_size, padding=padding, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        attention = torch.sigmoid(self.conv(concat))     # (B, 1, H, W)
        return x * attention


class CBAM(nn.Module):
    """
    Full Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention sequentially.

    Args:
        in_channels:      Feature map channels
        reduction_ratio:  Channel attention bottleneck ratio (default 16)
        kernel_size:      Spatial attention conv kernel size (default 7)

    Example:
        >>> cbam = CBAM(in_channels=1024)
        >>> x = torch.randn(4, 1024, 14, 14)
        >>> out = cbam(x)   # same shape: (4, 1024, 14, 14)
    """

    def __init__(self, in_channels: int,
                 reduction_ratio: int = 16,
                 kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)   # Channel refinement
        x = self.spatial_attention(x)   # Spatial refinement
        return x


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    model = CBAM(in_channels=1024, reduction_ratio=16, kernel_size=7)
    x = torch.randn(4, 1024, 14, 14)
    out = model(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"CBAM test passed. Input: {x.shape} → Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CBAM parameters: {n_params:,}")
