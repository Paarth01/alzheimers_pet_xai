"""
models/resnet_cbam.py
─────────────────────────────────────────────────────────────────────
ResNet50 + CBAM attention model for 4-class AD staging.

Architecture:
  ImageNet pretrained ResNet50
    ↓
  [Layer 1] → [Layer 2] → [Layer 3] → CBAM(1024) → [Layer 4]
    ↓
  AdaptiveAvgPool2d(1,1)
    ↓
  FC(2048 → 512) → ReLU → Dropout(0.5) → FC(512 → 4) → Softmax

CBAM is inserted after Layer 3 (feature maps: B x 1024 x 14 x 14)
to attend to high-level semantic features before the final residual
block — maximizing impact while preserving spatial resolution.
─────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional
from models.cbam import CBAM


class ResNet50CBAM(nn.Module):
    """
    ResNet50 backbone with CBAM attention for 4-class AD classification.

    Args:
        num_classes:      Number of output classes (default 4)
        pretrained:       Load ImageNet pretrained weights (default True)
        use_cbam:         Toggle CBAM module on/off for ablation (default True)
        reduction_ratio:  CBAM channel reduction ratio (default 16)
        kernel_size:      CBAM spatial attention kernel size (default 7)
        dropout_rate:     Dropout probability in classifier head (default 0.5)
    """

    def __init__(self,
                 num_classes: int = 4,
                 pretrained: bool = True,
                 use_cbam: bool = True,
                 reduction_ratio: int = 16,
                 kernel_size: int = 7,
                 dropout_rate: float = 0.5):
        super().__init__()
        self.use_cbam = use_cbam

        # ── Load pretrained ResNet50 backbone via timm ────────────
        backbone = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=0  # Remove classifier
        )

        # ── Extract ResNet layers ─────────────────────────────────
        self.conv1    = backbone.conv1      # 7×7 conv, stride 2
        self.bn1      = backbone.bn1
        self.act1     = backbone.act1
        self.maxpool  = backbone.maxpool

        self.layer1   = backbone.layer1    # 64  channels  → 56×56
        self.layer2   = backbone.layer2    # 256 channels  → 28×28 (after stride)
        self.layer3   = backbone.layer3    # 512 channels  → 14×14 (but timm uses 1024 here)
        self.layer4   = backbone.layer4    # 2048 channels → 7×7

        # ── CBAM after Layer 3 ────────────────────────────────────
        # ResNet50 Layer 3 output: (B, 1024, 14, 14)
        if use_cbam:
            self.cbam = CBAM(
                in_channels=1024,
                reduction_ratio=reduction_ratio,
                kernel_size=kernel_size
            )
        else:
            self.cbam = nn.Identity()

        # ── Global average pooling ────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier head ───────────────────────────────────────
        # ResNet50 Layer 4 output: (B, 2048, 7, 7) → pool → (B, 2048)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes, bias=True),
        )

        # ── Initialize new layers ─────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier head with Xavier uniform."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            logits: (B, num_classes)  — raw logits (use CrossEntropyLoss)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.layer1(x)   # (B, 256, 56, 56)
        x = self.layer2(x)   # (B, 512, 28, 28)
        x = self.layer3(x)   # (B, 1024, 14, 14)

        # ── CBAM attention ────────────────────────────────────────
        x = self.cbam(x)     # (B, 1024, 14, 14) — refined by attention

        x = self.layer4(x)   # (B, 2048, 7, 7)

        # Pooling and classify
        x = self.global_pool(x)   # (B, 2048, 1, 1)
        logits = self.classifier(x)  # (B, num_classes)
        return logits

    def get_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps at each stage.
        Used for Grad-CAM and feature visualization.

        Returns:
            dict with keys: layer1, layer2, layer3, cbam_out, layer4
        """
        features = {}

        x = self.conv1(x); x = self.bn1(x); x = self.act1(x); x = self.maxpool(x)
        x = self.layer1(x); features["layer1"] = x
        x = self.layer2(x); features["layer2"] = x
        x = self.layer3(x); features["layer3"] = x
        x = self.cbam(x);   features["cbam_out"] = x
        x = self.layer4(x); features["layer4"] = x

        return features

    def freeze_backbone(self, freeze_bn: bool = True):
        """
        Phase 1 training: Freeze backbone, train only classifier head.
        Optionally freeze BatchNorm layers.
        """
        for name, param in self.named_parameters():
            if "classifier" not in name and "cbam" not in name:
                param.requires_grad = False

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Phase 1] Trainable: {trainable:,} / {total:,} parameters")

    def unfreeze_all(self):
        """Phase 2 training: Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Phase 2] Trainable: {trainable:,} parameters (all layers)")

    def get_parameter_groups(self, lr_backbone: float = 1e-4,
                              lr_head: float = 1e-3) -> list:
        """
        Return parameter groups with differential learning rates.
        Backbone gets lower LR to preserve pretrained features.
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "cbam" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params,     "lr": lr_head,     "name": "head"},
        ]


def build_model(num_classes: int = 4, pretrained: bool = True,
                use_cbam: bool = True, dropout_rate: float = 0.5,
                device: str = "cuda") -> ResNet50CBAM:
    """
    Factory function to instantiate and move model to device.

    Args:
        num_classes:   Number of classification classes
        pretrained:    Use ImageNet pretrained weights
        use_cbam:      Enable/disable CBAM for ablation study
        dropout_rate:  Classifier head dropout
        device:        Target device ("cuda" | "cpu" | "mps")

    Returns:
        ResNet50CBAM model on target device
    """
    model = ResNet50CBAM(
        num_classes=num_classes,
        pretrained=pretrained,
        use_cbam=use_cbam,
        dropout_rate=dropout_rate
    )

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ResNet50+{'CBAM' if use_cbam else 'NoAttn'} | "
          f"Total params: {total_params:,} | Trainable: {trainable:,} | "
          f"Device: {device}")

    return model


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    model = build_model(num_classes=4, pretrained=False, device="cpu")
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4, 4), f"Expected (4,4), got {logits.shape}"
    print(f"Forward pass OK: input {x.shape} → logits {logits.shape}")

    # Ablation check
    model_no_cbam = build_model(num_classes=4, pretrained=False,
                                 use_cbam=False, device="cpu")
    logits2 = model_no_cbam(x)
    assert logits2.shape == (4, 4)
    print("Ablation model (no CBAM) OK")
