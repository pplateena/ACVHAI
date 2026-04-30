"""
ModelA — ResNet18 backbone (frozen) + trainable head, standard spatial conv.
ModelB — same backbone but first Conv2d replaced with FFTConv2d (frozen filter bank).
"""

import torch
import torch.nn as nn
from torchvision import models
from fft_conv import FFTConv2d


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def build_model_a(num_classes: int = 10) -> nn.Module:
    """ResNet18 with frozen backbone + fresh classification head."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    _freeze(backbone)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def build_model_b(num_classes: int = 10) -> nn.Module:
    """
    ResNet18 with frozen backbone, but backbone.layer1's first Conv2d
    is replaced by FFTConv2d (frozen weights = static filter bank).
    Head is trainable.
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    _freeze(backbone)

    # Replace the very first conv (7×7, stride 2) with FFTConv2d
    backbone.conv1 = FFTConv2d.from_conv2d(backbone.conv1)

    # Replace all Conv2d inside layer1 with FFTConv2d
    for name, module in backbone.layer1.named_modules():
        if isinstance(module, nn.Conv2d):
            parent = backbone.layer1
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            fft_layer = FFTConv2d.from_conv2d(module)
            setattr(parent, parts[-1], fft_layer)

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())