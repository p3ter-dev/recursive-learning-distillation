"""
src/models.py
─────────────────────────────────────────────────────────────────────────────
Neural network architectures for Fashion-MNIST classification.

Two architectures are provided, both configurable via the YAML config:

  SmallCNN   Teacher model, ~212K trainable parameters at default settings.
             Two conv blocks with BatchNorm + two FC layers.

  TinyCNN    Student model, ~26K trainable parameters at default settings.
             One conv block + two FC layers.  Deliberately under-parameterised
             relative to SmallCNN to give distillation meaningful work to do.

Both accept **kwargs to absorb any extra config keys passed from YAML without
raising errors.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("rld.models")


# ─── Teacher ─────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    Two-block convolutional network for Fashion-MNIST.

    Architecture (default conv_channels=[32, 32], fc_hidden=128):
        Input   : 1 × 28 × 28
        Block 1 : Conv(1→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)  → 32×14×14
        Block 2 : Conv(32→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2) → 32×7×7
        Flatten : 32×7×7 = 1 568
        FC      : Linear(1568, 128) → Dropout → ReLU
        Head    : Linear(128, 10)

    Trainable parameters: ≈ 212K at defaults.
    """

    def __init__(
        self,
        conv_channels: List[int] | None = None,
        fc_hidden: int = 128,
        dropout: float = 0.3,
        num_classes: int = 10,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 32]
        c0, c1 = conv_channels[0], conv_channels[1]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
            # Block 2
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14 → 7
        )
        flat_dim = c1 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ─── Student ─────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """
    Single-block convolutional network for Fashion-MNIST.

    Architecture (default conv_channels=[8], fc_hidden=64):
        Input   : 1 × 28 × 28
        Block 1 : Conv(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2) → MaxPool(2×2)
                  → 8×7×7
        Flatten : 8×7×7 = 392
        FC      : Linear(392, 64) → Dropout → ReLU
        Head    : Linear(64, 10)

    Trainable parameters: ≈ 26K at defaults.
    """

    def __init__(
        self,
        conv_channels: List[int] | None = None,
        fc_hidden: int = 64,
        dropout: float = 0.3,
        num_classes: int = 10,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [8]
        c0 = conv_channels[0]

        self.features = nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
            nn.MaxPool2d(2, 2),          # 14 → 7
        )
        flat_dim = c0 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ─── Canonical ResNet Architectures ──────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet18(nn.Module):
    """
    Standard ResNet-18 adapted for 1-channel 28x28 Fashion-MNIST inputs.
    Uses a 3x3 stride-1 stem to preserve spatial dimensions.
    Trainable parameters: ≈ 11.17M.
    """
    def __init__(self, num_classes: int = 10, **kwargs: object) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.flatten(1)
        return self.linear(out)


class ResNet8(nn.Module):
    """
    Lightweight 8-layer ResNet designed as an efficient residual student network.
    Trainable parameters: ≈ 78K.
    """
    def __init__(self, num_classes: int = 10, **kwargs: object) -> None:
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 1, stride=1)
        self.layer2 = self._make_layer(32, 1, stride=2)
        self.layer3 = self._make_layer(64, 1, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.flatten(1)
        return self.linear(out)


# ─── Registry & factory ──────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {
    "SmallCNN": SmallCNN,
    "TinyCNN": TinyCNN,
    "ResNet18": ResNet18,
    "ResNet8": ResNet8,
}


def build_model(cfg: dict, role: str = "model") -> nn.Module:
    """
    Instantiate a model from a configuration sub-dict.

    Parameters
    ----------
    cfg  : dict containing at least ``name`` (str) plus any constructor kwargs
           (e.g. ``conv_channels``, ``fc_hidden``, ``dropout``).
    role : human-readable label used only for logging.

    Returns
    -------
    Initialised, untrained ``nn.Module`` on CPU.
    """
    name = cfg.get("name", "")
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.  Available: {list(_REGISTRY)}"
        )
    model = _REGISTRY[name](**cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Built %s (%s) with %s trainable parameters.",
        role, name, f"{n_params:,}",
    )
    return model
