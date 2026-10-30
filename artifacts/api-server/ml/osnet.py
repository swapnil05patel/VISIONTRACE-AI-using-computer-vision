"""Eval-only OSNet x1.0 compatible with Kaiyang Zhou torchreid checkpoints (MIT).

Architecture follows deep-person-reid `torchreid.models.osnet` so official
`osnet_x1_0` weights can be loaded without depending on the torchreid package.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        reduced = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, in_channels, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return input_x * x


class OSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        branch = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(
            self.conv2d(x1)
        )
        x = self.conv3(branch)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return nn.functional.relu(x + identity, inplace=True)


class OSNet(nn.Module):
    def __init__(self, num_classes: int = 1000, channels: tuple[int, int, int, int] = (64, 256, 384, 512)):
        super().__init__()
        self.feature_dim = channels[3]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], 2, reduce_spatial_size=True)
        self.conv3 = self._make_layer(channels[1], channels[2], 2, reduce_spatial_size=True)
        self.conv4 = self._make_layer(channels[2], channels[3], 2, reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        reduce_spatial_size: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_channels, out_channels)]
        for _ in range(1, blocks):
            layers.append(OSBlock(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims: int, input_dim: int) -> nn.Sequential | None:
        if fc_dims is None or fc_dims < 0:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, fc_dims),
            nn.BatchNorm1d(fc_dims),
            nn.ReLU(inplace=True),
        )

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.featuremaps(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return y


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "", 1)] = value
    return cleaned


def load_osnet(weights: Path, device: str) -> OSNet:
    # Validate device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    
    checkpoint = torch.load(str(weights), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        state = checkpoint
    state = _strip_module_prefix(state)
    classifier_weight = state.get("classifier.weight")
    num_classes = int(classifier_weight.shape[0]) if classifier_weight is not None else 1000
    model = OSNet(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_core = [key for key in missing if not key.startswith("classifier")]
    if missing_core:
        raise RuntimeError(f"OSNet checkpoint is missing required tensors: {missing_core[:8]}")
    model.eval()
    model = model.to(device)
    return model
