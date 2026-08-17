"""Spatially Adapted Encoder (SAE)."""
from __future__ import annotations
from typing import Iterable

import torch
import torch.nn as nn

from omnitumor.modeling.adapters import VolumetricResidualAdapter


class SpatiallyAdaptedEncoder(nn.Module):
    """Wraps a frozen 2D image encoder with VRAs after each stage.

    The 3D volume is folded ``(B, C, D, H, W) -> (B*D, C, H, W)`` so it can
    pass through a 2D backbone; the per-stage VRAs re-inject inter-slice
    context. Only the adapters are trained.
    """

    def __init__(
        self,
        backbone: nn.Module,
        stage_channels: dict[str, int],
        reduction: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.adapters = nn.ModuleDict({
            name: VolumetricResidualAdapter(c, reduction, kernel_size)
            for name, c in stage_channels.items()
        })

    @property
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return [p for p in self.adapters.parameters() if p.requires_grad]

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        B, C, D, H, W = volume.shape
        x2d = volume.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        feats = self.backbone(x2d)
        out = {}
        for stage, f in feats.items():
            if stage in self.adapters:
                f = self.adapters[stage](f, depth=D)
            out[stage] = f
        return out
