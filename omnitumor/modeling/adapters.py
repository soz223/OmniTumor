"""Volumetric Residual Adapter (VRA).

Two interchangeable variants are provided:

* ``VolumetricResidualAdapter`` (reference): dense 4D layout
  ``(B*D, C, H, W) -> (B*D, C, H, W)``.

* ``TokenVolumetricResidualAdapter``: token-sequence layout
  ``(B*D, N, C) -> (B*D, N, C)``, used inside the FocalNet backbone
  wrapper where tokens have not yet been reshaped back to a feature map.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class VolumetricResidualAdapter(nn.Module):
    """Depth-wise 3D adapter: bottleneck-down -> 3x1x1 Conv3D -> bottleneck-up."""

    def __init__(self, dim: int, reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        hidden = max(1, dim // reduction)
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.conv3d = nn.Conv3d(
            hidden, hidden, kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), bias=True,
        )
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        # Zero-init so the adapter starts as a no-op.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        BD, C, H, W = x.shape
        B = BD // depth
        residual = x

        z = x.permute(0, 2, 3, 1).reshape(BD, H * W, C)
        z = self.ln(z)
        z = self.down(z)
        z = z.reshape(B, depth, H, W, -1).permute(0, 4, 1, 2, 3)
        z = self.conv3d(z)
        z = self.act(z)
        z = z.permute(0, 2, 3, 4, 1).reshape(BD, H * W, -1)
        z = self.up(z)
        z = z.reshape(BD, H, W, C).permute(0, 3, 1, 2)
        return residual + z


class TokenVolumetricResidualAdapter(nn.Module):
    """VRA variant operating on token sequences ``(B*D, N, C)``."""

    def __init__(self, dim: int, reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        hidden = dim // reduction
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.conv3d = nn.Conv3d(
            hidden, hidden, kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0), bias=True,
        )
        self.act = nn.GELU()
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        BD, N, C = x.shape
        B = BD // depth
        residual = x
        x = self.ln(x)
        x = self.down(x)
        H = W = int(math.sqrt(N))
        x = x.view(B, depth, H, W, -1).permute(0, 4, 1, 2, 3)
        x = self.conv3d(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(BD, N, -1)
        x = self.act(x)
        x = self.up(x)
        return residual + x
