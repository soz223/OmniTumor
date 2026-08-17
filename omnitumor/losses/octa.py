r"""Orthogonal Contrastive Text-Visual Alignment (OCTA) loss.

Given dense visual features ``F`` and a binary GT mask ``M``, foreground /
background prototypes are computed by masked GAP, then:

* the foreground prototype is pulled toward the text embedding
  :math:`z_{\rm text}` (cosine similarity -> 1);
* the background prototype is pushed orthogonal to :math:`z_{\rm text}`
  (|cos| -> 0), not anti-parallel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCTALoss(nn.Module):
    def __init__(self, lambda_orth: float = 0.5):
        super().__init__()
        self.lambda_orth = lambda_orth

    @staticmethod
    def _masked_gap(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        m = mask.float()
        denom = m.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
        return (feat * m).sum(dim=(-1, -2), keepdim=True) / denom

    def forward(
        self,
        feat: torch.Tensor,    # (B, C, h, w)
        mask: torch.Tensor,    # (B, 1, h, w)
        z_text: torch.Tensor,  # (B, C)
    ) -> dict[str, torch.Tensor]:
        v_fg = self._masked_gap(feat, mask).squeeze(-1).squeeze(-1)
        v_bg = self._masked_gap(feat, 1.0 - mask).squeeze(-1).squeeze(-1)

        v_fg = F.normalize(v_fg, dim=-1)
        v_bg = F.normalize(v_bg, dim=-1)
        z = F.normalize(z_text, dim=-1)

        cos_fg = (v_fg * z).sum(-1)
        cos_bg = (v_bg * z).sum(-1)

        align = (1.0 - cos_fg).mean()
        orth = cos_bg.abs().mean()
        loss = align + self.lambda_orth * orth
        return {"octa_loss": loss, "align": align, "orth": orth}
