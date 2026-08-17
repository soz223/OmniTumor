"""VRA-wrapped FocalNet backbone.

Inserts a ``TokenVolumetricResidualAdapter`` between every stage of the
frozen FocalNet backbone shipped with BiomedParse. The wrapper exposes the
same multi-scale feature dictionary (``res2 ... res5``) expected by the
downstream pixel decoder, while interleaving inter-slice context.
"""
from __future__ import annotations

import torch.nn as nn

from omnitumor.modeling.adapters import TokenVolumetricResidualAdapter


class VRAWrappedBackbone(nn.Module):
    def __init__(self, backbone, reduction: int = 4, kernel_size: int = 3):
        super().__init__()
        self.backbone = backbone
        self.depth = None

        self.adapters = nn.ModuleDict()
        dims = [int(backbone.embed_dim * (2 ** i)) for i in range(backbone.num_layers)]
        for i in range(backbone.num_layers):
            self.adapters[f'stage{i}'] = TokenVolumetricResidualAdapter(
                dims[i], reduction=reduction, kernel_size=kernel_size)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        D = self.depth
        assert D is not None, "Set .depth before forward"

        x = self.backbone.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.backbone.pos_drop(x)

        outs = {}
        for i in range(self.backbone.num_layers):
            layer = self.backbone.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            x = self.adapters[f'stage{i}'](x, D)
            x_out = x
            if i in self.backbone.out_indices:
                norm_layer = getattr(self.backbone, f'norm{i}')
                x_out_norm = norm_layer(x_out)
                out = x_out_norm.view(-1, H, W, self.backbone.num_features[i]) \
                                .permute(0, 3, 1, 2).contiguous()
                outs[f"res{i + 2}"] = out
        return outs

    @property
    def size_divisibility(self):
        return self.backbone.size_divisibility if hasattr(self.backbone, 'size_divisibility') else 32

    def output_shape(self):
        return self.backbone.output_shape() if hasattr(self.backbone, 'output_shape') else None
