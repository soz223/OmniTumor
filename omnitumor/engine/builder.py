"""Build the OmniTumor model by attaching VRA to a frozen BiomedParse SEEM."""
from __future__ import annotations
import os
import logging

import torch

from utilities.arguments import load_opt_from_config_files
from modeling.architectures import build_model
from modeling.BaseModel import BaseModel

from omnitumor.modeling.backbone import VRAWrappedBackbone

logger = logging.getLogger(__name__)


def build_omnitumor_model(pretrained_path: str, reduction: int = 4,
                          kernel_size: int = 3, device: str = 'cpu'):
    config_file = "configs/biomed_seg_lang_v1.yaml"
    opt = load_opt_from_config_files([config_file])
    opt['MODEL']['NAME'] = 'seem_model_v1'
    opt['MODEL']['HEAD'] = 'xdecoder_head'
    opt['MODEL']['ENCODER']['BINARY_CLASSES'] = True
    opt['device'] = device

    model = build_model(opt)
    base_model = BaseModel(opt, model)

    if os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state = ckpt['model'] if 'model' in ckpt else ckpt
        remapped = {f'model.{k}': v for k, v in state.items()}
        model_dict = base_model.state_dict()
        compatible = {k: v for k, v in remapped.items()
                      if k in model_dict}
        model_dict.update(compatible)
        base_model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(compatible)}/{len(state)} pretrained params")

    backbone = base_model.model.backbone
    wrapped = VRAWrappedBackbone(backbone, reduction=reduction, kernel_size=kernel_size)
    base_model.model.backbone = wrapped

    for name, param in base_model.named_parameters():
        param.requires_grad = ('adapters' in name)

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Trainable: {trainable/1e6:.1f}M / Total: {total/1e6:.1f}M")

    return base_model, opt
