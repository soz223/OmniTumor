#!/usr/bin/env python3
"""OmniTumor training entry point.

Run from inside a BiomedParse checkout that contains OmniTumorData and the
BiomedParse pretrained weight. The ``omnitumor`` package must be importable
(e.g., by placing this repo on ``PYTHONPATH``).
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

# Allow `python scripts/train.py` from the repo root without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnitumor.engine.trainer import train

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='OmniTumorData')
    p.add_argument('--metadata', default='dataset_metadata_v2.json')
    p.add_argument('--use_augmented_prompts', type=int, default=1)
    p.add_argument('--use_tau_sampler', type=int, default=1)
    p.add_argument('--pretrained', default='pretrained/biomedparse_v1.pt')
    p.add_argument('--output_dir', default='output/omnitumor')
    p.add_argument('--resume', default='')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--effective_batch', type=int, default=16)
    p.add_argument('--num_slices', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--reduction', type=int, default=4)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--lambda_orth', type=float, default=0.5)
    p.add_argument('--lambda_octa', type=float, default=1.0)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--val_every', type=int, default=5)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
