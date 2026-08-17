"""OmniTumorData dataset and SEEM-compatible collator."""
from __future__ import annotations
import os
import glob
import logging
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from detectron2.structures import BitMasks, Instances

logger = logging.getLogger(__name__)


class OmniTumorDataset(Dataset):
    """One ``(case, label)`` pair per sample, returning ``num_slices`` 2D slices."""

    def __init__(self, base_dir, metadata, split='train', num_slices=16,
                 image_size=1024, use_augmented_prompts=True):
        self.base_dir = base_dir
        self.num_slices = num_slices
        self.image_size = image_size
        self.use_augmented_prompts = use_augmented_prompts
        self.split = split

        split_ids = set(metadata['splits'][split])
        self.samples = []
        for s in metadata['samples']:
            if s['id'] not in split_ids:
                continue
            ds_info = metadata['datasets'][s['dataset']]
            for lid, linfo in ds_info['labels'].items():
                canonical = linfo['text_prompt']
                if canonical == '<routed_per_case>':
                    # ULS23: read per-sample text_prompts.
                    idx_in_labels = list(ds_info['labels'].keys()).index(lid)
                    canonical = s['text_prompts'][idx_in_labels]
                aug_pool = None
                if self.use_augmented_prompts and 'augmented_prompts_per_label' in s:
                    aug_pool = s['augmented_prompts_per_label'].get(lid)
                self.samples.append({
                    'case_id': s['case_id'],
                    'dataset': s['dataset'],
                    'image_dir': os.path.join(base_dir, s['image_path']),
                    'mask_dir': os.path.join(base_dir, s['mask_path']),
                    'pixel_value': linfo['pixel_value'],
                    'text_prompt': canonical,
                    'augmented_prompts': aug_pool,
                    'num_slices': s['num_slices'],
                })
        logger.info(f"[{split}] {len(self.samples)} (case, label) pairs; "
                    f"augmented_prompts={'on' if use_augmented_prompts else 'off'}")

    def cohort_index(self):
        cohorts = []
        cohort_to_idx = {}
        sample_idx = []
        for s in self.samples:
            ds = s['dataset']
            if ds not in cohort_to_idx:
                cohort_to_idx[ds] = len(cohorts)
                cohorts.append(ds)
            sample_idx.append(cohort_to_idx[ds])
        return cohorts, sample_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        slices = sorted(glob.glob(os.path.join(s['image_dir'], '*.png')))
        if len(slices) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.split == 'train' and s['augmented_prompts']:
            prompt = random.choice(s['augmented_prompts'])
        else:
            prompt = s['text_prompt']

        total = len(slices)
        D = min(self.num_slices, total)
        start = random.randint(0, total - D) if total > D else 0
        chosen = slices[start:start + D]

        slice_dicts = []
        for sl in chosen:
            img = np.array(Image.open(sl).convert('L'), dtype=np.float32)
            img_3ch = np.stack([img, img, img], axis=0)
            img_tensor = torch.from_numpy(img_3ch).float()

            mask_path = sl.replace('/images/', '/masks/')
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                m = (m == s['pixel_value']).astype(np.uint8)
            else:
                m = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            mask_tensor = torch.from_numpy(m).unsqueeze(0)
            H, W = img.shape[0], img.shape[1]

            instances = Instances((H, W))
            instances.gt_masks = BitMasks(mask_tensor.bool())
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances.gt_classes = torch.tensor([1], dtype=torch.int64)
            instances.is_things = torch.tensor([1], dtype=torch.int64)

            slice_dicts.append({
                'image': img_tensor,
                'text_prompt': prompt,
            })

        while len(slice_dicts) < self.num_slices:
            H, W = self.image_size, self.image_size
            instances = Instances((H, W))
            instances.gt_masks = BitMasks(torch.zeros((1, H, W), dtype=torch.bool))
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances.gt_classes = torch.tensor([1], dtype=torch.int64)
            instances.is_things = torch.tensor([1], dtype=torch.int64)
            slice_dicts.append({
                'image': torch.zeros((3, H, W), dtype=torch.float32),
                'instances': instances,
                'text_prompt': prompt,
            })

        return slice_dicts


def collate_seem(batch):
    flat = []
    depths = []
    prompts = []
    for item in batch:
        flat.extend(item)
        depths.append(len(item))
        prompts.append(item[0]['text_prompt'])
    return flat, depths, prompts
