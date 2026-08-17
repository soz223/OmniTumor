"""End-to-end training loop for OmniTumor."""
from __future__ import annotations
import os
import json
import math
import random
import logging
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from detectron2.structures import ImageList

from omnitumor.engine.builder import build_omnitumor_model
from omnitumor.data.dataset import OmniTumorDataset, collate_seem
from omnitumor.data.sampler import TauScheduledMultiDatasetSampler
from omnitumor.losses.octa import OCTALoss

logger = logging.getLogger(__name__)


def _build_cohort_sampler(train_ds, args):
    cohorts, sample_to_cohort = train_ds.cohort_index()
    cohort_sizes = [0] * len(cohorts)
    for ci in sample_to_cohort:
        cohort_sizes[ci] += 1
    cohort_to_global = [[] for _ in range(len(cohorts))]
    for gi, ci in enumerate(sample_to_cohort):
        cohort_to_global[ci].append(gi)

    class _CohortBackedSampler(TauScheduledMultiDatasetSampler):
        """Maps tau-sampler local indices to global ConcatDataset indices."""
        def __init__(self, *a, _lookup, **kw):
            super().__init__(*a, **kw)
            self._lookup = _lookup


    sampler = _CohortBackedSampler(
        cohort_sizes, epochs_total=args.epochs, seed=42,
    )
    logger.info(f"tau-sampler: cohorts={cohorts}")
    logger.info(f"tau-sampler: sizes={cohort_sizes}")
    logger.info(f"tau-sampler: schedule warmup/main/cooldown = "
                f"{sampler.warmup_end}/{sampler.main_end - sampler.warmup_end}/"
                f"{args.epochs - sampler.main_end} epochs (tau 0.5/1.0/0.5)")
    return sampler


def train(args):
    warnings.filterwarnings('ignore', message='.*torch.cuda.amp.*')

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    logger.info(f"Using GPU {args.gpu}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    metadata_path = os.path.join(args.data_dir, args.metadata)
    with open(metadata_path) as f:
        metadata = json.load(f)
    logger.info(f"Loaded metadata: {metadata_path}")

    train_ds = OmniTumorDataset(args.data_dir, metadata, 'train',
                                num_slices=args.num_slices, image_size=1024,
                                use_augmented_prompts=args.use_augmented_prompts)
    val_ds = OmniTumorDataset(args.data_dir, metadata, 'val',
                              num_slices=args.num_slices, image_size=1024,
                              use_augmented_prompts=False)

    if args.use_tau_sampler:
        train_sampler = _build_cohort_sampler(train_ds, args)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  collate_fn=collate_seem)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True,
                                  collate_fn=collate_seem)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_seem)

    model, opt = build_omnitumor_model(args.pretrained, reduction=args.reduction,
                                       kernel_size=args.kernel_size, device='cpu')
    model = model.to(device)

    seem = model.model
    seem.task_switch['grounding'] = False
    seem.task_switch['spatial'] = False
    backbone = seem.backbone
    lang_encoder = seem.sem_seg_head.predictor.lang_encoder
    pixel_decoder = seem.sem_seg_head.pixel_decoder
    predictor = seem.sem_seg_head.predictor
    criterion = seem.criterion

    octa_criterion = OCTALoss(lambda_orth=args.lambda_orth).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

    steps_per_epoch = len(train_loader)
    accum_steps = max(1, args.effective_batch // args.batch_size)
    opt_steps_per_epoch = max(1, steps_per_epoch // accum_steps)
    total_opt_steps = args.epochs * opt_steps_per_epoch
    warmup_opt_steps = args.warmup_epochs * opt_steps_per_epoch

    def lr_schedule(opt_step):
        if opt_step < warmup_opt_steps:
            return max(opt_step / max(warmup_opt_steps, 1), 0.01)
        progress = (opt_step - warmup_opt_steps) / max(total_opt_steps - warmup_opt_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler()

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', -1) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    weight_dict = criterion.weight_dict
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Epochs: {args.epochs}, Steps/epoch: {steps_per_epoch}, "
                f"Accum: {accum_steps}, Effective batch: {args.batch_size * accum_steps}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0

        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
            logger.info(f"epoch {epoch}: sampler tau = {train_sampler._current_tau()}")

        for step, (batched_inputs, depths, prompts) in enumerate(train_loader):
            D = depths[0]
            BD = len(batched_inputs)

            images_list = [x["image"].to(device) for x in batched_inputs]
            images_list = [(x - seem.pixel_mean) / seem.pixel_std for x in images_list]
            images = ImageList.from_tensors(images_list, seem.size_divisibility)

            lang_encoder.get_text_embeddings(seem.train_class_names, is_eval=False)
            targets = seem.prepare_targets(batched_inputs, images)

            backbone.depth = D
            with torch.amp.autocast('cuda'):
                features = backbone(images.tensor)

                feat = features['res5']
                _, C_feat, h, w = feat.shape
                masks_for_octa = [bi['instances'].gt_masks.tensor.float()[0] for bi in batched_inputs]
                masks_stack = torch.stack(masks_for_octa).to(device)
                masks_down = F.interpolate(
                    masks_stack.unsqueeze(1), size=(h, w), mode='nearest')

                with torch.no_grad():
                    lang_encoder.get_text_embeddings(prompts * D, is_eval=False)
                    text_emb = getattr(lang_encoder, 'default_text_embeddings')
                    if text_emb.dim() == 2:
                        text_emb = text_emb[0:1]
                    text_emb = text_emb.expand(BD, -1)
                    if text_emb.shape[-1] != C_feat:
                        text_emb = F.adaptive_avg_pool1d(
                            text_emb.unsqueeze(1), C_feat).squeeze(1)
                    lang_encoder.get_text_embeddings(seem.train_class_names, is_eval=False)

                octa_out = octa_criterion(feat, masks_down, text_emb)
                octa_loss = octa_out['octa_loss']

                mask_features, _, multi_scale_features = pixel_decoder.forward_features(features)
                outputs = predictor(multi_scale_features, mask_features, extra={}, task='seg')

                criterion.losses = ["labels", "masks"]
                extra_for_loss = {
                    'lang_logit': lang_encoder.logit_scale,
                    'class_embeddings': getattr(lang_encoder, 'default_text_embeddings'),
                }
                losses = criterion(outputs, targets, extra_for_loss)

                total_loss = torch.tensor(0.0, device=device)
                for k, v in losses.items():
                    if k in weight_dict:
                        total_loss = total_loss + v * weight_dict[k]

                total_loss = total_loss + args.lambda_octa * octa_loss
                total_loss = total_loss / accum_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            loss_val = total_loss.item() * accum_steps
            epoch_losses['total'] += loss_val
            epoch_losses['octa'] += octa_loss.item()
            for k, v in losses.items():
                if k in weight_dict and isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
            num_batches += 1

            if step % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                mem = torch.cuda.max_memory_allocated() / 1e9
                bce = losses.get('loss_mask_bce_0', torch.tensor(0.0))
                dice = losses.get('loss_mask_dice_0', torch.tensor(0.0))
                ce = losses.get('loss_mask_ce_0', torch.tensor(0.0))
                if isinstance(bce, torch.Tensor): bce = bce.item()
                if isinstance(dice, torch.Tensor): dice = dice.item()
                if isinstance(ce, torch.Tensor): ce = ce.item()
                logger.info(f"E{epoch} [{step}/{steps_per_epoch}] "
                            f"loss={loss_val:.4f} octa={octa_loss.item():.4f} "
                            f"bce={bce:.4f} dice={dice:.4f} ce={ce:.4f} "
                            f"lr={lr:.2e} mem={mem:.1f}GB")

        avg_loss = epoch_losses['total'] / max(num_batches, 1)
        avg_octa = epoch_losses['octa'] / max(num_batches, 1)
        logger.info(f"Epoch {epoch} done: avg_loss={avg_loss:.4f} avg_octa={avg_octa:.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'args': vars(args),
            }
            path = os.path.join(args.output_dir, f'omnitumor_epoch{epoch:03d}.pth')
            torch.save(ckpt, path)
            logger.info(f"Saved {path}")

        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi, (vbatch, vdepths, _) in enumerate(val_loader):
                    if vi >= 30:
                        break
                    vD = vdepths[0]
                    vimgs = [x["image"].to(device) for x in vbatch]
                    vimgs = [(x - seem.pixel_mean) / seem.pixel_std for x in vimgs]
                    vimages = ImageList.from_tensors(vimgs, seem.size_divisibility)

                    lang_encoder.get_text_embeddings(seem.train_class_names, is_eval=False)
                    vtargets = seem.prepare_targets(vbatch, vimages)

                    backbone.depth = vD
                    with torch.amp.autocast('cuda'):
                        vfeatures = backbone(vimages.tensor)
                        vmf, _, vmsf = pixel_decoder.forward_features(vfeatures)
                        voutputs = predictor(vmsf, vmf, extra={}, task='seg')

                        vextra_loss = {
                            'lang_logit': lang_encoder.logit_scale,
                            'class_embeddings': getattr(lang_encoder, 'default_text_embeddings'),
                        }
                        criterion.losses = ['labels', 'masks']
                        vl = criterion(voutputs, vtargets, vextra_loss)
                        vl_sum = sum(v * weight_dict.get(k, 0) for k, v in vl.items()
                                     if isinstance(v, torch.Tensor) and k in weight_dict)
                        val_losses.append(vl_sum.item())

            avg_val = sum(val_losses) / len(val_losses) if val_losses else 999
            logger.info(f"  Val loss: {avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'args': vars(args),
                }, os.path.join(args.output_dir, 'omnitumor_best.pth'))
                logger.info("  New best model saved")
            model.train()

    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'args': vars(args),
    }, os.path.join(args.output_dir, 'omnitumor_final.pth'))
    logger.info("Training complete")
