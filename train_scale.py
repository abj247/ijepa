# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# try:
#     # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
#     # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
#     # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
#     # --          TO EACH PROCESS
#     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
# except Exception:
#     pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.huggingface_dataset import make_huggingface

from src.helper import (
    load_checkpoint,
    init_opt)
from src.transforms import make_transforms

# [NEW] Import Scale-Aware Model
from src.models.vision_transformer_scale import (
    vit_tiny,
    vit_small,
    vit_base,
    vit_large,
    vit_huge,
    vit_giant
)
from src.loss_functions import (
    InterScaleContrastiveLoss,
    HierarchicalConsistencyLoss,
    OrthogonalScaleLoss
)

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(device, patch_size=16, crop_size=224, pred_depth=12, pred_emb_dim=384, model_name='vit_base'):
    # Helper to init model with scale-aware predictor
    if model_name == 'vit_tiny':
        model = vit_tiny(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    elif model_name == 'vit_small':
        model = vit_small(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    elif model_name == 'vit_base':
        model = vit_base(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    elif model_name == 'vit_large':
        model = vit_large(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    elif model_name == 'vit_huge':
        model = vit_huge(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    elif model_name == 'vit_giant':
        model = vit_giant(patch_size=patch_size, img_size=[crop_size], predictor_depth=pred_depth, predictor_embed_dim=pred_emb_dim)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)
    return model, model # Return same model as encoder and predictor (shared weights logic handled in train loop via copy)


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data'].get('root_path', None)  # Optional, only for ImageNet
    image_folder = args['data'].get('image_folder', None)  # Optional, only for ImageNet
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # -- AUX LOSS SETUP
    aux_loss_config = args.get('optimization', {}).get('aux_loss', None)
    aux_loss_fn = None
    aux_loss_weight = 0.0
    if aux_loss_config:
        name = aux_loss_config.get('name', 'inter_scale_contrastive')
        aux_loss_weight = float(aux_loss_config.get('weight', 0.0))
        margin = float(aux_loss_config.get('margin', 1.0))
        
        if name == 'inter_scale_contrastive':
            aux_loss_fn = InterScaleContrastiveLoss(margin=margin)
        elif name == 'hierarchical_consistency':
            aux_loss_fn = HierarchicalConsistencyLoss()
        elif name == 'orthogonal_scale':
            aux_loss_fn = OrthogonalScaleLoss()
        else:
            logger.warning(f"Unknown aux loss: {name}. Using None.")
            
        if aux_loss_fn:
            aux_loss_fn.to(device)
            logger.info(f"Using Aux Loss: {name} (weight={aux_loss_weight})")

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    # Note: init_model here is our local helper, not from src.helper
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    
    # In standard I-JEPA, predictor is part of the same class but we need to extract it?
    # Wait, in vision_transformer.py, VisionTransformer has NO predictor.
    # The predictor is separate.
    # Let's check src/helper.py's init_model to be sure.
    # Ah, I see. In src/helper.py, it calls vit_predictor separately.
    # So I should replicate that logic here but use my scale-aware predictor.
    
    from src.models.vision_transformer_scale import vit_predictor
    predictor = vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads)
    predictor.to(device)

    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    # Check if we should use HuggingFace dataset or ImageNet
    dataset_type = args['data'].get('dataset_type', 'imagenet')
    
    if dataset_type == 'huggingface':
        dataset_name = args['data'].get('dataset_name', 'tsbpp/fall2025_deeplearning')
        logger.info(f'Using HuggingFace dataset: {dataset_name}')
        _, unsupervised_loader, unsupervised_sampler = make_huggingface(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            dataset_name=dataset_name,
            split='train',
            drop_last=True)
    else:
        logger.info('Using ImageNet dataset')
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer_name = args['optimization'].get('optimizer', 'adamw')
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
        optimizer_name=optimizer_name)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        
        # Log number of batches
        logger.info(f'Starting epoch {epoch+1} with {len(unsupervised_loader)} iterations')

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            # [NEW] Generate Scale Indices
            # For each target block (masks_pred), decide if it is Local (0) or Global (1)
            # We will use a simple 50/50 probability for now, or maybe bias towards local?
            # Let's do 50/50.
            
            # masks_pred is a list of tensors. Each tensor is (B, num_tokens_in_block)
            # We need to generate a scale index for each block.
            # But wait, the predictor takes 'scale_indices' which we defined as (B, num_pred_tokens).
            # Since masks_pred is a list of blocks, and we process them together,
            # we should generate one scale ID per block per sample?
            # Or one scale ID per block (shared across batch)?
            # Usually masks are same for batch? No, masks are per sample if collator does it.
            # MBMaskCollator returns masks as list of tensors.
            # Let's check mask shape.
            # masks_pred[0] shape is (B, N_pred)
            
            # We will generate a random scale (0 or 1) for each block in masks_pred.
            # And we will create a tensor of that scale value matching the block size.
            
            B = imgs.shape[0]
            scale_indices_list = []
            
            # We need to store which blocks are global to pool their targets later
            block_scales = [] 
            
            for m in masks_pred:
                # m is (B, N)
                # Randomly choose scale 0 or 1 for this entire block
                # We can do it per sample, or per block.
                # Per block is simpler and likely what's intended (predict this region at scale S).
                # But wait, if we do it per block, then for the whole batch, block i is scale S.
                
                # Let's do per-block random choice.
                scale = np.random.randint(0, 3) # 0, 1, or 2
                
                # Store scale for this block to use in target generation
                # We need to know if it's scale 0 (8x8), 1 (16x16), or 2 (32x32)
                # Scale 0: No pooling
                # Scale 1: 2x2 pooling
                # Scale 2: 4x4 pooling
                block_scales.append(scale)
                
                # Create tensor of this scale
                # shape (B, N)
                s_tensor = torch.full_like(m, scale, dtype=torch.long, device=device)
                scale_indices_list.append(s_tensor)
                
            # Concatenate scale indices to match how predictor processes masks
            # In predictor: pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
            # Wait, predictor handles list of masks.
            # We passed scale_indices as a single tensor to predictor.forward.
            # We need to construct that tensor.
            # Predictor logic:
            # pos_embs = apply_masks(pos_embs, masks) -> returns concatenated tokens from all masks?
            # No, apply_masks returns (B, total_tokens, D) or similar?
            # Let's look at apply_masks in src/masks/utils.py
            # It usually gathers tokens.
            
            # If we pass scale_indices as a tensor (B, total_pred_tokens), we can use it.
            # Total pred tokens = sum of tokens in each mask block.
            # Concatenate scale indices to match how predictor processes masks
            # apply_masks concatenates in dim=0 (Batch dimension)
            # So we must concatenate scale indices in dim=0 as well.
            scale_indices = torch.cat(scale_indices_list, dim=0) # (M*B, N_per_block)
            
            # Wait, apply_masks might not just concatenate.
            # If masks are boolean? Or indices?
            # MBMaskCollator returns indices.
            # So apply_masks(x, masks) gathers x at indices.
            
            # Here we constructed scale_indices manually to match the *output* of apply_masks.
            # i.e., for every predicted token, we know its scale.
            # So we can pass this directly to predictor.
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        
                        # [NEW] Multi-scale Target Logic
                        # We need to handle Local vs Global targets differently.
                        # h is (B, N_patches, D)
                        
                        # We iterate over masks_pred and apply them individually
                        targets_list = []
                        for i, m in enumerate(masks_pred):
                            # m is (B, N_pred_tokens) indices
                            # Get features for this block
                            # h_block = apply_masks(h, [m]) # apply_masks expects list?
                            # apply_masks(x, masks)
                            
                            h_block = apply_masks(h, [m]) # (B, N_pred_tokens, D)
                            
                            scale = block_scales[i]
                            
                            if scale > 0:
                                # Scale 1 (2x2 pooling) or Scale 2 (4x4 pooling)
                                # We need to pool the features.
                                # But wait, the target block 'm' is a list of random indices.
                                # They are NOT necessarily a contiguous grid.
                                # Standard I-JEPA masks are random blocks.
                                # If we want to pool, we need the spatial structure.
                                # But 'h_block' is (B, N_pred_tokens, D). We lost spatial structure.
                                
                                # CRITICAL: We cannot do spatial pooling on random indices easily.
                                # However, I-JEPA masks ARE blocks. MBMaskCollator generates rectangular blocks.
                                # But they are flattened.
                                
                                # For "Global" prediction in the original 0/1 plan, we used .mean(dim=1).
                                # This averages ALL tokens in the block.
                                # This effectively simulates "Infinite Pooling" (The whole block becomes 1 vector).
                                
                                # If we want intermediate pooling (Scale 1 vs Scale 2), we need to be careful.
                                # Scale 1: 2x2 pooling.
                                # Scale 2: 4x4 pooling.
                                
                                # If the block size is small (e.g. 16 patches), 4x4 pooling = 1 vector (Average of all).
                                # If the block size is large, we might want sub-regions.
                                
                                # Given the complexity of reconstructing spatial structure from indices,
                                # let's simplify for this implementation:
                                # Scale 0: No pooling (Local).
                                # Scale 1: Average pool of the *entire* block (Global for that block).
                                # Scale 2: Average pool of the *entire* block (Global for that block).
                                
                                # Wait, if Scale 1 and 2 do the same thing (average everything), they are redundant?
                                # UNLESS we change the *size* of the block we predict?
                                # But MBMaskCollator fixes the block size.
                                
                                # Alternative:
                                # Scale 0: Predict individual patches.
                                # Scale 1: Predict average of the block.
                                # Scale 2: Predict average of the *entire image* (or larger context)?
                                # No, we can only predict what we mask.
                                
                                # Let's stick to the user's request of "Multi-Scale".
                                # If we can't easily do 2x2 pooling, maybe we just do:
                                # Scale 0: Local (Patch)
                                # Scale 1: Global (Block Average)
                                # Scale 2: Global (Block Average) -- effectively same target, but different embedding ID?
                                # That allows the model to learn "Medium" vs "Large" if we had different block sizes.
                                
                                # BUT, we can just use Scale 0 and Scale 1 for now if spatial pooling is hard.
                                # However, the user explicitly asked for 3 scales.
                                
                                # Let's implement "Hierarchical Pooling" approximation:
                                # Scale 0: h_block (Original)
                                # Scale 1: h_block mixed with average (50% local, 50% global)? No.
                                
                                # Let's go back to the "Average Pool" strategy.
                                # If we assume the block is roughly square, average pooling is a good proxy for "coarse" resolution.
                                # Maybe we can differentiate Scale 1 and Scale 2 by *how much* we pool?
                                # But we can't easily sub-pool without coordinates.
                                
                                # REVISED STRATEGY for 3 Scales on Unstructured Blocks:
                                # Scale 0: Exact Patch Features.
                                # Scale 1: Average of the Block.
                                # Scale 2: Average of the Block (Same target).
                                
                                # Wait, that's cheating.
                                # Let's look at MBMaskCollator. It returns masks.
                                # We don't have coordinates easily.
                                
                                # Okay, for 96x96 images with patch size 8:
                                # Grid is 12x12.
                                # A target block might be 4x4 patches (32x32 pixels).
                                # Scale 0: Predict 16 vectors.
                                # Scale 1: Predict 1 vector (Average of 16).
                                # Scale 2: Predict 1 vector (Average of 16).
                                
                                # This distinguishes "Detail" vs "Gist".
                                # To get an intermediate scale, we would need to pool 2x2 sub-regions.
                                
                                # Since we can't easily do 2x2 sub-pooling without rewriting the collator/masker,
                                # I will implement 3 scales where:
                                # Scale 0 = Local
                                # Scale 1 = Global (Average)
                                # Scale 2 = Global (Average)
                                
                                # This is a limitation of the current I-JEPA masking implementation.
                                # However, simply having 3 IDs allows the model to potentially learn different things if we *could* change the target.
                                
                                # Let's stick to 2 effective target types (Local vs Global) but map them to 3 IDs?
                                # No, that's confusing.
                                
                                # Let's stick to the 0/1 logic for now but call them "Fine" and "Coarse".
                                # If the user insists on 3 scales, we can add a 3rd ID that behaves like Coarse.
                                
                                # Actually, let's try to do 2x2 pooling by reshaping?
                                # If we assume the block is square...
                                # N_pred_tokens = H_b * W_b.
                                # If N=16, it's likely 4x4.
                                # We can reshape to (B, 4, 4, D), pool to (B, 2, 2, D), flatten to (B, 4, D), repeat to (B, 16, D).
                                
                                # Let's try to infer square shape.
                                N = h_block.shape[1]
                                Side = int(np.sqrt(N))
                                if Side * Side == N and scale == 1 and Side % 2 == 0:
                                    # We can do 2x2 pooling!
                                    # Reshape to (B, Side, Side, D)
                                    h_img = h_block.view(h_block.shape[0], Side, Side, h_block.shape[2])
                                    h_img = h_img.permute(0, 3, 1, 2) # (B, D, S, S)
                                    h_pool = F.avg_pool2d(h_img, kernel_size=2, stride=2) # (B, D, S/2, S/2)
                                    # Upsample back to S, S to match output size (nearest neighbor)
                                    h_up = F.interpolate(h_pool, size=(Side, Side), mode='nearest')
                                    h_block = h_up.permute(0, 2, 3, 1).reshape(h_block.shape[0], N, h_block.shape[2])
                                elif scale >= 1:
                                    # Scale 2 or Scale 1 (if not square): Full Average
                                    h_pooled = h_block.mean(dim=1, keepdim=True) # (B, 1, D)
                                    h_block = h_pooled.expand_as(h_block) # (B, N, D)

                                
                            targets_list.append(h_block)
                            
                        # Concatenate all targets
                        h = torch.cat(targets_list, dim=1)
                        
                        # Replicate for context views if needed (standard I-JEPA does this)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_target_dual(scale_b_val):
                    # Generate targets for Scale 0 AND Scale B (where B is 1 or 2)
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        
                        # Scale 0 Targets (No Pooling)
                        targets_0 = []
                        for m in masks_pred:
                            h_block = apply_masks(h, [m])
                            targets_0.append(h_block)
                        h_0 = torch.cat(targets_0, dim=1)
                        
                        # Scale B Targets
                        targets_b = []
                        for m in masks_pred:
                            h_block = apply_masks(h, [m])
                            
                            if scale_b_val == 2:
                                # Scale 2: Global Average Pool
                                h_pooled = h_block.mean(dim=1, keepdim=True)
                                h_block = h_pooled.expand_as(h_block)
                            elif scale_b_val == 1:
                                # Scale 1: 2x2 Pooling (if square)
                                N = h_block.shape[1]
                                Side = int(np.sqrt(N))
                                if Side * Side == N and Side % 2 == 0:
                                    # Reshape to (B, Side, Side, D)
                                    h_img = h_block.view(h_block.shape[0], Side, Side, h_block.shape[2])
                                    h_img = h_img.permute(0, 3, 1, 2) # (B, D, S, S)
                                    h_pool = F.avg_pool2d(h_img, kernel_size=2, stride=2) # (B, D, S/2, S/2)
                                    # Upsample back to S,S to match output shape
                                    h_up = F.interpolate(h_pool, size=(Side, Side), mode='nearest')
                                    h_block = h_up.permute(0, 2, 3, 1).reshape(h_block.shape[0], N, h_block.shape[2])
                                else:
                                    # Fallback to global average if not square/even
                                    h_pooled = h_block.mean(dim=1, keepdim=True)
                                    h_block = h_pooled.expand_as(h_block)
                                    
                            targets_b.append(h_block)
                        h_b = torch.cat(targets_b, dim=1)
                        
                        h_0 = repeat_interleave_batch(h_0, B, repeat=len(masks_enc))
                        h_b = repeat_interleave_batch(h_b, B, repeat=len(masks_enc))
                        
                        return h_0, h_b

                def forward_context(scale_b_val):
                    z = encoder(imgs, masks_enc)
                    
                    # [NEW] Dual Prediction Logic
                    if aux_loss_fn is not None:
                        # Create dual scales for Aux Loss
                        # Scale A = 0 (Fine)
                        # Scale B = scale_b_val (1 or 2)
                        
                        masks_pred_dual = masks_pred * 2 # Duplicate list [m1, m2, ..., m1, m2, ...]
                        
                        # We need to construct scale indices matching the order of masks_pred_dual
                        # First half: Scale 0
                        # Second half: Scale B
                        
                        scale_indices_list_dual = []
                        
                        # First half (Scale 0)
                        for m in masks_pred:
                            s = torch.zeros_like(m, dtype=torch.long, device=device)
                            scale_indices_list_dual.append(s)
                            
                        # Second half (Scale B)
                        for m in masks_pred:
                            s = torch.full_like(m, scale_b_val, dtype=torch.long, device=device)
                            scale_indices_list_dual.append(s)
                            
                        # Concatenate in dim 0 to match apply_masks
                        scale_indices_dual = torch.cat(scale_indices_list_dual, dim=0)
                        
                        # Forward pass with 2x blocks
                        z_pred = predictor(z, masks_enc, masks_pred_dual, scale_indices=scale_indices_dual)
                        
                        # Split output
                        # z_pred is (2*M*B, N, D)
                        total_rows = z_pred.shape[0]
                        half_rows = total_rows // 2
                        
                        z_0 = z_pred[:half_rows]
                        z_b = z_pred[half_rows:]
                        
                        return z_0, z_b
                        
                    else:
                        # Standard random scale training (if no aux loss)
                        z = predictor(z, masks_enc, masks_pred, scale_indices=scale_indices)
                        return z

                def loss_fn(z, h):
                    # z can be tuple (z_0, z_2) or single tensor
                    if isinstance(z, tuple):
                        z_0, z_2 = z
                        
                        # JEPA Loss: Both z_0 and z_2 should match the target h
                        # But wait, h (target) depends on the scale!
                        # Scale 0 target = Patch features.
                        # Scale 2 target = Pooled features.
                        
                        # We need to generate h_0 and h_2.
                        # forward_target() currently returns one h based on 'block_scales'.
                        # We need to update forward_target to return h_0 and h_2.
                        
                        # This requires refactoring forward_target too.
                        pass # Handled in train_step logic below
                    
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss


                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    if aux_loss_fn is not None:
                        # Dual Mode: Randomly choose Scale B (1 or 2)
                        # We want to train Scale 1 sometimes too.
                        scale_b_val = np.random.randint(1, 3) # 1 or 2
                        
                        h_0, h_b = forward_target_dual(scale_b_val)
                        z_0, z_b = forward_context(scale_b_val) # Returns tuple
                        
                        # [FIX] Reshape z to match h
                        # z is [M*B, N, D], h is [B, M*N, D]
                        # We need to rearrange z to [B, M*N, D]
                        M = len(masks_pred)
                        B_local = len(h_0) # Use local batch size
                        
                        # Ensure z_0 has expected shape
                        if z_0.shape[0] == M * B_local:
                            N = z_0.shape[1]
                            D = z_0.shape[2]
                            z_0 = z_0.view(M, B_local, N, D).permute(1, 0, 2, 3).reshape(B_local, M * N, D)
                            z_b = z_b.view(M, B_local, N, D).permute(1, 0, 2, 3).reshape(B_local, M * N, D)
                        
                        # Main JEPA Loss (Sum of both scales)
                        loss_jepa_0 = F.smooth_l1_loss(z_0, h_0)
                        loss_jepa_b = F.smooth_l1_loss(z_b, h_b)
                        loss_jepa = loss_jepa_0 + loss_jepa_b
                        
                        # Aux Loss
                        loss_aux = aux_loss_fn(z_0, z_b)
                        
                        loss = loss_jepa + (aux_loss_weight * loss_aux)
                        
                        # Apply AllReduce to total loss for logging/sync
                        loss = AllReduce.apply(loss)
                        
                    else:
                        # Standard Mode
                        h = forward_target()
                        z = forward_context()
                        loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
