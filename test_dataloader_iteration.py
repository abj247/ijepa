#!/usr/bin/env python3
"""Quick test to verify dataloader iteration"""

import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from src.datasets.huggingface_dataset import make_huggingface
from src.transforms import make_transforms
from src.masks.multiblock import MaskCollator as MBMaskCollator

print("Testing dataloader iteration...")

# Create transform
transform = make_transforms(
    crop_size=224,
    crop_scale=[0.3, 1.0],
    gaussian_blur=False,
    horizontal_flip=False,
    color_distortion=False,
    color_jitter=0.0
)

# Create mask collator
mask_collator = MBMaskCollator(
    input_size=224,
    patch_size=16,
    pred_mask_scale=[0.15, 0.2],
    enc_mask_scale=[0.85, 1.0],
    aspect_ratio=[0.75, 1.5],
    nenc=1,
    npred=4,
    allow_overlap=False,
    min_keep=10
)

# Create dataloader
dataset, dataloader, sampler = make_huggingface(
    transform=transform,
    batch_size=4,
    collator=mask_collator,
    pin_mem=True,
    num_workers=0,  # Use 0 for testing
    world_size=1,
    rank=0,
    dataset_name='tsbpp/fall2025_deeplearning',
    split='train',
    training=True,
    drop_last=True
)

print(f"Dataset length: {len(dataset)}")
print(f"Dataloader length: {len(dataloader)}")
print(f"Expected batches: {len(dataset) // 4}")

# Try to get first batch
print("\nAttempting to get first batch...")
try:
    batch = next(iter(dataloader))
    print("✓ Successfully got first batch!")
    udata, masks_enc, masks_pred = batch
    print(f"  Batch data shape: {udata[0].shape}")
    print(f"  Number of context masks: {len(masks_enc)}")
    print(f"  Number of prediction masks: {len(masks_pred)}")
except Exception as e:
    print(f"✗ Error getting batch: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
