#!/usr/bin/env python3
"""
Test script to verify HuggingFace dataset loading for I-JEPA.
"""

import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def test_dataset_loading():
    """Test loading the HuggingFace dataset."""
    print("=" * 60)
    print("Testing HuggingFace Dataset Loading")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        from src.datasets.huggingface_dataset import HuggingFaceDataset, make_huggingface
        from src.transforms import make_transforms
        import torch
        
        print("\n1. Testing basic dataset import...")
        dataset_name = 'tsbpp/fall2025_deeplearning'
        print(f"   Loading dataset: {dataset_name}")
        
        # Test basic HuggingFace loading
        print("\n2. Loading dataset directly from HuggingFace...")
        hf_dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
        print(f"   ✓ Dataset loaded successfully!")
        print(f"   ✓ Number of samples: {len(hf_dataset)}")
        
        # Check sample
        print("\n3. Inspecting first sample...")
        sample = hf_dataset[0]
        print(f"   ✓ Dataset columns: {list(sample.keys())}")
        
        if 'image' in sample:
            img = sample['image']
            print(f"   ✓ Image mode: {img.mode}")
            print(f"   ✓ Image size: {img.size}")
        else:
            print(f"   ✗ WARNING: 'image' column not found!")
            return False
        
        # Test with transforms
        print("\n4. Testing with I-JEPA transforms...")
        transform = make_transforms(
            crop_size=224,
            crop_scale=[0.3, 1.0],
            gaussian_blur=False,
            horizontal_flip=False,
            color_distortion=False,
            color_jitter=0.0
        )
        
        wrapped_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split='train',
            transform=transform,
            training=True
        )
        
        print(f"   ✓ Wrapped dataset created")
        print(f"   ✓ Dataset length: {len(wrapped_dataset)}")
        
        # Test getting a sample
        print("\n5. Testing sample retrieval with transforms...")
        img, target = wrapped_dataset[0]
        print(f"   ✓ Transformed image shape: {img.shape}")
        print(f"   ✓ Image dtype: {img.dtype}")
        print(f"   ✓ Target: {target}")
        
        # Test dataloader
        print("\n6. Testing DataLoader creation...")
        from src.masks.multiblock import MaskCollator as MBMaskCollator
        
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
        
        dataloader = torch.utils.data.DataLoader(
            wrapped_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            collate_fn=mask_collator
        )
        
        print(f"   ✓ DataLoader created successfully")
        
        # Get a batch
        print("\n7. Testing batch retrieval...")
        batch = next(iter(dataloader))
        imgs, masks_enc, masks_pred = batch
        print(f"   ✓ Batch images shape: {imgs[0].shape}")
        print(f"   ✓ Number of context masks: {len(masks_enc)}")
        print(f"   ✓ Number of prediction masks: {len(masks_pred)}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
