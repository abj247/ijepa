#!/usr/bin/env python3
"""
Script to check model parameter count for I-JEPA models.
Verifies that ViT-Small and ViT-Base are under 100M parameters.
"""

import sys
import argparse

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num_params):
    """Format parameter count in millions."""
    return f"{num_params / 1e6:.2f}M"

def check_model_size(model_name='vit_small', patch_size=16, img_size=224):
    """
    Check the parameter count for I-JEPA model (encoder + predictor).
    
    Args:
        model_name: Name of the model ('vit_small' or 'vit_base')
        patch_size: Patch size for the model
        img_size: Input image size
    """
    print("=" * 70)
    print(f"Checking Model Size: {model_name}")
    print("=" * 70)
    
    try:
        import torch
        import src.models.vision_transformer as vit
        from src.models.vision_transformer import VIT_EMBED_DIMS
        
        # Model configuration
        pred_depth = 6  # Predictor depth
        
        # Get embed dim for the model
        embed_dim = VIT_EMBED_DIMS[model_name]
        pred_emb_dim = embed_dim // 2  # Half of encoder dim
        
        print(f"\nConfiguration:")
        print(f"  Model: {model_name}")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Input size: {img_size}x{img_size}")
        print(f"  Encoder embed dim: {embed_dim}")
        print(f"  Predictor embed dim: {pred_emb_dim}")
        print(f"  Predictor depth: {pred_depth}")
        
        # Initialize encoder
        print(f"\nInitializing encoder...")
        encoder = vit.__dict__[model_name](
            img_size=[img_size],
            patch_size=patch_size
        )
        encoder_params = count_parameters(encoder)
        print(f"  Encoder parameters: {format_params(encoder_params)}")
        
        # Initialize predictor
        print(f"\nInitializing predictor...")
        predictor = vit.__dict__['vit_predictor'](
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=pred_emb_dim,
            depth=pred_depth,
            num_heads=encoder.num_heads
        )
        predictor_params = count_parameters(predictor)
        print(f"  Predictor parameters: {format_params(predictor_params)}")
        
        # Total parameters
        total_params = encoder_params + predictor_params
        print(f"\n{'=' * 70}")
        print(f"Total Parameters: {format_params(total_params)}")
        print(f"{'=' * 70}")
        
        # Check if under 100M
        if total_params < 100e6:
            print(f"✓ Model is under 100M parameters ({format_params(total_params)})")
            print(f"  Remaining budget: {format_params(100e6 - total_params)}")
            return True
        else:
            print(f"✗ Model exceeds 100M parameters ({format_params(total_params)})")
            print(f"  Excess: {format_params(total_params - 100e6)}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Check I-JEPA model parameter count')
    parser.add_argument('--model', type=str, default='vit_small',
                        choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help='Model architecture to check')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Input image size')
    
    args = parser.parse_args()
    
    success = check_model_size(
        model_name=args.model,
        patch_size=args.patch_size,
        img_size=args.img_size
    )
    
    # Also check both models if no specific model was requested
    if args.model == 'vit_small':
        print("\n" + "=" * 70)
        print("Also checking ViT-Base for comparison...")
        print("=" * 70)
        check_model_size(model_name='vit_base', patch_size=args.patch_size, img_size=args.img_size)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
