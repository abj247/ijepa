
import torch
import torch.nn as nn
from src.models.vision_transformer_scale import vit_small, vit_predictor

def test_scale_aware_model():
    print("Initializing Scale-Aware ViT Small...")
    # Initialize encoder (to get dims)
    model = vit_small(patch_size=16)
    
    # Initialize predictor
    predictor = vit_predictor(
        num_patches=model.patch_embed.num_patches,
        embed_dim=model.embed_dim,
        predictor_embed_dim=384,
        depth=6,
        num_heads=6
    )
    
    print("Model initialized.")
    
    # Dummy inputs
    B = 1
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    embed_dim = 384 # ViT-Small embed dim
    
    # Context features (output of encoder)
    N_context = 1
    x = torch.randn(B, N_context, embed_dim)
    
    # Masks
    masks_x = [torch.randint(0, num_patches, (B, N_context))]
    
    # Target Masks
    N_target = 1
    masks_pred = [torch.randint(0, num_patches, (B, N_target))]
    
    # Simulate Dual Prediction (Scale 0 and Scale 2)
    masks_pred_dual = masks_pred * 2 # [m1, m1]
    
    # Scale Indices: First half 0, Second half B (randomly 1 or 2)
    # We need to construct scale indices matching the order of masks_pred_dual
    # apply_masks stacks masks in dim 0.
    # So we must stack scale indices in dim 0.
    
    scale_b_val = 1 # Test Scale 1
    
    s0 = torch.zeros((B, N_target), dtype=torch.long)
    sb = torch.full((B, N_target), scale_b_val, dtype=torch.long)
    
    # Since masks_pred has 1 item, masks_pred_dual has 2 items.
    # The first item corresponds to s0.
    # The second item corresponds to sb.
    scale_indices = torch.cat([s0, sb], dim=0) # (2*B, N_target)
    
    print(f"Testing Dual Prediction with Scale 0 and Scale {scale_b_val}...")
    print(f"Scale indices shape: {scale_indices.shape}")
    
    print("Running forward pass...")
    try:
        output = predictor(x, masks_x, masks_pred_dual, scale_indices=scale_indices)
        print("Forward pass successful!")
        print("Output shape:", output.shape)
        
        # Expected shape: (2*B, N_target, embed_dim)
        # 2 blocks * 1 batch = 2 rows.
        expected_shape = (2*B, N_target, embed_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print("Shape check passed.")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_scale_aware_model()
