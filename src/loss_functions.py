
import torch
import torch.nn as nn
import torch.nn.functional as F

class InterScaleContrastiveLoss(nn.Module):
    """
    Forces the Fine prediction (Scale 0) to be distinct from the Coarse prediction (Scale 2)
    for the same spatial region.
    
    Formula: L = max(0, margin - ||z_fine - z_coarse||_2)
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z_fine, z_coarse):
        # z_fine: (B, N, D)
        # z_coarse: (B, N, D)
        
        # Calculate Euclidean distance
        dist = F.pairwise_distance(z_fine, z_coarse, p=2) # (B, N)
        
        # Contrastive loss: we want distance > margin
        loss = torch.clamp(self.margin - dist, min=0.0)
        
        return loss.mean()

class HierarchicalConsistencyLoss(nn.Module):
    """
    Forces the Coarse prediction (Scale 2) to be a summary (average) of the Fine predictions (Scale 0).
    
    Formula: L = || z_coarse - Average(z_fine) ||_2
    
    Note: This assumes z_fine and z_coarse are aligned. 
    If z_fine corresponds to multiple patches that pool into z_coarse, 
    the caller must handle the pooling before passing to this loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_fine, z_coarse):
        # z_fine: (B, N, D) - Should be already pooled or aligned to z_coarse
        # z_coarse: (B, N, D)
        
        # MSE or Smooth L1 between coarse and fine
        loss = F.smooth_l1_loss(z_coarse, z_fine)
        return loss

class OrthogonalScaleLoss(nn.Module):
    """
    Forces Fine and Coarse features to be orthogonal (uncorrelated).
    
    Formula: L = | z_fine . z_coarse |
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_fine, z_coarse):
        # Normalize vectors to focus on direction
        z_fine_norm = F.normalize(z_fine, dim=-1)
        z_coarse_norm = F.normalize(z_coarse, dim=-1)
        
        # Dot product
        dot_prod = (z_fine_norm * z_coarse_norm).sum(dim=-1) # (B, N)
        
        # Minimize absolute dot product (push towards 0)
        loss = torch.abs(dot_prod).mean()
        return loss
