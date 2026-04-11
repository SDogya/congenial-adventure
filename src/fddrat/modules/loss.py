import torch
import torch.nn as nn
import torch.nn.functional as F

class FDDRATLoss(nn.Module):
    def __init__(self, lambda_ratio: float = 1.0, beta_mse: float = 1.0):
        super().__init__()
        self.lambda_ratio = lambda_ratio
        self.beta_mse = beta_mse
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        p_stop_logits: torch.Tensor, 
        tau_target: torch.Tensor, 
        delta_a: torch.Tensor, 
        residual_target: torch.Tensor, 
        K_sampled: torch.Tensor, 
        H_l: int
    ) -> torch.Tensor:
        # Cross Entropy Loss
        # Shift logits to match original target boundaries and prevent looking ahead
        logits_shifted = logits[:, :-1, :]
        
        logits_flat = logits_shifted.reshape(-1, logits_shifted.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss_ce = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
        
        # Ratio Loss
        loss_ratio = F.binary_cross_entropy_with_logits(p_stop_logits.view(-1), tau_target.view(-1).float())
        
        # MSE Loss with strict masking for sequence end boundary rules
        mse_loss_raw = F.mse_loss(delta_a, residual_target, reduction='none') # [B, H_a, D_a]
        
        # Flatten spatial dims to average per item
        mse_loss_item = mse_loss_raw.mean(dim=[1, 2]) # [B]
        
        # Masking: do not penalize if K_sampled >= H_l
        mask = (K_sampled < H_l).float()
        
        # Strict summation division to prevent gradient decay (posterior collapse)
        masked_mse = (mse_loss_item * mask).sum() / (mask.sum() + 1e-8)
        
        loss_total = loss_ce + self.lambda_ratio * loss_ratio + self.beta_mse * masked_mse
        
        return loss_total
