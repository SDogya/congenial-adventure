import torch
import torch.nn as nn
import torch.nn.functional as F

class ShadowRouter(nn.Module):
    def __init__(self, D_v: int, alpha: float = 1.0):
        super().__init__()
        self.D_v = D_v
        self.alpha = alpha
        
        # Adaptive threshold shift mapped from visual features
        self.tau_mlp = nn.Linear(D_v, 1)
        
    def forward(self, q_t: torch.Tensor, k_prev: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        q_t:     [B, H_l, D_attn]
        k_prev:  [B, H_l, D_attn] 
        z_v:     [B, D_v]
        """
        # Cosine similarity over the hidden embedding dimension
        cos_sim = F.cosine_similarity(q_t, k_prev, dim=-1) # [B, H_l]
        
        # Adaptive shift
        tau_shift = self.tau_mlp(z_v) # [B, 1]
        
        # Compute stopping probability logits. alpha scales similarity.
        logits = self.alpha * cos_sim - tau_shift
        
        return logits
