import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class ContinuousResidualHead(nn.Module):
    def __init__(self, H_a: int, D_a: int, D_v: int):
        super().__init__()
        self.H_a = H_a
        self.D_a = D_a
        
        input_dim = (H_a * D_a) + D_v
        output_dim = H_a * D_a
        hidden_dim = int(input_dim * 1.5)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Init weights
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, a_coarse: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        Continuous Residual Head
        a_coarse: [B, H_a, D_a]
        z_v: [B, D_v]
        Returns: [B, H_a, D_a]
        """
        B = a_coarse.size(0)
        
        # Flatten macro-trajectory
        a_coarse_flat = a_coarse.reshape(B, -1)  # [B, H_a * D_a]
        
        # Concatenate with visual features
        x = torch.cat([a_coarse_flat, z_v], dim=1)  # [B, (H_a*D_a) + D_v]
        
        # Predict residuals
        delta_a_flat = self.mlp(x)  # [B, H_a * D_a]
        
        # Reshape to trajectory
        delta_a = delta_a_flat.reshape(B, self.H_a, self.D_a)  # [B, H_a, D_a]
        
        return delta_a
