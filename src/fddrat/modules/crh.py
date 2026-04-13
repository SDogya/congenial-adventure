import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class ContinuousResidualHead(nn.Module):
    def __init__(self, H_a: int, D_a: int, D_v: int, hidden_dim: int = 512):
        super().__init__()
        self.H_a = H_a
        self.D_a = D_a

        output_dim = H_a * D_a

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),   # ← автоматически подстраивается под реальный input_dim
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, a_coarse: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        B = a_coarse.size(0)
        a_coarse_flat = a_coarse.reshape(B, -1)
        x = torch.cat([a_coarse_flat, z_v], dim=1)
        delta_a_flat = self.mlp(x)
        return delta_a_flat.reshape(B, self.H_a, self.D_a)