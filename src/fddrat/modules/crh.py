import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class ContinuousResidualHead(nn.Module):
    def __init__(self, H_a: int, D_a: int, D_v: int, hidden_dim: int = 512):
        super().__init__()
        self.H_a = H_a
        self.D_a = D_a
        input_dim = H_a * D_a + D_v   # 112 + 250 = 362
        output_dim = H_a * D_a        # 112

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, a_coarse: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        B = a_coarse.size(0)
        x = torch.cat([a_coarse.reshape(B, -1), z_v], dim=1)
        return self.mlp(x).reshape(B, self.H_a, self.D_a)