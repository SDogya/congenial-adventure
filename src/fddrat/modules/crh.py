import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class ContinuousResidualHead(nn.Module):
    def __init__(self, H_a: int, D_a: int, D_v: int, hidden_dim: int = 512):
        super().__init__()
        self.H_a = H_a
        self.D_a = D_a
        output_dim = H_a * D_a

        # LazyLinear автоматически определяет input_dim при первом forward.
        # Это устраняет любое расхождение между obs_dim при __init__ и реальным z_v.
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialized = False

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, (nn.Linear,)) and not isinstance(m, nn.LazyLinear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self._initialized = True

    def forward(self, a_coarse: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        a_coarse: [B, H_a, D_a]
        z_v:      [B, D_v]
        Returns:  [B, H_a, D_a]
        """
        B = a_coarse.size(0)
        a_coarse_flat = a_coarse.reshape(B, -1)         # [B, H_a * D_a]
        x = torch.cat([a_coarse_flat, z_v], dim=1)      # [B, (H_a*D_a) + D_v]
        delta_a_flat = self.mlp(x)                       # [B, H_a * D_a]

        # Инициализируем веса после первой материализации LazyLinear
        if not self._initialized:
            self._init_weights()

        return delta_a_flat.reshape(B, self.H_a, self.D_a)