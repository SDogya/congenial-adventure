import torch
import torch.nn as nn


class ContinuousResidualHead(nn.Module):
    """Predicts a continuous residual correction over the coarse decoded trajectory.

    The head receives the stop-gradient coarse trajectory concatenated with the
    visual context and outputs delta_a in the same normalized action space.
    Input dim is always fixed (H_a * D_a + D_v), enabling a static CUDA graph.

    Formula from hypothesis:
        delta_a = CRH([stop_gradient(a_coarse) || z_v])

    Args:
        H_a: Action horizon (OAT decoder sample_horizon, e.g. 32).
        D_a: Action dimension (e.g. 7 for LIBERO).
        D_v: Visual feature dimension from FusedObservationEncoder.
        hidden_dim: MLP hidden layer width.
    """

    def __init__(self, H_a: int, D_a: int, D_v: int, hidden_dim: int = 512):
        super().__init__()
        self.H_a = H_a
        self.D_a = D_a
        input_dim = H_a * D_a + D_v
        output_dim = H_a * D_a

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
        """
        Args:
            a_coarse: Coarse decoded trajectory [B, H_a, D_a], stop-gradient applied by caller.
            z_v: Visual features [B, D_v].

        Returns:
            delta_a: Residual correction [B, H_a, D_a].
        """
        B = a_coarse.size(0)
        x = torch.cat([a_coarse.reshape(B, -1), z_v], dim=1)
        return self.mlp(x).reshape(B, self.H_a, self.D_a)
