import torch
import torch.nn as nn
import torch.nn.functional as F


class ShadowRouter(nn.Module):
    """Computes per-step stopping probability logits for any-time inference.

    The router operates in "shadow" — it is trained with a separate BCE loss
    (L_ratio) but its gradient is stopped from flowing into the AR backbone,
    enforcing Decoupled Training per the FD-DRAT hypothesis.

    Stopping probability at step t:
        p_t = sigmoid(alpha * cos(q_t, k_{t-1}) - tau(z_v))

    where q_t is the hidden state after token t and k_{t-1} is the previous
    hidden state. High cosine similarity between adjacent states signals that
    the sequence has converged and generation can stop early.

    Args:
        D_v: Visual feature dimension (used for adaptive threshold shift tau).
        alpha: Scaling factor on cosine similarity.
    """

    def __init__(self, D_v: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.tau_mlp = nn.Linear(D_v, 1)

    def forward(
        self,
        q_t: torch.Tensor,
        k_prev: torch.Tensor,
        z_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q_t:   Current hidden states [B, T, D_attn], detached from AR graph.
            k_prev: Previous hidden states [B, T, D_attn], detached from AR graph.
            z_v:   Visual features [B, D_v], detached from AR graph.

        Returns:
            Stopping logits [B, T] (apply sigmoid to get probabilities).
        """
        cos_sim = F.cosine_similarity(q_t, k_prev, dim=-1)  # [B, T]
        tau_shift = self.tau_mlp(z_v)                        # [B, 1]
        return self.alpha * cos_sim - tau_shift
