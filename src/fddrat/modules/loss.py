import torch
import torch.nn as nn
import torch.nn.functional as F


class FDDRATLoss(nn.Module):
    """Composite training objective for FD-DRAT.

    L_total = L_CE + lambda * L_ratio + beta * L_mse (masked)

    Components:
        L_CE:    Cross-entropy on AR token predictions (main autoregressive loss).
        L_ratio: BCE loss training the router to predict the target stopping ratio
                 tau_target (Decoupled Training — gradients stopped at hidden states).
        L_mse:   MSE between CRH residual output and the true residual target,
                 masked to zero when K_sampled == H_l (full sequence used,
                 no residual needed).

    Args:
        lambda_ratio: Weight for the router BCE loss.
        beta_mse:     Weight for the masked residual MSE loss.
    """

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
        H_l: int,
    ) -> torch.Tensor:
        """
        Args:
            logits:         AR logits [B, H_l+1, vocab_size].
            targets:        Token targets [B, H_l].
            p_stop_logits:  Router stopping logits [B, H_l].
            tau_target:     Router BCE targets [B, H_l], filled with target_ratio.
            delta_a:        CRH residual output [B, H_a, D_a].
            residual_target: True residual = a_target_norm - a_coarse_norm [B, H_a, D_a].
            K_sampled:      Per-sample dropout lengths [B].
            H_l:            Latent sequence length (OAT num_registers).

        Returns:
            Scalar total loss.
        """
        # Cross-entropy: shift logits left so position t predicts token t
        logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
        loss_ce = F.cross_entropy(logits_flat, targets.reshape(-1), ignore_index=-1)

        # Router BCE — forced to float32 even under bf16 AMP (numerically unstable otherwise)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            loss_ratio = F.binary_cross_entropy_with_logits(
                p_stop_logits.view(-1).float(),
                tau_target.view(-1).float(),
            )

        # Masked MSE: skip penalty when full sequence was used (K == H_l)
        mse_per_sample = F.mse_loss(delta_a, residual_target, reduction='none').mean(dim=[1, 2])
        mask = (K_sampled < H_l).float()
        loss_mse = (mse_per_sample * mask).sum() / (mask.sum() + 1e-8)

        return loss_ce + self.lambda_ratio * loss_ratio + self.beta_mse * loss_mse
