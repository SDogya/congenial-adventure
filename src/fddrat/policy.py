import torch
import torch.nn as nn
from typing import Dict, Any, List

try:
    from oat.policy.base_policy import BasePolicy
except ImportError:
    class BasePolicy(nn.Module):
        pass

from src.core.config_schema import FDDRATConfig
from src.fddrat.modules.crh import ContinuousResidualHead
from src.fddrat.modules.router import ShadowRouter
from src.fddrat.modules.loss import FDDRATLoss
from src.fddrat.tokenizer import FDDRATTok

class MaskedNestedDropout(nn.Module):
    def forward(self, x, K_sampled):
        return x

class DummyEncoder(nn.Module):
    def __init__(self, d_v):
        super().__init__()
        self.d_v = d_v
    def forward(self, obs):
        if len(obs.size()) > 2:
            B = obs.size(0)
            return torch.randn(B, self.d_v, device=obs.device)
        return obs

class DummyARModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
    def forward(self, x):
        B, L, _ = x.size()
        logits = torch.randn(B, L, self.vocab_size, device=x.device)
        hidden = torch.randn(B, L, self.d_model, device=x.device)
        # Mock q_t, k_prev mappings
        return logits, hidden, hidden

class FDDRATPolicy(BasePolicy):
    def __init__(self, cfg: FDDRATConfig):
        super().__init__()
        self.cfg = cfg
        
        self.obs_encoder = DummyEncoder(cfg.D_v)
        self.action_tokenizer = FDDRATTok()
        self.ar_model = DummyARModel(cfg.D_v, 1024)
        
        self.crh = ContinuousResidualHead(H_a=cfg.H_a, D_a=cfg.D_a, D_v=cfg.D_v)
        self.router = ShadowRouter(D_v=cfg.D_v)
        self.loss_fn = FDDRATLoss(lambda_ratio=cfg.lambda_ratio, beta_mse=cfg.beta_mse)
        self.dropout = MaskedNestedDropout()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train mode standard behavior over batch elements.
        """
        # 1. Vision feature extraction
        z_v = self.obs_encoder(batch['obs'])
        B = z_v.size(0)
        
        # 2. Tokenization Mock
        latents = batch['action'] 
        tokens = batch['action'].long()  
        
        # 3. Masking behavior
        # Safe random sample K ~ U
        K_sampled = torch.randint(1, self.cfg.H_l + 1, (B,), device=z_v.device)
        latents_masked = self.dropout(latents, K_sampled)
        
        # 4. AR Model Forward
        logits, q_t, k_prev = self.ar_model(latents_masked)
        
        # 5. Router Stop Probabilities
        p_stop = self.router(q_t, k_prev, z_v)
        
        # 6. CRH Integration & Gradient Isolation
        a_coarse = self.action_tokenizer.decode_coarse(latents_masked)
        a_coarse_detached = a_coarse.detach()
        
        delta_a = self.crh(a_coarse_detached, z_v)
        residual_target = batch['action'] - a_coarse_detached
        
        targets = tokens[..., 0] 
        tau_target = torch.rand_like(p_stop)
        
        loss = self.loss_fn(
            logits=logits,
            targets=targets,
            p_stop=p_stop,
            tau_target=tau_target,
            delta_a=delta_a,
            residual_target=residual_target,
            K_sampled=K_sampled,
            H_l=self.cfg.H_l
        )
        
        return {"loss": loss}

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
        router_crh_params = list(self.router.parameters()) + list(self.crh.parameters())
        base_params = [p for n, p in self.named_parameters() if 'router' not in n and 'crh' not in n]
        
        return [
            {"params": base_params},
            {"params": router_crh_params, "weight_decay": 0.0, "lr": 1e-4}
        ]

    def compile_decoder(self):
        self.action_tokenizer.decoder = torch.compile(self.action_tokenizer.decoder)
        self.crh = torch.compile(self.crh)

    def predict_action(self, obs: torch.Tensor):
        with torch.no_grad():
            pass
