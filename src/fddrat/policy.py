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
        self.blocks = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        B, L, _ = x.size()
        hidden = torch.randn(B, L, self.d_model, device=x.device)
        return self.head(self.blocks[-1](hidden))

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
        
        # Hook for capturing hidden states
        self._hooked_hidden = None
        def hook_fn(module, inp, out):
            self._hooked_hidden = out
            
        if hasattr(self.ar_model, 'blocks'):
            self.ar_model.blocks[-1].register_forward_hook(hook_fn)
        else:
            self.ar_model.register_forward_hook(hook_fn)
        
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
        
        # 3.5. BOS padding for AR models
        bos_emb = torch.zeros(B, 1, latents_masked.size(-1), device=latents_masked.device)
        latents_ar = torch.cat([bos_emb, latents_masked], dim=1) # [B, H_l+1, D_lat]
        
        # 4. AR Model Forward
        self._hooked_hidden = None
        logits = self.ar_model(latents_ar) # [B, H_l+1, Vocab]
        
        # Retrieve mapped hidden
        if self._hooked_hidden is None:
            self._hooked_hidden = torch.randn(B, self.cfg.H_l + 1, self.cfg.D_v, device=z_v.device)
            
        hidden_states = self._hooked_hidden
        
        # 5. Decoupled Routing Slicing
        q_t = hidden_states[:, 1:, :] # [B, H_l, D]
        k_prev = hidden_states[:, :-1, :] # [B, H_l, D]
        
        p_stop_logits = self.router(q_t, k_prev, z_v)
        
        # 6. CRH Integration & Denormalization
        a_coarse_norm = self.action_tokenizer.decode_coarse(latents_masked)
        
        if hasattr(self.action_tokenizer, 'normalizer') and 'action' in self.action_tokenizer.normalizer:
            a_coarse_denorm = self.action_tokenizer.normalizer['action'].unnormalize(a_coarse_norm)
        else:
            a_coarse_denorm = a_coarse_norm

        a_coarse_detached = a_coarse_denorm.detach()
        
        delta_a = self.crh(a_coarse_detached, z_v)
        residual_target = batch['action'] - a_coarse_detached
        
        targets = tokens[..., 0] 
        tau_target = torch.rand_like(p_stop_logits)
        
        loss = self.loss_fn(
            logits=logits,
            targets=targets,
            p_stop_logits=p_stop_logits,
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
