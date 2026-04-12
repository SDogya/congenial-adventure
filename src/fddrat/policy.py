import torch
import torch.nn as nn
from typing import Dict, Any, List

try:
    from oat.policy.base_policy import BasePolicy
except ImportError:
    class BasePolicy(nn.Module):
        pass

# Add real OAT dependencies per instructions
from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout

from src.core.config_schema import FDDRATConfig
from src.fddrat.modules.crh import ContinuousResidualHead
from src.fddrat.modules.router import ShadowRouter
from src.fddrat.modules.loss import FDDRATLoss
from src.fddrat.tokenizer import FDDRATTok

class ARModelWithHiddens(AutoregressiveModel):
    def forward(self, tokens: torch.LongTensor, cond: torch.Tensor):
        T_tok = tokens.shape[1]
        T_cond = cond.shape[1]
        
        tok_emb = self.tok_emb(tokens)
        tok_emb = self.drop(tok_emb + self.tok_pos_emb[:, :T_tok, :].to(tokens.device))

        cond_emb = self.cond_emb(cond)
        cond_emb = self.drop(cond_emb + self.cond_pos_emb[:, :T_cond, :].to(cond.device))
        cond_emb = self.encoder(cond_emb)

        tgt_mask = (torch.triu(torch.ones(T_tok, T_tok, device=tokens.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        out = self.decoder(
            tgt=tok_emb,
            memory=cond_emb,
            tgt_mask=tgt_mask,
            memory_mask=None,
        )
        hidden_states = self.ln_f(out)
        logits = self.head(hidden_states)
        return logits, hidden_states

class FDDRATPolicy(BasePolicy):
    def __init__(self, cfg: FDDRATConfig, shape_meta: dict = None):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta
        self.action_tokenizer = FDDRATTok()
        self.normalizer = None
        
        # 1. Observation Flow (Multimodal Fused Encoder)
        if shape_meta is not None:
            from oat.perception.fused_obs_encoder import FusedObservationEncoder
            from omegaconf import OmegaConf
            
            vision_dict = {"_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder", "crop_shape": [76, 76]}
            state_dict = {"_target_": "oat.perception.state_encoder.ProjectionStateEncoder", "out_dim": None}
            
            self.obs_encoder = FusedObservationEncoder(
                shape_meta=shape_meta,
                vision_encoder=OmegaConf.create(vision_dict),
                state_encoder=OmegaConf.create(state_dict)
            )
        else:
            self.obs_encoder = nn.Identity()
        
        # Vocab size logic parsed from quantizer
        if hasattr(self.action_tokenizer, 'quantizer'):
            self.vocab_size = self.action_tokenizer.quantizer.codebook_size + 1 # Account for BOS
            self.embedding_dim = self.action_tokenizer.quantizer.embedding_dim
        else:
            self.vocab_size = 1025 # Safe fallback if quantizer mock fails structurally
            self.embedding_dim = 256
            
        self.ar_model = ARModelWithHiddens(
            vocab_size=self.vocab_size,
            max_seq_len=cfg.H_l + 1,
            max_cond_len=1,
            cond_dim=cfg.D_v,
            n_emb=cfg.D_v
        )
        
        self.crh = ContinuousResidualHead(H_a=cfg.H_a, D_a=cfg.D_a, D_v=cfg.D_v)
        self.router = ShadowRouter(D_v=cfg.D_v)
        self.loss_fn = FDDRATLoss(lambda_ratio=cfg.lambda_ratio, beta_mse=cfg.beta_mse)
        self.dropout = MaskedNestedDropout(dim=self.embedding_dim)
        
        # Removed hook mechanisms to avoid FSDP sync deadlock
        
    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        self.normalizer = normalizer
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Train mode forward standard behavior over batch elements.
        """
        # 1. Vision feature extraction
        z_v = self.obs_encoder(batch['obs'])
        B = z_v.size(0)
        
        # 2. Tokenization via real encode
        latents, tokens = self.action_tokenizer.encode(batch['action'])
        
        # 3. Masking behavior
        K_sampled = torch.randint(1, self.cfg.H_l + 1, (B,), device=z_v.device)
        self.dropout.eval()
        latents_masked = self.dropout(latents, eval_keep_k=K_sampled.tolist())
        self.dropout.train()
        
        # OAT AR model expects discrete tokens not continuous latents
        # Apply dropout logic to tokens - for simplicity if tokenizer dropout isn't natively masking tokens:
        tokens_masked = tokens[..., 0].clone() # [B, H_l]
        
        # Create BOS token tensor 
        bos_id = getattr(self.action_tokenizer, 'bos_id', self.vocab_size - 1)
        bos_tokens = torch.full((B, 1), bos_id, device=tokens_masked.device, dtype=torch.long)
        tokens_ar = torch.cat([bos_tokens, tokens_masked], dim=1) # [B, H_l+1]
        
        # 4. AR Model Forward
        cond_input = z_v.unsqueeze(1)
        logits, hidden_states = self.ar_model(tokens_ar, cond=cond_input)
        
        # 5. Decoupled Routing Slicing
        q_t = hidden_states[:, 1:, :] # [B, H_l, D]
        k_prev = hidden_states[:, :-1, :] # [B, H_l, D]
        
        p_stop_logits = self.router(q_t, k_prev, z_v)
        
        # 6. CRH Integration & Denormalization
        a_coarse_norm = self.action_tokenizer.decode_coarse(latents_masked)
        a_coarse_norm_detached = a_coarse_norm.detach()
        
        delta_a_norm = self.crh(a_coarse_norm_detached, z_v)
        
        a_target_norm = batch['action']
        if self.normalizer is not None:
            a_target_norm = self.normalizer['action'].normalize(a_target_norm)
        
        residual_target = a_target_norm - a_coarse_norm_detached        
        targets = tokens_masked
        target_ratio = getattr(self.cfg, 'target_ratio', 0.5)
        tau_target = torch.full_like(p_stop_logits, target_ratio)
        
        loss = self.loss_fn(
            logits=logits,
            targets=targets,
            p_stop_logits=p_stop_logits,
            tau_target=tau_target,
            delta_a=delta_a_norm,
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

    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Any-Time Routing Inference sequence.
        """
        with torch.no_grad():
            B = obs.size(0)
            z_v = self.obs_encoder(obs)
            
            # Start with BOS 
            bos_id = getattr(self.action_tokenizer, 'bos_id', self.vocab_size - 1)
            tokens_in = torch.full((B, 1), bos_id, device=obs.device, dtype=torch.long)
            
            # Latents sequence for evaluation track
            latents = torch.zeros(B, 1, self.embedding_dim, device=obs.device)
            if hasattr(self.action_tokenizer, 'bos_id_emb'):
                 latents = self.action_tokenizer.bos_id_emb.expand(B, 1, -1)
            
            tokens_generated = []
            
            # Autoregressive generation boundary mapping
            threshold = 0.5 
            
            cond_input = z_v.unsqueeze(1)
            for t in range(self.cfg.H_l):
                logits_step, hidden_states = self.ar_model(tokens_in, cond=cond_input)
                
                # Take argmax at current step
                next_token = torch.argmax(logits_step[:, -1, :], dim=-1)
                tokens_generated.append(next_token)
                tokens_in = torch.cat([tokens_in, next_token.unsqueeze(1)], dim=1)
                
                # Convert token index back to latent embedding
                if hasattr(self.action_tokenizer, 'quantizer'):
                    next_latent = self.action_tokenizer.quantizer.indices_to_embedding(next_token.unsqueeze(-1))
                else:
                    next_latent = torch.zeros(B, 1, self.embedding_dim, device=obs.device)
                    
                latents = torch.cat([latents, next_latent], dim=1)
                
                # Early Exit Check on steps t > 0
                if t > 0:
                    q_t = hidden_states[:, -1:, :]
                    k_prev = hidden_states[:, -2:-1, :]
                    
                    p_stop_logit = self.router(q_t, k_prev, z_v)
                    p_stop_prob = torch.sigmoid(p_stop_logit)
                    
                    if (p_stop_prob > threshold).all():
                         break
            
            # Pad early terminated execution up to H_l
            curr_len = len(tokens_generated)
            if curr_len < self.cfg.H_l:
                 pad_size = self.cfg.H_l - curr_len
                 pad_id = getattr(self.cfg, 'mask_id', 0)
                 pad_tokens = torch.full((B, pad_size), pad_id, device=obs.device)
                 for i in range(pad_size):
                     tokens_generated.append(pad_tokens[:, i])
                 
                 # Assemble full padded tensor
                 if hasattr(self.action_tokenizer, 'quantizer'):
                     padded_latents = self.action_tokenizer.quantizer.indices_to_embedding(pad_tokens.unsqueeze(-1))
                 else:
                     padded_latents = torch.zeros(B, pad_size, self.embedding_dim, device=obs.device)
                 latents = torch.cat([latents, padded_latents], dim=1)
            
            # Strip BOS to extract true token span
            latents_filtered = latents[:, 1:, :]
            
            # 1. Decode coarse trajectory
            a_coarse_norm = self.action_tokenizer.decode_coarse(latents_filtered)
            
            # 2. Denormalize explicitly  
            if hasattr(self.action_tokenizer, 'normalizer') and 'action' in self.action_tokenizer.normalizer:
                 a_coarse = self.action_tokenizer.normalizer['action'].unnormalize(a_coarse_norm)
            else:
                 a_coarse = a_coarse_norm
                 
            # 3. CRH refinement step
            delta_a = self.crh(a_coarse, z_v)
            
            # Action Output Contract Match
            a_final = a_coarse + delta_a
            
            # Safe slice fallback to H_a if n_action_steps missing
            n_slice = getattr(self.cfg, 'n_action_steps', getattr(self.cfg, 'H_a', 16))
            return {"action": a_final[:, :n_slice]}
