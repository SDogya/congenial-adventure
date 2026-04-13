import torch
import torch.nn as nn
from typing import Dict, Any, List

from oat.model.common.normalizer import LinearNormalizer

try:
    from oat.policy.base_policy import BasePolicy
except ImportError:
    class BasePolicy(nn.Module):
        pass

from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout

from src.core.config_schema import FDDRATConfig
from src.fddrat.modules.crh import ContinuousResidualHead
from src.fddrat.modules.router import ShadowRouter
from src.fddrat.modules.loss import FDDRATLoss
from src.fddrat.tokenizer import FDDRATTok, DummyDecoder, DummyQuantizer, DummyNormalizer


class ARModelWithHiddens(AutoregressiveModel):
    """AutoregressiveModel subclass that also returns hidden states.

    The parent class only returns logits.  FD-DRAT needs the hidden states
    to compute cosine-similarity routing signals between adjacent positions.
    """

    def forward(
        self,
        tokens: torch.LongTensor,
        cond: torch.Tensor,
    ):
        """
        Args:
            tokens: Token sequence [B, T] (includes BOS at position 0).
            cond:   Conditioning tensor [B, 1, cond_dim] from obs encoder.

        Returns:
            logits:        [B, T, vocab_size]
            hidden_states: [B, T, n_emb]
        """
        T_tok = tokens.shape[1]
        T_cond = cond.shape[1]

        tok_emb = self.tok_emb(tokens)
        tok_emb = self.drop(tok_emb + self.tok_pos_emb[:, :T_tok, :].to(tokens.device))

        cond_emb = self.cond_emb(cond)
        cond_emb = self.drop(cond_emb + self.cond_pos_emb[:, :T_cond, :].to(cond.device))
        cond_emb = self.encoder(cond_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T_tok, device=tokens.device)

        out = self.decoder(tgt=tok_emb, memory=cond_emb, tgt_mask=tgt_mask, memory_mask=None)
        hidden_states = self.ln_f(out)
        return self.head(hidden_states), hidden_states


class FDDRATPolicy(BasePolicy):
    """Feature-Decoupled Dynamic Routing Action Transformer policy.

    Architecture (training forward pass):
        1. Obs → FusedObservationEncoder → z_v [B, D_obs]
        2. Action → FDDRATTok.encode → (latents [B, H_l, 4], tokens [B, H_l])
        3. Nested Dropout: K ~ U[1, H_l]; mask latents beyond K
        4. AR transformer: [BOS, tokens] + z_v → logits + hidden_states
        5. Router (decoupled): cos-sim of adjacent hidden states → p_stop logits
        6. CRH: stop_gradient(decode_coarse(latents_masked)) + z_v → delta_a
        7. FDDRATLoss: L_CE + lambda*L_ratio + beta*L_mse(masked)

    Inference (predict_action):
        Autoregressive loop up to H_l steps with early exit when router fires.
        Remaining latent slots zero-padded to prevent CRH hallucinations.

    Args:
        cfg:        FDDRATConfig with all hyperparameters.
        shape_meta: OAT shape_meta dict; if None, obs_encoder is nn.Identity.
    """

    # Required by LiberoRunner / BasePolicy interface
    n_obs_steps: int = 2
    n_action_steps: int = 32  # must match H_a

    def __init__(self, cfg: FDDRATConfig, shape_meta: dict = None):
        super().__init__()
        self.cfg = cfg
        self.shape_meta = shape_meta  # stored for eval warm-up and get_observation_ports

        # Action tokenizer — real checkpoint or lightweight mock for dry-runs
        if cfg.tokenizer_ckpt:
            self.action_tokenizer = FDDRATTok._load_from_oat_ckpt(cfg.tokenizer_ckpt)
        else:
            self.action_tokenizer = FDDRATTok(
                encoder=nn.Identity(),
                decoder=DummyDecoder(H_a=cfg.H_a, D_a=cfg.D_a),
                quantizer=DummyQuantizer(),
            )
            self.action_tokenizer.normalizer = DummyNormalizer()

        # nn.ModuleDict ensures action normalizer stats are serialised into checkpoints.
        # OAT's LinearNormalizer subclasses nn.Module, so this is safe.
        self.normalizer = nn.ModuleDict()

        # Observation encoder
        if shape_meta is not None:
            from oat.perception.fused_obs_encoder import FusedObservationEncoder
            from omegaconf import OmegaConf
            self.obs_encoder = FusedObservationEncoder(
                shape_meta=shape_meta,
                vision_encoder=OmegaConf.create({
                    "_target_": "oat.perception.robomimic_vision_encoder.RobomimicRgbEncoder",
                    "crop_shape": [76, 76],
                }),
                state_encoder=OmegaConf.create({
                    "_target_": "oat.perception.state_encoder.ProjectionStateEncoder",
                    "out_dim": None,
                }),
            )
        else:
            self.obs_encoder = nn.Identity()

        # Vocab size and latent embedding dim from quantizer
        q = self.action_tokenizer.quantizer
        self.vocab_size = q.codebook_size + 1  # +1 for BOS
        self.embedding_dim = getattr(q, 'embedding_dim', q.dim)

        # AR model: obs features project into AR internal dim (D_v) via cond_emb
        self.ar_model = ARModelWithHiddens(
            vocab_size=self.vocab_size,
            max_seq_len=cfg.H_l + 1,  # +1 for BOS
            max_cond_len=1,
            cond_dim=cfg.obs_dim,
            n_emb=cfg.D_v,
        )

        # CRH and Router operate on the raw obs dim, not the AR internal dim
        self.crh = ContinuousResidualHead(H_a=cfg.H_a, D_a=cfg.D_a, D_v=cfg.obs_dim)
        self.router = ShadowRouter(D_v=cfg.obs_dim)
        self.loss_fn = FDDRATLoss(lambda_ratio=cfg.lambda_ratio, beta_mse=cfg.beta_mse)
        self.dropout = MaskedNestedDropout(dim=self.embedding_dim)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Inject dataset normalizer (called by LitSystem.setup after datamodule is ready).

        Registers the action normalizer as a submodule so its scale/offset
        tensors are included in the checkpoint state_dict.
        """
        if hasattr(self.obs_encoder, 'set_normalizer'):
            self.obs_encoder.set_normalizer(normalizer)
        self.normalizer.update({'action': normalizer['action']})

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward pass computing the composite FD-DRAT loss.

        Args:
            batch: Dict with keys 'obs' and 'action'.
                   'obs'    — observation dict fed to FusedObservationEncoder.
                   'action' — ground-truth actions [B, H_a, D_a].

        Returns:
            {'loss': scalar tensor}
        """
        # 1. Observation encoding → z_v [B, D_obs]
        z_v = self.obs_encoder(batch['obs'])
        if z_v.dim() == 3:
            z_v = z_v[:, -1, :]  # take last obs step; robust to To>1 unlike squeeze(1)
        B = z_v.size(0)

        # 2. Tokenization: action → latents [B, H_l, 4] + tokens [B, H_l]
        latents, tokens = self.action_tokenizer.encode(batch['action'])
        if tokens.dim() == 3 and tokens.shape[-1] == 1:
            tokens = tokens.squeeze(-1)  # normalise to [B, H_l] for both real FSQ and mock

        # 3. Nested Dropout: sample K ~ U[1, H_l] per item, mask latents beyond K
        K_sampled = torch.randint(1, self.cfg.H_l + 1, (B,), device=z_v.device)
        self.dropout.eval()
        latents_masked = self.dropout(latents.clone(), eval_keep_k=K_sampled.tolist())
        self.dropout.train()

        # 4. Autoregressive forward: [BOS, token_0..H_l-1] → logits + hidden states
        bos_id = getattr(self.action_tokenizer, 'bos_id', self.vocab_size - 1)
        bos_tokens = torch.full((B, 1), bos_id, device=tokens.device, dtype=torch.long)
        tokens_ar = torch.cat([bos_tokens, tokens], dim=1)  # [B, H_l+1]
        logits, hidden_states = self.ar_model(tokens_ar, cond=z_v.unsqueeze(1))

        # 5. Decoupled routing — detach to prevent BCE loss from entering AR backbone
        q_t = hidden_states[:, 1:, :].detach()    # hidden after each token [B, H_l, D]
        k_prev = hidden_states[:, :-1, :].detach() # hidden before each token [B, H_l, D]
        p_stop_logits = self.router(q_t, k_prev, z_v.detach())

        # 6. CRH: stop_gradient(coarse trajectory) + z_v → residual delta_a
        a_coarse_norm = self.action_tokenizer.decode_coarse(latents_masked)
        a_coarse_detached = a_coarse_norm.detach()
        delta_a_norm = self.crh(a_coarse_detached, z_v)

        # 7. Residual target in normalised space
        a_target_norm = batch['action']
        if self.normalizer:
            a_target_norm = self.normalizer['action'].normalize(a_target_norm)
        residual_target = a_target_norm - a_coarse_detached

        tau_target = torch.full_like(p_stop_logits, self.cfg.target_ratio)
        loss = self.loss_fn(
            logits=logits,
            targets=tokens,
            p_stop_logits=p_stop_logits,
            tau_target=tau_target,
            delta_a=delta_a_norm,
            residual_target=residual_target,
            K_sampled=K_sampled,
            H_l=self.cfg.H_l,
        )
        return {"loss": loss}

    def get_optimizer_params(self) -> List[Dict[str, Any]]:
        """Return parameter groups for AdamW.

        Router and CRH use lr=1e-4 with no weight decay (small heads, fast
        convergence needed).  All other parameters use the global lr and WD.

        Uses identity-based filtering (not name-based) to avoid accidentally
        misclassifying future layers whose names happen to contain 'router'/'crh'.
        """
        router_crh_ids = {id(p) for p in list(self.router.parameters()) + list(self.crh.parameters())}
        base = [p for p in self.parameters() if id(p) not in router_crh_ids]
        router_crh = [p for p in self.parameters() if id(p) in router_crh_ids]
        return [
            {"params": base},
            {"params": router_crh, "weight_decay": 0.0, "lr": 1e-4},
        ]

    def get_observation_ports(self) -> List[str]:
        """Keys the LiberoRunner will pull from its obs_dict and pass to predict_action."""
        if self.shape_meta is None:
            return []
        return list(self.shape_meta['obs'].keys())

    def get_policy_name(self) -> str:
        return "fddrat"

    def compile_decoder(self) -> None:
        """torch.compile the decoder and CRH for static CUDA-graph acceleration."""
        self.action_tokenizer.decoder = torch.compile(self.action_tokenizer.decoder)
        self.crh = torch.compile(self.crh)

    @torch.inference_mode()
    def predict_action(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Any-Time Routing inference: generate up to H_l tokens with early exit.

        Stops early when the router fires (sigmoid > 0.5 for all items in batch).
        Remaining latent slots are zero-padded (non-zero padding causes CRH
        hallucinations — the static MLP has no mechanism to ignore stale input).

        Args:
            obs: Observation tensor fed directly to obs_encoder [B, ...].

        Returns:
            {'action': a_final [B, H_a, D_a]} in unnormalised action space.
        """
        B = obs.size(0)
        z_v = self.obs_encoder(obs)
        if z_v.dim() == 3:
            z_v = z_v[:, -1, :]  # take last obs step; robust to To>1

        bos_id = getattr(self.action_tokenizer, 'bos_id', self.vocab_size - 1)
        tokens_in = torch.full((B, 1), bos_id, device=obs.device, dtype=torch.long)
        cond_input = z_v.unsqueeze(1)

        # BOS latent slot
        latents = torch.zeros(B, 1, self.embedding_dim, device=obs.device)
        if hasattr(self.action_tokenizer, 'bos_id_emb'):
            latents = self.action_tokenizer.bos_id_emb.expand(B, 1, -1)

        tokens_generated = []
        # Note: each AR step re-runs the full prefix (O(H_l²) attention).
        # With H_l=8 this is negligible (64 ops). If H_l grows, add KV-cache
        # to ARModelWithHiddens to make inference O(H_l) instead.
        for t in range(self.cfg.H_l):
            logits_step, hidden_states = self.ar_model(tokens_in, cond=cond_input)

            next_token = torch.argmax(logits_step[:, -1, :], dim=-1)  # [B]
            tokens_generated.append(next_token)
            tokens_in = torch.cat([tokens_in, next_token.unsqueeze(1)], dim=1)

            # Convert token index to latent embedding for decoder input
            if hasattr(self.action_tokenizer, 'quantizer'):
                next_latent = self.action_tokenizer.quantizer.indices_to_embedding(
                    next_token.unsqueeze(-1)
                )
            else:
                next_latent = torch.zeros(B, 1, self.embedding_dim, device=obs.device)
            latents = torch.cat([latents, next_latent], dim=1)

            # Early exit check (skip t=0 — only BOS in context, signal not meaningful)
            if t > 0:
                q_t = hidden_states[:, -1:, :]
                k_prev = hidden_states[:, -2:-1, :]
                if (torch.sigmoid(self.router(q_t, k_prev, z_v)) > 0.5).all():
                    break

        # Zero-pad latents if stopped early (non-zero padding → CRH hallucinations)
        pad_size = self.cfg.H_l - len(tokens_generated)
        if pad_size > 0:
            latents = torch.cat([
                latents,
                torch.zeros(B, pad_size, self.embedding_dim, device=obs.device, dtype=latents.dtype),
            ], dim=1)

        # Strip BOS slot, decode, refine
        latents_filtered = latents[:, 1:, :]
        a_coarse_norm = self.action_tokenizer.decode_coarse(latents_filtered)
        a_final_norm = a_coarse_norm + self.crh(a_coarse_norm, z_v)

        # Denormalise — policy normalizer takes priority over tokenizer's
        if self.normalizer:
            a_final = self.normalizer['action'].unnormalize(a_final_norm)
        elif hasattr(self.action_tokenizer, 'normalizer') and \
                'action' in self.action_tokenizer.normalizer:
            a_final = self.action_tokenizer.normalizer['action'].unnormalize(a_final_norm)
        else:
            a_final = a_final_norm

        return {"action": a_final[:, :self.cfg.H_a]}
