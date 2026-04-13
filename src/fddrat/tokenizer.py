import torch
import torch.nn as nn

from oat.tokenizer.oat.tokenizer import OATTok


class DummyQuantizer(nn.Module):
    """Drop-in mock for FSQ(levels=[8,5,5,5]).

    Matches the real FSQ interface used by FDDRATPolicy:
        codebook_size = 8*5*5*5 = 1000
        dim / embedding_dim  = len(levels) = 4
        forward(x) -> (x, indices)   indices: [B, H_l] long
        indices_to_embedding(idx) -> zeros of shape [*idx.shape[:-1], dim]
    """

    def __init__(self):
        super().__init__()
        self.codebook_size = 1000
        self.dim = 4
        self.embedding_dim = self.dim  # alias used by FDDRATPolicy

    def forward(self, x: torch.Tensor):
        return x, torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)

    def indices_to_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*x.shape[:-1], self.dim, device=x.device)


class DummyDecoder(nn.Module):
    """Drop-in mock for SinglePassDecoder.

    Returns zeros with the correct output shape [B, H_a, D_a].

    Args:
        H_a: Action horizon (OAT sample_horizon, default 32).
        D_a: Action dimension (e.g. 7 for LIBERO).
    """

    def __init__(self, H_a: int = 32, D_a: int = 7):
        super().__init__()
        self.latent_horizon = 8  # OAT num_registers
        self.H_a = H_a
        self.D_a = D_a

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.H_a, self.D_a, device=x.device, dtype=x.dtype)


class DummyNormalizerField(nn.Module):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyNormalizer(nn.ModuleDict):
    """Passthrough normalizer used during mock/dry-run mode."""

    def __init__(self):
        super().__init__()
        self['action'] = DummyNormalizerField()


class FDDRATTok(OATTok):
    """OAT tokenizer wrapper for FD-DRAT.

    Adds:
        - ``decode_coarse(latents)``: bypasses the quantizer and directly
          calls the decoder, returning the continuous macro-trajectory.
        - ``_load_from_oat_ckpt(path)``: classmethod to load a trained OAT
          checkpoint and wrap it as FDDRATTok.
        - Zero-argument construction: auto-injects mock components for
          dry-run testing (no checkpoint needed).
    """

    @classmethod
    def _load_from_oat_ckpt(cls, ckpt_path: str) -> 'FDDRATTok':
        """Load a trained OAT tokenizer checkpoint and wrap it as FDDRATTok.

        Args:
            ckpt_path: Path to the .ckpt file produced by train_oattok workspace.

        Returns:
            FDDRATTok with real encoder/decoder/quantizer/normalizer weights.
        """
        from oat.tokenizer.base_tokenizer import BaseTokenizer
        oat_tok = BaseTokenizer.from_checkpoint(ckpt_path)
        instance = cls(
            encoder=oat_tok.encoder,
            decoder=oat_tok.decoder,
            quantizer=oat_tok.quantizer,
        )
        instance.normalizer.load_state_dict(oat_tok.normalizer.state_dict())
        return instance

    def __init__(self, *args, **kwargs):
        is_mocked = not args and not kwargs
        if is_mocked:
            kwargs['encoder'] = nn.Identity()
            kwargs['decoder'] = DummyDecoder()
            kwargs['quantizer'] = DummyQuantizer()
        super().__init__(*args, **kwargs)
        if is_mocked:
            self.normalizer = DummyNormalizer()

    def decode_coarse(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to a continuous macro-trajectory without quantization.

        OATTok always sets self.decoder in __init__, so this is a direct call.

        Args:
            latents: Latent tensor [B, H_l, latent_dim].

        Returns:
            Coarse trajectory [B, H_a, D_a] in normalized action space.
        """
        return self.decoder(latents)
