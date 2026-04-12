import torch
import torch.nn as nn

from oat.tokenizer.oat.tokenizer import OATTok

class DummyQuantizer(nn.Module):
    """Mimics FSQ(levels=[8,5,5,5]): codebook_size=1000, dim=4."""
    def __init__(self):
        super().__init__()
        self.codebook_size = 1000   # 8*5*5*5, matches real FSQ
        self.dim = 4                # len(levels), matches FSQ.dim attribute
        self.embedding_dim = self.dim   # alias used by FDDRATPolicy
    def forward(self, x): return x, torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device).unsqueeze(-1)
    def indices_to_embedding(self, x): return torch.zeros(*x.shape[:-1], self.dim, device=x.device)

class DummyDecoder(nn.Module):
    """Mimics SinglePassDecoder output shape: [B, H_a, D_a]."""
    def __init__(self, H_a: int = 16, D_a: int = 7):
        super().__init__()
        self.latent_horizon = 64    # H_l
        self.H_a = H_a
        self.D_a = D_a
    def forward(self, x, **kwargs):
        return torch.zeros(x.shape[0], self.H_a, self.D_a, device=x.device, dtype=x.dtype)

class DummyNormalizerField(nn.Module):
    def normalize(self, x): return x
    def unnormalize(self, x): return x

class DummyNormalizer(nn.ModuleDict):
    def __init__(self):
        super().__init__()
        self['action'] = DummyNormalizerField()

class FDDRATTok(OATTok):
    """
    Tokenizer tailored for FD-DRAT architecture, providing isolated access
    to coarse trajectory latent decodings.
    """

    @classmethod
    def _load_from_oat_ckpt(cls, ckpt_path: str) -> 'FDDRATTok':
        """Load a trained OAT tokenizer checkpoint and wrap it as FDDRATTok.

        Uses OAT's BaseTokenizer.from_checkpoint which handles the dill payload
        format (cfg + state_dicts) produced by OAT's workspace training script.

        Args:
            ckpt_path: Path to the .ckpt file produced by train_oattok workspace.

        Returns:
            FDDRATTok instance with real encoder/decoder/quantizer/normalizer weights.
        """
        from oat.tokenizer.base_tokenizer import BaseTokenizer
        oat_tok = BaseTokenizer.from_checkpoint(ckpt_path)
        instance = cls(
            encoder=oat_tok.encoder,
            decoder=oat_tok.decoder,
            quantizer=oat_tok.quantizer,
        )
        # Copy fitted normalizer statistics (action scale/offset from dataset)
        instance.normalizer.load_state_dict(oat_tok.normalizer.state_dict())
        return instance

    def __init__(self, *args, **kwargs):
        # Auto-inject mock components for dry-run testing (e.g. Kaggle pipeline test)
        is_mocked = False
        if len(args) == 0 and not kwargs:
            kwargs['encoder'] = nn.Identity()
            kwargs['decoder'] = DummyDecoder()
            kwargs['quantizer'] = DummyQuantizer()
            is_mocked = True
        super().__init__(*args, **kwargs)
        
        if is_mocked:
            self.normalizer = DummyNormalizer()
        
    def decode_coarse(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Bypasses final tokenization output and returns the raw, continuous
        macro-trajectory reconstructed by the underlying decoder.
        latents: [B, H_a, D_latent]
        """
        # Check against base framework
        if hasattr(self, 'decoder'):
            recons = self.decoder(latents)
            return recons
        return latents
