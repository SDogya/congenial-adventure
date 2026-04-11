import torch
import torch.nn as nn

from oat.tokenizer.oat.tokenizer import OATTok

class DummyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook_size = 1024
        self.embedding_dim = 256
    def forward(self, x): return x, torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device).unsqueeze(-1)
    def indices_to_embedding(self, x): return torch.zeros(*x.shape[:-1], self.embedding_dim, device=x.device)

class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_horizon = 64
    def forward(self, x, **kwargs): return x

class FDDRATTok(OATTok):
    """
    Tokenizer tailored for FD-DRAT architecture, providing isolated access
    to coarse trajectory latent decodings.
    """
    def __init__(self, *args, **kwargs):
        # Auto-inject mock components for dry-run testing (e.g. Kaggle pipeline test)
        if len(args) == 0 and not kwargs:
            kwargs['encoder'] = nn.Identity()
            kwargs['decoder'] = DummyDecoder()
            kwargs['quantizer'] = DummyQuantizer()
        super().__init__(*args, **kwargs)
        
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
