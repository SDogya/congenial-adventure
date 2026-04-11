import torch

try:
    from oat.tokenizer.oat.tokenizer import OATTok
except ImportError:
    class OATTok:
        pass

class FDDRATTok(OATTok):
    """
    Tokenizer tailored for FD-DRAT architecture, providing isolated access
    to coarse trajectory latent decodings.
    """
    def __init__(self, *args, **kwargs):
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
