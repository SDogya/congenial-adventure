import torch
import torch.nn as nn
from typing import Any, Dict
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.core.config_schema import ExperimentConfig
from src.fddrat.policy import FDDRATPolicy


class LitSystem(pl.LightningModule):
    """LightningModule orchestrating FD-DRAT training.

    Delegates optimizer parameter groups to the policy so that the router and
    CRH can use a separate learning rate and disable weight decay (they are
    small heads that benefit from faster, unregularised updates).

    Normalizer injection:
        ``set_normalizer`` is called from ``setup()`` via the trainer's
        datamodule reference.  It must happen before the first forward pass
        so action normalisation statistics are available.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = FDDRATPolicy(cfg.model, shape_meta=cfg.shape_meta)

    def setup(self, stage: str = None) -> None:
        """Inject dataset normalizer into the policy after datamodule is ready."""
        if self.trainer and getattr(self.trainer, 'datamodule', None):
            self.model.set_normalizer(self.trainer.datamodule.normalizer)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss = self.model(batch)["loss"]
        self.log("train_loss", loss, batch_size=batch['action'].size(0),
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx % 50 == 0:
            try:
                from IPython.display import clear_output
                clear_output(wait=True)
            except Exception:
                pass
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss = self.model(batch)["loss"]
        self.log("val_loss", loss, batch_size=batch['action'].size(0),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        """AdamW with cosine annealing; router/CRH use separate lr=1e-4, no WD.

        T_max is set to the total number of optimizer steps computed by Lightning
        from (num_epochs * batches_per_epoch / accumulate_grad_batches).
        Hard-coding T_max would cause LR to hit zero at the wrong time.
        """
        optimizer = torch.optim.AdamW(
            self.model.get_optimizer_params(),
            lr=self.cfg.learning_rate,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
