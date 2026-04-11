import torch
import torch.nn as nn
from typing import Any, Dict
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.core.config_schema import ExperimentConfig
from src.fddrat.policy import FDDRATPolicy

class LitSystem(pl.LightningModule):
    """
    LightningModule orchestrating the training logic and mathematics.
    """
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Initialize FDDRAT architecture
        self.model = FDDRATPolicy(cfg.model)
        
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self.model(batch)
        loss = out["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        out = self.model(batch)
        loss = out["loss"]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        # Delegate customized optimizer parameter groups from the policy model
        param_groups = self.model.get_optimizer_params()
        
        optimizer = torch.optim.AdamW(
            param_groups, 
            lr=self.cfg.learning_rate
        )
        # Cosine learning rate scheduling
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
