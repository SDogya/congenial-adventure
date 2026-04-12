import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from src.core.config_schema import ExperimentConfig

from oat.dataset.zarr_dataset import ZarrDataset

class LitDataModule(pl.LightningDataModule):
    """
    LightningDataModule orchestrating I/O operations with ZarrDataset.
    """
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer = None

    def setup(self, stage: str = None) -> None:
        obs_keys = [k for k in self.cfg.shape_meta["obs"].keys()]
        self.train_dataset = ZarrDataset(
            zarr_path=self.cfg.dataset_path,
            obs_keys=obs_keys,
            action_key='action',
            n_obs_steps=1,
            n_action_steps=self.cfg.model.H_a,
            val_ratio=0.1
        )
        self.val_dataset = self.train_dataset.get_validation_dataset()
        self.normalizer = self.train_dataset.get_normalizer()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=4
        )
