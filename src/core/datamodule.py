import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from src.core.config_schema import ExperimentConfig

class DummyDataset(Dataset):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
    def __len__(self):
        return 100
    def __getitem__(self, idx):
        # Mock tensors
        obs = torch.randn(self.cfg.model.D_v)
        action = torch.randn(self.cfg.model.H_a, self.cfg.model.D_a).clamp(-1, 1)
        return {"obs": obs, "action": action}

class LitDataModule(pl.LightningDataModule):
    """
    LightningDataModule orchestrating I/O operations.
    """
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = None) -> None:
        self.train_dataset = DummyDataset(self.cfg)
        self.val_dataset = DummyDataset(self.cfg)

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
