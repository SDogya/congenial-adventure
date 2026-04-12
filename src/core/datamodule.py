import copy
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.core.config_schema import ExperimentConfig

from oat.dataset.zarr_dataset import ZarrDataset
from oat.common.replay_buffer import ReplayBuffer
from oat.model.common.normalizer import LinearNormalizer


class LazyZarrDataset(ZarrDataset):
    """
    Drop-in replacement for ZarrDataset that opens the zarr lazily from disk
    instead of copying the entire dataset into RAM.

    ReplayBuffer.copy_from_path loads everything to a MemoryStore (~25 GB for
    LIBERO10 with 128×128 images).  ReplayBuffer.create_from_path opens the
    on-disk zarr directly so each sample is fetched from disk on demand.

    Also overrides get_normalizer() to skip RGB keys: the vision encoder
    normalises images internally (÷255), so we only need action + state stats.
    """
    def __init__(self, zarr_path: str, **kwargs):
        # Temporarily swap copy_from_path → create_from_path for this one call.
        # Using __dict__ + classmethod replacement to avoid touching OAT files.
        _orig = ReplayBuffer.__dict__['copy_from_path']

        @classmethod  # type: ignore[misc]
        def _lazy(cls, path, keys=None, **kw):
            return cls.create_from_path(path, mode='r')

        ReplayBuffer.copy_from_path = _lazy
        try:
            super().__init__(zarr_path, **kwargs)
        finally:
            ReplayBuffer.copy_from_path = _orig  # always restore

    def _sample_to_data(self, sample):
        data = super()._sample_to_data(sample)
        # ZarrDataset keeps uint8 arrays as-is; CUDA conv2d requires float.
        # Cast every rgb obs key to float32 here (before collation into tensors).
        for k in self.numeric_obs_keys:
            if 'rgb' in k or 'image' in k:
                data['obs'][k] = data['obs'][k].astype('float32')
        return data

    def get_normalizer(self, mode='limits', **kwargs):
        """Fit normalizer only on action + non-RGB state keys."""
        non_rgb = [k for k in self.numeric_obs_keys if 'rgb' not in k and 'image' not in k]
        data = {
            'action': self.replay_buffer[self.action_key],
            **{k: self.replay_buffer[k] for k in non_rgb},
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_validation_dataset(self):
        # copy() must preserve the lazy replay_buffer reference
        val = copy.copy(self)
        from oat.common.seq_sampler import SequenceSampler
        val.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val.train_mask = ~self.train_mask
        return val


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
        self.train_dataset = LazyZarrDataset(
            zarr_path=self.cfg.dataset_path,
            obs_keys=obs_keys,
            action_key='action',
            n_obs_steps=1,
            n_action_steps=self.cfg.model.H_a,
            val_ratio=0.1,
        )
        self.val_dataset = self.train_dataset.get_validation_dataset()
        self.normalizer = self.train_dataset.get_normalizer()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
