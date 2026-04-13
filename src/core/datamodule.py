import copy
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.core.config_schema import ExperimentConfig
from oat.dataset.zarr_dataset import ZarrDataset
from oat.common.replay_buffer import ReplayBuffer
from oat.model.common.normalizer import LinearNormalizer


class LazyZarrDataset(ZarrDataset):
    """ZarrDataset that opens the zarr store lazily from disk instead of RAM.

    The default ``ReplayBuffer.copy_from_path`` loads the entire dataset into
    a MemoryStore (~25 GB for LIBERO10 with 128×128 images), which kills the
    Kaggle kernel.  This subclass monkey-patches it to use
    ``create_from_path(mode='r')`` so each sample is fetched on demand.

    Also overrides ``get_normalizer`` to skip RGB keys (the vision encoder
    normalises images internally with ÷255) and ``_sample_to_data`` to cast
    uint8 RGB arrays to float32 required by CUDA conv2d.
    """

    def __init__(self, zarr_path: str, **kwargs):
        orig = ReplayBuffer.__dict__['copy_from_path']

        @classmethod  # type: ignore[misc]
        def _lazy(cls, path, keys=None, **kw):
            return cls.create_from_path(path, mode='r')

        ReplayBuffer.copy_from_path = _lazy
        try:
            super().__init__(zarr_path, **kwargs)
        finally:
            ReplayBuffer.copy_from_path = orig  # always restore, even on error

    def _sample_to_data(self, sample):
        """Cast uint8 RGB observations to float32 before collation."""
        data = super()._sample_to_data(sample)
        for k in self.numeric_obs_keys:
            if 'rgb' in k or 'image' in k:
                data['obs'][k] = data['obs'][k].astype('float32')
        return data

    def get_normalizer(self, mode: str = 'limits', **kwargs) -> LinearNormalizer:
        """Fit normalizer on action + non-RGB state keys only.

        RGB keys are excluded because the vision encoder normalises them
        internally (÷255), so including them would double-normalise.
        """
        non_rgb = [k for k in self.numeric_obs_keys if 'rgb' not in k and 'image' not in k]
        data = {
            'action': self.replay_buffer[self.action_key],
            **{k: self.replay_buffer[k] for k in non_rgb},
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_validation_dataset(self) -> 'LazyZarrDataset':
        """Return a shallow copy using the held-out episode mask."""
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
    """LightningDataModule wrapping LazyZarrDataset for LIBERO training.

    Exposes ``self.normalizer`` after ``setup()`` so that ``run.py`` can
    inject it into the policy before training starts.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer: LinearNormalizer = None

    def setup(self, stage: str = None) -> None:
        obs_keys = list(self.cfg.shape_meta["obs"].keys())
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
