import os
import sys

# Заглушаем rank 1 — чтобы прогресс-бар не дублировался
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
if _local_rank != 0:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

sys.path.insert(0, os.path.abspath('oat'))
sys.path.insert(0, os.path.abspath('hnet'))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import wandb
import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy
from omegaconf import OmegaConf

from src.core.config_schema import ExperimentConfig
from src.core.system import LitSystem
from src.core.datamodule import LitDataModule
from src.utils.setup import enforce_determinism

class WandbCheckpointUploader(pl.Callback):
    """Upload the best checkpoint to W&B as an artifact after each epoch.

    Uploads only when rank 0 has an actual checkpoint file on disk.
    Safe to use with DDP/FSDP — other ranks skip silently.
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank != 0:
            return
        ckpt_cb = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if ckpt_cb is None:
            return
        ckpt_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
        if not ckpt_path or not os.path.exists(ckpt_path):
            return
        artifact = wandb.Artifact(
            name=f"fddrat-epoch-{trainer.current_epoch:03d}",
            type="model",
            metadata={"epoch": trainer.current_epoch, "val_loss": trainer.callback_metrics.get("val_loss", -1)},
        )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)


# Register the configuration schema
cs = ConfigStore.instance()
cs.store(name="experiment_config", node=ExperimentConfig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: ExperimentConfig) -> None:
    # 1. Enforce strict reproducibility
    enforce_determinism(cfg.seed)

    # 2. Setup Data & System
    datamodule = LitDataModule(cfg)
    system = LitSystem(cfg)

    # 3. Setup W&B Logger
    logger = WandbLogger(
        project="VLA-experiment",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 4. Setup robust Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), "checkpoints"),
        filename="epoch_{epoch:03d}-val_loss_{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_weights_only=True
    )

    # 5. Trainer Initialization
    if cfg.strategy.use_fsdp:
        strategy = FSDPStrategy(sharding_strategy=cfg.strategy.sharding_strategy)
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, WandbCheckpointUploader()],
        max_epochs=10,
        strategy=strategy,
        precision=cfg.strategy.mixed_precision,
        sync_batchnorm=cfg.strategy.use_fsdp,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,  # kills notebook OOM from tqdm spam; use W&B instead
        log_every_n_steps=10,
    )

    # 6. Explicit Initialization
    datamodule.setup()
    system.model.set_normalizer(datamodule.normalizer)

    # 7. Execute Training
    trainer.fit(model=system, datamodule=datamodule)


if __name__ == "__main__":
    main()