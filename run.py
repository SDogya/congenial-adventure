import os
import sys
sys.path.insert(0, os.path.abspath('oat'))
sys.path.insert(0, os.path.abspath('hnet'))

import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from omegaconf import OmegaConf

from src.core.config_schema import ExperimentConfig
from src.core.system import LitSystem
from src.core.datamodule import LitDataModule
from src.utils.setup import enforce_determinism

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
        # FSDP Checkpointing optimization, ensures safe model save
        save_weights_only=True
    )
    
    # 5. Trainer Initialization
    fsdp_strategy = FSDPStrategy(sharding_strategy=cfg.strategy.sharding_strategy)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=10,
        strategy=fsdp_strategy,
        precision=cfg.strategy.mixed_precision,
        sync_batchnorm=True,
        accelerator="auto",
        devices="auto"
    )
    
    # 6. Explicit Initialization (Tech Lead Checkpoint)
    datamodule.setup()
    system.model.set_normalizer(datamodule.normalizer)
    
    # 7. Execute Training
    trainer.fit(model=system, datamodule=datamodule)

if __name__ == "__main__":
    main()
