from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FSDPConfig:
    """Fully Sharded Data Parallel strategy configuration.

    Set use_fsdp=False for single-GPU training (e.g. Kaggle T4).
    """
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"
    use_fsdp: bool = True


@dataclass
class FDDRATConfig:
    """Hyperparameters for the FD-DRAT policy.

    Dimension invariants (must match the trained OAT tokenizer):
        H_l:  Latent sequence length = OAT encoder num_registers (default 8).
        H_a:  Action horizon = OAT decoder sample_horizon (default 32).
        D_a:  Action dimension = 7 for LIBERO10.
        D_v:  AR transformer internal embedding dim; must be divisible by
              num_heads=12, so default 768 (768/12=64).
        obs_dim: Actual output dim of FusedObservationEncoder for LIBERO10.
                 Measured empirically: 2 RGB cameras + 10 state dims = 138.
    """
    lambda_ratio: float = 1.0
    beta_mse: float = 1.0
    target_ratio: float = 0.5
    H_a: int = 32        # OAT decoder sample_horizon (train_oattok.yaml: horizon=32)
    D_a: int = 7
    obs_dim: int = 138   # FusedObsEncoder output for LIBERO10 (2×RGB + 10 state)
    D_v: int = 768
    H_l: int = 8         # OAT encoder num_registers = latent sequence length
    tokenizer_ckpt: Optional[str] = None  # path to .ckpt; None → mock/dry-run


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration composed by Hydra."""
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    dataset_path: str = "data/libero/libero10_N500.zarr"
    shape_meta: dict = field(default_factory=lambda: {
        "obs": {
            # Shape format is (H, W, C) — OAT's RobomimicRgbEncoder expects channels-last
            "agentview_rgb":          {"type": "rgb",   "shape": [128, 128, 3]},
            "robot0_eye_in_hand_rgb": {"type": "rgb",   "shape": [128, 128, 3]},
            "robot0_eef_pos":         {"type": "state", "shape": [3]},
            "robot0_eef_quat":        {"type": "state", "shape": [4]},
            "robot0_gripper_qpos":    {"type": "state", "shape": [2]},
            "task_uid":               {"type": "state", "shape": [1]},
        },
        "action": {"shape": [7]},
    })
    strategy: FSDPConfig = field(default_factory=FSDPConfig)
    model: FDDRATConfig = field(default_factory=FDDRATConfig)
