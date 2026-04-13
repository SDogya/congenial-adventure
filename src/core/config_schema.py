from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"
    use_fsdp: bool = True   # False → single-GPU/DDP (e.g. Kaggle T4)

@dataclass
class FDDRATConfig:
    lambda_ratio: float = 1.0
    beta_mse: float = 1.0
    target_ratio: float = 0.5
    H_a: int = 16
    D_a: int = 7    # action dimension (matches shape_meta action.shape[0])
    obs_dim: int = 250
    D_v: int = 768
    H_l: int = 64
    tokenizer_ckpt: Optional[str] = None   # path to OAT tokenizer .ckpt; None → mock/dry-run

@dataclass
class ExperimentConfig:
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
        "action": {"shape": [7]}
    })
    strategy: FSDPConfig = field(default_factory=FSDPConfig)
    model: FDDRATConfig = field(default_factory=FDDRATConfig)
