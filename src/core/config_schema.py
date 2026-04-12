from dataclasses import dataclass, field
from typing import Any

@dataclass
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: str = "bf16"

@dataclass
class FDDRATConfig:
    lambda_ratio: float = 1.0
    beta_mse: float = 1.0
    target_ratio: float = 0.5
    H_a: int = 16
    D_a: int = 256
    D_v: int = 768
    H_l: int = 64

@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    dataset_path: str = "data/libero_10.zarr"
    shape_meta: dict = field(default_factory=lambda: {
        "obs": {
            "agentview_image": {"type": "rgb", "shape": [3, 224, 224]},
            "robot0_eye_in_hand_image": {"type": "rgb", "shape": [3, 224, 224]},
            "robot0_eef_pos": {"type": "state", "shape": [3]},
            "robot0_eef_quat": {"type": "state", "shape": [4]},
            "robot0_gripper_qpos": {"type": "state", "shape": [2]}
        },
        "action": {"shape": [7]}
    })
    strategy: FSDPConfig = field(default_factory=FSDPConfig)
    model: FDDRATConfig = field(default_factory=FDDRATConfig)
