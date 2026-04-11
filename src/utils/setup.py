import os
import torch
import pytorch_lightning as pl

def enforce_determinism(seed: int) -> None:
    """
    Enforces strict determinism for reproducible ML experiments.
    """
    # Disable stochastic heuristics in cuDNN
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Enable async error handling for collective operations (like NCCL)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Enforce deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True)
    
    # Lightning's seed_everything handles PRNGs for random, numpy, and torch
    pl.seed_everything(seed, workers=True)
