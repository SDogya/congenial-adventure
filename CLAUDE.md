# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup** (external deps must be cloned first):
```bash
git clone https://github.com/goombalab/hnet.git
git clone --recurse-submodules git@github.com:Chaoqi-LIU/oat.git
uv sync
```

**Training (multi-GPU / FSDP):**
```bash
python run.py                                          # default config (FSDP)
python run.py model=baseline seed=0 batch_size=64     # override via Hydra CLI
python run.py --multirun seed=0,1,2                   # Hydra multirun sweep
```

**Training (single GPU / Kaggle):**
```bash
python run.py strategy=single_gpu model.tokenizer_ckpt=path/to/latest.ckpt \
    dataset_path=data/libero/libero10_N500.zarr batch_size=16
```

**Evaluation:**
```bash
python scripts/eval_fddrat_libero.py -c path/to/ckpt -o path/to/output_dir
python scripts/eval_fddrat_libero.py -c checkpoints/ -o outputs/ -n 3  # avg over 3 runs
```

## Architecture

The project implements **FD-DRAT** (Feature-Decoupled Dynamic Routing Action Transformer), a VLA policy trained on the LIBERO robot manipulation benchmark.

**Entry point:** `run.py` — Hydra app that wires together config, data, model, W&B logger, FSDP strategy, and `pl.Trainer`.

**Config system (`conf/` + `src/core/config_schema.py`):**
- Hydra loads `conf/config.yaml` (which composes `model/baseline.yaml` + `strategy/fsdp.yaml`).
- All config is validated against typed Python dataclasses: `ExperimentConfig` → `FDDRATConfig` + `FSDPConfig`.
- `run.py` registers the schema via `ConfigStore` before the Hydra app starts.

**Training pipeline (`src/core/`):**
- `LitDataModule` wraps `oat.dataset.zarr_dataset.ZarrDataset` (LIBERO zarr format). It exposes a `normalizer` (fitted on train split) that must be injected into the policy before training.
- `LitSystem` is the `pl.LightningModule`. It holds `FDDRATPolicy` as `self.model` and delegates optimizer parameter groups to `model.get_optimizer_params()` (router/CRH use separate LR + no weight decay).
- **Normalizer injection order matters**: `run.py` explicitly calls `datamodule.setup()` then `system.model.set_normalizer(datamodule.normalizer)` before `trainer.fit()`.

**FD-DRAT policy (`src/fddrat/`):**
- `FDDRATPolicy` (`policy.py`): top-level module. Forward pass: obs → `FusedObservationEncoder` → `z_v`; actions → `FDDRATTok.encode` → tokens/latents; AR decoding → `ARModelWithHiddens`; routing → `ShadowRouter`; residual refinement → `ContinuousResidualHead`.
- `FDDRATTok` (`tokenizer.py`): wraps `oat.tokenizer.oat.tokenizer.OATTok`. Adds `decode_coarse()` for extracting the raw continuous macro-trajectory from latents. Auto-injects mock components (`DummyQuantizer`, `DummyDecoder`) when called with no args (dry-run / test mode).
- `ARModelWithHiddens` (`policy.py`): subclass of `oat`'s `AutoregressiveModel` that returns both `(logits, hidden_states)` instead of logits only.
- `ShadowRouter` (`modules/router.py`): computes stopping-probability logits from cosine similarity of adjacent hidden states + visual context `z_v`.
- `ContinuousResidualHead` (`modules/crh.py`): MLP that predicts a continuous residual `delta_a` from the flattened coarse trajectory + `z_v`.
- `FDDRATLoss` (`modules/loss.py`): `L = CE(AR) + λ·BCE(routing ratio) + β·masked_MSE(residual)`. MSE is masked to zero when `K_sampled >= H_l` (no residual penalty when full sequence is used).

**Key config params (`FDDRATConfig`):**
- `H_l=64`: AR token sequence length; `H_a=16`: action horizon output by CRH; `D_v=768`: visual embedding dim; `D_a=256`: action latent dim.
- `lambda_ratio`, `beta_mse`: loss weights; `target_ratio`: BCE target for the router.

**External dependencies:**
- `oat/` (cloned repo, added to `sys.path` in `run.py`): provides `ZarrDataset`, `LinearNormalizer`, `FusedObservationEncoder`, `AutoregressiveModel`, `OATTok`, `BasePolicy`, `BaseRunner`.
- `hnet/` (cloned repo, added to `sys.path`): available for future use.

**Inference (`predict_action`):** Autoregressive loop up to `H_l` steps with early exit when `ShadowRouter` fires (`sigmoid > 0.5`). Remaining slots are zero-padded (strict requirement — non-zero padding causes CRH hallucinations). CRH refinement and denormalization happen after the full sequence is assembled.
