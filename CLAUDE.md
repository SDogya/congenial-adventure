# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Pre-trained models (current state)

The repo currently ships two checkpoints that were trained and evaluated on Kaggle:

| File | Contents |
|------|----------|
| `src/model/model.ckpt` | OAT tokenizer (trained on LIBERO-10, `H_l=8`) |
| `src/model/eval_mod.ckpt` | FD-DRAT policy (trained on top of the above tokenizer) |

**If you want to run eval as-is** (no re-training), point the eval script at `src/model/eval_mod.ckpt`.  
**If you train new models**, replace `model.tokenizer_ckpt` and the eval `-c` path accordingly — the paths below use the pre-trained defaults.

---

## Commands

### Setup

**Clone and install (local machine):**
```bash
git clone --recurse-submodules git@github.com:Chaoqi-LIU/oat.git
uv sync
uv pip install -e ./oat   # installs LIBERO, robosuite, robomimic and other oat deps
```

`uv sync` only installs our minimal deps (torch, lightning, wandb, hydra). The oat subdir must be installed separately to pull in LIBERO and simulation deps.

**Auth (local only — Kaggle reads these from Secrets):**
```bash
wandb login                  # or: export WANDB_API_KEY=...
huggingface-cli login        # or: export HF_TOKEN=...  (read-only token is enough)
```

---

### Dataset

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('chaoqi-liu/libero10_N500.zarr', repo_type='dataset', local_dir='data/libero/')
"
# Unzip shards if downloaded as zip
```

---

### Training

**Train OAT tokenizer** (must run before FD-DRAT training):
```bash
uv run python oat/scripts/run_workspace.py --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=100 \
    task.dataset.zarr_path=data/libero/libero10_N500.zarr \
    training.checkpoint.save_path=src/model/model.ckpt
```

**Train FD-DRAT (single GPU / Kaggle):**
```bash
HYDRA_FULL_ERROR=1 MPLBACKEND=agg uv run run.py \
    strategy=single_gpu \
    model.tokenizer_ckpt=src/model/model.ckpt \
    dataset_path=data/libero/libero10_N500.zarr \
    model.H_l=8 \
    batch_size=16
# model.H_l=8 must match the tokenizer — default cfg.H_l=64 is a known bug
```

**Train FD-DRAT (multi-GPU / FSDP):**
```bash
python run.py model.tokenizer_ckpt=src/model/model.ckpt \
    dataset_path=data/libero/libero10_N500.zarr model.H_l=8
python run.py --multirun seed=0,1,2   # Hydra sweep
```

---

### Evaluation

**Headless Linux (server / Kaggle T4) — use EGL:**
```bash
# Do NOT set PYOPENGL_PLATFORM — MuJoCo uses its own EGL binding
MUJOCO_GL=egl MPLBACKEND=agg \
    uv run python scripts/eval_fddrat_libero.py \
    -c src/model/eval_mod.ckpt \
    -o eval_out/ \
    --n_test 50 --n_test_vis 5
```

**Local machine with display (Linux/macOS):**
```bash
# MUJOCO_GL=glfw uses the windowed renderer; omit on macOS (uses default)
MUJOCO_GL=glfw MPLBACKEND=agg \
    uv run python scripts/eval_fddrat_libero.py \
    -c src/model/eval_mod.ckpt \
    -o eval_out/ \
    --n_test 50 --n_test_vis 5
```

`n_parallel_envs` is hard-coded to 10 in the script — running all 50 envs simultaneously causes OOM on a 16 GB GPU.

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
- `H_l=8`: AR token sequence length **as used by the OAT tokenizer** (cfg default is 64 — always override with `model.H_l=8` or the nested-dropout range becomes wrong); `H_a=32`: action horizon; `D_v=768`: visual embedding dim; `D_a=7`: action dim.
- `lambda_ratio`, `beta_mse`: loss weights; `target_ratio`: BCE target for the router.

**External dependencies:**
- `oat/` (cloned repo, added to `sys.path` in `run.py`): provides `ZarrDataset`, `LinearNormalizer`, `FusedObservationEncoder`, `AutoregressiveModel`, `OATTok`, `BasePolicy`, `BaseRunner`.

**Inference (`predict_action`):** Autoregressive loop up to `H_l` steps with early exit when `ShadowRouter` fires (`sigmoid > 0.5`). Remaining slots are zero-padded (strict requirement — non-zero padding causes CRH hallucinations). CRH refinement and denormalization happen after the full sequence is assembled.
