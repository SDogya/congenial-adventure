# ── 1. Repos & deps ──────────────────────────────────────────────────────────
!git clone https://github.com/goombalab/hnet.git
!git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
!rm -rf congenial-adventure && git clone https://github.com/SDogya/congenial-adventure.git && cp -r congenial-adventure/. . && rm -rf congenial-adventure

import os
from kaggle_secrets import UserSecretsClient
os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('wandb')

!uv add "zarr<3.0.0" dill einops numba vector-quantize-pytorch accelerate huggingface_hub "robomimic<0.3.0" torchvision wrapt pillow pandas diffusers
!uv sync

# Patch OAT's lr_scheduler.py: newer diffusers removed Union/Optional/Optimizer exports
p = 'oat/oat/model/common/lr_scheduler.py'
txt = open(p).read()
marker = 'from diffusers.optimization import ('
if marker in txt and 'from typing import Union' not in txt:
    idx = txt.index(marker)
    end_idx = txt.index(')', idx) + 1
    header = (
        'from typing import Union, Optional\n'
        'from torch.optim import Optimizer\n'
        'from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION'
    )
    open(p, 'w').write(txt[:idx] + header + txt[end_idx:])
    print('lr_scheduler.py patched OK')
else:
    print('lr_scheduler.py already clean, skipping')



!rm -rf congenial-adventure && git clone https://github.com/SDogya/congenial-adventure.git && cp -r congenial-adventure/. . && rm -rf congenial-adventure

# ── 2. Dataset (скачиваем прямо туда, куда смотрит OAT по дефолту) ───────────
import os
from huggingface_hub import snapshot_download
from huggingface_hub import login
hf_token = UserSecretsClient().get_secret('hugface')

if hf_token:
    login(token=hf_token)
else:
    print("Ошибка: Секрет 'hugface' не найден.")


os.makedirs('/kaggle/working/oat/data/libero', exist_ok=True)
snapshot_download(
    repo_id='chaoqi-liu/libero10_N500.zarr',
    repo_type='dataset',
    local_dir='/kaggle/working/oat/data/libero'
)



# Распаковываем архив прямо в ту же папку
!unzip -o -q /kaggle/working/oat/data/libero/libero10_N500.zarr.zip -d /kaggle/working/oat/data/libero/

# Проверяем, что папка появилась
!ls -ld /kaggle/working/oat/data/libero/*
# ── 3. Train OAT tokenizer (~2-3h, 300 epochs) ───────────────────────────────
!uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment \
    task.tokenizer.dataset.zarr_path="/kaggle/working/oat/data/libero/libero10_N500.zarr"
# ── 4. Train FD-DRAT ─────────────────────────────────────────────────────────
!TOK=$(find /kaggle/working/output /kaggle/working/oat/output -name '*.ckpt' 2>/dev/null | sort | tail -1) && \
 HYDRA_FULL_ERROR=1 MPLBACKEND=agg uv run run.py \
    strategy=single_gpu \
    model.tokenizer_ckpt=$TOK \
    dataset_path=/kaggle/working/oat/data/libero/libero10_N500.zarr \
    batch_size=16


Seed set to 42
/kaggle/working/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/kaggle/working/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/kaggle/working/.venv/lib/python3.12/site-packages/lightning_fabric/connector.py:571: `precision=bf16` is supported for historical reasons but its usage is discouraged. Please set your precision to bf16-mixed instead!
Using bfloat16 Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
[rank: 1] Seed set to 42
/kaggle/working/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/kaggle/working/.venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
[W412 23:54:04.060794819 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[W412 23:54:04.070974287 Utils.hpp:137] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

wandb: WARNING The anonymous setting has no effect and will be removed in a future version.
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: sebersehmer (sebersehmer-nopeinc) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
]11;?]11;?wandb: ⢿ Waiting for wandb.init()...
wandb: ⣻ Waiting for wandb.init()...
wandb: ⣽ Waiting for wandb.init()...
wandb: ⣾ setting up run x9z1se1x (0.4s)
wandb: Tracking run with wandb version 0.25.1
wandb: Run data is saved locally in wandb/run-20260412_235406-x9z1se1x
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run absurd-snowflake-7
wandb: ⭐️ View project at https://wandb.ai/sebersehmer-nopeinc/VLA-experiment
wandb: 🚀 View run at https://wandb.ai/sebersehmer-nopeinc/VLA-experiment/runs/x9z1se1x
LOCAL_RANK: 1 - CUDA_VISIBLE_DEVICES: [0,1]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
[rank0]:[W412 23:54:11.954340810 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
[rank1]:[W412 23:54:11.954612755 Utils.hpp:112] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function operator())
/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/model_summary/model_summary.py:242: Precision bf16-mixed is not supported by the model summary.  Estimated model size in MB will not be accurate. Using 32 bits instead.
┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name  ┃ Type         ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ model │ FDDRATPolicy │  147 M │ train │     0 │
└───┴───────┴──────────────┴────────┴───────┴───────┘
Trainable params: 147 M                                                         
Non-trainable params: 18.7 K                                                    
Total params: 147 M                                                             
Total estimated model params size (MB): 590                                     
Modules in train mode: 496                                                      
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/utilities/_
pytree.py:21: `isinstance(treespec, LeafSpec)` is deprecated, use 
`isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.
no normalizer params for port agentview_rgb, skipping normalization.
no normalizer params for port robot0_eye_in_hand_rgb, skipping normalization.
no normalizer params for port agentview_rgb, skipping normalization.
no normalizer params for port robot0_eye_in_hand_rgb, skipping normalization.
Error executing job with overrides: ['strategy=single_gpu', 'model.tokenizer_ckpt=/kaggle/working/oat/output/20260412/225721_train_oattok_libero10_N500/checkpoints/latest.ckpt', 'dataset_path=/kaggle/working/oat/data/libero/libero10_N500.zarr', 'batch_size=16']
[rank1]: Traceback (most recent call last):
[rank1]:   File "/kaggle/working/run.py", line 74, in <module>
[rank1]:     main()
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/main.py", line 94, in decorated_main
[rank1]:     _run_hydra(
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank1]:     _run_app(
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank1]:     run_and_report(
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank1]:     raise ex
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank1]:     return func()
[rank1]:            ^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank1]:     lambda: hydra.run(
[rank1]:             ^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank1]:     _ = ret.return_value
[rank1]:         ^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 260, in return_value
[rank1]:     raise self._return_value
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 186, in run_job
[rank1]:     ret.return_value = task_function(task_cfg)
[rank1]:                        ^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/run.py", line 71, in main
[rank1]:     trainer.fit(model=system, datamodule=datamodule)
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
[rank1]:     call._call_and_handle_interrupt(
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
[rank1]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank1]:     return function(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
[rank1]:     self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
[rank1]:     results = self._run_stage()
[rank1]:               ^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1121, in _run_stage
[rank1]:     self._run_sanity_check()
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1150, in _run_sanity_check
[rank1]:     val_loop.run()
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/utilities.py", line 179, in _decorator
[rank1]:     return loop_run(self, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 146, in run
[rank1]:     self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 441, in _evaluation_step
[rank1]:     output = call._call_strategy_hook(trainer, hook_name, *step_args)
[rank1]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
[rank1]:     output = fn(*args, **kwargs)
[rank1]:              ^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 411, in validation_step
[rank1]:     return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 641, in __call__
[rank1]:     wrapper_output = wrapper_module(*args, **kwargs)
[rank1]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1699, in forward
[rank1]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank1]:          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1524, in _run_ddp_forward
[rank1]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 634, in wrapped_forward
[rank1]:     out = method(*_args, **_kwargs)
[rank1]:           ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/src/core/system.py", line 33, in validation_step
[rank1]:     out = self.model(batch)
[rank1]:           ^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/src/fddrat/policy.py", line 176, in forward
[rank1]:     delta_a_norm = self.crh(a_coarse_norm_detached, z_v)
[rank1]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/kaggle/working/src/fddrat/modules/crh.py", line 45, in forward
[rank1]:     x = torch.cat([a_coarse_flat, z_v], dim=1)  # [B, (H_a*D_a) + D_v]
[rank1]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: Tensors must have same number of dimensions: got 2 and 3
Error executing job with overrides: ['strategy=single_gpu', 'model.tokenizer_ckpt=/kaggle/working/oat/output/20260412/225721_train_oattok_libero10_N500/checkpoints/latest.ckpt', 'dataset_path=/kaggle/working/oat/data/libero/libero10_N500.zarr', 'batch_size=16']
Traceback (most recent call last):
  File "/kaggle/working/run.py", line 74, in <module>
    main()
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
           ^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
            ^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
        ^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
                       ^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/run.py", line 71, in main
    trainer.fit(model=system, datamodule=datamodule)
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
    call._call_and_handle_interrupt(
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
    return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
    return function(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
    self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1121, in _run_stage
    self._run_sanity_check()
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1150, in _run_sanity_check
    val_loop.run()
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/utilities.py", line 179, in _decorator
    return loop_run(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 146, in run
    self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 441, in _evaluation_step
    output = call._call_strategy_hook(trainer, hook_name, *step_args)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 411, in validation_step
    return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 641, in __call__
    wrapper_output = wrapper_module(*args, **kwargs)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1699, in forward
    else self._run_ddp_forward(*inputs, **kwargs)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1524, in _run_ddp_forward
    return self.module(*inputs, **kwargs)  # type: ignore[index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 634, in wrapped_forward
    out = method(*_args, **_kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/src/core/system.py", line 33, in validation_step
    out = self.model(batch)
          ^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/src/fddrat/policy.py", line 176, in forward
    delta_a_norm = self.crh(a_coarse_norm_detached, z_v)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/src/fddrat/modules/crh.py", line 45, in forward
    x = torch.cat([a_coarse_flat, z_v], dim=1)  # [B, (H_a*D_a) + D_v]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
[rank0]: Traceback (most recent call last):
[rank0]:   File "/kaggle/working/run.py", line 74, in <module>
[rank0]:     main()
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/main.py", line 94, in decorated_main
[rank0]:     _run_hydra(
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
[rank0]:     _run_app(
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 457, in _run_app
[rank0]:     run_and_report(
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
[rank0]:     raise ex
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
[rank0]:     return func()
[rank0]:            ^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
[rank0]:     lambda: hydra.run(
[rank0]:             ^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/hydra.py", line 132, in run
[rank0]:     _ = ret.return_value
[rank0]:         ^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 260, in return_value
[rank0]:     raise self._return_value
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/core/utils.py", line 186, in run_job
[rank0]:     ret.return_value = task_function(task_cfg)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/run.py", line 71, in main
[rank0]:     trainer.fit(model=system, datamodule=datamodule)
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 584, in fit
[rank0]:     call._call_and_handle_interrupt(
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 48, in _call_and_handle_interrupt
[rank0]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
[rank0]:     return function(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 630, in _fit_impl
[rank0]:     self._run(model, ckpt_path=ckpt_path, weights_only=weights_only)
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1079, in _run
[rank0]:     results = self._run_stage()
[rank0]:               ^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1121, in _run_stage
[rank0]:     self._run_sanity_check()
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 1150, in _run_sanity_check
[rank0]:     val_loop.run()
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/utilities.py", line 179, in _decorator
[rank0]:     return loop_run(self, *args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 146, in run
[rank0]:     self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 441, in _evaluation_step
[rank0]:     output = call._call_strategy_hook(trainer, hook_name, *step_args)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/trainer/call.py", line 329, in _call_strategy_hook
[rank0]:     output = fn(*args, **kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 411, in validation_step
[rank0]:     return self._forward_redirection(self.model, self.lightning_module, "validation_step", *args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 641, in __call__
[rank0]:     wrapper_output = wrapper_module(*args, **kwargs)
[rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1699, in forward
[rank0]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank0]:          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/parallel/distributed.py", line 1524, in _run_ddp_forward
[rank0]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/pytorch_lightning/strategies/strategy.py", line 634, in wrapped_forward
[rank0]:     out = method(*_args, **_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/src/core/system.py", line 33, in validation_step
[rank0]:     out = self.model(batch)
[rank0]:           ^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/src/fddrat/policy.py", line 176, in forward
[rank0]:     delta_a_norm = self.crh(a_coarse_norm_detached, z_v)
[rank0]:                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/kaggle/working/src/fddrat/modules/crh.py", line 45, in forward
[rank0]:     x = torch.cat([a_coarse_flat, z_v], dim=1)  # [B, (H_a*D_a) + D_v]
[rank0]:         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: Tensors must have same number of dimensions: got 2 and 3
wandb: 
wandb: 🚀 View run absurd-snowflake-7 at: https://wandb.ai/sebersehmer-nopeinc/VLA-experiment/runs/x9z1se1x
