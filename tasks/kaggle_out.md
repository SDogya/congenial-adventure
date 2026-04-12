# ── 3. Train OAT tokenizer (~2-3h, 300 epochs) ───────────────────────────────
# Запускаем из /kaggle/working — один venv, не создаём второй в oat/.venv
!uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment

[2026-04-12 17:33:12,973][hydra.utils][ERROR] - Error getting class at oat.workspace.train_oattok.TrainOATTokWorkspace: Error loading 'oat.workspace.train_oattok.TrainOATTokWorkspace':
ImportError("cannot import name 'Union' from 'diffusers.optimization' (/kaggle/working/.venv/lib/python3.12/site-packages/diffusers/optimization.py)")
Error executing job with overrides: ['task/tokenizer=libero/libero10', 'training.num_epochs=300', 'logging.project=VLA-experiment']
Traceback (most recent call last):
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 644, in _locate
    obj = getattr(obj, part)
          ^^^^^^^^^^^^^^^^^^
AttributeError: module 'oat.workspace' has no attribute 'train_oattok'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 650, in _locate
    obj = import_module(mod)
          ^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/kaggle/working/oat/oat/workspace/train_oattok.py", line 30, in <module>
    from oat.model.common.lr_scheduler import get_scheduler
  File "/kaggle/working/oat/oat/model/common/lr_scheduler.py", line 1, in <module>
    from diffusers.optimization import (
ImportError: cannot import name 'Union' from 'diffusers.optimization' (/kaggle/working/.venv/lib/python3.12/site-packages/diffusers/optimization.py)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/kaggle/working/oat/scripts/run_workspace.py", line 38, in main
    cls = hydra.utils.get_class(cfg._target_)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/utils.py", line 40, in get_class
    raise e
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/utils.py", line 31, in get_class
    cls = _locate(path)
          ^^^^^^^^^^^^^
  File "/kaggle/working/.venv/lib/python3.12/site-packages/hydra/_internal/utils.py", line 658, in _locate
    raise ImportError(
ImportError: Error loading 'oat.workspace.train_oattok.TrainOATTokWorkspace':
ImportError("cannot import name 'Union' from 'diffusers.optimization' (/kaggle/working/.venv/lib/python3.12/site-packages/diffusers/optimization.py)")

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
