# Инструкция: локальный запуск обучения FD-DRAT

Точное повторение k1.ipynb локально. Ячейки идут в том же порядке.

## Шаг 0 — Клонирование репозиториев

**Вариант A — чистая машина (Kaggle или новый сервер):**
```bash
git clone https://github.com/goombalab/hnet.git
git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
git clone https://github.com/SDogya/congenial-adventure.git
cp -r congenial-adventure/. .
rm -rf congenial-adventure
uv sync
```

**Вариант B — локально (уже находишься внутри congenial-adventure):**
```bash
git clone https://github.com/goombalab/hnet.git
git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
uv sync
```

## Шаг 1 — Переменные окружения
```bash
export WANDB_API_KEY=<ключ с wandb.ai>
export HF_TOKEN=<ключ с huggingface.co>
```

## Шаг 2 — Патчи OAT (один раз)
```python
# Патч lr_scheduler.py
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

# Заглушить print'ы про нормализацию
for path in [
    'oat/oat/perception/state_encoder.py',
    'oat/oat/perception/robomimic_vision_encoder.py',
]:
    txt = open(path).read()
    txt = txt.replace(
        'print(warning_msg(f"no normalizer params for port {port}, skipping normalization."))',
        'pass  # suppressed'
    ).replace(
        'print(f"no normalizer params for port {port}, skipping normalization.")',
        'pass  # suppressed'
    )
    open(path, 'w').write(txt)
```

## Шаг 3 — Датасет
```bash
python -c "
from huggingface_hub import login, snapshot_download
import os
login(token='$HF_TOKEN')
os.makedirs('oat/data/libero', exist_ok=True)
snapshot_download(
    repo_id='chaoqi-liu/libero10_N500.zarr',
    repo_type='dataset',
    local_dir='oat/data/libero'
)
"
unzip -o -q oat/data/libero/libero10_N500.zarr.zip -d oat/data/libero/
```
После этого должна существовать папка `oat/data/libero/libero10_N500.zarr/`.

## Шаг 4 — Обучение OAT токенайзера (~2-3 часа)
```bash
uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment \
    task.tokenizer.dataset.zarr_path="$(pwd)/oat/data/libero/libero10_N500.zarr"
```
Checkpoint появится в `oat/output/<дата>/<время>_train_oattok_libero10_N500/checkpoints/model.ckpt`.

## Шаг 5 — Обучение FD-DRAT
```bash
TOK=$(find oat/output -name 'model.ckpt' 2>/dev/null | sort | tail -1)
HYDRA_FULL_ERROR=1 MPLBACKEND=agg uv run run.py \
    strategy=single_gpu \
    model.tokenizer_ckpt=$TOK \
    dataset_path=oat/data/libero/libero10_N500.zarr \
    batch_size=16
```

## Отличия от Kaggle
| Kaggle | Локально |
|--------|----------|
| `UserSecretsClient().get_secret(...)` | `export WANDB_API_KEY=...` в shell |
| `/kaggle/working/oat/data/libero/...` | `oat/data/libero/...` (относительно корня репо) |
| `find /kaggle/working/output /kaggle/working -name model.ckpt` | `find oat/output -name model.ckpt` |
| 2× T4 GPU | адаптируется через `devices="auto"` |
