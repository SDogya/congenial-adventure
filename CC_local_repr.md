# Инструкция по воспроизведению FD-DRAT

Документ покрывает два сценария:
- **Kaggle** — точное повторение рабочего ноутбука `notebooks/should works.ipynb`
- **Локально** — запуск на собственном GPU (Linux/macOS)

Адресован LLM-агенту: все команды приведены дословно, все критические path-нюансы отмечены явно.

---

## СЦЕНАРИЙ A: Kaggle (рекомендуемый)

### Предварительные условия

1. Аккаунт Kaggle с доступом к GPU (T4 — стандартная конфигурация).
2. В настройках ноутбука добавить два **Kaggle Secrets**:
   - `wandb` — API-ключ с [wandb.ai/authorize](https://wandb.ai/authorize)
   - `hugface` — HuggingFace токен с правом на чтение (`read`)
3. В ноутбуке включить **Internet** (Notebook settings → Internet → On).
4. (Опционально, ускоряет загрузку) Добавить датасет
   `chaoqi-liu/libero10_N500.zarr` через Add data → Datasets.
   Если добавлен, он появится по пути
   `/kaggle/input/libero10n500zarr/libero10_N500.zarr.zip`.
   Если не добавлен — скачается через HuggingFace в Cell 2–3.

---

### Ячейки ноутбука (выполнять по порядку)

#### Cell 0 — Клонирование, зависимости, патчи OAT

```python
# ── Repos ─────────────────────────────────────────────────────────────────────
!git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
# Клонируем основной репозиторий и копируем содержимое в рабочую директорию
!rm -rf congenial-adventure && git clone https://github.com/SDogya/congenial-adventure.git \
    && cp -r congenial-adventure/. . && rm -rf congenial-adventure

import os
from kaggle_secrets import UserSecretsClient
os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('wandb')

# ── Зависимости ────────────────────────────────────────────────────────────────
# zarr<3.0 обязательно — oat не работает с zarr 3.x
# robomimic<0.3.0 обязательно — 0.3+ ломает robosuite интеграцию
!uv add "zarr<3.0.0" dill einops numba vector-quantize-pytorch accelerate \
    huggingface_hub "robomimic<0.3.0" torchvision wrapt pillow pandas \
    diffusers av gymnasium libero
!uv sync

# ── Патч 1: oat/oat/model/common/lr_scheduler.py ──────────────────────────────
# Проблема: новые версии diffusers убрали Union/Optional/Optimizer из публичного API.
# Без патча импорт упадёт с ImportError при старте обучения.
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

# ── Патч 2: заглушить лишние print'ы нормализатора ────────────────────────────
# Без патча лог обучения засоряется тысячами строк "no normalizer params for port ...".
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
print('normalizer prints suppressed OK')
```

> **PATH NOTE.** После `cp -r congenial-adventure/. .` рабочая директория Kaggle
> (`/kaggle/working/`) содержит файлы репозитория напрямую:
> `run.py`, `src/`, `conf/`, `scripts/` и т.д.
> OAT клонируется в `/kaggle/working/oat/`.
> Все последующие пути отсчитываются от `/kaggle/working/`.

---

#### Cell 1–3 — Датасет

Если датасет добавлен как Kaggle Dataset (рекомендуется — быстрее):
```python
import os, shutil
src = '/kaggle/input/libero10n500zarr/libero10_N500.zarr.zip'
dst_dir = '/kaggle/working/oat/data/libero'
os.makedirs(dst_dir, exist_ok=True)
shutil.copy(src, dst_dir)
```

Если нет — скачать с HuggingFace:
```python
import os
from huggingface_hub import login, snapshot_download
from kaggle_secrets import UserSecretsClient

login(token=UserSecretsClient().get_secret('hugface'))  # read-only токен достаточен
os.makedirs('/kaggle/working/oat/data/libero', exist_ok=True)
snapshot_download(
    repo_id='chaoqi-liu/libero10_N500.zarr',
    repo_type='dataset',
    local_dir='/kaggle/working/oat/data/libero'
)
```

Разархивировать:
```bash
!unzip -o -q /kaggle/working/oat/data/libero/libero10_N500.zarr.zip \
    -d /kaggle/working/oat/data/libero/
```

> **PATH NOTE.** После разархивирования должна существовать директория
> `/kaggle/working/oat/data/libero/libero10_N500.zarr/` (с косой чертой — это папка
> zarr-хранилища, не файл). Ни `train_oattok`, ни `run.py` не умеют сами разархивировать .zip.

---

#### Cell 4 — Обучение OAT токенизатора (опционально)

**Пропустить**, если используются готовые веса из `src/model/model.ckpt`
(pre-trained чекпоинт уже находится в репозитории).

Запускать только при полном переобучении с нуля:
```bash
!uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment \
    task.tokenizer.dataset.zarr_path="/kaggle/working/oat/data/libero/libero10_N500.zarr" \
    training.checkpoint.save_path="src/model/model.ckpt"
```

> **PATH NOTE.** Параметр `training.checkpoint.save_path` задан явно в `src/model/model.ckpt`.
> Без него OAT сохранит чекпоинт в `oat/outputs/<дата>/<время>_train_oattok.../checkpoints/last.ckpt`
> (имя папки содержит timestamp), и на следующем шаге придётся искать путь через `find`.
> Путь к `zarr_path` должен быть **абсолютным** — OAT запускается из `oat/`, а не из рабочей
> директории ноутбука.

---

#### Cell 5 — Обучение FD-DRAT

```python
import os
TOK = 'src/model/model.ckpt'  # pre-trained токенизатор из репо; при переобучении — тот же путь
print(f'Using tokenizer: {TOK}')
```

```bash
!HYDRA_FULL_ERROR=1 MPLBACKEND=agg uv run run.py \
    strategy=single_gpu \
    model.tokenizer_ckpt=src/model/model.ckpt \
    dataset_path=/kaggle/working/oat/data/libero/libero10_N500.zarr \
    model.H_l=8 \
    batch_size=16
```

> **CRITICAL BUG NOTE.** Параметр `model.H_l=8` **обязателен**.
> Дефолтное значение в конфиге — `H_l=64`, тогда как OAT токенизатор использует `H_l=8`.
> При `H_l=64` Nested Dropout применяется в ~12.5% шагов вместо ~50%, что делает CRH и роутер
> недообученными. Чекпоинт `src/model/eval_mod.ckpt` в репозитории обучен **с этим багом** —
> он даёт 6% SR. Для воспроизведения опубликованного результата баг оставить как есть;
> для получения реального потолка метода — передать `model.H_l=8`.

Чекпоинт по умолчанию сохраняется в `outputs/<дата>_<время>/checkpoints/`.
W&B логирует метрики в реальном времени (проект `VLA-experiment`).

---

#### Cell 6 — Оценка (eval)

```python
import os, subprocess, pathlib, yaml

# EGL — единственный рабочий рендерер на headless Kaggle GPU
os.environ['MUJOCO_GL'] = 'egl'
os.environ.pop('PYOPENGL_PLATFORM', None)  # НЕ устанавливать: MuJoCo использует свой EGL binding
os.environ['MPLBACKEND'] = 'agg'           # переопределить Jupyter inline backend

subprocess.run(['apt-get', 'install', '-y', '-q', 'libegl1'], check=False)

# Создать конфиг LIBERO — без него LiberoRunner падает при старте
_cfg_dir = pathlib.Path.home() / '.libero'
_cfg_dir.mkdir(exist_ok=True)
_cfg_file = _cfg_dir / 'config.yaml'

if not _cfg_file.exists():
    _pkg = subprocess.check_output(
        ['uv', 'run', 'python', '-c',
         'import libero.libero, pathlib; print(pathlib.Path(libero.libero.__file__).parent)'],
        text=True
    ).strip()
    _pkg = pathlib.Path(_pkg)
    yaml.dump({
        'benchmark_root': str(_pkg),
        'bddl_files':     str(_pkg / 'bddl_files'),
        'init_states':    str(_pkg / 'init_files'),
        'datasets':       str(_pkg.parent / 'datasets'),
        'assets':         str(_pkg / 'assets'),
    }, _cfg_file.open('w'))
    print(f'Created LIBERO config → {_cfg_file}')
else:
    print(f'LIBERO config already exists: {_cfg_file}')
```

```bash
!MPLBACKEND=agg uv run python scripts/eval_fddrat_libero.py \
    -c src/model/eval_mod.ckpt \
    -o eval_out/ \
    --n_test 50 \
    --n_test_vis 5
```

> **PATH NOTE.** `-c src/model/eval_mod.ckpt` — путь относительно `/kaggle/working/`.
> Чтобы оценить свой чекпоинт после обучения, заменить на путь из `outputs/` или найти:
> `find /kaggle/working/outputs -name '*.ckpt' | sort | tail -1`

> **OOM NOTE.** `n_parallel_envs` жёстко задан равным `10` в `scripts/eval_fddrat_libero.py`.
> При 50 роллаутах запускается 5 батчей по 10. При 50 параллельных воркерах — OOM на T4 16 GB.

---

## СЦЕНАРИЙ B: Локально

### Предварительные условия

- Linux или macOS, Python 3.12+, `uv` (`pip install uv`)
- GPU ≥ 16 GB VRAM для обучения; для eval достаточно 8 GB
- MuJoCo: на Linux с дисплеем — ничего дополнительного; headless — нужен EGL (`sudo apt install libegl1`)

### Шаг 0 — Клонирование

```bash
git clone https://github.com/SDogya/congenial-adventure.git
cd congenial-adventure
git clone --recurse-submodules https://github.com/Chaoqi-LIU/oat.git
uv sync
uv pip install -e ./oat   # устанавливает LIBERO, robosuite, robomimic и прочие зависимости OAT
```

> **PATH NOTE.** Все дальнейшие пути отсчитываются от корня репозитория `congenial-adventure/`.
> OAT находится в `./oat/`. Не нужно делать `cp -r` — локально файлы уже на месте.

### Шаг 1 — Переменные окружения

```bash
export WANDB_API_KEY=<ключ с wandb.ai/authorize>
export HF_TOKEN=<HuggingFace read-only токен>
```

### Шаг 2 — Патчи OAT (один раз после клонирования)

```python
# Выполнить: uv run python -c "exec(open('patch_oat.py').read())"
# или вставить в интерактивную сессию

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
    print('patched OK')

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
print('normalizer prints suppressed OK')
```

### Шаг 3 — Датасет

```bash
python -c "
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ['HF_TOKEN'])
os.makedirs('oat/data/libero', exist_ok=True)
snapshot_download(
    repo_id='chaoqi-liu/libero10_N500.zarr',
    repo_type='dataset',
    local_dir='oat/data/libero'
)
"
# Если скачан как .zip — разархивировать
unzip -o -q oat/data/libero/libero10_N500.zarr.zip -d oat/data/libero/
```

> **PATH NOTE.** Результат — директория `oat/data/libero/libero10_N500.zarr/`.

### Шаг 4 — Обучение OAT токенизатора (опционально)

Пропустить, если `src/model/model.ckpt` уже существует.

```bash
uv run python oat/scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=libero/libero10 \
    training.num_epochs=300 \
    logging.project=VLA-experiment \
    "task.tokenizer.dataset.zarr_path=$(pwd)/oat/data/libero/libero10_N500.zarr" \
    training.checkpoint.save_path=src/model/model.ckpt
```

> **PATH NOTE.** `$(pwd)/...` — абсолютный путь обязателен, OAT меняет CWD при запуске.
> `training.checkpoint.save_path` задан явно, иначе искать через
> `find oat/outputs -name 'last.ckpt' | sort | tail -1`.

### Шаг 5 — Обучение FD-DRAT

```bash
HYDRA_FULL_ERROR=1 MPLBACKEND=agg uv run run.py \
    strategy=single_gpu \
    model.tokenizer_ckpt=src/model/model.ckpt \
    dataset_path=oat/data/libero/libero10_N500.zarr \
    model.H_l=8 \
    batch_size=16
```

> **CRITICAL.** `model.H_l=8` обязателен — см. примечание в Сценарии A, Cell 5.

### Шаг 6 — Создать конфиг LIBERO (один раз, перед eval)

```python
import subprocess, pathlib, yaml

_cfg_dir = pathlib.Path.home() / '.libero'
_cfg_dir.mkdir(exist_ok=True)
_cfg_file = _cfg_dir / 'config.yaml'

if not _cfg_file.exists():
    _pkg = subprocess.check_output(
        ['uv', 'run', 'python', '-c',
         'import libero.libero, pathlib; print(pathlib.Path(libero.libero.__file__).parent)'],
        text=True
    ).strip()
    _pkg = pathlib.Path(_pkg)
    yaml.dump({
        'benchmark_root': str(_pkg),
        'bddl_files':     str(_pkg / 'bddl_files'),
        'init_states':    str(_pkg / 'init_files'),
        'datasets':       str(_pkg.parent / 'datasets'),
        'assets':         str(_pkg / 'assets'),
    }, _cfg_file.open('w'))
    print(f'Created: {_cfg_file}')
```

### Шаг 7 — Оценка

**Headless Linux (без дисплея):**
```bash
MUJOCO_GL=egl MPLBACKEND=agg \
    uv run python scripts/eval_fddrat_libero.py \
    -c src/model/eval_mod.ckpt \
    -o eval_out/ \
    --n_test 50 \
    --n_test_vis 5
```

**Linux с дисплеем:**
```bash
MUJOCO_GL=glfw MPLBACKEND=agg \
    uv run python scripts/eval_fddrat_libero.py \
    -c src/model/eval_mod.ckpt \
    -o eval_out/ \
    --n_test 50 \
    --n_test_vis 5
```

> **НЕ УСТАНАВЛИВАТЬ** `PYOPENGL_PLATFORM` — MuJoCo использует собственный EGL-биндинг,
> `PYOPENGL_PLATFORM=osmesa` вызывает `AttributeError: module 'OpenGL.GL' has no attribute 'GL_DEPTH_COMPONENT32'`.

---

## Таблица соответствия путей Kaggle ↔ локально

| Назначение | Kaggle | Локально |
|---|---|---|
| Рабочая директория | `/kaggle/working/` | `<repo>/` (корень репозитория) |
| OAT репозиторий | `/kaggle/working/oat/` | `<repo>/oat/` |
| Датасет zarr | `/kaggle/working/oat/data/libero/libero10_N500.zarr` | `<repo>/oat/data/libero/libero10_N500.zarr` |
| OAT токенизатор | `src/model/model.ckpt` | `src/model/model.ckpt` |
| FD-DRAT чекпоинт | `src/model/eval_mod.ckpt` | `src/model/eval_mod.ckpt` |
| LIBERO config | `~/.libero/config.yaml` | `~/.libero/config.yaml` |
| W&B ключ | Kaggle Secret `wandb` | `export WANDB_API_KEY=...` |
| HF токен | Kaggle Secret `hugface` | `export HF_TOKEN=...` |

---

## Известные ловушки — сводная таблица

| # | Симптом | Причина | Решение |
|---|---|---|---|
| 1 | `ImportError: cannot import name 'Union' from 'diffusers'` | OAT lr_scheduler.py использует старый API diffusers | Применить Патч 1 из Cell 0 |
| 2 | Тысячи строк `no normalizer params for port ...` в логе | OAT print'ы нормализатора | Применить Патч 2 из Cell 0 |
| 3 | `AttributeError: module 'OpenGL.GL' has no attribute 'GL_DEPTH_COMPONENT32'` | `PYOPENGL_PLATFORM=osmesa` установлен | Убрать эту переменную; использовать `MUJOCO_GL=egl` |
| 4 | `ValueError: Matplotlib is currently using module://matplotlib_inline...` | Jupyter переопределяет MPLBACKEND в subprocess | Передать `MPLBACKEND=agg` явно в env перед запуском eval |
| 5 | `RuntimeError: size mismatch, latents 64 vs pos_emb 8` при инференсе | `model.H_l=64` (дефолт) ≠ H_l=8 токенизатора | При обучении передавать `model.H_l=8`; инференс читает `decoder.latent_horizon` автоматически (исправлено в `policy.py`) |
| 6 | `RuntimeError: unexpected keys 'model.normalizer.action.*'` при загрузке чекпоинта | `LitSystem` не создаёт normalizer заранее | Уже исправлено в `src/core/system.py` через `on_load_checkpoint` |
| 7 | OOM при eval (`CUDA out of memory`) | Слишком много параллельных MuJoCo-воркеров | `n_parallel_envs=10` жёстко задан в `scripts/eval_fddrat_libero.py` — не менять на большее |
| 8 | OAT сохраняет чекпоинт в пути с timestamp | Дефолтный output dir OAT содержит дату/время | Передавать `training.checkpoint.save_path=src/model/model.ckpt` |
| 9 | `zarr.errors.GroupNotFoundError` или `FileNotFoundError` при обучении | Путь к датасету относительный, OAT меняет CWD | Передавать абсолютный путь: `$(pwd)/oat/data/libero/libero10_N500.zarr` |
| 10 | `FileNotFoundError: ~/.libero/config.yaml` | LIBERO config не создан | Выполнить Шаг 6 / Cell 6 перед eval |
