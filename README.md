# FD-DRAT

**Fixed-Dimension Decoupled Residual Action Tokenization** — политика управления роботом для бенчмарка LIBERO10.

Идея: поверх OAT-токенизатора навешан изолированный роутер (логика H-Net), который на инференсе обрывает авторегрессивную генерацию раньше `H_l` шагов. Чтобы ранний обрыв не портил точность, добавлен Continuous Residual Head (CRH), который доводит грубую траекторию до финальной.

Три компонента потерь:
- **L_CE** — кросс-энтропия AR-декодера (основная магистраль)
- **L_ratio** — BCE роутера (обучается предсказывать целевой коэффициент сжатия)
- **L_mse** — MSE остатка CRH (маскируется, если K == H_l)

---

## Структура проекта

```
.
├── run.py                      # точка входа (Hydra app)
├── k1.ipynb                    # ноутбук Kaggle: полный pipeline запуска
├── hypotesis.md                # математика и мотивация гипотезы FD-DRAT
├── CC_local_repr.md            # инструкция локального воспроизведения
├── pyproject.toml / requirements.txt
│
├── conf/                       # Hydra-конфиги
│   ├── config.yaml             # корневой конфиг (компонует model + strategy)
│   ├── model/
│   │   └── baseline.yaml       # дефолтные гиперпараметры FDDRATConfig
│   └── strategy/
│       ├── fsdp.yaml           # FSDP, несколько GPU
│       └── single_gpu.yaml     # single-GPU / Kaggle T4
│
├── src/
│   ├── core/
│   │   ├── config_schema.py    # типизированные датаклассы: ExperimentConfig, FDDRATConfig, FSDPConfig
│   │   ├── system.py           # LitSystem (pl.LightningModule): training/val step, оптимизатор
│   │   └── datamodule.py       # LitDataModule + LazyZarrDataset
│   │
│   ├── fddrat/
│   │   ├── policy.py           # FDDRATPolicy + ARModelWithHiddens
│   │   ├── tokenizer.py        # FDDRATTok (обёртка OATTok) + mock-компоненты
│   │   └── modules/
│   │       ├── crh.py          # ContinuousResidualHead (MLP остатка)
│   │       ├── router.py       # ShadowRouter (косинусная маршрутизация)
│   │       └── loss.py         # FDDRATLoss (CE + BCE + masked MSE)
│   │
│   └── utils/
│       └── setup.py            # enforce_determinism (seed, cudnn)
│
├── scripts/
│   └── eval_fddrat_libero.py   # оценка политики в симуляторе, p99 latency, JSON-лог
│
└── src/model/
    └── model.ckpt              # чекпоинт обученного OAT-токенизатора
```

---

## Ключевые файлы

### `run.py`
Hydra-приложение. Собирает `LitDataModule` + `LitSystem`, настраивает W&B logger, `ModelCheckpoint`, выбирает стратегию (FSDP или DDP), явно вызывает `datamodule.setup()` и `set_normalizer()` до `trainer.fit()`.

### `conf/`
Hydra компонует конфиг из `config.yaml` + переопределений CLI. Схема валидируется через `ConfigStore` и датаклассы из `config_schema.py`. Для Kaggle: `strategy=single_gpu`.

### `src/core/config_schema.py`
Три датакласса:
- `FSDPConfig` — стратегия (FSDP / single-GPU, mixed precision)
- `FDDRATConfig` — гиперпараметры модели: `H_l=8`, `H_a=32`, `D_a=7`, `D_v=768`, `obs_dim=138`, `tokenizer_ckpt`
- `ExperimentConfig` — seed, batch_size, lr, dataset_path, shape_meta

### `src/core/datamodule.py`
`LazyZarrDataset` — monkey-patch `ReplayBuffer.copy_from_path` → `create_from_path(mode='r')`, чтобы датасет (~25 ГБ) не грузился в RAM целиком. Исключает RGB-ключи из нормализатора (vision encoder нормализует их сам).

`LitDataModule` — оборачивает датасет в DataLoader, выставляет `self.normalizer` после `setup()`.

### `src/core/system.py`
`LitSystem` — `pl.LightningModule`. Делегирует группы параметров в `model.get_optimizer_params()` (роутер + CRH получают `lr=1e-4` без weight decay). Планировщик: `CosineAnnealingLR(T_max=estimated_stepping_batches)`.

### `src/fddrat/policy.py`
- `ARModelWithHiddens` — подкласс OAT `AutoregressiveModel`, возвращает `(logits, hidden_states)` вместо только logits.
- `FDDRATPolicy` — главный модуль. Прямой проход при обучении: obs → `FusedObservationEncoder` → `z_v`; action → `FDDRATTok.encode` → latents/tokens; Nested Dropout (K~U[1,H_l]); AR → logits + hidden; **роутер получает `.detach()` hidden states** (Decoupled Training); CRH получает `stop_gradient(a_coarse) || z_v`. Инференс: авторегрессивный цикл до H_l шагов с ранним выходом по сигналу роутера; незаполненные слоты обнуляются.

### `src/fddrat/tokenizer.py`
`FDDRATTok` — обёртка над OAT `OATTok`. Добавляет `decode_coarse(latents)` (декодер без квантизатора) и `_load_from_oat_ckpt(path)` для загрузки реального чекпоинта. Без аргументов создаёт mock (`DummyQuantizer`, `DummyDecoder`) для dry-run.

### `src/fddrat/modules/crh.py`
MLP с фиксированным входом `[stop_gradient(a_coarse) || z_v]` размером `[B, H_a*D_a + obs_dim]` = `[B, 362]`. Предсказывает непрерывный остаток `delta_a [B, H_a, D_a]`.

### `src/fddrat/modules/router.py`
`ShadowRouter` — вычисляет логиты остановки: `p_t = sigmoid(alpha * cos(q_t, k_{t-1}) - tau(z_v))`. Принимает hidden states с detach, не влияет на AR-граф.

### `src/fddrat/modules/loss.py`
`FDDRATLoss`: CE + λ·BCE(router) + β·masked_MSE(CRH). BCE форсируется в float32 даже при bf16 AMP. MSE маскируется нулём, когда K_sampled == H_l.

### `scripts/eval_fddrat_libero.py`
CLI (`click`) для оценки в LIBERO-симуляторе. Принимает `.ckpt` или директорию. Вызывает `policy.compile_decoder()` (torch.compile decoder + CRH), делает warm-up, профилирует p99 latency, сохраняет `eval_log.json` с success rate, latency и видео.

---

## Размерности (критично)

| Параметр | Значение | Откуда |
|----------|----------|--------|
| `H_l` | 8 | OAT `num_registers` |
| `H_a` | 32 | OAT `sample_horizon` (decoder output) |
| `D_a` | 7 | LIBERO action dim |
| `D_v` | 768 | AR internal dim (768 / 12 heads = 64) |
| `obs_dim` | 138 | `FusedObsEncoder`: 2×RGB camera + 10 state dims |
| Codebook | 1000 | FSQ levels=[8,5,5,5] |
| Latent dim | 4 | len(FSQ levels) |
