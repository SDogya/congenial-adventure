# Самое интересное

1. [Обзор инструментов](TASK/1/obzor.pdf) — сравнение DeepResearch, Claude Code, AutoResearch
2. [Логи общения с LLM](TASK/2_logs/) — обзоры статей, роли агентов, логи кодера и ревьювера
3. [Код и инструкция по запуску](#быстрый-старт) — ниже (для клода должен сработать файл CC_local_repr.md )
4. [Результаты eval](eval_results/) — `eval_log.json` с per-task SR и latency
5. [Отчёт (draft)](TASK/4_draft/draft.pdf)
6. https://wandb.ai/sebersehmer-nopeinc/VLA-experiment/reports/Untitled-Report--VmlldzoxNjUxMzU1MA?accessToken=dp2lkg5wxwpsk6agt7xlskcil1d121w7382tecduvuf2s8hc1x0b8290fgmkpx05 - графики с wandb, самые первые - токенизатор, стоит смотреть только на те, которые длились больше 20 минут.
---

# FD-DRAT

**Fixed-Dimension Decoupled Residual Action Tokenization** — VLA-политика управления роботом для бенчмарка [LIBERO-10](https://github.com/Lifelong-Robot-Learning/LIBERO).

Поверх [OAT](https://github.com/Chaoqi-LIU/oat)-токенизатора добавлен изолированный маршрутизатор (*Shadow Router*), который на инференсе прерывает авторегрессивную генерацию раньше `H_l` шагов. Для компенсации потери точности при раннем выходе предусмотрен *Continuous Residual Head (CRH)*, уточняющий грубую траекторию до финальной.

Три компонента потерь:
- **L_CE** — кросс-энтропия AR-декодера (основная магистраль)
- **L_ratio** — BCE роутера (обучается предсказывать целевой коэффициент сжатия)
- **L_mse** — MSE остатка CRH (маскируется, если K == H_l)

Подробнее про гипотезу и математику: [`hypotesis.md`](TASK/2_logs/hypotesis.md)  
Черновик статьи в формате Typst: [`paperdraft/fork.typ`](paperdraft/fork.typ)

---

## Быстрый старт

Полная пошаговая инструкция для **Kaggle** и **локального запуска**, включая path-ловушки и таблицу известных ошибок:

**[→ CC_local_repr.md](CC_local_repr.md)**

### Предобученные модели

В репозитории находятся два готовых чекпоинта — запустить eval можно без переобучения:

| Файл | Описание |
|------|----------|
| [`src/model/model.ckpt`](src/model/) | OAT токенизатор (~100 эпох на LIBERO-10) |
| [`src/model/eval_mod.ckpt`](src/model/) | FD-DRAT политика (~6 эпох, обучена с багом `H_l=64`) |

Подробнее: [`src/model/README.md`](src/model/README.md)

### Блокноты Kaggle

| Файл | Назначение |
|------|-----------|
| [`notebooks/should works.ipynb`](notebooks/should%20works.ipynb) | Рабочий ноутбук, на котором получены результаты из статьи |
| [`notebooks/k3.ipynb`](notebooks/k3.ipynb) | Чистая версия с `model.H_l=8` (исправлен баг) |
| [`notebooks/k1.ipynb`](notebooks/k1.ipynb) | Ранняя версия pipeline |

---

## Структура проекта

```
.
├── run.py                        # точка входа (Hydra app)
├── CC_local_repr.md              # инструкция по воспроизведению (Kaggle + локально)
├── hypotesis.md                  # математика и мотивация гипотезы FD-DRAT
├── pyproject.toml
│
├── notebooks/                    # Kaggle-ноутбуки
│   ├── should works.ipynb        # ← рабочий ноутбук с реальными результатами
│   ├── k3.ipynb                  # чистая версия (рекомендуется для новых запусков)
│   └── k1.ipynb                  # ранняя версия
│
├── paperdraft/                   # черновик статьи (Typst)
│   ├── fork.typ                  # основной текст
│   └── refs.bib
│
├── conf/                         # Hydra-конфиги
│   ├── config.yaml               # корневой конфиг
│   ├── model/baseline.yaml       # гиперпараметры FDDRATConfig
│   └── strategy/
│       ├── fsdp.yaml             # multi-GPU
│       └── single_gpu.yaml       # single-GPU / Kaggle T4
│
├── src/
│   ├── model/                    # предобученные чекпоинты
│   │   ├── model.ckpt            # OAT токенизатор
│   │   └── eval_mod.ckpt         # FD-DRAT политика
│   │
│   ├── core/                     # обучающий pipeline
│   │   ├── config_schema.py      # датаклассы конфига
│   │   ├── system.py             # LitSystem (pl.LightningModule)
│   │   └── datamodule.py         # LitDataModule + LazyZarrDataset
│   │
│   └── fddrat/                   # модель FD-DRAT
│       ├── policy.py             # FDDRATPolicy + ARModelWithHiddens
│       ├── tokenizer.py          # FDDRATTok (обёртка OATTok)
│       └── modules/
│           ├── crh.py            # ContinuousResidualHead
│           ├── router.py         # ShadowRouter
│           └── loss.py           # FDDRATLoss
│
├── scripts/
│   └── eval_fddrat_libero.py     # CLI оценки в LIBERO-симуляторе
│
├── eval_results/                 # результаты последнего eval
│   └── eval_log.json             # per-task SR, latency mean/p99
│
└── TASK/                         # материалы по заданию
    ├── 1/obzor.pdf               # обзор инструментов
    ├── 2_logs/                   # логи работы с LLM
    │   ├── papers/               # deep research по статьям (OAT, H-Net, BLT, VLA)
    │   ├── roles/                # промпты агентов (гипотезёр, кодер, ревьювер и др.)
    │   ├── tasks/                # логи задач по этапам
    │   └── NEURO_LOG/            # логи кодера и аудитора
    └── 4_draft/
        └── draft.pdf             # финальный отчёт
```

---

## Архитектура

**Точка входа:** `run.py` — Hydra-приложение, собирает конфиг, данные, модель, W&B logger, стратегию обучения и `pl.Trainer`.

**Конфиг ([`conf/`](conf/) + [`src/core/config_schema.py`](src/core/config_schema.py)):**  
Hydra загружает `config.yaml` и компонует `model/baseline.yaml` + `strategy/fsdp.yaml`.  
Схема валидируется через `ConfigStore` и датаклассы: `ExperimentConfig` → `FDDRATConfig` + `FSDPConfig`.  
Для Kaggle / single-GPU: `strategy=single_gpu`.

**Данные ([`src/core/datamodule.py`](src/core/datamodule.py)):**  
`LazyZarrDataset` — monkey-patch `ReplayBuffer.copy_from_path` → `create_from_path(mode='r')`, датасет (~25 ГБ) не грузится в RAM целиком. Нормализатор исключает RGB-ключи (их нормализует vision encoder самостоятельно).

**Обучение ([`src/core/system.py`](src/core/system.py)):**  
`LitSystem` — `pl.LightningModule`. Роутер + CRH обучаются с `lr=1e-4` без weight decay через отдельные группы параметров. Планировщик: `CosineAnnealingLR`. Порядок инициализации важен: `run.py` явно вызывает `datamodule.setup()` → `set_normalizer()` до `trainer.fit()`.

**Модель ([`src/fddrat/`](src/fddrat/)):**

| Класс | Файл | Назначение |
|-------|------|-----------|
| `FDDRATPolicy` | [`policy.py`](src/fddrat/policy.py) | Главный модуль. Forward: obs→`z_v`; action→latents/tokens; AR→logits+hiddens; router; CRH |
| `ARModelWithHiddens` | [`policy.py`](src/fddrat/policy.py) | Подкласс OAT `AutoregressiveModel`, возвращает `(logits, hidden_states)` |
| `ShadowRouter` | [`modules/router.py`](src/fddrat/modules/router.py) | `p_t = σ(α·cos(q_t, k_{t-1}) − τ(z_v))`, принимает `.detach()` hiddens |
| `ContinuousResidualHead` | [`modules/crh.py`](src/fddrat/modules/crh.py) | MLP, вход `[stop_grad(a_coarse) ‖ z_v]` = `[B, 362]`, выход `delta_a [B, 32, 7]` |
| `FDDRATLoss` | [`modules/loss.py`](src/fddrat/modules/loss.py) | CE + λ·BCE(router) + β·masked_MSE(CRH) |
| `FDDRATTok` | [`tokenizer.py`](src/fddrat/tokenizer.py) | Обёртка OATTok, добавляет `decode_coarse()`, mock-режим без аргументов |

**Инференс (`predict_action`):**  
Авторегрессивный цикл до `H_l` шагов с ранним выходом по сигналу роутера (`sigmoid > 0.5`). Незаполненные слоты **обнуляются** (ненулевое padding приводит к галлюцинациям CRH). Актуальный `H_l` читается из `decoder.latent_horizon`, а не из конфига.

---

## Ключевые параметры

| Параметр | Значение | Источник |
|----------|----------|---------|
| `H_l` | **8** | OAT `num_registers` — **передавать `model.H_l=8` при обучении**, дефолт 64 — баг |
| `H_a` | 32 | OAT `sample_horizon` (горизонт декодера) |
| `D_a` | 7 | LIBERO action dim |
| `D_v` | 768 | AR internal dim (768 / 12 heads = 64 per head) |
| `obs_dim` | 138 | `FusedObsEncoder`: 2×RGB 128×128 + 10-dim state |
| Codebook | 1000 | FSQ levels = [8, 5, 5, 5] |
| Latent dim | 4 | len(FSQ levels) |

---

## Результаты прототипа

Оценка на LIBERO-10: 50 роллаутов, BS=1, T4 GPU. Чекпоинт обучен с тремя стекающимися ограничениями: токенизатор ~100 эпох (рекомендуется ≥300), FD-DRAT ~6 из 10 эпох (прервано Kaggle), баг `H_l=64`.

**Mean SR: 6.0%** | Latency mean: 98.1 мс | p99: 310.7 мс

Подробности: [`eval_results/eval_log.json`](eval_results/eval_log.json)
