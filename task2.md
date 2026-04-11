Приветствую. Как Principal ML Tech Lead, я подготовил исчерпывающую, защищенную от ошибок спецификацию для Code Generator'а. 

Поскольку оригинальные репозитории (`oat` и `hnet`) уже находятся в корневой папке, мы не будем модифицировать их исходный код напрямую (чтобы не сломать их собственные тесты и импорты). Вместо этого мы создадим новый пакет `fddrat` внутри нашего проекта, который будет импортировать нужные модули, переопределяя и расширяя их логику. Также мы интегрируем это всё в нашу MLOps инфраструктуру (PyTorch Lightning + Hydra).

Передай эту инструкцию Code Generator'у **как есть**.

***

#  FD-DRAT: Implementation Master Plan & Code Directives

**To: Code Generator Agent**
**From: Principal ML Tech Lead**
**Objective:** Имплементация архитектуры FD-DRAT (Fixed-Dimension Decoupled Residual Action Tokenization) и её "бесшовная" интеграция в Lightning/Hydra пайплайн.

##  Архитектура директорий
Создай новую директорию `fddrat` внутри папки `src/` (чтобы следовать топологии проекта).
```text
project_root/
├── oat/           # [READ-ONLY] 
├── hnet/          # [READ-ONLY] 
└── src/
    ├── core/      # Lightning/Hydra (уже существует)
    └── fddrat/    # [TARGET] Твоя рабочая директория модели
        ├── __init__.py
        ├── modules/
        │   ├── crh.py        # Continuous Residual Head
        │   ├── router.py     # Shadow Router (из hnet)
        │   └── loss.py       # FDDRAT Loss
        ├── tokenizer.py      # Обертка над OATTok
        └── policy.py         # Главный класс FDDRATPolicy
```

---

##  ФАЗА 1: Реализация новых атомарных модулей (`src/fddrat/modules/`)

### 1.1 `crh.py`: Continuous Residual Head
Напиши класс `ContinuousResidualHead(nn.Module)`.
* **Правило статической размерности:** Модуль **не должен** использовать RNN или Attention. Только линейные слои (MLP).
* **Вход:** Конкатенация сплющенной макро-траектории и визуальных фичей.
* **Инициализация (`__init__`):** 
    * `input_dim = (H_a * D_a) + D_v`
    * `output_dim = H_a * D_a`
    * Слои: 3x `nn.Linear` с активациями `GELU` и LayerNorm. Последний слой без активации. Веса инициализировать через `trunc_normal_`.
* **Forward signature:** `def forward(self, a_coarse: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:`
    * `a_coarse` shape: `[B, H_a, D_a]`
    * `z_v` shape: `[B, D_v]`
    * **Шаги внутри:** Сделать `flatten` для `a_coarse` до `[B, H_a * D_a]`, сконкатенировать с `z_v` по `dim=1`, прогнать через MLP, сделать `reshape` обратно в `[B, H_a, D_a]`.

### 1.2 `loss.py`: Изолированная функция потерь
Напиши класс `FDDRATLoss(nn.Module)`.
* **Формула:** $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{ratio} + \beta \mathcal{L}_{MSE}$
* **Входы:** `logits`, `targets` (для CE), `p_stop`, `tau_target` (для Ratio), `delta_a`, `residual_target`, `K_sampled`, `H_l` (для MSE).
* **Критический нюанс (Маскирование MSE):** 
    ```python
    # Псевдокод для Кодера:
    mse_loss = F.mse_loss(delta_a, residual_target, reduction='none')
    # Не штрафуем, если авторегрессия дошла до конца (нет остатка)
    mask = (K_sampled < H_l).float().view(-1, 1, 1) 
    masked_mse = (mse_loss * mask).mean()
    ```

---

##  ФАЗА 2: Адаптация Legacy-модулей (`src/fddrat/`)

### 2.1 `modules/router.py`: Shadow Router
Импортируй логику из `hnet.modules.dc.RoutingModule`, но **полностью вырежи STE (Straight-Through Estimator)** и механизмы чанкинга.
* Напиши класс `ShadowRouter(nn.Module)`.
* **Инициализация:** Добавь `self.tau_mlp = nn.Linear(D_v, 1)` для адаптивного порога.
* **Forward pass:**
    * На вход приходят `q_t` и `k_{t-1}` (скрытые состояния трансформера из OAT, размерность `[B, H_l, D_attn]`) и `z_v` `[B, D_v]`.
    * Вычисли косинусное сходство по временному измерению: `cos_sim = cosine_similarity(q_t, k_{t-1})` -> `[B, H_l]`.
    * Вычисли сдвиг: `tau_shift = self.tau_mlp(z_v)` -> `[B, 1]`.
    * `p_stop = torch.sigmoid(alpha * cos_sim - tau_shift)`.
    * **Возврат:** Только тензор `p_stop` `[B, H_l]`. Никаких бинарных масок.

### 2.2 `tokenizer.py`: Токенизатор с доступом к Coarse Action
Унаследуй `FDDRATTok` от `oat.tokenizer.oat.tokenizer.OATTok`.
* **Цель:** В оригинальном OAT метод `decode` возвращает сразу финальные action. Нам нужно отделить процесс детокенизации (прогон латентов через FSQ-декодер) и вытащить $\hat{a}_{coarse}$.
* Добавь метод `decode_coarse(self, latents: torch.Tensor) -> torch.Tensor`. Он должен делать `recons = self.decoder(latents)` и возвращать непрерывную траекторию `[B, H_a, D_a]`.

---

##  ФАЗА 3: Сборка Магистрали (`src/fddrat/policy.py`)

Напиши класс `FDDRATPolicy`, наследуясь от `oat.policy.base_policy.BasePolicy`. 

### 3.1 Обучающий цикл (`forward` - Train Mode)
Напиши этот метод с абсолютной точностью:
1.  Извлеки `z_v` через `self.obs_encoder(batch['obs'])`.
2.  Закодируй $a_{target}$ (из `batch['action']`) через `action_tokenizer.encode` -> получи `latents` и `tokens`.
3.  Примени `MaskedNestedDropout` (с $K \sim U$). Получи маскированные токены $T_{1:K} \oplus \langle \text{MASK} \rangle_{K+1:H_l}$.
4.  Прогони токены через AR-модель для получения логитов (считаем $\mathcal{L}_{CE}$).
5.  Из скрытых слоев AR-модели вытащи `hidden_states`. Передай их в `self.router(hidden_states, z_v)` -> получи `p_stop`.
6.  **КРИТИЧЕСКИЙ ШАГ (CRH Integration):**
    ```python
    # 1. Детокенизация маскированного префикса
    a_coarse = self.action_tokenizer.decode_coarse(latents_masked)
    
    # 2. ИЗОЛЯЦИЯ ГРАДИЕНТОВ (Предотвращение Posterior Collapse)
    a_coarse_detached = a_coarse.detach() 
    
    # 3. Предсказание остатков
    delta_a = self.crh(a_coarse_detached, z_v)
    
    # 4. Расчет таргета для CRH
    residual_target = batch['action'] - a_coarse_detached
    ```
7.  Собери `FDDRATLoss` и верни **скалярный loss** (или словарь с ключом `'loss'`), чтобы `LitSystem.training_step` мог напрямую использовать его для `.backward()`.

### 3.2 Управление параметрами
Реализуй метод `get_optimizer_params(self)`, который вернет список словарей для конфигурации `AdamW` (например, выделит `crh` и `router` в отдельную группу параметров с другим lr или weight decay). Это нужно для делегирования из PyTorch Lightning.

### 3.3 Инференс с Early Exit (`predict_action`)
Оберни сборку ответов (early exit loop и crh pass), но **ВНИМАНИЕ**: поскольку мы обучаемся с FSDP, использование жесткого декоратора `@torch.compile` внутри исходного кода может вызывать ошибки Flatten Parameter. Оберни оптимизацию графа в отдельный опциональный метод `compile_decoder(self)`, чтобы мы могли вызывать его вручную на этапе инференса (уже после загрузки Full State Dict), а не во время FSDP-тренировки.

---

##  ФАЗА 4: MLOps Интеграция (Lightning, Hydra & Data)

Чтобы модель корректно работала в нашей инфраструктуре из ТЗ 1, сделай следующие инженерные правки:

1. **Hydra Dataclass (`src/core/config_schema.py`):**
   * Создай класс-схему `@dataclass class FDDRATConfig:` и помести туда гиперпараметры модели (`lambda_ratio`, `beta_mse`, `H_a`, `D_a`, параметры энкодера и AR-модели).
   * В главном конфиге `ExperimentConfig` обнови поле `model` на использование `FDDRATConfig` (как вариант по умолчанию).

2. **Lightning Module (`src/core/system.py`):**
   * Замени заглушку `self.model = nn.Linear(1, 1)` на инициализацию `self.model = FDDRATPolicy(cfg)`.
   * В `training_step`: прокидывай входящий `batch` внутрь `self.model(batch)`. Верни вычисленный лосс.
   * В `configure_optimizers`: инициализируй `torch.optim.AdamW`, вызывая `self.model.get_optimizer_params()` для проброса настроенных групп параметров из политики.

3. **DataModule (`src/core/datamodule.py`):**
   * Обнови `DummyDataset`. Вместо старого `{"data": 0.0, "label": 0}` датасет должен теперь генерировать фиктивные тензоры с теми ключами и формами, которые ожидает политика в методе `forward()` (например, `obs` с формой изображения/репзентации и `action` с формой `[H_a, D_a]`).
   * Размеры и параметры должны браться из `cfg`.

**УКАЗАНИЕ ДЛЯ КОДЕРА:** Выполни генерацию строго по этой инструкции, обновленному пути модулей и правилам интеграции. Это финальная, согласованная с техлидом сборка.