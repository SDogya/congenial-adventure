Основываясь на результатах последнего ревью и успешной интеграции фиксов (регистрация нормализатора через `nn.ModuleDict`, корректное слияние конфигураций Hydra и нулевой паддинг для защиты CRH), архитектура признана стабильной. 

Ниже представлен финальный алгоритмический чертеж (Blueprint) для Code Generator. Он жестко фиксирует математику тензоров и потоки данных для FSDP-совместимого обучения и Closed-Loop инференса в LIBERO.

***

### 1. **Status Quo**
1. **Исходная точка**: Инфраструктура обучения (PyTorch Lightning, ZarrDataset) и авторегрессионная магистраль (OAT-8 с FSQ) берутся из оригинального репозитория.
2. **Внедряемая дельта**: Полноценный пайплайн FD-DRAT. Включает теневую маршрутизацию с изоляцией градиентов, предсказание остатков в нормализованном пространстве и ранний выход (Any-Time Routing) с безопасным нулевым паддингом латентов.
3. **Точка интеграции**: Класс `FDDRATPolicy` (сохранение чекпоинтов через `nn.ModuleDict`), `LitDataModule` (иерархия Hydra-конфигов) и `eval_fddrat_libero.py` (замер p99 Latency).

---

### 2. **Algorithmic Blueprint (Чертеж)**

#### Фаза А: Обучение (Forward Pass / Decoupled Training)

- *Шаг 1:* Извлечение мультимодальных признаков.
  - *Вход:* Словарь наблюдений `obs`. Изображения $shape: [B, T_o, C, H, W]$ и стейт $shape: [B, T_o, D_s]$.
  - *Операция:* Прогон через `FusedObsEncoder`. Агрегация через CNN/ResNet и линейные проекции.
  - *Выход:* Вектор контекста `z_v` $shape: [B, D_v]$.

- *Шаг 2:* Токенизация Ground Truth траектории.
  - *Вход:* Сырая траектория `actions_gt` $shape: [B, H_a, D_a]$.
  - *Операция:* Вызов `action_tokenizer.encode()`. Внутри происходит нормализация, прогон через энкодер и дискретизация через FSQ.
  - *Выход:* Непрерывные латенты `latents` $shape: [B, H_l, D_{lat}]$ и дискретные индексы `tokens` $shape: [B, H_l]$.

- *Шаг 3:* Семплирование префикса (Nested Dropout).
  - *Вход:* `latents` $shape: [B, H_l, D_{lat}]$.
  - *Операция:* Семплирование длины $K \sim U[1, H_l]$. Замена среза $K+1 \dots H_l$ на обучаемый токен $\langle \text{MASK} \rangle$.
  - *Выход:* Маскированные латенты `latents_masked` $shape: [B, H_l, D_{lat}]$.

- *Шаг 4:* Авторегрессионный прогон магистрали.
  - *Входы:* `tokens` $shape: [B, H_l]$ и `z_v` $shape: [B, D_v]$.
  - *Операция:* Препендинг $\langle \text{BOS} \rangle$: `[B, 1] ⊕ [B, H_l] -> [B, H_l+1]`. Прогон через кастомную обертку AR-трансформера с условием `z_v`. Перехват `hidden_states` напрямую из выхода декодера (без хуков, для совместимости с FSDP).
  - *Выход:* `logits` $shape: [B, H_l+1, |V|]$, `hidden_states` $shape: [B, H_l+1, D_{emb}]$.

- *Шаг 5:* Теневая маршрутизация (Shadow Routing).
  - *Входы:* Срезы скрытых состояний $q_t =$ `hidden_states[:, 1:]` и $k_{t-1} =$ `hidden_states[:, :-1]` (оба $shape: [B, H_l, D_{emb}]$), а также `z_v` $shape: [B, D_v]$.
  - *Операция:* Вычисление косинусного сходства по $dim=-1$. Проекция $\tau(Z_v)$. Вычисление логитов: $P_{stop\_logits} = \alpha \cdot \text{cos\_sim}(q_t, k_{t-1}) - \tau(z_v)$.
  - *Выход:* `p_stop_logits` $shape: [B, H_l]$.

- *Шаг 6:* Детокенизация и изоляция градиентов (CRH Injection).
  - *Входы:* `latents_masked` $shape: [B, H_l, D_{lat}]$ и `z_v` $shape: [B, D_v]$.
  - *Операция:*
    1. Прогон через `decode_coarse` $\to$ `a_coarse_norm` $shape: [B, H_a, D_a]$.
    2. Остановка градиентов: `a_coarse_detached = a_coarse_norm.detach()`.
    3. Сплющивание в `[B, H_a \times D_a]`, конкатенация с `z_v` $\to$ `[B, (H_a \times D_a) + D_v]`.
    4. Прогон через MLP (CRH) и Reshape обратно $\to$ `[B, H_a, D_a]`.
  - *Выход:* `delta_a_norm` $shape: [B, H_a, D_a]$.

- *Шаг 7:* Сборка лосса в нормализованном пространстве.
  - *Входы:* `logits[:, :-1]`, `tokens`, `p_stop_logits`, `delta_a_norm`, `a_coarse_detached`, нормализованный `batch['action']`.
  - *Операция:* Вычисление $\mathcal{L}_{CE} + \lambda \mathcal{L}_{ratio} + \beta \mathcal{L}_{MSE}$. Наложение строгой бинарной маски $(K < H_l)$ на MSE.
  - *Выход:* `loss` $shape: [1]$.

#### Фаза B: Инференс (predict_action / Closed-Loop)

- *Шаг 1:* Any-Time Генерация.
  - *Вход:* `obs`.
  - *Операция:* Пошаговая авторегрессия. Расчет $\sigma(P_{stop\_logits})$. Если вероятность превышает `threshold`, прервать цикл (`break`).
  - *Выход:* `tokens_generated` $shape: [B, K_{exit}]$.

- *Шаг 2:* Безопасный паддинг (Zero-Padding Constraint).
  - *Вход:* `tokens_generated` $shape: [B, K_{exit}]$.
  - *Операция:* Перевод индексов в латентные эмбеддинги: `[B, K_{exit}, D_{lat}]`. Дополнение строго **нулевыми тензорами** (`torch.zeros`) до длины $H_l$: `[B, K_{exit}, D_{lat}] ⊕ [B, H_l - K_{exit}, D_{lat}] -> [B, H_l, D_{lat}]`.
  - *Выход:* `latents_padded` $shape: [B, H_l, D_{lat}]$.

- *Шаг 3:* Сложение в нормализованном пространстве и денормализация.
  - *Вход:* `latents_padded` $shape: [B, H_l, D_{lat}]$.
  - *Операция:*
    1. `a_coarse_norm = decode_coarse(latents_padded)`.
    2. `delta_a_norm = crh(a_coarse_norm, z_v)`.
    3. Сложение: `a_final_norm = a_coarse_norm + delta_a_norm`.
    4. Денормализация: `unnormalize(a_final_norm)`.
  - *Выход:* `a_final` $shape: [B, H_a, D_a]$.

---

### 3. **Legacy Code Mapping**

- **Конфигурации Hydra**: Структура OAT `task` (`libero10.yaml`) и `shape_meta` должна наследоваться в `conf/config.yaml` через механизм `defaults`.
- **Чекпоинтинг (Нормализатор)**: Внедрить `self.normalizer = nn.ModuleDict()` в `FDDRATPolicy.__init__`. При вызове метода `set_normalizer` использовать метод `.update()` для словаря, а не прямое присваивание, чтобы PyTorch Lightning смог сериализовать статистику датасета.
- **Обертка OAT Трансформера**: Оригинальный `oat.model.autoregressive.transformer` нужно обернуть в кастомный класс (например, `ARModelWithHiddens`), переопределив `forward` для возврата кортежа `(logits, hidden_states)`. Это безопасно для шардирования FSDP.

---

### 4. **Edge Cases Warning**

- **Внимание: FSDP All-Gather Deadlock**: Кодеру категорически запрещено использовать `register_forward_hook` для извлечения `hidden_states` в режиме FSDP. Разделенные по GPU веса не синхронизируются внутри хуков корректно. Обязать использовать только прямое возвращение тензоров из метода `forward`.
- **Внимание: CRH Hallucination Limit**: На этапе *Шага B2* (Инференс), если длина сгенерированной траектории $K_{exit}$ меньше $H_l$, пустой хвост эмбеддингов обязан инициализироваться строго нулями: `torch.zeros(...)`. Если использовать случайный шум или MASK-токены с плавающей арифметикой, статичная сеть `CRH` выдаст хаотичные остатки `delta_a_norm`, что приведет к катастрофическим сбоям манипулятора в симуляторе.
- **Внимание: Checkpoint Void**: Если нормализатор не будет сохранен через `ModuleDict`, `LitSystem.load_from_checkpoint` вернет политику, которая не сможет денормализовать действия. Манипулятор в симуляторе LIBERO застрянет на месте с микро-движениями порядка $10^{-4}$ рад.

<status_quo>
1. Исходная точка: Авторегрессионная магистраль OAT-8 с квантователем FSQ и иерархический маршрутизатор H-Net, интегрированные в PyTorch Lightning.
2. Внедряемая дельта: Исправленная фиксация нормализатора через `nn.ModuleDict`, внедрение `shape_meta` в Hydra-конфиги, и математически безопасное обнуление (zero-padding) латентных эмбеддингов для защиты CRH от галлюцинаций при раннем выходе (early exit).
3. Точка интеграции: Класс `FDDRATPolicy` (методы `forward` и `predict_action`), скрипт запуска `run.py` (инъекция нормализатора) и конфигурационный слой `conf/config.yaml`.
</status_quo>