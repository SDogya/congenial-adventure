

### 1. **Status Quo**
* **Исходная точка**: AR-магистраль на базе `OAT-8` (Nested Dropout, FSQ, Transformer) и модуль маршрутизации из `H-Net`, используемые как read-only зависимости.
* **Внедряемая дельта**: Архитектура FD-DRAT (Fixed-Dimension Decoupled Residual Action Tokenization), реализующая изоляцию градиентов (Gradient Decoupling), Any-Time Routing через сырые логиты и Continuous Residual Head (CRH) со статической размерностью.
* **Точка интеграции**: Новый мета-пакет `src/fddrat/`, инкапсулирующий модифицированную функцию потерь, обертки токенизатора с денормализацией и `FDDRATPolicy` для встраивания в PyTorch Lightning пайплайн.

---

### 2. **Algorithmic Blueprint (Чертеж)**

**Входные переменные (Global Context):**
* `actions_gt`: $shape: [B, H_a, D_a]$ (Сырая непрерывная траектория)
* `z_v`: $shape: [B, D_v]$ (Визуальные фичи от `obs_encoder`)

#### Шаг 1: OAT Encoding & Pre-processing
* *Вход:* `actions_gt` $shape: [B, H_a, D_a]$
* *Операция:* 1. Вызвать `action_tokenizer.encode(actions_gt)` для получения `latents` и дискретных `tokens`.
  2. Семплировать $K \sim U[1, H_l]$ $shape: [B]$.
  3. Применить `MaskedNestedDropout` к `latents` по маске $K$.
* *Выход:* `latents_masked` $shape: [B, H_l, D_{lat}]$, `tokens` $shape: [B, H_l]$.

#### Шаг 2: AR Forward & Hidden States Extraction
* *Входы:* `tokens` $shape: [B, H_l]$, `z_v` $shape: [B, D_v]$.
* *Операция:*
  1. Создать вектор $\langle \text{BOS} \rangle$ $shape: [B, 1]$.
  2. Конкатенация: `[B, 1] \oplus [B, H_l] \to [B, H_l + 1]`.
  3. Прогон через `AutoregressiveModel` с условием `z_v`. 
  4. Перехват скрытых состояний (через `register_forward_hook` к последнему attention-слою, так как оригинальный код OAT read-only).
* *Выход:* `logits` $shape: [B, H_l+1, |V|]$, `hidden_states` $shape: [B, H_l+1, D_{emb}]$.

#### Шаг 3: Decoupled Shadow Routing
* *Входы:* `hidden_states` $shape: [B, H_l+1, D_{emb}]$, `z_v` $shape: [B, D_v]$.
* *Операция:*
  1. Срез: $q_t =$ `hidden_states[:, 1:]` $shape: [B, H_l, D_{emb}]$.
  2. Срез: $k_{t-1} =$ `hidden_states[:, :-1]` $shape: [B, H_l, D_{emb}]$.
  3. Косинусное сходство по оси $D_{emb} \to$ `cos_sim` $shape: [B, H_l]$.
  4. Проекция $\tau(Z_v)$ через `nn.Linear` $\to$ `tau_shift` $shape: [B, 1]$.
  5. Расчет логитов: $\alpha \cdot \text{cos\_sim} - \text{tau\_shift}$.
* *Выход:* `p_stop_logits` $shape: [B, H_l]$ *(Сырые логиты, без sigmoid!)*.

#### Шаг 4: Coarse Trajectory Detachment & Denormalization
* *Вход:* `latents_masked` $shape: [B, H_l, D_{lat}]$.
* *Операция:*
  1. Прогон через OAT Decoder $\to$ `nsamples` $shape: [B, H_a, D_a]$ (нормализованные).
  2. **Денормализация:** `unnormalize(nsamples)` $\to$ `a_coarse` $shape: [B, H_a, D_a]$.
  3. Блокировка градиентов: `a_coarse.detach()`.
* *Выход:* `a_coarse_detached` $shape: [B, H_a, D_a]$.

#### Шаг 5: Continuous Residual Injection (CRH)
* *Входы:* `a_coarse_detached` $shape: [B, H_a, D_a]$, `z_v` $shape: [B, D_v]$.
* *Операция:*
  1. Flatten `a_coarse_detached` $\to [B, H_a \times D_a]$.
  2. Конкатенация: `[B, H_a \times D_a] \oplus [B, D_v] \to [B, (H_a \times D_a) + D_v]`.
  3. Прогон через 3-слойный MLP (GELU, LayerNorm).
  4. Reshape обратно в пространство действий.
* *Выход:* `delta_a` $shape: [B, H_a, D_a]$.

#### Шаг 6: Masked Loss Assembly
* *Операция:*
  1. $\mathcal{L}_{CE}$: `CrossEntropy(logits[:, :-1], tokens)`.
  2. $\mathcal{L}_{ratio}$: `BCEWithLogitsLoss(p_stop_logits, tau_target)` (где `tau_target` — тензор нулей или целевая разреженность).
  3. $\mathcal{L}_{MSE}$: Вычислить квадрат ошибки `(delta_a - (actions_gt - a_coarse_detached))^2`. 
  4. **Strict Masking:** Создать бинарную маску $M = (K < H_l) \to shape: [B]$. Сделать broadcast маски, умножить на $\mathcal{L}_{MSE}$ и вычислить сумму, разделенную на `M.sum() + eps`.
  5. $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{ratio} + \beta \mathcal{L}_{MSE}$.
* *Выход:* Скаляр `loss` $shape: [1]$.

---

### 3. **Legacy Code Mapping**

Указания по импортам для Кодера (Read-Only зона):
* **`oat.model.autoregressive.transformer.AutoregressiveModel`**: Импортировать без изменений. Настроить перехват скрытых состояний через PyTorch `register_forward_hook` в конструкторе `FDDRATPolicy`.
* **`oat.tokenizer.oat.model.token_dropout.MaskedNestedDropout`**: Использовать "as is" для генерации масок $K$.
* **`oat.tokenizer.oat.tokenizer.OATTok`**: Унаследовать и расширить в `FDDRATTok`.

---

### 4. **Edge Cases Warning (Критично для Кодера)**

⚠️ **Обязательно к исполнению при написании кода:**
1. **Router Instability (Hard Fail):** `ShadowRouter` **ОБЯЗАН** возвращать сырые логиты `[B, H_l]`. Никаких `torch.sigmoid` в forward-пассе роутера. Активация должна происходить математически безопасно внутри `F.binary_cross_entropy_with_logits` в `loss.py`.
2. **MSE Masking Collapse:** В `loss.py` категорически запрещено делать `masked_mse.mean()` по всему батчу. Маскирование должно обнулять потерю для элементов, где $K == H_l$, а усреднение — `sum() / (mask.sum() + 1e-8)`, чтобы штраф не размывался неактивными батчами.
3. **Denormalization Void:** Выход `decoder` внутри OATTok — это нормализованное пространство. Если вычесть его из сырого `batch['action']`, MSE сойдет с ума. Кодер обязан вызвать `self.normalizer['action'].unnormalize(nsamples)` перед `detach()`.
4. **AR Off-by-One Error:** Авторегрессионный прогон требует добавления `bos_id` в нулевую позицию $H_l$ токенов. Шейпы изменятся с `[B, H_l]` на `[B, H_l+1]`. В `CrossEntropy` необходимо передавать логиты со сдвигом `logits[:, :-1]`.
5. **Read-Only Enforcement:** Не менять исходники в папках `oat/` и `hnet/`. Все инъекции (например, добыча `hidden_states`) делать через хуки в `policy.py`.