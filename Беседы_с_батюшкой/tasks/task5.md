1. **Status Quo**: 
- Исходная точка: Использование заглушек (`DummyDataset`, `nn.Identity` для визуального энкодера) и отсутствие интеграции с симулятором для замера метрик.
- Внедряемая дельта: Подключение `ZarrDataset` и `FusedObsEncoder` для обработки реальных мультимодальных данных (изображения + стейт), перенос вычисления $\mathcal{L}_{MSE}$ в нормализованное пространство действий, а также интеграция скрипта эвалюации с профилировщиком `p99 latency`.
- Точка интеграции: `LitDataModule` (для датасета), `FDDRATPolicy.__init__` и `forward` (для энкодера и нормализации), а также форк скрипта `eval_fddrat_libero.py` для тестирования в среде.

---

2. **Algorithmic Blueprint (Чертеж)**

- *Шаг 1:* Интеграция мультимодального потока наблюдений (Vision & State).
  - *Вход:* Словарь `obs`, содержащий тензоры камер `image` $shape: [B, T_o, C, H, W]$ и проприоцептивный стейт `state` $shape: [B, T_o, D_s]$.
  - *Операция:* Прогон через `FusedObsEncoder`. Сверточные сети (ResNet/CNN) обрабатывают пространственные размерности, объединяются со стейтом и проецируются через линейные слои в единое латентное пространство признаков.
  - *Выход:* Вектор визуально-кинематического контекста `z_v` $shape: [B, D_v]$.

- *Шаг 2:* Извлечение и нормализация целевых макро-движений.
  - *Вход:* Сырой тензор действий `actions_raw` $shape: [B, H_a, D_a]$ из `ZarrDataset`.
  - *Операция:* Применение `LinearNormalizer` (аффинное преобразование с использованием предвычисленных min/max или mean/std датасета).
  - *Выход:* Нормализованный тензор `a_target_norm` $shape: [B, H_a, D_a]$.

- *Шаг 3:* Вычисление детокенизированной траектории (Без денормализации!).
  - *Вход:* Латентные токены с наложенным Nested Dropout `latents_masked` $shape: [B, H_l, D_{lat}]$.
  - *Операция:* Прогон через `decode_coarse`. **Остановка градиента** (`detach`). 
  - *Выход:* Грубая нормализованная траектория `a_coarse_norm_detached` $shape: [B, H_a, D_a]$.

- *Шаг 4:* Инъекция ортогональных остатков (CRH) в нормализованном пространстве.
  - *Вход:* Конкатенация сплющенного `a_coarse_norm_detached` $shape: [B, H_a \times D_a]$ и `z_v` $shape: [B, D_v]$.
  - *Операция:* Прогон через 3-слойный MLP (CRH). Предсказание высокочастотной поправки.
  - *Выход:* Нормализованный вектор остатков `delta_a_norm` $shape: [B, H_a, D_a]$.

- *Шаг 5:* Сборка $\mathcal{L}_{MSE}$ таргета.
  - *Вход:* `a_target_norm`, `a_coarse_norm_detached`, `delta_a_norm` — все тензоры $shape: [B, H_a, D_a]$.
  - *Операция:* Вычисление ошибки: `MSE(delta_a_norm, a_target_norm - a_coarse_norm_detached)`. Наложение бинарной маски $K < H_l$.
  - *Выход:* Скалярный лосс $\mathcal{L}_{MSE}$ $shape: [1]$.

- *Шаг 6:* Замер Latency в закрытом цикле (Closed-Loop Inference).
  - *Вход:* Инференс-наблюдение `obs` $shape: [1, T_o, C, H, W]$.
  - *Операция:* Старт таймера `t0 = time.perf_counter()`. Выполнение `predict_action(obs)`: `a_final_norm = a_coarse_norm + delta_a_norm`, затем применение `unnormalize(a_final_norm)`. Остановка таймера `t1 = time.perf_counter()`. Запись дельты в миллисекундах.
  - *Выход:* Денормализованное действие `a_final_raw` $shape: [1, H_a, D_a]$ и скаляр `latency_ms`.

---

3. **Legacy Code Mapping**

- `oat.dataset.zarr_dataset.ZarrDataset`: Заменить `DummyDataset` в `src/core/datamodule.py`. Подключить передачу `shape_meta` из конфигурации.
- `oat.model.common.normalizer.LinearNormalizer`: Инициализировать в `FDDRATPolicy` или извлекать из `ZarrDataset`. Использовать его методы `normalize()` и `unnormalize()`.
- `oat.perception.fused_obs_encoder.FusedObsEncoder`: Скопировать логику инициализации из `OATPolicy.__init__` и заменить `nn.Identity()` в `FDDRATPolicy.__init__`. 
- `scripts/eval_policy_sim.py`: Скопировать в `scripts/eval_fddrat_libero.py`. Изменить блок загрузки модели на `LitSystem.load_from_checkpoint(ckpt_path).model`. Добавить логирование массива `latency_ms` и вычисление 99-го перцентиля (`np.percentile(latency_list, 99)`).
- `slurm/oat/train_ply_libero10.slurm`: Оставить без изменений. Указать Кодеру использовать этот скрипт как входную точку для запуска Baseline-обучения OAT-4 и OAT-8.

---

4. **Edge Cases Warning**

- **Внимание: Архитектурный сдвиг (Normalization Mismatch)**: В предыдущей итерации `CRH` оперировал в денормализованном (сыром) пространстве действий. Согласно новым вводным, `a_target` берется *до денормализации*. Кодер обязан **удалить** вызов `unnormalize` перед прогоном `a_coarse` через `CRH` в методе `forward`. Денормализация теперь должна происходить строго и только в конце метода `predict_action` (после сложения `a_coarse_norm + delta_a_norm`).
- **Внимание: Инициализация Нормализатора**: `LinearNormalizer` в OAT требует предварительной статистики (mean/std или min/max). Убедитесь, что Кодер вызывает метод загрузки статистики из `ZarrDataset` в `LitDataModule.setup()` и передает ее в `FDDRATPolicy` до начала первого forward-пасса, иначе лоссы улетят в `NaN`.
- **Внимание: Синхронизация таймеров (CUDA Sync)**: При профилировании `p99 latency` в `eval_fddrat_libero.py`, перед `t1 = time.perf_counter()` Кодер обязан вызвать `torch.cuda.synchronize()`, иначе замеры времени будут отражать лишь время постановки задачи в очередь GPU, а не реальное время вычисления (wall-clock), что исказит доказательную базу против OAT-8.

<status_quo>
1. Исходная точка: Использование заглушек (`DummyDataset`, `nn.Identity` для визуального энкодера) и отсутствие интеграции с симулятором для замера метрик.
2. Внедряемая дельта: Подключение `ZarrDataset` и `FusedObsEncoder` для обработки реальных мультимодальных данных (изображения + стейт), перенос вычисления $\mathcal{L}_{MSE}$ в нормализованное пространство действий, а также интеграция скрипта эвалюации с профилировщиком `p99 latency`.
3. Точка интеграции: `LitDataModule` (для датасета), `FDDRATPolicy.__init__` и `forward` (для энкодера и нормализации), а также форк скрипта `eval_fddrat_libero.py` для тестирования в среде.
</status_quo>