
**Task:** Полный рефакторинг файла `src/fddrat/policy.py`. Необходимо удалить все заглушки (Dummy) и реализовать логику FD-DRAT строго по алгоритмическому чертежу.

**Action Items (Что нужно сделать):**

1. **Очистка и импорты:**
   - Удали `DummyEncoder` и `DummyARModel`.
   - Добавь импорты из read-only репозитория:
     `from oat.model.autoregressive.transformer import AutoregressiveModel`
     `from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout`

2. **Обновление `__init__`:**
   - Инициализируй реальный `AutoregressiveModel`. Параметры словаря возьми из квантизатора: `vocab_size = self.action_tokenizer.quantizer.codebook_size + 1`. Не забудь учесть `bos_id`.
   - Сохрани логику `register_forward_hook` для перехвата скрытых состояний, она написана верно.

3. **Переписывание `forward` (Train Mode):**
   - Убери мок токенизации. Замени на реальный вызов: `latents, tokens = self.action_tokenizer.encode(batch['action'])`.
   - Перед прогоном через `self.ar_model` добавь вектор `<BOS>` в начало `latents_masked` (размерность должна стать `[B, H_l + 1, D_lat]`).
   - Убедись, что перед передачей в CRH вызывается денормализация: `self.action_tokenizer.normalizer['action'].unnormalize(a_coarse_norm)`.

4. **Реализация `predict_action` (Inference Mode):**
   - Удали `pass`. Напиши цикл авторегрессионной генерации `for t in range(self.cfg.H_l):`.
   - Внутри цикла: прогоняй `ar_model`, извлекай логиты, бери `argmax`.
   - **Early Exit (Any-Time Routing):** На шагах `t > 0` извлекай `q_t` и `k_prev` из перехваченных `hidden_states`. Прогоняй через `self.router`. Если `torch.sigmoid(p_stop_logit) > threshold` — делай `break`.
   - После цикла добивай последовательность масками `cfg.mask_id` до фиксированной длины `H_l`.
   - Конвертируй токены обратно в `latents` через `self.action_tokenizer.quantizer.indices_to_embedding`.
   - Детокенизируй, денормализуй, пропусти через `self.crh` и верни сумму: `a_coarse + delta_a`.

**Ожидаемый результат:** Выдай полный, готовый к исполнению код для файла `src/fddrat/policy.py`.