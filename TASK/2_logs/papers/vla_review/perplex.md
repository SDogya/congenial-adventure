# Токенизация в Vision‑Language‑Action моделях: от BLT и H‑Net до OAT и Oat‑VLA

## 1. Формулировка задачи

Рассматривается класс VLA‑систем, в которых наблюдение \(o\) (RGB/видео, состояние робота), языковая инструкция \(\ell\) и действия \(a\) представлены в виде единого или согласованного поточного представления \(x_{1:n}\), обрабатываемого трансформером или иерархической сетью. Задача токенизации состоит в построении (возможно обучаемых) отображений из сырого пространства модальностей в дискретное или кусочно‑дискретное пространство токенов, по которому:[^1][^2]

\[
\mathcal{T}_\text{vis}: I \mapsto V_{1:K},\quad \mathcal{T}_\ell: \ell \mapsto L_{1:J},\quad \mathcal{T}_a: a_{1:H_a} \mapsto T_{1:H_l}.
\]

Для VLA‑политик вида \(\pi_\theta(a_{1:H_a}\mid o_{1:H_o}, \ell)\) технически критичны три свойства токенизаций действий и, частично, зрения:[^3]

1. Высокая степень компрессии: \(H_l\ll H_a D_a\) при терпимой реконструкционной ошибке.
2. Тотальная декодируемость: \(\mathcal{T}_a^{-1}\) определена на любом токен‑векторе из \(\mathcal{V}^{H_l}\).
3. Каузальный порядок токенов: левосторонняя факторизация \(p(T_i\mid T_{<i}, o_{1:H_o})\) согласована с семантической сложностью действий.

Аналогично для текста и иных модальностей рассматриваются либо классические BPE/SentencePiece токенизаторы, либо байтовые модели (BLT) и иерархические сети с динамическим чанкингом (H‑Net), в которых сама операция разбиения на единицы обрабатывается моделью.[^4][^5]

Визуальная часть в современных VLA (OpenVLA, π0, RT‑серия, Oat‑VLA) обычно реализуется ViT‑подобными энкодерами, дающими \(K\sim 200\text{--}400\) патч‑токенов на кадр 224×224; при длинных контекстах это становится основной вычислительной и память‑ограничивающей компонентой.[^2][^6]

Таким образом, задача токенизации в VLA сводится к совместной оптимизации:

- байтовой/символьной токенизации для языка и кода (BLT, H‑Net);
- сжатой и структурированной токенизации непрерывных действий (FAST, OAT);
- семантически осмысленной разреженной токенизации визуального потока (Oat‑VLA и объект‑агент‑центрические схемы).[^7][^8][^9]


## 2. Ключевые методы

### 2.1. BLT: Byte Latent Transformer (патч‑уровневая байтовая токенизация)

BLT устраняет внешний текстовый токенизатор и вводит двухуровневую архитектуру: лёгкий локальный «патчер» поверх байтов и крупный латентный трансформер над динамически сформированными патчами.[^5][^10]

Исходная последовательность байтов \(b_{1:L}\) обрабатывается локальной моделью, которая выбирает патчи переменной длины \(p_k\) согласно оценке энтропии следующего байта; длина патча \(\ell_k\) растёт в предсказуемых областях и уменьшается в сложных. Каждый патч кодируется в один латентный токен \(z_k\), над последовательностью \(z_{1:K}\) работает крупный трансформер. Семантически это соответствует токенизации на уровне переменных «байтовых фраз», где единица дискретизации определяется не заранее, а по информации в данных.[^11][^12]

При FLOP‑матчинге с BPE‑моделями до 8B параметров и 4T обучающих байтов BLT демонстрирует:

- bits‑per‑byte (BPB), сопоставимый или лучше LLaMA‑подобных токенизованных трансформеров при среднем размере патча 6–8 байт; при \(\text{ps}\approx 8\) достигается ≈50% экономии FLOP на инференсе за счёт меньшего числа шагов трансформера.[^13][^12]
- Возможность одновременного увеличения размера модели и среднего патча при фиксированном бюджете FLOP, что даёт более крутые скейлинг‑кривые, чем при увеличении только числа слоёв в BPE‑модели.[^14][^5]

С точки зрения VLA, BLT предлагает архитектурный паттерн: вынести токенизацию низкоуровневых потоков (текст, байты сенсоров, лог‑файлы) в лёгкий энкодер, который динамически решает, какие фрагменты сжимать агрессивно, а какие — оставлять более детализированными. Этот подход можно механически перенести на действия или траектории в VLA, если рассматривать действия как байт‑последовательность после квантования.[^15]


### 2.2. H‑Net: динамический чанкинг и иерархическая последовательная модель

H‑Net вводит явную иерархическую U‑Net‑подобную архитектуру с динамическим чанкинг‑слоем между внешними байтовыми уровнями и внутренней «ядерной» LM. Модель состоит из:[^16][^4]

- внешних энкодеров/декодеров на базе Mamba‑2 (SSM) для обработки «сырых» последовательностей длиной \(L^0\);
- основного трансформера \(\mathcal{M}\) над сжатой последовательностью \(x^S\) длиной \(L^S\ll L^0\);
- динамического механизма Chunk/Dechunk, который обучаемо решает, какие позиции передавать внутрь и как восстанавливать полный байтовый поток.

**Chunking.** Для каждой позиции \(t\) вычисляются проекции \(q_t, k_t\) и вероятность границы

\[
 p_t = \frac{1}{2}\Bigl(1 - \frac{q_t^\top k_{t-1}}{\lVert q_t\rVert \lVert k_{t-1}\rVert}\Bigr),\quad b_t = \mathbb{1}_{\{p_t\ge 0.5\}}.
\]

По индикаторам \(b_t\) выбирается подмножество векторов для сжатой последовательности, а параллельно добавляется ratio‑loss, принуждающий среднюю долю отобранных позиций \(F = L^S/L^0\) к заданному \(1/N\) (например, \(N=6\) даёт ≈4.5–5 байт на «чанк» — сравнимо с BPE).[^4]

**Dechunking и сглаживание.** Для восстановления применяется экспоненциальное сглаживание и повторение сжатых векторов по интервалам между границами, с использованием confidence‑коэффициента \(c_t\) и Straight‑Through Estimator (STE), что даёт дифференцируемый аналог копирования токенов назад в байтовое пространство.[^4]

H‑Net демонстрирует:

- при 1‑ступенчатой иерархии (\(6\)-DC) и байтовом входе BPB и zero‑shot качество на уровне сильного BPE‑трансформера аналогичного FLOP (≈760M параметров), при этом chunks автоматически достигают длины ≈4.5–5 байт;[^4]
- при 2‑ступенчатой иерархии ((3,3)‑DC) модель превосходит токенизованный трансформер в perplexity уже после ≈30B обучающих байтов и достигает downstream‑качества, соответствующего токенизированной модели вдвое большего размера (XL‑масштаб);[^4]
- существенно более высокую робастность к символьным и шальным шумам (perturbed HellaSwag) и лучший BPB на китайском, коде и ДНК‑последовательностях (до 3.6× экономии данных).[^17][^4]

Для VLA H‑Net даёт рецепт архитектуры, где дискретизация (tokenization) заменяется внутренней динамической субдискретизацией на нескольких уровнях: от байтов сенсоров/логов до «абстрактных действий» (супер‑токенов), над которыми работает основная VLA‑политика. Это особенно релевантно при объединении сырого протокольного потока (CAN, силомоментные датчики, аудио) с изображениями и языком.[^18]


### 2.3. FAST: частотная токенизация действий для VLA

FAST (Frequency‑space Action Sequence Tokenization) решает задачу токенизации непрерывных действий \(a_{1:H}\) путём аналитического сжатия в частотном пространстве:[^9]

1. Нормализация действий по квантилям 1/99% в диапазон \([-1,1]\) для устойчивости к выбросам.
2. DCT по времени для каждого измерения действия — получаем матрицу коэффициентов \(C \in \mathbb{R}^{D\times H}\), где низкие частоты кодируют общую форму траектории, высокие — локальные изменения.
3. Масштабирование и квантование коэффициентов (параметр \(\gamma\), типично 10) → разреженная целочисленная матрица.
4. Флэттенинг с упорядочиванием «сначала низкие частоты всех измерений», после чего обучается BPE‑кодировщик (vocab ~1024) для получения компактной последовательности токенов.[^9]

Получается токенизация \(\mathcal{T}_\text{FAST}: a_{1:H}\mapsto T_{1:n}\), где \(n\) существенно меньше \(H\cdot D\); для 1‑секундных чанков:

| Датасет | Частота | Naive (биннинг) | FAST (сред.) | Сжатие |
|---------|---------|-----------------|--------------|--------|
| BridgeV2 | 5 Hz | 35 | ~20 | 1.75× |
| DROID | 15 Hz | 105 | ~29 | 3.6× |
| Bussing | 20 Hz | 140 | ~28 | 5× |
| T‑shirt fold (bi‑manual) | 50 Hz | 700 | ~53 | 13× |

[^9]

Ключевой эффект — разрыв временной корреляции между соседними токенами; next‑token‑обучение на DCT‑коэффициентах остаётся информативным даже при больших частотах (50 Hz), где обычный поэлементный биннинг приводит к вырождению политики в «копирование прошлого действия».[^9]

Экспериментально FAST+ (универсальный токенизатор, обученный на ≈1M 1‑секундных чанков множества роботов) позволяет:

- обучить π0‑FAST (autoregressive VLA на PaliGemma‑3B) на 10k часов данных и достигнуть качества, сопоставимого с π0‑flow‑matching (диффузионным VLA), при ≈5× меньших затратах на обучение;[^9]
- впервые реализовать нулевой‑шот перенос политик, обученных на DROID, в новые физические окружения (3 кампуса, новые сцены) при простом языковом промптинге, чего не достигали OpenVLA и исходная работа по DROID.[^9]

FAST сохраняет полную декодируемость (обратный DCT + инверсия BPE) и задаёт естественный порядок токенов «от грубых частот к детальным», что функционально аналогично свойству P.3 в OAT.[^9]


### 2.4. OAT: Ordered Action Tokenization

OAT формализует три требования к токенизации действий (компрессия, тотальная декодируемость, каузальный порядок) и реализует их через обучаемый автоэнкодер с трансформером и Finite Scalar Quantization.[^3]

Пусть \(a_{1:H_a}\in \mathbb{R}^{H_a\times D_a}\) — чанк действий (типично \(H_a=32\)). Токенизатор \(\mathcal{T}\) реализуется так:

1. Конкатенация \(a_{1:H_a}\) с \(H_l\) learnable register‑токенами \(r_{1:H_l}\).
2. Трансформер‑энкодер \(\mathcal{E}_\phi\) обрабатывает последовательность \((a, r)\), агрегируя временную информацию в регистры. Выход: \(z_{1:H_l}\).
3. FSQ (уровни, например, [^19][^7][^7][^7], \(H_l=8, D_l=4\)) квантует \(z\) в дискретные токены \(T_{1:H_l}\in \mathcal{V}^{H_l}\), где \(|\mathcal{V}|\approx 10^3\).[^3]
4. Декодер \(\mathcal{D}_\theta\) (трансформер) реализует \(\mathcal{T}^{-1}: T_{1:H_l}\mapsto \hat{a}_{1:H_a}\) и обучается по MSE между \(a\) и \(\hat{a}\).

**Индукция порядка.** Чтобы получить каузальный порядок (P.3), OAT использует:[^8][^3]

- nested dropout по регистрам: на каждой итерации используется случайный префикс \(K\le H_l\), хвост маскируется. Это заставляет энкодер упаковывать наиболее важную информацию в ранние регистры; декодер учится восстанавливать \(a\) по любому префиксу.
- каузальное внимание между регистрами: \(r_i\) видит только \(r_j, j\le i\) и все action‑токены, что задаёт левосторонний поток информации и согласует порядок регистров с next‑token факторизацией.[^3]

**Авто‑регрессия по токенам.** Политика факторизует распределение

\[
 p(T_{1:H_l}\mid o_{1:H_o}) = \prod_{i=1}^{H_l} p(T_i\mid T_{<i}, o_{1:H_o}),
\]

после чего любой префикс \(T_{1:K}\) декодируется в чанк действий \(\hat{a}_{1:H_a}\) через \(\mathcal{T}^{-1}\). За счёт nested dropout гарантируется, что \(\mathcal{T}^{-1}\) — тотальная функция (P.2), а каждая длина префикса соответствует валидной (хотя и разной по точности) траектории.[^3]

**Результаты.** На 20+ задачах из LIBERO, RoboMimic, MetaWorld и RoboCasa OAT‑политики превосходят биннинг, FAST и QueST; кроме того, обеспечивается монотонный рост успеха по мере увеличения длины префикса:[^3]

| Политика | LIBERO | RoboMimic | MetaWorld | RoboCasa |
|----------|--------|-----------|-----------|----------|
| Bin | 14.4±0.6 | 39.5±1.2 | 14.5±0.7 | 27.7±0.9 |
| FAST | 23.0±0.5 | 24.0±1.5 | 7.1±0.7 | 13.2±1.1 |
| QueST | 48.2±0.6 | 66.9±0.8 | 17.9±0.9 | 52.3±1.9 |
| OAT‑1 | 11.7±0.7 | 50.8±1.4 | 11.3±0.4 | 47.7±1.3 |
| OAT‑4 | 46.4±0.6 | 65.3±0.9 | 19.5±0.8 | 51.7±1.0 |
| **OAT‑8** | **56.3±1.0** | **73.1±0.5** | **24.4±0.3** | **54.6±1.1** |

[^3]

Инференс‑латентность на A100 при OAT‑8 сопоставима с QueST (≈27–31 ms), а при OAT‑1…4 в 2–3 раза ниже (10–22 ms), что даёт настраиваемый trade‑off скорость/качество.[^3]

В реальном мире на ARX‑5 OAT‑8 достигает 16/20 успехов на Pick&Place Ball и Stack Cups против 14/20 у Diffusion Policy и 11/20 у QueST.[^3]


### 2.5. Oat‑VLA: объект‑агент‑центрическая токенизация зрения

Oat‑VLA адресует визуальную сторону VLA, заменяя плотную ViT‑патч‑токенизацию (например, 256 токенов для 224×224) на 16 семантических визуальных токенов на кадр: 7 объект‑центрических и 9 агент‑центрических.[^20][^7]

**Объект‑центрические токены.** RGB‑кадр \(o\) обрабатывается энкодером (DINOv2 + SigLIP) в \(K\) патч‑векторов \(v_k\). FT‑DINOsaUR выдаёт \(N=7\) масок \(m_n\); для каждой маски собираются патчи и усредняются:[^7]

\[
 t_n = \text{avg}\{ m_n^k \odot v_k \}_{k}\,,\quad n=1..7.
\]

Это устраняет дублирование одного и того же объекта в нескольких патчах и отбрасывает фон.[^7]

**Агент‑центрические токены.** Лёгкий Faster R‑CNN на ResNet детектирует 2D‑ключевую точку хватателя; вокруг неё выбирается фиксированная сетка 3×3 патчей → 9 токенов, обеспечивающих высокочастотную информацию о контактах.[^7]

В сумме Oat‑VLA подаёт в LLM (Llama‑2) 16 визуальных токенов вместо 256; это уменьшает число визуальных токенов на 93.75%. Архитектурно это реализуется как модуль перед OpenVLA‑совместимым «проектором» в пространство LLM, что позволяет переиспользовать готовые чекпоинты OpenVLA.[^7]

**Результаты.** На LIBERO Oat‑VLA при полном fine‑tuning и LoRA демонстрирует:

- ≈2× ускорение обучения (по шагам и по wall‑clock) до заданного уровня точности action tokens и успеха задач.[^20][^7]
- При LoRA‑дообучении на LIBERO (Spatial/Object/Goal/10) достигается средний успех 78.6±0.5% против 76.5±0.6% у OpenVLA (на тех же чекпоинтах Octo и Diffusion Policy показывают 75.1% и 72.4% соответственно).[^7]
- В реальных pick&place задачах с UFactory xArm 6 Oat‑VLA достигает 72%/46% успеха (in‑distribution / out‑of‑distribution) против 52%/29% у OpenVLA, демонстрируя меньше промахов по объектам и более точные выкладки.[^20]

Абляции показывают, что:

- один «глобальный» токен (attention‑pooling) даёт заметно худшее качество (60% средний успех),
- только объект‑токены без агент‑токенов — 61.3%;
- полная схема с 7 объект‑ и 9 агент‑токенами и avg‑pooling достигает 77.1%.

[^7]


## 3. Количественный ландшафт: методы × метрики × датасеты × годы

### 3.1. Токенизация действий

| Метод | Тип токенизации | Представление | Датасеты | Год | Ключевые метрики |
|-------|-----------------|---------------|----------|-----|-------------------|
| Наивный биннинг (RT‑2, OpenVLA) | пер‑измер., пер‑шаговое квантование (256 бинов) | \(H_a D_a\) токенов | BridgeV2, RT‑1/2, Open X‑Embodiment | 2023–2024[^21][^6] | низкая сжимаемость; деградация на высоких частотах; большие латентности |
| FAST | DCT + BPE, частотное представление | ≈30 токенов/сек/манипулятор | LIBERO, DROID, реальные задачи (bussing, folding, bagging) | 2024–2025[^9] | 2–13× сжатие, успешное обучение π0‑FAST на 10k ч данных, сопоставимо с π0‑flow, до 5× быстрее обучения |
| FSQ‑VQ (альтернативы) | обучаемый VQ‑автоэнкодер | H_l×D_l | те же | 2024[^9] | чувствителен к гиперпараметрам, хуже на 50 Hz задачах |
| QueST | VQ‑latent, без порядка | 8 токенов/чанк | LIBERO, др. | 2024[^3] | высокая компрессия, но отсутствие каузального порядка → хуже OAT |
| OAT | FSQ‑latent + nested dropout + causal attn | 1–8 ordered токенов | LIBERO, RoboMimic, MetaWorld, RoboCasa; реальные ARX‑5 | 2026[^3] | при OAT‑8: 56.3% (LIBERO) vs 48.2% (QueST) и выше, монотонный рост качества по K; сходные латентности с QueST |


### 3.2. Токенизация зрения

| Метод | Токены/кадр 224×224 | Представление | Датасеты | Год | Ключевые результаты |
|-------|---------------------|--------------|----------|-----|---------------------|
| Плотный ViT‑patch | 196–256 | равномерные патчи | RT‑1/2, OpenVLA, π0 | 2023–2024[^6][^22] | compute‑bottleneck; отношение vision:language токенов ≫1 |
| TokenLearner, Perceiver Resampler, Q‑Former | 32–128 | learned downsampling | VLM/VLA | 2021–2024[^22] | уменьшение числа токенов без явной объектной структуры |
| Oat‑VLA | 16 (7 объект + 9 агент) | объект‑ и агент‑центрические | LIBERO, Open X‑Embodiment subset, реальные pick&place | 2025[^7][^20] | 93.75% сокращение визуальных токенов, >2× ускорение обучения, +2 п.п. точности относительно OpenVLA |


### 3.3. Байтовая / текстовая токенизация

| Метод | Модальность | Тип | Масштаб | Год | Результаты |
|-------|-------------|-----|---------|-----|------------|
| BPE/SentencePiece | текст | статический словарь | стандарт для LLM/VLM | 2015–2024[^23] | хрупкость к шуму, морфологиям и смешанным модальностям |
| BLT | байты | динамические патчи по энтропии | до 8B, 4T байтов | 2024–2025[^5][^10] | матчит/превосходит BPE‑LLM по BPB и задачам при ≈50% экономии FLOP на инференсе |
| H‑Net | байты (и др.) | end‑to‑end динамический чанкинг | до XL (~1.3B) | 2025[^4] | 2‑ступ. байтовый H‑Net превосходит токенизированный трансформер по BPB и downstream, 3.6× экономия данных на ДНК |


## 4. Failure‑моды и ограничения

### 4.1. BLT и H‑Net в контексте VLA

BLT и H‑Net пока продемонстрированы преимущественно на языковых и байтовых корпусах; прямые интеграции в VLA отсутствуют. Основные риски переноса:[^5][^4]

- Визуальные и action‑потоки не являются IID‑байтами; грубое байтовое представление действий может разрушить геометрию action‑пространства, что критично для стабильности робота.
- Для real‑time управления жёсткие ограничения на латентность и предсказуемость вычислительной нагрузки; динамический чанкинг H‑Net имеет переменную стоимость шага в зависимости от структуры входа.[^4]
- Обучение H‑Net требует аккуратной настройки LR‑модуляции и нормировок; ранние версии сталкивались с нестабильностью при скейлинге и multi‑stage иерархиях, что может усугубиться при одновременном присутствии vision, language и action.[^16]


### 4.2. FAST и OAT

Для FAST ключевой failure‑модой является несогласованность частотного представления с downstream‑обучением при очень сложных, нелинейных траекториях: хотя DCT хорошо сжимает гладкие сигналы, локальные контакты и удары могут требовать множества высокочастотных коэффициентов, сокращение которых ухудшает точность. Кроме того:[^9]

- использование BPE поверх DCT приводит к частичной детокенизируемости: не всякая произвольная последовательность BPE‑токенов декодируется в валидную матрицу DCT фиксированного размера, что нарушает P.2 (тотальная декодируемость) в смысле OAT, хотя на практике это редко проявляется при обучении на реальных данных;[^3][^9]
- inference‑латентность при авторегрессивной декодизации сотен токенов в секунду для нескольких манипуляторов может быть чрезмерной для high‑rate управления без дополнительных приёмов (спекулятивный декодинг, chunked‑инференс).[^9]

Для OAT основное узкое место — стоимость предварительного обучения токенизатора. Хотя сам autoencoder относительно мал (2‑слойный энкодер + 4‑слойный декодер, d=256), требуется отдельный этап обучения на больших массивах action‑данных до сходимости по реконструкции. Failure‑кейсы:[^3]

- при выключении nested dropout (OAT×) токен‑пространство становится неупорядоченным, и качество падает до 35.2% (LIBERO) против 56.3% у полноценного OAT, приближаясь к QueST;[^3]
- слишком крупный кодбук FSQ (|\(\mathcal{V}\)|>~2^11) ухудшает моделируемость: рост энтропии токенов делает next‑token предсказание значительно сложнее, несмотря на лучшую реконструкцию (LIBERO падает с 56.3% до 46.9% при увеличении \(|\mathcal{V}|\) с 1000 до 4375).[^3]

Также OAT пока исследован только в sensorimotor‑режиме без основной VLA‑надстройки (VL‑планировщик + OAT‑policy), хотя авторы прямо указывают на возможность использования OAT как промежуточного слоя действий.[^3]


### 4.3. Oat‑VLA и визуальная токенизация

Oat‑VLA использует несколько внешних модулей: FT‑DINOsaUR, Faster R‑CNN, OpenVLA; несовместимость/дрейф одного из них (например, ошибочный детект хватателя) приводит к деградации всей схемы. Ограничения:[^7]

- тестирование только на single‑arm pick&place; для би‑мануальных и сложных манипуляций (складывание одежды, инструментальные задачи) требуется обобщение агент‑центрических токенов на несколько манипуляторов и более сложные контакты;[^7]
- выигрыш по latency для batch=1 на A100 невелик (≈6%), так как время доминируется загрузкой весов LLM; преимущества проявляются при крупных батчах и во время обучения, а не при онлайновом управлении.[^7]


## 5. Открытые проблемы и направления исследований

### 5.1. Унификация байтовых и action‑токенизаций

BLT и H‑Net показывают, что end‑to‑end динамический чанкинг байтов может превосходить ручные BPE‑токенизаторы; OAT демонстрирует, что подобный подход применим к действиям. Естественно исследовать «сквозные» VLA, в которых:

- наблюдения (включая текст, код, лог‑данные, возможно — байты изображений через VQ‑кодирование),
- абстрактные action‑токены (в стиле OAT или FAST),
- низкоуровневые continuous‑декодеры

объединены в иерархический H‑Net с несколькими стадиями DC. Требуются теоретические и эмпирические работы по стабильности таких архитектур и их скейлингу на мультимодальные потоки.[^4][^3]


### 5.2. Адаптивная глубина авторегрессии по action‑токенам

OAT даёт возможность prefix‑декодирования, но число токенов K фиксируется заранее; с точки зрения информации оптимально выбирать K по сложности конкретного чанка \(a_{1:H_a}\). Не решён вопрос, как:[^3]

- оценивать онлайновую «сложность» действия (через остаточную ошибку реконструкции, энтропию policy, value‑функции),
- динамически решать, когда прекращать генерацию токенов (аналог early exit в LM) без потери стабильности.[^8]


### 5.3. Токенизация действий для VLA с диффузионными/flow‑головами

Современные VLA (π0, DexVLA, Diffusion‑VLA) используют диффузионные или flow‑matching action‑experts, которые работают с continuous‑представлением; при этом discrete токены (FAST, OAT) могут служить либо supervised‑сигналом, либо планировочным слоем. Открыт вопрос оптимальной «стыковки»:[^24][^9]

- нужно ли обучать OAT/FAST совместно с диффузионной головой и как реализовать совместный градиентный поток;
- может ли OAT служить target‑пространством для flow‑matching, обеспечивая одновременно каузальное discrete‑представление и continuous‑восстановление.


### 5.4. Объект‑центрическая токенизация для сложных сцен и 3D

Oat‑VLA использует 2D‑маски FT‑DINOsaUR и single‑view RGB; для сложных 3D‑сцен (многокамерные наблюдения, плотные глубины, point clouds) требуется обобщённая объект‑центрическая токенизация, возможно встроенная в 3D‑VLA (OccLLaMA, RVT‑2, Uni3D). Вопросы:[^22][^7]

- как задать slot‑based объектные токены, которые инвариантны к виду и совместимы с текстовыми репрезентациями;
- как совместить объект‑ и affordance‑токены (карты взаимодействий, траектории) в едином пространстве action‑токенов (в духе survey по action tokenization).[^25]


### 5.5. Теория компромисса «компрессия–моделируемость–декодируемость» для VLA

Работы по OAT и FAST показывают, что оптимум для VLA далеко не совпадает с минимумом MSE реконструкции: избыточная компрессия (FAST без BPE, слишком крупные codebooks) ухудшает моделируемость токенов, а частичная декодируемость (FAST+BPE) создаёт лонг‑тейл невалидных последовательностей. Нужна более формальная теория rate–distortion–modelability trade‑off для мультимодальных VLA, аналогичная классическим result в информации, но учитывающая autoregressive‑обучение и RL/IL‑fine‑tюнинг.[^9][^3]


## 6. Критические ссылки

1. Liu et al., **“OAT: Ordered Action Tokenization”**, arXiv:2602.04215, 2026 — вводит FSQ‑autoencoder с nested dropout и каузальным вниманием; достигает SOTA среди action‑tokenizers и демонстрирует важность упорядоченного токен‑пространства для autoregressive политик.[^3]
2. Bendikas et al., **“Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action Models”**, CoRL 2025 — Oat‑VLA; 16 визуальных токенов/кадр, 93.75% сокращение числа визуальных токенов, >2× ускорение обучения и улучшение качества относительно OpenVLA.[^7]
3. Pertsch et al., **“Efficient Action Tokenization for Vision-Language-Action Models” (FAST)**, 2024 — DCT+ВРЕ токенизация действий; FAST+ как универсальный токенизатор; показано, что компрессия временных рядов критична для высокочастотных dexterous задач.[^9]
4. Zhong et al., **“A Survey on Vision-Language-Action Models: An Action Tokenization Perspective”**, arXiv:2507.01925, 2025 — систематизирует восемь типов action‑токенов (language, code, affordance, trajectory, goal, latent, raw, reasoning) и интерпретирует VLA как цепочку модулей, производящих всё более «исполняемые» action tokens.[^25]
5. Pagnoni et al., **“Byte Latent Transformer: Patches Scale Better Than Tokens”**, ACL 2025 / arXiv:2412.09871 — вводит BLT; динамические byte‑патчи по энтропии следующего байта; при FLOP‑матчинге достигает качества BPE‑LLM с ~50% экономии FLOP и лучшими скейлинг‑кривыми.[^10][^5]
6. Hwang, Wang, Gu, **“Dynamic Chunking for End-to-End Hierarchical Sequence Modeling” (H-Net)**, arXiv:2507.07955, 2025 — иерархический U‑Net с end‑to‑end динамическим чанкингом; двух‑ступенчатый byte‑H‑Net превосходит токенизированный трансформер в perplexity и downstream задачах, улучшая робастность и эффективность данных.[^4]
7. LoHoVLA / VLA‑survey, **“Vision-Language-Action Models: Concepts, Progress, Applications and Challenges”**, 2025 — обзор архитектур и тренд в VLA, включая вопросы представления действий и токенизации.[^26][^27]
8. Knowledge Insulating VLA (π0‑CO‑VLA), **“Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better”**, 2025 — анализирует взаимодействие discrete (FAST) и continuous (flow‑matching) action‑голов и влияние на перенос семантических знаний VLM.[^24]
9. OpenVLA, **“An Open-Source Vision-Language-Action Framework”**, 2024 — baseline VLA с простым биннингом действий и плотной ViT‑токенизацией зрения; служит точкой отсчёта для Oat‑VLA и FAST‑интеграций.[^6][^22]
10. RT‑серия (RT‑1/RT‑2/RT‑X), **“RT-2: Vision-Language-Action Models”**, 2023 — ранние крупномасштабные VLA, использующие ViT‑patch токенизацию и пер‑измеренное биннирование действий, демонстрируют проблему масштабирования token‑budget при росте задач и частоты.[^21]
11. General VLA survey, **“Vision-Language-Action Models: Concepts, Progress, Applications and Challenges”**, 2025 — даёт формализацию VLA как двух‑ступенчатой архитектуры (VLM‑перцепция + action‑decoder) и подчёркивает роль токенизации как основного узкого места.[^1][^26]
12. Action tokenizer for ICIL, **“Action Tokenizer Matters in In-Context Imitation Learning”**, 2025 — LipVQ‑VAE как action‑tokenizer с липшицевым ограничением, подчёркивает важность гладкости латентного action‑пространства для стабильного выполнения.[^28]
13. Oat‑VLA supplementary analyses — подробные абляции объект‑/агент‑центрических токенов, размерностей, batch size и throughput, показывающие, что выигрыш Oat‑VLA не сводится к увеличению батча.[^20][^7]
14. FAST ablations and DROID zero‑shot experiments — демонстрируют, что BPE‑шаг необходим для уменьшения количества нулевых DCT‑коэффициентов и улучшения обучения/инференса; без него политики деградируют.[^9]
15. Additional VLA surveys (full‑stack, action‑representation centric) — подтверждают, что выбор action‑токенов (raw vs latent vs trajectory vs language) является доминирующим архитектурным решением и что discrete tokenization остаётся узким местом для масштабируемых VLA.[^29][^30]

---

## References

1. [Vision-Language-Action Models: Concepts, Progress ... - arXiv](https://arxiv.org/html/2505.04769v1)

2. [Vision-Language-Action Models - Maths, CS & AI Compendium](https://henryndubuaku.github.io/maths-cs-ai-compendium/chapter%2011:%20autonomous%20systems/03.%20vision-language-action%20models/) - An open, intuition-first textbook covering mathematics, computer science, and artificial intelligenc...

3. [[Revue de papier] A Survey on Vision-Language-Action Models: An Action Tokenization Perspective](https://www.themoonlight.io/fr/review/a-survey-on-vision-language-action-models-an-action-tokenization-perspective) - This paper presents a comprehensive survey of Vision-Language-Action (VLA) models through the lens o...

4. [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955) - Major progress on language models (LMs) in recent years has largely resulted from moving away from s...

5. [Byte Latent Transformer: Patches Scale Better Than Tokens - arXiv](https://arxiv.org/abs/2412.09871) - We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the firs...

6. [Psi-Robot/Awesome-VLA-Papers: Paper list in the survey ...](https://github.com/Psi-Robot/Awesome-VLA-Papers) - Paper list in the survey: A Survey on Vision-Language-Action Models: An Action Tokenization Perspect...

7. [LoHoVLA: A Unified Vision-Language-Action Model for Long ... - arXiv](https://arxiv.org/html/2506.00411v1)

8. [[Literature Review] OAT: Ordered Action Tokenization - Moonlight](https://www.themoonlight.io/en/review/oat-ordered-action-tokenization) - The paper introduces **Ordered Action Tokenization (OAT)**, a novel learned action tokenizer designe...

9. [[PDF] Efficient Action Tokenization for Vision-Language-Action Models](https://www.pi.website/download/fast.pdf)

10. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://aclanthology.org/2025.acl-long.453/) - Artidoro Pagnoni, Ramakanth Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Ch...

11. [Byte Latent Transformer: Patches Scale Better Than Tokens | Aalto University](https://www.aalto.fi/en/events/byte-latent-transformer-patches-scale-better-than-tokens) - LLM seminar event on Jan 15th

12. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://huggingface.co/papers/2412.09871) - Join the discussion on this paper page

13. [[PDF] Byte Latent Transformer: Patches Scale Better Than Tokens](https://aclanthology.org/2025.acl-long.453.pdf)

14. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://kingy.ai/wp-content/uploads/2024/12/Byte-Latent-Transformer-Patches-Scale-Better-Than-Tokens-Paper-Summary.pdf)

15. [Results of the summarizing process for the arXiv paper: 2412.09871v1](https://www.summarizepaper.com/en/arxiv-id/2412.09871v1/) - Easy-to-read summary of the arXiv paper 2412.09871v1 entitled Byte Latent Transformer: Patches Scale...

16. [Beyond BPE tokenization; H-Nets - Harold Benoit](https://haroldbenoit.com/notes/ml/llms/architecture/beyond-bpe-tokenization;--h-nets)

17. [Revival of Architecture Research](https://goombalab.github.io/blog/2025/hnet-future/) - Homepage of the Goomba AI Lab @ CMU MLD.

18. [SSM-07: H Nets: The End of Tokenization? Dynamic Chunking and Hierarchical Abstraction](https://www.youtube.com/watch?v=0mYQOFxGE0E) - episode: "SSM-07: H-Nets: The End of Tokenization? Dynamic Chunking and Hierarchical Abstraction"
ti...

19. [jonyzhang2023/awesome-embodied-vla-va-vln](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln) - A curated list of state-of-the-art research in embodied AI, focusing on vision-language-action (VLA)...

20. [Object-Agent-centric Tokenization for Vision Language Action ...](https://www.alphaxiv.org/overview/2509.23655) - View recent discussion. Abstract: Vision-Language-Action (VLA) models offer a pivotal approach to le...

21. [Vision-language-action model - Wikipedia](https://en.wikipedia.org/wiki/Vision-language-action_model)

22. [[Literature Review] Vision-Language-Action Models for Robotics](https://www.themoonlight.io/en/review/vision-language-action-models-for-robotics-a-review-towards-real-world-applications) - Vision-Language-Action (VLA) models are an emerging paradigm in robotics, aiming to integrate vision...

23. [The Tokenization Bottleneck in Vision-Language Models - 9JAONCLOUD](https://9jaoncloud.com/the-tokenization-bottleneck-in-vision-language-models/) - The Tokenization Bottleneck in Vision-Language Models. Is tokenization, the seemingly simple step of...

24. [[Literature Review] Knowledge Insulating Vision-Language-Action ...](https://www.themoonlight.io/en/review/knowledge-insulating-vision-language-action-models-train-fast-run-fast-generalize-better) - Vision-language-action (VLA) models adapt pre-trained vision-language models (VLMs) for robotic cont...

25. [A Survey on Vision-Language-Action Models: An Action Tokenization Perspective](https://arxiv.org/abs/2507.01925) - The remarkable advancements of vision and language foundation models in multimodal understanding, re...

26. [Vision-Language-Action Models: Concepts, Progress ... - NASA ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250504769S/abstract) - Vision-Language-Action (VLA) models mark a transformative advancement in artificial intelligence, ai...

27. [[PDF] Vision-Language-Action Models: Concepts, Progress, Applications ...](https://onlineacademiccommunity.uvic.ca/implicitassociationtestsyessir/wp-content/uploads/sites/9812/2025/12/sapkota-1.pdf)

28. [Action Tokenizer Matters in In-Context Imitation Learning - arXiv](https://arxiv.org/html/2503.01206v2)

29. [[2510.07077] Vision-Language-Action Models for Robotics - arXiv](https://arxiv.org/abs/2510.07077) - Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models...

30. [An Anatomy of Vision-Language-Action Models](https://arxiv.org/html/2512.11362v2)

