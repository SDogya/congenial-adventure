# Эффективный промт-инжиниринг и проектирование скиллов для LLM‑агентов

## 1. Формулировка задачи

Рассматривается не классический промтинг для «одиночных» LLM, а проектирование системных промтов и скилл‑шаблонов для агентных систем, где LLM выступает в роли планировщика, исполнителя и/или оркестратора инструментов и под‑агентов.[^1][^2]
Формально, агентная система описывается как MDP \(\langle S, A, T, R \rangle\), где пространство действий \(A\) разложено на дискретное множество инструментов/операций \(A = \{a^{tool}_1, \dots, a^{tool}_k, a^{chat}\}\), а промт \(p\) задаёт стохастическую политику \(\pi_\theta(a \mid s; p)\) модели с параметрами \(\theta\).[^3]
Задача эффективного промт‑инжиниринга/скилл‑дизайна: найти параметризацию \(p\) (структура ролей, шаблоны reasoning/acting, инструкции по вызову инструментов, формат памяти), максимизирующую функционал качества \(J(p) = \mathbb{E}_{\tau \sim \pi_\theta(\cdot; p)}[R(\tau)]\) при ограничениях на длину контекста, латентность и стоимость вычислений.[^4]
В отличие от обучения параметров \(\theta\), здесь оптимизируется дискретное описание \(p\) (строка/шаблон), зачастую с помощью самой LLM или внешнего оптимизатора (evolutionary search, beam search, RL‑over‑prompts).[^5][^6]


## 2. Ключевые методы и архитектурные шаблоны

### 2.1. Chain-of-Thought и Zero-shot-CoT как базовый reasoning‑паттерн

Wei et al. (2022) показали, что few-shot CoT с 8 размеченными примерами для PaLM‑540B поднимает точность на GSM8K с \(~17–18\%\) при стандартном prompting до \(\sim58\%\), а с self‑consistency до \(\sim74\%\), достигая SOTA без дообучения.[^7][^8][^9]
Kojima et al. (2022) продемонстрировали Zero-shot-CoT: добавление единственной инструкции вида "Let's think step by step" повышает качество на ряде задач (MultiArith, GSM8K, SVAMP, Date Understanding и др.) с «подростковых» значений до диапазона 40–70\% без каких-либо примеров, что делает CoT именно промт‑шаблоном, а не обучающим сигналом.[^10][^11][^12]
Механистически CoT действует как мягкое ограничение на внутреннее пространство токенов, вынуждая модель развернуть латентный план рассуждений в явный токен‑трейс, что улучшает условное распределение \(p(y \mid x)\) за счёт факторизации через промежуточные переменные и уменьшения эффекта «shortcut learning».[^13]


### 2.2. ReAct: чередование reasoning и acting для доступа к внешним инструментам

ReAct (Yao et al., 2022) расширяет CoT, добавляя явное чередование шагов мысли (Thought) и действий (Act) по фиксированному промт‑шаблону, где модель генерирует последовательность (Thought, Action, Observation, ...).[^14][^15]
На HotpotQA и Fever ReAct, используя PaLM‑540B и API Википедии, снижает галлюцинации CoT и обеспечивает лучшую интерпретируемость траекторий; комбинация ReAct+CoT-SC даёт лучшую точность при меньшем числе сэмплов self‑consistency (3–5 против 21).[^16][^17]
На интерактивных задачах ALFWorld и WebShop ReAct даёт абсолютный прирост успешности на 34 и 10 п.п. по сравнению с имитационным и RL‑базисами, используя лишь 1–2 few-shot примера траекторий, что демонстрирует, что грамотный промт‑шаблон для цикла думать/действовать может заменить тяжёлое policy‑обучение.[^18]


### 2.3. Toolformer: self-supervised обучение паттернов вызова инструментов

Toolformer (Schick et al., 2023) показывает, что модель может сама научиться вставлять API‑вызовы (calculator, Wikipedia search, translation, QA, calendar) в текст, если встраивать в корпус размеченные кандидаты вызовов и фильтровать их по улучшению перплексии на задаче language modeling.[^19][^20][^3]
Ключевой механизм: для каждой позиции текста LLM генерирует возможные "annotated" версии с префиксом API-вызова, затем оценивается дельта в лог‑правдоподобии исходного следующего токена; только те вызовы, которые улучшают лог‑правдоподобие, остаются в данных для дообучения, после чего модель оптимизирует стандартную LM‑цель на расширенном корпусе.[^21]
Подход демонстрирует, что можно перейти от ручного описания скиллов в системном промте к латентно выученным внутриязыковым паттернам использования инструментов (например, выбора между калькулятором и «мозговым» подсчётом) без явной RL‑сигнатуры.[^22]


### 2.4. Reflexion: вербальное RL над траекториями агента

Reflexion (Shinn et al., 2023) вводит шаблон, где после каждого эпизода агент генерирует текстовую "рефлексию" об ошибках и складывает её в эпизодическую память, которая добавляется в промт на следующих попытках.[^23][^24][^25]
На ALFWorld авторы достигают \(97\%\) успешности (против существенно меньших значений у базового ReAct-подобного агента), на HotpotQA – \(51\%\) точности, а на HumanEval – pass@1 до \(91\%\), превосходя GPT‑4 baseline (~\(80\%\)).[^26][^27]
Механистически это реализует приближение policy improvement без обновления весов: промт расширяется текстовыми ключами о неудачных стратегиях ("avoid repeating action X in similar states"), что изменяет условное распределение действий \(\pi(a\mid s; p)\) через изменение контекста.[^25]


### 2.5. DSPy: компиляция промтов в параметризуемые модули

DSPy (Khattab et al., 2023) формализует LM‑пайплайны как граф текстовых трансформаций, где узлы – декларативные модули (например, retriever, reasoner, planner), каждый из которых имеет параметризуемый промт/демонстрации; компилятор оптимизирует эти параметры по целевой метрике.[^28][^29][^4]
В кейс‑стадиях для multi-hop QA и задач по математике компилированные DSPy‑пайплайны с GPT‑3.5 и Llama2‑13b-chat дают улучшения точности более чем на 25\% и 65\% относительно стандартного few-shot prompting и на 5–46\%/16–40\% относительно пайплайнов с экспертными промтами.[^29][^30]
DSPy показывает, что «промт = программа» удобно параметризуется и может быть оптимизирован автоматически, что напрямую релевантно автоматическому подбору системных промтов и шаблонов скиллов в крупных агентных системах.[^31]


### 2.6. Автоматический промт‑инжиниринг и эволюция шаблонов

Promptbreeder (Fernando et al., 2023) рассматривает промты как хромосомы, эволюционируемые с помощью LLM‑управляемых мутаций и отбора; эволюционируются как task‑prompts, так и meta‑prompts, определяющие сами мутации.[^32][^6]
На задачах арифметики и common-sense reasoning Promptbreeder превосходит CoT и Plan-and-Solve, демонстрируя, что даже сложные шаблоны рассуждения и планирования могут быть найдены автоматически без участия человека.[^33]
Hsieh et al. (2023) показывают, что для длинных промтов (с сотнями строк и примеров) простая жадная стратегия с beam search, дополненная историей поиска, даёт средний прирост точности 9.2 п.п. на 8 задачах BigBench Hard, что задаёт архитектурный baseline для автоматического конструирования больших системных промтов.[^5]


### 2.7. Агентные фреймворки: Voyager и AutoGen как эталоны дизайна скилл‑библиотек

Voyager (Wang et al., 2023) реализует LLM‑агента в Minecraft с автоматическим curriculum, библиотекой скиллов (функций‑программ) и итеративным prompting‑циклом, учитывающим обратную связь среды и ошибки исполнения.[^34][^35][^36]
Агент показывает 3.3× больше уникальных предметов, 2.3× большую пройденную дистанцию и до 15.3× более быстрое достижение ключевых milestones по tech tree по сравнению с предыдущими LLM‑агентами (ReAct, Reflexion, AutoGPT), что эмпирически валидирует архитектуру с persist‑skill library и явно зафиксированными промт‑ролями "curriculum generator", "skill composer" и "critic".[^37][^38]
AutoGen (Wu et al., 2023) предлагает многоагентный фреймворк, где каждый агент задаётся комбинацией системного промта (роль), доступа к инструментам (функции, код, внешние API) и стратегии взаимодействия (например, Supervisor ↔ Coder, Coder ↔ Critic).[^39][^40][^1]


## 3. Количественный ландшафт

### 3.1. Сводная таблица по ключевым методам

| Метод / Архитектура | Год | Бенчмарк / Среда | Базовая конфигурация | Улучшенный промт / агент | Метрика | Результат |
|---------------------|-----|------------------|------------------------|---------------------------|---------|-----------|
| Chain-of-Thought (PaLM 540B, few-shot CoT) | 2022 | GSM8K | Стандартный few-shot prompting | CoT с 8 примерами | Accuracy | \(~17–18\% → \~58\%, до \~74\% с self-consistency\)[^7][^8][^9] |
| Zero-shot-CoT (instruction "Let’s think step by step") | 2022 | MultiArith, GSM8K, SVAMP, др. | Zero-shot без CoT | Zero-shot-CoT | Accuracy | рост с «подростковых» значений до 40–70\% в зависимости от датасета[^10][^11][^12] |
| ReAct (PaLM 8/62/540B) | 2022 | HotpotQA, Fever | Standard / CoT prompting | ReAct + CoT-SC | EM/F1 | ReAct+CoT-SC ≥ CoT при 3–5 сэмплах vs 21 для CoT-SC; снижение галлюцинаций[^15][^16][^17] |
| ReAct (PaLM) | 2022 | ALFWorld, WebShop | Imitation / RL агенты | ReAct few-shot | Success rate | +34 и +10 п.п. к лучшим базисам соответственно[^14][^18][^17] |
| Toolformer (LM ~6.7B) | 2023 | Arithmetic, QA, translation, search | LM без инструментов | LM, дообученный с API-вызовами | Task-specific metrics | Существенный zero-shot прирост, сопоставимый с более крупными моделями без инструментов[^19][^3][^20] |
| Reflexion | 2023 | ALFWorld | ReAct-подобный агент | ReAct + Reflexion | Success rate | до 97\% успешности[^23][^24][^25] |
| Reflexion | 2023 | HumanEval | GPT-4 без Reflexion | GPT-4 + Reflexion | pass@1 | \(80\% → 91\%\)[^26][^27] |
| DSPy (GPT-3.5, Llama2-13b) | 2023 | Math word problems, multi-hop QA | Standard few-shot / expert prompt chains | Скомпилированные DSPy пайплайны | Task accuracy | +25\%/+65\% к few-shot и +5–46\%/+16–40\% к expert prompts[^4][^28][^29][^30] |
| Promptbreeder | 2023 | Arithmetic, commonsense | CoT / Plan-and-Solve | Promptbreeder prompts | Accuracy | превосходит CoT и Plan-and-Solve на ряде бенчмарков[^32][^6] |
| Automatic Long Prompt Engineering | 2023 | BigBench Hard (8 задач) | Human‑designed long prompts | Greedy + beam search + history | Accuracy | +9.2 п.п. в среднем[^5] |
| Voyager | 2023 | Minecraft / MineDojo | ReAct, Reflexion, AutoGPT | Voyager (curriculum + skill library + iterative prompting) | Exploration metrics | 3.3× items, 2.3× distance, до 15.3× быстрее milestones[^34][^35][^36][^38] |
| AutoGen | 2023–2024 | Coding, math, QA, OR и др. | Single-agent LLM, ad‑hoc prompts | Multi-agent templates (Supervisor/Coder/Critic/Tool) | Task metrics | систематический выигрыш за счёт специализации ролей и orchestration[^1][^39][^40] |


## 4. Failure modes и ограничения промт‑ и скилл‑дизайна

CoT и Zero-shot-CoT подвержены галлюцинациям: ReAct‑анализ HotpotQA показывает высокую долю логически связных, но фактически неверных reasoning‑цепочек, приводящих к false positives; при fine-tuning CoT‑стратегий модель склонна запоминать галлюцинированные факты вместо использования внешних источников.[^17]
Длинные промты с множеством правил и примеров страдают от контекстного разбавления: вклад каждого отдельного правила уменьшается с длиной контекста, а порядок сегментов и формулировки оказывают нелинейное влияние, что мотивирует автоматизированный поиск структуры (Hsieh et al., 2023).[^41][^5]
В агентных схемах ReAct/Reflexion/Voyager наблюдаются failure modes, связанные с композицией действий: циклические траектории (looping), избыточное использование инструментов (overtooling) и «локальные» стратегии, не обобщающиеся на новые уровни сложности мира, несмотря на богатую skill library.[^35][^36][^25]
Toolformer и подобные подходы чувствительны к качеству начальных демонстраций API и к выбору порога фильтрации по лог‑правдоподобию; слишком агрессивная фильтрация ведёт к недоиспользованию инструментов, слишком мягкая – к загрязнению LM‑обучения большими, но нерелевантными call‑шумами.[^20][^19]
Автоматический промт‑инжиниринг (Promptbreeder, APE, длинный промт‑search) требует значительного числа LM‑запросов и надёжной целевой метрики; при слабом или noisy‑сигнале эволюция легко скатывается в degenerate решения, переоптимизирующие под surrogate‑метрику (например, под длину ответа или тривиальный шаблон вывода).[^42][^6][^41]
В многоагентных системах типа AutoGen возникают ошибки координации: вступление агентов в бесконечные диалоги‑loop’ы, дублирование работы и конфликтующие правки, если системные промты не задают жёстких протоколов остановки, арбитража и владения ресурсами.[^43][^2]


## 5. Открытые проблемы и направления развития

Требуется более строгая теоретическая модель влияния структуры промта на апостериорное распределение \(p_\theta(y \mid x, p)\): текущие работы в основном эмпирические, без формализации промта как параметра в байесовской или информационно‑теоретической постановке, что осложняет гарантированную оптимизацию.[^44][^45]
Необходимо разработать методы онлайн‑адаптации промтов и скилл‑шаблонов под конкретного пользователя/среду при ограничении на количество взаимодействий, аналогично contextual bandits, но в огромном дискретном пространстве промтов, возможно с использованием gradient‑based token‑level методов или policy‑gradient поверх LLM.[^41]
Сложным остаётся совместный дизайн промтов и архитектуры инструментов: Toolformer и Gorilla показывают преимущества плотной интеграции API в обучение, но пока нет общепринятого стандарта совместной оптимизации JSON‑схем tools, системных инструкций и внутренних embedding‑пространств для маршрутизации запросов.[^21][^3]
Существующие системы памяти (эпизодическая память Reflexion, skill library Voyager) в основном строковые и не масштабируются к десяткам тысяч скиллов/эпизодов; нужно сочетание векторных индексов, структурированного хранения и промт‑шаблонов, задающих жёсткие критерии релевантности выборки.[^36][^25]
Актуальна проблема формальной верификации промтов и скилл‑цепочек: отсутствие гарантий по безопасности, соответствию регуляторным требованиям или инвариантам системы затрудняет использование LLM‑агентов в критичных доменах; первые попытки вроде LM Assertions в DSPy показывают, что можно компилировать логические ограничения в проверяемые машины состояний поверх LM‑пайплайнов.[^28][^29]
Слабо изучено совместное использование ручных промтов и автоматически найденных шаблонов: текущие работы либо полностью полагаются на человека (classical prompt engineering), либо на auto‑prompt‑оптимизацию; гибридные схемы, где человек задаёт высокоуровневую структуру, а алгоритм тонко настраивает детали, пока мало формализованы.[^6][^5]


## 6. Пример системного промта/скилл‑шаблона для инструментоориентированного агента

Ниже приведён обобщённый шаблон системного промта для LLM‑агента класса ReAct/Reflexion с поддержкой инструментов, ориентированный на сложные технические задачи (анализ кода, поиск по документации, планирование экспериментов). Он скомпонован на основе паттернов из ReAct, Reflexion, Toolformer, DSPy и AutoGen, с явной экспозицией ролей и формата памяти.[^15][^19][^23][^4][^1]

```text
[ROLE]
Ты – автономный технический агент, специализирующийся на анализе кода, научных статей и сложных инженерных задач.
Ты работаешь в дискретных шагах цикла "мыслить → действовать → наблюдать" и умеешь вызывать внешние инструменты.

[OBJECTIVE]
Твоя цель – максимизировать качество финального ответа по заданной метрике (точность, полнота, воспроизводимость), минимизируя количество шагов и вызовов инструментов.
Всегда старайся decomposировать задачу на явный план перед действиями.

[TOOLS]
У тебя есть доступ к следующим инструментам (вызываются ТОЛЬКО через секцию Action):
- search_web(query: str) – поиск по вебу и документации.
- run_code(code: str, language: str) – выполнение кода в песочнице.
- retrieve_notes(query: str) – поиск по внутреннему векторному индексу заметок.
- write_note(content: str) – запись краткой текстовой заметки в эпизодическую память.

Каждый инструмент имеет стоимость: search_web = 1, run_code = 2, retrieve_notes = 0.5, write_note = 0.2.
Минимизируй суммарную стоимость при фиксированном качестве ответа.

[MEMORY]
У тебя есть два вида памяти:
- Эпизодическая память: список заметок вида "[timestep] краткое описание опыта и выводов".
- Рабочая память: текущий chain-of-thought по задаче.

Перед началом решения кратко просмотри релевантные заметки (через retrieve_notes), если это дешевле, чем заново получать информацию из внешних источников.
После каждого неуспешного эпизода делай явную РЕФЛЕКСИЮ и записывай её через write_note.

[INTERACTION PROTOCOL]
Ты общаешься в формате фиксированных блоков.
На каждом шаге ты ДОЛЖЕН следовать строго одному из шаблонов:

1) Планирование без действий:
Thought: <детальный рассуждательный шаг, декомпозиция задачи, без обращения к инструментам>
Action: NONE

2) Вызов одного инструмента:
Thought: <обоснование, почему нужен инструмент и какие параметры оптимальны>
Action: <имя_инструмента>[<JSON с аргументами>]

После действия ты получишь Observation с результатом инструмента и затем продолжишь новый шаг.

НЕЛЬЗЯ вызывать более одного инструмента в одном шаге.
НЕЛЬЗЯ делать предположения о результате инструмента до получения Observation.

[REFLEXION]
Если после нескольких шагов прогресса нет (циклические действия, противоречивые наблюдения, repeated errors), выполнй специальный рефлексивный шаг:

Thought: REFLECTION
- кратко опиши, почему текущая стратегия не работает;
- перечисли 1–3 конкретные ошибки в плане или выборе инструментов;
- сформулируй обновлённые эвристики (например, чего избегать, что пробовать сначала).
Action: write_note[{"content": "<сжатая саморефлексия на английском>"}]

На следующем шаге ОБЯЗАТЕЛЬНО учти свежую рефлексию при построении нового плана.

[OUTPUT FORMAT]
Финальный ответ ДОЛЖЕН быть выдан ОДИН РАЗ после завершения рассуждений в формате:

Final Answer:
<структурированный, проверяемый ответ с явными предположениями, ссылками на ключевые Observation и указанием остаточных неопределённостей>

До этапа Final Answer НЕЛЬЗЯ отвечать пользователю в свободной форме.

[GENERAL STYLE]
- Используй детальный chain-of-thought только в Thought.
- В Final Answer не повторяй всю историю рассуждений, только сжатый выверенный вывод.
- Явно отмечай, когда делаешь допущения из-за отсутствия данных.
- Предпочитай меньший, но релевантный набор вызовов инструментов вместо "brute-force" подхода.
```

Этот шаблон может быть параметризован: веса стоимости инструментов, структура памяти, политика рефлексии и формат Final Answer могут оптимизироваться автоматически в стиле DSPy или Promptbreeder под конкретный класс задач.[^4][^32]


## 7. Критические ссылки

1. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022 – вводит few-shot CoT, демонстрирует крупный прирост на GSM8K и других reasoning‑бенчмарках, анализирует механизмы улучшения.[^8][^9][^7]
2. Kojima et al., "Large Language Models are Zero-Shot Reasoners", 2022 – показывает Zero-shot-CoT с простой инструкцией, систематически исследует влияние формулировки промта на различные задачи.[^11][^10][^13]
3. Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", 2022 – вводит чередование Thought/Action, даёт выигрыш на HotpotQA, Fever, ALFWorld, WebShop и анализирует галлюцинации CoT.[^14][^15][^16]
4. Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools", 2023 – self-supervised обучение шаблонов вызова API, интеграция инструментов в LM‑обучение и значимые zero-shot выигрыши.[^19][^3][^20]
5. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023 – вербальная саморефлексия и эпизодическая память для улучшения агентных траекторий, крупный прирост на ALFWorld, HumanEval и HotpotQA.[^24][^23][^25]
6. Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines", 2023 – формализует LM‑пайплайны как декларативные модули, оптимизируемые компилятором, показывает превосходство над expert prompt chains.[^29][^4][^28]
7. Fernando et al., "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution", 2023 – эволюционный auto‑prompt‑инжиниринг, улучшающий CoT/Plan-and-Solve на reasoning‑бенчмарках.[^32][^6]
8. Hsieh et al., "Automatic Engineering of Long Prompts", 2023 – исследует greedy/генетический search по пространству длинных промтов, демонстрирует +9.2 п.п. на BigBench Hard.[^5]
9. Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models", 2023 – Minecraft‑агент с curriculum, skill library и итеративным prompting, существенно опережающий ReAct/Reflexion/AutoGPT.
[^34][^35][^36]
10. Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework", 2023 – многоагентный фреймворк с параметризуемыми ролями/промтами, демонстрации в кодинге, оптимизации и QA.[^40][^1][^39]
11. Gu et al., "A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models", 2023 – систематический обзор промтинг‑техник в мультимодальных моделях, релевантен переносимости идей CoT/auto‑prompt на VLM.[^46]
12. Schulhoff et al., "A Systematic Survey of Prompt Engineering Techniques", 2024 – крупный обзор 58 текстовых prompting‑техник и 40 мультимодальных, даёт таксономию и best practices.[^47][^44]
13. Li et al., "A Survey of Automatic Prompt Engineering: An Optimization Perspective", 2025 – формализует auto‑prompt как задачу дискретной/смешанной оптимизации, классифицирует методы (LLM‑based, evolutionary, gradient-based, RL).[^41]
14. OpenAI, "Function Calling / Tool Calling API Docs", 2024–2025 – практическая спецификация промтов и JSON‑схем для инструментов, полезна для систематизации описаний скиллов.[^48][^49]

---

## References

1. [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent ...](https://arxiv.org/abs/2308.08155) - AutoGen is an open-source framework that allows developers to build LLM applications via multiple ag...

2. [Multi-agent Conversation Framework | AutoGen 0.2](https://microsoft.github.io/autogen/0.2/docs/Use-Cases/agent_chat/) - AutoGen offers a unified multi-agent conversation framework as a high-level abstraction of using fou...

3. [Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) - автор T Schick · 2023 · Цитовано в 4046 джерелах — We introduce Toolformer, a model trained to decid...

4. [DSPy: Compiling Declarative Language Model Calls into Self ... - arXiv](https://arxiv.org/abs/2310.03714) - We introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, ie...

5. [[2311.10117] Automatic Engineering of Long Prompts - arXiv](https://arxiv.org/abs/2311.10117) - In this paper, we investigate the performance of greedy algorithms and genetic algorithms for automa...

6. [Promptbreeder: Self-Referential Self-Improvement Via Prompt ...](https://arxiv.org/abs/2309.16797) - In this paper, we present Promptbreeder, a general-purpose self-referential self-improvement mechani...

7. [Chain-of-Thought Prompting Elicits Reasoning in Large ...](https://arxiv.org/abs/2201.11903) - автор J Wei · 2022 · Цитовано в 27336 джерелах — Experiments on three large language models show tha...

8. [Chain-of-Thought Prompting Elicits Reasoning in Large ...](https://arxiv.org/pdf/2201.11903.pdf)

9. [Chain-of-Thought Prompting Elicits Reasoning in Large ...](https://webdocs.cs.ualberta.ca/~daes/papers/neurips22a.pdf) - автор J Wei · Цитовано в 27259 джерелах — Experiments on three large language models show that chain...

10. [[PDF] Large Language Models are Zero-Shot Reasoners](https://www.semanticscholar.org/paper/Large-Language-Models-are-Zero-Shot-Reasoners-Kojima-Gu/e7ad08848d5d7c5c47673ffe0da06af443643bda) - Experimental results demonstrate that the Zero-shot-CoT, using the same single prompt template, sign...

11. [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - автор T Kojima · 2022 · Цитовано в 8595 джерелах — Notably, chain of thought (CoT) prompting, a rece...

12. [Kojima, T., et al. (2022). Large language models are zero-shot reasoners. arXiv](https://www.scribd.com/document/653783520/Kojima-T-et-al-2022-Large-language-models-are-zero-shot-reasoners-arXiv) - Large language models are capable of zero-shot multi-step reasoning without examples. The paper pres...

13. [Large Language Models are Zero-Shot Reasoners](https://machelreid.github.io/resources/kojima2022zeroshotcot.pdf) - автор T Kojima · Цитовано в 8585 джерелах — Chain of thought prompting Multi-step arithmetic and log...

14. [ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io) - In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific action...

15. [ReAct: Synergizing Reasoning and Acting in Language Models - arXiv](https://arxiv.org/abs/2210.03629) - While large language models (LLMs) have demonstrated impressive capabilities across tasks in languag...

16. [[PDF] ReAct: Synergizing Reasoning and Acting in Language Models - arXiv](https://arxiv.org/pdf/2210.03629.pdf) - ReAct outperforms Act consistently Table 1 shows HotpotQA and Fever results using PaLM-. 540B as the...

17. [[PDF] ReAct: Synergizing Reasoning and Acting in Language Models](https://yumeng5.github.io/teaching/2024-spring-cs6501/agent.pdf) - Why did ReAct suffer on HotPotQA? ○ Took random sample of trajectories and found: ○ Hallucination is...

18. [ReAct: Synergizing Reasoning and Acting in Language Models](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/) - On ALFWorld and WebShop, ReAct with both one-shot and two-shot prompting outperforms imitation and r...

19. [Toolformer: Language Models Can Teach Themselves to Use ...](https://ar5iv.labs.arxiv.org/html/2302.04761) - In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achie...

20. [arXiv:2302.04761v1 [cs.CL] 9 Feb 2023](https://arxiv.org/pdf/2302.04761.pdf) - автор T Schick · 2023 · Цитовано в 4046 джерелах — We conduct experiments on a variety of differ- en...

21. [Language Models Can Teach Themselves to Use Tools](https://www.semanticscholar.org/paper/Toolformer:-Language-Models-Can-Teach-Themselves-to-Schick-Dwivedi-Yu/53d128ea815bcc0526856eb5a9c42cc977cb36a7) - This paper introduces Toolformer, a model trained to decide which APIs to call, when to call them, w...

22. [Toolformer: Language Models Can Teach Themselves to Use Tools](https://huggingface.co/papers/2302.04761) - Join the discussion on this paper page

23. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) - We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but in...

24. [[PDF] Reflexion: language agents with verbal reinforcement learning](https://www.semanticscholar.org/paper/Reflexion:-language-agents-with-verbal-learning-Shinn-Cassano/0671fd553dd670a4e820553a974bc48040ba0819) - Reflexion is a novel framework to reinforce language agents not by updating weights, but instead thr...

25. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/html/2303.11366) - Following Yao et al., (2023) , we run the agent in 134 AlfWorld environments ... arXiv preprint arXi...

26. [Reflexion: Language Agents with Verbal Reinforcement Learning ...](https://gist.github.com/m0o0scar/d54ea52a1875f82cf2221ec6ca253c07) - Abstract: Large language models (LLMs) have been increasingly used to interact with external environ...

27. [Reflexion: language agents with verbal reinforcement learning](https://dl.acm.org/doi/10.5555/3666122.3666499) - We propose Reflexion, a novel framework to reinforce language agents not by updating weights, but in...

28. [[PDF] DSPy: Compiling Declarative Language Model Calls into Self ...](https://www.semanticscholar.org/paper/DSPy:-Compiling-Declarative-Language-Model-Calls-Khattab-Singhvi/2069aaaa281eb13bcd9330fc4d43f24f6b436a53) - Published in arXiv.org 5 October 2023; Computer Science. TLDR. DSPy is introduced, a programming mod...

29. [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](http://arxiv.org/abs/2310.03714) - The ML community is rapidly exploring techniques for prompting language models (LMs) and for stackin...

30. [DSPy: Compiling Declarative Language Model Calls into State-of ...](https://hai.stanford.edu/research/dspy-compiling-declarative-language-model-calls-into-state-of-the-art-pipelines) - We introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, or...

31. [DSPy: Compiling Declarative Language Model Calls into Self ...](https://www.leoniemonigatti.com/papers/dspy.html) - A guide to getting started with the DSPy framework from what is DSPy to a full end-to-end DSPy examp...

32. [[PDF] Promptbreeder: Self-Referential Self-Improvement Via Prompt ...](https://www.semanticscholar.org/paper/Promptbreeder:-Self-Referential-Self-Improvement-Fernando-Banarse/7fe071ea76e49bc3e573beb53f07721630954247) - Promptbreeder is a general-purpose self-referential self-improvement mechanism that evolves and adap...

33. [ArXivQA/papers/2309.16797.md at main · taesiri/ArXivQA](https://github.com/taesiri/ArXivQA/blob/main/papers/2309.16797.md) - WIP - Automated Question Answering for ArXiv Papers with Large Language Models (https://arxiv.taesir...

34. [Voyager: An open-ended embodied agent with large language models](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mWGyYMsAAAAJ&citation_for_view=mWGyYMsAAAAJ%3Axtoqd-5pKcoC) - G Wang, Y Xie, Y Jiang, A Mandlekar, C Xiao, Y Zhu, L Fan, A Anandkumar, arXiv preprint arXiv:2305.1...

35. [VOYAGER: An Open-Ended Embodied Agent](http://arxiv.org/pdf/2305.16291.pdf)

36. [Voyager: An Open-Ended Embodied Agent with Large Language ...](https://arxiv.org/abs/2305.16291) - We introduce Voyager, the first LLM-powered embodied lifelong learning agent in Minecraft that conti...

37. [Voyager | An Open-Ended Embodied Agent with Large Language ...](https://voyager.minedojo.org)

38. [Paper page - Voyager: An Open-Ended Embodied Agent with Large ...](https://huggingface.co/papers/2305.16291) - Join the discussion on this paper page

39. [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent ...](https://huggingface.co/papers/2308.08155) - Join the discussion on this paper page

40. [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent ...](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/) - We present AutoGen, an open-source framework that allows developers to build LLM applications by com...

41. [A Survey of Automatic Prompt Engineering](https://arxiv.org/html/2502.11560v1) - This paper presents the first comprehensive survey on automated prompt engineering through a unified...

42. [Automatic Prompt Engineer (APE)](https://fnl.es/Science/Papers/Prompt+Engineering/Automatic+Prompt+Engineer+(APE)) - We propose Automatic Prompt Engineer1 (APE) for automatic instruction generation and selection. Exte...

43. [Autogen: Enabling next-gen llm applications via multi-agent conversation framework](https://scholar.google.com/citations?amp=&amp=&amp=&citation_for_view=IiSNwnAAAAAJ%3AzCpYd49hD24C&hl=en&user=IiSNwnAAAAAJ&view_op=view_citation) - Q Wu, G Bansal, J Zhang, Y Wu, S Zhang, E Zhu, B Li, L Jiang, X Zhang, C Wang, arXiv preprint arXiv:...

44. [A Systematic Survey of Prompt Engineering Techniques - arXiv](https://arxiv.org/abs/2406.06608) - Generative Artificial Intelligence (GenAI) systems are increasingly being deployed across diverse in...

45. [A Systematic Survey of Prompt Engineering in Large Language ...](https://arxiv.org/abs/2402.07927) - Prompt engineering has emerged as an indispensable technique for extending the capabilities of large...

46. [A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models | Semantic Scholar](https://www.semanticscholar.org/paper/A-Systematic-Survey-of-Prompt-Engineering-on-Models-Gu-Han/06d8562831c32844285a691c5250d04726df3c61) - This paper aims to provide a comprehensive survey of cutting-edge research in prompt engineering on ...

47. [The Prompt Report: A Systematic Survey of Prompting Techniques](https://sanderschulhoff.com/Prompt_Survey_Site/) - PromptSurvey

48. [Function calling | OpenAI API](https://developers.openai.com/api/docs/guides/function-calling) - Function calling (also known as tool calling) provides a powerful and flexible way for OpenAI models...

49. [Prompting Best Practices for Tool Use (Function Calling)](https://community.openai.com/t/prompting-best-practices-for-tool-use-function-calling/1123036) - OpenAI Developer Community · Prompting Best Practices for Tool Use (Function Calling) ... system pro...

