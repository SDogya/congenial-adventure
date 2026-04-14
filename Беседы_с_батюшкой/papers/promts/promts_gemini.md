# **Оптимизация промптов и архитектура мультиагентных систем: SOTA методы, топологии графов вычислений и паттерны навыков**

## **1\. Математическая и системная формулировка проблемы**

В контексте мультиагентных систем (MAS) на базе больших языковых моделей, процесс выполнения задачи моделируется как ориентированный ациклический граф (DAG) $\\mathcal{G} \= (\\mathcal{V}, \\mathcal{E})$, где множество узлов $\\mathcal{V} \= \\{a\_1, a\_2, \\dots, a\_N\\}$ представляет $N$ агентов, а направленные ребра $\\mathcal{E} \\subseteq \\mathcal{V} \\times \\mathcal{V}$ определяют поток информации.1 Наличие ребра $(a\_i, a\_j)$ означает, что строковый выход агента $a\_i$ конкатенируется с входным контекстом $a\_j$. Каждый узел $N\_i$ характеризуется кортежем $\\Pi\_i \= (\\psi\_i, C\_i, \\mathcal{W}\_i)$, где $\\psi\_i$ — базовая LLM, $C\_i$ — контекст (включающий системный промпт $p\_i$ и эпизодическую память $M\_i$), а $\\mathcal{W}\_i$ — множество доступных инструментов.3  
Каждый агент управляется промптом $p\_i$, выбираемым из пула кандидатов $\\mathcal{P}\_i \= \\{p\_i^1, p\_i^2, \\dots, p\_i^K\\}$.4 Проблема оптимизации мультиагентной системы сводится к поиску оптимального набора промптов $P^\* \= (p\_1^\*, p\_2^\*, \\dots, p\_N^\*) \\in \\mathcal{P}\_1 \\times \\dots \\times \\mathcal{P}\_N$, который минимизирует глобальную функцию потерь $\\mathcal{L}$ на валидационном множестве $\\mathcal{D}\_{valid}$.4 Пространство поиска $\\mathcal{O}(K^N)$ растет экспоненциально, а целевая функция недифференцируема классическими методами обратного распространения ошибки из\-за дискретной природы токенов и black-box архитектуры API-вызовов.  
Для преодоления комбинаторного взрыва оценка качества $\\mathcal{T}(\\tilde{P})$ комбинации промптов $\\tilde{P}$ формализуется через метрику Joint Quality Score. Она вычисляется как произведение индивидуальных оценок агентов $g(\\tilde{p}\_i)$ (компетентность при условии корректного входа) и оценок ребер $g(\\tilde{p}\_i, \\tilde{p}\_j)$ (надежность передачи семантического контекста между узлами) 2:

$$\\mathcal{T}(\\tilde{P}) \= \\left(\\prod\_{i=1}^N g(\\tilde{p}\_i)\\right) \\left(\\prod\_{(i,j) \\in \\mathcal{E}} g(\\tilde{p}\_i, \\tilde{p}\_j)\\right)$$  
Данная формулировка строго доказывает эквивалентность задачи оптимизации промптов MAS задаче вывода максимума апостериорной вероятности (Maximum a Posteriori, MAP) при допущении равномерного априорного распределения на множестве конфигураций промптов.4  
Смещение фокуса с изолированного промптинга на системную инженерию контекста (Context Engineering) требует оптимизации не только инструкций, но и топологии $\\mathcal{G}$. Задача расширяется до поиска оптимальной пары $(P^\*, \\mathcal{G}^\*)$, где $\\mathcal{G}^\* \\in \\Omega\_{\\mathcal{G}}$ выбирается из дискретного пространства допустимых паттернов взаимодействия (Chain, Tree, Graph, Debate, Reflection).7 Внедрение механизмов автоматического дифференцирования через текст и графовых суррогатных моделей позволяет аппроксимировать глобальный оптимум за полиномиальное время, минуя полное исчерпывающее тестирование $\\mathcal{O}(K^N)$ конфигураций.

## **2\. Ключевые методы оптимизации и архитектурные подходы**

### **2.1. Текстовое автоматическое дифференцирование (TextGrad и TEP)**

Фреймворк TextGrad (2025) реализует механизм обратного распространения ошибки, где числовые градиенты заменены текстовым фидбеком $\\nabla\_{\\text{LLM}}$.9 В вычислительном графе переменными выступают системные промпты, few-shot примеры или молекулярные структуры.5 Оптимизация узла $p\_i$ выполняется путем вычисления текстового градиента $\\nabla\_{\\text{LLM}} \\mathcal{L}\_i := \\text{LLM}(\\text{feedback}(q\_i, a\_i, \\text{gold}\_i, p\_i^{(t)}))$, после чего LLM-оптимизатор применяет этот градиент для генерации $p\_i^{(t+1)}$.11  
При масштабировании глубины DAG в TextGrad возникает проблема затухающих и взрывающихся текстовых градиентов (vanishing/exploding textual gradients). Фидбек либо экспоненциально увеличивается в длине, перегружая контекстное окно и провоцируя энтропийный коллапс внимания, либо теряет семантическую специфичность при прохождении множества узлов вверх по графу.12 Подход Textual Equilibrium Propagation (TEP) решает эту проблему за счет адаптации концепции Equilibrium Propagation из моделей на основе энергии (energy-based models). TEP заменяет глобальное распространение критики на локальные ограниченные пертурбации. Алгоритм разделен на две фазы: свободная фаза (free phase), в которой локальные LLM-критики итеративно доводят промпт до состояния равновесия (отсутствие предложений по улучшению), и фаза сдвига (nudged phase), применяющая проксимальные изменения с жестко ограниченной интенсивностью модификации на основе прямого распространения сигналов о глобальных целях (forward signaling).12  
Для предотвращения переобучения (overfitting) базового TextGrad под узкий контекст задачи применяется фреймворк Reflection-Enhanced Meta-Optimization (REMO). REMO интегрирует блок памяти на основе RAG, функционирующий как «журнал ошибок» (mistake notebook), и самоадаптивный оптимизатор. LLM-метаконтроллер синтезирует рефлексивные инсайты на уровне эпох, накапливая знания о межсессионной оптимизации, что стабилизирует траекторию градиентного спуска в текстовом пространстве.13

### **2.2. Вероятностный MAP-вывод и суррогатные GNN-модели (MAPRO и MASPOB)**

Алгоритм Multi-Agent PRompt Optimization (MAPRO) аппроксимирует решение задачи максимизации Joint Quality Score $\\mathcal{T}(\\tilde{P})$ с использованием алгоритма передачи сообщений Language-guided Max-Product Belief Propagation (LMPBP).2 Для обработки циклических зависимостей в графах применяется преобразование дерева сочленений (junction-tree transformation), гарантирующее точное вычисление MAP-оценок.6  
Ключевым механизмом MAPRO является топологически-ориентированное присвоение кредита (topology-aware credit assignment). Пул кандидатов обновляется через LLM-мутатор, принимающий на вход глобальный фидбек $f\_g$ (результат выполнения всего конвейера) и локальный фидбек $f\_l$. Последний формируется путем обхода топологии в обратном порядке: каждый агент генерирует «обвинения» (downstream blames) в адрес предшествующих узлов на основе полученного от них контекста и глобального результата $f\_g$.2 Мутации ограничиваются набором малых правок $\\mathcal{M}$, эмулируя концепцию trust region из Proximal Policy Optimization (PPO), чтобы предотвратить семантический дрейф промпта.2  
Альтернативный подход Multi-Agent System Prompt Optimization via Bandits (MASPOB) решает проблему сильной связанности промптов (topology-induced coupling), индуцированной топологией. MASPOB интегрирует графовую нейронную сеть (GNN) для кодирования структурных априорных знаний графа выполнения.1 GNN изучает топологически-осведомленные репрезентации семантики промптов. Оптимизация декомпозируется на одномерные подзадачи с использованием координатного подъема (coordinate ascent), что снижает сложность поиска с экспоненциальной до линейной. Исследование пространства управляется алгоритмом контекстуальных многоруких бандитов (LinUCB) с критерием Upper Confidence Bound (UCB), что радикально повышает эффективность использования бюджета API-вызовов по сравнению с эволюционными алгоритмами.1

### **2.3. Совместный поиск архитектур и промптов (MASS и MasRouter)**

Фреймворк Multi-Agent System Search (MASS) доказывает, что оптимизация инструкций и few-shot примеров на уровне отдельных блоков обеспечивает больший прирост производительности, чем простое масштабирование числа агентов.7 Оптимизация выполняется в три этапа 7:

1. **Блочная оптимизация (Block-level):** Применение автоматических оптимизаторов (например, APO) для настройки функционала каждого агента в изоляции.  
2. **Топологическая оптимизация (Workflow Topology):** Сэмплирование конфигураций системы из пространства, взвешенного по влиянию (influence-weighted space). Метрика инкрементального влияния вычисляет прирост производительности каждой топологии относительно базового агента, направляя поиск в сторону комбинаций с наивысшим градиентом полезности.  
3. **Глобальная оптимизация (Workflow-level):** Адаптация (fine-tuning) промптов как единой интегрированной сущности с учетом выявленных взаимозависимостей агентов в зафиксированной топологии.

Проблема статической маршрутизации решается в архитектуре Multi-Agent System Routing (MasRouter). Она использует каскадную сеть контроллеров для динамического конструирования MAS под конкретный запрос.8 Первый модуль определяет режим коллаборации $F\_{\\theta t}$, используя вариационную модель со скрытыми переменными (variational latent variable model) для отображения запроса в латентное пространство топологий. Второй модуль $F\_{\\theta r}$ распределяет роли через структурированный вероятностный каскад (structured probabilistic cascade), учитывая зависимости графа вычислений. Третий модуль $F\_{\\theta m}$ решает задачу мультиномиального распределения для маршрутизации конкретных LLM-бекендов к узлам, оптимизируя баланс между вычислительными затратами и метриками точности.8

### **2.4. Декларативное программирование конвейеров (DSPy, GEPA и AutoPDL)**

Переход от конкатенации строк к абстрактным сигнатурам формализован в DSPy. Сигнатуры имеют вид context, question \-\> answer и скрывают низкоуровневые детали форматирования.16 Телепромптеры (оптимизаторы) в DSPy настраивают веса LLM и текстовые инструкции.  
Алгоритм GEPA (Genetic-Pareto), интегрированный в DSPy, представляет собой рефлексивный оптимизатор, адаптивно эволюционирующий текстовые компоненты. В отличие от стандартных алгоритмов подкрепления, GEPA принимает не только скалярные метрики, но и текстовый фидбек. Встроенный механизм интроспекции анализирует текстовое обоснование оценки и генерирует гипотезы по улучшению, формируя высокоэффективные промпты за минимальное число развертываний (rollouts).18  
Фреймворк AutoPDL формулирует задачу как проблему AutoML над комбинаторным пространством агентивных паттернов (ReAct, ReWOO, CoT) и демонстраций.19 Алгоритм Successive Halving итеративно отбрасывает нижнюю половину наименее приспособленных конфигураций на валидационном наборе $\\mathcal{D}\_{valid}$, решая задачу $\\argmin\_{Ap \\in AP} \\mathcal{L}(Ap, \\mathcal{D}\_{valid})$. Итоговое решение транслируется в читаемый и исполняемый код на языке Prompt Declaration Language (PDL), обеспечивая возможность source-to-source оптимизации.19

### **2.5. Библиотеки навыков и воплощенные агенты (Voyager и Nex-N1)**

Агенты непрерывного обучения (lifelong learners), такие как Voyager, смещают парадигму от низкоуровневых действий к генерации исполняемого кода, который инкапсулирует темпорально-протяженные и композиционные навыки (temporally extended, compositional skills).21 Механизм итеративного промптинга замыкает цикл управления через три вектора обратной связи: состояние среды (environment feedback), трассировка ошибок парсера/интерпретатора (execution errors) и самоверификация LLM (self-verification), где модель выступает в роли критика собственного кода.21  
Успешно скомпилированные функции (например, Python или JavaScript методы) индексируются через векторные эмбеддинги их AST-представлений и документации, пополняя библиотеку навыков (Skill Library). При поступлении нового запроса от автоматического учебного плана (automatic curriculum), агент выполняет $k$-NN извлечение топ-5 релевантных навыков, загружая их в контекст исполнения.21  
Масштабирование сред исполнения обеспечивается инфраструктурой NexAU (Agent Universe) и NexGAP (General Agent-data Pipeline). Эти системы используют протокол Model Context Protocol (MCP) для генерации синтетических траекторий в аутентичных условиях исполнения. NexA4A автоматически синтезирует разнообразные архитектуры агентов из спецификаций на естественном языке, превращая конструирование среды из ручной инженерии в задачу генеративного языкового моделирования.23

### **2.6. Архитектура SOTA-промпта: Bento-Box XML паттерн**

В промышленных системах (2025-2026) парадигма "Prompt Engineering" эволюционировала в "Context Engineering".24 Доминирующим архитектурным шаблоном стала структура "Bento-Box", изолирующая императивные директивы, динамические данные и протоколы рассуждений через строгую иерархию XML-тегов.25  
Использование XML-разметки обусловлено механикой токенизации и работы голов внимания (attention heads) в архитектуре Transformer. Теги создают жесткие семантические границы в пространстве эмбеддингов, минимизируя интерференцию между инструкциями и шумом пользовательского ввода (предотвращая атаки prompt injection).27  
Ниже приведена анатомия технического SOTA-промпта для агента оркестрации:

XML

\<system\_directive\>  
  \<role\>Senior SRE Agent\</role\>  
  \<operational\_constraints\>  
    \<rule\>Использовать метод Фейнмана для декомпозиции графа зависимостей.\</rule\>  
    \<rule\>Формировать вывод строго на основе \<telemetry\_data\>. Доступ к параметрической памяти для вывода конфигураций запрещен (Zero-Guessing policy).\</rule\>  
    \<rule\>При исчерпании контекста внутри \<telemetry\_data\>, инициировать вызов инструмента \`query\_splunk\_logs\` с параметрами в формате JSON schema.\</rule\>  
  \</operational\_constraints\>  
\</system\_directive\>

\<infrastructure\_state\>  
  \<topology\>  
    \<vlan id="10" cidr="10.0.1.0/24" type="management" /\>  
    \<vlan id="20" cidr="10.0.2.0/24" type="production" /\>  
  \</topology\>  
  \<cluster\_status version="v1.31"\>Nodes: 4 Ready, 1 NotReady (prod-worker-3)\</cluster\_status\>  
\</infrastructure\_state\>

\<telemetry\_data\>  
  \<cli\_output source="kubectl describe node prod-worker-3"\>  
    Conditions: Ready \- False, Reason: KubeletNotReady,   
    Message: container runtime network not ready: NetworkReady=false   
    reason:NetworkPluginNotReady message:calico: CNI plugin not initialized  
  \</cli\_output\>  
\</telemetry\_data\>

\<reasoning\_protocol\>  
  \<step id="1"\>Проанализировать \<cli\_output\> на совпадение с паттернами сетевых отказов (CNI).\</step\>  
  \<step id="2"\>Сопоставить инцидент с \<infrastructure\_state\> для определения радиуса поражения (blast radius).\</step\>  
  \<step id="3"\>Синтезировать \<resolution\_plan\> с пошаговыми CLI-командами.\</step\>  
\</reasoning\_protocol\>

\<context\_pinning\>  
 : Перед вызовом любых мутирующих инструментов (tools), сгенерируйте Wakeup-Call. Резюмируйте текущий вектор состояния инцидента в 3 тезисах внутри тега \<status\_quo\>.  
\</context\_pinning\>

1\. Инициировать \<reasoning\_protocol\>.  
2\. Сформировать payload для инструмента \`restart\_cni\_daemonset\`.

Механика работы шаблона основана на нескольких структурных принципах:

1. **Context Pinning (закрепление контекста):** Размещение критических триггеров безопасности (\<context\_pinning\>) в самом конце контекстного окна компенсирует эффект спада внимания (attention decay), заставляя модель учитывать ограничения на этапе вычисления логитов первого генерируемого токена.28  
2. **Wakeup-Calls:** Принудительная генерация тега \<status\_quo\> выполняет функцию "chain-of-thought" верификации. Вынуждая LLM сжать текущий контекст перед действием, агент обновляет свои скрытые состояния (hidden states), что радикально снижает вероятность галлюцинаций в параметрах API-вызовов.25  
3. **Изоляция полезной нагрузки (Payload Isolation):** Разделение \<telemetry\_data\> и \<infrastructure\_state\> защищает агента от семантического дрейфа, когда шумные данные из логов ошибочно интерпретируются как инструкции.27

## **3\. Количественный ландшафт (Бенчмарки)**

| Фреймворк / Алгоритм | Архитектура LLM | Оптимизируемая метрика | Датасет / Среда | Год | Конкретный результат | Источник |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **TextGrad** | GPT-4o | Zero-Shot Accuracy | **GPQA** (Google-Proof QA) | 2025 | Улучшение с 51.0% до **55.0%** (SOTA для GPT-4o) | 9 |
| **TextGrad \+ TEP** | GPT-4o / Qwen3 | Pass Rate | **BigCodeBench** | 2026 | Снижение затухания градиента: **35.71% $\\pm$ 0.10%** | 12 |
| **MAPRO** (LMPBP) | Llama 3.3-70b | Exact Match (EM) | **GSM8K** | 2026 | **93.48% $\\pm$ 0.42%** | 29 |
| **MAPRO** (LMPBP) | Claude 3.5 Haiku | Pass Rate | **MBPP-Plus** | 2026 | **76.54% $\\pm$ 0.67%** | 2 |
| **MASS** (3-stage opt) | Gemini 1.5 Pro | Avg Accuracy | **MATH** | 2025 | Базовый: 62.3% $\\rightarrow$ **84.0%** | 30 |
| **MASS** (3-stage opt) | Gemini 1.5 Pro | Exact Match (EM) | **HotpotQA** | 2025 | Улучшение до **91.0%** | 32 |
| **MASPOB** (GNN+LinUCB) | GPT-4o | EM | **DROP** | 2026 | **82.28% $\\pm$ 0.55%** | 1 |
| **AutoPDL** (Succ. Halving) | Granite 13B | Accuracy | **FEVER** | 2025 | Прирост **\+67.5 pp** (6.5% $\\rightarrow$ 74.0%) | 19 |
| **AutoPDL** (Succ. Halving) | LLaMA 3.3 70B | Exact Match (EM) | **GSM8K** | 2025 | Прирост **\+9.9 pp** (до 95.4%) | 19 |
| **A-MapReduce** | OpenAI o3 / Gemini 2.5 | Item F1 / Runtime | **Web Retrieval** | 2026 | **\+17.5% Item F1**, снижение runtime на 47% | 33 |
| **Voyager** | GPT-4 | Unique Items / Tech Tree | **Minecraft** | 2023 | **3.3x** больше предметов, в 15.3x быстрее | 21 |

Количественный анализ подтверждает фундаментальный сдвиг в оптимизации: локальная настройка агентов до их топологического масштабирования является наиболее эффективной по затратам токенов. Аблирование MASS показывает, что простое увеличение числа агентов в топологиях Majority Voting или Self-Consistency насыщается значительно раньше, чем системы, прошедшие через многоэтапный поиск (stage-wise optimization).7  
На бенчмарке MATH фреймворк MASS достигает 84.0%, в то время как классические дебаты или самосогласованность (self-consistency) останавливаются на уровне 76-80%.30 Внедрение GNN в MASPOB для кодирования графов вычислений дает прямой прирост \+3.08% на DROP и \+1.47% на MBPP в сравнении с версиями без GNN, подтверждая влияние топологически-осведомленного поиска.1 Алгоритм TEP в связке с TextGrad нивелирует падение точности в глубоких графах, доводя Pass Rate на BigCodeBench до 35.71% $\\pm$ 0.10%.12

## **4\. Модели отказов и системные ограничения (Failure Modes)**

Индустриальный анализ миллионов трассировок выявил дискретную таксономию семантических и архитектурных сбоев мультиагентных систем (WFGY 16 Failure Modes, Arize AI).28 Данные сбои не являются случайными артефактами, а представляют собой детерминированные следствия механики внимания и авторегрессионной генерации.

### **4.1. Шум поиска и перегрузка контекста (Hallucination & Chunk Drift)**

Агрессивная индексация корпоративных баз без жесткого структурирования приводит к коллапсу энтропии внимания. Дамп неструктурированных чанков в контекст вызывает эффект "Lost in the Middle". Агент получает релевантный документ, но игнорирует его, формируя ответ из соседних нерелевантных блоков.28 Решение: отказ от метрик Precision@K в пользу оценки использования на уровне текстовых интервалов (span-level usage metrics) для верификации активации логитов на целевых чанках.

### **4.2. Галлюцинирование аргументов API (Silent Failures / Bluffing)**

Модель сталкивается с конфликтом параметрических знаний и предоставленного контекста.28 При несовпадении схем агент генерирует параметры "вслепую". Например, передает { "user\_id": "843A" } вместо требуемого базой данных { "customer\_uuid": "843A" }. Запрос возвращает пустой JSON-ответ (валидный, но нулевой). LLM не идентифицирует сбой схемы и уверенно синтезирует нарратив об "отсутствии данных в системе", что является сайлент-отказом (silent failure).28

### **4.3. Рекурсивные циклы и налог на поллинг (Logic Spirals)**

В асинхронных задачах (индексация, загрузка файлов) агенты с высокой степенью автономности попадают в циклы гиперактивного ожидания. Получив от API статус 200 OK (но в теле status: "processing"), агент извиняется и немедленно повторяет вызов. Отсутствие внутреннего счетчика контрфактических рассуждений приводит к генерации сотен запросов за секунды, исчерпывая лимиты API и бюджет токенов при внешне "здоровой" телеметрии сети.28

### **4.4. Конфликт контекста и инстинктов базовой модели (The Reflex Problem)**

Предварительное обучение (Pre-training bias) может переопределять строгие ограничения системного промпта. Модель, обученная на миллионах диалогов лояльной техподдержки, обладает нейронным "рефлексом" предлагать компенсацию. При внедрении в \<context\> документа с политикой "Возврат средств запрещен", агент со слабой инструктивной настройкой игнорирует документ в пользу параметрического инстинкта "быть вежливым", что приводит к системным уязвимостям.28

### **4.5. Коллапс интерпретации и дрейф семантики (Interpretation Collapse)**

В длинных цепочках рассуждений (Long Reasoning Chains) семантика промежуточных шагов размывается. Эмбеддинговое косинусное сходство (Cosine Similarity) дает ложноположительные результаты на семантически противоположных, но лексически схожих чанках. Это приводит к многоагентному хаосу (Multi-Agent Chaos), когда один агент тихой перезаписью (quiet overwrite) уничтожает логику другого агента в разделяемом блоке памяти.34

### **4.6. Дедлоки и сбои развертывания (Bootstrap Ordering)**

При инициализации MAS в продакшене возникают состояния кругового ожидания (Deployment Deadlock) и состояния гонки (Infra races). Микроагенты (например, Retrieval Agent) пытаются выполнить API-вызовы к векторной базе до завершения работы индексатора, что приводит к падению всей системы на первом вызове (Pre-Deploy Collapse).34

## **5\. Открытые проблемы**

В литературе 2025-2026 годов выделен спектр нерешенных архитектурных и математических вызовов, ограничивающих применение агентных систем в критических инфраструктурах.36

### **5.1. Верифицируемость стохастических планировщиков (Verifiability vs. Adaptability)**

Переход от жестких символьных графов (symbolic constraints) к нейро-ориентированным вероятностным планировщикам (LLM-based planning) уничтожает формальную верифицируемость. Не существует математических гарантий того, что агент в открытой среде не нарушит правила безопасности при композиции инструментов. Актуальной задачей является разработка структур управления по принципу "governance-by-construction", интегрирующих методы классической теории управления с непредсказуемой динамикой вероятностных сетей. Требуется физическая и программная изоляция когнитивного планировщика от слоев выполнения через строгие Role-Based Access Control (RBAC).36

### **5.2. Интероперабельность и типизированные контракты (Shared Protocols)**

Мультиагентные системы остаются замкнутыми в монолитных фреймворках разработчика. Острой проблемой является отсутствие стандартизированных протоколов взаимодействия агентов (подобно REST или gRPC в Web Services). Индустрии необходимы единые спецификации (например, развитие Model Context Protocol, MCP), типизированные контракты и брокеры сообщений для динамической композиции автономных агентов из различных экосистем в единый вычислительный граф.36

### **5.3. Безопасность цепочек поставок навыков (Skill Supply-Chain Security)**

С распространением открытых библиотек навыков (Skill Libraries) возник новый вектор атак. В кейсе ClawHavoc (2025) злоумышленники внедрили более 1200 вредоносных процедурных навыков в публичные репозитории. Автономные агенты, динамически загружающие эти навыки, выполняли инъекции промптов и эксфильтрацию API-ключей и учетных данных в фоновом режиме. Исследования сфокусированы на разработке механизмов "trust-tiered execution" — создании песочниц (WebAssembly, microVMs) для изоляции среды выполнения недоверенного агентского кода.38

### **5.4. Масштабируемая иерархическая память (Scalable Memory Tiering)**

Текущие методы борьбы с ограничениями контекстного окна (уплотнение, суммаризация) приводят к катастрофическому забыванию фактов (catastrophic forgetting). Применение рекурсивных языковых моделей (Recursive Language Models, RLM), где корень графа порождает sub-LLM для чтения чанков, на практике вызывает экспоненциальный взрыв галлюцинаций (hallucination multiplication) и колоссальные штрафы по задержке (latency penalties).40 Ко-дизайн архитектуры памяти (разделение на L1 кэш, L2 RAG, L3 графы знаний) и когнитивных блоков обслуживания остается одной из самых ресурсоемких открытых проблем.36

### **5.5. Кибер-физические системы безопасности (Cyber-Physical Safety Frameworks)**

В робототехнике и АСУ ТП высокоуровневые дискурсивные LLM-планировщики должны сосуществовать с низкоуровневыми реактивными слоями безопасности. Ошибка генерации плана не должна приводить к физическому ущербу оборудования. Открыт вопрос интеграции непрерывного физического контроля (continuous physical constraints) с дискретной символьной логикой языковых моделей, чтобы аппаратный уровень мог переопределять галлюцинации LLM с нулевой задержкой.36

## **Ключевые работы и их технический вклад**

| Название статьи и Авторы | Год | Технический вклад | Идентификатор |
| ----- | :---- | :---- | :---- |
| *Optimizing generative AI by backpropagating language model feedback (TextGrad)*, Yuksekgonul et al. | 2025 | Формализует автоматическое дифференцирование составных AI-систем, заменяя числовые градиенты текстовым фидбеком $\\nabla\_{\\text{LLM}}$. | 42 |
| *MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference*, Zhang et al. | 2026 | Преобразует оптимизацию промптов MAS в вероятностный MAP-вывод, используя max-product передачу сообщений для полиномиального поиска. | 4 |
| *Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies (MASS)*, Zhou et al. | 2025 | Доказывает превосходство 3-этапной ко-оптимизации промптов и топологий графов вычислений над масштабированием числа агентов. | 7 |
| *MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks*, Hong et al. | 2026 | Применяет GNN и алгоритмы бандитов (LinUCB) для моделирования связанности промптов (topology-induced coupling) при строгом бюджете. | 1 |
| *AutoPDL: Automatic Prompt Optimization for LLM Agents*, Spiess et al. | 2025 | Транслирует задачу в AutoML-поиск паттернов (ReAct, CoT) через Successive Halving, генерируя код на Prompt Declaration Language. | 19 |
| *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*, Agrawal et al. | 2025 | Разрабатывает эволюционный оптимизатор с использованием интроспекции LLM на основе текстового фидбека (экосистема DSPy). | 18 |
| *Voyager: An Open-Ended Embodied Agent with Large Language Models*, Wang et al. | 2023 | Внедряет парадигму непрерывного обучения через формирование библиотеки верифицируемых исполняемых навыков (AST-кода). | 21 |
| *Textual Equilibrium Propagation (TEP)*, Zhang et al. | 2026 | Решает проблему взрывающихся текстовых градиентов в глубоких DAG через локальные пертурбации и фазы локального равновесия. | 12 |
| *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines*, Khattab et al. | 2024 | Заменяет ручное написание промптов на декларативные сигнатуры и автоматическую компиляцию/подстройку весов конвейеров. | 43 |
| *metaTextGrad: Automatically optimizing language model optimizers*, Xu et al. | 2025 | Создает мета-оптимизатор для автоматической адаптации структур и инструкций самих алгоритмов языковой оптимизации. | 44 |
| *The Prompt Report: A Systematic Survey of Prompt Engineering Techniques*, Schulhoff et al. | 2024 | Предоставляет крупнейшую таксономию (58 паттернов) и словарь (33 термина) методов префиксного промптинга и декомпозиции. | 45 |
| *Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*, Zhang et al. | 2026 | Внедряет фреймворк ACE для оффлайн/онлайн оптимизации контекста без размеченных данных, базируясь на естественном фидбеке среды. | 46 |
| *WFGY Problem Map 2.0 (16 Failure Modes)*, WFGY Foundation | 2026 | Систематизирует 16 критических аппаратных и семантических отказов RAG-архитектур, от дрейфа чанков до коллапса интерпретации. | 34 |
| *MasRouter: Multi-Agent System Routing*, Yue et al. | 2025 | Представляет каскадную сеть контроллеров для динамического определения режимов коллаборации, ролей и маршрутизации LLM. | 8 |

#### **Works cited**

1. MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks \- arXiv.org, accessed April 10, 2026, [https://arxiv.org/html/2603.02630v1](https://arxiv.org/html/2603.02630v1)  
2. MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference, accessed April 10, 2026, [https://arxiv.org/html/2510.07475v1](https://arxiv.org/html/2510.07475v1)  
3. A Survey of Self-Evolving Agents What, When, How, and Where to Evolve on the Path to Artificial Super Intelligence \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2507.21046v4](https://arxiv.org/html/2507.21046v4)  
4. \[2510.07475\] MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2510.07475](https://arxiv.org/abs/2510.07475)  
5. : Automatic “Differentiation” via Text \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2406.07496v1](https://arxiv.org/html/2406.07496v1)  
6. \[Literature Review\] MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference \- Moonlight, accessed April 10, 2026, [https://www.themoonlight.io/en/review/mapro-recasting-multi-agent-prompt-optimization-as-maximum-a-posteriori-inference](https://www.themoonlight.io/en/review/mapro-recasting-multi-agent-prompt-optimization-as-maximum-a-posteriori-inference)  
7. Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies | OpenReview, accessed April 10, 2026, [https://openreview.net/forum?id=uCKvHweh1g](https://openreview.net/forum?id=uCKvHweh1g)  
8. MasRouter: Learning to Route LLMs for Multi-Agent ... \- ACL Anthology, accessed April 10, 2026, [https://aclanthology.org/2025.acl-long.757.pdf](https://aclanthology.org/2025.acl-long.757.pdf)  
9. TextGrad: Automatic" Differentiation" via Text, accessed April 10, 2026, [https://arxiv.org/pdf/2406.07496](https://arxiv.org/pdf/2406.07496)  
10. (PDF) TextGrad: Automatic "Differentiation" via Text \- ResearchGate, accessed April 10, 2026, [https://www.researchgate.net/publication/381318980\_TextGrad\_Automatic\_Differentiation\_via\_Text](https://www.researchgate.net/publication/381318980_TextGrad_Automatic_Differentiation_via_Text)  
11. A Prompt Optimization System Based on Center-Aware Textual Gradients \- MDPI, accessed April 10, 2026, [https://www.mdpi.com/2079-8954/13/9/748](https://www.mdpi.com/2079-8954/13/9/748)  
12. Textual Equilibrium Propagation for Deep Compound AI Systems \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2601.21064v3](https://arxiv.org/html/2601.21064v3)  
13. Reflection-Enhanced Meta-Optimization: Integrating TextGrad-style Prompt Optimization with Memory-Driven Self-Evolution \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2508.18749v1](https://arxiv.org/html/2508.18749v1)  
14. Zhi Hong \- CatalyzeX, accessed April 10, 2026, [https://www.catalyzex.com/author/Zhi%20Hong](https://www.catalyzex.com/author/Zhi%20Hong)  
15. Agentic Design Patterns, accessed April 10, 2026, [https://irp.cdn-website.com/ca79032a/files/uploaded/Agentic-Design-Patterns.pdf](https://irp.cdn-website.com/ca79032a/files/uploaded/Agentic-Design-Patterns.pdf)  
16. DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines, accessed April 10, 2026, [https://arxiv.org/html/2312.13382v2](https://arxiv.org/html/2312.13382v2)  
17. LLMs-in-Production/chapters/chapter\_7/DSPy local tutorial.ipynb at main \- GitHub, accessed April 10, 2026, [https://github.com/IMJONEZZ/LLMs-in-Production/blob/main/chapters/chapter\_7/DSPy%20local%20tutorial.ipynb](https://github.com/IMJONEZZ/LLMs-in-Production/blob/main/chapters/chapter_7/DSPy%20local%20tutorial.ipynb)  
18. 1\. GEPA Overview \- DSPy, accessed April 10, 2026, [https://dspy.ai/api/optimizers/GEPA/overview/](https://dspy.ai/api/optimizers/GEPA/overview/)  
19. Automatic Prompt Optimization for LLM Agents \- AutoPDL \- arXiv, accessed April 10, 2026, [https://arxiv.org/pdf/2504.04365](https://arxiv.org/pdf/2504.04365)  
20. \[2504.04365\] AutoPDL: Automatic Prompt Optimization for LLM Agents \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2504.04365](https://arxiv.org/abs/2504.04365)  
21. Voyager | An Open-Ended Embodied Agent with Large Language Models, accessed April 10, 2026, [https://voyager.minedojo.org/](https://voyager.minedojo.org/)  
22. Voyager: An LLM-powered learning agent in Minecraft : r/MachineLearning \- Reddit, accessed April 10, 2026, [https://www.reddit.com/r/MachineLearning/comments/13sc0pp/voyager\_an\_llmpowered\_learning\_agent\_in\_minecraft/](https://www.reddit.com/r/MachineLearning/comments/13sc0pp/voyager_an_llmpowered_learning_agent_in_minecraft/)  
23. Nex-N1: Agentic Models Trained via a Unified Ecosystem for Large-Scale Environment Construction \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2512.04987v1](https://arxiv.org/html/2512.04987v1)  
24. Effective context engineering for AI agents \- Anthropic, accessed April 10, 2026, [https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)  
25. Advanced Prompt Engineering in 2026? : r/PromptEngineering, accessed April 10, 2026, [https://www.reddit.com/r/PromptEngineering/comments/1r8yl5j/advanced\_prompt\_engineering\_in\_2026/](https://www.reddit.com/r/PromptEngineering/comments/1r8yl5j/advanced_prompt_engineering_in_2026/)  
26. Product updates \- MockFlow, accessed April 10, 2026, [https://mockflow.com/updates](https://mockflow.com/updates)  
27. Effective Prompt Engineering: Mastering XML Tags for Clarity, Precision, and Security in LLMs | by Tech for Humans | Medium, accessed April 10, 2026, [https://medium.com/@TechforHumans/effective-prompt-engineering-mastering-xml-tags-for-clarity-precision-and-security-in-llms-992cae203fdc](https://medium.com/@TechforHumans/effective-prompt-engineering-mastering-xml-tags-for-clarity-precision-and-security-in-llms-992cae203fdc)  
28. Why AI Agents Break: A Field Analysis of Production Failures \- Arize AI, accessed April 10, 2026, [https://arize.com/blog/common-ai-agent-failures/](https://arize.com/blog/common-ai-agent-failures/)  
29. MAPRO: Recasting Multi-Agent Prompt Optimization as Maximum a Posteriori Inference \- arXiv, accessed April 10, 2026, [https://arxiv.org/pdf/2510.07475](https://arxiv.org/pdf/2510.07475)  
30. Optimizing Multi-Agent Workflows: Inside Google's New MASS Framework | by Hadi Tabani, accessed April 10, 2026, [https://medium.com/@haditabani\_91415/optimizing-multi-agent-workflows-inside-googles-new-mass-framework-0b0341a727b4](https://medium.com/@haditabani_91415/optimizing-multi-agent-workflows-inside-googles-new-mass-framework-0b0341a727b4)  
31. Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies \- ResearchGate, accessed April 10, 2026, [https://www.researchgate.net/publication/388686619\_Multi-Agent\_Design\_Optimizing\_Agents\_with\_Better\_Prompts\_and\_Topologies](https://www.researchgate.net/publication/388686619_Multi-Agent_Design_Optimizing_Agents_with_Better_Prompts_and_Topologies)  
32. Reasoning Topology Matters: Network-of-Thought for Complex Reasoning Tasks \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2603.20730v1](https://arxiv.org/html/2603.20730v1)  
33. Multi-Agent System Search \- Emergent Mind, accessed April 10, 2026, [https://www.emergentmind.com/topics/multi-agent-system-search-mass](https://www.emergentmind.com/topics/multi-agent-system-search-mass)  
34. onestardao/WFGY\_RAG\_Problem\_Map\_Index: 16 real ... \- GitHub, accessed April 10, 2026, [https://github.com/onestardao/WFGY-ProblemMap-Index](https://github.com/onestardao/WFGY-ProblemMap-Index)  
35. Suggestion: reference WFGY 16 Problem Map in the RAG / evaluation sections · Issue \#730 · dair-ai/Prompt-Engineering-Guide \- GitHub, accessed April 10, 2026, [https://github.com/dair-ai/Prompt-Engineering-Guide/issues/730](https://github.com/dair-ai/Prompt-Engineering-Guide/issues/730)  
36. From Prompt–Response to Goal-Directed Systems: The Evolution of Agentic AI Software Architecture \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2602.10479](https://arxiv.org/html/2602.10479)  
37. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows, accessed April 10, 2026, [https://arxiv.org/html/2506.23260v1](https://arxiv.org/html/2506.23260v1)  
38. Clawed and Dangerous: Can We Trust Open Agentic Systems? \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2603.26221v1](https://arxiv.org/html/2603.26221v1)  
39. SoK: Agentic Skills — Beyond Tool Use in LLM Agents \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2602.20867v1](https://arxiv.org/html/2602.20867v1)  
40. Think, But Don't Overthink: Reproducing Recursive Language Models \- arXiv.org, accessed April 10, 2026, [https://arxiv.org/html/2603.02615v1](https://arxiv.org/html/2603.02615v1)  
41. Recursive Language Models: MIT's Solution to Infinite Context (Without Summarization) | by Reliable Data Engineering | Mar, 2026 | Medium, accessed April 10, 2026, [https://medium.com/@reliabledataengineering/recursive-language-models-mits-solution-to-infinite-context-without-summarization-d0386e862053](https://medium.com/@reliabledataengineering/recursive-language-models-mits-solution-to-infinite-context-without-summarization-d0386e862053)  
42. Optimizing generative AI by backpropagating language model feedback \- PubMed, accessed April 10, 2026, [https://pubmed.ncbi.nlm.nih.gov/40108317/](https://pubmed.ncbi.nlm.nih.gov/40108317/)  
43. DSPy, accessed April 10, 2026, [https://dspy.ai/](https://dspy.ai/)  
44. NeurIPS Poster metaTextGrad: Automatically optimizing language model optimizers, accessed April 10, 2026, [https://neurips.cc/virtual/2025/poster/120272](https://neurips.cc/virtual/2025/poster/120272)  
45. The Prompt Report: A Systematic Survey of Prompt Engineering Techniques \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2406.06608](https://arxiv.org/abs/2406.06608)  
46. \[2510.04618\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2510.04618](https://arxiv.org/abs/2510.04618)