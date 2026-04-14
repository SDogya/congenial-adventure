# **Byte Latent Transformer: Autoregressive Sequence Modeling via Entropy-Guided Dynamic Patching**

## **1\. Problem Formulation**

Autoregressive sequence modeling fundamentally seeks to approximate the joint probability distribution of a discrete sequence $X \= (x\_1, x\_2, \\dots, x\_T)$ via the chain rule of probability: $P(X) \= \\prod\_{i=1}^{T} P(x\_i | x\_{\<i})$. In dominant Large Language Model (LLM) architectures, the sequence $X$ is constructed by mapping a raw uncompressed byte array $B \= (b\_1, b\_2, \\dots, b\_L)$ into a compressed token space utilizing a heuristic, non-differentiable pre-processing function $f\_{tokenizer}: \\mathcal{B}^\* \\to \\mathcal{V}^\*$, typically instantiated via Byte-Pair Encoding (BPE) or WordPiece algorithms.1  
This tokenization mapping achieves critical computational tractability by compressing the sequence length ($T \\ll L$), thereby mitigating the quadratic time and space complexity $\\mathcal{O}(T^2 d)$ of standard dense self-attention mechanisms.1 However, defining the state space via a static vocabulary $\\mathcal{V}$ introduces severe structural limitations. Expanding $\\mathcal{V}$ to increase sequence compression directly inflates the embedding and projection layers, precipitating Embedding Sparsity Risk (ESR), wherein aggressive vocabulary expansion yields under-trained, inactive "zombie parameters".4 Furthermore, static boundaries fail to capture character-level orthographic semantics 5, systematically degrade morphologically rich and non-Latin scripts via the "byte premium" (whereby out-of-distribution characters require multiple tokens) 7, and exhibit catastrophic topological fragmentation under adversarial noise or spelling perturbations.8  
Conversely, operating directly on the byte sequence ($x\_i \\in \\{0, 1, \\dots, 255\\}$) preserves lossless continuous features and eliminates out-of-vocabulary (OOV) failure modes.9 Yet, naive byte-level models like ByT5 incur a $1.2\\times$ to $10\\times$ Floating Point Operations (FLOP) overhead due to the expanded sequence length $L \\approx 4T$, saturating memory bandwidth and bounding inference latency.11  
The Byte Latent Transformer (BLT) redefines the objective by learning a dynamic partition function $\\Pi: \\mathcal{B}^\* \\to \\mathcal{P}^\*$. BLT maps the raw byte sequence $B$ into a sequence of variable-length latent patches $P \= (p\_1, p\_2, \\dots, p\_K)$ where $K \< T$. The patch boundaries are determined dynamically by the instantaneous conditional entropy of the data distribution, eliminating the static vocabulary $\\mathcal{V}$ while reallocating compute density strictly to high-complexity sequence regions.6

## **2\. Key Methods: Architecture and Implementation**

BLT bypasses the tokenization bottleneck through a hierarchical, three-stage computational pipeline: a Local Encoder to compress variable-length byte spans, a Latent Global Transformer to autoregressively process the high-dimensional patch space, and a Local Decoder to project latent patches back into raw byte probabilities.15

### **2.1 Hash $n$-gram Embeddings**

To inject localized contextual priors into individual byte representations prior to pooling, the architecture augments the base byte embeddings with hash $n$-gram embeddings. Storing an exhaustive table of exact $n$-gram frequencies scales as $256^n$, exceeding memory limits for larger spans. BLT utilizes rolling polynomial hashing to project arbitrary byte spans into a dimension-constrained embedding table $\\mathbf{E}^{hash} \\in \\mathbb{R}^{V \\times d}$.13  
For a byte sequence, the embedding $e\_i$ at index $i$ is formulated as the sum of the unigram byte embedding $x\_i$ and the hash embeddings for multiple preceding sequence lengths $n \\in \\{3, 4, 5, 6, 7, 8\\}$:

$$e\_i \= x\_i \+ \\sum\_{n=3}^{8} \\mathbf{E}\_n^{hash}\\Big(\\text{Hash}(g\_{i,n})\\Big)$$  
The rolling hash function applied to the $n$-gram $g\_{i,n} \= (b\_{i-n+1}, \\dots, b\_i)$ is defined mathematically as:

$$\\text{Hash}(g\_{i,n}) \= \\left( \\sum\_{j=1}^{n} b\_j a^{j-1} \\right) \\pmod V$$  
where $b\_j$ denotes the integer value of the byte, $a$ acts as the radix multiplier, and the collision space $V$ is constrained. Empirical ablations dictate a vocabulary size $V \= 500,000$ per table. Expanding $V$ beyond $10^6$ to $2 \\times 10^6$ entries exhibits diminishing returns in Bits-Per-Byte (BPB) minimization. Shorter spans ($n \\in $) provide disproportionately higher signal density for structural formatting like JSON or XML compared to longer spans ($n \\in $).13

### **2.2 Entropy-Guided Dynamic Segmentation**

The segmentation logic governing the boundaries of patch $p\_k$ operates independently of future sequence data to satisfy the strict autoregressive incremental patching constraint: $f\_p(x\_{\\le i})$.14 BLT calculates boundary coordinates utilizing a pre-trained, auxiliary byte-level causal language model $M\_\\theta$. This entropy model predicts the probability distribution of the next byte $p\_\\theta(x\_i | x\_{\<i})$.20  
The uncertainty at step $i$ is formalized via Shannon entropy:

$$H(x\_i) \= \-\\sum\_{v=0}^{255} p\_\\theta(x\_i \= v | x\_{\<i}) \\log p\_\\theta(x\_i \= v | x\_{\<i})$$  
A patch boundary is instantiated whenever the local complexity meets specific thresholding rules. The framework formalizes two primary gating mechanisms 22:

1. **Global Threshold:** A boundary triggers if $H(x\_i) \> \\tau\_H$.  
2. **Monotonicity Constraint:** A boundary triggers based on the discrete derivative of the entropy, isolating sudden transitions in predictability: $H(x\_i) \- H(x\_{i-1}) \> \\tau\_{\\Delta}$.

The threshold $\\tau$ dictates the granularity-to-efficiency ratio. Lowering $\\tau$ shortens the average patch size ($ps$), decreasing data compression but increasing model granularity. BLT explicitly targets $\\tau$ configurations yielding average patch sizes of $ps=6$ and $ps=8$ bytes. The entropy model $M\_\\theta$ requires minimal capacity; empirical data indicates diminishing improvements in boundary precision for $M\_\\theta$ scaling beyond 50 Million parameters with a localized context window restricted to 512 bytes.13

### **2.3 Local Encoder Architecture**

The Local Encoder maps the varying-length byte sequence spanning a defined patch into a static-dimensional latent representation $p\_k$. Following the application of hash $n$-gram embeddings, the sequence is processed via a lightweight transformer block utilizing a local block-causal attention mask. Each byte attends strictly to a fixed sliding window $w\_{\\mathcal{E}}$ of preceding bytes, bounded by document endpoints but permitted to bridge intra-document patch boundaries.13  
Dimensionality reduction is executed via a multi-headed cross-attention module adapted from the Perceiver architecture.5 An initialization vector $\\mathbf{P}\_{0, j}$ is constructed by max-pooling the byte embeddings belonging to patch $j$. This vector operates as the query matrix $\\mathbf{Q}$, while the local encoder's intermediate byte representations function as the keys $\\mathbf{K}$ and values $\\mathbf{V}$.  
The cross-attention update to project $\\mathbb{R}^{L \\times d} \\to \\mathbb{R}^{1 \\times d}$ per patch is:

$$\\mathbf{P}\_l \= \\mathbf{P}\_{l-1} \+ \\mathbf{W}\_o \\left( \\text{Softmax}\\left( \\frac{\\mathbf{Q} \\mathbf{K}^T}{\\sqrt{d\_k}} \\right) \\mathbf{V} \\right)$$  
where $\\mathbf{Q}\_j \= \\mathbf{W}\_q(\\mathbf{P}\_{l-1, j})$, $\\mathbf{K}\_i \= \\mathbf{W}\_k(\\mathbf{h}\_{l-1, i})$, and $\\mathbf{V}\_i \= \\mathbf{W}\_v(\\mathbf{h}\_{l-1, i})$. To restrict cross-contamination, the self-attention masking strategy isolates queries to attend exclusively to the specific keys and values corresponding to the bytes explicitly bounded within patch $j$.13 The local encoder specifies cross-attention heads $k=2$.8

### **2.4 Latent Global Transformer**

The Latent Global Transformer acts as the core reasoning engine, consuming the vast majority of the FLOP budget. It processes the sequence of patch embeddings $P \= (p\_1, \\dots, p\_K)$ autoregressively.  
Unlike token models, the global model utilizes a block-causal mask over the patch sequence, predicting the latent state of the subsequent patch $p\_{k+1}$ conditioned on $p\_{\\le k}$.13 For the 8B parameter scale architecture, the backbone replicates the LLaMA 3 structural specifications to isolate the efficacy of patching versus tokenization: 32 hidden layers, intermediate FFN dimension sizes scaled accordingly, a hidden dimension of $d\_{model} \= 4096$, 32 attention heads, and Grouped-Query Attention (GQA).8 For larger average patch sizes ($ps=8$), the latent model configuration scales up the encoder depth to 3 layers to compensate for higher data compression per patch.13  
By operating over patch lengths $ps=8$, the sequence length is halved compared to BPE (which averages $\\approx 4$ bytes/token). This fundamentally alters the Hoffmann scaling laws, shifting the inference compute boundary to yield up to a 50% reduction in global attention FLOPs.11

### **2.5 Local Decoder and Autoregressive Generation**

The Local Decoder projects the predicted latent patch vector $p\_{k+1}$ back into the raw byte domain. It implements an inverted variation of the cross-attention block: the raw byte embeddings act as queries $\\mathbf{Q}$, while the latent patch representation serves as the keys $\\mathbf{K}$ and values $\\mathbf{V}$.5  
The generation mechanism is strictly step-wise and dynamically bounded by the entropy model. During inference for patch $k+1$:

1. The decoder receives the global latent vector $p\_{k+1}$ alongside the terminating hidden state of the preceding byte.  
2. The decoder yields raw byte $b\_t$.  
3. The auxiliary entropy model evaluates $H(b\_t)$. If $H(b\_t) \\le \\tau\_H$, $b\_t$ is appended to the local context, and the decoder emits $b\_{t+1}$.  
4. If $H(b\_t) \> \\tau\_H$, the threshold is breached. The byte $b\_t$ acts as a terminal signal for patch $k+1$. It is discarded from the immediate patch representation.  
5. The accumulated bytes are passed to the Local Encoder to produce the finalized latent state for patch $k+1$, triggering the Latent Global Transformer to compute $p\_{k+2}$.27

This isolated, per-patch decoding ensures that the local module retains no persistent memory of prior patches outside of the representation passed down from the global latent state, strictly decoupling the local byte reconstruction from global sequence dependency.27

## **3\. Quantitative Landscape**

The empirical validation of BLT spans absolute scaling limits, zero-shot task reasoning, inference efficiency boundaries, and sub-character morphological alignment. Evaluating byte-level models against subword models necessitates the use of Bits-Per-Byte (BPB)—a tokenizer-agnostic metric derived from the cross-entropy loss normalized over sequence bytes—rather than arbitrary per-token perplexity.8  
The comparative matrix (Table 1\) tracks BLT against dominant contemporary architectures: traditional BPE (Llama 3), hierarchical fixed-stride architectures (MegaByte), and whitespace-heuristic patching (SpaceByte).

### **Table 1: Model Architectures and Quantitative Benchmarks (8B Scale)**

| Architecture Strategy | Parameter Scale | Average Patch / Token Size | BPB (C4/Wiki) | Inference FLOPs (Relative) | MMLU (5-shot) | ARC-Challenge | HellaSwag | HumanEval |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Llama 3.1 (BPE)** 13 | 8B | $\\approx 4.0$ bytes | 0.890 | 1.0x | 68.6 | 40.6 | 72.6 | 54.9 |
| **ByT5** 11 | 1B (Base) | 1.0 bytes | 0.950 | \>1.2x | \- | \- | \- | \- |
| **MegaByte** 18 | 1.1B | 4.0 bytes (Fixed) | 1.140 | 1.2x | 25.1 | 23.4 | 38.9 | 9.6 |
| **SpaceByte** 29 | 1.1B | Dynamic (Space) | 0.910 | 0.9x | 46.5 | 30.5 | 65.0 | 9.6 |
| **BLT-Entropy (ps=6)** 29 | 8B | 6.0 bytes | 0.810 | 0.6x | 65.7 | 43.2 | 73.7 | 51.3 |
| **BLT-Entropy (ps=8)** 29 | 8B | 8.0 bytes | **0.795** | **0.5x** | 64.1 | 41.5 | 71.9 | 49.0 |

Note: BLT models evaluated above operate in a FLOP-controlled regime (trained on 4T bytes). The Llama 3.1 baseline benefits from 15T tokens. Metrics derived from benchmark reports.12

### **3.1 Compute-Optimal Trajectories and Efficiency**

BLT is the first tokenizer-free architecture to match or slightly exceed the zero-shot capabilities of Llama 3 under strict compute-optimal training configurations.23 At an average patch size of 8 bytes, BLT operates at approximately 50% of the inference FLOPs required by equivalent BPE models, as the $\\mathcal{O}(K^2 d)$ global transformer operates on significantly fewer steps $K$. However, Table 1 reveals a measurable degradation in highly structured logic and coding datasets (HumanEval drops from 54.9 to 49.0). Pushing compression to $ps=8$ heavily bottlenecks the local encoder's capacity to embed complex syntax into a single $d=4096$ dimension vector, shifting the burden disproportionately to the global network.13

### **3.2 Orthographic Robustness and Morphological Processing**

Tokenization mapping introduces critical vulnerabilities to typographical permutations. On adversarial evaluation sets, BLT definitively outperforms BPE counterparts. When evaluated on noised iterations of HellaSwag—featuring casing alterations, typos, and character omissions—BLT models surpass the Llama 3 tokenizer baselines by a definitive average of \+8.0 accuracy points.5  
For fine-grained character manipulations, the CUTE benchmark (evaluating spelling, composition, and string manipulation) exhibits a massive delta, with BLT registering an advantage of over \+25.0 points against Llama 3 8B. In Grapheme-to-Phoneme (G2P) alignment tasks hosted by Phonology Bench, BLT correctly maps continuous text strings to phonemic components, vastly outperforming token-restricted models.5 The hash $n$-gram embeddings preserve the spatial adjacency of bytes, precluding the topological destruction caused by OOV subword fallbacks.9

## **4\. Failure Modes & Limitations**

Despite substantial algorithmic efficiency gains, the practical deployment of BLT is constrained by distinct hardware interaction bottlenecks, inference scheduling defects, and secondary module overheads.

### **4.1 Batch Scheduling Asynchrony and "Compute Bubbles"**

A critical failure mode of dynamic entropy patching resides in the sequence-length variance across batched execution. Standard autoregressive generation computes one token per step synchronously across a batch size $N$. In BLT, the length of the latent patch varies per sequence; Sequence $A$ may trigger an entropy boundary at 3 bytes, while Sequence $B$ requires 14 bytes.25  
The Latent Global Transformer relies on the Local Encoder completing the patch representation. Consequently, the scheduling engine must wait for the longest byte-sequence in the batch to finalize its patch boundary before executing the global forward pass. This mismatch generates "compute bubbles"—idle GPU tensor core cycles where sequences that reached their entropy threshold prematurely halt execution. Resolving this necessitates complex asynchronous decoding algorithms or forced patching thresholds that truncate execution, the latter of which artificially degrades the entropy logic and spikes BPB.25

### **4.2 Non-Standard Masking and FlexAttention Compilation Overhead**

While the localized cross-attention modules minimize theoretical FLOPs, modern GPU hardware achieves peak utilization exclusively on contiguous dense matrix multiplications optimized via FlashAttention-2. The local encoder and decoder depend on bespoke block-causal attention masks to bound the window $w\_\\mathcal{E}$ and invert query/key roles over variable lengths.13  
Executing these sparse, varying-length cross-attention interactions currently mandates utilizing PyTorch's FlexAttention API. Profiling reports indicate that FlexAttention suffers from compiler overhead, graph compilation bugs, and sub-optimal hardware utilization compared to standard causal masking.35 Consequently, the time-to-first-byte (TTFB) and inter-token latency (ms/token) often bottleneck at memory bandwidth saturation rather than compute saturation, diluting the 50% theoretical FLOP reduction in real-world deployments.1

### **4.3 Entropy Model Drift and Prefix Debt**

The reliance on an isolated, frozen auxiliary model $M\_\\theta$ for conditional entropy calculations introduces "prefix debt." As the generation context lengthens, the statistical predictability of the byte stream inherently shifts (entropy drift). A static hyperparameter threshold $\\tau\_H$ calibrated for the initial prompt sequence frequently results in excessively coarse patches as the context extends, forcing the latent dimensions to over-compress data.1 This boundary rigidity drives the noticeable degradation in performance on extended context logic tests like HumanEval (Table 1), as the non-differentiable $M\_\\theta$ fails to adaptively refine its confidence thresholds relative to the primary model's state.38

### **4.4 KV-Cache Fragmentation**

In traditional causal LLMs, the Key-Value (KV) cache grows linearly and contiguously. In the BLT architecture, the KV cache is heavily bifurcated. The Latent Global Transformer maintains a persistent cache of patch keys/values. Conversely, the Local Decoder reconstructs bytes dynamically per patch and flushes its byte-level KV cache upon crossing the entropy boundary $\\tau\_H$.27 This continuous allocation, initialization, and de-allocation of localized memory buffers prevents contiguous memory mapped database (NVMe) offloading 39, escalating memory fragmentation and demanding advanced paging algorithms (e.g., PagedAttention) engineered specifically for dual-scale resets.34

## **5\. Open Problems**

The paradigm shift toward continuous byte latent modeling introduces specific unresolved technical vectors actively debated in the literature:  
**1\. End-to-End Differentiable Patching:** The dependency on a hard-coded heuristic threshold $\\tau$ and an external language model $M\_\\theta$ isolates the segmentation policy from the downstream objective loss. Developing fully differentiable boundary detection algorithms—potentially leveraging Gumbel-Softmax discrete relaxations, reinforcement learning eviction policies (e.g., RazorAttention), or internal gradient routing—remains a paramount theoretical challenge.36  
**2\. Omnimodal Generalization:** While BLT succeeds over UTF-8 text, the true theoretical ceiling of token-free architecture is modality-agnostic omnimodal processing (raw audio WAV bytes, RGB pixel streams, and compiled binary executables). Determining if the scalar entropy boundary equation $H(x\_i) \> \\tau\_H$ naturally aligns with the high-variance continuous entropy topologies of audio or image arrays, without requiring modality-specific delimiter adaptations (e.g., SpaceByte), is under active investigation by frameworks like MBLM and bGPT.9  
**3\. Hardware-Software Co-Design for Dual-Scale Attention:** Bridging the gap between theoretical FLOP efficiency and empirical GPU utilization demands optimized kernels for nested, variable-length block masking. Open questions persist regarding the mathematical restructuring of cross-attention matrices to bypass FlexAttention graph compilation overheads, potentially mapping the hierarchical memory directly to SRAM/HBM hierarchies to prevent memory-bandwidth starvation during local-decoder unrolling.43  
**4\. Mitigating Information Loss at Large Patch Sizes:** Increasing the compression scalar (e.g., pushing to $ps=12$) consistently degrades zero-shot logic task performance. Resolving how to expand the representational capacity of a single $d$-dimensional patch vector $p\_k$ to encapsulate higher byte-level variance without proportionally inflating the $d\_{model}$ global dimension—perhaps through dynamic multi-vector projection per patch or MoE (Mixture of Experts) allocation—remains critically unsolved.44

## **6\. Critical References**

1. **Pagnoni et al. (2024)**. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Introduces the BLT framework, demonstrating dynamic entropy patching achieves 8B-parameter compute-parity with Llama 3 while halving inference FLOPs.  
2. **Yu et al. (2023)**. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers.* Establishes the foundation for hierarchical byte-level transformers utilizing fixed-stride spatial patching to achieve sub-quadratic attention over million-byte context windows.  
3. **Xue et al. (2022)**. *ByT5: Towards a token-free future with pre-trained byte-to-byte models.* Provides the empirical baseline for pure byte-to-byte encoder-decoder models, isolating their extreme robustness to noise at the cost of high sequence-length FLOP penalties.  
4. **Hwang et al. (2025)**. *H-Net: Dynamic Chunking for Hierarchical Sequence Modeling.* Explores temporally-dynamic downsampling and boundary detection logic specifically for nested modalities without relying on tokenizer heuristics.  
5. **Slagle (2024)**. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* Investigates modality-specific static patching tied to whitespace delimiters, performing competitively on natural text but degrading severely on generic byte sequences.  
6. **Neitemeier et al. (2025)**. *Hierarchical autoregressive transformers: Combining byte- and word-level processing for robust, adaptable language models.* Analyzes the batching synchronization and scheduling complexities inherent in variable-length patch generation protocols.  
7. **Wang et al. (2024)**. *MambaByte: Token-free selective state space model.* Adapts token-free byte-level processing to recurrent state-space models, validating continuous scaling independent of global/local transformer architectures.  
8. **Wu et al. (2024)**. *bGPT: Byte-level generative pre-training for omnimodal representations.* Extends static patching architectures into true omnimodality, proving byte-level models can autoregressively process audio, image, and raw CPU execution traces.  
9. **Liu et al. (2025)**. *SuperBPE: Efficient Tokenization via Boundary Awareness.* Contrasts the tokenizer-free approach by maximizing vocabulary scales into "superwords" spanning cross-space boundaries to forcibly reduce sequence lengths.  
10. **Dagan et al. (2024)**. *Getting the most out of your tokenizer for pre-training and domain adaptation.* Empirically quantifies the fragility of static tokenizers when transitioning models across distinct domains, validating the necessity of dynamic sequence representations.  
11. **Petrov et al. (2024)**. *Language model tokenizers introduce unfairness between languages.* Quantifies the "byte premium" where morphologically rich languages suffer disproportionate token fragmentation and mapping errors compared to English data.  
12. **Huang et al. (2025)**. *OverEncoding: Integrating N-gram structures directly into representation space.* Explores the integration of $N$-gram hash mappings directly into initial embeddings, providing the structural basis for BLT's local encoder hash augmentation.  
13. **Abeywickrama et al. (2025)**. *EntroPE: Entropy-based patching for time series.* Demonstrates the generalizability of entropy-guided sequence boundary detection by adapting the BLT thresholding model directly to continuous time-series forecasting.

#### **Works cited**

1. The Bitter Lesson is coming for Tokenization | ⛰️ lucalp, accessed April 10, 2026, [https://lucalp.dev/bitter-lesson-tokenization-and-blt/](https://lucalp.dev/bitter-lesson-tokenization-and-blt/)  
2. ByteFlow: Language Modeling through Adaptive Byte Compression without a Tokenizer, accessed April 10, 2026, [https://arxiv.org/html/2603.03583v1](https://arxiv.org/html/2603.03583v1)  
3. EvaByte: Efficient Byte-level Language Models at Scale \- HKU NLP Group, accessed April 10, 2026, [https://hkunlp.github.io/blog/2025/evabyte/](https://hkunlp.github.io/blog/2025/evabyte/)  
4. Beyond the Tokenization Bottleneck: A Conceptual Framework for Efficiency-Plasticity Trade-Offs in ASEAN Large Language Model A \- Cureus Journals, accessed April 10, 2026, [https://www.cureusjournals.com/articles/13958-beyond-the-tokenization-bottleneck-a-conceptual-framework-for-efficiency-plasticity-trade-offs-in-asean-large-language-model-adaptation.pdf](https://www.cureusjournals.com/articles/13958-beyond-the-tokenization-bottleneck-a-conceptual-framework-for-efficiency-plasticity-trade-offs-in-asean-large-language-model-adaptation.pdf)  
5. Paper Review: Byte Latent Transformer: Patches Scale Better Than Tokens | by Andrew Lukyanenko, accessed April 10, 2026, [https://artgor.medium.com/paper-review-byte-latent-transformer-patches-scale-better-than-tokens-18539c34f177](https://artgor.medium.com/paper-review-byte-latent-transformer-patches-scale-better-than-tokens-18539c34f177)  
6. \[2412.09871\] Byte Latent Transformer: Patches Scale Better Than Tokens \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2412.09871](https://arxiv.org/abs/2412.09871)  
7. A Linguistic Approach to Crosslingual and Multilingual NLP \- UC San Diego, accessed April 10, 2026, [https://escholarship.org/content/qt8k37q7j4/qt8k37q7j4.pdf](https://escholarship.org/content/qt8k37q7j4/qt8k37q7j4.pdf)  
8. (PDF) Byte Latent Transformer: Patches Scale Better Than Tokens \- ResearchGate, accessed April 10, 2026, [https://www.researchgate.net/publication/387078383\_Byte\_Latent\_Transformer\_Patches\_Scale\_Better\_Than\_Tokens](https://www.researchgate.net/publication/387078383_Byte_Latent_Transformer_Patches_Scale_Better_Than_Tokens)  
9. Byte Language Models: A Tokenization-Free Approach, accessed April 10, 2026, [https://www.emergentmind.com/topics/byte-language-models-blms](https://www.emergentmind.com/topics/byte-language-models-blms)  
10. Tokenization Deep Dive: Why It Matters More Than You Think \- Let's Data Science, accessed April 10, 2026, [https://letsdatascience.com/blog/tokenization-deep-dive-why-it-matters-more-than-you-think](https://letsdatascience.com/blog/tokenization-deep-dive-why-it-matters-more-than-you-think)  
11. Byte Latent Transformer (BLT) \- Emergent Mind, accessed April 10, 2026, [https://www.emergentmind.com/topics/byte-latent-transformer-blt](https://www.emergentmind.com/topics/byte-latent-transformer-blt)  
12. A Comparative Analysis of Byte-Level and Token-Level Transformer Models in Natural Language Processing \- Greg Robison, accessed April 10, 2026, [https://gregrobison.medium.com/a-comparative-analysis-of-byte-level-and-token-level-transformer-models-in-natural-language-9fb4331b6acc](https://gregrobison.medium.com/a-comparative-analysis-of-byte-level-and-token-level-transformer-models-in-natural-language-9fb4331b6acc)  
13. Byte Latent Transformer: Patches Scale Better Than Tokens \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2412.09871v1](https://arxiv.org/html/2412.09871v1)  
14. This might be a dumb question but how many bits are in a token? : r/LocalLLaMA \- Reddit, accessed April 10, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1hl3iwa/this\_might\_be\_a\_dumb\_question\_but\_how\_many\_bits/](https://www.reddit.com/r/LocalLLaMA/comments/1hl3iwa/this_might_be_a_dumb_question_but_how_many_bits/)  
15. A Comprehensive Guide to Byte Latent Transformer Architecture \- DigitalOcean, accessed April 10, 2026, [https://www.digitalocean.com/community/tutorials/what-is-byte-latent-transformer](https://www.digitalocean.com/community/tutorials/what-is-byte-latent-transformer)  
16. Spend Your FLOPs Wisely: December Papers \- Graphcore, accessed April 10, 2026, [https://www.graphcore.ai/posts/spend-your-flops-wisely-december-papers](https://www.graphcore.ai/posts/spend-your-flops-wisely-december-papers)  
17. Inputs to Byte Latent Transformer \- Sagar Sarkale, accessed April 10, 2026, [https://sagarsarkale.com/blog/genai/inputs-to-byte-latent-transformer/](https://sagarsarkale.com/blog/genai/inputs-to-byte-latent-transformer/)  
18. PptxGenJS Presentation, accessed April 10, 2026, [https://www.cs.toronto.edu/\~cmaddis/courses/csc2541\_w25/presentations/li\_hsu\_bytelatent.pdf](https://www.cs.toronto.edu/~cmaddis/courses/csc2541_w25/presentations/li_hsu_bytelatent.pdf)  
19. Byte Latent Transformer (BLT) \- Hugging Face, accessed April 10, 2026, [https://huggingface.co/docs/transformers/en/model\_doc/blt](https://huggingface.co/docs/transformers/en/model_doc/blt)  
20. Byte Latent Transformer: Improved Transformer architecture for LLMs | by Mehul Gupta | Data Science in Your Pocket | Medium, accessed April 10, 2026, [https://medium.com/data-science-in-your-pocket/byte-latent-transformer-improved-transformer-architecture-for-llms-f1589e15dd21](https://medium.com/data-science-in-your-pocket/byte-latent-transformer-improved-transformer-architecture-for-llms-f1589e15dd21)  
21. PatchDNA: A Flexible and Biologically-Informed Alternative to Tokenization for DNA, accessed April 10, 2026, [https://www.biorxiv.org/content/10.1101/2025.11.28.691095v2.full-text](https://www.biorxiv.org/content/10.1101/2025.11.28.691095v2.full-text)  
22. Byte Latent Transformer (BLT) by Meta AI: A Tokenizer-free LLM \- AI Papers Academy, accessed April 10, 2026, [https://aipapersacademy.com/byte-latent-transformer/](https://aipapersacademy.com/byte-latent-transformer/)  
23. Byte Latent Transformer: Patches Scale Better Than Tokens | by DrKilngon | Medium, accessed April 10, 2026, [https://medium.com/@DrKilngon/byte-latent-transformer-patches-scale-better-than-tokens-f40106f8dd3c](https://medium.com/@DrKilngon/byte-latent-transformer-patches-scale-better-than-tokens-f40106f8dd3c)  
24. BLT Deep Dive: Hello bytes, goodbye tokens | by Kevin Rohling | Medium, accessed April 10, 2026, [https://medium.com/@krohling/blt-deep-dive-hello-bytes-goodbye-tokens-315ef8032668](https://medium.com/@krohling/blt-deep-dive-hello-bytes-goodbye-tokens-315ef8032668)  
25. A Family of LLMs Liberated from Static Vocabularies \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2603.15953v1](https://arxiv.org/html/2603.15953v1)  
26. Precursors to Byte Latent Transformer | Sagar Sarkale, accessed April 10, 2026, [https://sagarsarkale.com/blog/genai/precursors-to-byte-latent-transformer/](https://sagarsarkale.com/blog/genai/precursors-to-byte-latent-transformer/)  
27. \[D\] In Byte Latent Transformer, how is the decoded patch boundary determined? \- Reddit, accessed April 10, 2026, [https://www.reddit.com/r/MachineLearning/comments/1hli20i/d\_in\_byte\_latent\_transformer\_how\_is\_the\_decoded/](https://www.reddit.com/r/MachineLearning/comments/1hli20i/d_in_byte_latent_transformer_how_is_the_decoded/)  
28. MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers | Request PDF, accessed April 10, 2026, [https://www.researchgate.net/publication/401460638\_MEGABYTE\_Predicting\_Million-byte\_Sequences\_with\_Multiscale\_Transformers](https://www.researchgate.net/publication/401460638_MEGABYTE_Predicting_Million-byte_Sequences_with_Multiscale_Transformers)  
29. From Bytes to Ideas: Language Modeling with Autoregressive U-Nets \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2506.14761v1](https://arxiv.org/html/2506.14761v1)  
30. Overcome the limitations of traditional tokenization-based models | by Abhishek Karmakar, accessed April 10, 2026, [https://medium.com/@karmakarabhishek5/overcome-the-limitations-of-traditional-tokenization-based-models-988bc7284840](https://medium.com/@karmakarabhishek5/overcome-the-limitations-of-traditional-tokenization-based-models-988bc7284840)  
31. Byte Latent Transformer: Patches Scale Better Than Tokens | Kingy AI, accessed April 10, 2026, [https://kingy.ai/wp-content/uploads/2024/12/Byte-Latent-Transformer-Patches-Scale-Better-Than-Tokens-Paper-Summary.pdf](https://kingy.ai/wp-content/uploads/2024/12/Byte-Latent-Transformer-Patches-Scale-Better-Than-Tokens-Paper-Summary.pdf)  
32. From Tokens to Patches: Byte Latent Transformers for Spelling-Intensive Tasks \- Medium, accessed April 10, 2026, [https://medium.com/tr-labs-ml-engineering-blog/from-tokens-to-patches-byte-latent-transformers-for-spelling-intensive-tasks-1aa6a632276f](https://medium.com/tr-labs-ml-engineering-blog/from-tokens-to-patches-byte-latent-transformers-for-spelling-intensive-tasks-1aa6a632276f)  
33. Trace \- lecture\_01 \- Stanford CS336, accessed April 10, 2026, [https://cs336.stanford.edu/lectures/?trace=lecture\_01](https://cs336.stanford.edu/lectures/?trace=lecture_01)  
34. Trustworthy and Efficient LLMs Meet Databases \- arXiv, accessed April 10, 2026, [https://arxiv.org/pdf/2412.18022](https://arxiv.org/pdf/2412.18022)  
35. Multi-Token Attention | Hacker News, accessed April 10, 2026, [https://news.ycombinator.com/item?id=43562384](https://news.ycombinator.com/item?id=43562384)  
36. Transformers, accessed April 10, 2026, [https://aarnphm.xyz/thoughts/Transformers](https://aarnphm.xyz/thoughts/Transformers)  
37. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2507.07955v2](https://arxiv.org/html/2507.07955v2)  
38. ANCHORED DECODING: PROVABLY REDUCING COPYRIGHT RISK FOR ANY LANGUAGE MODEL \- OpenReview, accessed April 10, 2026, [https://openreview.net/pdf?id=63Is26x1qu](https://openreview.net/pdf?id=63Is26x1qu)  
39. NeurIPS Poster Scaling Embedding Layers in Language Models, accessed April 10, 2026, [https://neurips.cc/virtual/2025/poster/116760](https://neurips.cc/virtual/2025/poster/116760)  
40. Trustworthy and Efficient LLMs Meet Databases \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2412.18022v1](https://arxiv.org/html/2412.18022v1)  
41. Dynamic Tokenization via Reinforcement Patching: End-to-end Training and Zero-shot Transfer \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2603.26097v1](https://arxiv.org/html/2603.26097v1)  
42. Multiscale Byte Language Models A Hierarchical Architecture for Causal Million-Length Sequence Modeling \- arXiv.org, accessed April 10, 2026, [https://arxiv.org/html/2502.14553](https://arxiv.org/html/2502.14553)  
43. Scaling Embedding Layers in Language Models \- OpenReview, accessed April 10, 2026, [https://openreview.net/pdf?id=gH4BRa4ZP3](https://openreview.net/pdf?id=gH4BRa4ZP3)  
44. Length-MAX Tokenizer for Language Models \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2511.20849v1](https://arxiv.org/html/2511.20849v1)  
45. ConceptMoE: Adaptive Token-to-Concept Compression for Implicit Compute Allocation, accessed April 10, 2026, [https://www.researchgate.net/publication/400237399\_ConceptMoE\_Adaptive\_Token-to-Concept\_Compression\_for\_Implicit\_Compute\_Allocation](https://www.researchgate.net/publication/400237399_ConceptMoE_Adaptive_Token-to-Concept_Compression_for_Implicit_Compute_Allocation)  
46. THE EFFICIENCY GAP IN BYTE MODELING \- OpenReview, accessed April 10, 2026, [https://openreview.net/pdf/acf781bc76df838197fea3c96e2b787cb42f9cdf.pdf](https://openreview.net/pdf/acf781bc76df838197fea3c96e2b787cb42f9cdf.pdf)