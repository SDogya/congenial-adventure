# **Ordered Action Tokenization (OAT): Algorithmic Mechanics, Information-Theoretic Trade-offs, and Scaling Bottlenecks**

## **Problem Formulation**

### **Markov Decision Processes and Continuous Control Factorization**

The integration of sequence modeling architectures into robotic manipulation requires framing physical control within a discrete autoregressive distribution. In a fully observable Markov Decision Process defined by the tuple $\\mathcal{M} \= \\langle \\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma \\rangle$, the system state is represented by an observation history $o\_{1:H\_o} \\in \\mathcal{S}$. The policy network $\\pi\_\\theta$ must output a corresponding sequence of optimal continuous actions. To mitigate the compounding temporal variance inherent in step-by-step physical execution, contemporary manipulation paradigms utilize action chunking. The policy predicts an open-loop kinematic sequence, denoted as an action chunk $a\_{1:H\_a} \\in \\mathbb{R}^{H\_a \\times D\_a}$, where $H\_a$ dictates the temporal prediction horizon and $D\_a$ represents the dimensionality of the actuation vector (e.g., end-effector poses, Euler angles, and gripper states).1  
Because transformer-based architectures inherently operate over categorical probability distributions optimized via cross-entropy loss, applying autoregressive generation to robotic control requires discretizing the continuous manifold $\\mathbb{R}^{H\_a \\times D\_a}$.2 The probability of executing a specific action chunk must be factorized over discrete states. This necessitates an intermediate mapping interface, functioning as a structural bridge between the continuous physical dynamics and the discrete token space modeled by the transformer.

### **The Action Tokenization Mapping $\\mathcal{T}$**

Action tokenization formalizes a deterministic mathematical mapping $\\mathcal{T} : \\mathbb{R}^{H\_a \\times D\_a} \\to \\mathcal{V}^{H\_l}$, which projects the continuous physical trajectory into a sequence of $H\_l$ discrete tokens drawn from a finite categorical vocabulary $\\mathcal{V}$.1 A corresponding detokenizer $\\mathcal{T}^{-1} : \\mathcal{V}^{H\_l} \\to \\mathbb{R}^{H\_a \\times D\_a}$ serves as the inverse function, recovering the approximated continuous chunk $\\hat{a}\_{1:H\_a}$ for closed-loop execution by the low-level motor controllers.1  
The factorization of the autoregressive policy $\\pi\_\\theta$ over the tokenized space is strictly defined by the chain rule of probability. Given the observation history $o\_{1:H\_o}$, the policy models the joint distribution of the latent token sequence as:

$$P\_\\theta(T\_{1:H\_l} | o\_{1:H\_o}) \= \\prod\_{i=1}^{H\_l} P\_\\theta(T\_i | T\_{\<i}, o\_{1:H\_o})$$  
where each $T\_i \\in \\mathcal{V}$ represents an individual generated token.3

### **Baseline Discretization Failures: Analytical vs. Latent Approaches**

Existing systems generally implement the mapping $\\mathcal{T}$ through either analytical discretization or unstructured latent quantization, both of which introduce severe mathematical and operational bottlenecks.2  
Analytical discretization methods, heavily utilized in architectures like Gato 4, RT-1, and OpenVLA 6, construct the token space by independently partitioning each dimension of the action vector. For example, OpenVLA discretizes each of the $D\_a$ dimensions into 256 uniform bins 7, while Gato utilizes a $\\mu$-law transformation to encode continuous joint torques into the range $\[-1, 1\]$, subsequently quantizing them into 1024 bins.4 This naive approach results in an unstructured, non-hierarchical serialization where the latent sequence length becomes a direct multiple of the continuous horizon: $H\_l \= H\_a \\times D\_a$. In standard manipulation environments with multi-degree-of-freedom robotic arms, this analytical mapping inflates $H\_l$ to sequence lengths exceeding 300 tokens per chunk.1 Such massive context expansion dramatically increases the computational complexity of the autoregressive generation ($O(N^2)$ attention overhead) and induces severe inference latency ($\>500$ ms).1  
Conversely, latent quantization techniques, such as QueST 8 and Vector-Quantized Behavior Transformers (VQ-BeT) 9, utilize learned autoencoders to compress the action chunk into a compact latent space before vector quantization. While solving the sequence length inflation issue, these methods fail to construct a structured dependency graph across the generated tokens. The resulting latent space is unordered and symmetric, meaning the mutual information distributed across the tokens $T\_1 \\dots T\_{H\_l}$ follows no predictable hierarchy.2 This symmetry explicitly violates the sequential, left-to-right conditional dependency structure optimized during next-token prediction.2

### **The Three Strict Tokenization Desiderata**

To construct a token manifold that organically aligns with the chain-rule factorization of autoregressive policies without violating physical execution constraints, the tokenizer $\\mathcal{T}$ must simultaneously satisfy three strict mathematical conditions 2:

1. **P1. High Compression:** The mapping function must project the input data into a significantly lower-dimensional latent sequence such that $H\_l \\ll H\_a \\times D\_a$.2 This bounds the inference FLOPs, minimizes the self-attention context footprint, and prevents temporal compounding errors caused by excessive autoregressive rollout steps.  
2. **P2. Total Decodability:** The detokenizer $\\mathcal{T}^{-1}$ must mathematically guarantee a total function across the entirety of the vocabulary space $\\mathcal{V}^{H\_l}$.2 Every arbitrary combination of generated tokens must decode to a bounded, valid continuous action sequence. Partial decodability paradigms violate this condition by allowing specific out-of-distribution token permutations to map to undefined states, undefined coordinates, or catastrophic kinematic singularities that trigger low-level safety halts in physical robots.  
3. **P3. Causal Ordering:** The token representation must enforce a strict left-to-right information hierarchy.2 The mutual information between the generated sequence prefix and the continuous physical action must monotonically increase, functioning as an implicit progressive coding scheme.8 Mathematically, $I(T\_{1:k}; a\_{1:H\_a}) \> I(T\_{1:k-1}; a\_{1:H\_a})$ for all $k \\in \\{1, \\dots, H\_l\\}$. Early tokens in the sequence must capture high-variance global structure (dominant trajectory modes), while later tokens must conditionally capture low-variance residual features (fine-grained kinematic adjustments).1

Ordered Action Tokenization (OAT) provides a closed-form architectural solution formulated to satisfy all three constraints simultaneously, yielding a causally ordered, highly compressed, totally decodable sequence mapping.2

## **Key Methods**

The OAT framework discretizes the action sequence $a\_{1:H\_a}$ via a structured autoencoding architecture that incorporates a transformer-based causal masking matrix, Finite Scalar Quantization (FSQ), and a specific nested dropout objective designed to mathematically enforce the P3 causal ordering constraint.2

### **The OAT Autoencoder Architecture**

The architecture comprises a 2-layer transformer encoder ($\\mathcal{E}\_\\phi$) and a 4-layer transformer decoder ($\\mathcal{D}\_\\theta$).11 To standardize representational capacity across empirical validation, both the encoder and decoder are parameterized with a model dimension of $D\_{model} \= 256$ and a multi-head attention topology utilizing $H\_{head} \= 64$.11  
The input to the encoder $\\mathcal{E}\_\\phi$ is constructed by concatenating the continuous physical action chunk $a\_{1:H\_a}$ with a sequence of learnable register tokens $r\_{1:H\_l} \\in \\mathbb{R}^{D\_{model}}$.2 The system sets the latent target horizon to $H\_l \= 8$.11 The concatenated sequence is processed through the encoder's self-attention layers to project the kinematic trajectory data into the registers:

$$z\_{1:H\_l} \= \\mathcal{E}\_\\phi(a\_{1:H\_a} \\oplus r\_{1:H\_l})$$  
where $z\_{1:H\_l}$ represents the continuous register latents prior to quantization.2

### **Causal Masking of Register Tokens**

To prevent bidirectional information leakage across the register tokens and strictly enforce the temporal progressive hierarchy defined by P3, OAT modifies the standard self-attention operation within the encoder $\\mathcal{E}\_\\phi$.8 A deterministic causal attention mask $M \\in \\{-\\infty, 0\\}^{H\_l \\times H\_l}$ is applied exclusively to the register-to-register attention interactions.14  
The mask matrix is structured such that the $i$-th register token $r\_i$ can only attend to the $j$-th register token if the condition $i \\ge j$ is met.8

$$\\text{Attention}(Q\_r, K\_r, V\_r) \= \\text{softmax}\\left(\\frac{Q\_r K\_r^T}{\\sqrt{D\_{head}}} \+ M\\right) V\_r$$  
This intervention guarantees that $r\_1$ functions as an independent, globally-aware vector summarizing the entire trajectory $a\_{1:H\_a}$, while $r\_2$ conditions its representation exclusively on the trajectory and the residual features already encoded by $r\_1$. This causal masking acts as the primary architectural enforcer of left-to-right information flow.8

### **Finite Scalar Quantization (FSQ) Mechanics**

Following the extraction of the continuous register latents $z\_{1:H\_l}$, the vectors are subjected to a discrete information bottleneck. OAT abandons standard Vector Quantization (VQ-VAE) methodologies, which maintain learnable codebooks updated via moving averages or auxiliary commitment losses.15 Instead, it employs Finite Scalar Quantization (FSQ).2  
FSQ directly projects continuous scalar variables onto a deterministically bounded integer grid through implicit rounding, eliminating codebook collapse and index collapse pathologies typical in latent quantization.15 OAT parameterizes the latent dimension width to $D\_l \= 4$ per token and applies specific FSQ binning levels defined as $L \= $ across the four dimensions.11  
The effective implicit codebook size $|\\mathcal{V}|$ is the product of the defined grid levels:

$$|\\mathcal{V}| \= 8 \\times 5 \\times 5 \\times 5 \= 1000$$  
This quantization topology limits the absolute informational capacity of each generated token chunk to an upper bound of $I(Z; T) \\le 80$ bits.15 The quantization operation is executed as:

$$\\hat{z}\_{1:H\_l} \= \\text{FSQ}(z\_{1:H\_l})$$  
where $\\hat{z}$ denotes the discrete token representation.1

### **Nested Dropout and Implicit Progressive Coding**

While the causal mask $M$ restricts cross-register attention, it does not mathematically guarantee that the optimization landscape prioritizes embedding global information in $r\_1$ over $r\_8$. To ensure the causal ordering (P3) is rigorously embedded into the latent representation, OAT modifies the standard autoencoding loss via nested dropout.1  
During the training phase, a truncation index $K$ is stochastically sampled from a specified prior distribution $p(\\cdot)$ over the range $\[H\_l\]$.2 The quantized sequence $\\hat{z}\_{1:H\_l}$ is subjected to tail dropout: tokens positioned at indices from $K+1$ through $H\_l$ are discarded and deterministically replaced by a universally broadcast, learnable $\\langle \\texttt{MASK} \\rangle$ token.1

$$\\hat{z}\_{masked} \= \\hat{z}\_{1:K} \\oplus \\langle \\texttt{MASK} \\rangle\_{K+1:H\_l}$$  
The conditional decoder $\\mathcal{D}\_\\theta$ processes this artificially truncated sequence to output the reconstructed action chunk $\\hat{a}\_{1:H\_a}$.2 The network parameters $\\{ \\phi, r, \\theta, \\texttt{MASK} \\}$ are iteratively updated to minimize the standard Mean Squared Error (MSE) reconstruction loss via gradient descent with learning rate $\\eta$:

$$\\mathcal{L} \= \\|\\hat{a}\_{1:H\_a} \- a\_{1:H\_a}\\|\_2^2$$

$$\\{ \\phi, r, \\theta, \\texttt{MASK} \\} \\leftarrow \\{ \\phi, r, \\theta, \\texttt{MASK} \\} \- \\eta \\nabla \\mathcal{L}$$  
This optimization objective forces the model into an implicit Principal Component Analysis (PCA)-style compression regime.2 Because the decoder must constantly reconstruct the full horizon $H\_a$ under stochastically constrained partial information (prefixes), the tokenizer is mathematically coerced to allocate the highest-variance, lowest-frequency kinematic features (the global path structure) to the earliest tokens.11 Subsequent tokens are reserved exclusively for modeling high-frequency, low-variance residuals (fine-grained pose adjustments).3 This mechanism organically aligns the token generation priority with Shannon's optimal code length principles.3

### **Detokenization and Autoregressive Inference Algorithm**

Because nested dropout trains the decoder $\\mathcal{D}\_\\theta$ to act as a robust total function over any padded prefix combination, OAT fundamentally supports variable-length, anytime prefix-based detokenization during autoregressive execution.1  
The inference logic proceeds as follows 1:

1. Given observation history $o\_{1:H\_o}$, the detokenizer $\\mathcal{T}^{-1} \= \\{\\mathcal{D}\_\\theta, \\texttt{MASK}\\}$, and a target generation depth $K \\le H\_l$.  
2. Initialize an empty token prefix sequence $T\_{1:K} \\leftarrow \\emptyset$.  
3. For step $i \= 1$ to $K$:  
   * Sample the next token $T\_i \\sim \\pi(\\cdot | T\_{\<i}, o\_{1:H\_o})$  
   * Append $T\_i$ to $T\_{1:K}$  
4. Pad the sequence tail using the learned mask: $T\_{1:H\_l} \\leftarrow T\_{1:K} \\oplus \\langle \\texttt{MASK} \\rangle\_{K+1:H\_l}$  
5. Execute the detokenization pass through the decoder to recover the control signals: $\\hat{a}\_{1:H\_a} \\leftarrow \\mathcal{T}^{-1}(T\_{1:H\_l})$

This algorithmic property yields a precise, linear trade-off between computational inference latency (determined by the scalar limit $K$) and kinematic action fidelity.1 Terminating generation at $K=2$ produces a highly compressed, rapid execution of generalized motion vectors, whereas extending generation to $K=8$ demands higher computational overhead to calculate precise insertion trajectories.12

### **Competing Methodologies: FAST, VQ-BeT, and QueST**

To contextualize the architectural advantages of OAT, it is necessary to examine the mechanistic limitations of competing tokenization models.  
Frequency-space Action Sequence Tokenization (FAST) 17 achieves compression by mapping continuous trajectories into the frequency domain via a Discrete Cosine Transform (DCT). FAST truncates high-frequency spectral coefficients to reduce the sequence length.17 However, the DCT mapping is not a total function over random token sequences. Because truncation indiscriminately severs high-frequency spatial dependencies, arbitrary generation sequences map to kinematically invalid combinations in the physical domain. FAST severely violates the P2 constraint (Total Decodability), resulting in partial decodability risks where policies output erratic, unsafe joint torques during physical rollout.1  
Vector-Quantized Behavior Transformer (VQ-BeT) 9 utilizes Residual Vector Quantization (RVQ) to compress action states. While RVQ inherently defines a hierarchical codebook structure (where subsequent codebooks encode the residual errors of prior codebooks), the model suffers from codebook collapse—a common failure in standard VQ paradigms where the continuous distribution bypasses significant portions of the discrete grid.16 Furthermore, VQ-BeT lacks the explicit temporal dependency constraints over its generated tokens, preventing stable autoregressive factorization over long horizons.10  
QueST 8 employs a learned latent quantizer generating fixed-length sequences (e.g., exactly 8 tokens). Without nested dropout or causal register masking, QueST generates an unstructured, homogeneous sequence. The latent space does not prioritize global features over residuals, meaning $T\_1$ possesses the same informational variance distribution as $T\_8$. This symmetric sequence creates compounding entropy during autoregressive generation, as the probability distribution $P\_\\theta(T\_i | T\_{\<i}, o\_{1:H\_o})$ cannot rely on a coarse-to-fine inductive bias to prune low-probability coordinate branches.11 Furthermore, QueST cannot utilize prefix-based early stopping, as truncating an un-ordered sequence results in catastrophic reconstruction failure.1

## **Quantitative Landscape**

The empirical validation of OAT evaluates the intersection of inference latency, data compression metrics, and simulated/real-world closed-loop success rates. To isolate the direct impact of the tokenization mapping, all compared autoregressive policies (Bin, FAST, QueST, OAT) utilize the exact same backbone architecture: a 4-layer transformer decoder with $D\_{model}=256$ and $H\_{head}=64$.11 Comparisons against non-autoregressive paradigms utilize Diffusion Policy (DP) parameterized via a ResNet-18 or SigLIP vision backbone.11

### **Inference Latency and Compression Limits**

The deterministic token sequence length defines the strict boundary on sequential generation latency. The per-dimension binning methodologies (Bin) produce massive $H\_l$ counts that prohibit high-frequency real-time execution. OAT reduces the $H\_l$ requirement by a factor of 48 compared to upper-bound Binning while matching QueST in base capacity.  
| Tokenization Method | Latent Horizon ($H\_l$) | Codebook / Vocabulary Size ($|\\mathcal{V}|$) | Action Chunk Inference Latency (ms) 1 | | :--- | :--- | :--- | :--- | | **Bin** 8 | 224 – 384 | 256 per dimension | 517.0 – 888.0 | | **FAST** 8 | Variable (\~8–16) | N/A (DCT coefficient states) | Variable (Low) | | **QueST** 1 | 8 | 1000 | 27.4 | | **OAT 8** (Full Decode) 8 | 8 | 1000 | 27.4 | | **OAT 4** (Prefix Decode) | 4 | 1000 | \~13.7 (Calculated $O(K)$ bounded) |

### **Simulation Benchmark Methodologies**

OAT was evaluated across four standard simulated manipulation suites: LIBERO 1, RoboMimic 1, MetaWorld 1, and RoboCasa.8 These environments isolate varying degrees of kinematic complexity, ranging from simple spatial traversals (MetaWorld) to high-precision insertions and long-horizon multi-object interactions (LIBERO, RoboMimic). The evaluation records the mean task success rate compiled over 5 independent seed initializations with 50 closed-loop evaluation rollouts per seed per task.2

### **Quantitative Results: Simulation**

The success rate matrix confirms a strictly monotonic performance gradient for OAT relative to the generation depth $K$. OAT$\_K$ indicates the policy was truncated to generate exactly $K$ tokens prior to padding and detokenization.1

| Method / Policy | Tokenization Depth | LIBERO (%) | RoboMimic (%) | MetaWorld (%) |
| :---- | :---- | :---- | :---- | :---- |
| **Diffusion Policy (DP)** | Continuous / Continuous | 57.6 ± 1.2 (SigLIP) | 71.4 ± 1.0 | 25.1 ± 0.8 |
| **Bin** | Full | 14.4 ± 0.6 | 39.5 ± 1.2 | 14.5 ± 0.7 |
| **FAST** | Full | 23.0 ± 0.5 | 24.0 ± 1.5 | 7.1 ± 0.7 |
| **QueST** | Full ($K=8$) | 48.2 ± 0.6 | 66.9 ± 0.8 | 17.9 ± 0.9 |
| **OAT 1** | Prefix ($K=1$) | 11.7 ± 0.7 | 50.8 ± 1.4 | 11.3 ± 0.4 |
| **OAT 2** | Prefix ($K=2$) | 39.8 ± 0.5 | 52.5 ± 1.2 | 16.4 ± 0.3 |
| **OAT 4** | Prefix ($K=4$) | 46.4 ± 0.6 | 65.3 ± 0.9 | *N/A* |
| **OAT 8** | Prefix ($K=8$) | 56.3 ± 1.0 | 73.1 ± 0.5 | 24.4 ± 0.6 |

OAT 8 effectively establishes state-of-the-art results for autoregressive generation, tracking or exceeding the non-autoregressive Diffusion Policy baselines across multiple domains.3 The performance of FAST completely collapses on MetaWorld (7.1%), corroborating the hypothesis that spectral coefficient truncation produces highly invalid physical bounds in coordinate-sensitive tasks.1 QueST plateaus significantly below OAT 8 on LIBERO (48.2% vs 56.3%), verifying the efficiency limits of unordered token predictions.3

### **Physical Rollout Dynamics and Validation**

Real-world validation utilized a fixed-base tabletop robotic arm executing two discrete primitives: "Pick & Place Ball" and "Stack Cups".3 Objects were initialized at randomized coordinates across the table surface. Execution fidelity was measured via binary task completion across 20 independent trials per task.11

| Method | Pick & Place Ball (Successes / 20\) | Stack Cups (Successes / 20\) |
| :---- | :---- | :---- |
| **Diffusion Policy (DP)** | 14 | 11 |
| **Bin** | 4 | 8 |
| **FAST** | 8 | 6 |
| **QueST** | 11 | 8 |
| **OAT 1** | 7 | 3 |
| **OAT 2** | 11 | 9 |
| **OAT 4** | 13 | 12 |
| **OAT 8** | 16 | 16 |

The physical rollout trajectory characteristics diverge sharply depending on the tokenization matrix. Bin generates excessively jittery motion profiles due to the sequence latency causing continuous $H\_a$ action chunk overlaps to desynchronize from the true state matrix.11 FAST produces erratic, overly aggressive acceleration curves, triggering safety threshold halts.11 Conversely, OAT demonstrates highly smoothed continuous motion splines, with insertion precision scaling proportionately to $K \\to 8$.3

### **Ablation Analysis: The Mathematical Necessity of Ordering**

To causally isolate the effect of the P3 condition (Causal Ordering) on policy stability, an architectural ablation test (referred to as OAT$*{\\times}$) was executed.\[2, 3, 11\] The OAT$*{\\times}$ topology utilized the exact dimension bounds ($H\_l=8$, $|\\mathcal{V}|=1000$) and identical attention structures but disabled the nested dropout mechanism $p(\\cdot)$ during training, thus producing a symmetric, unordered latent manifold equivalent in behavior to QueST.2

| Ablation Condition | Token Ordering Status | Nested Dropout Mechanism | LIBERO Success Rate (%) |
| :---- | :---- | :---- | :---- |
| **OAT 8** (Full Model) | P3 Enforced | Active | 56.3 ± 1.0 |
| **OAT$\_{\\times}$** (Ablated) | Unordered Space | Disabled | 35.2 |
| **QueST** | Unordered Space | Disabled | 48.2 ± 0.6 |

The removal of the dropout distribution collapsed the LIBERO success rate by 21.1 absolute percentage points.8 The ablated OAT$\_{\\times}$ structure actually underperforms the QueST baseline, verifying that standard FSQ quantization parameters are insufficient for autoregressive mapping unless explicit progressive sequence ordering forces an inductive bias across the transformer heads.3 The causal hierarchy fundamentally reduces sequence entropy, constraining the vast categorical probability space at generation step $i$ using the deterministic spatial boundaries established at step $i-1$.19

## **Failure Modes & Limitations**

### **The Compression Gap: Information-Theoretic Scaling Bounds**

While OAT elegantly resolves the autoregressive sequence alignment problem, it introduces an immutable limit on Vision-Language-Action (VLA) scaling dynamics. The phenomenon, codified analytically as the *Compression Gap* (Shiba, 2026\) 15, exposes a fundamental information-theoretic bottleneck inherent to all discrete tokenization regimes.  
The mapping from physical observation to execution traverses a defined Markov chain: $O \\to Z\_v \\to T \\to A$, where $O$ represents the vision-language observation sequence, $Z\_v$ is the continuous visual representation output by the vision encoder, $T$ is the discrete action token manifold, and $A$ is the final kinematic action.15 By the Data Processing Inequality, the mutual information transferred to the robotic execution is strictly bounded by the tightest intermediate constraint:

$$I(O; A) \\le \\min(I(O; Z\_v), I(Z\_v; T))$$  
For the OAT architecture, the capacity limit is statically defined by the FSQ fixed codebook. Utilizing $H\_l \= 8$ and an FSQ grid resolving $|\\mathcal{V}| \= 1000$, the tokenizer imposes a mathematically rigid ceiling: $I(Z\_v; T) \\le 80$ bits per action chunk.15  
When the vision encoder is upgraded—for instance, transitioning from a low-dimensional ResNet-18 (64-dim) to high-capacity architectures like SigLIP (1152-dim) or DINOv2 ViT-L/14 (1024-dim)—the representational capacity $I(O; Z\_v)$ increases dramatically.15 However, this scaling fails to propagate through OAT.

### **Vision Encoder Bottlenecks and Parameter Saturation**

A factorial experiment conducted on the LIBERO-10 suite isolates the exact threshold of the Compression Gap. In continuous action pathways utilizing score-matching (Diffusion Policy, DP), no discrete bottleneck exists ($I(Z\_v; T) \\to \\infty$). Therefore, the vision encoder $I(O; Z\_v)$ acts as the binding constraint. Upgrading the vision backbone propagates end-to-end, yielding strictly monotonic performance curves.15

| Vision Encoder Architecture | Output Dimensionality | DP M-Size Success Rate (%) | OAT M-Size Success Rate (%) |
| :---- | :---- | :---- | :---- |
| **ResNet-18** | 64 | 36.4 | 53.8 |
| **SigLIP** | 1152 | 57.6 | 57.4 |
| **SigLIP 2** | 1152 | 62.8 | 44.2 |
| **DINOv2 ViT-L/14** | 1024 | 63.8 | 51.0 |

As detailed in the matrix above, Diffusion Policy tracks visual representation quality perfectly, scaling from 36.4% to 63.8%.15 Conversely, OAT plateaus and fluctuates randomly within a constrained margin (53.8% $\\to$ 51.0%). Because the 80-bit codebook bottleneck becomes the new binding constraint ($I(Z\_v; T) \< I(O; Z\_v)$), downstream task performance cannot improve, regardless of the upstream visual richness.15  
Furthermore, simply increasing the VLA transformer parameter count from Medium (M) to Large (L) yields severe attenuation. For Diffusion Policy, scaling model capacity with a SigLIP encoder yields a \+12.4% gain (57.6% $\\to$ 70.0%). For OAT, scaling $M \\to L$ via the identical encoder configuration actually degrades performance (53.8% $\\to$ 48.0%).15 Adding autoregressive modeling capacity fundamentally cannot bypass the truncation of the 80-bit visual-spatial bottleneck.15

### **Codebook Entropy and Dimensional Saturation Limits**

A naive algorithmic hypothesis assumes that drastically expanding the codebook capacity (e.g., modifying the FSQ layers to $L \= $, $|\\mathcal{V}| \> 65000$) will eliminate the Compression Gap by shifting $I(Z\_v; T) \> I(O; Z\_v)$. However, empirical codebook size analyses prove this intervention causes immediate downstream failure.14  
As the scalar quantization grid capacity expands, the vocabulary space required for the cross-entropy projection layer in the transformer inflates exponentially. This induces massive token entropy and latent vector sparsity.14 The discrete target distributions become overly diffuse, severely degrading the policy's ability to minimize next-token prediction loss during gradient descent.14 The model collapses as it struggles to associate continuous spatial features with sparse categorical distributions, indicating that OAT is permanently trapped between the 80-bit Compression Gap and the instability of high-entropy codebooks.14

### **Kinematic Degradation at Low Prefix Lengths**

The physical execution mechanism exhibits specific edge-condition vulnerabilities when intentionally operating under aggressive latency constraints. A primary documented failure mode for $\\text{OAT}\_{\<4}$ (detokenization utilizing $K \\le 3$) is the "proximity failure".11  
Because the nested dropout algorithm strictly mandates that $T\_1$ and $T\_2$ encode generalized, low-frequency coordinate structures, decoding sequences truncated at $K=2$ produces a mathematically accurate but incomplete representation of the spatial objective.11 During physical rollout on tasks demanding precise millimeter-scale alignment (e.g., Stack Cups), the end-effector seamlessly traverses free space and enters the immediate vicinity of the target.11 However, absent the high-frequency residual parameters normally embedded in $T\_5 \\dots T\_8$, the decoded continuous action array contains an artificially smoothed trajectory curve. As a result, the robotic manipulator halts prematurely just prior to grasping, or outputs singular contact matrices, triggering intrinsic torque-safety limits and rendering the system entirely stationary.11 Therefore, sub-$15$ms inference targets result in high failure distributions on insertion mechanics.19

## **Open Problems**

The strict trade-off architectures necessitated by discrete token mapping have isolated several specific, mathematically unresolved technical trajectories currently active in the literature.

### **1\. Dynamic Formulation of Adaptive Autoregressive Depth**

Current implementations of OAT execute under static prefix definitions; the parameter $K$ must be defined as a constant external to the network topology prior to the generation step.1 However, kinetic complexity varies radically over the temporal span of a single manipulation task.3 Free-space transit exhibits near-zero spatial variance, requiring only $K=1$ resolution, whereas surface contact and object interaction necessitate full $K=8$ detail. Developing an analytical framework for "adaptive autoregressive depth" remains an open problem. The literature lacks a mechanism that natively formulates an early-exit criteria function—where the network intrinsically models a terminal state probability $P(\\texttt{STOP} | T\_{\<K}, o\_{1:H\_o})$ capable of halting generation dynamically when the sequence information exceeds the task variance bounds without disrupting the left-to-right causal masking.3

### **2\. Hybrid Encoding to Bypass the 80-bit Ceiling**

The 80-bit structural ceiling demonstrated by the Compression Gap fundamentally prevents OAT from scaling linearly with parameter counts and dataset expansion.15 Research is heavily shifting toward defining hybrid tokenization spaces that circumvent purely discrete capacity bounds. Current theoretical propositions involve appending unquantized continuous residual vectors conditionally dependent on the discrete categorical prefix. Determining how to propagate continuous gradients through the categorical generation steps without suffering from the temporal instability of diffusion trajectories represents an active, unresolved architectural paradox within transformer policy physics.15

### **3\. Variance-Aware Temporal Drift Evaluation**

The current benchmark methodology evaluates policy success utilizing localized peak probability success rates derived from standard independent evaluation rollouts.15 However, the literature has not yet formalized a robust statistical framework for tracking "temporal drift variance" over infinite or extended rollout horizons $H\_a \\to \\infty$.15 Because autoregressive sequences accumulate compounding categorical errors, an unresolved question exists regarding how $T\_{\<K}$ prediction decay at step $t$ specifically amplifies the kinematic boundary variance at step $t \+ 100$. Mapping the deterministic error bounds of FSQ quantization against the chaotic temporal expansion of closed-loop physics simulators is essential for calculating exact mean-time-to-failure in OAT robotic deployments.15

## **Critical References**

1. Liu, C. et al. (2026). *OAT: Ordered Action Tokenization.* Formulates the OAT architecture by integrating FSQ with nested dropout and causal register attention, establishing prefix-based anytime decoding paradigms and setting the state-of-the-art tokenized baseline success rate.1  
2. Shiba, T. (2026). *The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling.* Explicitly defines the 80-bit information bottleneck inherent in FSQ tokenizers, proving mathematically and empirically that OAT architecture strictly blocks feature gradient propagation from advanced vision encoders.15  
3. Mete, A. et al. (2024). *Quest: Self-supervised skill abstractions for learning continuous control.* Introduces the baseline methodology for unstructured learned latent action tokenization, inadvertently validating the requirement for explicit causal sequence hierarchy in transformers.8  
4. Pertsch, K. et al. (2025). *FAST: Efficient action tokenization for vision-language-action models.* Develops frequency-domain compression mechanisms using Discrete Cosine Transforms (DCT), noting the specific degradation of total decodability parameters upon frequency truncation.8  
5. Lee, S. et al. (2024). *Vector-Quantized Behavior Transformer (VQ-BeT).* Addresses multi-modal action prediction parameters by tokenizing complex physical behaviors via hierarchical Residual Vector Quantization (RVQ), serving as the principal precursor methodology to FSQ grid scaling.9  
6. Mentzer, F. et al. (2023). *Finite scalar quantization: VQ-VAE made simple.* Establishes the mathematical foundation and gradient mapping functions of the fixed integer-grid quantization matrix utilized within OAT’s discrete bottleneck sequence.15  
7. Zhao, T. et al. (2023). *Action Chunking with Transformers (ACT).* Analyzes the initial network baselines and continuous policy logic optimized against discrete formulation limits in generalist VLA manipulation paradigms.15  
8. Kim, M. et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* Demonstrates the extensive, large-scale sequence scaling limits for robotic primitives utilizing standard 256-bin analytical token generation distributions.6  
9. Levine, S. et al. (2024). *Octo: An open-source generalist robot policy.* Explores transformer-based physical diffusion parameters and standard task/observation tokenizers mapped over extremely large Open X-Embodiment datasets.1  
10. Reed, S. et al. (2022). *A Generalist Agent (Gato).* Outlines the foundational deep-learning $\\mu$-law mapping and binning continuous-to-discrete action tokenization strategies that initially yielded massive physical context-length inflation parameters.4

#### **Works cited**

1. OAT: Ordered Action Tokenization \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2602.04215v2](https://arxiv.org/html/2602.04215v2)  
2. (PDF) OAT: Ordered Action Tokenization \- ResearchGate, accessed April 10, 2026, [https://www.researchgate.net/publication/400459544\_OAT\_Ordered\_Action\_Tokenization](https://www.researchgate.net/publication/400459544_OAT_Ordered_Action_Tokenization)  
3. \[Literature Review\] OAT: Ordered Action Tokenization \- Moonlight, accessed April 10, 2026, [https://www.themoonlight.io/en/review/oat-ordered-action-tokenization](https://www.themoonlight.io/en/review/oat-ordered-action-tokenization)  
4. Deepmind Gato \- Silverton Consulting, accessed April 10, 2026, [https://silvertonconsulting.com/tag/deepmind-gato/](https://silvertonconsulting.com/tag/deepmind-gato/)  
5. arXiv:2205.06175v3 \[cs.AI\] 11 Nov 2022, accessed April 10, 2026, [https://arxiv.org/pdf/2205.06175](https://arxiv.org/pdf/2205.06175)  
6. OpenVLA: An Open-Source Vision-Language-Action Model, accessed April 10, 2026, [https://openvla.github.io/](https://openvla.github.io/)  
7. OpenVLA: An open-source vision-language-action model for robotic manipulation. \- GitHub, accessed April 10, 2026, [https://github.com/openvla/openvla](https://github.com/openvla/openvla)  
8. OAT: Ordered Action Tokenization \- alphaXiv, accessed April 10, 2026, [https://www.alphaxiv.org/overview/2602.04215v2](https://www.alphaxiv.org/overview/2602.04215v2)  
9. VQ-BeT: Behavior Generation with Latent Actions \- Seungjae Lee, accessed April 10, 2026, [https://sjlee.cc/vq-bet/](https://sjlee.cc/vq-bet/)  
10. Behavior Generation with Latent Actions \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2403.03181v1](https://arxiv.org/html/2403.03181v1)  
11. OAT: Ordered Action Tokenization \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2602.04215v1](https://arxiv.org/html/2602.04215v1)  
12. Meet OAT: The New Action Tokenizer Bringing LLM-Style Scaling and Flexible, Anytime Inference to the Robotics World \- MarkTechPost, accessed April 10, 2026, [https://www.marktechpost.com/2026/02/08/meet-oat-the-new-action-tokenizer-bringing-llm-style-scaling-and-flexible-anytime-inference-to-the-robotics-world/](https://www.marktechpost.com/2026/02/08/meet-oat-the-new-action-tokenizer-bringing-llm-style-scaling-and-flexible-anytime-inference-to-the-robotics-world/)  
13. \[2602.04215\] OAT: Ordered Action Tokenization \- arXiv, accessed April 10, 2026, [https://arxiv.org/abs/2602.04215](https://arxiv.org/abs/2602.04215)  
14. \[論文評述\] OAT: Ordered Action Tokenization \- Moonlight, accessed April 10, 2026, [https://www.themoonlight.io/tw/review/oat-ordered-action-tokenization](https://www.themoonlight.io/tw/review/oat-ordered-action-tokenization)  
15. The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2604.03191v1](https://arxiv.org/html/2604.03191v1)  
16. ICML Poster STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization, accessed April 10, 2026, [https://icml.cc/virtual/2025/poster/44123](https://icml.cc/virtual/2025/poster/44123)  
17. Unlocking robotics with action tokenization | by Dave Davies | Online Inference \- Medium, accessed April 10, 2026, [https://medium.com/online-inference/unlocking-robotics-with-action-tokenization-15e4160c1aee](https://medium.com/online-inference/unlocking-robotics-with-action-tokenization-15e4160c1aee)  
18. FAST: Efficient Action Tokenization for Vision-Language-Action Models \- arXiv, accessed April 10, 2026, [https://arxiv.org/html/2501.09747v1](https://arxiv.org/html/2501.09747v1)  
19. OAT, accessed April 10, 2026, [https://ordered-action-tokenization.github.io/](https://ordered-action-tokenization.github.io/)  
20. 机器学习2026\_4\_6 \- arXiv每日学术速递, accessed April 10, 2026, [http://www.arxivdaily.com/thread/78470](http://www.arxivdaily.com/thread/78470)  
21. awesome-daily-AI-arxiv/hot\_topic/Embodied\_AI.md at main \- GitHub, accessed April 10, 2026, [https://github.com/Tavish9/awesome-daily-AI-arxiv/blob/main/hot\_topic/Embodied\_AI.md](https://github.com/Tavish9/awesome-daily-AI-arxiv/blob/main/hot_topic/Embodied_AI.md)  
22. Trajectory Flow Matching with Applications to Clinical Time Series Modelling | Request PDF \- ResearchGate, accessed April 10, 2026, [https://www.researchgate.net/publication/397200124\_Trajectory\_Flow\_Matching\_with\_Applications\_to\_Clinical\_Time\_Series\_Modelling](https://www.researchgate.net/publication/397200124_Trajectory_Flow_Matching_with_Applications_to_Clinical_Time_Series_Modelling)  
23. \[Literature Review\] Behavior Generation with Latent Actions, accessed April 10, 2026, [https://www.themoonlight.io/en/review/behavior-generation-with-latent-actions](https://www.themoonlight.io/en/review/behavior-generation-with-latent-actions)  
24. DIFFUSION TRANSFORMER POLICY \- OpenReview, accessed April 10, 2026, [https://openreview.net/pdf?id=PvvXDazPMs](https://openreview.net/pdf?id=PvvXDazPMs)  
25. An Open-Source Generalist Robot Policy \- Octo, accessed April 10, 2026, [https://octo-models.github.io/paper.pdf](https://octo-models.github.io/paper.pdf)  
26. Deepmind's new model Gato is amazing\! \- Louis-François Bouchard, aka What's AI, accessed April 10, 2026, [https://www.louisbouchard.ai/deepmind-gato/](https://www.louisbouchard.ai/deepmind-gato/)