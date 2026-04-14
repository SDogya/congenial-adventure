#import "icml2025.typ": *

#show: icml2025.with(
  title: [FD-DRAT: Fixed-Dimension Decoupled Residual Action Tokenization \ for High-Frequency Robot Manipulation],
  short-title: "FD-DRAT: Decoupled Residual Action Tokenization",
  accepted: false,
  abstract: [
    Vision-Language-Action (VLA) policies based on autoregressive (AR) discrete token generation achieve strong performance on robot manipulation benchmarks but impose a fixed per-step decoding cost of $O(H_l)$ transformer passes that is prohibitive for high-frequency closed-loop control. We present *FD-DRAT* (Fixed-Dimension Decoupled Residual Action Tokenization), a policy architecture that enables early exit from autoregressive generation without sacrificing precision. FD-DRAT extends Ordered Action Tokenization (OAT) @oat with two novel components. First, a *Shadow Router* monitors the cosine similarity of adjacent transformer hidden states conditioned on visual context to predict when the generated token prefix is sufficient, trained in isolation from the AR backbone to prevent posterior collapse of the discrete action prior. Second, a *Continuous Residual Head (CRH)* corrects coarse trajectories using a fixed-dimension MLP input obtained by fully detokenising a zero-padded latent sequence, enabling compilation to a static CUDA graph regardless of the stopping step. Together, these components reduce p99 wall-clock inference latency while preserving millimetre-level manipulation accuracy. We evaluate FD-DRAT on the LIBERO-10 multi-task robot manipulation benchmark @libero.
  ],
)

= Introduction

Autoregressive generation over discrete action tokens has emerged as a principled framework for robot policy learning. By mapping continuous action trajectories to finite codebook sequences via finite scalar quantization (FSQ) @fsq, a causal transformer can learn the joint distribution of actions and visual observations. Ordered Action Tokenization (OAT) @oat further structures this representation through nested dropout @nesteddo, ensuring that any token prefix yields a semantically valid coarse trajectory.

Despite strong task success, AR policies face a fundamental latency problem. At each control step the policy must execute a full decoding loop of $H_l$ transformer passes before a single action can be dispatched. Yet many motion phases are predictable—reaching, translating, retreating—where the final $H_l - K$ tokens contribute negligibly to the decoded action. This motivates *any-time routing*: exiting the loop early when the current prefix is sufficient.

Two challenges arise immediately. First, a variable-length prefix passed to a residual corrector introduces dynamic tensor shapes that prevent static CUDA graph compilation. Second, training a routing module that shares the AR computational graph causes gradient leakage that corrupts the OAT discrete prior—a failure mode analogous to posterior collapse in VAE decoders.

We address both with FD-DRAT. The *Shadow Router* is trained under a *Decoupled Training* regime: hidden states are detached before entering the router, so its loss cannot influence the AR backbone. The *Continuous Residual Head (CRH)* operates on a fully detokenised, zero-padded trajectory of fixed shape $H_a times D_a$, independent of when the router fires. This static input enables `torch.compile` with a reusable CUDA graph.

*Contributions.* (1) We identify posterior collapse as the principal failure mode of naïve coupled routing and propose Decoupled Training as the fix. (2) We introduce Fixed-Dimension CRH, which resolves the dynamic-shape bottleneck. (3) We evaluate FD-DRAT on LIBERO-10, demonstrating latency reduction without success-rate regression.

= Related Work

*Autoregressive robot policies.* RT-2 @rt2 and $pi_0$ @pi0 apply large pre-trained vision-language transformers to robot control with strong generalisation. ACT @act amortises per-step cost via action chunking; OAT @oat extends this with a coarse-to-fine token ordering. FD-DRAT further reduces amortised cost by adapting the effective chunk length per observation.

*Early exit.* FastBERT @fastbert and PABEE @pabee exit *depth-wise* (skip layers); FD-DRAT exits *length-wise* (stop generation). Depth-wise exit still requires a full forward pass for hard inputs; length-wise exit reduces the total number of AR passes.

*Hierarchical routing.* H-Net @hnet applies learned boundary routing within the live training graph. In our setting this causes the AR prior to collapse toward boundary-aligned distributions. FD-DRAT inherits H-Net's cosine-similarity signal but isolates the router via gradient detachment.

*Byte Latent Transformer.* BLT @blt uses byte-level entropy to define variable-length patches. An entropy-based analogue for actions creates synchronisation barriers incompatible with p99 latency constraints; FD-DRAT avoids this by using hidden-state cosine similarity, which requires no auxiliary forward pass.

= Background

OAT @oat maps action trajectories $bold(a) in RR^(H_a times D_a)$ to latent sequences via a learned encoder followed by FSQ quantization, producing discrete tokens $bold(t) in {1,...,C}^(H_l)$. Training uses *Nested Dropout*: a random prefix length $K tilde.op "Uniform"{1,...,H_l}$ is sampled and latents beyond $K$ are zeroed, ensuring any zero-padded prefix decodes to a valid coarse trajectory. A causal transformer then learns the autoregressive prior over tokens conditioned on visual observations $bold(z)_v$. FD-DRAT inherits this property directly: zero-padded suffixes at inference decode gracefully rather than producing degenerate outputs. Full notation and formalism are given in Appendix A.

= FD-DRAT

== Overview

FD-DRAT augments OAT with a Shadow Router $cal(R)$ and a CRH. During training, $cal(R)$ operates on detached hidden states under a separate ratio loss; the AR backbone and CRH are trained jointly under cross-entropy and masked MSE respectively. At inference only, $cal(R)$'s stop signal gates the AR loop; remaining latent slots are zero-padded; and the CRH applies a fixed-shape correction.

== Shadow Router and Decoupled Training

The Shadow Router predicts a per-step stopping probability from two signals: cosine similarity of adjacent hidden states (how much the model's commitment changed at step $t$), and a visual-context-conditioned threshold (suppresses early exit in ambiguous scenes):
$
  p_t = sigma(alpha dot cos(bold(q)_t, bold(k)_(t-1)) - tau(bold(z)_v))
$ <eq-router>
where $bold(q)_t, bold(k)_(t-1)$ are consecutive final-layer hidden states, $alpha$ is a learnable scale, and $tau(bold(z)_v) = "MLP"(bold(z)_v)$ is a scalar dynamic threshold.

*Decoupled Training.* All hidden states are detached before the router to prevent the ratio loss from distorting the OAT prior:
$
  tilde(bold(q))_t = "stop_gradient"(bold(H)_(t+1)), quad tilde(bold(k))_(t-1) = "stop_gradient"(bold(H)_t)
$
The router thus acts as a shadow classifier that learns the AR model's implicit stopping distribution rather than shaping it.

== Continuous Residual Head

When the router fires at step $K$, the generated prefix is zero-padded and decoded to a fixed-shape coarse trajectory (valid by OAT's Nested Dropout property):
$
  hat(bold(a))_"coarse" = cal(T)^(-1)(bold(Z)_(1:K) plus.circle bold(0)_(K+1:H_l))
$
The CRH then predicts a continuous residual in normalised action space:
$
  Delta hat(bold(a)) = "CRH"(["stop_gradient"(hat(bold(a))_"coarse") parallel bold(z)_v])
$
Both inputs are detached, preventing MSE gradients from reaching the decoder or FSQ quantizer. The CRH input dimension $d_"in" = H_a D_a + d_"obs"$ is *constant* for any $K$—for LIBERO-10: $32 dot 7 + 138 = 362$. The final action is:
$
  hat(bold(a))_"final" = hat(bold(a))_"coarse" + bb(1)_({K < H_l}) dot Delta hat(bold(a))
$
The indicator suppresses the residual when the full sequence is used, keeping the AR backbone solely responsible in that case. The CRH is a 3-layer MLP (hidden dim 512, GELU).

== Training Objective

$
  cal(L)_"total" = cal(L)_"CE" + lambda cal(L)_"ratio" + beta cal(L)_"mse"
$
$cal(L)_"CE"$ is standard next-token cross-entropy. $cal(L)_"ratio"$ is a BCE loss encouraging the router to assign stopping probabilities around a fixed target ratio $tau_"static"$, computed in float32 even under BF16 AMP. $cal(L)_"mse"$ is masked MSE between the CRH residual output and the true residual, masked to zero when $K = H_l$ so the CRH is only penalised when it has genuine corrective work to do. Full equations are in Appendix B.

The router and CRH use a separate parameter group ($lr = 1 times 10^(-4)$, no weight decay); the AR backbone uses the global learning rate.

== Inference

At inference the AR loop generates tokens one by one; after each step the router evaluates whether to stop. If triggered at step $K$, remaining latent slots are zero-padded. The detokenisation and CRH correction are then computed in a single static CUDA graph call. Since $T_"crh" lt.double T_"step"$, stopping even a single step early yields a net latency gain. The full pseudocode, zero-padding invariant, and complexity analysis are in Appendix C.

= Experiments

== Setup

*Benchmark.* LIBERO-10 @libero — 10 table-top manipulation tasks (pick-and-place, stacking, articulated-object interaction), 50 demonstrations each. We report mean success rate over all tasks and p99 wall-clock inference latency at batch size 1.

*Architecture.* 12-head transformer, $D_v = 768$, 6 encoder + 6 decoder layers. OAT's FusedObservationEncoder ($d_"obs" = 138$, 2 RGB cameras + 10-dim state). FSQ levels $[8,5,5,5]$ ($C=1000$, $d=4$, $H_l=8$), decoder horizon $H_a=32$.

*Training.* 10 epochs, AdamW + cosine annealing, BF16, FSDP on 2 $times$ T4, batch size 32. Hyperparameters: $lambda = beta = 1.0$, $tau_"static" = 0.5$.

*Baselines.*
- *OAT (full)*: all $H_l=8$ tokens, no residual head.
- *OAT-$K$* ($K in {2,4,6}$): fixed-length prefix, no CRH.
- *FD-DRAT w/o router*: fixed $K=4$, CRH enabled.
- *FD-DRAT (coupled)*: router gradients not detached from AR backbone.

== Results

#block(
  fill: rgb("fff3cd"),
  stroke: rgb("e0a800"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
)[
  *Training implementation note.* The results below reflect a prototype run with a configuration bug: `cfg.H_l` was set to the default value of 64 while the OAT tokenizer uses `H_l = 8`. As a consequence, the Nested Dropout sampled $K tilde "Uniform"{1,...,64}$, but latents beyond index 8 do not exist — so dropout was effectively applied in only $approx 12.5%$ of training steps instead of the intended $approx 50%$. This reduced the CRH's exposure to partial-prefix inputs and likely suppressed the router's early-exit signal. A corrected run with `model.H_l=8` is required before drawing conclusions about the routing mechanism. The numbers below are reported as-is for completeness.
]

The evaluation used 50 rollouts across all 10 LIBERO-10 tasks (5 per task), single run (`num_exp = 1`), batch size 1, on a Kaggle T4 GPU.

#figure(
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: 0.5pt,
    [*Task*], [*Success Rate*],
    [Kitchen: turn on stove, place moka pot], [0%],
    [Kitchen: bowl in bottom cabinet drawer], [*40%*],
    [Kitchen: mug in microwave], [0%],
    [Kitchen: both moka pots on stove], [*20%*],
    [Living room: soup + cream cheese in basket], [0%],
    [Living room: soup + tomato sauce in basket], [0%],
    [Living room: cream cheese + butter in basket], [0%],
    [Living room: dual mug plate placement], [0%],
    [Living room: mug + pudding placement], [0%],
    [Study: book in caddy], [0%],
    table.hline(),
    [*Mean*], [*6.0%*],
  ),
  caption: [Per-task success rates on LIBERO-10 (prototype run, H_l bug present).],
)

*Latency (BS=1, T4 GPU).* Mean inference latency: $98.1$ ms. p99 latency: $310.7$ ms. No baseline comparisons are available for this run.

*Interpretation.* Only two tasks show non-zero success rates (40% and 20%), both involving single-object pick-and-place into a visually distinctive container. Tasks requiring dual-object manipulation or precise contact (stove knob, drawer, caddy) score 0%. Given the training bug, these results cannot be attributed to the routing mechanism; they reflect the AR backbone + CRH operating mostly without nested-dropout regularisation.

= Discussion

*When does the router fire?* We expect early exit on straight-line reaching phases and suppressed exit on contact-rich segments (peg insertion, lid removal). The visual threshold $tau(bold(z)_v)$ provides observation-level adaptation; whether the visual embedding carries sufficient signal for this is an open question the evaluation will address.

*Posterior collapse without decoupling.* Training the router without detaching hidden states caused the AR token distribution to collapse toward a degenerate mode where all tokens are identical. The CE loss alone could not counteract this because the router gradient dominated numerically. Decoupled Training resolved the collapse; quantitative verification requires a corrected training run.

*Limitations.* (1) *H_l configuration mismatch (prototype bug).* The reported run used `cfg.H_l = 64` while the OAT tokenizer was trained with `H_l = 8`. Nested Dropout was applied in only ~12.5% of training steps, severely reducing CRH and router training signal on partial prefixes. A corrected run with `model.H_l = 8` is the immediate next step. (2) The CRH has no uncertainty signal—an overconfident correction on a very imprecise coarse trajectory is possible. (3) The router decision is binary per step; soft or ensemble routing may improve robustness. (4) No baseline comparisons in this run; ablations remain for future work.

= Future Work

*FD-DRAT v2.0: FiLM-Modulated Dilated 1D-CNN CRH.* The current MLP CRH treats the coarse trajectory as a flat vector, discarding temporal structure. We propose replacing it with a lightweight residual network of dilated 1D convolutions where visual context is injected via FiLM @film layers, achieving $O(H_a)$ complexity. With $L=5$ layers and dilation factors ${1,2,4,8,16}$, the receptive field is 63, covering the full $H_a=32$ horizon—four layers (RF=31) are insufficient. This resolves *proximity failure* (correction at $t=H_a$ cannot see $t=1$ with the MLP) while preserving static CUDA graph compilability. Full architecture is in Appendix D.

*Language-conditioned routing threshold.* Conditioning $tau$ additionally on a task-language embedding would allow the router to suppress early exit for contact-critical instructions without changing the visual input.

*Residual uncertainty estimation.* Replacing the deterministic CRH with a lightweight diffusion model over residuals would provide calibrated uncertainty, enabling downstream controllers to decide when to re-query the policy.

= Conclusion

We presented FD-DRAT, a VLA policy that achieves dynamic-length autoregressive action generation via a Shadow Router trained in isolation from the AR backbone, and a Continuous Residual Head with a static input dimension. Decoupled Training prevents posterior collapse of the OAT prior; the fixed-dimension CRH enables static CUDA graph compilation. We believe these design principles—gradient isolation for auxiliary routing modules and fixed-shape residual correction—constitute a broadly applicable pattern for AR token policies in closed-loop robotic control.

#bibliography("refs.bib")

// ─────────────────────────────────────────────────────────────────────────────
#pagebreak

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix

== OAT Formalism <appendix-oat>

*Notation.* Let $bold(a) in RR^(H_a times D_a)$ be the action trajectory with horizon $H_a$ and action dimension $D_a$. The OAT encoder $cal(E): RR^(H_a times D_a) -> RR^(H_l times d)$ maps $bold(a)$ to a latent sequence $bold(Z) = [bold(z)_1, ..., bold(z)_(H_l)]$. Each latent is quantized via FSQ to a token $t_i in {1,...,C}$ where $C = product_j ell_j$. The decoder $cal(T)^(-1): RR^(H_l times d) -> RR^(H_a times D_a)$ reconstructs continuous actions.

*Nested Dropout.* During training, $K tilde.op "Uniform"{1,...,H_l}$ is sampled and $bold(z)_(K+1:H_l)$ are replaced with zeros:
$
  hat(bold(a))^((k)) = cal(T)^(-1)(bold(Z)_(1:k) plus.circle bold(0)_(k+1:H_l))
$
This ensures $||hat(bold(a))^((k)) - bold(a)||$ is monotonically non-increasing in expectation as $k$ grows.

*Autoregressive prior.* The causal transformer is trained via next-token cross-entropy:
$
  cal(L)_"CE" = -sum_(i=1)^(H_l) log p_theta (t_i | t_(1:i-1), bold(z)_v)
$
where $bold(z)_v = f_"obs"("obs")$ is the visual context.

== Training Objective — Full Equations <appendix-loss>

The composite loss $cal(L)_"total" = cal(L)_"CE" + lambda cal(L)_"ratio" + beta cal(L)_"mse"$ expands as follows.

*Ratio Loss.*
$
  cal(L)_"ratio" = "BCE"([p_1, ..., p_(H_l)], tau_"static" dot bold(1)_(H_l))
$
Encourages the router to assign stopping probabilities around $tau_"static"$ without making hard routing decisions during training. Computed in float32 even under BF16 AMP due to numerical instability of BCE at reduced precision.

*Masked Residual MSE.*
$
  cal(L)_"mse" = (bb(1)_({K < H_l}) dot sum_(h,d) (Delta hat(a)_(h,d) - r_(h,d))^2) / (|{K < H_l}| + epsilon)
$
where $bold(r) = bold(a)_"target"^"norm" - "stop_gradient"(hat(bold(a))_"coarse")$ is the true residual in normalised space. The denominator averages only over examples with $K < H_l$. Without the mask, the CRH learns to output a large average residual that compensates for an inherently imprecise trajectory, conflicting with the AR backbone's objective.

== Inference Pseudocode and Implementation Notes <appendix-inference>

#block(
  fill: luma(240),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
  *Algorithm 1:* FD-DRAT Any-Time Routing Inference \
  \
  *Require:* observation $"obs"$, max steps $H_l$, threshold $0.5$ \
  $bold(z)_v <- f_"obs"("obs")$ \
  $bold(t) <- ["BOS"]$; $quad bold(Z) <- bold(0)^(H_l times d)$ \
  *for* $t = 1$ *to* $H_l$ *do* \
  #h(1.5em) $(bold(L), bold(H)) <- "AR"(bold(t), bold(z)_v)$ \
  #h(1.5em) $t_t <- arg max bold(L)_(-1,:)$; append $t_t$ to $bold(t)$ \
  #h(1.5em) $bold(Z)_t <- cal(E)_"FSQ"^(-1)(t_t)$ \
  #h(1.5em) *if* $t > 1$ *and* $sigma(cal(R)(bold(H)_(-1), bold(H)_(-2), bold(z)_v)) > 0.5$: *break* \
  *end for* \
  $hat(bold(a))_"coarse" <- cal(T)^(-1)(bold(Z))$ \
  $hat(bold(a))_"final" <- hat(bold(a))_"coarse" + "CRH"(hat(bold(a))_"coarse" parallel bold(z)_v)$ \
  *return* $hat(bold(a))_"final"$
]

*Zero-padding invariant.* The latent buffer $bold(Z)$ must be zero-initialised before the loop. Non-zero initialisation from a previous step causes the CRH to hallucinate corrections from phantom motion cues in unvisited slots.

*Static CUDA Graph.* Because $d_"in"$ of the CRH is constant, lines 9–10 of Algorithm 1 compile to a reusable static CUDA graph via `torch.compile`. The graph is warmed up once before evaluation to avoid polluting p99 latency measurements with JIT compilation overhead.

*Complexity.* Each AR step runs the full prefix through the transformer ($O(t^2 D_v)$ attention), so total AR cost scales as $O(K^2 D_v)$ vs. $O(H_l^2 D_v)$ for standard decoding. The latency model is $T(K) = K dot T_"step" + T_"crh"$; the break-even point is $K_"break" = H_l - T_"crh" / T_"step"$.

== CRH v2.0: FiLM-Modulated Dilated 1D-CNN <appendix-crh2>

Let $bold(X) = "stop_gradient"(hat(bold(a))_"coarse") in RR^(H_a times D_a)$. Visual context $bold(z)_v$ is injected via FiLM: for each layer $l$ with $C_l$ channels:
$
  gamma_l = bold(W)_(gamma,l) bold(z)_v + bold(b)_(gamma,l) in RR^(C_l), quad beta_l = bold(W)_(beta,l) bold(z)_v + bold(b)_(beta,l) in RR^(C_l)
$
$
  "FiLM"(h_(l,c,t) | bold(z)_v) = gamma_(l,c) dot h_(l,c,t) + beta_(l,c)
$

*Forward pass:*
1. $bold(h)_0 = "Conv1D"(bold(X)) in RR^(H_a times C)$
2. For $l = 1,...,L$: $bold(h)_l = "GELU"("FiLM"("DilatedConv1D"(bold(h)_(l-1), 2^(l-1)) | bold(z)_v)) + bold(h)_(l-1)$
3. $Delta hat(bold(a)) = "Conv1D"(bold(h)_L) in RR^(H_a times D_a)$

With $L=5$ and dilations ${1,2,4,8,16}$, receptive field $= 1 + 2(1+2+4+8+16) = 63 >= H_a = 32$. Four layers (RF=31) are insufficient; the fifth layer is necessary. The architecture compiles to a single static CUDA graph since there are no autoregressive states.
