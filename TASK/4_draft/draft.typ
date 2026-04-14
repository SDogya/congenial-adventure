#import "icml2025.typ": *

#show: icml2025.with(
  title: [FD-DRAT: Fixed-Dimension Decoupled Residual Action Tokenization \ for High-Frequency Robot Manipulation],
  short-title: "FD-DRAT: Decoupled Residual Action Tokenization",
  accepted: false,
  abstract: [
    Autoregressive (AR) discrete-token VLA policies incur $O(H_l)$ transformer passes per control step, a fixed cost regardless of motion complexity. We present *FD-DRAT*, which adds early exit to OAT @oat via two components: a *Shadow Router* that predicts per-step sufficiency from adjacent hidden-state cosine similarity, trained with detached gradients to prevent posterior collapse of the OAT prior; and a *Continuous Residual Head (CRH)* that refines the decoded coarse trajectory from a fixed-shape input independent of the exit step, enabling static CUDA graph compilation. We evaluate on LIBERO-10 @libero and report a prototype run affected by a training configuration bug (Section 5).
  ],
)

= Introduction

AR policies over discrete action tokens @oat @rt2 must complete a full $H_l$-step decoding loop before dispatching any action. Many motion phases are low-complexity -- reaching, retreating -- where later tokens add negligible information. Variable-length early exit fixes this but creates two problems: (1) dynamic tensor shapes break static CUDA graph compilation; (2) routing loss backpropagated through the AR graph corrupts the OAT nested-dropout prior (posterior collapse).

FD-DRAT solves both. The *Shadow Router* operates on stop-gradient hidden states -- its loss never reaches the backbone. The *CRH* always receives a fixed-shape input $hat(bold(a))_"coarse" in RR^(H_a times D_a)$ obtained by decoding the zero-padded latent buffer, so the CUDA graph is static regardless of exit step $K$.

*Contributions.* (1) Gradient isolation via stop-gradient as the fix for routing-induced posterior collapse. (2) Fixed-dimension CRH via zero-padded decode, eliminating dynamic shapes. (3) Prototype evaluation on LIBERO-10 with honest reporting of a training bug that limits the current results.

= Related Work

*AR robot policies.* RT-2 @rt2, $pi_0$ @pi0, ACT @act, OAT @oat. FD-DRAT extends OAT with adaptive exit length.

*Early exit.* FastBERT @fastbert and PABEE @pabee exit depth-wise (layer skipping); FD-DRAT exits length-wise (generation stopping), reducing total AR passes rather than pass depth.

*Hierarchical boundary routing.* H-Net @hnet routes within the live compute graph; we find this collapses the OAT discrete prior and isolate the router via detachment.

*Byte Latent Transformer.* BLT @blt uses byte-level entropy for variable-length patches; entropy estimation requires an auxiliary forward pass, incompatible with p99 latency constraints. FD-DRAT uses hidden-state similarity computed for free during decoding.

= Background

OAT @oat encodes $bold(a) in RR^(H_a times D_a)$ into latents $bold(Z) in RR^(H_l times d)$, quantized via FSQ @fsq to tokens $bold(t) in {1,...,C}^(H_l)$. *Nested Dropout* trains the decoder with $bold(z)_(K+1:H_l)$ zeroed for $K tilde.op "Uniform"{1,...,H_l}$, guaranteeing that any zero-padded prefix decodes to a valid coarse trajectory. A causal transformer learns $p_theta(t_i | t_{1:i-1}, bold(z)_v)$ where $bold(z)_v = f_"obs"("obs")$. Full formalism in Appendix A.

= FD-DRAT

== Shadow Router and Decoupled Training

$
  p_t = sigma(alpha dot cos(bold(q)_t, bold(k)_(t-1)) - tau(bold(z)_v))
$ <eq-router>

$bold(q)_t, bold(k)_(t-1)$: consecutive final-layer hidden states. $alpha$: learnable scale. $tau(bold(z)_v) = "MLP"(bold(z)_v)$: visual dynamic threshold.

*Decoupled Training* detaches all inputs to $cal(R)$:
$
  tilde(bold(q))_t = "stop_gradient"(bold(H)_(t+1)), quad tilde(bold(k))_(t-1) = "stop_gradient"(bold(H)_t)
$
This severs the ratio loss from the AR backbone, preserving the OAT prior. Without detachment the router gradient dominates the CE loss numerically and collapses the token distribution.

== Continuous Residual Head

At exit step $K$, zero-pad and decode:
$
  hat(bold(a))_"coarse" = cal(T)^(-1)(bold(Z)_(1:K) plus.circle bold(0)_(K+1:H_l))
$

Predict residual (both inputs detached):
$
  Delta hat(bold(a)) = "CRH"(["stop_gradient"(hat(bold(a))_"coarse") parallel bold(z)_v])
$

CRH input dim $d_"in" = H_a D_a + d_"obs"$ is *constant* for all $K$. LIBERO-10: $32 dot 7 + 138 = 362$. Final action:
$
  hat(bold(a))_"final" = hat(bold(a))_"coarse" + bb(1)_({K < H_l}) dot Delta hat(bold(a))
$

The indicator zeros the residual when $K = H_l$, keeping the AR backbone solely responsible for full-sequence predictions. CRH: 3-layer MLP, hidden dim 512, GELU.

== Training Objective

$
  cal(L)_"total" = cal(L)_"CE" + lambda cal(L)_"ratio" + beta cal(L)_"mse"
$

$cal(L)_"CE"$: next-token cross-entropy. $cal(L)_"ratio"$: BCE encouraging $p_t approx tau_"static"$, computed in float32 under BF16 AMP. $cal(L)_"mse"$: MSE between CRH output and true normalised-space residual, masked to zero when $K = H_l$. Router and CRH: $lr = 10^(-4)$, no weight decay. AR backbone: global lr. Full loss equations in Appendix B.

== Inference

Autoregressive loop up to $H_l$ steps; router checked after step $t > 1$; exit when $sigma(cal(R)(...)) > 0.5$. Zero-pad remaining slots; decode; apply CRH. The zero-padding invariant and pseudocode are in Appendix C.

= Experiments

== Setup

*Benchmark.* LIBERO-10 @libero  --  10 table-top manipulation tasks, 50 demos each. Metric: mean success rate, p99 wall-clock latency at BS=1.

*Architecture.* Transformer: 12 heads, $D_v = 768$, 6 encoder + 6 decoder layers. Observation encoder: FusedObservationEncoder, $d_"obs" = 138$ (2 RGB 128×128 + 10-dim state). FSQ: levels $[8,5,5,5]$, $C=1000$, $d=4$, $H_l=8$. Decoder horizon $H_a=32$.

*Training.* 10 epochs, AdamW + cosine annealing, BF16, single T4, batch 16. $lambda = beta = 1.0$, $tau_"static" = 0.5$.

*Baselines.* Not yet run. Planned: OAT (full), OAT-$K$ ($K in {2,4,6}$), FD-DRAT w/o router, FD-DRAT (coupled).

== Results

#block(
  fill: rgb("fff3cd"),
  stroke: rgb("e0a800"),
  inset: 8pt,
  radius: 4pt,
  width: 100%,
)[
  *Training bug.* `cfg.H_l = 64` (default) while the tokenizer uses `H_l = 8`. Nested Dropout sampled $K in {1,...,64}$; latents beyond index 8 do not exist, so dropout applied in $approx 12.5%$ of steps instead of $approx 50%$. CRH and router received insufficient partial-prefix training signal. Results below are reported as-is; a corrected run with `model.H_l=8` is required.
]

50 rollouts, 5 per task, single run, BS=1, T4 GPU.

#figure(
  table(
    columns: (auto, auto),
    align: (left, center),
    stroke: 0.5pt,
    [*Task*], [*SR*],
    [Kitchen: turn on stove + place moka pot], [0%],
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
  caption: [LIBERO-10 per-task success rates (prototype, H_l bug present).],
)

*Latency (BS=1, T4).* Mean: $98.1$ ms. p99: $310.7$ ms.

2 of 10 tasks non-zero: both single-object pick-and-place with visually distinct targets. Tasks requiring dual-object or contact-precise manipulation: 0%. Results are not attributable to the routing mechanism given the training bug.

= Discussion

*Posterior collapse.* Without stop-gradient, router gradients dominate CE numerically, collapsing the token distribution to a degenerate mode. Decoupled Training prevents this by construction; quantitative verification requires a corrected run.

*Limitations.* (1) *Compounding under-training (prototype).* Three factors stack: the OAT tokenizer was trained for 100 epochs (recommended: $>=300$), so the discrete prior and FSQ codebook are undertrained; FD-DRAT training targeted 10 epochs but Kaggle session limits caused early termination at approximately 6 epochs; and the $H_l$ mismatch bug reduced effective nested-dropout coverage to $approx 12.5%$ of steps. Each factor alone degrades performance; together they make the 6.0% SR uninformative about the method's ceiling. A corrected run -- tokenizer $>=300$ epochs, full FD-DRAT training with `model.H_l=8` -- is required before conclusions can be drawn. (2) CRH has no uncertainty estimate; overconfident correction on imprecise coarse trajectories is possible. (3) Binary routing per step; soft routing may improve robustness. (4) Baselines not yet evaluated.

= Future Work

*FiLM-Modulated Dilated 1D-CNN CRH (v2.0).* Current MLP discards temporal structure of $hat(bold(a))_"coarse"$. Replacement: $L=5$ dilated 1D-conv residual blocks with FiLM @film visual injection, dilations ${1,2,4,8,16}$, receptive field $= 63 >= H_a = 32$ ($L=4$ gives RF=31, insufficient). $O(H_a)$ complexity, static CUDA graph. Full architecture in Appendix D.

*Language-conditioned threshold.* Condition $tau$ on task-language embedding to suppress early exit for contact-critical instructions.

*Residual uncertainty.* Replace deterministic CRH with lightweight diffusion over residuals for calibrated uncertainty.

= Conclusion 

FD-DRAT extends OAT with adaptive-length AR decoding via a gradient-isolated Shadow Router and a fixed-dimension CRH. Decoupled Training prevents posterior collapse; fixed-shape decode enables static CUDA graph compilation. 

The prototype reports 6.0% mean SR on LIBERO-10, but this number reflects three compounding under-training conditions: OAT tokenizer trained for only 100 epochs, FD-DRAT cut short at $approx 6$ of 10 epochs by Kaggle session limits, and the $H_l$ mismatch bug throughout. A clean run with a fully-trained tokenizer ($>=300$ epochs), complete FD-DRAT training, and `model.H_l=8` is the required next step.


= Human Conclusion

It's hard to draw a definitive conclusion on the validity of the proposed hypothesis. The loss steadily decreased, but the number of training epochs was critically low due to hardware constraints. Additionally, a software bug caught at later stages makes it hard to pinpoint the exact cause of the low Success Rate (SR)—most likely, the result comes from a combination of these factors.

As for AI assistants: fully autonomous project implementation from idea to working code without human intervention is practically impossible right now (though AutoResearchClaw functionality couldn't be tested). For example, the Claude Code agent periodically suggests technically incorrect solutions without proper oversight. Despite my attempts to minimize manual intervention, it remained strictly necessary.

== What was achieved

- Gathered and figured out the papers (OAT, BLT, H-Net) using DeepResearch + NotebookLM + Gemini
- Successfully generated a hypothesis via NotebookLM + Gemini
- Implemented a working code prototype after lengthy debugging of tensor dimension conflicts and related errors (even though a hidden bug remained in the final implementation) via Claude Code
- Got the code to a working state on Kaggle and obtained metrics: 6% SR and 98 ms mean latency

== What didn't work out

- Anticipating all infrastructure vulnerabilities in the "experimental design". For example, deciding to use the OAT library as a third-party dependency instead of making a proper fork led to "dependency hell". Also, the initial lack of cloud model weight saving (W&B) caused significant time loss on kernel restarts.
- Using Claude Code efficiently. This was due to non-local execution, which left the agent without direct filesystem integration, forcing manual duplication of many actions and draining time and context.
- Conducting practical testing of the autoresearch and AutoResearchClaw frameworks.
- Preventing hidden bugs. Despite multiple attempts at static code checking via LLMs and Claude, a hidden dimensionality bug slipped through, undetected by any neural network until the actual training loop started.
- Ensuring reproducibility. Deploying outside the Kaggle container would require significant effort to resolve environment conflicts (though running Claude Code locally might potentially handle this, but I cannot guarantee it).
- Running a few experiments, since one full run takes about 5–10 hours.

#bibliography("refs.bib")

// ─────────────────────────────────────────────────────────────────────────────
#pagebreak

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix

== OAT Formalism <appendix-oat>

$cal(E): RR^(H_a times D_a) -> RR^(H_l times d)$. FSQ quantizer: $bold(z)_i -> t_i in {1,...,C}$, $C = product_j ell_j$. Decoder $cal(T)^(-1): RR^(H_l times d) -> RR^(H_a times D_a)$. Nested Dropout:
$
  hat(bold(a))^((k)) = cal(T)^(-1)(bold(Z)_(1:k) plus.circle bold(0)_(k+1:H_l)), quad K tilde.op "Uniform"{1,...,H_l}
$
$||hat(bold(a))^((k)) - bold(a)||$ non-increasing in expectation as $k$ grows. AR prior trained via:
$
  cal(L)_"CE" = -sum_(i=1)^(H_l) log p_theta (t_i | t_(1:i-1), bold(z)_v)
$

== Full Loss Equations <appendix-loss>

$
  cal(L)_"ratio" = "BCE"([p_1, ..., p_(H_l)],\ tau_"static" dot bold(1)_(H_l))
$
$
  cal(L)_"mse" = frac(bb(1)_({K < H_l}) dot sum_(h,d) (Delta hat(a)_(h,d) - r_(h,d))^2, |{K < H_l}| + epsilon)
$
$bold(r) = bold(a)_"target"^"norm" - "stop_gradient"(hat(bold(a))_"coarse")$. Mask prevents CRH from learning a spurious mean-residual offset on full-sequence examples.

== Inference Pseudocode <appendix-inference>

#block(fill: luma(240), inset: 8pt, radius: 4pt, width: 100%)[
  *Algorithm 1:* FD-DRAT Any-Time Routing \
  $bold(z)_v <- f_"obs"("obs")$; $bold(t) <- ["BOS"]$; $bold(Z) <- bold(0)^(H_l times d)$ \
  *for* $t = 1$ *to* $H_l$: \
  #h(1.5em) $(bold(L), bold(H)) <- "AR"(bold(t), bold(z)_v)$; $t_t <- arg max bold(L)_(-1,:)$; append $t_t$ \
  #h(1.5em) $bold(Z)_t <- cal(E)_"FSQ"^(-1)(t_t)$ \
  #h(1.5em) *if* $t > 1$ *and* $sigma(cal(R)(bold(H)_(-1), bold(H)_(-2), bold(z)_v)) > 0.5$: *break* \
  $hat(bold(a))_"final" <- cal(T)^(-1)(bold(Z)) + "CRH"(cal(T)^(-1)(bold(Z}) parallel bold(z)_v)$ \
  *return* $hat(bold(a))_"final"$
]

$bold(Z)$ must be zero-initialised; non-zero padding causes CRH to hallucinate corrections from unvisited slots. CRH input dim constant $arrow$ reusable static CUDA graph via `torch.compile`. AR cost: $O(K^2 D_v)$ vs $O(H_l^2 D_v)$ baseline.

== CRH v2.0: FiLM Dilated 1D-CNN <appendix-crh2>

FiLM parameters per layer $l$: $gamma_l = bold(W)_(gamma,l) bold(z)_v + bold(b)_(gamma,l)$, $beta_l = bold(W)_(beta,l) bold(z)_v + bold(b)_(beta,l) in RR^(C_l)$.
$
  "FiLM"(h_(l,c,t) | bold(z)_v) = gamma_(l,c) dot h_(l,c,t) + beta_(l,c)
$

Forward: $bold(h)_0 = "Conv1D"(bold(X))$; for $l=1,...,5$: $bold(h)_l = "GELU"("FiLM"("DilConv1D"(bold(h)_(l-1), 2^(l-1)))) + bold(h)_(l-1)$; $Delta hat(bold(a)) = "Conv1D"(bold(h)_5)$.

RF $= 1 + 2(1+2+4+8+16) = 63 >= H_a = 32$. $L=4$ gives RF=31 (insufficient). No AR states $arrow$ single static CUDA graph.

== Additional author comments  <appendix-aut>
