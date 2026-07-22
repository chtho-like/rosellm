# Kimi K3: Architecture, Systems, Evaluation, and Disclosure Boundary

**Verified through:** 2026-07-22. Kimi K3 was announced on 2026-07-16. The
full weights were scheduled for release by 2026-07-27, and the technical report
was not public at this cutoff. This chapter therefore separates facts disclosed
in Moonshot's launch material from mechanisms established in earlier primary
papers and from hypotheses that remain unverified.

## 1. Executive reading

Kimi K3 is a 2.8-trillion-total-parameter sparse Mixture-of-Experts (MoE)
model with a one-million-token context window and documented text, image, and
video input. Its disclosed architecture combines four largely orthogonal axes:

```text
sequence axis:    3 x Kimi Delta Attention -> 1 x Gated MLA
channel axis:     every token-mixing sublayer -> Stable LatentMoE
depth axis:       Block Attention Residuals selects earlier representations
deployment axis: MXFP4 weights, MXFP8 activations, balanced expert parallelism
```

This is more precise than calling K3 either “a 2.8T MoE” or “a linear-attention
model.” Kimi Delta Attention (KDA) compresses most sequence history into a
fixed-size recurrent state. Periodic global Multi-head Latent Attention (MLA)
layers retain exact content-addressable access. Attention Residuals (AttnRes)
learns which earlier depth representations a later sublayer should read.
Stable LatentMoE moves routed expert computation into a lower-dimensional
latent space so that many more experts can be offered and selected without a
proportional communication penalty.

The launch blog is not a technical report. It does **not** disclose K3's layer
count, hidden width, head count, KDA state dimensions, latent expert dimension,
shared-expert count, active parameter count, visual encoder, visual projector,
frame policy, training-token count, data mixture, compute, reinforcement
learning recipe, or most component equations. Claims about these fields from
secondary launch analyses are not confirmed K3 facts.

## 2. Evidence ledger

| Claim | Evidence at the cutoff | Status |
|---|---|---|
| 2.8T total parameters; 1M context | launch blog and current API guide | disclosed |
| 896 experts; effectively 16 active | launch blog | disclosed, but active parameters unknown |
| 3 KDA layers to 1 Gated MLA layer | K3 architecture figure | disclosed diagram |
| Stable LatentMoE topology | K3 figure plus the LatentMoE paper | topology disclosed; K3 dimensions unknown |
| Block AttnRes backbone | K3 figure plus the AttnRes report | family disclosed; K3 block count unknown |
| Quantile Balancing, Per-Head Muon, SiTU | launch prose | names and intended roles only |
| MXFP4 weights and MXFP8 activations from SFT onward | launch prose | disclosed; quantization recipe unknown |
| native vision and image/video API input | launch blog and API guide | disclosed interface; visual internals unknown |
| approximately 2.5x scaling efficiency over K2 | launch blog | vendor claim without a published metric or curve |
| open weights by 2026-07-27 | launch schedule | future commitment at this cutoff |

“First open 3T-class model” refers to **total parameter scale and intended open
weight release**, not to three trillion parameters being evaluated for every
token. It is also not an evaluation result. Moonshot's accompanying size chart
uses a 2026-07-16 cutoff and a vendor-selected set of flagship open models.

### 2.1 What the size and scaling charts actually say

The launch chart's last disclosed point for each included company is:

| Company | Flagship point in the chart | Total parameters | Date |
|---|---|---:|---|
| Moonshot AI | Kimi K3 | 2,800B | 2026-07-16 |
| DeepSeek | DeepSeek V4 Pro | 1,600B | 2026-04-22 |
| Xiaomi | MiMo V2.5 Pro | 1,020B | 2026-04-27 |
| Thinking Machines | Inkling | 975B | 2026-07-15 |
| Z.AI | GLM-5 | 744B | 2026-02-11 |
| MiniMax | MiniMax M3 | 428B | 2026-06-01 |
| Alibaba | Qwen 3.5 | 397B | 2026-02-16 |

K2's 1,040B point led this selected chart from 2025-07-11 until DeepSeek V4
Pro's 1,600B point on 2026-04-22, which explains the “nine of twelve months”
language. K3 then retook the total-parameter frontier in July. This is a
parameter-count timeline, not a quality, active-compute, memory, latency, data,
or training-compute comparison. “Only open and flagship models” leaves model
selection and the meaning of flagship under the chart author's control.

K3 has 2.692x K2's total parameters. The disclosed routed-expert pool grows
from K2's 384 to K3's 896, or 2.333x, while routed top-$k$ grows from 8 to 16.
K2 disclosed 32.6B active parameters; K3 does not disclose an active count, so
per-token compute growth cannot be derived from the launch numbers.

The separate “approximately 2.5x overall scaling efficiency” statement is not
defined. The blog attributes it jointly to architecture, training methods, and
data recipes but supplies no loss-vs-FLOPs curve, scaling-law equation, token
budget, active FLOPs, baseline, or task aggregation. It must not be translated
into “2.5x faster,” “2.5x fewer active parameters,” or “2.5x cheaper.” It is a
currently unreproducible vendor scaling claim.

## 3. Reading the architecture figure

The right side of the official figure depicts a repeating hybrid backbone:

```text
embedding
  -> [KDA -> Stable LatentMoE] x 3
  -> [Gated MLA -> Stable LatentMoE] x 1
  -> repeat through depth
```

The red side buses do not depict ordinary token attention. They carry the token
embedding and completed block representations to an AttnRes operator before
each token-mixing or channel-mixing sublayer. Each operator has its own learned
pseudo-query $w$ and produces depth-selection weights $\alpha$.

The left side contains two module enlargements:

- KDA projects queries, keys, and values, applies short convolutions, normalizes
  queries and keys, and uses learned decay, delta-update, and output gates.
- Stable LatentMoE keeps the router and shared expert path at model width,
  down-projects the routed path, dispatches only the latent representation to
  selected experts, then normalizes and up-projects the combined routed output.

The drawn number of experts is schematic. It must not be used to infer the
number of shared experts, the hidden sizes, or whether every K3 MoE layer has an
identical expert layout.

## 4. Kimi Delta Attention: the sequence-memory path

### 4.1 From linear attention to a gated delta rule

Ordinary linear attention can be read as an online associative memory:

$$
S_t = S_{t-1} + k_t v_t^\top,
\qquad
o_t = S_t^\top q_t.
$$

$S_t$ accumulates key-value associations, but the unbounded sum has no rule for
correcting stale or conflicting entries. DeltaNet instead performs an online
gradient step on the reconstruction loss
$\tfrac12\operatorname{norm}(S^\top k_t-v_t)^2$:

$$
S_t = (I-\beta_t k_tk_t^\top)S_{t-1}
      +\beta_t k_tv_t^\top.
$$

The rank-one term first erases or corrects the state's prediction along the
current key direction, then writes the new association. Gated DeltaNet adds one
scalar forget factor per head. KDA makes the forget factor channel-wise:

$$
S_t = (I-\beta_t k_tk_t^\top)
      \operatorname{Diag}(\alpha_t)S_{t-1}
      +\beta_t k_tv_t^\top,
\qquad
o_t=S_t^\top q_t.
$$

The distinction is important. A scalar gate gives an entire head one memory
lifetime; $\operatorname{Diag}(\alpha_t)$ lets different key channels forget
at different rates. The Kimi Linear report interprets this learned,
data-dependent decay as both memory management and an implicit positional or
recency mechanism.

### 4.2 Neural parameterization established by Kimi Linear

The precursor Kimi Linear model uses, per head,

$$
q_t,k_t = \operatorname{L2Norm}
  (\operatorname{Swish}(\operatorname{ShortConv}(W_{q/k}x_t))),
$$

$$
v_t = \operatorname{Swish}(\operatorname{ShortConv}(W_vx_t)),
\quad
\alpha_t=f(W_{\alpha,\mathrm{up}}W_{\alpha,\mathrm{down}}x_t),
\quad
\beta_t=\sigma(W_\beta x_t).
$$

Its output path applies head-wise RMSNorm and a data-dependent low-rank gate
before the output projection. K3's module drawing is consistent with this
family, but K3's exact ranks, dimensions, activation $f$, convolution width,
and gate placement remain undisclosed.

### 4.3 Why it can train in parallel and decode recurrently

The token-by-token equation looks sequential. Kimi Linear derives a chunkwise
form using a compact WY-style representation of the rank-one updates. The
implementation is recurrent **between** chunks and parallel **within** a chunk,
turning most work into dense Tensor Core matrix multiplications.

KDA is also a constrained Diagonal-Plus-Low-Rank (DPLR) transition:

$$
D-a_tb_t^\top,
\qquad
D=\operatorname{Diag}(\alpha_t),\quad
a_t=\beta_tk_t,\quad
b_t=\operatorname{Diag}(\alpha_t)k_t.
$$

Tying the low-rank factors to the same key removes secondary chunking and
several matrix multiplications needed by a general DPLR operator. In the Kimi
Linear experiments, the bespoke kernel was nearly twice as fast as the compared
DPLR kernel through 64K sequence length.

For a fixed chunk size $C$ and head dimension $d_h$, the precursor report gives
the per-head KDA attention work as

$$
6Td_h^2+3TCd_h+TC^2,
$$

which is linear in sequence length $T$ for fixed $C$ and $d_h$. Full attention
has a dominant $2T^2d_h$ term. During autoregressive decoding, a KDA layer keeps
a fixed $d_k\times d_v$ matrix state instead of a key/value entry for every
past token.

### 4.4 Why K3 still has periodic full attention

Finite-state linear attention compresses history and can lose exact random
access or collide associations. The official K3 figure retains the precursor's
uniform 3:1 pattern: three KDA token-mixing layers followed by one Gated MLA
layer. The global layer periodically restores exact content-addressable access;
MLA compresses the cached keys and values into a latent representation.

Kimi Linear reported up to 75% KV-cache reduction and up to roughly 6x decoding
throughput at one-million-token context for its 48B-total/3B-active research
model. Those are **not K3 measurements**. The 3:1 topology transfers; the
earlier benchmark numbers do not.

The exact equation for K3's **Gated MLA** is not published. “Gated” could refer
to an output, head, latent, or residual gate. The safe conclusion is only that
the periodic global-attention contribution has learned selectivity. It should
not be equated with similarly named third-party MLA variants.

## 5. Attention Residuals: attention over depth

### 5.1 The problem with the usual residual stream

In a PreNorm transformer, repeatedly adding sublayer outputs yields a hidden
state of the schematic form

$$
h_l=h_0+\sum_{i<l}F_i(h_i).
$$

Every earlier contribution has coefficient one. As depth grows, the residual
norm can grow and each individual representation can become diluted. AttnRes
replaces uniform accumulation with learned, token-dependent selection over
earlier **depths**.

### 5.2 Full AttnRes

For source representations $v_i$, Full AttnRes uses a learned pseudo-query
$w_l$ for layer $l$ and normalized source representations as keys:

$$
\alpha_{i\to l}=
\frac{\exp(w_l^\top\operatorname{RMSNorm}(v_i))}
{\sum_{j<l}\exp(w_l^\top\operatorname{RMSNorm}(v_j))},
\qquad
h_l=\sum_{i<l}\alpha_{i\to l}v_i.
$$

The pseudo-query is a layer parameter, but the keys depend on each token's
earlier representations. The resulting depth weights are therefore
input-dependent. This is not another pass over sequence positions: for each
token position, it asks which earlier **layer output** should be read.

Full AttnRes stores $O(Ld)$ representations per token and performs $O(L^2d)$
depth mixing. Arithmetic is modest because depth $L$ is far smaller than
sequence length, but activation recomputation and pipeline parallelism make
storage and cross-stage communication expensive.

### 5.3 Block AttnRes

Block AttnRes partitions $L$ sublayers into $N$ groups. Inside block $n$, it
maintains a normal residual partial sum

$$
b_n^i=\sum_{j\in B_n,\,j\le i}F_j(h_j).
$$

Across completed blocks, it attends over the embedding $b_0$, completed block
representations $b_1,\ldots,b_{n-1}$, and, after the first sublayer, the current
partial sum $b_n^{i-1}$. This compresses depth history from individual layer
outputs to block summaries and reduces storage and communication from $O(Ld)$
to $O(Nd)$. $N=L$ recovers Full AttnRes; $N=1$ approaches an ordinary residual
stream with the embedding kept as a separate source.

The AttnRes report found that about eight blocks recovered most of Full
AttnRes's benefit in its experiments. It reported a validation-loss point
equivalent to a baseline trained with 1.25x more compute, less than 4% measured
training overhead under pipeline parallelism, and less than 2% typical inference
latency overhead. These are **AttnRes research-model results**, not a disclosure
of K3's block count or an explanation of K3's separate 2.5x scaling claim.

### 5.4 Why the system implementation is nontrivial

The AttnRes report uses two main optimizations:

1. Cross-stage caches retain block representations already received by a
   physical pipeline rank. Later virtual stages send only newly completed
   blocks instead of retransmitting the full history; the peak transition cost
   falls by the virtual-pipeline factor in the report's schedule.
2. A two-phase inference algorithm batches all pseudo-queries in one block
   against completed block representations, then processes the evolving
   intra-block partial sum sequentially and merges the two results with exact
   online-softmax statistics.

For long prefill, block representations are sequence-sharded across
tensor-parallel devices. The paper's 128K-token, eight-block example falls from
15 GB total block-state storage to about 1.9 GB per device with eight-way
sharding, and below 0.3 GB per device with 16K chunked prefill. These figures
explain the feasibility technique; they do not reveal K3's hidden width or its
actual memory footprint.

## 6. Stable LatentMoE: capacity without proportional routed cost

### 6.1 Canonical LatentMoE topology

For model width $d$ and routed latent width $\ell<d$, canonical LatentMoE keeps
the router on the full token $x$ but sends a shared down-projection to routed
experts:

$$
y_{\text{routed}}=
W_{\mathrm{up}}
\left(
\sum_{i\in\operatorname{TopK}(r(x))}
p_iE_i(W_{\mathrm{down}}x;\ell)
\right).
$$

Shared experts, when present, operate at full model width and are added outside
that routed latent path. Dispatch and combine communicate $\ell$-dimensional
rather than $d$-dimensional activations. Routed expert input/output matrices are
also narrower. If $\alpha=d/\ell$, the paper's cost model reduces routed
all-to-all payload and expert-weight bandwidth by approximately $\alpha$.

The saved budget can be spent in two ways:

- increase the number of experts while keeping top-$k$ unchanged to improve
  efficiency; or
- increase both expert count and top-$k$ by $\alpha$ to increase expert-mixture
  diversity at approximately unchanged routed communication and weight-loading
  cost.

The K3 figure matches this topology. It does not disclose $d$, $\ell$,
$d/\ell$, expert intermediate width, shared-expert count, or which LatentMoE
variant K3 uses. “Stable” denotes K3-specific additions whose complete recipe
is still pending.

### 6.2 What 16 of 896 does and does not mean

The routed selection rate is

$$
\frac{16}{896}=1.7857\%,
$$

so 98.2143% of routed experts are inactive for one token. The possible unordered
top-16 subsets number $\binom{896}{16}\approx10^{33.86}$, illustrating the
combinatorial capacity, although real routers use a highly non-uniform fraction
of those combinations.

It is invalid to estimate active parameters as $2.8\text{T}\times16/896$.
Embeddings, attention/KDA, projections, norms, routers, shared experts, and the
latent up/down projections are not multiplied by that fraction, and routed
experts may not dominate total parameters in the same ratio. Moonshot has not
published K3's active parameter count. Secondary estimates such as “30B,”
“50B,” or “16/896 of 2.8T” are therefore unsupported at this cutoff.

### 6.3 Quantile Balancing

Extreme sparsity turns routing imbalance into both an optimization problem and
a systems straggler problem. The K3 blog says Quantile Balancing derives expert
allocation directly from router-score quantiles, eliminating heuristic bias
updates and a sensitive balancing hyperparameter.

A public mathematical reconstruction by Jianlin Su frames balanced top-$k$
routing as maximizing router scores subject to approximately equal expert
loads. With token thresholds $a_i$ and expert offsets $b_j$, an alternating
quantile procedure has the form:

1. select the token's top-$k$ experts from adjusted scores $s_{ij}-b_j$;
2. update each token threshold from its relevant order statistic;
3. update each expert offset from the required score quantile over tokens; and
4. carry the offsets forward rather than applying a hand-tuned sign update.

This explains the launch phrase, but it is not yet a confirmed K3 algorithm.
The report must still specify the batch/domain over which quantiles are taken,
distributed approximation, causality, tie handling, capacity, router weights,
and whether the offset affects selection only or also combine weights.

Quantile balancing is not free of trade-offs. Exact balance can force some
lower-score routes, while weak balance permits hot experts and stragglers. The
technical question is how K3 trades semantic specialization against hardware
regularity.

## 7. Optimizer and activation-control disclosures

### 7.1 Per-Head Muon

Muon maintains gradient momentum for a matrix and approximately applies its
polar factor, commonly through Newton-Schulz iterations. If
$M=UDV^\top$, the idealized update direction is $UV^\top$: singular
directions are equalized rather than scaled in proportion to their raw gradient
singular values. Moonshot's scalable Muon work added weight decay and
shape-aware update-RMS matching; its scaling-law experiments reported matching
AdamW loss with about 52% of the compute.

K3's new phrase “Per-Head Muon” says attention heads are optimized
independently. Conceptually, a fused attention projection gradient is reshaped
into head matrices and orthogonalized per head, avoiding one singular spectrum
coupling otherwise specialized heads. The blog does not disclose which of
Q/K/V/O, KDA, MLA, or low-rank matrices use this treatment, how rectangular
matrices are oriented, the Newton-Schulz polynomial and iteration count, or the
per-head learning-rate/RMS rule. It should not be described as a published
K3 optimizer equation yet.

### 7.2 SiTU

The only official expansion is **Sigmoid Tanh Unit**, and the only official
role is better activation control. No equation or placement is published. The
name is compatible with a sigmoid-gated tanh family, but it does not prove
$\sigma(a)\times\operatorname{tanh}(b)$, a replacement for SwiGLU, or an
expert-specific gate.
Claims that SiTU replaces a particular K3 activation are hypotheses until the
report or weights expose the module.

### 7.3 Gated MLA

The figure places Gated MLA in every fourth token-mixing position and says it
improves attention selectivity. MLA itself compresses the key/value path into a
latent representation to reduce cache and bandwidth. The precise K3 gate is
unknown; it may gate heads, values, the latent path, the output, or the residual
contribution. Only its macro placement and stated purpose are disclosed.

## 8. Quantization-aware training

Moonshot says K3 applies Quantization-Aware Training (QAT) **from supervised
fine-tuning onward**, using MXFP4 weights and MXFP8 activations.

The Open Compute Project MX specification defines a 32-element microscaling
block with one shared eight-bit E8M0 power-of-two scale. MXFP4 elements use an
E2M1 four-bit format; MXFP8 supports E4M3 or E5M2 eight-bit elements. Compared
with one scale for a whole tensor, the local scale reduces the effect of
outliers while retaining compact elements.

“From SFT onward” does not say that the entire pretraining run used four-bit
weights. In a typical QAT pipeline, the forward pass simulates or uses target
rounding, clipping, and scales while higher-precision master weights and
optimizer state may still be retained. K3's exact master precision, scale
calculation, rounding, accumulators, excluded layers, router precision,
KDA-state precision, MLA/KV-cache precision, and measured quality delta are
unknown.

At the raw element level, 2.8T weights require about 5.6 TB in BF16 and 1.4 TB
at four bits. The latter is only a lower bound: MX scales, packing, metadata,
unquantized parameters, runtime workspaces, states, and replication add memory.
The OCP format is vendor-neutral, which supports portability, but not every
accelerator has equally efficient native MXFP4 instructions.

## 9. Distributed training and inference

### 9.1 Fully balanced expert parallelism

With expert parallelism, experts are sharded across accelerators and each token
is dispatched by all-to-all communication to the owners of its selected
experts. A hot expert makes its rank the step's straggler. Variable token counts
also create dynamic buffer shapes that hinder preallocation, graph capture, and
kernel fusion.

K3's launch says its method maintains fully balanced expert-parallel execution
with static shapes and no host synchronization on the critical path. The likely
systems consequences are fixed communication sizes, device-side coordination,
and predictable kernels. The exact mechanism—capacity slots, padding, token
reordering, dropping, replication, or a combination—has not been published.
Quantile Balancing and static-shape execution complement one another, but they
are not necessarily the same algorithm.

### 9.2 Why Moonshot recommends a 64-accelerator supernode

Top-16 routing across 896 experts generates many destinations. LatentMoE makes
each routed message narrower, but it does not eliminate all-to-all traffic.
Keeping expert ranks inside one large high-bandwidth fabric avoids turning
inter-node links into the dominant latency.

At raw MXFP4 element size, the 1.4 TB lower-bound weight payload averages about
21.9 GB across 64 accelerators; BF16 would average 87.5 GB. Real deployments
need more memory for scales, non-quantized tensors, cache/state, buffers,
runtime, and possibly expert replication. Thus 64 is an operationally sensible
recommendation, not proof that every 64-device topology will serve K3 well and
not a formal minimum for every offload configuration.

### 9.3 KDA prefix caching

Conventional prefix caching reuses the key/value tensors produced by an
identical token prefix. A KDA layer instead ends the prefix with a recurrent
matrix state. A hybrid K3 cache must preserve both:

```text
KDA layers:       recurrent/chunk boundary state
Gated MLA layers: latent key/value prefix blocks
AttnRes:          required block representations and metadata
```

Branching, block hashing, chunk boundaries, state composition, eviction, and
transfer between prefill and decode workers therefore differ from a pure KV
cache. Moonshot says it contributed a KDA prefix-cache implementation to vLLM,
scheduled for release with the model. The implementation was not available at
this cutoff, so its state layout and reuse granularity cannot yet be described
as fact.

### 9.4 Mooncake and the price claim

Mooncake separates compute-heavy long-prompt prefill from memory-bandwidth-heavy
autoregressive decode. Its scheduler chooses prefill and decode workers using
cache locality, load, and time-to-first-token/time-between-token service goals.
The original system stores reusable cache blocks across GPU memory, host DRAM,
and SSD, and transfers cache layer by layer so transfer can overlap compute.

K3's API launch price is \$0.30 per million cache-hit input tokens, \$3.00 per
million cache-miss input tokens, and \$15.00 per million output tokens. The input
cache-hit discount is 10x. At exactly a 90% hit rate, the weighted input price
would be

$$
0.9(0.30)+0.1(3.00)=\$0.57/\text{MTok},
$$

81% below the all-miss input price; output cost remains separate. Moonshot says
its coding workload exceeds 90% cache hits, which is plausible because agents
repeatedly resend stable system prompts, repository context, tool definitions,
and conversation prefixes. It is a provider workload statistic, not a
guarantee for arbitrary applications.

The current API automatically attempts prefix caching when the previous prompt
exceeds 256 tokens; users do not supply a cache ID or TTL. Prefix identity and
stable ordering therefore matter economically as well as computationally.

## 10. Native vision and video: what is actually known

The launch says K3's native multimodal architecture understands text, images,
and video “within the same model.” The API accepts an image as base64 or an
uploaded `ms://` media identifier, and video through an uploaded media ID.
Public internet image URLs are not accepted by the current guide.

This establishes first-class visual input and a shared model endpoint. It does
not answer the internal representation questions:

- visual encoder family and size;
- image patch size, tiling, resolution limits, and resampler/projector;
- video decode rate, scene-aware sampling, temporal tubelets, pooling, and
  audio handling;
- number and placement of visual positions in the one-million-token context;
- whether visual tokens pass through every K3 block or modality-specific
  experts;
- pretraining image/video quantity, ordering mixture, SFT, RL, and rewards.

K2.5's disclosed MoonViT-3D plus MLP projector is a useful predecessor but
cannot be copied into K3 as fact. A modality-specific vision encoder would also
not contradict “native”: native multimodality ordinarily means first-class
joint training and backbone integration, not raw pixels entering a text
embedding table without a visual front end.

K3's output contract remains text, code, structured output, and tool calls. The
blog does not disclose a visual codebook, diffusion/flow decoder, Variational
Autoencoder, or video decoder. Its image/video editing demonstrations may use
code, renderers, media tools, and iterative visual feedback; they are not proof
that the K3 checkpoint directly samples image or video latents.

## 11. Agent behavior and product-system composition

Long-horizon coding and knowledge work are not just single-pass checkpoint
properties. The demonstrated system combines the model with a harness, terminal
and browser tools, sandboxes, artifact renderers, context management, Kimi Work
or Kimi Code orchestration, and sometimes concurrent subagents.

K3's current API features include:

- always-on reasoning with `reasoning_effort` set to `low`, `high`, or `max`;
- streamed `reasoning_content` before final `content`;
- strict JSON Schema output and partial-prefix continuation;
- custom tool calls and tool-choice constraints;
- tool definitions inserted dynamically through a system message; and
- one-million-token context with automatic prefix caching.

The maximum completion-token field defaults to 131,072 and can be set as high
as 1,048,576. The service fixes temperature at 1.0, top-p at 0.95, `n=1`, and
both frequency and presence penalties at zero.

### Preserved thinking is an API state contract

K3 was trained with preserved thinking history. Every later request in a
multi-turn or tool loop must return the complete previous assistant message,
including `reasoning_content`, not merely the final `content`. This does not
mean hidden neural state is transferred magically between calls: the harness
serializes the previous reasoning into the next model input. It consumes context
and billable tokens.

Dropping or rewriting that history, or switching an existing session from a
different model into K3, creates a prompt distribution different from the one
used in training. Moonshot explicitly warns that output can become highly
unstable. This is why a benchmark score is partly a model-plus-harness result.

## 12. What the launch demonstrations establish

| Demonstration | Disclosed result | Correct evidential reading |
|---|---|---|
| GPU kernel optimization | four tasks, identical sandbox, up to 24 hours | best trajectory under a long tool loop, not one-shot coding accuracy |
| MiniTriton compiler | DSL, tile IR over MLIR, passes, PTX, runtime; nanoGPT convergence | coherent end-to-end artifact claim; code and independent reproduction absent |
| game/front-end/CAD | code iterated from live screenshots and visual inputs | evidence for vision-in-the-loop agent use, not pixel generation |
| chip design | 48-hour run, Nangate 45 nm, <=4 mm2, 100 MHz, simulated >8,700 token/s | simulated proof of concept on an old open library, not fabricated silicon |
| I-Love-Q research | 20+ papers, 300+ equations of state, 3,000+ Python lines, about two hours | curated case without released artifact/rubric at the cutoff |
| AI-ASIC report | 120+ refinement rounds, 2.8K+ searches/fetches, 1.1K+ terminal pulls, 11K+ pages | activity and system-scale counts, not direct correctness metrics |
| GWTC-5 analysis | 391 events, 20+ concurrent subagents, 7 visualizations, 2 tables | Kimi Work orchestration plus K3, not a bare checkpoint test |
| video editing | 56 source clips, motion-matched cuts, beat sync, audio processing, revisions | multimodal planning/tool use; internal video tokenizer and editor tools undisclosed |

The kernel chart contains four panes. The exact values are:

| Task/metric | K3 | Fable 5 | Opus 4.8 | GPT 5.6 Sol | GPT 5.5 |
|---|---:|---:|---:|---:|---:|
| AttnRes speedup over FLA Triton | 59.66 | 57.12 | — | 17.34 | 30.76 |
| DSA speedup over FLA Triton | 55.13 | 57.34 | 43.61 | 48.86 | 38.23 |
| MLA-512 achieved TFLOPS | 517.8 | 492.7 | 393.6 | 361.5 | 386.2 |
| KDA-GPGPU speedup over FLA Triton | 73.56 | 56.46 | 56.99 | 66.66 | — |

The prose does not expand the chart label `DSA`; reading it as DeepSeek Sparse
Attention is plausible but remains an inference. Fable 5 was evaluated by a
third party and may include fallback. Some trajectories used small precision
shortcuts within the evaluator's numerical tolerance.

## 13. Complete launch benchmark table

The following reproduces all 35 rows in the launch table. A dash means the
blog did not report a result. These are vendor-collected or vendor-assembled
numbers, not an independent reproduction.

### 13.1 Coding

| Benchmark | K3 | Fable 5 | GPT 5.6 Sol | Opus 4.8 | GPT 5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSWE | 67.5 | 70.0 | 73.0 | 59.0 | 67.0 | 46.2 |
| Program Bench | 77.8 | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 84.6 | 88.8 | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | 86.6 | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE Marathon | 42.0 | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrain Bench | 36.6 | 41.4 | 34.6 | 34.1 | 28.4 | 34.3 |
| MLS Bench Lite | 48.3 | 49.9 | 46.2 | 42.8 | 35.5 | 40.4 |
| KCB 2.0, internal | 72.9 | 76.9 | 64.8 | 71.7 | 69.0 | 64.2 |

### 13.2 Productivity and agents

| Benchmark | K3 | Fable 5 | GPT 5.6 Sol | Opus 4.8 | GPT 5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| GDPval-AA, Elo | 1668 | 1760 | 1748 | 1600 | 1494 | 1514 |
| BrowseComp | 91.2 | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA, F1 | 95.0 | 94.2 | — | 93.1 | — | — |
| Toolathlon | 73.2 | 77.9 | 74.9 | 76.2 | 73.5 | 59.9 |
| MCP Atlas | 84.2 | 84.7 | 83.6 | 83.6 | 82.8 | 82.6 |
| AutomationBench | 30.8 | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| JobBench | 52.9 | 57.4 | 46.5 | 48.4 | 38.3 | 43.4 |
| AA-Briefcase, Elo | 1548 | 1583 | 1495 | 1354 | 1158 | 1260 |
| APEX-Agents | 41.0 | 43.3 | 39.9 | 39.4 | 38.5 | 35.6 |
| OfficeQA Pro | 63.3 | 69.9* | 63.2* | 63.9* | 60.9* | 41.4 |
| SpreadsheetBench 2 | 34.8 | 34.7* | 32.4* | 31.55* | 29.05* | 28.12 |
| DECK Bench, internal | 73.5 | 73.0 | 74.7 | 66.9 | 68.2 | 68.6 |

### 13.3 Reasoning

| Benchmark | K3 | Fable 5 | GPT 5.6 Sol | Opus 4.8 | GPT 5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| GPQA Diamond | 93.5 | 92.6 | 94.1 | 91.0 | 93.5 | 91.2 |
| Humanity's Last Exam | 43.5 | 53.3 | 44.5 | 49.8* | 41.4* | — |
| Humanity's Last Exam, tools | 56.0 | 63.0 | 58.0 | 57.9* | 52.2* | — |

### 13.4 Vision

| Benchmark | K3 | Fable 5 | GPT 5.6 Sol | Opus 4.8 | GPT 5.5 | GLM-5.2 |
|---|---:|---:|---:|---:|---:|---:|
| MMMU-Pro | 81.6 | 81.2 | 83.0 | 78.9 | 81.2 | — |
| MMMU-Pro with Python | 83.4 | 86.5 | 84.6 | 82.7 | 83.2 | — |
| CharXiv-RQ | 84.8 | 88.9 | 84.6 | 80.5 | 84.1 | — |
| CharXiv with Python | 91.3 | 93.5 | 89.1 | 89.9 | 89.0 | — |
| MathVision | 94.3 | 94.8 | 95.8 | 86.7 | 92.2 | — |
| MathVision with Python | 97.8 | 98.6 | 97.8 | 97.1 | 96.8 | — |
| BabyVision with Python | 85.7 | 90.5 | 88.9 | 81.2 | 83.6 | — |
| ZeroBench, pass@5 | 23 | 23 | 17 | 17 | 22 | — |
| ZeroBench with Python, pass@5 | 41 | 46 | 35 | 34 | 41 | — |
| WorldVQA, ForceAnswer | 51.0 | 56.7 | 41.8 | 39.1 | 38.5 | — |
| OmniDoc | 91.1 | 89.8 | 85.8 | 87.9 | 89.4 | — |
| PerceptionBench, internal | 58.5 | 57.2 | 59.7 | 47.2 | 55.8 | — |

The asterisk is reproduced from Moonshot's table. The accessible launch text
does not define it globally, so it should not be silently interpreted as one
uniform provenance class.

## 14. Benchmark methodology and comparability limits

All K3 results use maximum reasoning effort, temperature 1.0, and top-p 1.0.
Depending on the benchmark, Moonshot used KimiCode, Claude Code, or Codex. This
creates several important qualifications:

- K3 uses KimiCode on DeepSWE, Terminal-Bench, Program Bench, FrontierSWE, MLS
  Bench Lite, and at least one KCB run. Competitors sometimes use their best
  reported harness rather than the same harness.
- The official DeepSWE leaderboard gives K3 67.3 under mini-SWE-agent; the table
  gives 67.5 under KimiCode.
- SWE Marathon uses Moonshot's H20-calibrated branch. Correctness and anti-cheat
  checks remain, but GPU images, performance gates, and reference oracles differ
  from the official hardware setting. Fable 5 fell back on 35% of these tasks.
- PostTrain Bench uses three-run averages on H20 rather than the official H100,
  with different model-specific harnesses.
- FrontierSWE dominance was recomputed from raw scores on 2026-07-16.
- KCB 2.0, DECK Bench, and PerceptionBench are internal. Their samples and full
  rubrics are unavailable; 10% of KCB tasks triggered GPT 5.6 Sol's cyber guard.
- OfficeQA Pro provides complete PDF corpora rendered as images, with no
  machine-readable text. Model+harness OCR and visual navigation are therefore
  part of the result.
- MCP Atlas uses the 500-task public subset, a 100-turn limit, and Gemini 3.1 Pro
  as judge. AutomationBench uses the 600-task public subset.
- BrowseComp uses context compaction at 300K tokens for the reported 91.2. With
  the raw 1M window and no context management, K3 scores 90.4.
- Vision scores are three-run averages except ZeroBench, which is run five
  times. MMMU-Pro preserves original ordering and prepends images to text.
- GDPval-AA, AA-Briefcase, and APEX scores are cited from Artificial Analysis,
  while several other competitor scores come from vendor pages or leaderboards.

The launch benchmark's top-p 1.0 also differs from the current public API
guide, which fixes top-p at 0.95. That implies a separate evaluation path or a
documentation/configuration difference and is another reason not to assume the
public endpoint reproduces every launch score bit-for-bit.

The table therefore supports “K3 is broadly frontier-competitive and usually
behind the strongest proprietary models overall,” which Moonshot itself states.
It does not support a clean single-model ranking independent of harness,
reasoning budget, tool policy, safety policy, hardware, context management, and
provenance.

## 15. Disclosed limitations

Moonshot names three limitations:

1. **Thinking-history sensitivity.** Omitting historical reasoning or switching
   models mid-session can make generation highly unstable.
2. **Excessive proactivity.** Training emphasizes difficult long-horizon work,
   so K3 may resolve small ambiguities or make decisions on the user's behalf.
   Production systems should use explicit behavioral constraints, tool
   allowlists, permissions, budgets, and confirmation before irreversible
   actions.
3. **User-experience gap.** Moonshot says overall conversational/product
   experience still noticeably trails Claude Fable 5 and GPT 5.6 Sol despite
   competitive benchmarks.

These admissions expose a general lesson: capability, instruction following,
calibrated initiative, harness compatibility, and user experience are distinct
axes. Long-horizon task-completion training can improve persistence while
worsening the tendency to over-act under ambiguity.

## 16. Availability at the cutoff

- Kimi.com and mobile apps on iOS, Android, and HarmonyOS;
- Kimi Work 3.1.0 or later on Windows and Apple silicon Macs;
- Kimi Code through the `/model` selector;
- API model identifier `kimi-k3`; and
- enterprise account separation and member management.

The launch initially advertised maximum thinking only, with lower efforts to
follow. By 2026-07-22, the current API guide documented `low`, `high`, and `max`,
with `max` as default. This is a useful example of why launch prose and current
API behavior need separate dates.

## 17. Questions the technical report and weights must answer

1. Exact layer count, model width, attention/KDA heads and state dimensions.
2. Total versus active parameter accounting and shared-expert structure.
3. Stable LatentMoE's latent width, expert width, stability additions, capacity,
   overflow, and router equations.
4. Exact Quantile Balancing and fully balanced expert-parallel algorithms.
5. Per-Head Muon parameter grouping and hyperparameters; SiTU and Gated MLA
   equations and ablations.
6. Pretraining tokens, modalities, data composition, curriculum, compute, and
   the definition of the claimed 2.5x scaling efficiency.
7. Visual/video encoder, sampling, projector/resampler, token budget, joint
   training stages, and visual RL.
8. MX quantization block/scaling details, exception layers, accumulator and
   state/cache precision, plus quality and throughput ablations.
9. KDA/MLA/AttnRes cache serialization, hashing, branching, composition, and
   vLLM implementation.
10. Reproducible artifacts and judging protocols for the launch cases and
    internal evaluations.

## Primary sources

- Moonshot AI, [Kimi K3 launch blog](https://www.kimi.com/de/blog/kimi-k3),
  [K3 API guide](https://platform.kimi.com/docs/guide/kimi-k3-quickstart),
  [thinking-mode guide](https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model),
  and [visual-input guide](https://platform.kimi.com/docs/guide/use-kimi-vision-model).
- Moonshot AI, [Kimi Linear](https://arxiv.org/abs/2510.26692) and
  [official implementation](https://github.com/MoonshotAI/Kimi-Linear).
- Kimi Team, [Attention Residuals](https://arxiv.org/abs/2603.15031) and
  [official implementation](https://github.com/MoonshotAI/Attention-Residuals).
- NVIDIA and collaborators,
  [LatentMoE](https://arxiv.org/abs/2601.18089).
- Moonshot AI, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
  and [Moonlight implementation](https://github.com/MoonshotAI/Moonlight).
- Moonshot AI and Tsinghua University,
  [Mooncake](https://arxiv.org/abs/2407.00079).
- Open Compute Project,
  [Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
- Jianlin Su, [MoE journey: quantile balancing](https://kexue.fm/archives/11619),
  used only as a public mathematical reconstruction while K3's official
  algorithm remains unpublished.

Continue with [Kimi model lineage and native vision](kimi.md) for the K2 to K3
history and [representation and fusion](architecture.md) for the cross-vendor
meaning of native multimodality.
