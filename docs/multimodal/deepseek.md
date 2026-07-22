# DeepSeek's Multimodal Portfolio: VL, Janus, OCR, and V4

**Verified through:** 2026-07-22. The most important boundary is that DeepSeek's
current flagship V4 API is officially text-only even though DeepSeek has several
separate open multimodal research families.

## 1. One organization, four distinct model lines

“DeepSeek supports images” and “the DeepSeek base model supports images” are not
equivalent statements. The public portfolio is segmented:

| Line | Primary goal | Input | Output | Relationship to the flagship text model |
|---|---|---|---|---|
| DeepSeek-VL/VL2 | general visual understanding and grounding | image + text | text, boxes, coordinates | separate VLM checkpoints using DeepSeek language backbones |
| Janus/Janus-Pro/JanusFlow | unified image understanding and generation | image/text | text and images | separate multimodal research checkpoints |
| DeepSeek-OCR/OCR 2 | high-resolution document parsing and optical context compression | page/scene image + prompt | text, Markdown, layout serialization | specialized encoder + small DeepSeek MoE decoder |
| DeepSeek V3/V4 | general reasoning, coding, tools, and agents | text | text/code/tools | production flagship API line; V4 is text-only at the cutoff |

An application can compose them:

```text
image or PDF
  -> DeepSeek-OCR, VL2, or another vision service
  -> text/Markdown/coordinates
  -> DeepSeek V4 reasoning and tool use
```

That is a strong production architecture, but it is a cascade. V4 does not see
the original pixels after conversion.

DeepSeek's own
[GitHub Copilot integration guide](https://api-docs.deepseek.com/quick_start/agent_integrations/github_copilot)
states that V4 is text-only and describes using another installed model to
produce image descriptions before sending them to V4. This is unusually clear
first-party evidence for the boundary.

## 2. DeepSeek-VL: the first general understanding path

The [DeepSeek-VL paper](https://arxiv.org/abs/2403.05525) uses a hybrid visual
encoder:

- SigLIP-L at 384 by 384 supplies semantic features;
- a SAM-B path at 1024 by 1024 preserves higher-resolution local detail;
- an adaptor/projector aligns visual features with the language width; and
- a DeepSeek language model autoregressively produces text.

The design is input-side visual understanding. It does not contain Janus's image
generation path. Its fixed-resolution dual encoder also requires two visual
preprocessing streams, which motivated VL2's dynamic tiling redesign.

## 3. DeepSeek-VL2, tensor by tensor

The [DeepSeek-VL2 report](https://arxiv.org/abs/2412.10302) identifies three
modules:

```text
dynamic-resolution image tiles
  -> SigLIP-SO400M-384 vision encoder
  -> pixel-shuffle compression + two-layer MLP adaptor
  -> DeepSeekMoE language model with MLA
  -> text and grounding serialization
```

### 3.1 Dynamic tiling

For a high-resolution image, VL2 chooses a candidate canvas
$(m\cdot384,n\cdot384)$ with $mn\leq9$ that minimizes padding after
aspect-ratio-preserving resize. It creates:

- one 384 by 384 global thumbnail; and
- $m\times n$ local 384 by 384 tiles.

The report disables dynamic tiling when more than two images are supplied to
control context and compute. This is a concrete example of an API-level media
count changing the vision preprocessing policy.

Each tile passes through SigLIP-SO400M-384 and yields a 27 by 27 grid:

$$
27\times27=729
$$

visual embeddings, each of width 1,152.

### 3.2 Pixel shuffle and sequence layout

A 2 by 2 pixel-shuffle operation reorganizes neighboring spatial features,
reducing each tile from 729 positions to

$$
14\times14=196
$$

positions while increasing feature width before projection. VL2 then inserts
row-newline and view-separator embeddings so the one-dimensional language
sequence retains the layout of the global and local grids.

The report gives the full sequence length as

$$
N_v=210+1+m\cdot14\left(n\cdot14+1\right),
$$

where 210 is the global 14 by 14 grid plus 14 row markers and one position is the
view separator. A two-layer MLP maps this sequence into the language-model
embedding space.

### 3.3 Language backbone and training

VL2 uses DeepSeekMoE backbones with Multi-head Latent Attention (MLA). Public
variants have 3B, 16B, and 27B total LLM parameters, with the report listing
0.57B, 2.4B, and 4.1B activated LLM parameters before counting the vision
component. Product/model-card summaries sometimes round or include components
differently; preserve the report's denominator when comparing numbers.

Training has three stages:

1. align the dynamic vision encoder and adaptor while the language model is
   fixed;
2. jointly pretrain vision encoder, adaptor, and LLM on roughly 800B image-text
   tokens; and
3. supervised instruction tuning with all modules trainable while loss is
   applied to answer text tokens.

This is a jointly optimized VLM, not a runtime OCR-to-text proxy, despite its
recognizable encoder-projector-LLM decomposition.

## 4. DeepSeek-OCR: optical compression, not just character recognition

The [DeepSeek-OCR paper](https://arxiv.org/abs/2510.18234) asks an LLM-centric
question: how many visual positions are needed to reconstruct a page that would
occupy many more text positions? OCR supplies measurable source and target
sequences for studying that compression.

### 4.1 DeepEncoder architecture

```text
image
  -> SAM-base ViTDet, patch size 16, window-dominant attention (~80M)
  -> two 3x3 stride-2 convolutions
  -> 16x reduction in spatial token count
  -> CLIP-L global-attention stack (~300M, patch embed removed)
  -> concatenate CLIP and compressed SAM features (2,048 width)
  -> linear projector to 1,280
  -> DeepSeek-3B-MoE decoder (~570M active)
  -> OCR/layout text
```

The serial window-then-global design is deliberate. A 1024 by 1024 image with
patch size 16 first produces

$$
64\times64=4096
$$

patch positions. Two stride-2 convolutions reduce the spatial grid by four in
each dimension, so only 256 positions enter dense global attention. This keeps
high-resolution local processing relatively cheap and prevents global
attention activation from scaling over all 4,096 positions.

The released inference artifact confirms a 2,048-to-1,280 linear projector and
concatenation of the CLIP and SAM streams. These are artifact-confirmed details,
not an inference from a diagram.

### 4.2 Resolution and visual-token modes

| Mode | Input resolution | Encoder output positions |
|---|---:|---:|
| Tiny | 512 by 512 | 64 |
| Small | 640 by 640 | 100 |
| Base | 1024 by 1024 | 256 |
| Large | 1280 by 1280 | 400 |
| Gundam | local 640 tiles + global 1024 view | $100k+256$, for a bounded tile count $k$ |
| Gundam-M | local 1024 tiles + global 1280 view | $256k+400$ |

These modes expose the fidelity/sequence trade-off directly. The paper reports
about 97% decoding precision when ground-truth text positions are less than ten
times the visual positions in its compression study, and about 60% around 20x.
Those figures belong to the authors' benchmark/protocol; they are not a theorem
or a guarantee for arbitrary fonts, languages, corruptions, or layouts.

### 4.3 MoE decoder

The 3B DeepSeekMoE decoder activates six of 64 routed experts plus two shared
experts, approximately 570M parameters per position. It reconstructs text and
layout markup autoregressively from the continuous compressed visual sequence.

This means optical compression does not make long-context processing free:

- rendering or page images must be produced;
- the vision encoder still performs work;
- lossy visual compression can hallucinate plausible text;
- reconstructed output tokens still incur autoregressive decoding cost; and
- general language reasoning over a visual memory was not established by OCR
  reconstruction alone.

### 4.4 Training data and operational role

The paper discloses about 30M PDF pages across roughly 100 languages, including
coarse and finer annotations, plus about 3M Word-derived examples, scene OCR,
charts, formulas, chemical structures, geometry, and general visual data. It
reports production throughput above 200,000 pages/day on one A100-40G, and a
larger internal data-generation deployment. These are vendor measurements whose
document distribution and operational stack matter.

The immediate capability is high-throughput document-to-training-data parsing.
The research hypothesis is that optical representations could become a compact
long-context memory medium. The former is demonstrated more directly than the
latter.

## 5. DeepSeek-OCR 2: an LLM-style vision encoder

The [OCR 2 paper](https://arxiv.org/abs/2601.20552) retains the overall
encoder-projector-MoE decoder but replaces OCR 1's CLIP global stack with
**DeepEncoder V2**, initialized from a Qwen2-0.5B-like decoder-only transformer.

### 5.1 Visual tokenizer

SAM-base plus two stride-2 convolutions still provide local perception and 16x
spatial-position compression. The final convolutional width changes from 1,024
to 896 to match the LLM-style encoder.

### 5.2 Causal-flow queries

For each view, DeepEncoder V2 concatenates compressed visual embeddings with an
equal number of learned query embeddings:

```text
[visual prefix: m positions] [learned causal queries: m positions]
```

The block attention mask is

$$
M=
\begin{bmatrix}
\mathbf 1_{m\times m} & \mathbf 0_{m\times m}\\
\mathbf 1_{m\times m} & \operatorname{LowerTri}(\mathbf 1_{m\times m})
\end{bmatrix}.
$$

Therefore:

- visual-prefix positions attend bidirectionally to other visual positions and
  cannot see later query positions;
- query $q_i$ sees all visual positions and only queries $q_{\leq i}$; and
- only the final query outputs are projected and sent to the DeepSeek decoder.

The queries do not perform a hard permutation or emit indices saying “read patch
17 next.” They learn an ordered sequence of continuous summaries. Calling this
“dynamic semantic reordering” is the authors' interpretation of the causal
query mechanism, not a literal sorting algorithm.

### 5.3 Concrete released configuration

The official code instantiates:

| Item | Value |
|---|---:|
| LLM-style encoder layers | 24 |
| hidden width | 896 |
| query heads | 14 |
| key/value heads | 2 |
| feed-forward width | 4,864 |
| local 768 view queries | 144 |
| global 1024 view queries | 256 |
| projector | linear 896 -> 1,280 |

With $k$ local crops, the decoder receives

$$
N_v=144k+256,
\qquad 0\leq k\leq6,
$$

or 256 to 1,120 visual positions per image in the reported multi-crop policy.

### 5.4 Training stages

1. Pretrain the visual tokenizer and LLM-style encoder with a lightweight
   decoder and next-token objective; the report states 160 A100 GPUs, 40K
   iterations, and about 100M image-text pair samples.
2. Freeze the SAM-convolution tokenizer, then jointly improve the LLM-style
   encoder queries and DeepSeek decoder for 15K iterations.
3. Freeze all of DeepEncoder V2 and continue training the decoder for 20K
   iterations to consume data faster.

These are unusually concrete disclosures, but they describe OCR 2, not a future
DeepSeek V5 or the current V4 production model.

## 6. What OCR 2 says about DeepSeek's direction

Section 6.2 of the OCR 2 paper explicitly presents a research direction toward
native multimodality: one LLM-style encoder could share key/value projections,
attention, and feed-forward layers across modalities while using
modality-specific learned queries to compress text, extract speech features, or
reorganize vision.

This is a proposal and early validation path, not a released omni model. The
careful planning interpretation is:

1. OCR 1 explores visual tokens as compact document memory.
2. OCR 2 tests whether language-model-style causal queries can replace a CLIP
   global encoder and improve reading order.
3. The authors propose extending the shared encoder idea to text and audio.
4. No public DeepSeek flagship checkpoint at the cutoff implements that complete
   shared omni encoder.

The paper therefore reveals research intent more clearly than product schedule.

## 7. Janus: understanding and image generation in one system

DeepSeek's [Janus repository](https://github.com/deepseek-ai/Janus) contains
Janus, Janus-Pro, and JanusFlow. These models are the answer to “does DeepSeek
have an LLM-centered model that can generate images?”—yes, as a separate
research family.

### 7.1 Janus and Janus-Pro

Janus deliberately decouples two visual representations:

- an understanding encoder supplies continuous semantic visual features; and
- a vector-quantized image tokenizer supplies discrete codes for generation.

Both paths connect to one autoregressive transformer. For understanding, image
features condition text-token prediction. For generation, the transformer
predicts discrete image codebook entries, which a VQ decoder converts back to
pixels.

The separate encoders solve a real conflict: a representation invariant enough
for semantic understanding is not necessarily detailed enough for pixel
reconstruction. “Unified model” therefore does not mean one identical encoder
for every operation.

Janus-Pro scales data and model size and refines training, but its public 1B/7B
checkpoints remain far smaller and separate from DeepSeek V4.

### 7.2 JanusFlow

JanusFlow replaces discrete autoregressive image generation with rectified flow.
The language transformer participates in predicting the velocity/noise-removal
direction for continuous image latents over multiple integration steps, while
text generation remains autoregressive. A VAE-like image representation and
flow solver still surround the transformer.

This is “image generation in an LLM framework,” but not one forward pass that
writes a JPEG. The system iteratively evolves continuous latents and decodes
them into pixels.

## 8. Current capability and planning matrix

| Question | Evidence-bounded answer |
|---|---|
| Can the current DeepSeek V4 API accept images? | No; official integration documentation calls V4 text-only. |
| Can a DeepSeek-branded open model understand images? | Yes: VL/VL2, OCR/OCR 2, and Janus checkpoints. |
| Can a DeepSeek-branded open model generate images? | Yes: Janus/Janus-Pro and JanusFlow. |
| Is DeepSeek-OCR a general native multimodal flagship? | No; it is a specialized OCR/compression model with some general-vision data. |
| Is OCR 2's shared omni encoder shipping in V4? | No public evidence; the paper presents it as future work. |
| Does DeepSeek disclose a public audio foundation model? | No comparable current DeepSeek audio checkpoint was identified at the cutoff. |
| Does the portfolio indicate multimodal research intent? | Yes, across VL, Janus, optical compression, causal visual encoding, and proposed shared modality queries. |

## 9. Primary sources

- DeepSeek-AI, [DeepSeek-VL](https://arxiv.org/abs/2403.05525).
- DeepSeek-AI, [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) and
  [official code](https://github.com/deepseek-ai/DeepSeek-VL2).
- DeepSeek-AI, [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
  and [official code](https://github.com/deepseek-ai/DeepSeek-OCR).
- DeepSeek-AI, [DeepSeek-OCR 2: Visual Causal Flow](https://arxiv.org/abs/2601.20552)
  and [official code](https://github.com/deepseek-ai/DeepSeek-OCR-2).
- DeepSeek-AI, [Janus](https://arxiv.org/abs/2410.13848),
  [Janus-Pro](https://arxiv.org/abs/2501.17811),
  [JanusFlow](https://arxiv.org/abs/2411.07975), and the
  [Janus repository](https://github.com/deepseek-ai/Janus).
- DeepSeek, [V4 official release](https://api-docs.deepseek.com/news/news260424/)
  and [text-only vision-proxy statement](https://api-docs.deepseek.com/quick_start/agent_integrations/github_copilot).

For the text-model pretraining, reasoning, agentic RL, and V4 lineage, continue
to the [DeepSeek Agentic RL case study](../agentic-rl/case-studies/deepseek.md).
