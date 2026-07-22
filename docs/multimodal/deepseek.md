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

For the 7B-family model, the exact visual path is:

```text
1024x1024 image
  -> SAM-B / ViTDet -> 64x64x256 feature map
  -> interpolate to 96x96, then two stride-2 convolutions
  -> 24x24x1024 = 576 high-resolution positions

384x384 image
  -> SigLIP-L -> 24x24x1024 = 576 semantic positions

position-wise split projections -> concatenate channels
  -> 576 positions x 2,048 features
  -> GELU/MLP -> DeepSeek LLM embedding width
```

This is channel fusion, not sequence concatenation: corresponding SAM and
SigLIP positions are fused into 576 combined positions instead of making a
1,152-position sequence. The released
[hybrid tower](https://github.com/deepseek-ai/DeepSeek-VL/blob/681bffb4519856ad27cc17531aacde31ddf6f1a7/deepseek_vl/models/clip_encoder.py#L126-L203)
executes the two towers separately, while the
[split projector](https://github.com/deepseek-ai/DeepSeek-VL/blob/681bffb4519856ad27cc17531aacde31ddf6f1a7/deepseek_vl/models/projector.py#L47-L86)
projects each stream to half the language width before concatenation. The report
lists the 1B-family variant with SigLIP alone; the dual-tower description applies
to the larger model.

Training also matters to what “ViT plus LLM” means:

1. freeze both visual towers and the LLM, and warm up only the adaptor on about
   1.25M caption pairs plus 2.5M rendered-document OCR pairs;
2. keep the visual towers frozen but jointly train the adaptor and LLM on a
   language-and-multimodal mixture, because multimodal-only continuation caused
   language forgetting in the authors' ablation; and
3. supervised-fine-tune SigLIP, the adaptor, and the LLM, while SAM-B remains
   frozen for memory reasons and loss is applied only to answer/special text
   tokens.

The design is input-side visual understanding. Continuous image embeddings
replace image placeholder positions inside the decoder-only language sequence;
ordinary causal self-attention then mixes them with text. It does not contain
Janus's image generation path. Its fixed-resolution dual encoder also requires
two visual preprocessing streams, which motivated VL2's dynamic tiling redesign.

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

### 3.2 Pixel unshuffle and sequence layout

The paper calls the operation pixel shuffle, but the released implementation is
the downsampling direction normally called **pixel unshuffle** or
**space-to-depth**: it pads the odd 27 by 27 grid to 28 by 28, unfolds each 2 by
2 neighborhood into the channel dimension, and applies an MLP. It therefore
reduces each tile from 729 positions to

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

The
[released adaptor](https://github.com/deepseek-ai/DeepSeek-VL2/blob/ef9f91e2b6426536b83294c11742c27be66361b1/deepseek_vl2/models/modeling_deepseek_vl_v2.py#L56-L110)
shows the 2 by 2 unfold explicitly. The
[sequence assembly](https://github.com/deepseek-ai/DeepSeek-VL2/blob/ef9f91e2b6426536b83294c11742c27be66361b1/deepseek_vl2/models/modeling_deepseek_vl_v2.py#L409-L454)
reconstructs the local tile grid before adding learned newline and view-separator
embeddings. This deterministic reshape does not choose salient patches: before
the MLP, all four neighboring feature vectors remain present as channels. Any
lossiness comes from the subsequent learned projection and finite language
context, not from averaging four patches.

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

VL2 still has important scope boundaries. It serializes grounding boxes as text
coordinates normalized into integer bins from 0 to 999; it does not produce pixel masks or
images through a media decoder. Its report is image-centric and does not disclose
a shared image/video encoder comparable to later Qwen or Kimi systems. Mixture of
Experts (MoE) sparsifies feed-forward computation and MLA reduces key/value-cache
width, but neither mechanism makes the initial visual-token prefill free.

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
not an inference from a diagram: the
[two stride-2 convolutions](https://github.com/deepseek-ai/DeepSeek-OCR/blob/09eaf526153e7a01ed16c9dea8c96282aaea29c0/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py#L166-L181)
map 256 to 512 to 1,024 channels, and the
[released model assembly](https://github.com/deepseek-ai/DeepSeek-OCR/blob/09eaf526153e7a01ed16c9dea8c96282aaea29c0/DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepseek_ocr.py#L281-L298)
instantiates the linear 2,048-to-1,280 projector.

Here “visual token” means a **continuous embedding occupying one decoder input
position**. DeepSeek-OCR does not quantize the page into image-code identifiers
as Janus does for generation, and it cannot losslessly recover arbitrary bytes
from a page. The decoder generates ordinary vocabulary tokens conditioned on a
lossy continuous page representation and its learned language prior.

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
  cannot see any query positions;
- query $q_i$ sees all visual positions and only queries $q_{\leq i}$; and
- only the final query outputs are projected and sent to the DeepSeek decoder.

The queries do not perform a hard permutation or emit indices saying “read patch
17 next.” They learn an ordered sequence of continuous summaries. Calling this
“dynamic semantic reordering” is the authors' interpretation of the causal
query mechanism, not a literal sorting algorithm.

The artifact makes this distinction especially clear. It concatenates visual
features and learned queries, assigns the former non-causal token type 0 and the
latter causal token type 1, then slices out only the query half. See the
[custom attention mask](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/2f3699ebbb96fa8af32212e8c170f2cc28730fad/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepencoderv2/qwen2_d2e.py#L136-L172)
and
[query assembly](https://github.com/deepseek-ai/DeepSeek-OCR-2/blob/2f3699ebbb96fa8af32212e8c170f2cc28730fad/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepencoderv2/qwen2_d2e.py#L248-L284).
An attention mask changes information flow but does not automatically make the
matrix multiplication sparse; the released dense masked-attention path can still
compute over the full $2m$-position encoder sequence.

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
pixels. The released
[multimodal model assembly](https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/janus/models/modeling_vlm.py#L190-L263)
contains separate understanding vision/alignment modules, generation vision and
alignment modules, a generation embedding table and head, and one shared causal
language model.

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

## 8. Industry comparison: where the designs actually differ

The phrase “ViT plus LLM” is too coarse to distinguish modern systems. Most
open vision-language models contain a visual encoder, a bridge, and a language
backbone. The material differences are the resolution policy, the information
bottleneck, where modalities first share trainable layers, which losses update
those layers, and whether the output space contains only text or also media.

### 8.1 General visual understanding front ends

| System | Image/video path | Visual-budget mechanism | Bridge and language core | Primary architectural trade-off |
|---|---|---|---|---|
| [LLaVA](https://arxiv.org/abs/2304.08485) | fixed-resolution CLIP ViT | one fixed patch grid | linear/MLP connector into a dense LLM | simplest strong baseline; limited native resolution and connector capacity |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | frozen image encoder | 32 learned Querying Transformer (Q-Former) queries in the reported design | Q-Former bridges to a frozen LLM | few trainable parameters and fixed budget, but a tight learned bottleneck |
| DeepSeek-VL | fixed 384 SigLIP plus 1024 SAM-B | fuse two 576-position grids by channels | hybrid MLP into DeepSeek-LLM | semantic/detail complementarity at the cost of two preprocessors and fixed resolution |
| DeepSeek-VL2 | global thumbnail plus up to nine 384 local tiles | 729 to 196 positions per tile by space-to-depth | two-layer MLP into DeepSeekMoE with MLA | explicit global/local geometry and efficient sparse FFNs; tile boundaries and a bounded image policy |
| [InternVL 2.5](https://arxiv.org/abs/2412.05271) | global thumbnail plus dynamic 448 tiles; fixed-size frames for video | 1,024 to 256 positions per tile by pixel unshuffle | two-layer MLP into several dense LLM families | structurally close to VL2, with much larger optional InternViT encoders and broader video training |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | variable-resolution image/video patches | visual position count varies with input resolution | projector plus a language model using Multimodal Rotary Position Embedding (M-RoPE) | fewer artificial tile seams and explicit temporal/2D position axes, but high-resolution inputs can create long visual sequences |
| [Kimi K2.5](https://arxiv.org/abs/2602.02276) | MoonViT-3D uses Native-resolution Vision Transformer (NaViT) packing; groups up to four frames | variable patch packing plus temporal pooling | MLP into the K2 MoE backbone, followed by extensive joint text/vision training | shared image/video representation and deep training integration, with substantial continued-pretraining cost |

DeepSeek-VL2 is architecturally closest to InternVL 2.5, not to a radically new
end-to-end pixel transformer. Both split large images into a thumbnail and local
tiles, deterministically move 2 by 2 spatial neighborhoods into channels, use an
MLP connector, and let a decoder-only LLM attend over the resulting continuous
embeddings. The important differences are 384-pixel SigLIP versus 448-pixel
InternViT tiles, VL2's explicit learned row/view separators, InternVL's optional
300M/6B vision encoders, and DeepSeek's MoE/MLA language core.

Qwen2-VL and Kimi K2.5 instead emphasize variable-resolution patch sequences.
They avoid reconstructing a large image as a fixed tile montage, although
padding, packing, token caps, and rescaling still exist in implementation. Their
visual activation and language-prefill costs grow more directly with retained
patch count. Kimi additionally shares its encoder across images and four-frame
spatiotemporal chunks, while the VL2 report does not establish an equivalent
video architecture.

This comparison also explains why “uses ViT plus LLM” and “jointly trained
multimodal model” can both be true. Kimi K2.5, DeepSeek-VL2, InternVL, and Qwen2-VL
all retain modality-specific front ends. The stronger integration claim comes
from continued joint pretraining and shared reasoning layers, not from deleting
the vision encoder.

### 8.2 Five different meanings of visual-token compression

| Mechanism | Representative systems | Position mapping | Learned content selection? | What it preserves or risks |
|---|---|---:|---|---|
| space-to-depth / pixel unshuffle | VL2, InternVL | approximately $HW\rightarrow HW/4$ | no | preserves each local feature as a channel before projection; introduces a learned width bottleneck afterward |
| fixed learned query set | BLIP-2 Q-Former | hundreds of patches $\rightarrow 32$ queries | yes | constant LLM cost; may omit dense text or small objects when the query budget is too small |
| local attention, convolutional compression, then global attention | DeepSeek-OCR | $HW\rightarrow HW/16$ before global attention | convolution is learned but locally structured | high native resolution with bounded global activation; page detail is compressed before whole-page reasoning |
| equal-count causal query serialization | DeepSeek-OCR 2 | $m$ compressed features $\rightarrow m$ ordered query outputs | yes | adds semantic ordering without reducing the already compressed count; the encoder processes a $2m$-position concatenation |
| variable patch sequence with packing/pooling | Qwen2-VL, Kimi K2.5 | retained positions scale with pixels/frames | mostly through encoder attention and pooling | spends more positions when detail is available; worst-case activation and context use require hard caps |

Token count alone is therefore not a fair information-rate metric. A VL2
position contains four neighboring SigLIP vectors before projection; a BLIP-2
position is a learned global query; an OCR position is a continuous page summary;
and a Janus generation token is a discrete codebook identifier. Equal counts do
not imply equal pixels, information, compute, or recoverability.

For a decoder-only VLM with $T$ text positions and $N_v$ visual positions, a
dense-attention prefill has the rough scaling

$$
C_{\mathrm{attn}}\propto(T+N_v)^2.
$$

This is why connector-side compression matters even when the connector has few
parameters. MoE primarily reduces the active feed-forward work per position;
it does not sparsify this attention matrix. MLA compresses cached key/value
representations and improves decode-time bandwidth, but it does not erase the
vision encoder or the initial prefill.

### 8.3 OCR now spans several different product architectures

| Family | Representative implementation | Processing structure | Strength | Main failure boundary |
|---|---|---|---|---|
| classical pipeline | detector/layout model + line recognizer + reading-order/postprocessor | multiple specialized stages with explicit boxes and confidence | modular, debuggable, replaceable, often strong for exact transcription | detection and ordering errors cascade; integration is operationally complex |
| end-to-end document model | [Donut](https://arxiv.org/abs/2111.15664), [GOT-OCR2.0](https://arxiv.org/abs/2409.01704) | page encoder directly conditions an autoregressive structured-text decoder | one learned page-to-output objective; no external OCR engine | implicit alignment and language-prior hallucination; task serialization can be brittle |
| general-purpose VLM | VL2, Qwen-VL, InternVL | general visual encoder and large language reasoner prompted to transcribe/parse | OCR can be combined with QA and reasoning in one conversation | often thousands of visual positions and unnecessary general-model cost for batch parsing |
| coarse-to-fine parser | [MinerU2.5](https://arxiv.org/abs/2509.22186) | low-resolution global layout pass, then native-resolution recognition of selected crops | spends high-resolution compute only where layout says it is needed | multi-pass orchestration; global layout mistakes can suppress local recovery |
| compact dynamic-resolution parser | [PaddleOCR-VL](https://arxiv.org/abs/2510.14528) | NaViT-style encoder plus an ERNIE-4.5-0.3B language model in a 0.9B system | 109-language target and compact page/element parsing | disclosed benchmark and deployment claims remain distribution-dependent |
| optical-compression VLM | DeepSeek-OCR/OCR 2 | high-resolution tokenizer, aggressive position compression, small-active MoE text decoder | unusually explicit token/fidelity study and high-throughput page generation | lossy visual memory, autoregressive output cost, and dependence on learned linguistic/layout priors |

Donut's historical contribution was to replace a separate OCR engine with a
Swin-style page encoder and autoregressive text decoder. GOT-OCR2.0 broadened
that idea into a 580M end-to-end model for text, formulas, tables, charts, music,
geometry, and prompted regions, with a high-compression encoder and long-context
decoder. DeepSeek-OCR's distinctive contribution is not merely “OCR without an
OCR engine”; it makes the visual-position budget itself the experiment and puts
the 16-fold compressor before dense global visual attention.

MinerU2.5 makes a different systems choice. It does not ask one page embedding
to retain every glyph. It first discovers layout cheaply, then revisits selected
regions at native resolution. This trades a second pass and explicit crop
orchestration for content-dependent compute. PaddleOCR-VL instead pursues a
small, multilingual, dynamically sized end-to-end model. Neither is simply a
larger or smaller DeepSeek-OCR; the compute graph is different.

OCR 2's author-run OmniDocBench v1.5 comparison reports an overall score of
91.09 versus 87.36 for the nine-crop OCR 1 baseline, with reading-order edit
distance falling from 0.085 to 0.057 and similar maximum visual budgets (1,120
versus 1,156). This is evidence that the causal-query encoder helped under that
protocol, not proof that causal serialization universally beats every OCR
architecture. The same table reports PaddleOCR-VL at 92.86 and MinerU2.5 at
90.67, but most non-DeepSeek rows were imported from the benchmark repository;
model revisions and inference policies still need parity before causal claims.

An independent 2026
[semantic-corruption study](https://arxiv.org/abs/2601.03714) reports that
DeepSeek-OCR's accuracy falls sharply when linguistic coherence is destroyed and
that fewer visual positions increase reliance on language priors. This does not
invalidate the vendor's ordinary-document results; it identifies a fundamental
trade-off. A highly compressed generative OCR model can reconstruct plausible
language while being less robust than a character-centric recognizer on random
strings, identifiers, adversarial substitutions, or unfamiliar notation.

For high-stakes transcription, a robust production design can therefore retain
the original image, run an explicit recognizer with coordinates/confidence, run
an end-to-end parser for layout and structure, and reconcile disagreements
instead of treating fluent Markdown as proof of exactness.

### 8.4 OCR 2 causal queries versus BLIP-2 Q-Former queries

Both designs use learned queries, but their information graphs are different:

- BLIP-2's Q-Former uses a small fixed query set that cross-attends to a frozen
  visual encoder. Query self-attention is bidirectional, so the result is a
  compact semantic set rather than an explicitly ordered reading sequence.
- OCR 2 prepends $m$ visual embeddings to $m$ learned queries in a decoder-only
  transformer. Every query sees all visual positions, but query $q_i$ sees only
  $q_1$ through $q_i$. The output therefore has a trainable causal order and
  preserves one query slot per compressed spatial position.
- Visual positions cannot see any position in the query suffix in OCR 2's mask. They
  remain a bidirectionally encoded prefix; causal aggregation happens only in
  the query suffix.
- OCR 2's mechanism is not chiefly a token-count compressor. SAM plus the two
  stride-2 convolutions already performed the 16-fold spatial compression. The
  causal-query stack is an ordering and knowledge-compression stage.

This is also why “Visual Causal Flow” must not be confused with optical flow,
diffusion flow, or an emitted scan-path permutation. It is a masked-attention
construction whose learned query outputs are interpreted as a semantic reading
flow.

### 8.5 Understanding plus image generation

| Architecture | Visual representation | Shared component | Image objective and decoder | Unification trade-off |
|---|---|---|---|---|
| Janus / Janus-Pro | continuous semantic features for understanding; separate discrete VQ codes for generation | one autoregressive language transformer | next-code cross-entropy, then VQ decoder | avoids forcing semantic and reconstructive features to be identical, but retains two visual paths |
| [Chameleon](https://arxiv.org/abs/2405.09818) / [Emu3](https://arxiv.org/abs/2409.18869) | text and media represented as discrete token IDs | one early-fusion autoregressive transformer | next-token prediction, then media tokenizer decoder | conceptually uniform sequence modeling; image quantization and long code sequences constrain fidelity/compute |
| [Transfusion](https://arxiv.org/abs/2408.11039) | discrete text plus continuous image patches | one transformer over mixed sequences | language loss plus diffusion loss, then image decoder | avoids discrete image-code prediction but retains iterative denoising and modality-specific input/output layers |
| JanusFlow | continuous understanding features and continuous generation latents | language transformer participates in both paths | rectified-flow velocity objective, numerical integration, then VAE-like decoder | continuous generation without VQ code classification, but still has aligners, a latent decoder, and repeated solver steps |
| tool cascade | text LLM plus a separate diffusion/DiT service | no required shared model weights | tool call invokes external image generator | easiest production composition; multimodal only at the product/orchestration layer |

Janus is therefore more unified than an LLM calling Stable Diffusion, but less
representationally unified than Chameleon or Emu3. Its decoupling is deliberate:
semantic invariance helps recognition, while pixel reconstruction needs texture,
color, and local detail that semantic encoders often discard. JanusFlow changes
the generation objective, not the fact that pixels are produced through a latent
decoder. None of these models literally emits RGB or JPEG bytes from ordinary
language logits without a media representation and decoder.

### 8.6 Practical model-selection consequences

| Requirement | Natural starting family | Reason |
|---|---|---|
| broad image chat, grounding, chart QA | VL2, Qwen-VL, InternVL, Kimi | general visual pretraining and a language reasoning core |
| long video or fast motion | Kimi MoonViT-3D or a modern Qwen/video-trained model | explicit temporal representation; VL2 and OCR are not equivalent substitutes |
| high-throughput PDF-to-Markdown | OCR 2, PaddleOCR-VL, MinerU2.5 | document-specialized data and bounded page compute |
| exact serial numbers, legal identifiers, or compliance transcription | classical recognizer plus cross-checking parser | explicit character/coordinate confidence and lower tolerance for language-prior completion |
| one checkpoint that understands and generates images | Janus/JanusFlow, Chameleon/Emu3-like, or Transfusion-like systems | the output space and training objective contain an image-generation path |
| strongest dedicated image aesthetics | specialized diffusion/flow model behind a tool | a general LLM-centered checkpoint need not be the best image generator |

### 8.7 What the portfolio implies about DeepSeek's strategy

The disclosed and artifact-confirmed facts support a careful interpretation:

1. DeepSeek-VL and VL2 pursue general understanding through increasingly
   efficient continuous visual front ends and DeepSeek language backbones.
2. Janus decouples semantic and reconstructive representations rather than
   forcing one visual tokenizer to optimize incompatible objectives.
3. OCR 1 specializes in moving high-resolution local perception before a severe
   bottleneck and global reasoning; OCR 2 then replaces the CLIP-style global
   encoder with an LLM-style causal-query encoder.
4. OCR 2 discloses that OCR services both online image/document reading for
   DeepSeek LLMs and offline pretraining-data production. That is direct evidence
   for a production cascade, not evidence that the flagship LLM itself consumes
   pixels.
5. The authors propose a future shared omni encoder with modality-specific
   queries. No released flagship, audio path, integration date, or training
   recipe proves that the proposal has become a production base model.

It is reasonable to infer a design philosophy of **specialized modality front
ends, aggressive token economy, reusable DeepSeek language/MoE components, and
separate representations when objectives conflict**. It is not reasonable to
infer that VL2, OCR 2, Janus, and V4 are already one hidden omni checkpoint. The
public evidence shows a portfolio of related experiments and services, not one
fully consolidated model lineage.

## 9. Current capability and planning matrix

| Question | Evidence-bounded answer |
|---|---|
| Can the current DeepSeek V4 API accept images? | No; official integration documentation calls V4 text-only. |
| Can a DeepSeek-branded open model understand images? | Yes: VL/VL2, OCR/OCR 2, and Janus checkpoints. |
| Can a DeepSeek-branded open model generate images? | Yes: Janus/Janus-Pro and JanusFlow. |
| Is DeepSeek-OCR a general native multimodal flagship? | No; it is a specialized OCR/compression model with some general-vision data. |
| Is OCR 2's shared omni encoder shipping in V4? | No public evidence; the paper presents it as future work. |
| Does DeepSeek disclose a public audio foundation model? | No comparable current DeepSeek audio checkpoint was identified at the cutoff. |
| Does the portfolio indicate multimodal research intent? | Yes, across VL, Janus, optical compression, causal visual encoding, and proposed shared modality queries. |

## 10. Primary sources

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
- Liu et al., [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485),
  and Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597).
- Qwen Team, [Qwen2-VL](https://arxiv.org/abs/2409.12191); OpenGVLab,
  [InternVL 2.5](https://arxiv.org/abs/2412.05271); and Kimi Team,
  [Kimi K2.5](https://arxiv.org/abs/2602.02276).
- Kim et al., [Donut](https://arxiv.org/abs/2111.15664); Wei et al.,
  [GOT-OCR2.0](https://arxiv.org/abs/2409.01704); Cui et al.,
  [PaddleOCR-VL](https://arxiv.org/abs/2510.14528); and Niu et al.,
  [MinerU2.5](https://arxiv.org/abs/2509.22186).
- Meta, [Chameleon](https://arxiv.org/abs/2405.09818); Wang et al.,
  [Emu3](https://arxiv.org/abs/2409.18869); and Zhou et al.,
  [Transfusion](https://arxiv.org/abs/2408.11039).
- Liang et al.,
  [Visual Merit or Linguistic Crutch?](https://arxiv.org/abs/2601.03714), an
  independent DeepSeek-OCR robustness study rather than a DeepSeek disclosure.

For the text-model pretraining, reasoning, agentic RL, and V4 lineage, continue
to the [DeepSeek Agentic RL case study](../agentic-rl/case-studies/deepseek.md).
