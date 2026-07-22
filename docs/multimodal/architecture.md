# Representation, Fusion, and the Meaning of Native Multimodality

**Verified through:** 2026-07-22.

## 1. Why pixels cannot simply be text tokens

An RGB image with height $H$ and width $W$ contains $3HW$ scalar channel
values. A 1024 by 1024 image already has more than three million channel values;
treating every channel value as an ordinary autoregressive position would make
attention and generation prohibitively expensive.

A Vision Transformer (ViT) first divides the image into patches of side length
$P$. Ignoring padding, the patch count is

$$
N_{\text{patch}}
=\frac{H}{P}\frac{W}{P}.
$$

Each flattened patch is linearly embedded into a vector. Position information
is then added so attention can distinguish spatial locations. The result is a
sequence of continuous visual vectors, not text-token identifiers.

Video adds a time axis. A naive frame-wise path creates approximately
$T N_{\text{patch}}$ vectors for $T$ sampled frames. Long-video models therefore
sample frames, use temporal patches, pool neighboring frames, or route only a
subset of spatiotemporal features. The sampling policy is part of model behavior:
a model cannot reason about a moment that preprocessing discarded.

## 2. The complete image/video understanding path

### 2.1 Media decoding and canonicalization

Before neural inference, ordinary software:

- decodes JPEG, PNG, WebP, PDF pages, or a video container;
- applies orientation and color conversion;
- selects frames or temporal clips;
- rescales, pads, tiles, or packs variable-resolution inputs; and
- records crop, page, frame, and coordinate metadata needed for grounding.

This layer is architecturally important. Tiling preserves small text but
increases token count and can break objects across boundaries. Downscaling is
cheap but may erase fine detail. Native-resolution packing minimizes artificial
resizing, yet its variable sequence lengths complicate batching and memory.

### 2.2 Visual tokenization or encoding

There are two major representation types:

1. **Continuous features.** A ViT, convolutional network, or hybrid encoder emits
   floating-point embeddings. These are preferred for understanding because
   they retain rich features without forcing every patch into a finite codebook.
2. **Discrete visual codes.** A vector-quantized tokenizer maps image regions to
   codebook identifiers. These identifiers can be predicted like word tokens and
   decoded back into pixels, making them attractive for autoregressive media
   generation.

The terms “visual token” and “image token” are overloaded. In a Kimi K2.5-style
input path they usually mean continuous embeddings occupying sequence
positions. In a Chameleon-, Emu3-, or Janus-generation path they can mean
discrete codebook indices. Only the latter can be directly sampled as categorical
next-token outputs.

### 2.3 Compression, resampling, and projection

The visual encoder width $d_v$ rarely equals the language-model embedding width
$d_l$. A projector maps

$$
P_\phi:\mathbb R^{N_v\times d_v}
\longrightarrow\mathbb R^{N'_v\times d_l}.
$$

An MLP projector may change only width, while a pixel-shuffle layer, convolution,
Perceiver resampler, or query transformer also reduces sequence length from
$N_v$ to $N'_v$. Compression saves context and attention cost, but it creates an
information bottleneck. OCR is a particularly hard test because one lost stroke
can change a character.

The projector is not merely a file-format converter. During training, it learns
which visual distinctions must survive in the language embedding space. A very
small adapter can align already strong representations; it cannot recover visual
detail that the encoder or preprocessing discarded.

### 2.4 Sequence assembly

Projected visual embeddings replace sentinel image positions or are inserted
between text segments:

```text
<bos> text tokens <image_start> v1 v2 ... vN <image_end> text tokens ...
```

They need not share token IDs with words. The model runner directly constructs
an `inputs_embeds` tensor whose rows come from the text embedding table or the
visual projector. Modality, two-dimensional, temporal, crop, and separator
embeddings can be added before the shared transformer.

With ordinary causal attention, later text tokens attend to all earlier visual
positions. Some models use cross-attention instead: the language stream remains
separate and selected layers query an external visual memory. Cross-attention
can isolate compute and cache media once, while single-stream early fusion makes
all ordinary self-attention layers available for cross-modal interaction.

### 2.5 Shared reasoning and decoding

The language backbone performs the same layer operations over projected visual
and text embeddings. A sparse MoE routes each sequence position through selected
feed-forward experts, but the visual front end remains a separate computation.
For understanding-only models, the output head predicts text vocabulary tokens.
Coordinates, boxes, points, SVG, HTML, or tool calls are usually serialized as
text or special tokens.

This explains why an image-understanding model does not automatically generate
images: its output head has learned a distribution over text vocabulary, not a
distribution over image codes or continuous image latents.

## 3. Fusion architectures

| Architecture | Media path | Cross-modal interaction | Main trade-off |
|---|---|---|---|
| caption/OCR cascade | media -> separate model -> text | text-only LLM sees a description | simple and replaceable, but loses unmentioned detail |
| encoder + projector + decoder-only LLM | ViT -> MLP/resampler -> input embeddings | ordinary self-attention over one sequence | dominant open VLM pattern; context cost grows with visual tokens |
| encoder + cross-attention LLM | media encoder stays external | selected LLM layers query visual memory | clean compute separation; extra architectural blocks |
| shared discrete-token transformer | image/video tokenizer -> code IDs | one autoregressive stream and vocabulary partitions | supports interleaved understanding/generation; long visual sequences and quantization loss |
| shared backbone with modality losses | continuous image latents plus text tokens | common transformer, next-token loss for text and diffusion/flow loss for images | high-quality generation without discretizing every output patch |
| tool-routed composition | LLM plans; specialist models execute | exchanged prompts, embeddings, or artifacts | operationally strong but not one native model |

## 4. Training determines how integrated the components become

A common progression is:

1. **Unimodal initialization.** Start from a pretrained vision encoder and text
   LLM. This provides strong local features and language knowledge.
2. **Alignment.** Freeze most components and train a projector or a small visual
   module so image representations land in a useful language space.
3. **Joint multimodal pretraining.** Unfreeze some or all components and mix
   captions, interleaved documents, OCR, grounding, image-code pairs, video, and
   text. This is where the language backbone learns that visual embeddings are
   first-class evidence.
4. **Instruction tuning.** Train response behavior and formatting. Freezing or
   excluding vision here does not erase multimodal pretraining; Kimi K2.5's
   “zero-vision SFT” is a concrete example.
5. **Multimodal reinforcement learning.** Apply verifiable rewards for OCR edit
   distance, boxes, points, segmentation, counting, visual coding, or computer
   use. This can improve both the visual policy and language reasoning path.

Late attachment normally means visual features are added after the text model is
already trained and only a limited adapter/SFT stage connects them. Early or
continual fusion means visual data is present through a substantial backbone
training phase. It does not mean the system has no visual encoder.

## 5. A rigorous native-multimodal checklist

When a vendor uses “native,” ask:

1. Were all claimed modalities included from initial pretraining, from a long
   continued-pretraining stage, or only at SFT?
2. Are media embeddings processed inside the main backbone or converted into
   text by a separate service?
3. Are the vision/audio front end, projector, and backbone jointly optimized?
4. Are unimodal and cross-modal examples mixed throughout training?
5. Is there one context that can interleave modalities and preserve references
   across turns?
6. Does the model emit media, or only text about media?
7. Are image/video/audio output decoders inside the released checkpoint or
   external tools?
8. Does the technical report disclose these facts, or does only a product page
   use the label?

The result should be a vector, not a yes/no label. For example, Kimi K2.5 is
sequence-native, deeply training-integrated through about 15T continued tokens,
and backbone-shared after MoonViT projection, while remaining architecturally a
MoonViT-3D + MLP + K2 MoE composition. It is not generation-native for images.

Kimi K3 is API-native for image/video input and officially described as having
native vision. Until its technical report and weights are released, its visual
encoder, projector, fusion depth, training mixture, and media output decoder
remain unknown.

## 6. Systems consequences

### Context and attention cost

If a prompt has $N_t$ text positions and $N_v$ visual positions, dense
self-attention sees $N_t+N_v$ positions. The quadratic attention term scales as

$$
O\left((N_t+N_v)^2d\right).
$$

Visual token compression therefore affects time to first token, memory, and how
much textual context remains. It also affects fidelity; token count is an
information budget, not just a billing unit.

### Batching and pipeline imbalance

Images have different resolutions and videos have different frame counts. If the
vision encoder lives in pipeline stage zero, that stage receives variable work
and memory while later language stages remain regular. Kimi K2.5's Decoupled
Encoder Process replicates the relatively small vision encoder, load-balances
visual forward passes, discards intermediate activations, trains the backbone,
then recomputes the vision encoder for backward. The report claims about 90% of
text-only training efficiency; this is a vendor result, not a universal number.

### Caching

Static visual embeddings can be cached, but media preprocessing, crop layout,
model revision, and projector weights become cache identity. Video frame
sampling also needs a deterministic manifest. Reusing a text prefix cache while
silently changing visual embeddings is incorrect even if the visible prompt
string is unchanged.

## 7. Evaluation failure modes

- **Text prior masquerading as vision:** the decoder guesses plausible document
  content after visual detail was lost.
- **Tiling boundary loss:** a table, object, or word is split across crops.
- **Reading-order failure:** raster order disagrees with columns, captions, or
  diagrams.
- **Temporal aliasing:** sparse frame sampling misses a short event.
- **Grounding drift:** the right concept is named but its coordinates are wrong.
- **modality conflict:** visual tuning improves perception while degrading text,
  or vice versa.
- **benchmark leakage:** synthetic captions or OCR corpora overlap evaluation
  content.
- **tool substitution:** an external OCR or browser tool is credited to the base
  model.

A serious evaluation therefore records media preprocessing, resolution, visual
token budget, frame policy, prompt, tool access, decoding settings, and whether
the tested endpoint is a fixed checkpoint or a changing product.

## Primary sources

- Moonshot AI, [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276),
  Sections 4.2–4.5.
- Moonshot AI, [Kimi K3 official technical blog](https://www.kimi.com/blog/kimi-k3)
  and [API guide](https://platform.kimi.com/docs/guide/kimi-k3-quickstart).
- DeepSeek-AI, [DeepSeek-VL2](https://arxiv.org/abs/2412.10302).
- DeepSeek-AI, [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) and
  [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552).
- Meta, [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818).
- Meta, [Transfusion: Predict the Next Token and Diffuse Images with One
  Multi-Modal Model](https://openreview.net/forum?id=SI2hI0frk6).
- BAAI, [Emu3: Next-Token Prediction Is All You Need](https://arxiv.org/abs/2409.18869).
