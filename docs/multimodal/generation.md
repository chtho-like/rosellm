# How LLM-Centered Systems Generate Images and Other Media

**Verified through:** 2026-07-22.

## 1. Is there an LLM that directly generates an image?

Yes, if “directly” means that a language-style transformer itself predicts the
image representation. Chameleon, Emu3, and DeepSeek Janus are public examples
of autoregressive transformers that predict discrete visual codes. Transfusion
and DeepSeek JanusFlow show a second route in which a shared language-centered
transformer participates in diffusion or rectified-flow prediction over
continuous image latents.

No, if “directly” means that an ordinary text-only LLM emits raw RGB pixels or a
JPEG byte stream with no image representation layer. Practical systems still
need at least one of:

- a visual tokenizer and image detokenizer;
- a Variational Autoencoder (VAE) and diffusion/flow decoder;
- a specialist image model invoked as a tool; or
- a renderer for code such as SVG, HTML, or a 3D scene description.

The meaningful question is therefore not “is there any image module?” but
“which part of image generation is learned inside the shared backbone, and
which representation/decoder turns its output into pixels?”

## 2. Scheme A: LLM planner plus a separate image generator

```text
user request
  -> LLM interprets intent, expands prompt, plans edits and safety constraints
  -> separate diffusion/flow image model
  -> image
  -> optional vision model critiques image
  -> revised prompt or edit mask -> regenerate
```

This is common in products because each component can be upgraded, scaled, and
moderated independently. The LLM contributes instruction following, world
knowledge, composition planning, text rendering specifications, and iterative
critique. The image model contributes spatial synthesis.

Advantages:

- specialist image quality and mature editing controls;
- independent deployment and routing;
- the same LLM can select among multiple generators; and
- failures can be isolated to planning, rendering, or critique.

Limitations:

- the generator may ignore details in the expanded prompt;
- semantic state can be lost at the text boundary;
- identity and geometry can drift between iterations; and
- a product's smooth single endpoint can hide the fact that two or more models
  are involved.

This is product-level multimodality, not necessarily a native generative
backbone.

## 3. Scheme B: autoregressive discrete visual tokens

### 3.1 Visual tokenizer

An image tokenizer encodes image $x$ into a lower-resolution latent grid and
quantizes each latent to one entry in a learned codebook:

$$
z=E(x),
\qquad c_i=\operatorname*{argmin}_{k}\Vert z_i-e_k\Vert_2.
$$

The code indices $c_1,\ldots,c_N$ are discrete, like token identifiers. A
detokenizer $D$ reconstructs the image:

$$
\hat x=D(e_{c_1},\ldots,e_{c_N}).
$$

The tokenizer is lossy: its codebook size, spatial downsampling, and perceptual
loss determine the maximum detail that later generation can recover.

### 3.2 One next-token model

Text tokens and image codes can be placed in one vocabulary/sequence. The
transformer learns

$$
p(s_1,\ldots,s_L)
=\prod_{i=1}^{L}p(s_i\mid s_{<i}),
$$

where a position can be a word, control marker, image code, video code, or action
token. Text-to-image generation starts with text and an image-start marker, then
samples visual codes until the image-end marker.

This provides an unusually literal form of “LLM generates the image”: the same
causal-transformer mechanism that predicts words predicts image-code IDs. It
still needs a visual tokenizer/detokenizer and a spatial serialization order.

### 3.3 Public examples

- [Chameleon](https://arxiv.org/abs/2405.09818) trains an early-fusion
  token-based model over interleaved text and images, supporting understanding,
  generation, and mixed-modal documents.
- [Emu3](https://arxiv.org/abs/2409.18869) tokenizes text, images, and video into
  discrete sequences and trains one transformer with next-token prediction,
  including text-to-image, text-to-video, perception, and future prediction.
- [DeepSeek Janus](https://arxiv.org/abs/2410.13848) uses one autoregressive
  transformer but deliberately separates the continuous understanding encoder
  from the discrete image-generation tokenizer.

### 3.4 Strengths and bottlenecks

Strengths:

- one causal objective and sampling interface;
- arbitrary interleaving of text and media;
- natural in-context continuation and editing formulations; and
- scaling machinery resembles LLM training.

Bottlenecks:

- image sequences are long, making left-to-right decoding expensive;
- raster order is not a natural semantic order;
- local errors propagate to later codes;
- categorical codebooks impose quantization artifacts;
- text and image token frequencies/entropy differ dramatically; and
- high-resolution video multiplies spatial positions by time.

Multiscale tokenizers, blockwise generation, speculative decoding, parallel
token prediction, and stronger codebooks mitigate these costs but make “just
next-token prediction” operationally more complex than the slogan suggests.

## 4. Scheme C: specialist diffusion or flow model conditioned by language

A conventional latent diffusion system encodes an image into a continuous
latent $z_0$, adds noise to obtain $z_t$, and trains a denoiser to predict noise,
velocity, or a related target conditioned on text $y$:

$$
\mathcal L_{\text{diff}}
=\mathbb E_{z_0,\epsilon,t}
\left[\Vert\epsilon-f_\theta(z_t,t,y)\Vert_2^2\right].
$$

A Diffusion Transformer (DiT) represents noisy latent patches as transformer
positions, but it is not automatically an LLM: its inputs, timestep
conditioning, objective, and iterative sampling path are image-generative.

The text conditioner may be a CLIP/T5-style encoder, an LLM representation, or
an external prompt planner. Image generation requires many denoising steps or a
distilled/few-step solver, followed by VAE decoding.

Advantages:

- continuous latents avoid a hard visual codebook;
- parallel denoising updates all patches per step;
- excellent image fidelity and editing/control ecosystems; and
- spatial inductive biases are easier to preserve.

Limitations:

- text generation and image generation remain different objectives and sampling
  procedures;
- iterative denoising costs multiple model evaluations; and
- a separate text encoder/generator can create semantic misalignment.

## 5. Scheme D: one transformer, next-token text plus diffusion images

[Transfusion](https://openreview.net/forum?id=SI2hI0frk6) demonstrates a shared
transformer trained with different losses by modality:

$$
\mathcal L
=\mathcal L_{\text{text-CE}}
+\lambda\mathcal L_{\text{image-diffusion}}.
$$

Text positions use causal next-token cross entropy. Image patches remain
continuous and use a diffusion objective, with attention masks arranged so
image spans can interact appropriately while the text stream remains causal.

This is generation-native at the shared-backbone level without pretending that
images are ordinary word tokens. It trades architectural purity for a better
match between each modality and its prediction problem.

The important insight is that “one model” need not mean “one loss” or “one
sampling algorithm.” Shared parameters can support categorical text decoding
and continuous image denoising through modality-specific heads and schedules.

## 6. Scheme E: autoregressive language plus rectified-flow images

Rectified flow learns a vector field that transports a simple noise sample
toward the data distribution. In a simplified straight interpolation,

$$
z_t=(1-t)z_0+t z_1,
$$

and the model learns a velocity field for integrating from noise to a clean
latent.

[DeepSeek JanusFlow](https://arxiv.org/abs/2411.07975) integrates rectified-flow
image generation into a language-model framework while retaining
autoregressive text generation. It keeps understanding and generation encoders
decoupled and aligns their representations during unified training.

Its generation path is approximately:

```text
text condition + noisy continuous image latent + time
  -> shared language-centered transformer / flow prediction
  -> repeat ODE integration steps
  -> clean latent
  -> VAE decoder
  -> pixels
```

The LLM framework performs material generative computation, yet the flow solver
and image decoder remain indispensable.

## 7. Scheme F: generate a symbolic visual program

An LLM can generate SVG, HTML/CSS, canvas commands, Blender/Python, CAD
operations, shader code, or a scene graph. A deterministic renderer then creates
the image or animation.

This path is particularly strong when exact text, layout, geometry, or
editability matters:

```text
instruction -> LLM -> structured visual program -> renderer -> image/video
```

It is not pixel-native generation, but it may be more controllable and auditable
than a diffusion model. Kimi's screenshot-to-code and “vision in the loop” agent
examples fit this broader category when the model writes code, renders it,
inspects the screenshot, and iterates.

## 8. How understanding and generation can share a model

Understanding wants representations invariant to irrelevant pixel changes and
rich in semantics. Generation wants representations that preserve exact color,
texture, position, and reconstructable detail. Forcing one encoder to do both
can produce gradient conflict.

Common resolutions are:

- **separate encoders, shared transformer:** Janus;
- **continuous understanding encoder plus VQ generation tokenizer:** Janus and
  related unified autoregressive systems;
- **separate modality losses/heads on a shared transformer:** Transfusion;
- **representation alignment between distinct encoders:** JanusFlow;
- **fully discrete shared stream:** Chameleon and Emu3; and
- **specialists connected by an LLM agent:** production cascades.

Thus the most capable “unified” model may intentionally contain modality-specific
components. Uniformity of parameters is not the objective; coherent information
exchange and trainable end-to-end behavior are.

## 9. Image editing and reference consistency

Text-to-image is only one operation. Editing requires the system to preserve
some information from source image $x$ while changing attributes specified by
instruction $y$.

Typical mechanisms include:

- encode the source to visual codes and autoregressively continue/replace a
  masked span;
- add noise only to a masked latent region and condition denoising on the
  unmasked source;
- cross-attend to reference-image embeddings;
- condition on structural controls such as depth, edges, pose, segmentation,
  or layout; and
- use a vision-language model to critique identity, text, and geometry across
  iterative generations.

A chat interface that remembers an image does not prove pixel-level identity
preservation. Evaluation must measure source fidelity, requested change,
unrequested change, text rendering, and multi-turn consistency separately.

## 10. Video extends every bottleneck

Discrete autoregression can serialize spatiotemporal codes, as Emu3 does, but
token count and sequential latency grow rapidly. Diffusion/flow video models
update spatiotemporal latent blocks in parallel per denoising step but require
large memory and careful temporal attention.

Video systems commonly add:

- causal or bidirectional temporal attention;
- three-dimensional patching;
- frame-rate and keyframe conditioning;
- temporal compression/tokenizers;
- chunked generation with overlap and state reuse; and
- separate audio, motion, and camera-control streams.

Kimi K2.5's four-frame MoonViT pooling is an **input compression technique for
video understanding**, not a video-generation decoder.

## 11. Speech makes the same distinction visible

Kimi-Audio shows a common speech pattern:

```text
waveform -> acoustic/semantic tokenizer -> shared Audio LLM
shared Audio LLM -> discrete semantic audio codes
  -> flow-matching acoustic detokenizer -> BigVGAN vocoder -> waveform
```

The transformer can natively predict speech-semantic codes while a specialized
decoder and vocoder synthesize high-bandwidth audio. This mirrors visual-token
or latent image generation and demonstrates why “native output” never means
“no modality decoder.”

## 12. Comparison matrix

| Scheme | Shared reasoning/generation backbone | Media representation predicted | Media decoder | Typical sampling |
|---|---|---|---|---|
| LLM + external generator | no | prompt/control only | separate diffusion/flow model | text once, image iteratively |
| Chameleon/Emu3-style | yes | discrete image/video codes | VQ-style detokenizer | autoregressive code by code |
| Janus-style | yes | discrete image codes; separate understanding features | VQ decoder | autoregressive code by code |
| Transfusion-style | yes | continuous noisy image patches | VAE decoder | iterative diffusion |
| JanusFlow-style | yes | continuous flow velocity/latent path | VAE decoder | iterative ODE/flow |
| symbolic program | LLM produces program | SVG/HTML/CAD/scene tokens | deterministic renderer | autoregressive program then render |

## 13. What to ask before accepting “native image generation”

1. Does the released checkpoint contain the image-token or latent prediction
   head?
2. Is image generation trained jointly with language, or invoked as a tool?
3. What image tokenizer/VAE and downsampling ratio are used?
4. Is sampling autoregressive, diffusion, flow, masked parallel prediction, or a
   hybrid?
5. Does one transformer handle both modalities, and which parameters are
   shared?
6. Can the model emit interleaved text and images, or only one final image?
7. Are editing/reference images directly encoded or reduced to text?
8. What resolution, token/step count, latency, and memory are required?
9. Are typography, identity, and spatial-relation benchmarks independently
   reproduced?
10. Is the product endpoint a fixed model or an orchestrated stack?

Closed-model marketing may answer only the capability questions. Without a
technical report or artifacts, it is valid to say the system behaves as a
native generator at the interface while its internal scheme remains unknown.

## Primary sources

- Meta, [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818).
- BAAI, [Emu3: Next-Token Prediction Is All You Need](https://arxiv.org/abs/2409.18869)
  and [official repository](https://github.com/baaivision/Emu3).
- Meta, [Transfusion: Predict the Next Token and Diffuse Images with One
  Multi-Modal Model](https://openreview.net/forum?id=SI2hI0frk6).
- DeepSeek-AI, [Janus](https://arxiv.org/abs/2410.13848),
  [Janus-Pro](https://arxiv.org/abs/2501.17811), and
  [JanusFlow](https://arxiv.org/abs/2411.07975).
- Moonshot AI, [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio).
