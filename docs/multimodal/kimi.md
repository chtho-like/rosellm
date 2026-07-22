# Kimi Model Lineage, MoonViT, and “Native Vision”

**Verified through:** 2026-07-22. Architectural claims use Moonshot's technical
reports, released repositories/model cards, and current API documentation.
Kimi K3's technical report and full weights were not yet public at this cutoff.

## 1. Direct conclusions

1. **Kimi K2 itself is a text-first LLM.** It is the 1.04T-parameter sparse MoE
   backbone and agentic-training foundation released in July 2025.
2. **Kimi K2.5 is structurally MoonViT-3D + MLP projector + K2 MoE.** Saying
   “ViT + LLM” is correct but incomplete: Moonshot continued training the
   backbone on roughly 15T mixed tokens, jointly optimized text and vision, and
   later applied joint text/vision reinforcement learning.
3. **A separate vision encoder does not disqualify a model from being natively
   multimodal.** “Native” normally describes training and first-class model
   integration, not the absence of modality-specific front ends.
4. **K2.5/K2.6/K3 visual understanding is not native image generation.** Their
   disclosed output path is text, code, structured content, and tool calls.
   Moonshot can expose product image tools without proving that the K2/K3
   checkpoint emits image latents itself.
5. **K3's “native vision” is official but under-specified.** The current sources
   disclose image/video API input and a new KDA/AttnRes/LatentMoE language
   architecture, but do not yet disclose K3's visual encoder, projector, visual
   token budget, or multimodal training stages. K2.5 internals cannot be copied
   forward as facts.

## 2. Why so many models begin with K2

`K2` names a model generation and lineage. The characters after it identify a
checkpoint, capability specialization, or major post-training/continued-training
revision. They are not parameter counts, strict semantic versioning, or a claim
that every internal tensor stayed identical.

| Public name | Relationship to the lineage | Main disclosed change | Architecture disclosure |
|---|---|---|---|
| Kimi K2 Base/Instruct | foundation of the generation | text MoE pretraining and agentic post-training | detailed technical report |
| K2-Instruct-0905 | dated K2 instruct refresh | 256K context and behavior updates | model card; limited training detail |
| K2 Thinking | K2 reasoning/agent checkpoint | interleaved reasoning and tool calls; native INT4 post-training | release/model card; incomplete recipe |
| K2.5 | major continued-pretraining branch from near-final K2 | MoonViT-3D, image/video, about 15T additional mixed tokens, joint multimodal RL | detailed technical report and weights |
| K2.6 | K2.5-family product/model advance | long-horizon coding, design, and larger agent-swarm operation | launch/model card; no equally detailed new training report |
| K2.7 Code | built on K2.6 | coding specialization and lower reported thinking-token use | release/model card; training recipe largely unknown |
| K3 | new generation | 2.8T, KDA, Attention Residuals, Stable LatentMoE, 1M context, native vision | announcement/API now; full report pending |

The decimal is therefore a family label chosen by Moonshot. Treating `K2.5` as
“K2 plus a small patch” is wrong: adding a 400M-class visual encoder and roughly
15T continued tokens is a large training event. Treating `K2.6` as a proven new
base architecture is also wrong because its public training disclosure is much
thinner.

### Where are K2.1 through K2.4?

There are no publicly released Moonshot checkpoints named `K2.1`, `K2.2`,
`K2.3`, or `K2.4` in the official research timeline, model catalog, or
Moonshot Hugging Face organization at this cutoff. The public sequence between
the original K2 and K2.5 instead uses capability or date labels:

```text
K2 Base/Instruct
  -> K2-Instruct-0905
  -> K2 Thinking
  -> K2.5
```

Moonshot's K2.5 announcement and report explain that K2.5 continues from K2
with roughly 15T mixed visual/text tokens, MoonViT, multimodal training, and new
agent behavior. They do **not** explain why the exact decimal `.5` was chosen.
The safest interpretation is therefore:

- **disclosed:** K2.5 is a major K2-lineage continuation, not the fifth public
  patch after four missing releases;
- **inferred:** `.5` communicates a substantial intermediate-generation or
  “half-step toward the next generation” position; and
- **unknown:** whether Moonshot used K2.1–K2.4 as private experiment names,
  checkpoints, or internal milestones.

Model-family decimals are product/research labels, not a promise to publish
every intervening number. By contrast, `0905` is explicitly a date-stamped
instruct refresh and `Thinking` is a capability label; neither needs to be
renumbered as K2.1.

Earlier names such as Kimi k1.5 and Kimi-VL are related research/product
ancestors, not K2 decimal checkpoints. Kimi-Audio is a separate modality branch.

## 3. K2: the language backbone inherited by K2.5

The [Kimi K2 technical report](https://arxiv.org/abs/2507.20534) discloses:

| Component | K2 value |
|---|---:|
| total parameters | 1.04T |
| active parameters per token | 32.6B |
| transformer layers | 61 |
| hidden width | 7,168 |
| routed experts | 384 |
| routed experts selected per token | 8 |
| shared experts | 1 |
| attention heads | 64 |
| attention | Multi-head Latent Attention (MLA) |
| pretraining exposure | 15.5T tokens |

This table matters because “K2.5 uses K2” means the visual vectors ultimately
enter a large, already capable K2 embedding/residual stream. It does not mean the
K2 text checkpoint could accept images before the visual path and multimodal
training were added.

The active-parameter figure cannot be recovered by multiplying total parameters
by $8/384$. Attention, embeddings, dense/shared components, and expert shapes
contribute differently.

## 4. K2.5 end-to-end visual path

Moonshot explicitly divides K2.5 into three components:

```text
image/video bytes
  -> decode and native-resolution patch packing
  -> MoonViT-3D visual encoder
  -> temporal pooling for video
  -> short MLP projector
  -> visual embeddings in K2's model width
  -> interleaved visual and text positions
  -> K2 sparse-MoE language backbone
  -> text/code/coordinates/tool calls
```

The simplified diagram is therefore accurate:

```text
MoonViT-3D -> MLP projector -> K2 MoE
```

It becomes misleading only when interpreted as “an unchanged text API calls a
separate caption service.” In K2.5, projected visual vectors are input positions
for the K2 backbone and the system undergoes substantial multimodal continued
pretraining.

### 4.1 MoonViT-3D

The [K2.5 report](https://arxiv.org/abs/2602.02276), Section 4.2, states that
MoonViT-3D is initialized from SigLIP-SO-400M and extends the MoonViT design from
Kimi-VL.

For images:

1. preserve variable original resolutions rather than forcing one fixed square;
2. divide each image into spatial patches;
3. flatten patches into a one-dimensional sequence;
4. use NaViT-style “patch n' pack” to pack sequences from different image sizes
   efficiently; and
5. run visual self-attention with spatial/packing metadata.

“Native resolution” means the encoder accepts variable shapes through packing.
It does not mean every sensor pixel becomes one LLM token or that resizing and
resource limits disappear.

For video:

1. group up to four consecutive frames as one spatiotemporal volume;
2. flatten and pack their two-dimensional patches into one sequence;
3. use the same MoonViT parameters and embedding space for image and video;
4. let attention operate across space and the short temporal group; and
5. temporally pool corresponding patches before the projector.

Pooling provides a reported fourfold reduction in temporal tokens, allowing a
longer video under the same context budget. It is lossy compression: fast events
still depend on frame sampling and the information retained by pooling.

### 4.2 MLP projector

The projector maps MoonViT output width into K2's 7,168-dimensional model space
and provides a trainable alignment bridge. The report calls it a short MLP but
does not publish enough layer-by-layer tensor detail to justify inventing a
specific hidden width, activation, or token count.

For visual vectors $V\in\mathbb R^{N_v\times d_v}$, the conceptual operation is

$$
V'=P_\phi(V),
\qquad V'\in\mathbb R^{N'_v\times 7168}.
$$

Temporal pooling can make $N'_v<N_v$ for video. The projector's job is not to
turn an image into Chinese or English words; it produces continuous embeddings
that the K2 transformer can attend over.

### 4.3 Sequence integration

Media positions and text positions are assembled into the same backbone input
sequence. The K2 transformer then performs cross-modal reasoning through its
ordinary attention and MoE layers. The text output head remains autoregressive,
which is why OCR, boxes, code, and actions can be serialized, but pixels are not
directly emitted.

### 4.4 API transport is not visual tokenization

The OpenAI-compatible request schema uses fields named `image_url` and
`video_url`, but the word `url` is easy to misread. In the current
[Kimi visual-input guide](https://platform.kimi.com/docs/guide/use-kimi-vision-model),
the dependable payload forms are:

| Client-side form | Value placed in `image_url.url` or `video_url.url` | Suitable use |
|---|---|---|
| inline data URL | `data:image/png;base64,...` or `data:video/mp4;base64,...` | small, one-off media |
| uploaded-media reference | `ms://<file_id>` after `POST /v1/files` with `purpose="image"` or `purpose="video"` | large or repeatedly referenced media |

The guide explicitly says arbitrary HTTP or HTTPS image URLs are not currently
supported. The generic
[chat schema](https://platform.kimi.com/docs/api/chat) likewise documents
base64 data URLs and `ms://` file references as the two supported media forms.
`ms://` is an internal Moonshot Storage reference, not a public URL that the
model downloads from the Internet. The field name is inherited from an API
content-block convention; it does not prove URL fetching.

For native visual input, uploading with `purpose="image"` or `purpose="video"`
is different from `purpose="file-extract"`. The former preserves media for the
vision path. The latter asks the file service to extract text that an application
then places in a prompt, which is a text cascade rather than native visual
inference.

The two equivalent request shapes are conceptually:

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
```

and, after a separate multipart upload:

```json
{"type":"video_url","video_url":{"url":"ms://file-id"}}
```

The `message.content` value must remain an array of typed content blocks; JSON
stringifying that entire array into an ordinary text message loses the media
type. Base64 increases the transmitted size and request parsing cost, so upload
plus `file_id` is the practical path for large videos and media reuse. The
current guide caps the request body at 100 MB, the file API caps an individual
upload at 100 MB, recommends no more than 4K for images and 1080p for video, and
offers a token-count endpoint because media token cost is dynamic.

#### Who selects the video frames?

The basic API caller sends a video container, not a manually prepared list of
JPEG frames. Nevertheless, a model cannot consume MP4/H.264 bytes directly.
Ordinary media and model-serving code must still perform the following work on
Moonshot's side:

```text
MP4/MOV/WebM bytes or ms:// object
  -> demux and decode the compressed video stream
  -> select timestamped frames under a visual-token budget
  -> resize/canonicalize and retain timing metadata
  -> group up to four consecutive selected frames
  -> spatial patching and spatiotemporal packing
  -> MoonViT-3D attention
  -> patch-wise temporal pooling
  -> MLP projection into K2 width
  -> interleave visual embeddings with text
  -> K2 MoE prefill and autoregressive text/tool output
```

There are therefore **two distinct temporal reductions**:

1. **clip-level sampling** chooses a bounded set of frames from the complete
   decoded video; and
2. **model-level compression** groups up to four consecutive retained frames and
   temporally pools corresponding patches before the MLP projector.

The API guide confirms the first boundary indirectly by saying that a video is
represented by a variable number of key frames and that cost grows with key-frame
count and resolution. It does not publish the production sampling policy, frame
rate, scene-change logic, maximum retained-frame count, or whether “key frame”
means codec I-frames versus model-selected frames. K2.5's paper discloses a
specific uniform-frame policy for benchmark evaluation, but that is not evidence
for the production API policy.

The paper discloses the second boundary directly: MoonViT-3D treats up to four
consecutive frames as one spatiotemporal volume, jointly packs their 2D patches,
and applies lightweight temporal pooling before projection. This is reported to
reduce temporal positions by up to fourfold. It does not recover an event that
the earlier sampling stage omitted.

#### Is MoonViT a separate model at inference time?

It is a separately parameterized vision-encoder submodule—about 400M parameters
for K2.5—not a captioning API that returns prose to K2. The MLP passes its
continuous outputs directly into the jointly trained K2 backbone. A serving
system may schedule this submodule on separate workers or co-locate it with the
language prefill workers; either way, it remains one end-to-end model graph from
media to answer.

Moonshot publicly describes a Decoupled Encoder Process for **training**: the
small vision encoder is replicated, visual work is load-balanced by patch count,
its outputs are gathered to pipeline stage zero, and its forward pass is later
recomputed for backpropagation. That is evidence that the encoder is
computationally separable, but it does not disclose the exact production API
topology. K3's current API exposes the same media abstraction while its visual
encoder and serving internals remain undisclosed.

The disclosed K2.5 path is visual-only. Neither the MoonViT-3D architecture nor
the current visual API guide establishes that the audio track inside a video is
encoded. Applications that require speech or sound should not assume it is heard;
they need a documented audio-capable path or an explicit ASR/audio-model branch.

## 5. Why K2.5 is more than a bolted-on adapter

### 5.1 Vision-tower training and bridge alignment

The report describes an initial MoonViT-3D stage over image-text and video-text
pairs. Targets include captions, alternative text, grounding boxes, and OCR.
Unlike Kimi-VL's earlier recipe, the K2.5 report explicitly says this stage omits
contrastive loss and uses conditional caption-generation cross-entropy alone.

The first alignment stage updates the ViT while aligning it with the smaller
Moonlight-16B-A3B language model and consumes about 1T caption-style tokens. A
short second stage updates only the MLP projector to bridge the visual encoder to
the 1T K2 backbone before joint training.

### 5.2 Joint continual pretraining

K2.5 starts from a checkpoint near the final K2 model and processes roughly 15T
additional tokens across visual and text data. The report's controlled ablation
finds early, low-ratio vision fusion (10:90 vision/text in that comparison)
better than later, higher-ratio alternatives under a fixed token budget.

The important causal claim is limited: the ablation supports Moonshot's K2.5
recipe. It does not prove that 10% vision is universally optimal or disclose
every production mixture weight.

Data categories include captions, interleaved documents, OCR, visual knowledge,
grounding, images paired with HTML/React/SVG, GUI screenshots and actions,
videos, points, boxes, contours, and segmentation. Long-context stages extend
the context toward 262,144 positions.

### 5.3 Zero-vision SFT is not zero-vision pretraining

K2.5's main supervised fine-tuning stage uses text-only examples. Moonshot calls
this **zero-vision SFT**, not a text-only model. The report says human-designed
visual trajectories at SFT generalized worse; strong joint pretraining allowed
text instruction behavior to transfer across modalities.

This distinction prevents a common misreading:

```text
joint multimodal pretraining -> text-only behavioral SFT -> joint text/vision RL
```

Images can also be manipulated through an IPython tool during agent training.
Tool-mediated image operations and direct visual embeddings can coexist.

### 5.4 Joint text/vision reinforcement learning

Visual RL tasks include grounding, counting, charts/documents, visual science,
OCR, points, boxes, segmentation, and synthetic puzzles. Disclosed reward
signals include Intersection over Union, soft F1, Gaussian point matching,
normalized OCR edit distance, and count error.

This stage trains response and action behavior over visual evidence. It is
stronger evidence for integrated visual reasoning than an API wrapper, while
still not supplying an image-generation decoder.

## 6. K3 and the precise meaning of “native vision”

Moonshot's [K3 technical blog](https://www.kimi.com/blog/kimi-k3) and
[API guide](https://platform.kimi.com/docs/guide/kimi-k3-quickstart) disclose:

- 2.8T total parameters;
- 896 experts with 16 selected in the disclosed sparse design;
- Kimi Delta Attention (KDA), Attention Residuals (AttnRes), and Stable
  LatentMoE;
- a one-million-token context window;
- “native vision” / “natively multimodal” positioning; and
- direct image and video input through the Kimi API.

The visual API accepts base64 images or uploaded media identifiers. Current
documentation lists image and video understanding, while the model's response is
text/structured output/tool calls.

At the cutoff, Moonshot says full weights will be released by 2026-07-27 and
architecture/training/evaluation details will arrive with the technical report.
Therefore the following remain unknown:

- whether K3 retains MoonViT-3D or uses a new visual encoder;
- the visual encoder size, patching, resolution and frame policy;
- projector/resampler structure and visual token budget;
- which K3 layers see visual positions and whether modality experts exist;
- the quantity and schedule of image/video pretraining;
- visual SFT/RL data and rewards; and
- any internal image/video generation head.

The responsible interpretation is: K3 has a first-class visual input contract
and Moonshot claims deep model integration, but “native” is not evidence that
K3 eliminated a ViT or can synthesize pixels. The pending report must answer
those architectural questions.

For a component-by-component derivation of KDA, Block AttnRes, Stable
LatentMoE, Quantile Balancing, Per-Head Muon, MX quantization, expert-parallel
training, Mooncake serving, preserved-thinking semantics, launch cases, and all
35 benchmark rows, continue to the dedicated
[Kimi K3 architecture and systems deep dive](kimi-k3.md).

## 7. Does Kimi natively generate images?

No public K2.5/K2.6/K3 architecture source at the cutoff describes a visual
codebook, Variational Autoencoder (VAE), diffusion/flow image head, or pixel
decoder attached to the checkpoint. Their disclosed visual path is input-side.

A Kimi product can still create visual artifacts by:

- writing HTML, SVG, game, or CAD code and inspecting rendered screenshots;
- calling a separate image-generation tool; or
- orchestrating specialist media services.

Those are valuable agent capabilities, but they are not the same as sampling
image tokens or latents from the K3 model itself.

## 8. Other modalities: Kimi-Audio

Moonshot's separate [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio)
checkpoint supports audio understanding, transcription, and speech generation.
Its public architecture has three stages:

1. an audio tokenizer produces 12.5 Hz discrete semantic codes plus continuous
   acoustic features from a downsampled Whisper encoder;
2. an Audio LLM initialized from a text LLM processes multimodal inputs through
   shared transformer layers and uses parallel heads for text and discrete audio
   semantic tokens; and
3. a flow-matching detokenizer plus BigVGAN vocoder turns predicted audio codes
   into a streaming waveform.

This is an example of generation-native audio: the language-centered transformer
predicts audio semantic tokens, but a specialized detokenizer and vocoder are
still required. It is a separate branch, not evidence that every K2/K3 endpoint
accepts or emits speech.

## 9. Capability matrix

| Model | Text in | Image in | Video in | Audio in | Text/tools out | Image out in checkpoint | Speech out |
|---|---:|---:|---:|---:|---:|---:|---:|
| K2 | yes | no | no | no | yes | no | no |
| K2.5 | yes | yes | yes | no disclosed | yes | no disclosed | no |
| K2.6 | yes | yes | yes | no disclosed | yes | no disclosed | no |
| K2.7 Code | yes | yes | yes | no disclosed | yes | no disclosed | no |
| K3 API | yes | yes | yes | no documented in current guide | yes | no disclosed | no disclosed |
| Kimi-Audio | yes | no | no | yes | yes | no | yes |

“No disclosed” is deliberately different from “impossible.” It records the
public checkpoint/API boundary rather than guessing about internal products.

## Primary sources

- Moonshot AI, [official research timeline](https://www.kimi.com/en/blog/),
  [Kimi API model catalog](https://platform.kimi.com/docs/models), and
  [official Hugging Face model inventory](https://huggingface.co/moonshotai/models).
- Moonshot AI, [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
  and [official repository](https://github.com/MoonshotAI/Kimi-K2).
- Moonshot AI, [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
  and [official repository](https://github.com/MoonshotAI/Kimi-K2.5).
- Moonshot AI, [Kimi-VL](https://github.com/MoonshotAI/Kimi-VL).
- Moonshot AI, [Kimi K3 technical blog](https://www.kimi.com/blog/kimi-k3),
  [K3 API guide](https://platform.kimi.com/docs/guide/kimi-k3-quickstart), and
  [visual-input guide](https://platform.kimi.com/docs/guide/use-kimi-vision-model).
- Moonshot AI, [chat content-block schema](https://platform.kimi.com/docs/api/chat)
  and [media file upload API](https://platform.kimi.com/docs/api/files-upload).
- Moonshot AI, [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio).

For the broader training, optimizer, long-context, and agent lineage, see the
[Moonshot/Kimi Agentic RL case study](../agentic-rl/case-studies/kimi.md).
