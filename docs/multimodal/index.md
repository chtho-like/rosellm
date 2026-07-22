# Multimodal Foundation Models: Architecture and Evidence Map

**Verified through:** 2026-07-22. This section uses model-developer papers,
released code and model cards, and official API documentation. Product claims
and vendor benchmarks are not independent reproductions.

Multimodal systems are easiest to understand when three separate questions stay
separate:

1. **How is a modality represented?** Pixels, frames, waveforms, and documents
   must become tensors, continuous embeddings, or discrete codes.
2. **Where does cross-modal reasoning happen?** A language backbone may consume
   projected visual embeddings, cross-attend to an encoder, or process every
   modality in one shared token stream.
3. **What can the system emit?** Text generation, image understanding, image
   synthesis, video understanding, and speech generation require different
   output heads or decoders even when a backbone is shared.

Calling a model “native multimodal” answers none of those questions by itself.
The phrase is useful only after its integration level and output capabilities
are specified.

## The shortest correct mental model

For a mainstream image-understanding Large Multimodal Model (LMM), the path is:

```text
image or video
  -> decode, resize/tile/sample, patchify
  -> vision encoder
  -> token compression or resampling
  -> projector into the language-model width
  -> interleaved visual embeddings and text-token embeddings
  -> shared autoregressive language backbone
  -> text, structured output, coordinates, code, or tool calls
```

This is often summarized as “Vision Transformer (ViT) + projector + Large
Language Model (LLM).” That summary is structurally correct for Kimi K2.5,
DeepSeek-VL2, and many other systems, but it does **not** imply that vision is a
late product plug-in. If the visual path and language backbone are jointly
trained over a large multimodal mixture, visual tokens participate in ordinary
backbone layers, and visual tasks are included in post-training, the resulting
system can reasonably be described as deeply integrated or native at the
training and interface levels while still containing a ViT.

Conversely, a single API that accepts images can be a cascade:

```text
image -> separate OCR/caption model -> text -> text-only LLM
```

That system is multimodal at the product interface but not at the main model.

## Input and output are independent axes

| Capability | Required path | What it does **not** imply |
|---|---|---|
| image understanding | image encoder/tokenizer -> reasoning backbone | image generation |
| video understanding | frame/spatiotemporal encoder -> reasoning backbone | native video synthesis |
| document OCR | high-resolution visual encoder -> text decoder | general visual reasoning |
| text-to-image | semantic conditioning -> image-token or latent decoder | ability to inspect an input image |
| image editing | encode source image + instruction -> conditional image decoder | one shared model for every stage |
| speech understanding | acoustic tokenizer/encoder -> reasoning backbone | speech output |
| speech generation | semantic/acoustic codes -> waveform decoder/vocoder | general audio understanding |

The practical mistake to avoid is inferring output modality from input
modality. Kimi K2.5 and K3 accept visual input but their disclosed model output
path is text, code, structured content, and tool calls. DeepSeek's Janus branch,
not its current V4 API model, is the disclosed image-generation branch.

## A five-level meaning of “native”

This knowledge base reports “native multimodal” using five explicit levels:

| Level | Test | Example interpretation |
|---|---|---|
| **API-native** | one documented endpoint accepts media directly | convenient product contract; architecture may still be cascaded |
| **sequence-native** | modality embeddings occupy first-class positions in the model context | the language backbone can directly attend over visual/audio representations |
| **training-native** | modalities are present through substantial joint pretraining, not attached only during final instruction tuning | cross-modal behavior is learned with the backbone rather than only through a thin adapter |
| **backbone-native** | a shared transformer performs material computation for multiple modalities | modality encoders, tokenizers, and output heads may still differ |
| **generation-native** | the trained system can emit non-text codes or latents through a learned media decoder | image/audio/video output, not merely understanding |

No level requires feeding raw pixels or waveform samples into an ordinary text
embedding table. Modality-specific front ends are usually desirable because
two-dimensional locality, temporal sampling, and acoustic bandwidth have very
different inductive biases and sequence lengths.

## Current Kimi and DeepSeek snapshot

| Family or branch | Inputs | Outputs | Public architectural reading |
|---|---|---|---|
| Kimi K2 | text | text, code, tools | text-first 1T Mixture-of-Experts (MoE) backbone |
| Kimi K2.5/K2.6 | text, image, video | text, code, tools | MoonViT-3D + MLP projector + K2 MoE, jointly continued-pretrained |
| Kimi K3 | text, image, video through the API | text, code, tools | “native vision” is official; exact visual stack is still undisclosed pending the technical report |
| Kimi-Audio | text, audio | text and speech | separate audio-tokenizer/Audio-LLM/flow-and-vocoder branch |
| DeepSeek V4 | text | text, code, tools | current flagship API is explicitly text-only |
| DeepSeek-VL/VL2 | text, image | text and grounding coordinates | encoder-projector-LLM understanding branch |
| DeepSeek-OCR/OCR 2 | document or scene image + prompt | text/Markdown/layout representation | high-resolution optical compression and reading-order branch |
| DeepSeek Janus/Janus-Pro | text, image | text and generated images | unified transformer with separate understanding and generation representations |
| DeepSeek JanusFlow | text, image | text and generated images | autoregressive language modeling plus rectified-flow image generation |

The table describes public checkpoints and documented APIs, not an undisclosed
internal routing layer. A consumer application may add OCR, search, image
generation, or media-processing tools around any base model.

## Reading path

1. [Representation, fusion, and the meaning of native](architecture.md) derives
   the full image/video-to-token path and separates integration levels.
2. [Kimi model lineage and visual architecture](kimi.md) explains why K2.5 is
   both “ViT + LLM” and a jointly trained multimodal model, what the K2 suffixes
   mean, and what is actually known about K3.
3. [DeepSeek's multimodal portfolio](deepseek.md) separates VL, Janus, OCR, and
   the current text-only V4 line, with source-level OCR and VL2 structures.
4. [How LLM-centered systems generate images](generation.md) compares
   cascades, discrete visual-token autoregression, diffusion, rectified flow,
   and mixed-objective shared transformers.
5. For training and agent details outside the modality path, continue to the
   [Moonshot Kimi](../agentic-rl/case-studies/kimi.md) and
   [DeepSeek](../agentic-rl/case-studies/deepseek.md) evidence reconstructions.

## Evidence rules for this section

- A model name or decimal suffix is lineage metadata, not architectural proof.
- “Supports image/video” is an input-interface statement unless an output media
  path is separately disclosed.
- A released projector or model configuration is a confirmed artifact; an
  architecture diagram in a vendor paper is a disclosed fact.
- A product demonstration is not evidence for a training recipe.
- K2.5 internals are not assigned to K3 merely because K3 follows K2.5.
- Active parameters cannot be estimated by multiplying total MoE parameters by
  the fraction of selected experts; attention, embeddings, shared experts, and
  expert-internal structure remain outside that shortcut.

The repository-wide [research standard](../research-method.md) defines the
Disclosed, Confirmed, Reproduced, Inferred, and Unknown evidence classes used in
the model-specific chapters.
