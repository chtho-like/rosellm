# Dev Notes (rosellm)

This folder records design/implementation notes, profiling results, and benchmark artifacts produced during development.

- `2026-03-03_roseinfer_v2_deepseek_v2_lite_fp8_design.md`: detailed design for a vLLM-v1-style `roseinfer` rewrite targeting DeepSeek-V2-Lite-Chat-FP8 on 2×H100 with DPA(=DP-Attention)+EP.
- `2026-03-03_deepseek_official_infra_backends_plan.md`: integration plan for DeepSeek official infra backends (FlashMLA / DeepEP / DeepGEMM) plus FlashAttention/FlashInfer.
- `2026-03-03_roseinfer_v2_runtime_execution_model.md`: how rank0 runs HTTP+Scheduler+Driver+Worker0, and how step broadcast/gather works without deadlocks.
- `2026-03-03_roseinfer_v2_dev_environment.md`: recommended Python tooling (`uv`) and optional CUDA kernel libraries installation philosophy.
- `2026-03-03_roseinfer_v2_scheduler_models_industry.md`: industry patterns for “distributed scheduling” (DP routing / DPA / PD disaggregation) and why we start with a centralized scheduler inside one DPA+EP instance.
