#!/bin/bash

python -m roseinfer.offline_example \
  --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt \
  --tokenizer-name gpt2 \
  --prompts "hi, " "hello," "how" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95 \
  --do-sample
