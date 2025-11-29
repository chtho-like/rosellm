#!/bin/bash
python generate.py \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt \
  --tokenizer-name gpt2 \
  --vocab-size 50257 \
  --max-position-embeddings 1024 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.0 \
  --prompt "Hello, " \
  --max-new-tokens 500 \
  --temperature 0.99 \
  --top-p 0.99 \
  --do-sample \
  --device cuda

