#!/bin/bash
python generate.py \
  --checkpoint-path checkpoints/gpt2_small_ddp_edu.pt \
  --tokenizer-name gpt2 \
  --vocab-size 50257 \
  --max-position-embeddings 1024 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.0 \
  --prompt "You have said" \
  --max-new-tokens 100 \
  --temperature 0.99 \
  --top-p 0.99 \
  --do-sample \
  --device cpu

