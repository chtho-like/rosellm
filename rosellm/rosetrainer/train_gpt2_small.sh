#!/bin/bash
torchrun --nproc_per_node=2 train_ddp.py \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.1 \
  --max-position-embeddings 1024 \
  --seq-len 1024 \
  --batch-size 2 \
  --num-steps 10000 \
  --lr 3e-4 \
  --train-data data/gutenberg/*.txt \
  --tokenizer-name gpt2 \
  --max-tokens 100000000 \
  --data-seed 42 \
  --val-ratio 0.001 \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt

