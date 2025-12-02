#!/bin/bash
python train_minimal.py \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --d-ff 3072 \
  --dropout 0.1 \
  --max-position-embeddings 1024 \
  --seq-len 1024 \
  --batch-size 2 \
  --grad-accum-steps 2 \
  --num-steps 50 \
  --grad-clip-norm 1.0 \
  --lr 3e-4 \
  --train-data data/train.txt \
  --tokenizer-name gpt2 \
  --max-tokens 100000 \
  --data-seed 42 \
  --use-wandb \
  --checkpoint-path checkpoints/gpt2_small_minimal.pt

