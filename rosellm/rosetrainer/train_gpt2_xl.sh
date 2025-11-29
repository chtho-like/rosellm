#!/bin/bash
torchrun --nproc_per_node=2 train_ddp.py \
  --n-layers 48 \
  --n-heads 25 \
  --d-model 1600 \
  --d-ff 6400 \
  --dropout 0.1 \
  --max-position-embeddings 1024 \
  --seq-len 1024 \
  --batch-size 1 \
  --num-steps 2000 \
  --lr 3e-4 \
  --train-data data/train.txt \
  --tokenizer-name gpt2 \
  --checkpoint-path checkpoints/gpt2_xl_ddp.pt

