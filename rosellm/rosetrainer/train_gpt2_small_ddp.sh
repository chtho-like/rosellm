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
  --grad-accum-steps 2 \
  --grad-clip-norm 1.0 \
  --num-steps 122000 \
  --lr 3e-4 \
  --data-mode fineweb_npy \
  --train-npy data/edu_fineweb10B/edufineweb_train_000001.npy \
  --val-npy data/edu_fineweb10B/edufineweb_val_000000.npy \
  --val-max-tokens 1000000 \
  --eval-steps 1000 \
  --tokenizer-name gpt2 \
  --data-seed 42 \
  --use-wandb \
  --wandb-project rosetrainer \
  --wandb-run-name gpt2_small_ddp_edu_fineweb10B \
  --checkpoint-path checkpoints/gpt2_small_ddp.pt

