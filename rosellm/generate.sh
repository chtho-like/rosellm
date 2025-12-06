#!/bin/bash

python -m roseinfer.cli_generate --checkpoint-path rosetrainer/checkpoints/gpt2_small_ddp_edu_amp_bf16_init.pt --tokenizer-name gpt2 --max-new-tokens 1000 --stream --prompt "hi, " --top-k 40 --top-p 0.99 --do-sample
