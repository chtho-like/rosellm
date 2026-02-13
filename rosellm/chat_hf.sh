#!/bin/bash

python -m roseinfer.cli_generate \
  --hf-model-id gpt2 \
  --interactive \
  --stream \
  --max-new-tokens 256 \
  --top-k 40 \
  --top-p 0.95 \
  --do-sample
