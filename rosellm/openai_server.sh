#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

python -m roseinfer.server \
  --hf-model-id gpt2 \
  --host 127.0.0.1 \
  --port 8888 \
  --kv-cache-max-concurrency 16
