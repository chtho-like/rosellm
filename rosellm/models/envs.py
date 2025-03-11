import os
import uuid

import torch

CACHE_DIR = os.getenv("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
SESSION_ID = uuid.uuid4()

_torch_version = torch.__version__
