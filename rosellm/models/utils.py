import torch.nn as nn

from rosellm.models.opt import OPTForCausalLM

MODEL_CLASSES = {
    'opt': OPTForCausalLM,
}

def get_model(model_name: str) -> nn.Module:
    for name, cls in MODEL_CLASSES.items():
        if name in model_name:
            return cls.from_pretrained(model_name)
    raise ValueError(f"Model {model_name} not found.")


