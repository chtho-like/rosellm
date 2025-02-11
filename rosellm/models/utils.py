import torch.nn as nn

from rosellm.models.opt import OPTForCausalLM

MODEL_CLASSES = {
    'opt': OPTForCausalLM,
}

def get_model(model_name: str) -> nn.Module:
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model {model_name} not found.")
    return MODEL_CLASSES[model_name].from_pretrained(model_name)

