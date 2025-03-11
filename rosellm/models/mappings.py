from rosellm.models.lazy_mapping import _LazyAutoMapping, _LazyConfigMapping

CONFIG_MAPPING_NAMES = {
    "qwen2": "Qwen2Config",
}

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = {
    "qwen2": "Qwen2ForCausalLM",
}

CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
