import importlib
from collections import OrderedDict


class _LazyConfigMapping(OrderedDict):
    def __init__(self, mapping):
        self._mapping = mapping
        self._modules = {}

    def __getitem__(self, key):
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = key
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "rosellm.models"
            )
        return getattr(self._modules[module_name], value)

    def keys(self):
        return list(self._mapping.keys())

    def values(self):
        return [self[key] for key in self._mapping.keys()]

    def items(self):
        return [(key, self[key]) for key in self._mapping.keys()]

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        return item in self._mapping


class _LazyAutoMapping(OrderedDict):
    def __init__(self, config_mapping, model_mapping):
        # E.g. config_mapping = {"qwen2": "Qwen2Config"}
        self._config_mapping = config_mapping
        # E.g. model_mapping = {"qwen2": "Qwen2ForCausalLM"}
        self._model_mapping = model_mapping
        # E.g. reverse_config_mapping = {"Qwen2Config": "qwen2"}
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._modules = {}

    def __len__(self):
        return len(
            set(self._config_mapping.keys()).intersection(
                self._model_mapping.keys(),
            )
        )

    def __getitem__(self, key):
        model_type = self._reverse_config_mapping[key.__name__]
        # E.g. model_type = "qwen2"
        if model_type in self._model_mapping:
            # E.g. model_name = "Qwen2ForCausalLM"
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type
        if module_name not in self._modules:
            # Loading the module dynamically.
            self._modules[module_name] = importlib.import_module(
                f".{module_name}", "rosellm.models"
            )
        return getattr(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys

    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item):
        if (
            not hasattr(item, "__name__")
            or item.__name__ not in self._reverse_config_mapping
        ):
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping.keys()
