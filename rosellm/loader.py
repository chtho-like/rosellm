import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from torch import nn 
from config import LoadConfig, ModelConfig, LLMConfig

class BaseModelLoader:
    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config 
    
    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self, *, llm_config: LLMConfig) -> nn.Module:
        raise NotImplementedError

class DefaultModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
    
    def _prepare_weights(
        self, 
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
    ) -> Tuple[str, List[str], bool]:
        is_local = os.path.isdir(model_name_or_path)
    
    def download_model(self, model_config: ModelConfig) -> None:
        pass 
    
    def load_model(self, llm_config: LLMConfig) -> nn.Module:
        pass


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    return DefaultModelLoader(load_config)
