import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from torch import nn 
from config import LoadConfig, ModelConfig, LLMConfig
import huggingface_hub
import fnmatch

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
        if is_local:
            model_folder = model_name_or_path
        else:
            model_folder = download_weights(
                model_name_or_path, 
            )
    
    def download_model(self, model_config: ModelConfig) -> None:
        pass 
    
    def load_model(self, llm_config: LLMConfig) -> nn.Module:
        pass

def download_weights(
    model_name_or_path: str,
    allow_patterns: List[str],
    revision: Optional[str] = None,
):
    if not huggingface_hub.constants.HF_HUB_OFFLINE:
        # Before downloading, check whether the model exists.
        fs = huggingface_hub.HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    return DefaultModelLoader(load_config)
