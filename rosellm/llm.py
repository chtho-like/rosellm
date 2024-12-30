from typing import Optional, Type, Union, Dict, Any, List
from config import TaskOption
from input import SamplingParams
from output import RequestOutput 
from engine import EngineArgs, LLMEngine

class LLM:
    engine_class : Type[LLMEngine]
    llm_engine: LLMEngine
    def __init__(
        self,
        # The name or path of a HuggingFace Transformers model.
        model: str, 
        *,
        # Core model settings.
        task: TaskOption = "auto",
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        tokenizer_revision: Optional[str] = None,
        skip_tokenizer_init: bool = False,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        # Model execution settings.
        dtype: str = "auto",
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
        
        # Performance settings.
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        
        # Optional features.
        allowed_local_media_path: str = "",
        seed: int = 0,
        **kwargs,
    ) -> None:
        """Initializes a vLLM instance for large language model inference.

        Args:
            model (str): The name or path of a HuggingFace 
                Transformers model.
            task (TaskOption, optional): The task to run. Options are 
                "auto", "generate", "embedding", "embed", "classify", 
                "score", "reward". Defaults to "auto".
            tokenizer (Optional[str], optional): The name or path of a 
                HuggingFace Transformers tokenizer. If None, uses the 
                default tokenizer for the model. Defaults to None.
            tokenizer_mode (str, optional): The tokenizer mode. "auto" 
                will use the fast tokenizer if available, "slow" will 
                always use the slow tokenizer. Defaults to "auto".
            tokenizer_revision (Optional[str], optional): The specific 
                tokenizer version to use. Can be a branch name, tag 
                name, or commit id. Defaults to None.
            skip_tokenizer_init (bool, optional): If True, skips 
                initialization of tokenizer and detokenizer. Expects 
                valid prompt_token_ids and None for prompt from input. 
                Defaults to False.
            revision (Optional[str], optional): The specific model 
                version to use. Can be a branch name, tag name, or 
                commit id. Defaults to None.
            trust_remote_code (bool, optional): Trust remote code 
                (e.g., from HuggingFace) when downloading the model 
                and tokenizer. Security risk in untrusted environments. 
                Defaults to False.
            dtype (str, optional): Data type for model weights and 
                activations. Supports 'float32', 'float16', and 
                'bfloat16'. 'auto' uses model config's torch_dtype. 
                Defaults to "auto".
            quantization (Optional[str], optional): Method to quantize 
                model weights. Supports "awq", "gptq", and "fp8" 
                (experimental). None uses model config or dtype. 
                Defaults to None.
            tensor_parallel_size (int, optional): Number of GPUs to 
                use for distributed execution with tensor parallelism. 
                Defaults to 1.
            compilation_config (Optional[Union[int, Dict[str, Any]]], 
                optional): Level of compilation optimization (if int) 
                or full compilation configuration (if dict). Defaults 
                to None.
            gpu_memory_utilization (float, optional): Ratio (0-1) of 
                GPU memory to reserve for model weights, activations, 
                and KV cache. Higher values improve throughput but 
                risk OOM. Defaults to 0.9.
            swap_space (float, optional): Size (GiB) of CPU memory 
                per GPU for swap space. Needed when best_of > 1. Set 
                to 0 if best_of=1. Defaults to 4.
            cpu_offload_gb (float, optional): Size (GiB) of CPU 
                memory for offloading model weights. Increases virtual 
                GPU memory at cost of CPU-GPU transfer. Defaults to 0.
            enforce_eager (bool, optional): If True, disables CUDA 
                graph and forces eager execution. If False, uses 
                hybrid of CUDA graph and eager execution. Defaults to 
                False.
            max_seq_len_to_capture (int, optional): Maximum sequence 
                length for CUDA graphs. Longer sequences fall back to 
                eager mode. Affects encoder-decoder models similarly. 
                Defaults to 8192.
            disable_custom_all_reduce (bool, optional): Whether to 
                disable custom all-reduce implementation. See 
                ParallelConfig documentation. Defaults to False.
            disable_async_output_proc (bool, optional): Whether to 
                disable async output processing. May reduce 
                performance. Defaults to False.
            allowed_local_media_path (str, optional): Directories 
                where API requests can read local media files. 
                Security risk in untrusted environments. Defaults to 
                "".
            seed (int, optional): Seed for random number generator 
                used in sampling. Defaults to 0.
        """
        engine_args = EngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            **kwargs,
        )
        self.engine_class = LLMEngine
        self.llm_engine = self.engine_class.from_engine_args(
            engine_args
        )

    def generate(
        self,
        prompts: Union[str, List[str]] = None,
        sampling_params: Optional[Union[SamplingParams, 
                                        List[SamplingParams]]] = None,
    ) -> List[RequestOutput]:
        pass
