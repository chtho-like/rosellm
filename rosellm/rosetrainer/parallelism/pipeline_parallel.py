from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

import torch
import torch.distributed as dist

# References:
# [1] Huang, Y. et al. "GPipe: Efficient Training of Giant Neural Networks using
#     Pipeline Parallelism." NeurIPS (2019)
# [2] Narayanan, D. et al. "Memory-Efficient Pipeline-Parallel DNN Training."
#     ICML (2021)
# [3] Narayanan, D. et al. "Efficient Large-Scale Language Model Training on GPU
#     Clusters Using Megatron-LM." arXiv:2104.04473 (2021)
# [4] DeepSpeed PipeDream implementation:
#     https://github.com/microsoft/DeepSpeed/tree/master/deepspeed/runtime/pipe
# [5] Megatron-LM implementation:
#     https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/p2p/


class PipelineStage(Enum):
    """Enum representing different stages in the pipeline."""

    FORWARD = 0
    BACKWARD = 1


class MicrobatchInfo:
    """Class to track microbatch state in the pipeline."""

    def __init__(self, id: int, batch_size: int):
        """
        Initialize microbatch info.

        Args:
            id: Microbatch identifier.
            batch_size: Size of the microbatch.
        """
        self.id = id
        self.batch_size = batch_size
        self.stage = PipelineStage.FORWARD
        self.forward_output: Optional[Dict[str, torch.Tensor]] = None
        self.gradient: Optional[Dict[str, torch.Tensor]] = None
        self.inputs: Dict[str, torch.Tensor] = {}
        self.loss: Optional[torch.Tensor] = None


class PipelineParallel:
    """
    Pipeline parallelism implementation for RoseLLM.
    Splits model across multiple GPUs along the layer dimension.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        local_rank: int,
        world_size: int,
        num_microbatches: int,
        pp_size: int,
        pp_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize the pipeline parallel manager.

        Args:
            model: The PyTorch model for this pipeline stage.
            local_rank: Local rank of the current process.
            world_size: Total number of processes.
            num_microbatches: Number of microbatches to split each batch into.
            pp_size: Size of pipeline parallel group.
            pp_group: Optional existing pipeline parallel process group.
        """
        self.model = model
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_microbatches = num_microbatches
        self.pp_size = pp_size

        # Ensure world_size is divisible by pp_size
        assert (
            world_size % pp_size == 0
        ), "World size must be divisible by pipeline parallel size"

        # Set up process groups if not provided
        if pp_group is None:
            self.pp_group = self._initialize_process_group()
        else:
            self.pp_group = pp_group

        # Get pipeline stage rank
        self.pp_rank = dist.get_rank(self.pp_group)
        self.is_first_stage = self.pp_rank == 0
        self.is_last_stage = self.pp_rank == self.pp_size - 1

        # Device for this stage
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Keep track of microbatches
        self.active_microbatches: Deque[MicrobatchInfo] = deque()

    def _initialize_process_group(self) -> dist.ProcessGroup:
        """Initialize the pipeline parallel process group."""
        # Make sure global process group is initialized
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed is not initialized. Call dist.init_process_group first."
            )

        # Calculate the number of PP groups
        num_pp_groups = self.world_size // self.pp_size

        # Determine the PP group this rank belongs to
        pp_group_id = self.local_rank // self.pp_size

        # Create PP groups
        pp_groups = []
        for i in range(num_pp_groups):
            ranks_in_group = [i * self.pp_size + j for j in range(self.pp_size)]
            pp_groups.append(dist.new_group(ranks=ranks_in_group))

        # Return the PP group for this rank
        return pp_groups[pp_group_id]  # type: ignore[no-any-return]

    def _send_tensors(self, tensors: Dict[str, torch.Tensor], dest: int) -> None:
        """Send tensors to the next stage."""
        # Send number of tensors first
        num_tensors = torch.tensor([len(tensors)], device=self.device)
        dist.send(num_tensors, dest, group=self.pp_group)

        # Send each tensor name and data
        for name, tensor in tensors.items():
            # Send name length
            name_bytes = name.encode("utf-8")
            name_length = torch.tensor([len(name_bytes)], device=self.device)
            dist.send(name_length, dest, group=self.pp_group)

            # Send name bytes
            name_tensor = torch.tensor(
                [ord(c) for c in name], dtype=torch.uint8, device=self.device
            )
            dist.send(name_tensor, dest, group=self.pp_group)

            # Send tensor shape
            shape = torch.tensor(list(tensor.shape), device=self.device)
            shape_length = torch.tensor([len(shape)], device=self.device)
            dist.send(shape_length, dest, group=self.pp_group)
            dist.send(shape, dest, group=self.pp_group)

            # Send tensor data
            dist.send(tensor, dest, group=self.pp_group)

    def _recv_tensors(self, src: int) -> Dict[str, torch.Tensor]:
        """Receive tensors from the previous stage."""
        # Receive number of tensors
        num_tensors = torch.tensor([0], device=self.device)
        dist.recv(num_tensors, src, group=self.pp_group)

        # Receive each tensor
        tensors = {}
        for _ in range(int(num_tensors.item())):
            # Receive name length
            name_length = torch.tensor([0], device=self.device)
            dist.recv(name_length, src, group=self.pp_group)

            # Receive name bytes
            name_tensor = torch.zeros(
                int(name_length.item()), dtype=torch.uint8, device=self.device
            )
            dist.recv(name_tensor, src, group=self.pp_group)
            name = "".join([chr(int(b)) for b in name_tensor.tolist()])

            # Receive tensor shape
            shape_length = torch.tensor([0], device=self.device)
            dist.recv(shape_length, src, group=self.pp_group)
            shape = torch.zeros(
                int(shape_length.item()), dtype=torch.int64, device=self.device
            )
            dist.recv(shape, src, group=self.pp_group)

            # Receive tensor data
            tensor = torch.zeros(tuple(shape.tolist()), device=self.device)
            dist.recv(tensor, src, group=self.pp_group)

            tensors[name] = tensor

        return tensors

    def _forward_step(self, microbatch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a forward step on the current microbatch."""
        # Move tensors to device
        microbatch = {k: v.to(self.device) for k, v in microbatch.items()}

        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(**microbatch)

        # Convert outputs to dict if it's not already
        if not isinstance(outputs, dict):
            # Handle various output types
            if hasattr(outputs, "to_dict"):
                outputs = outputs.to_dict()
            elif hasattr(outputs, "__dict__"):
                outputs = {
                    k: v
                    for k, v in outputs.__dict__.items()
                    if not k.startswith("_") and isinstance(v, torch.Tensor)
                }
            else:
                outputs = {"output": outputs}

        return outputs  # type: ignore[no-any-return]

    def _backward_step(
        self, microbatch_info: MicrobatchInfo
    ) -> Dict[str, torch.Tensor]:
        """Perform a backward step on the current microbatch."""
        # Get output from forward pass
        outputs = microbatch_info.forward_output
        if outputs is None:
            # Handle the case where no forward outputs were saved
            return {}

        # Get incoming gradients
        gradients = microbatch_info.gradient
        if gradients is None:
            # Handle the case where no gradients were received
            return {}

        # Enable autograd for backward pass
        with torch.set_grad_enabled(True):
            # Set gradients for output tensors
            for name, tensor in outputs.items():
                if name in gradients and tensor.requires_grad:
                    tensor.backward(gradients[name])

        # Collect input gradients to send to previous stage
        input_gradients = {}
        for name, tensor in microbatch_info.inputs.items():
            if hasattr(tensor, "grad") and tensor.grad is not None:
                input_gradients[name] = tensor.grad

        return input_gradients

    def train_batch(
        self,
        batch: Dict[str, torch.Tensor],
        loss_fn: Optional[Callable] = None,
    ) -> Optional[torch.Tensor]:
        """
        Process a batch using pipeline parallelism.

        Args:
            batch: Dictionary containing the input tensors.
            loss_fn: Optional loss function if this is the last stage.

        Returns:
            Loss tensor (on last stage) or None.
        """
        # Split batch into microbatches
        microbatches = self._split_batch(batch, self.num_microbatches)

        # Initialize tracking
        self.active_microbatches.clear()

        # Pipeline schedule: forward passes then backward passes
        # First process all forward passes
        for i, microbatch in enumerate(microbatches):
            # Create tracking info
            mb_size = microbatch.get("batch_size", 1)
            mb_info = MicrobatchInfo(
                id=i,
                batch_size=(
                    int(mb_size) if isinstance(mb_size, torch.Tensor) else mb_size
                ),
            )
            mb_info.inputs = microbatch

            # Process this microbatch
            self._process_microbatch_forward(mb_info)

            # Store for backward pass
            self.active_microbatches.append(mb_info)

        # Then process all backward passes in reverse order
        loss: Optional[torch.Tensor] = None
        while self.active_microbatches:
            mb_info = self.active_microbatches.pop()
            mb_info.stage = PipelineStage.BACKWARD

            # If this is the last stage, compute loss
            if self.is_last_stage and loss_fn is not None:
                outputs = mb_info.forward_output
                inputs = mb_info.inputs

                if outputs is not None:
                    # Compute loss
                    with torch.set_grad_enabled(True):
                        for name, tensor in outputs.items():
                            outputs[name] = tensor.detach().requires_grad_(True)

                        batch_loss = loss_fn(outputs, inputs)
                        batch_loss.backward()
                        mb_info.loss = batch_loss

                        # Collect gradients
                        mb_info.gradient = {}
                        for name, tensor in outputs.items():
                            if tensor.grad is not None:
                                mb_info.gradient[name] = tensor.grad

            # Process backward pass
            self._process_microbatch_backward(mb_info)

            # Accumulate loss from last stage
            if self.is_last_stage and mb_info.loss is not None:
                loss = mb_info.loss if loss is None else loss + mb_info.loss

        # Return average loss if on last stage
        if self.is_last_stage and loss is not None:
            return loss / self.num_microbatches

        return None

    def _split_batch(
        self, batch: Dict[str, torch.Tensor], num_chunks: int
    ) -> List[Dict[str, Any]]:
        """Split a batch into microbatches."""
        # Determine batch size from first tensor dimension
        sample_tensor = next(iter(batch.values()))
        batch_size = sample_tensor.size(0)

        # Calculate microbatch size
        microbatch_size = batch_size // num_chunks

        # Handle uneven division
        if batch_size % num_chunks != 0:
            # Use ceiling division to ensure all data is used
            microbatch_size = (batch_size + num_chunks - 1) // num_chunks
            # Adjust num_chunks accordingly
            num_chunks = (batch_size + microbatch_size - 1) // microbatch_size

        # Split each tensor in the batch
        microbatches = []
        for i in range(num_chunks):
            start_idx = i * microbatch_size
            end_idx = min(start_idx + microbatch_size, batch_size)

            microbatch = {}
            for name, tensor in batch.items():
                microbatch[name] = tensor[start_idx:end_idx]

            # Store the actual size
            microbatch["batch_size"] = end_idx - start_idx  # type: ignore[assignment]
            microbatches.append(microbatch)

        return microbatches

    def _process_microbatch_forward(self, mb_info: MicrobatchInfo) -> None:
        """Process a microbatch in the forward pass."""
        if self.is_first_stage:
            # First stage: use the provided inputs
            outputs = self._forward_step(mb_info.inputs)
            mb_info.forward_output = outputs

            # Send outputs to next stage
            if not self.is_last_stage:
                self._send_tensors(outputs, self.pp_rank + 1)
        elif self.is_last_stage:
            # Last stage: receive inputs from previous stage
            inputs = self._recv_tensors(self.pp_rank - 1)
            mb_info.inputs = inputs

            # Forward step
            outputs = self._forward_step(inputs)
            mb_info.forward_output = outputs
        else:
            # Middle stage: receive inputs, process, and send outputs
            inputs = self._recv_tensors(self.pp_rank - 1)
            mb_info.inputs = inputs

            # Forward step
            outputs = self._forward_step(inputs)
            mb_info.forward_output = outputs

            # Send to next stage
            self._send_tensors(outputs, self.pp_rank + 1)

    def _process_microbatch_backward(self, mb_info: MicrobatchInfo) -> None:
        """Process a microbatch in the backward pass."""
        if self.is_first_stage:
            # First stage: receive gradients, backward, no send
            if not self.is_last_stage:
                gradients = self._recv_tensors(self.pp_rank + 1)
                mb_info.gradient = gradients

            # Backward step (no need to send further back)
            self._backward_step(mb_info)
        elif self.is_last_stage:
            # Last stage: backward using loss gradients, send gradients back
            input_gradients = self._backward_step(mb_info)

            # Send gradients to previous stage
            self._send_tensors(input_gradients, self.pp_rank - 1)
        else:
            # Middle stage: receive gradients, backward, send gradients
            gradients = self._recv_tensors(self.pp_rank + 1)
            mb_info.gradient = gradients

            # Backward step
            input_gradients = self._backward_step(mb_info)

            # Send gradients to previous stage
            self._send_tensors(input_gradients, self.pp_rank - 1)
