from typing import Dict, List, Tuple

from rosellm.master.block_manager import BlockSpaceManager
from rosellm.sequence import Sequence
from rosellm.sequence import SequenceGroup
from rosellm.sequence import SequenceStatus

class Scheduler:
    def __init__(
        self,
        controller: List,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        self.controller = controller
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        # Initialize the block manager.
        self.block_manager = BlockSpaceManager(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )
        # Serving sequence groups (FIFO).
        self.serving: List[SequenceGroup] = []
        # Mapping: group_id => num_steps.
        self.num_steps: Dict[int, int] = {}
        # Mapping: group_id => max_num_steps.
        self.max_num_steps: Dict[int, int] = {}
        # Mapping: group_id => stop_token_ids.
        self.stop_token_ids: Dict[int, List[int]] = {}

        # Swapped sequence groups (LIFO).
        self.swapped: List[SequenceGroup] = []
        
        # Pending sequence groups (FIFO).
        self.pending: List[SequenceGroup] = []

        # Maps src_block_number to dst_block_number.
        self.blocks_to_swap_in: Dict[int, int] = {}
        self.blocks_to_swap_out: Dict[int, int] = {}
        self.blocks_to_copy: Dict[int, int] = {}

    def _free_seq(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free(seq)
    
    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.seqs:
            seq.status = SequenceStatus.SERVING
        self.serving.append(seq_group)
        self.num_steps[seq_group.group_id] = 0
    
    def _append(self, seq_group: SequenceGroup) -> None:
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.FINISHED:
                continue
            ret = self.block_manager.append(seq)
            if ret is not None:
                src_block, dst_block = ret
                self.blocks_to_copy[src_block] = dst_block

    def _swap_in(self, seq_group: SequenceGroup) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        self.blocks_to_swap_in.update(mapping)
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.SWAPPED:
                seq.status = SequenceStatus.SERVING
        self.serving.append(seq_group)
    
    def _swap_out(self, seq_group: SequenceGroup) -> None:
        assert self.block_manager.can_swap_out(seq_group)
        mapping = self.block_manager.swap_out(seq_group)
        self.blocks_to_swap_out.update(mapping)
        for seq in seq_group.seqs:
            if seq.status == SequenceStatus.SERVING:
                seq.status = SequenceStatus.SWAPPED
        self.swapped.append(seq_group)
    
    def prepare(self) -> None:
        # 1. Prepare new slots for the serving sequences.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # That is, the most recently added sequence group 
        # is the first to be swapped out.
        victim_idx = len(self.serving) - 1
        for i, seq_group in enumerate(self.serving):
            if i > victim_idx:
                # The i-th sequence group has already been swapped out.
                break
            # OOM. Swap out the victim sequence groups.
            while not self.block_manager.can_append(seq_group):
                victim_seq_group = self.serving[victim_idx]
                self._swap_out(victim_seq_group)
                victim_idx -= 1
                if i > victim_idx:
                    # No other sequence groups can be swapped out.
                    break
            else:
                self._append(seq_group)
        self.serving = self.serving[:victim_idx + 1]
        # 2. Swap in the swapped sequences if possible.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # The swapped sequences are in LIFO order.
        for i, seq_group in enumerate(reversed(self.swapped)):
            if self.block_manager.can_swap_in(seq_group):
                self._swap_in(seq_group)
                self._append(seq_group)
            else:
                # OOM. Stop swapping.
                self.swapped = self.swapped[:len(self.swapped) - i]
                break
        else:
            # All swapped sequences are swapped in.
            self.swapped.clear()
        # 3. Join new sequences if possible.
        # NOTE: Here we implicitly assume FCFS scheduling.
        # TODO: Add a heuristic to control the maximum batch size.
        if not self.swapped:
            for i, seq_group in enumerate(self.pending):
                if self.block_manager.can_allocate(seq_group):
                    self._allocate(seq_group)
                else:
                    # TODO: Consider race condition.
                    self.pending = self.pending[i:]
                    break
        else:
            self.pending.clear()

    def step(self) -> None:
        assert self.blocks_to_swap_in is not None or self.blocks_to_swap_out is not None
        # Execute the first stage.
        self.controllers[0].execute_stage(
            self.blocks_to_swap_in.copy(),
            self.blocks_to_swap_out.copy(),
            self.blocks_to_copy.copy(),
        )
        self.blocks_to_swap_in.clear()
        self.blocks_to_swap_out.clear()
        self.blocks_to_copy.clear()

    def post_step(
        self,
        next_tokens: Dict[int, Tuple[int, int]],
    ) -> None:
        # Update the running sequences and free blocks.
        for seq_group in self.serving:
            group_id = seq_group.group_id
            self.num_steps[group_id] += 1
            stop_token_ids = self.stop_token_ids[group_id]
            for seq in seq_group.seqs:
                if seq.status == SequenceStatus.FINISHED:
                    continue
                parent_seq_id, next_token = next_tokens[seq.seq_id]
                if seq.seq_id != parent_seq_id:
                    # The sequence is a fork of the parent sequence
                    # through beam search.
                    # Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(parent_seq_id)
                    seq.logical_token_blocks = parent_seq.logical_token_blocks.copy()
                    self.block_manager.fork(parent_seq, seq)
                # Append a new token to the sequence.
                seq.append(next_token)
                # Check if the sequence has generated a stop token.
                if next_token in stop_token_ids:
                    self._free_seq(seq)
                    continue
                # Check if the sequence has reached the maximum number of steps.
                if self.num_steps[group_id] == self.max_num_steps[group_id]:
                    self._free_seq(seq)
                    continue
        # Update the serving status.
        serving: List[SequenceGroup] = []
        for seq_group in self.serving:
            if seq_group.num_seqs(status=SequenceStatus.FINISHED) == len(seq_group.seqs):
                del self.num_steps[seq_group.group_id]
                del self.max_num_steps[seq_group.group_id]
                del self.stop_token_ids[seq_group.group_id]
                # TODO: Return the seq_group to the client.
            else:
                serving.append(seq_group)
        self.serving = serving
