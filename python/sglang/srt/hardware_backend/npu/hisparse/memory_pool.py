"""Separate logical index-K capacity from the physical paged MLA hot cache."""

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.base_hisparse_memory_pool import (
    HiSparseDevicePoolMappingMixin,
)
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc


class NPUHiSparseTokenToKVPool(HiSparseDevicePoolMappingMixin, NPUMLATokenToKVPool):
    def __init__(self, *args, host_to_device_ratio=2, **kwargs):
        if kwargs.get("use_combined_kv_buffer") or kwargs.get("modelslim_cache_layout"):
            raise ValueError(
                "NPU HiSparse requires separate plain latent/RoPE KV buffers."
            )
        if kwargs["dtype"] not in (torch.bfloat16, torch.float16):
            raise ValueError("NPU HiSparse requires BF16/FP16 KV storage.")
        super().__init__(*args, **kwargs)
        self.host_to_device_ratio = host_to_device_ratio
        self.host_capacity_tokens = self.size
        self.bytes_per_token = (
            self.kv_lora_rank + self.qk_rope_head_dim
        ) * self.store_dtype.itemsize

    def translate_loc_from_full_to_compressed(self, indices):
        return indices

    def translate_loc_to_hisparse_device(self, indices):
        # Preserve -1 for allocator continuation checks. KV stores use page zero
        # as a padding sink instead; index-K always keeps the original logical ID.
        return self.full_to_hisparse_device_index_mapping[indices.long()]

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v):
        loc, _, _ = unwrap_write_loc(loc_info)
        physical = self.translate_loc_to_hisparse_device(loc).clamp_min(0)
        super().set_kv_buffer(layer, physical, cache_k, cache_v)

    def set_mla_kv_buffer(
        self, layer, loc_info, cache_k_nope, cache_k_rope, layer_id_override=None
    ):
        if layer_id_override is not None and layer_id_override != layer.layer_id:
            raise NotImplementedError(
                "NPU HiSparse does not support MLA layer overrides."
            )
        self.set_kv_buffer(layer, loc_info, cache_k_nope, cache_k_rope)

    def transfer_values_on_device(self, dst_indices, src_indices):
        # Used when the upstream allocator relocates physical hot-cache slots.
        # Clone before index_copy_ so overlapping source/destination sets are safe.
        src = src_indices.long()
        dst = dst_indices.long()
        for buffers in (self.k_buffer, self.v_buffer):
            for buffer in buffers:
                rows = buffer.view(-1, buffer.shape[-1])
                rows.index_copy_(0, dst, rows.index_select(0, src))
