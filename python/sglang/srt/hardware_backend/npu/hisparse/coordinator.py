"""Use native HiSparse scheduling/LRU metadata with Ascend storage and kernels."""

import torch
from sgl_kernel_npu.hisparse import copy_cache_planned, plan_hisparse_swap

from sglang.srt.hardware_backend.npu.hisparse.config import validate_geometry
from sglang.srt.hardware_backend.npu.hisparse.host_pool import NPUHiSparseHostPool
from sglang.srt.hardware_backend.npu.hisparse.layout import (
    build_hot_cache_page_table,
    normalize_topk,
)
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator


class NPUHiSparseCoordinator(HiSparseCoordinator):
    def __init__(self, **kwargs):
        if kwargs.get("num_draft_tokens", 0):
            raise NotImplementedError(
                "NPU HiSparse speculative swap is not implemented."
            )
        pool = kwargs["token_to_kv_pool_allocator"].get_hisparse_kvcache()
        validate_geometry(
            page_size=pool.page_size,
            device_buffer_size=kwargs["device_buffer_size"],
            top_k=kwargs["top_k"],
            host_to_device_ratio=kwargs["host_to_device_ratio"],
        )
        super().__init__(**kwargs)
        # The base returns physical locations. Paged SFA additionally needs
        # cache-row slot IDs (original sequence positions are no longer valid).
        self.top_k_cache_slots = torch.full_like(self.top_k_device_locs_buffer, -1)
        if not self.enable_prefetch:
            batch = self.req_to_device_buffer.shape[0]
            self._miss_src = torch.full(
                (batch, self.top_k), -1, dtype=torch.int64, device=self.device
            )
            self._miss_dst = torch.full(
                (batch, self.top_k), -1, dtype=torch.int32, device=self.device
            )
            self._miss_count = torch.zeros(batch, dtype=torch.int32, device=self.device)

    def _create_mla_host_pool(self, host_to_device_ratio):
        return NPUHiSparseHostPool(self.mem_pool_device, host_to_device_ratio)

    def _run_swap_in_kernel(
        self,
        req_pool_indices,
        compressed_seq_lens,
        top_k_result,
        layer_id,
        record_plan=False,
    ):
        # Planning is always separate from IO on NPU. When native prefetch is
        # enabled, the same plan is replayed for subsequent shared-index layers.
        del record_plan
        topk = normalize_topk(top_k_result)
        if topk.shape[1] != self.top_k:
            raise ValueError("NPU HiSparse Top-K width differs from coordinator.top_k.")
        batch = req_pool_indices.shape[0]
        local_layer = layer_id - self.mem_pool_device.start_layer
        plan_hisparse_swap(
            topk_tokens=topk,
            seq_lens=compressed_seq_lens,
            req_pool_indices=req_pool_indices,
            num_real_reqs=self.num_real_reqs,
            device_buffer_tokens=self.req_device_buffer_tokens[local_layer],
            device_buffer_locs=self.req_device_buffer_token_locs[local_layer],
            host_cache_locs=self.req_to_host_pool,
            lru_slots=self.lru_slots[local_layer],
            hot_buffer_size=self.device_buffer_size,
            topk_device_locs=self.top_k_device_locs_buffer[:batch, : self.top_k],
            topk_cache_slots=self.top_k_cache_slots[:batch, : self.top_k],
            miss_src=self._miss_src[:batch],
            miss_dst=self._miss_dst[:batch],
            miss_count=self._miss_count[:batch],
        )
        self._run_copy_only_kernel(batch, layer_id)
        return self.top_k_device_locs_buffer[:batch, : self.top_k]

    def _run_copy_only_kernel(self, num_reqs, skip_layer):
        if self.skip_io:
            raise ValueError("NPU HiSparse does not support the skip-IO debug mode.")
        layer = skip_layer - self.mem_pool_device.start_layer
        host = self.mem_pool_host
        device = self.mem_pool_device
        copy_cache_planned(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_k=host.k_buffer[layer],
            host_rope=host.rope_buffer[layer],
            host_k_ptr=host.k_device_ptrs[layer],
            host_rope_ptr=host.rope_device_ptrs[layer],
            device_k=device.k_buffer[layer].view(-1, device.kv_lora_rank),
            device_rope=device.v_buffer[layer].view(-1, device.qk_rope_head_dim),
        )

    def prepare_attention(self, req_pool_indices, seq_lens, topk, layer_id):
        topk = normalize_topk(topk)
        # Inherited dispatch performs anchor swap/plan, schedules skip-layer IO
        # on the prefetch stream, and inserts event waits before skip attention.
        self.swap_in_selected_pages(req_pool_indices, seq_lens, topk, layer_id)
        batch = req_pool_indices.shape[0]
        pages = build_hot_cache_page_table(
            self.req_to_device_buffer, req_pool_indices, self.page_size
        )
        slots = self.top_k_cache_slots[:batch, : self.top_k]
        valid_reqs = torch.arange(batch, device=seq_lens.device) < self.num_real_reqs[0]
        # seq_len constrains original Top-K in the planner. SFA sees cache-slot
        # coordinates instead: hot slots [0,C) plus newest-token slot C.
        lengths = torch.where(
            valid_reqs, seq_lens.clamp(max=self.device_buffer_size + 1), 0
        )
        return slots, pages, lengths.to(torch.int32)
