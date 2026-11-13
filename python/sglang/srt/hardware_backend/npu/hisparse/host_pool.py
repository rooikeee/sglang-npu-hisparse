"""Registered, split latent/RoPE Host storage for native HiSparse lifecycle."""

from uuid import uuid4

import torch
from sgl_kernel_npu.hisparse import copy_kv_rows
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    free_shm_by_name,
)

from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin


class NPUHiSparseHostPool(HiSparseHostPoolMixin, HostKVCache):
    def __init__(self, device_pool, host_to_device_ratio):
        self.layer_num = self.target_layer_num = device_pool.layer_num
        self._transfer_device = device_pool.k_buffer.device
        self._device_id = self._transfer_device.index
        self._owned_mappings = []
        self._destroyed = False
        self._prefix = f"hisparse_{uuid4().hex}"
        self.kv_lora_rank = device_pool.kv_lora_rank
        self.qk_rope_head_dim = device_pool.qk_rope_head_dim
        self.token_stride_size = device_pool.bytes_per_token
        super().__init__(
            dtype=device_pool.store_dtype,
            device_capacity_tokens=device_pool.size,
            size_per_token=self.token_stride_size * self.layer_num,
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            host_to_device_ratio=host_to_device_ratio,
            host_size=0,
            page_size=device_pool.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )

    def get_size_per_token(self):
        return self.size_per_token

    def get_mapping_lengths(self):
        return [
            self.size * width * self.dtype.itemsize
            for _ in range(self.layer_num)
            for width in (self.kv_lora_rank, self.qk_rope_head_dim)
        ]

    def init_kv_buffer(self):
        self.k_buffer, self.rope_buffer = [], []
        self.k_device_ptrs, self.rope_device_ptrs = [], []
        try:
            for layer in range(self.layer_num):
                for label, width, tensors, pointers in (
                    ("k", self.kv_lora_rank, self.k_buffer, self.k_device_ptrs),
                    (
                        "rope",
                        self.qk_rope_head_dim,
                        self.rope_buffer,
                        self.rope_device_ptrs,
                    ),
                ):
                    name = f"{self._prefix}_{layer}_{label}"
                    tensor, _, device_ptr = create_shm_tensor(
                        (self.size, width), self.dtype, self._device_id, name
                    )
                    self._owned_mappings.append(name)
                    tensors.append(tensor)
                    pointers.append(device_ptr)
        except Exception:
            self.destroy()
            raise
        # Generic host-pool bookkeeping only; transfers below use both split
        # buffers explicitly. No concatenated duplicate of the KV data is kept.
        return self.k_buffer

    def _transfer_layer(
        self, device_pool, host_indices, device_indices, layer, to_host
    ):
        host_indices = host_indices.to(self._transfer_device, dtype=torch.int64)
        device_indices = device_indices.to(self._transfer_device, dtype=torch.int64)
        for host, ptr, device in (
            (
                self.k_buffer[layer],
                self.k_device_ptrs[layer],
                device_pool.k_buffer[layer],
            ),
            (
                self.rope_buffer[layer],
                self.rope_device_ptrs[layer],
                device_pool.v_buffer[layer],
            ),
        ):
            device = device.view(-1, device.shape[-1])
            if to_host:
                copy_kv_rows(device, host, device_indices, host_indices, dst_ptr=ptr)
            else:
                copy_kv_rows(host, device, host_indices, device_indices, src_ptr=ptr)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend="kernel"
    ):
        if io_backend != "kernel":
            raise ValueError("NPU HiSparse Host transfers use the UniDex kernel.")
        for layer in range(self.layer_num):
            self._transfer_layer(device_pool, host_indices, device_indices, layer, True)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend="kernel"
    ):
        if io_backend != "kernel":
            raise ValueError("NPU HiSparse Host transfers use the UniDex kernel.")
        self._transfer_layer(device_pool, host_indices, device_indices, layer_id, False)

    def get_data_page(self, index, flat=True):
        raise NotImplementedError("NPU HiSparse Host pool is not an L3 storage pool.")

    def get_dummy_flat_data_page(self):
        raise NotImplementedError("NPU HiSparse Host pool is not an L3 storage pool.")

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError("NPU HiSparse Host pool is not an L3 storage pool.")

    def destroy(self):
        # HiSparseCoordinator.destroy drains backup/prefetch streams first.
        # Do NOT call free_shm(): it frees every mapping in the process, including
        # mappings owned by unrelated pools or another coordinator.
        if self._destroyed:
            return
        while self._owned_mappings:
            free_shm_by_name(self._owned_mappings[-1], self._device_id)
            self._owned_mappings.pop()
        self.k_buffer = []
        self.rope_buffer = []
        self.kv_buffer = []
        self._destroyed = True
