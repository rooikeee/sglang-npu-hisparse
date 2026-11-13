"""Read persistent paged hot-cache KV directly, without a selected-KV gather."""

import torch
import torch_npu


def forward_hisparse(
    backend,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache,
    q_rope,
    k_rope,
    topk_indices,
):
    del v
    pool = backend.token_to_kv_pool
    if q_rope is None or k_rope is None:
        raise ValueError("NPU HiSparse requires separate MLA RoPE tensors.")
    if save_kv_cache:
        # The pool remaps only the attention KV store. The indexer has already
        # stored index-K at the original allocator-logical out_cache_loc.
        pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, k_rope)
    k_cache, rope_cache = pool.get_kv_buffer(layer.layer_id)
    metadata = backend.forward_metadata
    if forward_batch.forward_mode.is_decode():
        coordinator = forward_batch.hisparse_coordinator
        if coordinator is None:
            raise RuntimeError("NPU HiSparse decode is missing its coordinator.")
        indices, block_table, kv_lengths = coordinator.prepare_attention(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            topk_indices,
            layer.layer_id,
        )
        q_lengths = torch.arange(1, q.shape[0] + 1, dtype=torch.int32, device=q.device)
        # Slots are ordered by cache residency, NOT chronological token position.
        # Causality has already been enforced on the model's selected logical IDs
        # by the planner; a causal mask on slot IDs would be incorrect here.
        sparse_mode = 0
    elif forward_batch.forward_mode.is_extend_without_speculative():
        indices = topk_indices
        # Indexer metadata must remain in logical space. Build a separate physical
        # page table for attention; never overwrite metadata.block_tables.
        logical_starts = metadata.block_tables.long() * pool.page_size
        block_table = (
            (pool.translate_loc_to_hisparse_device(logical_starts) // pool.page_size)
            .to(torch.int32)
            .contiguous()
        )
        q_lengths = metadata.actual_seq_lengths_q
        if q_lengths is None:
            q_lengths = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
        kv_lengths = forward_batch.seq_lens
        sparse_mode = 3
    else:
        raise NotImplementedError("NPU HiSparse supports ordinary prefill/decode only.")
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    result, _, _ = torch_npu.npu_sparse_flash_attention(
        query=q.contiguous(),
        key=k_cache,
        value=k_cache,
        query_rope=q_rope.contiguous(),
        key_rope=rope_cache,
        sparse_indices=indices.to(torch.int32).contiguous(),
        scale_value=layer.scaling,
        actual_seq_lengths_query=q_lengths.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=kv_lengths.to(device=q.device, dtype=torch.int32),
        block_table=block_table,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=sparse_mode,
        attention_mode=2,
        return_softmax_lse=False,
    )
    return result
