"""Metadata-only conversion from a HiSparse hot-cache row to paged SFA input."""

import torch


def build_hot_cache_page_table(req_to_device_buffer, req_pool_indices, page_size):
    """Map cache-row pages to physical pages, without gathering any KV bytes.

    Upstream HiSparse allocates each request's cache in whole pages and keeps
    tokens within each page consecutive. Hot-slot i therefore addresses
    block_table[i // page_size] * page_size + i % page_size. Page zero is a
    permanent padding sink; padded request IDs must never index arbitrary rows.
    """
    valid = (req_pool_indices >= 0) & (req_pool_indices < req_to_device_buffer.shape[0])
    safe_reqs = req_pool_indices.clamp(0, req_to_device_buffer.shape[0] - 1).long()
    page_starts = req_to_device_buffer[safe_reqs, ::page_size]
    return (
        torch.where(valid[:, None], page_starts // page_size, 0)
        .to(torch.int32)
        .contiguous()
    )


def normalize_topk(topk):
    if topk.ndim == 2:
        return topk.to(torch.int32).contiguous()
    if topk.ndim == 3 and topk.shape[1] == 1:
        return topk[:, 0].to(torch.int32).contiguous()
    if topk.ndim == 4 and topk.shape[1:3] == (1, 1):
        return topk[:, 0, 0].to(torch.int32).contiguous()
    raise ValueError(
        "NPU HiSparse expects single-step Top-K [B,K], [B,1,K] or [B,1,1,K]."
    )
