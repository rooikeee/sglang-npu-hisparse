"""CPU-only contracts for HiSparse's NPU paged-attention metadata.

Can run directly with unittest; no SGLang server, torch_npu or compiled kernels
are imported. The real NPU LRU planner intentionally remains a placeholder.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

_ROOT = Path(__file__).resolve().parents[4]


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config = _load(
    "npu_hisparse_config", "python/sglang/srt/hardware_backend/npu/hisparse/config.py"
)
_layout = _load(
    "npu_hisparse_layout", "python/sglang/srt/hardware_backend/npu/hisparse/layout.py"
)


class TestNPUHiSparseAttention(unittest.TestCase):
    def _forward(self, decode):
        fake_npu = types.ModuleType("torch_npu")
        fake_npu.npu_sparse_flash_attention = lambda **kwargs: (
            torch.zeros_like(kwargs["query"]),
            None,
            None,
        )
        with patch.dict(sys.modules, {"torch_npu": fake_npu}):
            attention = _load(
                "npu_hisparse_attention",
                "python/sglang/srt/hardware_backend/npu/hisparse/attention.py",
            )
        logical_pages = torch.tensor([[1, 2]], dtype=torch.int32)
        original_pages = logical_pages.clone()
        mapping = torch.zeros(64, dtype=torch.int64)
        mapping[16] = 5 * 16
        mapping[32] = 7 * 16
        keys = torch.zeros(10, 16, 1, 8)
        rope = torch.zeros(10, 16, 1, 2)
        pool = types.SimpleNamespace(
            page_size=16,
            get_kv_buffer=lambda _: (keys, rope),
            translate_loc_to_hisparse_device=lambda ids: mapping[ids],
        )
        metadata = types.SimpleNamespace(
            block_tables=logical_pages, actual_seq_lengths_q=None
        )
        coordinator = types.SimpleNamespace(
            prepare_attention=lambda *args: (
                torch.tensor([[16, 3]], dtype=torch.int32),
                torch.tensor([[5, 7]], dtype=torch.int32),
                torch.tensor([17], dtype=torch.int32),
            )
        )
        batch = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: decode,
                is_extend_without_speculative=lambda: not decode,
            ),
            hisparse_coordinator=coordinator,
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([30]),
            extend_seq_lens=torch.tensor([1]),
        )
        backend = types.SimpleNamespace(
            token_to_kv_pool=pool, forward_metadata=metadata
        )
        layer = types.SimpleNamespace(layer_id=0, scaling=0.5)
        with patch.object(
            fake_npu,
            "npu_sparse_flash_attention",
            wraps=fake_npu.npu_sparse_flash_attention,
        ) as call:
            attention.forward_hisparse(
                backend,
                torch.zeros(1, 2, 8),
                None,
                None,
                layer,
                batch,
                False,
                torch.zeros(1, 2, 2),
                torch.zeros(1, 1, 2),
                torch.tensor([[4, 20]]),
            )
        torch.testing.assert_close(metadata.block_tables, original_pages)
        kwargs = call.call_args.kwargs
        self.assertIs(kwargs["key"], keys)
        self.assertIs(kwargs["key_rope"], rope)
        self.assertEqual(kwargs["layout_kv"], "PA_BSND")
        self.assertEqual(kwargs["block_table"].tolist(), [[5, 7]])
        return kwargs

    def test_decode_uses_cache_coordinates_without_causal_slot_mask(self):
        kwargs = self._forward(decode=True)
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertEqual(kwargs["sparse_indices"].tolist(), [[[16, 3]]])
        self.assertEqual(kwargs["actual_seq_lengths_kv"].tolist(), [17])

    def test_prefill_preserves_logical_indexer_table(self):
        kwargs = self._forward(decode=False)
        self.assertEqual(kwargs["sparse_mode"], 3)
        self.assertEqual(kwargs["sparse_indices"].tolist(), [[[4, 20]]])
        self.assertEqual(kwargs["actual_seq_lengths_kv"].tolist(), [30])


class TestNPUHiSparseLayout(unittest.TestCase):
    def setUp(self):
        self.page_size = 16
        # Physical pages are deliberately nonconsecutive. The second page is
        # the reserved newest-token page in a cache with C=16 hot slots.
        pages = torch.tensor([[0, 0], [5, 2], [7, 3]])
        self.cache_rows = (
            pages[:, :, None] * self.page_size + torch.arange(self.page_size)
        ).reshape(3, -1)
        self.cache_rows[0] = 0

    def test_reordered_requests_and_reserved_newest_token(self):
        requests = torch.tensor([2, 1])
        table = _layout.build_hot_cache_page_table(self.cache_rows, requests, 16)
        self.assertEqual(table.tolist(), [[7, 3], [5, 2]])
        cache_slots = torch.tensor([[0, 15, 16], [1, 7, 16]])
        actual = table.gather(1, cache_slots // 16) * 16 + cache_slots % 16
        expected = self.cache_rows[requests].gather(1, cache_slots)
        torch.testing.assert_close(actual.long(), expected)

    def test_invalid_request_ids_cannot_read_last_request(self):
        requests = torch.tensor([-1, 99, 1])
        table = _layout.build_hot_cache_page_table(self.cache_rows, requests, 16)
        self.assertEqual(table.tolist(), [[0, 0], [0, 0], [5, 2]])

    def test_paged_attention_matches_explicit_gather(self):
        # Independent dense attention reference for selected cache slots. This
        # checks addressing when a request's pages are physically fragmented.
        torch.manual_seed(9)
        kv = torch.randn(8 * 16, 8)
        query = torch.randn(2, 8)
        requests = torch.tensor([2, 1])
        slots = torch.tensor([[16, 3, 15], [8, 1, 16]])
        table = _layout.build_hot_cache_page_table(self.cache_rows, requests, 16)
        paged_addresses = table.gather(1, slots // 16) * 16 + slots % 16
        paged_kv = kv[paged_addresses.long()]
        gathered = kv[self.cache_rows[requests]][torch.arange(2)[:, None], slots]

        def attention(keys):
            logits = torch.einsum("bd,bkd->bk", query, keys)
            return torch.einsum("bk,bkd->bd", logits.softmax(-1), keys)

        torch.testing.assert_close(attention(paged_kv), attention(gathered))

    def test_topk_layouts_preserve_invalid_tail(self):
        tokens = torch.tensor([[12, 4, -1], [8, -1, -1]])
        for value in (tokens, tokens[:, None], tokens[:, None, None]):
            result = _layout.normalize_topk(value)
            self.assertEqual(result.dtype, torch.int32)
            self.assertEqual(result.tolist(), tokens.tolist())
        with self.assertRaises(ValueError):
            _layout.normalize_topk(torch.zeros(2, 3, 4))

    def test_reject_incompatible_page_cache_geometries(self):
        base = dict(
            page_size=128, device_buffer_size=4096, top_k=2048, host_to_device_ratio=2
        )
        _config.validate_geometry(**base)
        for change in (
            {"page_size": 1},
            {"page_size": 1040},
            {"device_buffer_size": 2050},
            {"device_buffer_size": 1024},
            {"device_buffer_size": 32768},
            {"host_to_device_ratio": 1.5},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                _config.validate_geometry(**(base | change))


if __name__ == "__main__":
    unittest.main()
