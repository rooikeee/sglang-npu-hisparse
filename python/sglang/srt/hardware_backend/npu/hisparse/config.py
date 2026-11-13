"""Supported geometry and combinations for the first Ascend HiSparse port."""

import json


def validate_geometry(*, page_size, device_buffer_size, top_k, host_to_device_ratio):
    if page_size is None or page_size <= 0 or page_size > 1024 or page_size % 16:
        raise ValueError(
            "NPU HiSparse requires a page size in [16, 1024], aligned to 16."
        )
    if top_k <= 0 or device_buffer_size < top_k:
        raise ValueError(
            "NPU HiSparse device_buffer_size must be >= the model index_topk."
        )
    if device_buffer_size % page_size:
        raise ValueError("NPU HiSparse device_buffer_size must be page-aligned.")
    # The upstream LRU slot table uses int16, including the newest-token slot.
    if device_buffer_size > 32767:
        raise ValueError("NPU HiSparse device_buffer_size must fit int16 LRU slots.")
    if not isinstance(host_to_device_ratio, int) or host_to_device_ratio < 1:
        raise ValueError(
            "NPU HiSparse host_to_device_ratio must be a positive integer."
        )


def validate_npu_hisparse(cfg, hf_config):
    from sglang.srt.configs.model_config import (
        get_dsa_index_kpool,
        get_dsa_index_kpool_compress,
        get_dsa_index_topk,
        is_deepseek_dsa,
    )
    from sglang.srt.environ import envs
    from sglang.srt.utils import get_bool_env_var

    if not is_deepseek_dsa(hf_config):
        raise ValueError(
            "NPU HiSparse currently supports DSA MLA models only (not V4/M3)."
        )
    if (
        getattr(hf_config, "kv_lora_rank", None) != 512
        or getattr(hf_config, "qk_rope_head_dim", None) != 64
    ):
        raise ValueError("NPU HiSparse SFA requires MLA latent/RoPE dimensions 512/64.")
    if cfg.kv_cache_dtype not in ("auto", "bfloat16", "float16"):
        raise ValueError("NPU HiSparse currently supports BF16/FP16 KV only.")
    if cfg.attention_backend != "ascend" or any(
        getattr(cfg, name, None) not in (None, "ascend")
        for name in ("prefill_attention_backend", "decode_attention_backend")
    ):
        raise ValueError(
            "NPU HiSparse requires the ascend prefill/decode attention backend."
        )
    if any(
        getattr(cfg, name, None) is not None
        for name in ("dsa_prefill_backend", "dsa_decode_backend")
    ):
        raise ValueError(
            "NPU HiSparse uses --attention-backend ascend; omit DSA backend overrides."
        )
    unsupported = {
        "PD disaggregation": cfg.disaggregation_mode != "null",
        "speculative decoding": (cfg.speculative_algorithm or "NONE").upper() != "NONE",
        "pipeline parallelism": getattr(cfg, "pp_size", 1) != 1,
        "decode context parallelism": getattr(cfg, "dcp_size", 1) != 1,
        "attention context parallelism": getattr(cfg, "attn_cp_size", 1) != 1,
        "prefill context parallelism": getattr(cfg, "enable_prefill_cp", False),
        "HiCache": cfg.enable_hierarchical_cache,
        "external KV linker": getattr(
            cfg, "enable_unified_cache_external_linker", False
        ),
        "skip-IO debug mode": envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get(),
        "legacy sparse KV offload": envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get(),
        "fused MLA preprocessing": get_bool_env_var("SGLANG_NPU_USE_MLAPO"),
        "NZ KV layout": get_bool_env_var("SGLANG_USE_FIA_NZ"),
        "compressed indexer": get_dsa_index_kpool(hf_config) != 1
        or get_dsa_index_kpool_compress(hf_config),
        "KV sharding": getattr(cfg, "enable_kv_cache_sharding", False),
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise ValueError("NPU HiSparse does not yet support: " + ", ".join(rejected))
    options = json.loads(cfg.hisparse_config or "{}")
    configured_top_k = options.get("top_k", 2048)
    validate_geometry(
        page_size=cfg.page_size,
        device_buffer_size=options.get("device_buffer_size", 2 * configured_top_k),
        top_k=get_dsa_index_topk(hf_config),
        host_to_device_ratio=options.get("host_to_device_ratio", 2),
    )
