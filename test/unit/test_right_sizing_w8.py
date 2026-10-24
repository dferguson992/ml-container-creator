# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wave 8 (R1 / R2 / R2b / R10.1) — the right-sizing checks added to
templates/do/lib/python/right_sizing.py.

Configs: gemma-3-12b-it-config.json is the public config.json of the
google/gemma-3-12b-it architecture (unsloth mirror, the findings-doc model);
gemma-4-e4b-it-config.json is fresh-ai-ml's staged config (Wave 6.1 fixture).
The synthetic dense config below is hand-built so the KV formula can be checked
by hand. Instance catalogs are synthetic (behavior, not frozen SKUs).
"""
import json
import os
import re
import subprocess
import sys

import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PY = os.path.join(_REPO, "templates", "do", "lib", "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import instance_sizer  # noqa: E402
import right_sizing as rs  # noqa: E402
import serve_manifest  # noqa: E402
import target_manifest  # noqa: E402

_FIX = os.path.join(_REPO, "test", "fixtures", "right-sizing")
with open(os.path.join(_FIX, "gemma-3-12b-it-config.json"), encoding="utf-8") as _f:
    GEMMA3 = json.load(_f)
with open(os.path.join(_FIX, "gemma-4-e4b-it-config.json"), encoding="utf-8") as _f:
    GEMMA4 = json.load(_f)
GIB = float(1 << 30)

# A dense Llama-style config: 32 layers, 32 heads, 8 KV heads, hidden 4096
# (head_dim omitted on purpose: derived as 4096 / 32 = 128).
DENSE = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096,
         "intermediate_size": 14336, "num_hidden_layers": 32, "num_attention_heads": 32,
         "num_key_value_heads": 8, "max_position_embeddings": 8192, "vocab_size": 128256,
         "tie_word_embeddings": False, "torch_dtype": "bfloat16"}

CATALOG = {
    "test.gpu24x1": {"gpus": 1, "gpuMemoryGb": 24, "gpuArchitecture": "Ampere"},
    "test.gpu32x1": {"gpus": 1, "gpuMemoryGb": 32, "gpuArchitecture": "Blackwell"},
    "test.gpu48x1": {"gpus": 1, "gpuMemoryGb": 48, "gpuArchitecture": "Ada Lovelace"},
    "test.gpu32x2": {"gpus": 2, "gpuMemoryGb": 32, "gpuArchitecture": "Blackwell"},
    "test.gpu16x1": {"gpus": 1, "gpuMemoryGb": 16, "gpuArchitecture": "Turing"},
    "test.cpu": {"gpus": 0, "category": "cpu"},
}


def ctx(**kw):
    base = dict(engine="vllm", model_config=DENSE, dtype_bytes=2, weights_bytes=int(16e9),
                weights_source="measured", instance_type="test.gpu48x1", catalog=CATALOG,
                gpu_memory_gib=48.0, gpu_memory_utilization=0.9, tp=1, serving_gpus=1,
                max_model_len=8192, max_model_len_source="VLLM_MAX_MODEL_LEN",
                max_model_len_var="VLLM_MAX_MODEL_LEN", lora_enabled=False)
    base.update(kw)
    if "instance" not in kw:
        base["instance"] = CATALOG.get(base["instance_type"])
    return rs.SizingContext(**base)


def only(findings, check):
    out = [f for f in findings if f.check == check]
    assert out, f"no {check} finding"
    return out


# ── KV formula (hand-computed) ────────────────────────────────────────────────


def test_kv_per_token_dense_hand_computed():
    # 2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes = 131,072 B/token
    lay = rs.kv_layout(DENSE)
    assert rs.kv_cache_bytes(lay, 1, 2) == 2 * 32 * 8 * 128 * 2
    assert not lay["sliding"]


def test_kv_tp_shards_heads_and_replicates_when_tp_exceeds_kv_heads():
    lay = rs.kv_layout({**DENSE, "num_key_value_heads": 2})
    one = rs.kv_cache_bytes(lay, 1, 2, tp=1)
    assert rs.kv_cache_bytes(lay, 1, 2, tp=2) == one          # 2 heads split over 2 GPUs
    assert rs.kv_cache_bytes(lay, 1, 2, tp=4) == 2 * one      # each of 4 GPUs holds 1 head


def test_kv_gemma3_upper_is_384_kib_per_token_and_lower_is_window_aware():
    lay = rs.kv_layout(GEMMA3)  # nested text_config, sliding_window_pattern 6
    assert rs.kv_cache_bytes(lay, 1, 2) == 48 * 2 * 8 * 256 * 2 == 384 * 1024
    full = [x for x in lay["layers"] if x["kind"] == "full"]
    assert len(full) == 8 and lay["window"] == 1024
    lower = rs.kv_cache_bytes(lay, 131072, 2, window_aware=True)
    assert lower == 8 * 8192 * 131072 + 40 * 8192 * 1024


def test_kv_gemma4_excludes_kv_shared_layers_and_uses_global_head_dim():
    lay = rs.kv_layout(GEMMA4)
    # 42 layers, 18 KV-shared -> 24 own a cache; 4 full (global_head_dim 512), 20 sliding (256), 2 KV heads
    assert len(lay["layers"]) == 24 and lay["shared"] == 18
    assert rs.kv_cache_bytes(lay, 1, 2) == 4 * (2 * 2 * 512 * 2) + 20 * (2 * 2 * 256 * 2)


def test_kv_dtype_bytes():
    assert rs.kv_dtype_bytes("auto", 2) == 2
    assert rs.kv_dtype_bytes("fp8_e4m3", 2) == 1
    assert rs.kv_dtype_bytes("float32", 2) == 4


# ── context-memory ────────────────────────────────────────────────────────────


def gemma3_on_one_32gb_gpu(**kw):
    """The findings-doc failure: Gemma-3-12B, bf16, 131,072 tokens, ~24 GB weights, one 32 GB GPU."""
    args = dict(model_config=GEMMA3, weights_bytes=int(24e9), instance_type="test.gpu32x1", gpu_memory_gib=32.0,
                max_model_len=131072)
    args.update(kw)
    return ctx(**args)


def test_gemma3_findings_doc_case_blocks_with_computed_fixes():
    f = only(rs.check_context_memory(gemma3_on_one_32gb_gpu()), "context-memory")[0]
    assert f.severity == "block"
    d = f.data
    assert d["need_lower_bytes"] > d["usable_bytes"] and d["sliding"]
    # Full-attention-safe fix is in the ~8K range the findings doc computed; the
    # window-aware fix (the one that clears the block) is larger and both are named.
    assert 4096 <= d["max_fit_len_full_attention"] <= 12288
    assert d["max_fit_len_window_aware"] > d["max_fit_len_full_attention"]
    assert f"VLLM_MAX_MODEL_LEN={d['max_fit_len_window_aware']}" in f.fix
    # The fix actually fits: re-run at the proposed length -> no block.
    fixed = rs.check_context_memory(gemma3_on_one_32gb_gpu(max_model_len=d["max_fit_len_window_aware"]))
    assert fixed[0].severity != "block"
    fixed_full = rs.check_context_memory(gemma3_on_one_32gb_gpu(max_model_len=d["max_fit_len_full_attention"]))
    assert fixed_full[0].severity == "info"


def test_smallest_fitting_instance_fix_is_the_least_gpu_memory_that_fits():
    f = rs.check_context_memory(gemma3_on_one_32gb_gpu())[0]
    pick = f.data["smallest_fitting_instance"]
    assert pick in CATALOG and pick != "test.gpu32x1"
    need_ok = [n for n, e in CATALOG.items() if e.get("gpus")
               and rs.memory_terms(gemma3_on_one_32gb_gpu(), 131072, e["gpus"])["need_lower_bytes"]
               <= rs.usable_bytes(e["gpuMemoryGb"], e["gpus"], 0.9)]
    total = {n: CATALOG[n]["gpus"] * CATALOG[n]["gpuMemoryGb"] for n in need_ok}
    assert total[pick] == min(total.values())
    assert f"use {pick}" in f.fix


def test_only_upper_bound_over_warns_for_sliding_window_model():
    # 2 x 32 GB: window-aware fits, full-attention-on-every-layer would not.
    c = gemma3_on_one_32gb_gpu(instance_type="test.gpu32x2", tp=2, serving_gpus=2, weights_bytes=int(40e9))
    f = rs.check_context_memory(c)[0]
    assert f.data["need_lower_bytes"] < f.data["usable_bytes"] * rs.WARN_FROM
    assert f.data["need_upper_bytes"] > f.data["usable_bytes"]
    assert f.severity == "warn"
    # The fix names the conservative (full-attention) length, not the current one.
    assert "full attention on every layer" in f.fix
    assert f"VLLM_MAX_MODEL_LEN={f.data['max_fit_len_full_attention']}" in f.fix
    assert "smallest_fitting_instance" not in f.data or \
        CATALOG[f.data["smallest_fitting_instance"]]["gpus"] * CATALOG[f.data["smallest_fitting_instance"]]["gpuMemoryGb"] > 64


def test_margin_band_warns_dense():
    # weights chosen so the estimate lands between WARN_FROM and BLOCK_OVER
    usable = 48 * GIB * 0.9
    kv = rs.kv_cache_bytes(rs.kv_layout(DENSE), 8192, 2)
    w = int((usable * 0.95 - kv) / (1 + instance_sizer.OVERHEAD_FACTOR))
    f = rs.check_context_memory(ctx(weights_bytes=w))[0]
    assert rs.WARN_FROM <= f.data["ratio_upper"] <= rs.BLOCK_OVER
    assert f.severity == "warn"


def test_fits_is_info_dense():
    assert rs.check_context_memory(ctx())[0].severity == "info"


def test_quantized_model_is_capped_at_warn():
    blocked = rs.check_context_memory(gemma3_on_one_32gb_gpu())[0]
    capped = rs.check_context_memory(gemma3_on_one_32gb_gpu(quantization="awq"))[0]
    assert blocked.severity == "block" and capped.severity == "warn"
    assert "conservative" in capped.message


def test_missing_config_skips_never_blocks():
    out = rs.check_context_memory(ctx(model_config=None, missing={"model_config": "gated repo"}))
    assert [f.severity for f in out] == ["skip"] and "gated repo" in out[0].message
    assert rs.exit_code(out) == 0
    assert rs.check_context_memory(ctx(weights_bytes=None))[0].severity == "skip"
    assert rs.check_context_memory(ctx(instance_type="test.cpu"))[0].severity == "skip"
    assert rs.check_context_memory(ctx(instance_type="unknown.type", instance=None))[0].severity == "skip"


def test_lora_buffer_and_draft_join_the_memory_sum():
    base = rs.memory_terms(ctx(), 8192)
    with_lora = rs.memory_terms(ctx(lora_enabled=True, serving_rank_limit=64, max_loras=4), 8192)
    assert with_lora["lora_buffer_bytes"] > 0
    assert with_lora["need_upper_bytes"] == base["need_upper_bytes"] + with_lora["lora_buffer_bytes"]
    with_draft = rs.memory_terms(ctx(draft={"model": "d", "weights_bytes": int(1e9)}), 8192)
    assert with_draft["draft_weights_bytes"] == int(1e9 * (1 + instance_sizer.OVERHEAD_FACTOR))


def test_thresholds_are_overridable_by_env(monkeypatch):
    monkeypatch.setenv("MLCC_SIZING_BLOCK_OVER", "2.0")
    assert rs.threshold("BLOCK_OVER") == 2.0
    assert rs.check_context_memory(gemma3_on_one_32gb_gpu())[0].severity != "block"


# ── weights (OQ-8.9) ──────────────────────────────────────────────────────────


def test_weights_estimate_uses_instance_sizer_constants():
    est, info = rs.estimate_weights_bytes(DENSE, "bfloat16")
    assert info["bytes_per_param"] == instance_sizer.BYTES_PER_PARAM["bfloat16"]
    assert est == int(info["params"] * instance_sizer.BYTES_PER_PARAM["bfloat16"])
    # Llama-3-8B-shaped dims -> ~8B parameters
    assert 7.5e9 < info["params"] < 8.5e9


def test_weights_estimate_counts_multimodal_towers():
    _, info = rs.estimate_params(GEMMA3)
    assert info["towers"]["vision_config"] > 0
    f = rs.check_vision_weights(ctx(model_config=GEMMA3, weights_source="estimate"))[0]
    assert f.severity == "info" and "added to the weights estimate" in f.message
    assert "text-only" in rs.check_vision_weights(ctx())[0].message


def test_context_from_env_records_weights_source(tmp_path):
    staged = tmp_path / "model"
    staged.mkdir()
    (staged / "config.json").write_text(json.dumps(DENSE))
    with open(staged / "model.safetensors", "wb") as f:
        f.truncate(1234567)
    env = {"MODEL_SERVER": "vllm", "STAGED_MODEL_PATH": str(staged), "DEPLOYMENT_TARGET": "eks",
           "HP_INSTANCE_TYPE": _single_gpu_instances()[0][1]}
    c = rs.context_from_env(env, fetch_http=lambda *a, **k: None)
    assert c.weights_source == "measured" and c.weights_bytes == 1234567
    (staged / "model.safetensors").unlink()
    c = rs.context_from_env(env, fetch_http=lambda *a, **k: None)
    assert c.weights_source == "estimate" and c.weights_bytes > 0


# ── context-limit ─────────────────────────────────────────────────────────────


def test_context_limit_blocks_over_max_position_embeddings():
    f = rs.check_context_limit(ctx(max_model_len=16384))[0]
    assert f.severity == "block" and "VLLM_MAX_MODEL_LEN=8192" in f.fix
    assert rs.check_context_limit(ctx(max_model_len=4096))[0].severity == "info"


def test_context_limit_respects_rope_scaling_like_vllm():
    yarn = {**DENSE, "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192}}
    assert rs.derived_max_model_len(yarn)[0] == 32768
    llama3 = {**DENSE, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}
    assert rs.derived_max_model_len(llama3)[0] == 8192        # vLLM does not scale llama3
    assert rs.derived_max_model_len(GEMMA3)[0] == 131072      # nor Gemma-3


def test_context_limit_skip_and_unset():
    assert rs.check_context_limit(ctx(model_config=None))[0].severity == "skip"
    unset = rs.check_context_limit(ctx(max_model_len=None, max_model_len_source="model default"))[0]
    assert unset.severity == "info" and "8,192" in unset.message


# ── tp-divisibility ───────────────────────────────────────────────────────────


def test_tp_divisibility():
    assert rs.check_tp_divisibility(ctx(tp=1))[0].severity == "info"
    over = rs.check_tp_divisibility(ctx(tp=2, serving_gpus=1))
    assert any(f.severity == "block" and "exceeds" in f.message for f in over)
    heads = rs.check_tp_divisibility(ctx(tp=3, serving_gpus=4, instance={"gpus": 4, "gpuMemoryGb": 48}))
    assert any(f.severity == "block" and "num_attention_heads" in f.message for f in heads)
    assert rs.check_tp_divisibility(ctx(instance_type="", instance=None))[0].severity == "skip"


# ── dtype-arch ────────────────────────────────────────────────────────────────


def test_dtype_arch():
    assert rs.check_dtype_arch(ctx())[0].severity == "info"
    turing = rs.check_dtype_arch(ctx(instance_type="test.gpu16x1"))[0]
    assert turing.severity == "block" and "bf16" in turing.message
    ampere_fp8 = rs.check_dtype_arch(ctx(instance_type="test.gpu24x1", quantization="fp8"))
    assert [f.severity for f in ampere_fp8] == ["warn"]
    assert rs.check_dtype_arch(ctx(instance_type="test.gpu48x1", quantization="fp8"))[0].severity == "info"
    assert rs.check_dtype_arch(ctx(instance={"gpus": 1, "gpuMemoryGb": 8, "gpuArchitecture": "Mystery"}))[0].severity \
        == "skip"


def test_every_catalogued_gpu_architecture_is_mapped():
    """instances.json carries gpuArchitecture for every GPU entry (Task 1); each
    must map to a compute capability so dtype-arch is a real check, not a skip."""
    cat = rs.load_instance_catalog([os.path.join(_REPO, "servers", "lib", "catalogs", "instances.json")])
    archs = {str(e.get("gpuArchitecture")).lower() for e in cat.values() if e.get("gpus")}
    assert archs and archs <= set(rs.GPU_ARCH_COMPUTE_CAPABILITY)


# ── quant-engine / arch-engine-version ───────────────────────────────────────


def test_quant_engine():
    assert rs.check_quant_engine(ctx())[0].severity == "info"
    assert rs.check_quant_engine(ctx(quantization="awq"))[0].severity == "skip"   # vllm declares no list
    assert rs.check_quant_engine(ctx(quantization="awq", quantization_methods=["gptq"]))[0].severity == "warn"
    assert rs.check_quant_engine(ctx(quantization="awq", quantization_methods=["awq"]))[0].severity == "info"


def test_vllm_manifest_declares_no_quantization_list_so_the_check_skips():
    assert serve_manifest.quantization_methods("vllm") is None


def test_arch_engine_version():
    support = {"LlamaForCausalLM": "v0.4.0", "NewForCausalLM": "v9.0.0"}
    assert rs.check_arch_engine_version(ctx(arch_support=support, engine_version="0.31.0"))[0].severity == "info"
    new = {**DENSE, "architectures": ["NewForCausalLM"]}
    f = rs.check_arch_engine_version(ctx(model_config=new, arch_support=support, engine_version="0.31.0"))[0]
    assert f.severity == "block" and "v9.0.0" in f.message
    other = {**DENSE, "architectures": ["UnlistedForCausalLM"]}
    assert rs.check_arch_engine_version(ctx(model_config=other, arch_support=support,
                                            engine_version="0.31.0"))[0].severity == "skip"
    assert rs.check_arch_engine_version(ctx(arch_support=support, engine_version=None))[0].severity == "skip"


# ── spec-draft-memory / ic-packing / concurrency / disk / quota ──────────────


def test_spec_draft_memory():
    assert rs.check_spec_draft_memory(ctx())[0].severity == "info"
    assert rs.check_spec_draft_memory(ctx(draft={"model": "d", "missing": "offline"}))[0].severity == "skip"
    f = rs.check_spec_draft_memory(ctx(draft={"model": "d", "algorithm": "draft-model", "weights_bytes": int(2e9),
                                              "source": "estimate"}))[0]
    assert f.severity == "info" and "adds" in f.message


def test_ic_packing():
    assert rs.check_ic_packing(ctx())[0].severity == "skip"
    c = ctx(uses_inference_components=True, instance_type="test.gpu32x2",
            ic_confs=[{"_name": "default", "IC_GPU_COUNT": "1", "IC_COPY_COUNT": "2"}])
    assert rs.check_ic_packing(c)[0].severity == "info"
    c.ic_confs.append({"_name": "big", "IC_GPU_COUNT": "4", "IC_COPY_COUNT": "1"})
    sev = [f.severity for f in rs.check_ic_packing(c)]
    assert sev.count("block") == 2  # per-copy > instance GPUs, and the total


def test_concurrency_kv_is_info_and_reports_capacity():
    f = rs.check_concurrency_kv(ctx())[0]
    assert f.severity == "info" and f.data["kv_capacity_tokens_full_attention"] > 0
    g = rs.check_concurrency_kv(ctx(model_config=GEMMA3, weights_bytes=int(24e9), max_model_len=8192))[0]
    assert g.data["kv_capacity_tokens_window_aware"] > g.data["kv_capacity_tokens_full_attention"]
    assert rs.check_concurrency_kv(ctx(max_num_seqs=100000, max_num_seqs_var="VLLM_MAX_NUM_SEQS"))[0].severity == "info"
    assert rs.check_concurrency_kv(ctx(model_config=None))[0].severity == "skip"


def test_disk_model_size():
    assert rs.check_disk_model_size(ctx())[0].severity == "skip"
    small = rs.check_disk_model_size(ctx(volume_gb=10, volume_source="ENDPOINT_VOLUME_SIZE"))[0]
    assert small.severity == "block"
    assert rs.check_disk_model_size(ctx(volume_gb=100, volume_source="ENDPOINT_VOLUME_SIZE"))[0].severity == "info"


def test_quota():
    assert rs.check_quota(ctx())[0].severity == "skip"                                  # cluster capacity
    assert rs.check_quota(ctx(quota_usage="endpoint usage"))[0].severity == "skip"      # not --network
    zero = ctx(quota_usage="endpoint usage", network=True, quota={"name": "x for endpoint usage", "value": 0.0})
    assert rs.check_quota(zero)[0].severity == "warn"
    ok = ctx(quota_usage="endpoint usage", network=True, quota={"name": "x for endpoint usage", "value": 4.0})
    assert rs.check_quota(ok)[0].severity == "info"
    q, why = rs.lookup_quota("ml.x", "endpoint usage", {}, run=lambda cmd: (0, '[["ml.x for endpoint usage", 2.0, "L-1"]]'))
    assert q == {"name": "ml.x for endpoint usage", "value": 2.0, "code": "L-1"} and not why
    q, why = rs.lookup_quota("ml.x", "endpoint usage", {}, run=lambda cmd: (255, ""))
    assert q is None and why


# ── context derivation from targets.d / serve.d ──────────────────────────────


def test_sagemaker_ic_target_reads_ic_env_engine_vars_and_ic_gpu_count():
    env = {"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "realtime-inference", "INSTANCE_TYPE": "ml.g5.12xlarge",
           "HP_INSTANCE_TYPE": "ml.g7.12xlarge", "IC_GPU_COUNT": "2", "VLLM_MAX_MODEL_LEN": "1024",
           "IC_ENV_VLLM_MAX_MODEL_LEN": "4096", "IC_ENV_VLLM_TENSOR_PARALLEL_SIZE": "2", "ENABLE_LORA": "true"}
    c = rs.context_from_env(env, fetch_http=lambda *a, **k: None)
    assert c.max_model_len == 4096 and c.max_model_len_source == "IC_ENV_VLLM_MAX_MODEL_LEN"
    assert c.instance_type == "ml.g5.12xlarge"            # not a stale HP_INSTANCE_TYPE
    assert c.serving_gpus == 2 and c.tp == 2 and c.uses_inference_components
    assert c.lora_enabled is True and c.quota_usage == "endpoint usage"


def test_kubernetes_target_reads_bare_engine_vars_and_hp_gpu_count():
    env = {"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "hyperpod", "HP_INSTANCE_TYPE": "ml.g5.12xlarge",
           "HP_GPU_COUNT": "4", "VLLM_MAX_MODEL_LEN": "2048", "IC_ENV_VLLM_MAX_MODEL_LEN": "9999",
           "HP_LORA_ENABLED": "false"}
    c = rs.context_from_env(env, fetch_http=lambda *a, **k: None)
    assert c.target == "hyperpod-eks"
    assert c.max_model_len == 2048 and c.tp == 4 and c.lora_enabled is False and not c.quota_usage
    assert c.gpu_memory_utilization_source.startswith("serve.d/vllm")


def test_gpu_memory_utilization_comes_from_the_engine_manifest():
    sizing = serve_manifest.sizing_map("vllm")
    c = rs.context_from_env({"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "eks"}, fetch_http=lambda *a, **k: None)
    assert c.gpu_memory_utilization == sizing["gpu_memory_utilization"]["default"]
    c = rs.context_from_env({"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "eks",
                             "VLLM_" + sizing["gpu_memory_utilization"]["suffix"]: "0.8"},
                            fetch_http=lambda *a, **k: None)
    assert c.gpu_memory_utilization == 0.8


def test_hf_config_fetch_is_skipped_with_reason_when_offline():
    c = rs.context_from_env({"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "eks", "MODEL_NAME": "org/not-cached-xyz"},
                            fetch_http=lambda *a, **k: None)
    assert c.model_config is None and "Hugging Face" in c.missing["model_config"]


def test_lora_checks_skip_when_lora_is_off():
    out = rs.check_all(ctx(lora_enabled=False), ["lora-rank-fit", "lora-buffer-memory"])
    assert [f.severity for f in out] == ["skip", "skip"]


# ── derive, don't hardcode: drift guards ─────────────────────────────────────


def test_module_has_no_hardcoded_engine_setting_names():
    """Engine env-var names come from serve.d (dimension_map / sizing_map /
    capability_map), never literals in the module."""
    src = open(os.path.join(_PY, "right_sizing.py"), encoding="utf-8").read()
    names = set(serve_manifest.dimension_map("vllm").values())
    names |= {v["suffix"] for v in serve_manifest.sizing_map("vllm").values() if v.get("suffix")}
    for suffix in names:
        assert not re.search(r'["\'](VLLM_|IC_ENV_VLLM_)?' + suffix + r'["\']', src), suffix
        assert ("VLLM_" + suffix) not in src, suffix


def test_target_descriptor_sizing_fields_are_consistent():
    for t in target_manifest.list_targets():
        d = target_manifest.read_manifest(t)
        if d["family"] == "kubernetes":
            cc = d.get("cluster_contract") or {}
            assert cc, f"{t}: kubernetes targets declare their read-only cluster_contract"
            path = os.path.join(_REPO, "templates", "do", cc["script"])
            assert os.path.isfile(path), path
            body = open(path, encoding="utf-8").read()
            if path.endswith(".sh"):
                assert re.search(r"^" + re.escape(cc["entry"]) + r"\(\)", body, re.M)
            else:
                assert f'add_parser("{cc["entry"]}")' in body
            assert not d.get("service_quota_usage"), f"{t}: cluster capacity has no SageMaker quota"
        else:
            assert d.get("service_quota_usage"), f"{t}: SageMaker targets name their quota usage kind"
            assert "cluster_contract" not in d


def _single_gpu_instances():
    cat = rs.load_instance_catalog([os.path.join(_REPO, "servers", "lib", "catalogs", "instances.json")])
    one = sorted((e["gpuMemoryGb"], n) for n, e in cat.items() if e.get("gpus") == 1 and e.get("gpuMemoryGb"))
    return one[0], one[-1]


def test_cli_check_exit_codes_block_and_fit(tmp_path):
    (small_mem, small), (big_mem, big) = _single_gpu_instances()
    weights = int(small_mem * GIB)  # weights alone fill the smallest single-GPU instance
    if big_mem * 0.85 < weights * 1.15 / GIB:
        pytest.skip("catalog has no single-GPU instance large enough for the fit half of this test")
    staged = tmp_path / "m"
    staged.mkdir()
    (staged / "config.json").write_text(json.dumps(DENSE))
    with open(staged / "model.safetensors", "wb") as f:
        f.truncate(weights)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VLLM_", "TUNE_", "HP_", "IC_ENV_"))}
    env.update({"MODEL_SERVER": "vllm", "DEPLOYMENT_TARGET": "eks", "HP_INSTANCE_TYPE": small,
                "STAGED_MODEL_PATH": str(staged), "HP_LORA_ENABLED": "false"})
    script = os.path.join(_PY, "right_sizing.py")
    blocked = subprocess.run([sys.executable, script, "check", "--json"], capture_output=True, text=True, env=env)
    assert blocked.returncode == 2, blocked.stdout + blocked.stderr
    sev = {f["check"]: f["severity"] for f in json.loads(blocked.stdout)}
    assert sev["context-memory"] == "block" and set(sev.values()) <= set(rs.SEVERITIES)
    env["VLLM_MAX_MODEL_LEN"] = "1024"
    env["HP_INSTANCE_TYPE"] = big
    ok = subprocess.run([sys.executable, script, "check"], capture_output=True, text=True, env=env)
    assert ok.returncode in (0, 1), ok.stdout
