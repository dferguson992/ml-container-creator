#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""right_sizing.py — centralized serving right-sizing checks (Wave 6.1, R3.1).

Purpose: ONE home for "does this serving configuration fit" decisions, so deploy,
         do/adapter and do/tune ask the same question the same way. Wave 8
         extends this module with more checks; Wave 6.1 registers two:

           lora-rank-fit       an adapter's rank (adapter_config.json `r` and
                               `rank_pattern`) vs the serving max_lora_rank
           lora-buffer-memory  the LoRA buffer vLLM preallocates for
                               max_loras x max_lora_rank, vs a cap of 10% of the
                               GPU memory left after weights (owner decision D3)

         It also owns the shared `effective_lora_rank` derivation both k8s
         deploy drivers call (R2.3): techniques DRIVE max_lora_rank —
             effective = snap_up(max(recorded adapter ranks, TUNE_LORA_RANK_*,
                                     manifest default), allowed)
         and an explicit <engine>_MAX_LORA_RANK wins (with a warning when it is
         below what recorded adapters need).

Shapes:
  Finding        one check result: check, severity (info|warn|block), message, fix
  SizingContext  the pure inputs every check reads (no I/O inside a check)
  CHECKS         the check registry (name -> callable(ctx) -> [Finding])

Where knowledge comes from (derive, don't hardcode — ADR-008):
  - rank default + allowed set, the rank / max-loras env-var names:
      serve.d/<engine>/manifest.json capability_map (via serve_manifest.py)
  - per-GPU memory: the instance catalog (.mlcc/instances.json gpuMemoryGb)
  - model dims + dtype: the staged model's config.json
  - weights: the staged model's *.safetensors sizes

Memory formula (vLLM 0.31 create_lora_weights, non-fully-sharded; see the
W6 follow-up investigation Q5):
    bytes/GPU = max_loras x rank x dtype_bytes x sum_layers sum_modules (in_local + out_local)
      ColumnParallel (q, k, v, gate, up): in_local = in,      out_local = out / TP
      RowParallel    (o, down):           in_local = in / TP, out_local = out
It is a slight upper bound (KV-shared layers are counted with k/v).

CLI (stdlib only; for bash callers):
    right_sizing.py check [--checks a,b] [--json]
    right_sizing.py effective-lora-rank [--rank-only] [--exclude-technique T] [--json]
    right_sizing.py adapter-rank <s3-uri|local dir|adapter_config.json|.tar.gz>
    right_sizing.py rank-fit --name N --rank R --limit L [--limit-kind live|effective] [--target T]
    right_sizing.py live-limit [--pods-json F] [--iec-json F] [--env-json F]   (R3.2 live limit)
  Exit codes: 0 = fits (info only), 1 = warnings, 2 = blocked (or usage error).
  Context comes from the environment (do/config is sourced by the caller).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import serve_manifest  # noqa: E402  (sibling module; same directory)

SEVERITIES = ("info", "warn", "block")
GIB = float(1 << 30)
# Owner decision D3 (2026-10-23): the LoRA buffer may use at most 10% of the GPU
# memory left after weights. A sizing policy, not engine knowledge.
LORA_BUFFER_BUDGET_FRACTION = 0.10
# vLLM's own gpu_memory_utilization default; overridden by <prefix>GPU_MEMORY_UTILIZATION.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "half": 2, "float32": 4, "float": 4}


# ── Shapes ─────────────────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One check result. severity: info (FYI) | warn (proceed) | block (refuse)."""

    check: str
    severity: str
    message: str
    fix: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
            "data": self.data,
        }


@dataclass
class AdapterRank:
    """A rank requirement and where it came from (adapter name / config var)."""

    name: str
    rank: int
    source: str = ""


@dataclass
class SizingContext:
    """Pure inputs for every check. Built from the environment by
    context_from_env(); tests construct it directly."""

    engine: str = "vllm"
    rank_env_var: str = ""
    rank_default: int | None = None
    rank_allowed: list[int] = field(default_factory=list)
    explicit_rank: int | None = None
    # Ranks recorded adapters need (adapter_config.json r / rank_pattern).
    adapters: list[AdapterRank] = field(default_factory=list)
    # Ranks recorded at tune submit (TUNE_LORA_RANK_<TECH>).
    tune_ranks: list[AdapterRank] = field(default_factory=list)
    # The limit lora-rank-fit / lora-buffer-memory evaluate. kind:
    #   "live"      the running deployment's limit (do/adapter pre-check)
    #   "effective" the limit the next deploy will apply
    serving_rank_limit: int | None = None
    limit_kind: str = "effective"
    # True when serving_rank_limit was auto-raised above the manifest default
    # (only an auto-raise is blocked by the memory cap; an explicit choice warns).
    rank_auto_raised: bool = False
    target: str = ""
    max_loras: int | None = None
    tp: int = 1
    model_config: dict | None = None
    dtype_bytes: int | None = None
    gpu_memory_gib: float | None = None
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    weights_bytes: int | None = None
    budget_fraction: float = LORA_BUFFER_BUDGET_FRACTION


# ── Check registry ─────────────────────────────────────────────────────────────

CHECKS: dict[str, Callable[[SizingContext], list[Finding]]] = {}


def register(name: str):
    """Decorator: register a check under `name` (Wave 8 adds more)."""

    def _wrap(fn):
        CHECKS[name] = fn
        return fn

    return _wrap


def run_checks(ctx: SizingContext, names: list[str] | None = None) -> list[Finding]:
    """Run the named checks (all when None), in registry order."""
    out: list[Finding] = []
    for name, fn in CHECKS.items():
        if names is None or name in names:
            out.extend(fn(ctx))
    return out


def exit_code(findings: list[Finding]) -> int:
    """0 = fits, 1 = warnings, 2 = blocked."""
    sev = {f.severity for f in findings}
    if "block" in sev:
        return 2
    if "warn" in sev:
        return 1
    return 0


# ── Rank helpers ───────────────────────────────────────────────────────────────


def adapter_config_rank(cfg: dict | None) -> int | None:
    """The largest rank a PEFT adapter_config.json needs: max(r, *rank_pattern).
    vLLM's PEFTHelper.validate_legal rejects max(r, *rank_pattern.values()) >
    max_lora_rank, so the pattern counts too. None when unreadable."""
    if not isinstance(cfg, dict):
        return None
    ranks = []
    r = cfg.get("r")
    if isinstance(r, int) and r > 0:
        ranks.append(r)
    pattern = cfg.get("rank_pattern")
    if isinstance(pattern, dict):
        ranks.extend(v for v in pattern.values() if isinstance(v, int) and v > 0)
    return max(ranks) if ranks else None


def snap_up(value: int, allowed: list[int]) -> int | None:
    """The smallest allowed value >= value; `value` itself when the engine accepts
    any integer (allowed == []); None when value exceeds every allowed value."""
    if not allowed:
        return value
    for a in sorted(allowed):
        if a >= value:
            return a
    return None


def _fmt_gib(nbytes: float) -> str:
    return f"{nbytes / GIB:.2f} GiB"


# ── Memory model ───────────────────────────────────────────────────────────────


def _text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) and tc.get("hidden_size") else cfg


def model_dtype_bytes(cfg: dict | None) -> int:
    """LoRA weights use the model dtype (vLLM lora_dtype=auto). Default bf16."""
    if not isinstance(cfg, dict):
        return 2
    t = _text_config(cfg)
    for src in (t, cfg):
        for key in ("dtype", "torch_dtype"):
            val = src.get(key)
            if isinstance(val, str) and val.lower() in _DTYPE_BYTES:
                return _DTYPE_BYTES[val.lower()]
    return 2


def lora_module_dims(cfg: dict) -> list[tuple[int, int, str]]:
    """Every LoRA-wrapped linear in the language model as (in, out, parallel),
    parallel ∈ {"col", "row"}. Reads hidden/intermediate sizes, heads, KV heads,
    head_dim and (Gemma-style) layer_types + global_head_dim from config.json.
    MoE models count every expert's MLP (upper bound). Raises KeyError/TypeError
    when the config lacks the basic dims."""
    t = _text_config(cfg)
    hidden = int(t["hidden_size"])
    layers = int(t["num_hidden_layers"])
    heads = int(t["num_attention_heads"])
    kv_heads = int(t.get("num_key_value_heads") or heads)
    head_dim = int(t.get("head_dim") or hidden // heads)
    inter = int(t.get("intermediate_size") or 4 * hidden)
    layer_types = t.get("layer_types") if isinstance(t.get("layer_types"), list) else []
    global_head_dim = t.get("global_head_dim")
    global_kv = t.get("num_global_key_value_heads")
    experts = int(t.get("num_experts") or t.get("n_routed_experts") or 0)
    expert_inter = int(t.get("moe_intermediate_size") or t.get("expert_intermediate_size") or inter)

    dims: list[tuple[int, int, str]] = []
    for i in range(layers):
        full = i < len(layer_types) and layer_types[i] == "full_attention"
        d = int(global_head_dim) if (full and global_head_dim) else head_dim
        k = int(global_kv) if (full and global_kv) else kv_heads
        dims += [
            (hidden, heads * d, "col"),  # q_proj
            (hidden, k * d, "col"),      # k_proj
            (hidden, k * d, "col"),      # v_proj
            (heads * d, hidden, "row"),  # o_proj
        ]
        mlp_inter, copies = (expert_inter, experts) if experts > 0 else (inter, 1)
        for _ in range(copies):
            dims += [
                (hidden, mlp_inter, "col"),  # gate_proj
                (hidden, mlp_inter, "col"),  # up_proj
                (mlp_inter, hidden, "row"),  # down_proj
            ]
    return dims


def lora_buffer_bytes_per_gpu(cfg: dict, rank: int, max_loras: int, tp: int,
                              dtype_bytes: int | None = None) -> float:
    """Per-GPU bytes vLLM preallocates for the LoRA A/B stacks (formula above)."""
    tp = max(int(tp or 1), 1)
    dtype_bytes = dtype_bytes or model_dtype_bytes(cfg)
    per_rank = 0.0
    for n_in, n_out, par in lora_module_dims(cfg):
        if par == "col":
            per_rank += n_in + n_out / tp
        else:
            per_rank += n_in / tp + n_out
    return float(max_loras) * float(rank) * float(dtype_bytes) * per_rank


def lora_buffer_budget_bytes(gpu_memory_gib: float, utilization: float, weights_bytes: int,
                             tp: int, fraction: float = LORA_BUFFER_BUDGET_FRACTION) -> float:
    """fraction x (per-GPU memory x utilization - per-GPU weights)."""
    tp = max(int(tp or 1), 1)
    left = gpu_memory_gib * GIB * utilization - float(weights_bytes) / tp
    return max(left, 0.0) * fraction


def _memory_inputs_missing(ctx: SizingContext) -> list[str]:
    missing = []
    if not isinstance(ctx.model_config, dict):
        missing.append("model config.json")
    if not ctx.gpu_memory_gib:
        missing.append("per-GPU memory (instance catalog)")
    if ctx.weights_bytes is None:
        missing.append("model weight size")
    if not ctx.max_loras:
        missing.append("max_loras")
    return missing


def max_rank_within_budget(ctx: SizingContext) -> int | None:
    """The largest allowed rank whose buffer fits the budget (None if none fit or
    inputs are missing). Only meaningful for engines with an allowed set."""
    if _memory_inputs_missing(ctx) or not ctx.rank_allowed:
        return None
    budget = lora_buffer_budget_bytes(ctx.gpu_memory_gib, ctx.gpu_memory_utilization,
                                      ctx.weights_bytes, ctx.tp, ctx.budget_fraction)
    best = None
    for r in sorted(ctx.rank_allowed):
        if lora_buffer_bytes_per_gpu(ctx.model_config, r, ctx.max_loras, ctx.tp, ctx.dtype_bytes) <= budget:
            best = r
    return best


# ── Checks ─────────────────────────────────────────────────────────────────────


def _rank_fix(ctx: SizingContext, need: int) -> str:
    snapped = snap_up(need, ctx.rank_allowed) or need
    var = ctx.rank_env_var or "the engine's MAX_LORA_RANK"
    target = ctx.target or "<target>"
    if target in ("realtime-inference", "managed-inference", "realtime"):
        return (
            f"Raise the base inference component's limit to {snapped}: set IC_ENV_{var}={snapped} in "
            f"do/config and recreate it: ./do/deploy --target realtime-inference --force-ic. "
            f"vLLM fixes max_lora_rank at start."
        )
    surge = " (add --surge-node on a single-node HyperPod instance group)" if target in ("hyperpod-eks", "hyperpod") else ""
    return (
        f"Raise the serving limit to {snapped}, then redeploy: set {var}={snapped} in do/config "
        f"(or unset it so deploy derives it from the recorded tune adapters) and run "
        f"./do/deploy --target {target} --force{surge}. vLLM fixes max_lora_rank at start, "
        f"so the pod must restart."
    )


@register("lora-rank-fit")
def check_lora_rank_fit(ctx: SizingContext) -> list[Finding]:
    """Each adapter's rank must be <= the serving max_lora_rank. Against the LIVE
    limit (do/adapter pre-check) an excess blocks; against the effective limit
    (an explicit override below need) it warns — the explicit value wins."""
    limit = ctx.serving_rank_limit
    if limit is None:
        return [Finding("lora-rank-fit", "info", "no serving max_lora_rank known; rank fit not checked")]
    out = []
    for a in ctx.adapters:
        if a.rank > limit:
            sev = "block" if ctx.limit_kind == "live" else "warn"
            where = "the running deployment's" if ctx.limit_kind == "live" else "the configured"
            out.append(Finding(
                "lora-rank-fit", sev,
                f"adapter '{a.name}' has LoRA rank {a.rank}, above {where} max_lora_rank {limit}; "
                f"vLLM rejects it (\"LoRA rank {a.rank} is greater than max_lora_rank {limit}\").",
                fix=_rank_fix(ctx, a.rank),
                data={"adapter": a.name, "rank": a.rank, "limit": limit, "source": a.source},
            ))
    return out


@register("lora-buffer-memory")
def check_lora_buffer_memory(ctx: SizingContext) -> list[Finding]:
    """The LoRA buffer at the serving rank must fit 10% of the post-weights GPU
    memory. Above the cap an AUTO-RAISED rank blocks (never silently capped);
    an explicit rank only warns (the user's choice wins)."""
    rank = ctx.serving_rank_limit
    missing = _memory_inputs_missing(ctx)
    if rank is None or missing:
        why = ", ".join(missing) if missing else "no serving rank"
        return [Finding("lora-buffer-memory", "info", f"LoRA buffer memory not checked (missing: {why})")]
    need = lora_buffer_bytes_per_gpu(ctx.model_config, rank, ctx.max_loras, ctx.tp, ctx.dtype_bytes)
    budget = lora_buffer_budget_bytes(ctx.gpu_memory_gib, ctx.gpu_memory_utilization,
                                      ctx.weights_bytes, ctx.tp, ctx.budget_fraction)
    data = {
        "rank": rank, "max_loras": ctx.max_loras, "tp": ctx.tp,
        "buffer_bytes_per_gpu": int(need), "budget_bytes_per_gpu": int(budget),
        "gpu_memory_gib": ctx.gpu_memory_gib, "utilization": ctx.gpu_memory_utilization,
        "weights_bytes_per_gpu": int(ctx.weights_bytes / max(ctx.tp, 1)),
    }
    summary = (
        f"LoRA buffer at max_lora_rank {rank} (max_loras {ctx.max_loras}, TP {ctx.tp}) is "
        f"{_fmt_gib(need)} per GPU; the cap is {int(ctx.budget_fraction * 100)}% of "
        f"({ctx.gpu_memory_gib:g} GiB x {ctx.gpu_memory_utilization:g} - {_fmt_gib(data['weights_bytes_per_gpu'])} weights) "
        f"= {_fmt_gib(budget)}"
    )
    if need <= budget:
        return [Finding("lora-buffer-memory", "info", summary + " — fits.", data=data)]
    fits = max_rank_within_budget(ctx)
    drivers = [a for a in ctx.adapters + ctx.tune_ranks if fits is None or a.rank > fits]
    names = ", ".join(f"{a.name} (r{a.rank})" for a in drivers) or "the configured rank"
    data["max_rank_within_budget"] = fits
    data["adapters"] = [a.name for a in drivers]
    if ctx.rank_auto_raised:
        return [Finding(
            "lora-buffer-memory", "block",
            summary + f" — over the cap. Raising max_lora_rank to {rank} for {names} would take "
            f"KV-cache memory beyond the {int(ctx.budget_fraction * 100)}% budget, so deploy refuses rather than cap it.",
            fix=(f"Serve these adapters on a larger GPU instance, or lower max_loras ({ctx.max_loras}), "
                 f"or set {ctx.rank_env_var or 'MAX_LORA_RANK'} explicitly to accept the cost"
                 + (f" (largest rank that fits: {fits})" if fits else "") + "."),
            data=data,
        )]
    return [Finding("lora-buffer-memory", "warn", summary + f" — over the cap (explicit rank; proceeding). Driven by: {names}.",
                    data=data)]


# ── effective_lora_rank (R2.3) ────────────────────────────────────────────────


def effective_lora_rank(ctx: SizingContext) -> dict:
    """Derive the serving max_lora_rank. Returns
    {rank, required, derived, explicit, default, allowed, findings[]}.

    required = max(recorded adapter ranks, TUNE_LORA_RANK_*), 0 when none
    derived  = snap_up(max(required, default), allowed)
    rank     = explicit when set (wins, with a warning if below required), else derived
    The memory cap is evaluated at `rank` (an auto-raise above the cap blocks).
    """
    findings: list[Finding] = []
    all_needs = ctx.adapters + ctx.tune_ranks
    required = max((a.rank for a in all_needs), default=0)
    default = ctx.rank_default or 0
    floor = max(required, default)
    derived = snap_up(floor, ctx.rank_allowed) if floor else None
    if floor and derived is None:
        top = [a for a in all_needs if a.rank > max(ctx.rank_allowed)]
        findings.append(Finding(
            "lora-rank-fit", "block",
            f"required LoRA rank {floor} ({', '.join(f'{a.name} r{a.rank}' for a in top)}) exceeds the "
            f"largest rank {ctx.engine} supports ({max(ctx.rank_allowed)}).",
            fix="Retune that technique with a supported lora_rank.",
            data={"required": floor, "allowed": ctx.rank_allowed},
        ))
    rank = derived
    auto_raised = bool(derived and default and derived > default)
    if ctx.explicit_rank is not None:
        rank = ctx.explicit_rank
        auto_raised = False
        if ctx.rank_allowed and ctx.explicit_rank not in ctx.rank_allowed:
            findings.append(Finding(
                "lora-rank-fit", "block",
                f"{ctx.rank_env_var}={ctx.explicit_rank} is not a value {ctx.engine} accepts "
                f"({', '.join(str(a) for a in ctx.rank_allowed)}); the engine would fail to start.",
                fix=f"Set {ctx.rank_env_var} to {snap_up(ctx.explicit_rank, ctx.rank_allowed) or max(ctx.rank_allowed)} "
                    f"or unset it so deploy derives it.",
            ))
        if required and ctx.explicit_rank < required:
            need = [a for a in all_needs if a.rank > ctx.explicit_rank]
            findings.append(Finding(
                "lora-rank-fit", "warn",
                f"explicit {ctx.rank_env_var}={ctx.explicit_rank} is below what recorded adapters need "
                f"({', '.join(f'{a.name} r{a.rank}' for a in need)}); those adapters will be refused at load.",
                fix=f"Unset {ctx.rank_env_var} (deploy derives {derived}) or raise it to {derived}.",
                data={"explicit": ctx.explicit_rank, "required": required, "derived": derived},
            ))
    if rank is not None:
        mem_ctx = SizingContext(**{**ctx.__dict__, "serving_rank_limit": rank, "rank_auto_raised": auto_raised,
                                   "limit_kind": "effective"})
        findings.extend(check_lora_buffer_memory(mem_ctx))
    return {
        "rank": rank,
        "required": required,
        "derived": derived,
        "explicit": ctx.explicit_rank,
        "default": ctx.rank_default,
        "allowed": ctx.rank_allowed,
        "auto_raised": auto_raised,
        "findings": [f.to_dict() for f in findings],
    }


# ── Environment / fetch layer (the only I/O) ──────────────────────────────────


def _aws_region_args(env: dict) -> list[str]:
    region = env.get("AWS_REGION") or ""
    return ["--region", region] if region else []


def fetch_bytes(uri: str, env: dict | None = None) -> bytes | None:
    """Read an S3 object (aws s3 cp <uri> -) or a local file. None on failure."""
    env = env if env is not None else dict(os.environ)
    if not uri:
        return None
    if not uri.startswith("s3://"):
        try:
            with open(uri, "rb") as f:
                return f.read()
        except OSError:
            return None
    try:
        proc = subprocess.run(["aws", "s3", "cp", uri, "-", *_aws_region_args(env)],
                              capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def read_adapter_config(uri: str, env: dict | None = None,
                        fetch: Callable[[str, dict | None], bytes | None] = fetch_bytes) -> dict | None:
    """adapter_config.json for an adapter at an S3 prefix / local dir / the json
    itself / a .tar.gz (SMAI adapters). None when unreadable."""
    if not uri:
        return None
    if uri.endswith(".tar.gz") or uri.endswith(".tgz"):
        blob = fetch(uri, env)
        if not blob:
            return None
        try:
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if os.path.basename(member.name) == "adapter_config.json":
                        fh = tf.extractfile(member)
                        return json.loads(fh.read()) if fh else None
        except (tarfile.TarError, ValueError, OSError):
            return None
        return None
    target = uri if uri.endswith(".json") else uri.rstrip("/") + "/adapter_config.json"
    blob = fetch(target, env)
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def fetch_weights_bytes(model_uri: str, env: dict | None = None) -> int | None:
    """Total *.safetensors bytes directly under the model prefix (S3 or local)."""
    env = env if env is not None else dict(os.environ)
    if not model_uri:
        return None
    if not model_uri.startswith("s3://"):
        try:
            return sum(os.path.getsize(os.path.join(model_uri, n)) for n in os.listdir(model_uri)
                       if n.endswith(".safetensors")) or None
        except OSError:
            return None
    no_scheme = model_uri[len("s3://"):]
    bucket, _, prefix = no_scheme.partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    try:
        proc = subprocess.run(
            ["aws", "s3api", "list-objects-v2", "--bucket", bucket, "--prefix", prefix,
             "--delimiter", "/", "--output", "json", *_aws_region_args(env)],
            capture_output=True, timeout=60, check=False, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        contents = json.loads(proc.stdout or "{}").get("Contents") or []
    except ValueError:
        return None
    total = sum(int(o.get("Size", 0)) for o in contents if str(o.get("Key", "")).endswith(".safetensors"))
    return total or None


def _instance_catalog_candidates() -> list[str]:
    project_root = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    repo_root = os.path.abspath(os.path.join(project_root, ".."))
    return [
        os.path.join(project_root, ".mlcc", "instances.json"),
        os.path.join(repo_root, "servers", "lib", "catalogs", "instances.json"),
        os.path.join(os.getcwd(), "servers", "lib", "catalogs", "instances.json"),
    ]


def gpu_memory_gib_for(instance_type: str, catalog_paths: list[str] | None = None) -> float | None:
    """Per-GPU memory from the instance catalog (gpuMemoryGb), or None."""
    if not instance_type:
        return None
    for path in catalog_paths or _instance_catalog_candidates():
        try:
            with open(path, encoding="utf-8") as f:
                cat = json.load(f)
        except (OSError, ValueError):
            continue
        entry = (cat.get("catalog") or cat).get(instance_type) if isinstance(cat, dict) else None
        if isinstance(entry, dict) and entry.get("gpuMemoryGb"):
            return float(entry["gpuMemoryGb"])
    return None


def _int_or_none(val) -> int | None:
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return None


def _float_or(val, default: float) -> float:
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return default


def recorded_adapter_ranks(env: dict, fetch: Callable = fetch_bytes) -> list[AdapterRank]:
    """Ranks of every recorded tune adapter (TUNE_ADAPTER_PATH_<TECH>[_<SLUG>]),
    read from each adapter_config.json (r + rank_pattern). De-duplicated by URI;
    an unreadable config is skipped (the TUNE_LORA_RANK_* record still counts)."""
    seen: dict[str, AdapterRank] = {}
    for key in sorted(env):
        if not key.startswith("TUNE_ADAPTER_PATH_"):
            continue
        uri = (env.get(key) or "").strip()
        if not uri or uri in seen:
            continue
        rank = adapter_config_rank(read_adapter_config(uri, env, fetch))
        if rank:
            seen[uri] = AdapterRank(name=key[len("TUNE_ADAPTER_PATH_"):].lower(), rank=rank, source=key)
    return list(seen.values())


def recorded_tune_ranks(env: dict) -> list[AdapterRank]:
    """TUNE_LORA_RANK_<TECH> values recorded by do/tune at submit (R2.2)."""
    out = []
    for key in sorted(env):
        if key.startswith("TUNE_LORA_RANK_"):
            r = _int_or_none(env.get(key))
            if r and r > 0:
                out.append(AdapterRank(name=key[len("TUNE_LORA_RANK_"):].lower(), rank=r, source=key))
    return out


def context_from_env(env: dict | None = None, *, rank_only: bool = False,
                     fetch: Callable = fetch_bytes,
                     fetch_weights: Callable = fetch_weights_bytes) -> SizingContext:
    """Build a SizingContext from do/config-style environment variables."""
    env = dict(os.environ) if env is None else env
    engine = env.get("MODEL_SERVER") or "vllm"
    try:
        caps = serve_manifest.capability_map(engine)
        defaults = serve_manifest.capability_defaults(engine)
        allowed = serve_manifest.lora_rank_allowed(engine)
        prefix = serve_manifest.env_var_prefix(engine)
        dims = serve_manifest.dimension_map(engine)
    except serve_manifest.ManifestError:
        caps, defaults, allowed, prefix, dims = {}, {}, [], "", {}
    rank_var = (caps.get("lora_max_rank") or {}).get("envVar", "")
    max_var = (caps.get("lora_max") or {}).get("envVar", "")
    tp_var = f"{prefix}{dims['tensor_parallel_degree']}" if prefix and dims.get("tensor_parallel_degree") else ""
    explicit = _int_or_none(env.get(rank_var)) if rank_var else None
    max_loras = (_int_or_none(env.get(max_var)) if max_var else None) or _int_or_none(defaults.get("lora_max"))
    tp = (_int_or_none(env.get(tp_var)) if tp_var else None) or _int_or_none(env.get("HP_GPU_COUNT")) or 1
    ctx = SizingContext(
        engine=engine,
        rank_env_var=rank_var,
        rank_default=_int_or_none(defaults.get("lora_max_rank")),
        rank_allowed=allowed,
        explicit_rank=explicit,
        tune_ranks=recorded_tune_ranks(env),
        target=env.get("DEPLOYMENT_TARGET", ""),
        max_loras=max_loras,
        tp=tp,
        gpu_memory_utilization=_float_or(env.get(f"{prefix}GPU_MEMORY_UTILIZATION") if prefix else None,
                                         DEFAULT_GPU_MEMORY_UTILIZATION),
    )
    ctx.adapters = recorded_adapter_ranks(env, fetch)
    if rank_only:
        return ctx
    model_uri = (env.get("STAGED_MODEL_PATH") or "").strip()
    if model_uri:
        blob = fetch(model_uri.rstrip("/") + "/config.json", env)
        try:
            ctx.model_config = json.loads(blob) if blob else None
        except ValueError:
            ctx.model_config = None
        ctx.weights_bytes = fetch_weights(model_uri, env)
    if ctx.model_config:
        ctx.dtype_bytes = model_dtype_bytes(ctx.model_config)
    ctx.gpu_memory_gib = gpu_memory_gib_for(env.get("HP_INSTANCE_TYPE") or env.get("INSTANCE_TYPE") or "")
    return ctx


def live_rank_limit(engine: str, pods: dict | None = None, iec: dict | None = None,
                    env_map: dict | None = None) -> tuple[int | None, str]:
    """The RUNNING deployment's max_lora_rank (R3.2 pre-check), from — in order —
    the serving pods' container env, the InferenceEndpointConfig worker env, or a
    SageMaker IC's container Environment map; else the engine manifest default
    (what the engine runs with when nothing overrides it). Returns (limit, source)."""
    try:
        var = (serve_manifest.capability_map(engine).get("lora_max_rank") or {}).get("envVar", "")
        default = serve_manifest.lora_rank_default(engine)
    except serve_manifest.ManifestError:
        var, default = "", None
    if var:
        for pod in (pods or {}).get("items") or []:
            for c in (pod.get("spec") or {}).get("containers") or []:
                for e in c.get("env") or []:
                    if e.get("name") == var and _int_or_none(e.get("value")):
                        return _int_or_none(e.get("value")), "pod env"
        for e in (((iec or {}).get("spec") or {}).get("worker") or {}).get("environmentVariables") or []:
            if e.get("name") == var and _int_or_none(e.get("value")):
                return _int_or_none(e.get("value")), "InferenceEndpointConfig env"
        if isinstance(env_map, dict) and _int_or_none(env_map.get(var)):
            return _int_or_none(env_map.get(var)), "inference component env"
    return default, "engine default (no override in the deployment)"


def _load_json_file(path: str):
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────


def _print_findings(findings: list[dict], stream) -> None:
    icon = {"info": "ℹ️ ", "warn": "⚠️ ", "block": "❌"}
    for f in findings:
        print(f"   {icon.get(f['severity'], '')} [{f['check']}] {f['message']}", file=stream)
        if f.get("fix"):
            print(f"      Fix: {f['fix']}", file=stream)


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="right_sizing.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_check = sub.add_parser("check")
    p_check.add_argument("--checks", default="")
    p_check.add_argument("--json", action="store_true")
    p_eff = sub.add_parser("effective-lora-rank")
    p_eff.add_argument("--rank-only", action="store_true",
                       help="skip the memory inputs (no model config/weights reads)")
    p_eff.add_argument("--json", action="store_true")
    p_eff.add_argument("--exclude-technique", default="",
                       help="ignore TUNE_LORA_RANK_<T> / TUNE_ADAPTER_PATH_<T>* (the limit BEFORE this tune)")
    p_ar = sub.add_parser("adapter-rank")
    p_ar.add_argument("uri")
    p_live = sub.add_parser("live-limit")
    p_live.add_argument("--pods-json", default="")
    p_live.add_argument("--iec-json", default="")
    p_live.add_argument("--env-json", default="")
    p_fit = sub.add_parser("rank-fit")
    p_fit.add_argument("--name", required=True)
    p_fit.add_argument("--rank", type=int, required=True)
    p_fit.add_argument("--limit", type=int, required=True)
    p_fit.add_argument("--limit-kind", default="live", choices=["live", "effective"])
    p_fit.add_argument("--target", default="")
    args = parser.parse_args(argv)

    if args.cmd == "adapter-rank":
        rank = adapter_config_rank(read_adapter_config(args.uri))
        print("" if rank is None else rank)
        return 0

    if args.cmd == "live-limit":
        limit, source = live_rank_limit(os.environ.get("MODEL_SERVER") or "vllm",
                                        _load_json_file(args.pods_json), _load_json_file(args.iec_json),
                                        _load_json_file(args.env_json))
        print(json.dumps({"limit": limit, "source": source}))
        return 0

    if args.cmd == "rank-fit":
        ctx = context_from_env(rank_only=True, fetch=lambda _u, _e=None: None)
        ctx.adapters = [AdapterRank(args.name, args.rank, "adapter_config.json")]
        ctx.serving_rank_limit = args.limit
        ctx.limit_kind = args.limit_kind
        ctx.target = args.target or ctx.target
        findings = [f.to_dict() for f in check_lora_rank_fit(ctx)]
        _print_findings([f for f in findings if f["severity"] != "info"], sys.stdout)
        return exit_code([Finding(**f) for f in findings])

    if args.cmd == "effective-lora-rank":
        env = dict(os.environ)
        if args.exclude_technique:
            tech = args.exclude_technique.upper()
            env = {k: v for k, v in env.items()
                   if k != f"TUNE_LORA_RANK_{tech}" and k != f"TUNE_ADAPTER_PATH_{tech}"
                   and not k.startswith(f"TUNE_ADAPTER_PATH_{tech}_")}
        ctx = context_from_env(env, rank_only=args.rank_only)
        result = effective_lora_rank(ctx)
        _print_findings([f for f in result["findings"] if f["severity"] != "info"
                         or os.environ.get("MLCC_SIZING_VERBOSE")], sys.stderr)
        print(json.dumps(result))
        return exit_code([Finding(**f) for f in result["findings"]])

    # check
    ctx = context_from_env()
    eff = effective_lora_rank(ctx)
    ctx.serving_rank_limit = eff["rank"]
    ctx.rank_auto_raised = eff["auto_raised"]
    names = [n.strip() for n in args.checks.split(",") if n.strip()] or None
    findings = run_checks(ctx, names)
    if args.json:
        print(json.dumps([f.to_dict() for f in findings]))
    else:
        _print_findings([f.to_dict() for f in findings], sys.stdout)
    return exit_code(findings)


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
