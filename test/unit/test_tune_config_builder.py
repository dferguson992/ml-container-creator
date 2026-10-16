# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""BL128 config-builder CORE — unit tests (tasks 2.2, and coverage for 1/2/3/4/5/7).

Feature: v2-w3-02-bl128

The CORE (`build_config`) is descriptor-source-AGNOSTIC: every test here drives it with
IN-MEMORY fabricated `hyperparameters` blocks (NOT a real tune.d technique), proving the
core has no tune.d coupling (Requirement 7.4). Covers control/tier derivation (R2.3),
progressive disclosure (R2), validation (R1.3), precedence + non-interactive parity
(R4.1), required-no-default fail-fast (R4.3), and unknown-knob rejection (R3.3).

No pyarrow / questionary needed — the core's prompter is injectable.
"""

import importlib.util
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB = os.path.join(_REPO_ROOT, "templates", "do", "lib", "python")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_LIB, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builder = _load("tune_config_builder")


# A fabricated descriptor fragment — NOT a real technique (proves agnosticism, R7.4).
HP = {
    "learning_rate": {"type": "number", "required": False, "default": 1e-5,
                      "min": 0, "max": 1, "control": "number_input", "tier": "primary"},
    "global_batch_size": {"type": "integer", "required": False, "default": 256,
                          "enum": [128, 256, 512], "control": "dropdown", "tier": "primary"},
    "warmup_ratio": {"type": "number", "required": False, "default": 0.1,
                     "min": 0, "max": 1, "control": "number_input", "tier": "advanced"},
    "debug_dump_path": {"type": "string", "required": False, "control": "text_input",
                        "tier": "tertiary"},
    # A leaner MTRL-shaped knob: NO control, NO tier, NO default.
    "advantage_method": {"type": "string", "required": True, "enum": ["grpo", "rloo"]},
}


# ── effective control / tier derivation (R2.3) ────────────────────────────────

def test_effective_control_uses_descriptor_when_present():
    assert builder.effective_control(HP["global_batch_size"]) == "dropdown"
    assert builder.effective_control(HP["learning_rate"]) == "number_input"


def test_effective_control_derived_when_absent():
    # enum → dropdown; numeric bounds → number_input; else text_input
    assert builder.effective_control({"type": "string", "enum": ["a", "b"]}) == "dropdown"
    assert builder.effective_control({"type": "number", "min": 0, "max": 1}) == "number_input"
    assert builder.effective_control({"type": "string"}) == "text_input"
    # the MTRL-shaped knob (enum, no control) derives dropdown
    assert builder.effective_control(HP["advantage_method"]) == "dropdown"


def test_effective_tier_defaults_to_primary_when_absent():
    assert builder.effective_tier(HP["warmup_ratio"]) == "advanced"
    assert builder.effective_tier(HP["advantage_method"]) == "primary"  # no tier → primary


# ── non-interactive: defaults applied, validated (R4.2) ───────────────────────

def test_non_interactive_applies_defaults():
    # advantage_method is required with no default → must be supplied; give it.
    out = builder.build_config(HP, {"advantage_method": "grpo"}, interactive=False)
    assert out["errors"] == []
    assert out["resolved"]["learning_rate"] == 1e-5
    assert out["resolved"]["global_batch_size"] == 256
    assert out["resolved"]["warmup_ratio"] == 0.1          # advanced default still applied
    assert "debug_dump_path" not in out["resolved"]        # tertiary, optional, no default → omitted
    assert out["resolved"]["advantage_method"] == "grpo"


def test_required_knob_with_no_value_fails_fast():
    out = builder.build_config(HP, {}, interactive=False)
    assert any("advantage_method" in e and "required" in e for e in out["errors"])


# ── precedence: default < supplied (R4.1) ─────────────────────────────────────

def test_supplied_overrides_default():
    out = builder.build_config(HP, {"global_batch_size": "512", "advantage_method": "rloo"},
                               interactive=False)
    assert out["errors"] == []
    assert out["resolved"]["global_batch_size"] == 512     # supplied wins, coerced to int


# ── validation (R1.3) ─────────────────────────────────────────────────────────

def test_enum_violation_named_with_choices():
    out = builder.build_config(HP, {"global_batch_size": "99", "advantage_method": "grpo"},
                               interactive=False)
    assert any("global_batch_size" in e and "one of" in e for e in out["errors"])


def test_range_violation_named_with_bound():
    out = builder.build_config(HP, {"learning_rate": "5", "advantage_method": "grpo"},
                               interactive=False)
    assert any("learning_rate" in e and "<=" in e for e in out["errors"])


def test_type_violation_named():
    out = builder.build_config(HP, {"global_batch_size": "notanint", "advantage_method": "grpo"},
                               interactive=False)
    assert any("global_batch_size" in e for e in out["errors"])


# ── unknown supplied knob rejected (R3.3) ─────────────────────────────────────

def test_unknown_supplied_knob_rejected_naming_valid_set():
    out = builder.build_config(HP, {"not_a_knob": "x", "advantage_method": "grpo"},
                               interactive=False)
    assert any("not_a_knob" in e and "Valid knobs" in e for e in out["errors"])


# ── progressive disclosure + interactive==non-interactive parity (R2, R4.1) ───

def test_interactive_prompts_primary_and_advanced_on_optin_not_tertiary():
    prompted = []

    def prompter(name, knob, control, default):
        prompted.append(name)
        # Return the default (simulating the user accepting it); required knob gets a value.
        return default if default is not None else "grpo"

    out = builder.build_config(HP, {}, interactive=True, include_advanced=True,
                               prompter=prompter)
    assert out["errors"] == []
    # primary + advanced prompted; tertiary NEVER prompted
    assert "learning_rate" in prompted and "global_batch_size" in prompted
    assert "warmup_ratio" in prompted              # advanced, because include_advanced
    assert "debug_dump_path" not in prompted       # tertiary never prompted
    assert "advantage_method" in prompted          # no tier → primary → prompted


def test_advanced_not_prompted_without_optin():
    prompted = []

    def prompter(name, knob, control, default):
        prompted.append(name)
        return default if default is not None else "grpo"

    builder.build_config(HP, {}, interactive=True, include_advanced=False, prompter=prompter)
    assert "warmup_ratio" not in prompted          # advanced, opt-in OFF
    # advanced knob still resolves to its default (R2.4), just not prompted
    out = builder.build_config(HP, {"advantage_method": "grpo"}, interactive=False)
    assert out["resolved"]["warmup_ratio"] == 0.1


def test_interactive_and_non_interactive_resolve_identically_for_same_inputs():
    # With everything supplied, the interactive path (prompter never consulted because
    # knobs are in `supplied`) resolves identically to non-interactive.
    supplied = {"learning_rate": "2e-5", "global_batch_size": "128",
                "warmup_ratio": "0.2", "advantage_method": "grpo"}

    def prompter(*a):
        raise AssertionError("should not prompt when the knob is supplied")

    interactive = builder.build_config(HP, dict(supplied), interactive=True,
                                       include_advanced=True, prompter=prompter)
    non_interactive = builder.build_config(HP, dict(supplied), interactive=False)
    assert interactive["resolved"] == non_interactive["resolved"]
    assert interactive["errors"] == non_interactive["errors"] == []


# ── a knob with no default, prompted empty, stays unset unless required ────────

def test_optional_no_default_prompted_empty_is_omitted():
    hp = {"opt": {"type": "string", "required": False, "tier": "primary"}}

    def prompter(name, knob, control, default):
        return ""   # user accepts "no value"

    out = builder.build_config(hp, {}, interactive=True, prompter=prompter)
    assert out["errors"] == []
    assert "opt" not in out["resolved"]


# ── tune.d adapter (task 2.1) — the ONLY tune.d-aware entry ───────────────────

def test_adapter_reads_tune_d_and_runs_core(tmp_path):
    # Point the adapter at a temp tune.d with a populated block (the committed
    # descriptors are pre-sync / empty). Proves the adapter reads tune.d and feeds the
    # core — while the core itself (tested above) stays tune.d-agnostic.
    import json
    d = tmp_path / "tune.d" / "rlvr"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "technique": "rlvr", "reward_source": "rule",
        "variants": {"lora": {
            "peft": "lora", "artifact_kind": "lora_adapter",
            "dataset_contract": {"format": "parquet", "schema": "verl-5field",
                                 "prompt_shape": "message-list",
                                 "reward_model_style": "rule", "verified": True},
            "hyperparameters": {
                "learning_rate": {"type": "number", "required": False, "default": 1e-5,
                                  "min": 0, "max": 1, "control": "number_input", "tier": "primary"},
            },
            "_synced": {"smtj": {"from": "s3://b/x_v1.0.0.json", "version": "1.0.0"}},
        }},
    }))
    out = builder.build_from_tune_descriptor(
        "rlvr", "lora", {}, interactive=False, tune_dir=str(tmp_path / "tune.d"))
    assert out["errors"] == []
    assert out["resolved"]["learning_rate"] == 1e-5


def test_adapter_empty_presync_descriptor_is_valid_empty_config():
    # A committed (pre-sync) descriptor has hyperparameters:{} → the adapter returns an
    # empty resolved config, NOT a crash (R: no descriptor is a valid empty knob set).
    out = builder.build_from_tune_descriptor("sft", "lora", {}, interactive=False)
    assert out["errors"] == []
    assert out["resolved"] == {}
