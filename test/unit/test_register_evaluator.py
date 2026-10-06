"""Unit tests for register_evaluator (BL117 native AI Registry evaluators).

All offline: the sagemaker.ai_registry SDK and the role/domain resolution are
mocked, so behavior — not any AWS state — is asserted.

Validates: Req 1.1/1.2 (REWARD_FUNCTION ARN path), Req 1.3 (name budget),
Req 2.1 (REWARD_PROMPT), Req 1.5/7 (region/SDK skip is a clear not-supported).
"""
import sys
import types

import pytest

import register_evaluator as re_mod


# ── name budget (Req 1.3) ──────────────────────────────────────────────────────

class TestNameBudget:
    def test_short_name_ok(self):
        ok, reason = re_mod.check_name_budget("math-reward")
        assert ok is True
        assert reason == ""

    def test_long_name_rejected_with_reason(self):
        ok, reason = re_mod.check_name_budget("x" * 40)
        assert ok is False
        assert "64-char" in reason or "too long" in reason

    def test_budget_is_derived_not_hardcoded(self):
        # The budget is derived from the Lambda prefix + timestamp reserve.
        name_at_budget = "x" * re_mod.EVALUATOR_NAME_BUDGET
        assert re_mod.check_name_budget(name_at_budget)[0] is True
        assert re_mod.check_name_budget(name_at_budget + "x")[0] is False


# ── native create: fake SDK (both types) ───────────────────────────────────────

def _install_fake_evaluator_sdk(created):
    """Install fake sagemaker.ai_registry.{air_constants,evaluator} modules."""
    air = types.ModuleType("sagemaker.ai_registry.air_constants")
    air.REWARD_FUNCTION = "REWARD_FUNCTION"
    air.REWARD_PROMPT = "REWARD_PROMPT"

    ev_mod = types.ModuleType("sagemaker.ai_registry.evaluator")

    class Evaluator:
        def __init__(self, **kw):
            self.arn = "arn:aws:sagemaker:us-west-2:1:hub-content/H/JsonDoc/x/1.0.0"
            self.version = "1.0.0"
            self.status = "Available"
            self.type = ("RewardPrompt" if kw.get("type") == "REWARD_PROMPT"
                         else "RewardFunction")
            self.reference = kw.get("source")

        @staticmethod
        def create(**kwargs):
            created.update(kwargs)
            return Evaluator(**kwargs)

    ev_mod.Evaluator = Evaluator

    saved = {k: sys.modules.get(k) for k in (
        "sagemaker", "sagemaker.ai_registry",
        "sagemaker.ai_registry.air_constants", "sagemaker.ai_registry.evaluator",
    )}
    sys.modules["sagemaker"] = sys.modules.get("sagemaker") or types.ModuleType("sagemaker")
    sys.modules["sagemaker.ai_registry"] = types.ModuleType("sagemaker.ai_registry")
    sys.modules["sagemaker.ai_registry.air_constants"] = air
    sys.modules["sagemaker.ai_registry.evaluator"] = ev_mod

    def cleanup():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return cleanup


class TestCreateEvaluatorNative:
    def test_reward_function_passes_type_and_source(self, monkeypatch):
        created = {}
        cleanup = _install_fake_evaluator_sdk(created)
        monkeypatch.setattr(re_mod, "check_ai_registry_available", lambda: True, raising=False)
        # check_ai_registry_available lives on ai_registry_native; patch there.
        import ai_registry_native
        monkeypatch.setattr(ai_registry_native, "check_ai_registry_available", lambda: True)
        try:
            info = re_mod._create_evaluator_native(
                name="rf", source="arn:aws:lambda:us-west-2:1:function:rf",
                evaluator_type="reward_function", region="us-west-2",
                role="arn:role", domain_id="d-1",
            )
        finally:
            cleanup()
        assert created["type"] == "REWARD_FUNCTION"
        assert created["source"] == "arn:aws:lambda:us-west-2:1:function:rf"
        assert created["role"] == "arn:role"
        assert created["domain_id"] == "d-1"
        assert info["type"] == "RewardFunction"
        assert info["domain_tagged"] is True

    def test_reward_prompt_passes_prompt_type(self, monkeypatch):
        created = {}
        cleanup = _install_fake_evaluator_sdk(created)
        import ai_registry_native
        monkeypatch.setattr(ai_registry_native, "check_ai_registry_available", lambda: True)
        try:
            info = re_mod._create_evaluator_native(
                name="jp", source="s3://b/judge.txt",
                evaluator_type="reward_prompt", region="us-east-1",
            )
        finally:
            cleanup()
        assert created["type"] == "REWARD_PROMPT"
        assert info["type"] == "RewardPrompt"
        assert "domain_id" not in created
        assert info["domain_tagged"] is False

    def test_unsupported_region_skips(self):
        import ai_registry_native
        with pytest.raises(ai_registry_native.NativeRegistrySkipped) as ei:
            re_mod._create_evaluator_native(
                name="rf", source="arn:...", evaluator_type="reward_function",
                region="eu-central-1",
            )
        assert "does not support" in str(ei.value)

    def test_missing_sdk_skips_with_upgrade_message(self, monkeypatch):
        import ai_registry_native
        monkeypatch.setattr(ai_registry_native, "check_ai_registry_available", lambda: False)
        with pytest.raises(ai_registry_native.NativeRegistrySkipped) as ei:
            re_mod._create_evaluator_native(
                name="rf", source="arn:...", evaluator_type="reward_function",
                region="us-west-2",
            )
        assert "upgrade sagemaker" in str(ei.value)
