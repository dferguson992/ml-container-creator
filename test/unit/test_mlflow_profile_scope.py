# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for the MLflow shared-catalog correctness contract.

The MLflow store is profile-shared (one serverless App per bootstrap profile).
These tests pin the invariants that keep that store correct and prevent a
regression to a local/per-project tracking store or an experiment-"0"-only
write:

1. No code path emits a local-file tracking URI (sqlite:// / mlflow.db /
   ./mlruns / file://) — MLflow's silent CWD default must never be relied on.
2. Write paths target a PER-PROJECT experiment (sanitized project name), not the
   literal "Default".
3. Read scope resolves to the active project's experiment id + "0" (read-both)
   by default, and to every experiment when all_projects=True.
4. The tune path receives its tracking ref from the profile: do/lib/profile.sh
   maps mlflowAppArn -> MLFLOW_APP_ARN.

The behavioral assertions use fake MLflow client objects (no network, no
./mlruns written); the write-path and profile.sh assertions are source-level so
they do not require a configured MLflow server.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

_LIB = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "templates", "do", "lib", "python")
)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import mlcc_mlflow  # noqa: E402

_TEMPLATES_DO = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "templates", "do")
)


# ── Fake MLflow client ──────────────────────────────────────────────────────


class _FakeExperiment:
    def __init__(self, experiment_id, name):
        self.experiment_id = experiment_id
        self.name = name


class _FakeClient:
    """Minimal client exposing the surface _resolve_experiment_ids uses."""

    def __init__(self, experiments):
        # experiments: list of (experiment_id, name)
        self._by_name = {name: _FakeExperiment(eid, name) for eid, name in experiments}
        self._all = [_FakeExperiment(eid, name) for eid, name in experiments]

    def get_experiment_by_name(self, name):
        return self._by_name.get(name)

    def search_experiments(self):
        return list(self._all)


# ── 2. Per-project experiment name (write-side derivation) ───────────────────


def test_project_experiment_name_sanitizes_explicit_name():
    # Explicit project name wins and is sanitized (slashes/spaces removed).
    assert mlcc_mlflow._project_experiment_name("my/proj name") == "my--projname"


def test_project_experiment_name_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("PROJECT_NAME", "env-project")
    assert mlcc_mlflow._project_experiment_name() == "env-project"


def test_project_experiment_name_defaults_when_nameless(monkeypatch):
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    # Nameless context preserves today's behavior (the Default experiment).
    assert mlcc_mlflow._project_experiment_name("") == "Default"


# ── 3. Read-scope resolution (active + "0" read-both; --all) ─────────────────


def test_resolve_experiment_ids_reads_active_project_and_zero():
    client = _FakeClient([("0", "Default"), ("7", "my-project")])
    ids = mlcc_mlflow._resolve_experiment_ids(client=client, project_name="my-project")
    # Read-both: the active project's id PLUS the legacy "0" experiment.
    assert ids == ["7", "0"]


def test_resolve_experiment_ids_missing_experiment_falls_back_to_zero():
    # Project experiment not created yet → just ["0"], never an error.
    client = _FakeClient([("0", "Default")])
    ids = mlcc_mlflow._resolve_experiment_ids(client=client, project_name="not-created-yet")
    assert ids == ["0"]


def test_resolve_experiment_ids_nameless_is_zero_only(monkeypatch):
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    client = _FakeClient([("0", "Default"), ("7", "my-project")])
    ids = mlcc_mlflow._resolve_experiment_ids(client=client, project_name="")
    assert ids == ["0"]


def test_resolve_experiment_ids_all_projects_returns_every_id():
    client = _FakeClient([("0", "Default"), ("7", "a"), ("8", "b")])
    ids = mlcc_mlflow._resolve_experiment_ids(client=client, all_projects=True)
    assert set(ids) == {"0", "7", "8"}


def test_resolve_experiment_ids_no_client_degrades_to_zero():
    # No client and mlflow absent/unconfigured → the safe pre-BL163 default.
    ids = mlcc_mlflow._resolve_experiment_ids(client=None, project_name="whatever")
    assert ids == ["0"]


# ── 1. No local-file tracking anywhere in the runtime python ─────────────────

_LOCAL_FILE_PATTERNS = [
    re.compile(r"sqlite://"),
    re.compile(r"mlflow\.db"),
    re.compile(r"\./mlruns"),
    re.compile(r"\bfile://"),
    re.compile(r"set_tracking_uri"),
]


def _runtime_python_files():
    for root, _dirs, files in os.walk(os.path.join(_TEMPLATES_DO, "lib", "python")):
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


def test_no_local_file_tracking_uri_in_runtime_python():
    offenders = []
    for path in _runtime_python_files():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for pat in _LOCAL_FILE_PATTERNS:
            if pat.search(text):
                offenders.append((os.path.basename(path), pat.pattern))
    assert offenders == [], (
        "local-file MLflow tracking introduced (the shared serverless store must "
        f"not be bypassed): {offenders}"
    )


# ── 2. Write paths use the per-project experiment, not "Default" ─────────────


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_register_dataset_sets_per_project_experiment():
    text = _read(os.path.join(_TEMPLATES_DO, "lib", "python", "register_dataset.py"))
    assert "_project_experiment_name(" in text
    # The hardcoded Default-experiment write is gone.
    assert 'set_experiment("Default")' not in text


def test_register_model_sets_per_project_experiment():
    text = _read(os.path.join(_TEMPLATES_DO, "lib", "python", "register_model.py"))
    assert "_project_experiment_name(" in text
    assert 'set_experiment("Default")' not in text


# ── 4. Tune path ref comes from the profile (profile.sh mapping) ─────────────


def test_profile_sh_maps_mlflow_app_arn_from_profile():
    text = _read(os.path.join(_TEMPLATES_DO, "lib", "profile.sh"))
    # profile.sh maps the profile key mlflowAppArn to the env var tune_submit.py
    # consumes, and exports it so the Python process sees it.
    assert "_PROFILE_mlflowAppArn" in text
    assert "MLFLOW_APP_ARN" in text
    assert re.search(r"export\s+MLFLOW_APP_ARN", text), (
        "profile.sh must export MLFLOW_APP_ARN so tune_submit.py reads it from os.environ"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
