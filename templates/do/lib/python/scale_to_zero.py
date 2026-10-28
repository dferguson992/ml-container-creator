# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scale-to-zero capability helper (Wave 10, ADR-014).

Purpose: the one place bash reads a target's ``scale_to_zero`` descriptor block
         from (via target_manifest.py), so do/deploy, deploy.d/realtime-inference,
         clean.d/realtime-inference, do/status and do/add-ic never branch on a
         target name for this capability (ADR-008).

CLI (one-shot, for bash callers):
    python3 do/lib/python/scale_to_zero.py shell <target>
        prints STZ_SUPPORT=..., STZ_MECHANISM=..., STZ_WAKE_SIGNAL=...,
        STZ_REQUESTS_AT_ZERO=..., STZ_IDLE_BILLING=..., STZ_ENABLED=true|false,
        STZ_REASON=... as shell-quoted assignments for ``eval``.

  Exit codes: 0 success; 2 usage; 3 unknown target.
"""

from __future__ import annotations

import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import target_manifest  # noqa: E402

# descriptor key -> shell variable
_SHELL_VARS = (
    ("support", "STZ_SUPPORT"),
    ("mechanism", "STZ_MECHANISM"),
    ("wake_signal", "STZ_WAKE_SIGNAL"),
    ("requests_at_zero", "STZ_REQUESTS_AT_ZERO"),
    ("idle_billing", "STZ_IDLE_BILLING"),
    ("enabled", "STZ_ENABLED"),
    ("reason", "STZ_REASON"),
)


def capability(target: str) -> dict:
    """The target's scale_to_zero block (alias-aware; see target_manifest)."""
    return target_manifest.scale_to_zero(target)


def shell_assignments(target: str) -> list[str]:
    """``VAR='value'`` lines for the block; null -> '', booleans -> true/false."""
    block = capability(target)
    lines = []
    for key, var in _SHELL_VARS:
        value = block.get(key)
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif value is None:
            text = ""
        else:
            text = str(value)
        lines.append(f"{var}={shlex.quote(text)}")
    return lines


def _cli(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "shell":
        try:
            print("\n".join(shell_assignments(argv[1])))
        except target_manifest.ManifestError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3
        return 0
    print("usage: scale_to_zero.py shell <target>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
