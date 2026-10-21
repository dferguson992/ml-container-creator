from __future__ import annotations
"""Technique-aware tuning config builder (BL128).

Purpose: render / validate / resolve a technique's hyperparameter knobs from a
         Tune_Descriptor, for both the interactive `do/tune` flow and the
         non-interactive (flag/config) path.

The CORE (`build_config`) is descriptor-source-AGNOSTIC (Requirement 7): it takes a
`hyperparameters` map (the descriptor variant's synced block — the shared
`$defs/hyperparameter` entry shape from BL129) plus already-resolved supplied values,
and returns the resolved knob→value map + any validation errors. It knows nothing about
`tune.d/`, technique names, `_synced`, or `reward_source`. The ONLY `tune.d`-aware code
is the thin `build_from_tune_descriptor` adapter below; the user-authored `do/train`
plugin UX (BL154) will supply its own descriptor source and call the same core.

Precedence (Requirement 4.1): descriptor default < config file < interactive answer /
explicit flag (the user-supplied value wins). An interactive answer and a flag occupy
the same highest tier; a single run supplies a knob through at most one of them, so the
caller merges config + flags into `supplied` (flag-wins) and the core prompts only for
knobs NOT in `supplied`.

Callers: do/tune (via .tune_helper.py `build-config`). The interactive prompt layer is
`questionary` (wired in the do/tune adapter / helper, mirroring deploy_prompts.py); the
core's `prompter` is injectable so it is unit-testable without a TTY.
"""

# No third-party imports at module top — the CORE needs no questionary/pyarrow. The
# tune.d adapter imports tune_manifest lazily; the helper wires questionary.


# ── effective control / tier derivation (R2.3) ────────────────────────────────

_VALID_TIERS = ("primary", "advanced", "tertiary")
_VALID_CONTROLS = ("number_input", "dropdown", "text_input")


def effective_control(knob: dict) -> str:
    """The knob's EFFECTIVE input control: the descriptor `control` when present and
    valid, else derived — `enum` → dropdown, numeric bounds → number_input, else
    text_input. Never assumes `control` is present (MTRL's leaner schema omits it)."""
    control = knob.get("control")
    if control in _VALID_CONTROLS:
        return control
    if isinstance(knob.get("enum"), list) and knob["enum"]:
        return "dropdown"
    if knob.get("type") in ("integer", "number") or "min" in knob or "max" in knob:
        return "number_input"
    return "text_input"


def effective_tier(knob: dict) -> str:
    """The knob's EFFECTIVE disclosure tier: the descriptor `tier` when present and
    valid, else `primary` (so a tier-less MTRL knob is never silently dropped)."""
    tier = knob.get("tier")
    return tier if tier in _VALID_TIERS else "primary"


# ── value coercion + validation (R1.3) ────────────────────────────────────────

def _coerce(knob_name: str, knob: dict, value):
    """Coerce a string/whatever value to the knob's declared type. Returns
    (coerced, error_or_None). Non-fatal: a coercion failure is returned as an error
    string so the caller collects it rather than raising."""
    t = knob.get("type", "string")
    if value is None:
        return None, None
    try:
        if t == "integer":
            return int(str(value)), None
        if t == "number":
            return float(str(value)), None
        if t == "boolean":
            s = str(value).strip().lower()
            if s in ("true", "1", "yes"):
                return True, None
            if s in ("false", "0", "no"):
                return False, None
            return None, f"knob '{knob_name}' expects a boolean (true/false), got '{value}'"
        return str(value), None
    except (ValueError, TypeError):
        return None, f"knob '{knob_name}' expects type {t}, got '{value}'"


def _validate(knob_name: str, knob: dict, value) -> str | None:
    """Validate a coerced value against the knob's constraints. Returns an error
    string naming the knob + allowed range/choices, or None when valid."""
    enum = knob.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        return f"knob '{knob_name}' must be one of {enum}; got {value!r}"
    lo, hi = knob.get("min"), knob.get("max")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(lo, (int, float)) and value < lo:
            return f"knob '{knob_name}' must be >= {lo}; got {value}"
        if isinstance(hi, (int, float)) and value > hi:
            return f"knob '{knob_name}' must be <= {hi}; got {value}"
        step = knob.get("step")
        if isinstance(step, (int, float)) and step > 0 and lo is not None:
            # Allow a small float tolerance for step alignment.
            n = (value - lo) / step
            if abs(n - round(n)) > 1e-9:
                return f"knob '{knob_name}' must align to step {step} from {lo}; got {value}"
    return None


# ── the CORE (descriptor-source-agnostic) ─────────────────────────────────────

def build_config(hyperparameters: dict, supplied: dict, *, interactive: bool,
                 include_advanced: bool = False, prompter=None) -> dict:
    """Resolve a technique's knobs from its `hyperparameters` map + supplied values.

    Args:
        hyperparameters: the descriptor variant's synced knob map (knob -> entry).
        supplied: already-resolved values from config file + flags (flag-wins merged by
            the caller). A knob present here is NOT prompted.
        interactive: True when a TTY is present and the user did not pass
            --non-interactive. When True and a primary/advanced knob is not in
            `supplied`, the user is prompted via `prompter`.
        include_advanced: the "advanced options" opt-in (R2.2). advanced knobs are
            prompted only when True; tertiary knobs are never prompted (R2.2).
        prompter: callable(name, knob, control, default) -> raw value, for interactive
            knobs. Injectable for tests; None forces the non-interactive path.

    Returns:
        {"resolved": {knob: coerced_value}, "errors": [str, ...]}. The caller decides
        fail-fast on a non-empty `errors`. A `required` knob that resolves to no value
        is an error (R4.3). Unknown supplied knobs (not in `hyperparameters`) are errors
        (R3.3).
    """
    resolved: dict = {}
    errors: list[str] = []

    # (R3.3) reject supplied knobs that are not in this technique's descriptor.
    for name in supplied:
        if name not in hyperparameters:
            known = ", ".join(sorted(hyperparameters)) or "(none)"
            errors.append(
                f"unknown hyperparameter '{name}' for this technique. "
                f"Valid knobs: {known}."
            )

    for name, knob in hyperparameters.items():
        tier = effective_tier(knob)
        has_default = "default" in knob
        value = None
        source = None

        if name in supplied:
            value = supplied[name]
            source = "supplied"
        elif interactive and prompter is not None and (
            tier == "primary" or (tier == "advanced" and include_advanced)
        ):
            # Prompt (R2.1/R2.2). tertiary is never prompted; advanced only on opt-in.
            raw = prompter(name, knob, effective_control(knob),
                           knob.get("default") if has_default else None)
            if raw is not None and str(raw) != "":
                value = raw
                source = "prompt"
            elif has_default:
                value = knob["default"]
                source = "default"
        elif has_default:
            value = knob["default"]
            source = "default"

        if value is None:
            # No supplied/prompted/default value. A required knob is an error (R4.3);
            # an optional knob with no default is simply omitted.
            if knob.get("required") is True:
                errors.append(
                    f"required hyperparameter '{name}' has no value "
                    f"(no default, config value, or flag/answer)."
                )
            continue

        # Coerce + validate every resolved value (identical for prompt/flag — R3.4).
        # A descriptor default is trusted (already valid) but coercing is harmless.
        coerced, cerr = _coerce(name, knob, value)
        if cerr:
            errors.append(cerr)
            continue
        verr = _validate(name, knob, coerced)
        if verr:
            errors.append(verr)
            continue
        resolved[name] = coerced

    return {"resolved": resolved, "errors": errors}


# ── tune.d adapter (the ONLY tune.d-aware entry) ──────────────────────────────

def build_from_tune_descriptor(technique, variant, supplied, *, interactive,
                               include_advanced=False, prompter=None, tune_dir=None):
    """Read the `tune.d/<technique>` descriptor's hyperparameters and run the core.

    This is the single place that knows about tune.d/ and technique names; the CORE
    (`build_config`) has no such coupling. BL154's do/train adapter is a sibling that
    reads a user-authored descriptor (with its own runtime schema validation) and calls
    the same core.
    """
    import tune_manifest
    hp = tune_manifest.read_hyperparameters(technique, variant, tune_dir)
    return build_config(hp, supplied, interactive=interactive,
                        include_advanced=include_advanced, prompter=prompter)


# ── questionary prompt layer (interactive) ────────────────────────────────────

def _questionary_prompter(name, knob, control, default):
    """Prompt for one knob via questionary, modeled on deploy_prompts.py. Returns the
    user's raw string (or None). A dropdown (enum) → questionary.select; numeric/text →
    questionary.text pre-filled with the default. Validation is re-done by the core, so
    this layer only needs to collect input.

    BL178: the prompt UI is rendered to STDERR (not stdout). do/tune invokes
    `build-config` inside a command substitution `bc_result=$(... build-config ...)`
    that CAPTURES stdout — the final {resolved,errors} JSON. prompt_toolkit's default
    output is stdout, so without this the prompt would draw into the captured pipe
    (invisible → the CLI appears to hang) and corrupt the JSON. Routing the prompt
    output to stderr keeps stdout clean (JSON only) and the prompt visible on the
    terminal; input still comes from the controlling tty. Mirrors deploy_prompts.py's
    stderr convention."""
    import sys
    import questionary
    # Render the prompt UI to stderr so it does not land in do/tune's captured stdout.
    _out = None
    try:
        from prompt_toolkit.output.defaults import create_output
        _out = create_output(stdout=sys.stderr)
    except Exception:  # noqa: BLE001 — if unavailable, fall back to default (stdout)
        _out = None
    _kw = {"output": _out} if _out is not None else {}
    label = knob.get("display_name") or name
    desc = knob.get("description") or knob.get("hint")
    message = f"{label}:" if not desc else f"{label} ({desc}):"
    if control == "dropdown" and isinstance(knob.get("enum"), list) and knob["enum"]:
        choices = [str(c) for c in knob["enum"]]
        default_str = str(default) if default is not None else None
        res = questionary.select(
            message, choices=choices,
            default=default_str if default_str in choices else None,
            **_kw,
        ).ask()
        return res
    # BL-UX: numeric (and any min/max-bounded) knob — SHOW the allowed range in the
    # message AND validate INLINE so an out-of-range value is rejected AT the prompt
    # (naming the range) instead of silently accepted and only caught by the core
    # after all prompts finish. The inline check reuses the core _coerce/_validate so
    # it agrees exactly with the authoritative validation.
    lo, hi = knob.get("min"), knob.get("max")
    if lo is not None or hi is not None:
        if lo is not None and hi is not None:
            message = f"{message[:-1]} [{lo}–{hi}]:" if message.endswith(":") else f"{message} [{lo}–{hi}]"
        elif lo is not None:
            message = f"{message[:-1]} [>= {lo}]:" if message.endswith(":") else f"{message} [>= {lo}]"
        else:
            message = f"{message[:-1]} [<= {hi}]:" if message.endswith(":") else f"{message} [<= {hi}]"

        def _validate_input(text):
            text = (text or "").strip()
            if text == "":
                return True  # empty → accept the default (handled by the caller)
            coerced, cerr = _coerce(name, knob, text)
            if cerr:
                return cerr
            verr = _validate(name, knob, coerced)
            return verr if verr else True

        return questionary.text(
            message, default=str(default) if default is not None else "",
            validate=_validate_input, **_kw,
        ).ask()
    return questionary.text(
        message, default=str(default) if default is not None else "",
        **_kw,
    ).ask()


def cmd_build_config(args):
    """`.tune_helper.py build-config` — resolve a technique's knobs and print
    {resolved, errors} as JSON for do/tune to consume. `--supplied` is the merged
    config-file + --set map (flag-wins) as JSON. Interactive unless --non-interactive
    or no TTY; prompts via questionary."""
    import json
    import sys
    from common import _output

    try:
        supplied = json.loads(args.supplied) if args.supplied else {}
    except (ValueError, TypeError):
        from common import _error_exit
        _error_exit("--supplied must be valid JSON (a {knob: value} object)")
    if not isinstance(supplied, dict):
        from common import _error_exit
        _error_exit("--supplied must be a JSON object")

    interactive = (not getattr(args, "non_interactive", False)) and sys.stdin.isatty()
    prompter = _questionary_prompter if interactive else None

    result = build_from_tune_descriptor(
        args.technique, getattr(args, "variant", None), supplied,
        interactive=interactive, include_advanced=getattr(args, "advanced", False),
        prompter=prompter, tune_dir=getattr(args, "tune_dir", None),
    )
    _output(result)
