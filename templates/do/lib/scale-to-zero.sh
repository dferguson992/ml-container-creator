# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helper: scale-to-zero (Wave 10, ADR-014). Sourced by do/deploy,
# deploy.d/realtime-inference, clean.d/realtime-inference, do/status, do/add-ic.
#
# Nothing here branches on a target NAME. Flag acceptance, the refusal reason,
# the warning wording and the mechanism come from the target's descriptor
# (targets.d/<target>/manifest.json "scale_to_zero", read through
# lib/python/scale_to_zero.py). The mechanism arms below are keyed on the
# descriptor's "mechanism" value (one arm in v2: sagemaker-ic-step), so enabling
# another target is a descriptor change plus one arm (ADR-014 Consequences).
#
# Config vars (do/config, runtime-owned, preserved by regenerate):
#   SCALE_TO_ZERO          true | false (default false; off = no behavior change)
#   SCALE_MAX_CAPACITY     1..20 (default 1): max IC copies and MaxInstanceCount
#   SCALE_TO_ZERO_STATUS   ok | failed | "" (written after the scaling setup)
#
# bash 3.2: no associative arrays, no ${v^^}, no mapfile.

_STZ_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_STZ_PY="${_STZ_LIB_DIR}/python/scale_to_zero.py"
_STZ_TM="${_STZ_LIB_DIR}/python/target_manifest.py"
STZ_MAX_CAPACITY_LIMIT=20

# _stz_refuse <reason> <remedy> — contract violation (exit 3), uniform format.
_stz_refuse() {
    if type _contract_violation >/dev/null 2>&1; then
        _contract_violation "scale-to-zero" "$1" "$2"
    fi
    echo "❌ Contract violation: scale-to-zero"
    echo "   $1"
    echo "   → $2"
    exit 3
}

# stz_load <target> — sets STZ_SUPPORT, STZ_MECHANISM, STZ_WAKE_SIGNAL,
#   STZ_REQUESTS_AT_ZERO, STZ_IDLE_BILLING, STZ_ENABLED, STZ_REASON from the
#   descriptor. Returns 1 (and STZ_ENABLED=false) for an unknown target.
stz_load() {
    local _target="$1" _out=""
    STZ_ENABLED=false
    STZ_SUPPORT="" STZ_MECHANISM="" STZ_WAKE_SIGNAL="" STZ_REQUESTS_AT_ZERO="" STZ_IDLE_BILLING="" STZ_REASON=""
    [ -n "${_target}" ] || return 1
    _out=$(python3 "${_STZ_PY}" shell "${_target}" 2>/dev/null) || return 1
    eval "${_out}"
    return 0
}

# stz_parse_flags <--scale-to-zero value> <--scale-max value>
#   R2.4: validates the raw flag values before any AWS call; a bad value exits 1
#   (the do/deploy bad-flag convention). Sets:
#     STZ_FLAGS_GIVEN  1 when either flag was passed
#     STZ_FLAG_ON      true | false | "" (not passed)
#     STZ_FLAG_MAX     1..20 | "" (not passed)
stz_parse_flags() {
    local _on="$1" _max="$2"
    STZ_FLAGS_GIVEN="" STZ_FLAG_ON="" STZ_FLAG_MAX=""
    if [ -n "${_on}" ]; then
        case "${_on}" in
            on) STZ_FLAG_ON=true ;;
            off) STZ_FLAG_ON=false ;;
            *)
                echo "❌ --scale-to-zero must be 'on' or 'off' (got '${_on}')" >&2
                exit 1
                ;;
        esac
        STZ_FLAGS_GIVEN=1
    fi
    if [ -n "${_max}" ]; then
        case "${_max}" in
            *[!0-9]*)
                echo "❌ --scale-max must be an integer from 1 to ${STZ_MAX_CAPACITY_LIMIT} (got '${_max}')" >&2
                exit 1
                ;;
        esac
        _max=$((10#${_max}))
        if [ "${_max}" -lt 1 ] || [ "${_max}" -gt "${STZ_MAX_CAPACITY_LIMIT}" ]; then
            echo "❌ --scale-max must be an integer from 1 to ${STZ_MAX_CAPACITY_LIMIT} (got '$2')" >&2
            exit 1
        fi
        STZ_FLAG_MAX="${_max}"
        STZ_FLAGS_GIVEN=1
    fi
    return 0
}

# stz_refuse_disabled <target>
#   R2.2: a target whose descriptor says enabled=false refuses the flags with
#   the descriptor's reason (exit 3).
stz_refuse_disabled() {
    local _target="$1"
    if ! stz_load "${_target}"; then
        _stz_refuse "unknown deployment target '${_target}'" "pick a target with ./do/deploy --target <target>"
    fi
    if [ "${STZ_ENABLED}" != "true" ]; then
        _stz_refuse "${_target} does not support --scale-to-zero in this release: ${STZ_REASON}" \
            "remove --scale-to-zero / --scale-max (see docs/scale-to-zero.md for the per-target matrix)"
    fi
}

# stz_refuse_unsupported_config <target>
#   R2.3: mechanism-specific refusals (exit 3), checked against the current
#   config before anything is created. Requires stz_load to have run.
stz_refuse_unsupported_config() {
    local _target="$1"
    case "${STZ_MECHANISM}" in
        sagemaker-ic-step)
            if [ -n "${CAPACITY_RESERVATION_ARN:-}" ]; then
                _stz_refuse "${_target}: CAPACITY_RESERVATION_ARN is set; reserved capacity is billed whether or not the endpoint is at zero, so scale-to-zero saves nothing" \
                    "unset CAPACITY_RESERVATION_ARN in do/config, or use --scale-to-zero off"
            fi
            if [ -n "${INSTANCE_POOLS:-}" ]; then
                _stz_refuse "${_target}: INSTANCE_POOLS is set; scale-to-zero with heterogeneous instance pools is unverified" \
                    "unset INSTANCE_POOLS in do/config (single instance type), or use --scale-to-zero off"
            fi
            if [ "${ENDPOINT_EXTERNAL:-false}" = "true" ] || { [ -n "${ENDPOINT_STRATEGY:-}" ] && [ "${ENDPOINT_STRATEGY}" != "new" ]; }; then
                _stz_refuse "${_target}: ENDPOINT_STRATEGY is '${ENDPOINT_STRATEGY:-existing}'; scale-to-zero changes the endpoint config, which this project only owns for a new endpoint" \
                    "use --endpoint-strategy new, or use --scale-to-zero off"
            fi
            ;;
        *)
            _stz_refuse "${_target}: scale-to-zero mechanism '${STZ_MECHANISM}' is not implemented" \
                "set enabled=false in targets.d/${_target}/manifest.json or add the mechanism arm to do/lib/scale-to-zero.sh"
            ;;
    esac
}

# stz_active <target> — 0 when SCALE_TO_ZERO=true and the target is enabled.
stz_active() {
    [ "${SCALE_TO_ZERO:-false}" = "true" ] || return 1
    stz_load "$1" || return 1
    [ "${STZ_ENABLED}" = "true" ]
}

# stz_target_live <target> — 0 when the target's status var holds its success
# status (both read from the descriptor).
stz_target_live() {
    local _target="$1" _var="" _ok=""
    _var=$(python3 "${_STZ_TM}" status_var "${_target}" 2>/dev/null) || return 1
    _ok=$(python3 "${_STZ_TM}" success_status "${_target}" 2>/dev/null) || return 1
    [ -n "${_var}" ] && [ "${!_var:-}" = "${_ok}" ]
}

# stz_apply_flags <target> [persist]
#   R2.1–R2.3: refuse on a disabled target or unsupported config, then export
#   SCALE_TO_ZERO / SCALE_MAX_CAPACITY for this run. With persist=1 (not a
#   dry run) they are also written to do/config through _update_config.
stz_apply_flags() {
    local _target="$1" _persist="${2:-0}"
    [ -n "${STZ_FLAGS_GIVEN:-}" ] || return 0
    stz_refuse_disabled "${_target}"
    if [ -n "${STZ_FLAG_ON}" ]; then
        SCALE_TO_ZERO="${STZ_FLAG_ON}"
    fi
    if [ -n "${STZ_FLAG_MAX}" ]; then
        SCALE_MAX_CAPACITY="${STZ_FLAG_MAX}"
    fi
    export SCALE_TO_ZERO="${SCALE_TO_ZERO:-false}"
    export SCALE_MAX_CAPACITY="${SCALE_MAX_CAPACITY:-1}"
    if [ "${SCALE_TO_ZERO}" = "true" ]; then
        stz_refuse_unsupported_config "${_target}"
    fi
    if [ "${_persist}" = "1" ] && type _update_config >/dev/null 2>&1; then
        _update_config "SCALE_TO_ZERO" "${SCALE_TO_ZERO}"
        _update_config "SCALE_MAX_CAPACITY" "${SCALE_MAX_CAPACITY}"
    fi
    return 0
}
