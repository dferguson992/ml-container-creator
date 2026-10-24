// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Wave 6 — do/adapter (incl. BL169 on Kubernetes) and do/add-ic on the shared
 * resolver + contract, executed end-to-end in a sandbox with stubbed aws /
 * kubectl / lora_vllm.py / .register_helper.py.
 *
 *   R5 / BL169  the k8s path ACCEPTS --from-* (was refused), resolves through the
 *               shared resolver, and hot-loads it via lora_vllm.py — Wave 6.1
 *               (BL189): by its IN-POD path under the model mount, never s3://;
 *               every Running pod (R1.6); rank pre-check vs the live limit (R3.2)
 *   R5.3 / R8.5 a full_model is refused on do/adapter; a lora_adapter on do/add-ic
 *   R2 / R8.3   --weights / --model-data are deprecated aliases of --from-weights
 *               (same resolved source + the notice); --model-data-url is untouched
 *   R9 / R8.6   a crashing / failing hot-load helper is SURFACED, not swallowed
 *   R6.2        do/add-ic --from-registry (stretch) resolves a full-model version
 */
import { describe, it, before, after } from 'mocha';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { makeSandbox, runBash, sandboxEnv, ACTIVE_STATUS_CONFIG, REPO } from '../helpers/additive-sandbox.js';

const AWS_STUB = `#!/bin/bash
# Stub aws CLI: cluster lookups succeed. S3 reads fail (advisory sniff → unknown)
# unless STUB_RANK is set, in which case adapter_config.json reads return r=STUB_RANK.
case "$1 $2" in
  "sagemaker describe-cluster") echo "arn:aws:eks:us-west-2:111122223333:cluster/hp-eks" ;;
  "eks update-kubeconfig") exit 0 ;;
  "s3 cp")
    if [ -n "\${STUB_RANK:-}" ] && [[ "$3" == */adapter_config.json ]]; then
      echo "{\\"r\\": \${STUB_RANK}, \\"rank_pattern\\": {}}"; exit 0
    fi
    echo "stub aws: $*" >&2; exit 1 ;;
  *) echo "stub aws: $*" >&2; exit 1 ;;
esac
`;
// kubectl stub (Wave 6.1): Running pod names (STUB_PODS), pod JSON carrying the
// live VLLM_MAX_LORA_RANK (STUB_LIVE_RANK), and the live InferenceEndpointConfig
// (S3 model source s3://core/models/m, prefetch STUB_PREFETCH; STUB_SOURCE=huggingface
// for an HF-sourced deployment).
const KUBECTL_STUB = `#!/bin/bash
pods="\${STUB_PODS:-proj-pod-0}"
case "$1 $2" in
  "get inferenceendpointconfig")
    if [ "\${STUB_SOURCE:-s3}" = "huggingface" ]; then
      echo '{"spec":{"modelSourceConfig":{"modelSourceType":"huggingface","prefetchEnabled":true}}}'
    else
      echo "{\\"spec\\":{\\"modelSourceConfig\\":{\\"modelSourceType\\":\\"s3\\",\\"s3Storage\\":{\\"bucketName\\":\\"core\\"},\\"modelLocation\\":\\"models/m\\",\\"prefetchEnabled\\":\${STUB_PREFETCH:-false}}}}"
    fi ;;
  "get pod")
    case " $* " in
      *" -o json "*)
        echo "{\\"items\\":[{\\"spec\\":{\\"containers\\":[{\\"env\\":[{\\"name\\":\\"VLLM_MAX_LORA_RANK\\",\\"value\\":\\"\${STUB_LIVE_RANK:-64}\\"}]}]}}]}" ;;
      *) for p in $pods; do echo "$p"; done ;;
    esac ;;
  port-forward*) exec sleep 30 ;;
  *) case "$1" in port-forward) exec sleep 30 ;; *) exit 1 ;; esac ;;
esac
`;
// lora_vllm.py stub: records every call; LORA_STUB selects the outcome;
// LORA_FAIL_CALL=<n> fails the n-th load (multi-pod partial failure).
const LORA_STUB = `#!/usr/bin/env python3
import json, os, sys
with open(os.environ["LORA_LOG"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
with open(os.environ["LORA_LOG"]) as f:
    n_loads = sum(1 for line in f if line.startswith("load "))
mode = os.environ.get("LORA_STUB", "ok")
cmd = sys.argv[1]
if mode == "crash":
    print("Traceback (most recent call last):\\nModuleNotFoundError: No module named 'requests'", file=sys.stderr)
    sys.exit(1)
if cmd == "list":
    print(json.dumps([{"name": "a1", "path": "/opt/ml/model/.mlcc-adapters/proj/a1"}]))
    sys.exit(0)
if mode == "409" and cmd == "load":
    print(json.dumps({"success": False, "error": "adapter exists", "status_code": 409}))
    sys.exit(1)
if cmd == "load" and str(n_loads) == os.environ.get("LORA_FAIL_CALL", ""):
    print(json.dumps({"success": False, "error": "boom", "status_code": 500}))
    sys.exit(1)
print(json.dumps({"success": True}))
`;
const REGISTER_STUB = `#!/usr/bin/env python3
import json, os, sys
md = {}
if os.environ.get("STUB_KIND"):
    md["mlcc.artifact_kind"] = os.environ["STUB_KIND"]
print(json.dumps({"arn": "x", "version": "3", "modelDataUrl": os.environ.get("STUB_URI", "s3://b/reg/adapter/"), "metadata": md}))
`;
function k8sConfig(extra = []) {
    return [
        ACTIVE_STATUS_CONFIG,
        'export DEPLOYMENT_TARGET="hyperpod-eks"',
        'export PROJECT_NAME="proj"',
        'export AWS_REGION="us-west-2"',
        'export HP_LORA_ENABLED="true"',
        'export HP_CLUSTER_NAME="hp"',
        'export HP_LOCAL_PORT="18080"',
        'export HP_PF_WAIT="0"',
        ...extra,
        ''
    ].join('\n');
}
// The deployment's canonical adapter home (model prefix + .mlcc-adapters, from the
// target descriptor) and the in-pod path it maps to.
const PFX = 's3://core/models/m/.mlcc-adapters/proj';
const POD = '/opt/ml/model/.mlcc-adapters/proj';
const URL = 'http://localhost:18080';
describe('Wave 6 / 6.1 — do/adapter on Kubernetes: --from-* sources, local-path hot-load (BL169 / BL189)', function () {
    this.timeout(120000);
    let sb;
    let log;
    before(() => {
        sb = makeSandbox({ scripts: ['adapter'], pyHelpers: true });
        sb.write('bin/aws', AWS_STUB);
        sb.write('bin/kubectl', KUBECTL_STUB);
        sb.write('do/lib/python/lora_vllm.py', LORA_STUB);
        sb.write('do/.register_helper.py', REGISTER_STUB);
        log = join(sb.root, 'lora.log');
    });
    after(() => sb.cleanup());
    const run = (args, { config = [], env = {} } = {}) => {
        sb.write('do/config', k8sConfig(config), 0o644);
        sb.write('lora.log', '', 0o644);
        const r = runBash(join(sb.doDir, 'adapter'), args,
            sandboxEnv(sb, { HOME: sb.root, LORA_LOG: log, ...env }), { timeout: 30000 });
        return { ...r, calls: readFileSync(log, 'utf8').trim().split('\n').filter(Boolean) };
    };
    it('add --from-weights under the model prefix hot-loads the IN-POD path (never s3://)', () => {
        const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`]);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, [`load ad1 ${POD}/ad1 ${URL}`]);
        assert.match(r.stdout, /Pod path: \/opt\/ml\/model\/\.mlcc-adapters\/proj\/ad1/);
    });
    it('--weights is a deprecated alias: same hot-load + the notice on stderr', () => {
        const r = run(['add', 'ad1', '--weights', `${PFX}/ad1/`]);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, [`load ad1 ${POD}/ad1 ${URL}`]);
        assert.match(r.stderr, /--weights is deprecated; use --from-weights/);
    });
    it('R1.3 (Part A): a source outside the model prefix is refused with the exact copy command, before any engine call', () => {
        const r = run(['add', 'tuned', '--from-tune'], {
            config: ['export TUNE_OUTPUT_PATH_LATEST="s3://b/tune/out/"', 'export TUNE_OUTPUT_TYPE_LATEST="adapter"']
        });
        assert.strictEqual(r.code, 1, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, [], 'no s3:// path may ever reach the engine');
        assert.match(r.stdout, /outside this deployment's model prefix/);
        assert.match(r.stdout, /aws s3 sync s3:\/\/b\/tune\/out\/ s3:\/\/core\/models\/m\/\.mlcc-adapters\/proj\/tuned\//);
        assert.ok(!/Contract violation/.test(r.stdout), 'the k8s source-verb refusal must stay gone');
    });
    it('add --from-tune whose output is under the model prefix hot-loads it', () => {
        const r = run(['add', 'tuned', '--from-tune'], {
            config: [`export TUNE_OUTPUT_PATH_LATEST="${PFX}/tuned/"`, 'export TUNE_OUTPUT_TYPE_LATEST="adapter"']
        });
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, [`load tuned ${POD}/tuned ${URL}`]);
    });
    it('add --from-registry hot-loads the registered adapter version (mapped to the pod path)', () => {
        const r = run(['add', 'reg', '--from-registry', 'arn:aws:sagemaker:us-west-2:1:model-package/proj/3'],
            { env: { STUB_KIND: 'lora_adapter', STUB_URI: `${PFX}/reg/` } });
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, [`load reg ${POD}/reg ${URL}`]);
    });
    it('a full_model source is refused BEFORE any cluster call (kind gate, exit 3)', () => {
        const r = run(['add', 'full', '--from-tune'], {
            config: ['export TUNE_OUTPUT_PATH_LATEST="s3://b/tune/m/"', 'export TUNE_OUTPUT_TYPE_LATEST="model"']
        });
        assert.strictEqual(r.code, 3, r.stdout + r.stderr);
        assert.match(r.stdout, /Contract violation: artifact-kind/);
        assert.match(r.stdout, /do\/add-ic/);
        assert.match(r.stdout, /do\/add-pod/);
        assert.deepStrictEqual(r.calls, [], 'nothing may be hot-loaded');
        assert.ok(!/Port-forward/.test(r.stdout), 'refusal precedes the port-forward');
    });
    it('a registry full_model is refused too (any source)', () => {
        const r = run(['add', 'full', '--from-registry', 'arn:aws:sagemaker:us-west-2:1:model-package/proj/4'],
            { env: { STUB_KIND: 'full_model' } });
        assert.strictEqual(r.code, 3);
        assert.match(r.stdout, /Contract violation: artifact-kind/);
    });
    it('an unknown kind (raw S3) degrades to the content check and still loads', () => {
        const r = run(['add', 'legacy', '--from-weights', `${PFX}/legacy/`]);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.match(r.stderr, /artifact_kind unknown/);
        assert.match(r.stdout, /Content check/);
        assert.strictEqual(r.calls.length, 1);
    });
    it('update --from-weights unloads then loads; remove and list still work', () => {
        const u = run(['update', 'ad1', '--from-weights', `${PFX}/v2/`]);
        assert.strictEqual(u.code, 0, u.stdout + u.stderr);
        assert.deepStrictEqual(u.calls, [`unload ad1 ${URL}`, `load ad1 ${POD}/v2 ${URL}`]);
        const rm = run(['remove', 'ad1']);
        assert.strictEqual(rm.code, 0);
        assert.deepStrictEqual(rm.calls, [`unload ad1 ${URL}`]);
        const ls = run(['list']);
        assert.strictEqual(ls.code, 0);
        assert.match(ls.stdout, /a1\s+\/opt\/ml\/model\/\.mlcc-adapters\/proj\/a1/);
    });
    it('rejects two sources at once and a missing source', () => {
        const both = run(['add', 'x', '--from-weights', 's3://b/w/', '--from-tune']);
        assert.strictEqual(both.code, 1);
        assert.match(both.stdout, /mutually exclusive/);
        const none = run(['add', 'x']);
        assert.strictEqual(none.code, 1);
        assert.match(none.stdout, /requires a source/);
    });
    it('a failed resolution stops before the cluster with the resolver error', () => {
        const r = run(['add', 'x', '--from-tune', 'dpo']);
        assert.strictEqual(r.code, 1);
        assert.match(r.stderr, /Could not resolve --from-tune/);
        assert.deepStrictEqual(r.calls, []);
    });
    describe('Wave 6.1 R1.4 — deployments that cannot take mounted adapters refuse with the fix', () => {
        it('an HF-sourced hyperpod-eks deployment points at do/stage + redeploy', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { STUB_SOURCE: 'huggingface' } });
            assert.strictEqual(r.code, 1, r.stdout + r.stderr);
            assert.match(r.stdout, /not served from S3/);
            assert.match(r.stdout, /\.\/do\/stage/);
            assert.deepStrictEqual(r.calls, []);
        });
        it('a deployment still on prefetch points at the LoRA redeploy (and --surge-node)', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { STUB_PREFETCH: 'true' } });
            assert.strictEqual(r.code, 1, r.stdout + r.stderr);
            assert.match(r.stdout, /prefetchEnabled: true/);
            assert.match(r.stdout, /--surge-node/);
            assert.deepStrictEqual(r.calls, []);
        });
        it('plain eks without the S3 model mount names EKS_MODEL_MOUNT=s3', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], {
                config: ['export DEPLOYMENT_TARGET="eks"', 'export DEPLOYMENT_TARGET_EKS_STATUS="Running"']
            });
            assert.strictEqual(r.code, 1, r.stdout + r.stderr);
            assert.match(r.stdout, /EKS_MODEL_MOUNT=s3/);
            assert.deepStrictEqual(r.calls, []);
        });
        it('plain eks WITH the S3 model mount maps under STAGED_MODEL_PATH', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], {
                config: ['export DEPLOYMENT_TARGET="eks"', 'export DEPLOYMENT_TARGET_EKS_STATUS="Running"',
                    'export EKS_MODEL_MOUNT="s3"', 'export STAGED_MODEL_PATH="s3://core/models/m/"']
            });
            assert.strictEqual(r.code, 0, r.stdout + r.stderr);
            assert.deepStrictEqual(r.calls, [`load ad1 ${POD}/ad1 ${URL}`]);
        });
    });
    describe('Wave 6.1 R1.6 — every Running pod', () => {
        it('loads on each pod and reports per pod', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { STUB_PODS: 'p0 p1' } });
            assert.strictEqual(r.code, 0, r.stdout + r.stderr);
            assert.deepStrictEqual(r.calls, [`load ad1 ${POD}/ad1 ${URL}`, `load ad1 ${POD}/ad1 ${URL}`]);
            assert.match(r.stdout, /pod\/p0/);
            assert.match(r.stdout, /pod\/p1/);
            assert.match(r.stdout, /2\/2 succeeded/);
        });
        it('a failure on any pod exits non-zero and names the pod', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { STUB_PODS: 'p0 p1', LORA_FAIL_CALL: '2' } });
            assert.strictEqual(r.code, 1, r.stdout + r.stderr);
            assert.strictEqual(r.calls.length, 2, 'the second pod is still attempted');
            assert.match(r.stdout, /1\/2 succeeded/);
            assert.match(r.stdout, /✗ pod\/p1/);
        });
        it('remove unloads from every pod', () => {
            const r = run(['remove', 'ad1'], { env: { STUB_PODS: 'p0 p1 p2' } });
            assert.strictEqual(r.code, 0, r.stdout + r.stderr);
            assert.strictEqual(r.calls.length, 3);
        });
    });
    describe('Wave 6.1 R3.2 — rank pre-check against the LIVE limit, before any engine call', () => {
        it('an adapter above the live max_lora_rank is refused with the exact fix', () => {
            const r = run(['add', 'big', '--from-weights', `${PFX}/big/`], { env: { STUB_RANK: '128', STUB_LIVE_RANK: '64' } });
            assert.strictEqual(r.code, 1, r.stdout + r.stderr);
            assert.deepStrictEqual(r.calls, [], 'the engine is never called');
            assert.match(r.stdout, /Rank check: adapter r128 vs serving max_lora_rank 64 \(pod env\)/);
            assert.match(r.stdout, /VLLM_MAX_LORA_RANK=128/);
            assert.match(r.stdout, /\.\/do\/deploy --target hyperpod-eks --force/);
        });
        it('an adapter within the live limit loads', () => {
            const r = run(['add', 'ok', '--from-weights', `${PFX}/ok/`], { env: { STUB_RANK: '32', STUB_LIVE_RANK: '64' } });
            assert.strictEqual(r.code, 0, r.stdout + r.stderr);
            assert.strictEqual(r.calls.length, 1);
        });
    });
    describe('BL165 — hot-load helper failures are surfaced (R9.1 / R8.6)', () => {
        it('a crashing lora_vllm.py shows its traceback + guidance (no silent exit)', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { LORA_STUB: 'crash' } });
            assert.strictEqual(r.code, 1);
            assert.match(r.stdout, /lora_vllm\.py load failed/);
            assert.match(r.stdout, /No module named 'requests'/);
            assert.match(r.stdout, /Failed to load LoRA adapter 'ad1'/);
        });
        it('a vLLM 409 is reported as already-exists (previously aborted by set -e before the message)', () => {
            const r = run(['add', 'ad1', '--from-weights', `${PFX}/ad1/`], { env: { LORA_STUB: '409' } });
            assert.strictEqual(r.code, 1);
            assert.match(r.stdout, /already exists/);
            assert.match(r.stdout, /do\/adapter remove ad1/);
        });
        it('a crashing list helper is an error, not "No LoRA adapters loaded"', () => {
            const r = run(['list'], { env: { LORA_STUB: 'crash' } });
            assert.strictEqual(r.code, 1);
            assert.match(r.stdout, /Could not list LoRA adapters/);
            assert.ok(!/No LoRA adapters loaded/.test(r.stdout));
        });
    });
});
describe('Wave 6 — do/adapter (SMAI) --weights deprecated alias (R2.2)', function () {
    this.timeout(60000);
    it('add --weights prints the notice and parses as --from-weights', () => {
        const sb = makeSandbox({
            scripts: ['adapter'],
            config: [ACTIVE_STATUS_CONFIG, 'export DEPLOYMENT_TARGET="realtime-inference"', 'export ENABLE_LORA="true"',
                'export PROJECT_NAME="proj"', 'export AWS_REGION="us-west-2"', ''].join('\n')
        });
        try {
            sb.write('bin/aws', AWS_STUB);
            const r = runBash(join(sb.doDir, 'adapter'), ['add', 'ad1', '--weights', 's3://b/a.tar.gz'], sandboxEnv(sb));
            assert.match(r.stderr, /--weights is deprecated; use --from-weights/);
            // It gets past parsing + URI validation to the base-IC lookup (no IC in the sandbox).
            assert.match(r.stdout, /Adding adapter: ad1/);
            assert.match(r.stderr, /Cannot determine base inference component/, 'BL165: the base-IC error must not be swallowed by $(...)');
        } finally {
            sb.cleanup();
        }
    });
});

describe('Wave 6 — do/add-ic on the resolver + contract (R6 / R2)', function () {
    this.timeout(60000);
    let sb;
    before(() => {
        sb = makeSandbox({ scripts: ['add-ic'] });
        sb.write('bin/aws', AWS_STUB);
        sb.write('do/.register_helper.py', REGISTER_STUB);
    });
    after(() => sb.cleanup());

    const run = (args, config = [], env = {}) => {
        sb.write('do/config', [ACTIVE_STATUS_CONFIG, 'export DEPLOYMENT_TARGET="realtime-inference"',
            'export PROJECT_NAME="proj"', 'export AWS_REGION="us-west-2"', ...config, ''].join('\n'), 0o644);
        // stdin is empty: the interactive prompts after resolution end the run.
        return runBash(join(sb.doDir, 'add-ic'), args, sandboxEnv(sb, env));
    };

    it('--from-weights and the deprecated --model-data resolve to the same model data', () => {
        const canonical = run(['ic1', '--from-weights', 's3://b/model.tar.gz']);
        const alias = run(['ic1', '--model-data', 's3://b/model.tar.gz']);
        const line = /📦 Using --from-weights model data: s3:\/\/b\/model\.tar\.gz/;
        assert.match(canonical.stdout, line);
        assert.match(alias.stdout, line);
        assert.ok(!/deprecated/.test(canonical.stderr));
        assert.match(alias.stderr, /--model-data is deprecated; use --from-weights/);
    });

    it('--from-tune with a full-model output is accepted', () => {
        const r = run(['ic1', '--from-tune'], ['export TUNE_OUTPUT_PATH_LATEST="s3://b/m/"', 'export TUNE_OUTPUT_TYPE_LATEST="model"']);
        assert.match(r.stdout, /📦 Using --from-tune model data: s3:\/\/b\/m\//);
    });

    it('a lora_adapter is refused with a pointer to do/adapter (exit 3)', () => {
        const r = run(['ic1', '--from-tune'], ['export TUNE_OUTPUT_PATH_LATEST="s3://b/a/"', 'export TUNE_OUTPUT_TYPE_LATEST="adapter"']);
        assert.strictEqual(r.code, 3, r.stdout + r.stderr);
        assert.match(r.stdout, /Contract violation: artifact-kind/);
        assert.match(r.stdout, /do\/adapter add <name> --from-tune/);
    });

    it('stretch R6.2: --from-registry resolves a full-model version; an adapter version is refused', () => {
        const ok = run(['ic1', '--from-registry', 'arn:aws:sagemaker:us-west-2:1:model-package/proj/5'], [],
            { STUB_KIND: 'full_model', STUB_URI: 's3://b/full/model.tar.gz' });
        assert.match(ok.stdout, /📦 Using --from-registry model data: s3:\/\/b\/full\/model\.tar\.gz/);
        const bad = run(['ic1', '--from-registry', 'arn:aws:sagemaker:us-west-2:1:model-package/proj/6'], [],
            { STUB_KIND: 'lora_adapter' });
        assert.strictEqual(bad.code, 3);
    });

    it('rejects two sources at once', () => {
        const r = run(['ic1', '--from-tune', '--from-weights', 's3://b/m/']);
        assert.strictEqual(r.code, 1);
        assert.match(r.stdout, /mutually exclusive/);
    });

    it('on hyperpod-eks the derived refusal points at do/adapter add <name> --from-tune (not --load-lora)', () => {
        sb.write('do/config', [ACTIVE_STATUS_CONFIG, 'export DEPLOYMENT_TARGET="hyperpod-eks"', ''].join('\n'), 0o644);
        const r = runBash(join(sb.doDir, 'add-ic'), ['ic1'], sandboxEnv(sb));
        assert.strictEqual(r.code, 3);
        assert.match(r.stdout, /do\/adapter add <name> --from-tune/);
        assert.ok(!/--load-lora/.test(r.stdout));
    });
});

describe('Wave 6 — R2.3 trap: --model-data-url is NOT renamed (regression guard)', () => {
    const register = readFileSync(join(REPO, 'templates/do/register'), 'utf8');
    const registerModel = readFileSync(join(REPO, 'templates/do/lib/python/register_model.py'), 'utf8');
    const registerHelper = readFileSync(join(REPO, 'templates/do/.register_helper.py'), 'utf8');

    it('do/register still forwards --model-data-url', () => {
        assert.ok(register.includes('"--model-data-url"'), 'do/register must keep --model-data-url');
        assert.ok(!register.includes('--from-weights'), 'the --from-weights rename must not reach do/register');
    });

    it('the register helper parser still defines --model-data-url (consumed by register_model.py)', () => {
        assert.match(registerHelper, /add_argument\("--model-data-url"/);
        assert.match(registerModel, /model_data_url/);
        assert.ok(!registerHelper.includes('--from-weights') && !registerModel.includes('--from-weights'));
    });
});
