// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Wave 6 — do/adapter (incl. BL169 on Kubernetes) and do/add-ic on the shared
 * resolver + contract, executed end-to-end in a sandbox with stubbed aws /
 * kubectl / lora_vllm.py / .register_helper.py.
 *
 *   R5 / BL169  the k8s path ACCEPTS --from-* (was refused), resolves through the
 *               shared resolver, and hot-loads RESOLVED_URI via lora_vllm.py
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
# Stub aws CLI: cluster lookups succeed, every S3 probe fails (advisory sniff → unknown).
case "$1 $2" in
  "sagemaker describe-cluster") echo "arn:aws:eks:us-west-2:111122223333:cluster/hp-eks" ;;
  "eks update-kubeconfig") exit 0 ;;
  *) echo "stub aws: $*" >&2; exit 1 ;;
esac
`;
const KUBECTL_STUB = `#!/bin/bash
case "$1" in
  get) echo "proj-pod-0" ;;
  port-forward) exec sleep 30 ;;
  *) exit 1 ;;
esac
`;
// lora_vllm.py stub: records every call; LORA_STUB selects the outcome.
const LORA_STUB = `#!/usr/bin/env python3
import json, os, sys
with open(os.environ["LORA_LOG"], "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
mode = os.environ.get("LORA_STUB", "ok")
cmd = sys.argv[1]
if mode == "crash":
    print("Traceback (most recent call last):\\nModuleNotFoundError: No module named 'requests'", file=sys.stderr)
    sys.exit(1)
if cmd == "list":
    print(json.dumps([{"name": "a1", "path": "s3://b/a1/"}]))
    sys.exit(0)
if mode == "409" and cmd == "load":
    print(json.dumps({"success": False, "error": "adapter exists", "status_code": 409}))
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
        ...extra,
        ''
    ].join('\n');
}

describe('Wave 6 — do/adapter on Kubernetes: --from-* hot-load (BL169 / R5)', function () {
    this.timeout(120000);
    let sb;
    let log;
    before(() => {
        sb = makeSandbox({ scripts: ['adapter'] });
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

    it('add --from-weights hot-loads the S3 URI straight through', () => {
        const r = run(['add', 'ad1', '--from-weights', 's3://b/w/']);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, ['load ad1 s3://b/w/ http://localhost:18080']);
        assert.match(r.stdout, /loaded from s3:\/\/b\/w\//);
    });

    it('--weights is a deprecated alias: same hot-load + the notice on stderr', () => {
        const r = run(['add', 'ad1', '--weights', 's3://b/w/']);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, ['load ad1 s3://b/w/ http://localhost:18080']);
        assert.match(r.stderr, /--weights is deprecated; use --from-weights/);
    });

    it('headline: add --from-tune hot-loads the tuned LoRA (no hand-copied URI)', () => {
        const r = run(['add', 'tuned', '--from-tune'], {
            config: ['export TUNE_OUTPUT_PATH_LATEST="s3://b/tune/out/"', 'export TUNE_OUTPUT_TYPE_LATEST="adapter"']
        });
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, ['load tuned s3://b/tune/out/ http://localhost:18080']);
        assert.ok(!/Contract violation/.test(r.stdout), 'the k8s source-verb refusal must be gone');
    });

    it('add --from-registry hot-loads the registered adapter version', () => {
        const r = run(['add', 'reg', '--from-registry', 'arn:aws:sagemaker:us-west-2:1:model-package/proj/3'],
            { env: { STUB_KIND: 'lora_adapter' } });
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.deepStrictEqual(r.calls, ['load reg s3://b/reg/adapter/ http://localhost:18080']);
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
        const r = run(['add', 'legacy', '--from-weights', 's3://b/legacy/']);
        assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        assert.match(r.stderr, /artifact_kind unknown/);
        assert.match(r.stdout, /Content check/);
        assert.strictEqual(r.calls.length, 1);
    });

    it('update --from-weights unloads then loads; remove and list still work', () => {
        const u = run(['update', 'ad1', '--from-weights', 's3://b/v2/']);
        assert.strictEqual(u.code, 0, u.stdout + u.stderr);
        assert.deepStrictEqual(u.calls, ['unload ad1 http://localhost:18080', 'load ad1 s3://b/v2/ http://localhost:18080']);
        const rm = run(['remove', 'ad1']);
        assert.strictEqual(rm.code, 0);
        assert.deepStrictEqual(rm.calls, ['unload ad1 http://localhost:18080']);
        const ls = run(['list']);
        assert.strictEqual(ls.code, 0);
        assert.match(ls.stdout, /a1\s+s3:\/\/b\/a1\//);
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

    describe('BL165 — hot-load helper failures are surfaced (R9.1 / R8.6)', () => {
        it('a crashing lora_vllm.py shows its traceback + guidance (no silent exit)', () => {
            const r = run(['add', 'ad1', '--from-weights', 's3://b/w/'], { env: { LORA_STUB: 'crash' } });
            assert.strictEqual(r.code, 1);
            assert.match(r.stdout, /lora_vllm\.py load failed/);
            assert.match(r.stdout, /No module named 'requests'/);
            assert.match(r.stdout, /Failed to load LoRA adapter 'ad1'/);
        });

        it('a vLLM 409 is reported as already-exists (previously aborted by set -e before the message)', () => {
            const r = run(['add', 'ad1', '--from-weights', 's3://b/w/'], { env: { LORA_STUB: '409' } });
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
