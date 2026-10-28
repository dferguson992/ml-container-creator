// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * Wave 10 (ADR-014 R2.2-R2.4, R8.2) — do/deploy --scale-to-zero / --scale-max
 * validation and refusals, run against the REAL templates/do/deploy under
 * /bin/bash (3.2 on macOS) in a sandbox project with a stub `aws` that logs every
 * call. The disabled targets are found by iterating the descriptors (no target
 * literals), so a descriptor flip changes what this test expects.
 */
import { describe, it, afterEach } from 'mocha';
import assert from 'node:assert';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { makeSandbox, sandboxEnv, runBash, loadDescriptors, DO_SRC } from '../helpers/additive-sandbox.js';

const descriptors = loadDescriptors();
const disabled = descriptors.filter((d) => !d.scale_to_zero.enabled);
const enabled = descriptors.filter((d) => d.scale_to_zero.enabled);

describe('Wave 10 — do/deploy scale-to-zero flags (behavioral, bash 3.2)', function () {
    this.timeout(60000);
    let sb;
    afterEach(() => sb && sb.cleanup());

    function setup(target, extra = []) {
        sb = makeSandbox({ scripts: ['deploy'], pyHelpers: true });
        sb.write('do/lib/scale-to-zero.sh', readFileSync(join(DO_SRC, 'lib/scale-to-zero.sh')), 0o644);
        sb.write('do/lib/python/scale_to_zero.py', readFileSync(join(DO_SRC, 'lib/python/scale_to_zero.py')), 0o644);
        sb.write('do/config', [
            'export PROJECT_NAME="stz-test"',
            `export DEPLOYMENT_TARGET="${target}"`,
            'export INSTANCE_TYPE="ml.g5.xlarge"',
            ...extra,
            ''
        ].join('\n'), 0o644);
        // The driver must never run on a refusal; record it if it does.
        for (const d of descriptors) {
            sb.write(`do/${d.deploy_script}`, '#!/bin/bash\necho DRIVER_RAN >> "${SCRIPT_DIR}/../driver.log"\n');
        }
        sb.write('bin/aws', `#!/bin/bash\necho "aws $*" >> "${sb.root}/calls.log"\nexit 0\n`);
    }
    const calls = () => (existsSync(join(sb.root, 'calls.log')) ? readFileSync(join(sb.root, 'calls.log'), 'utf8') : '');
    const driverRan = () => existsSync(join(sb.root, 'driver.log'));
    const config = () => readFileSync(join(sb.doDir, 'config'), 'utf8');
    const deploy = (args) => runBash(join(sb.doDir, 'deploy'), args, sandboxEnv(sb, { HOME: sb.root }), { timeout: 50000 });

    it('at least one target refuses and one accepts (sanity on the descriptor set)', () => {
        assert.ok(disabled.length >= 1 && enabled.length >= 1);
    });

    for (const d of disabled) {
        for (const args of [['--scale-to-zero', 'on'], ['--scale-to-zero', 'off'], ['--scale-max', '2']]) {
            it(`${d.target}: ${args.join(' ')} exits 3 with the descriptor reason, before any AWS call`, () => {
                setup(d.target);
                const r = deploy(args);
                assert.strictEqual(r.code, 3, r.stdout + r.stderr);
                assert.ok(r.stdout.includes(d.scale_to_zero.reason), `reason not printed:\n${r.stdout}`);
                assert.strictEqual(calls(), '', 'no aws call');
                assert.ok(!driverRan(), 'deploy driver must not run');
                assert.ok(!/SCALE_TO_ZERO/.test(config()), 'nothing persisted');
            });
        }
    }

    for (const [args, what] of [
        [['--scale-to-zero', 'yes'], /--scale-to-zero must be 'on' or 'off'/],
        [['--scale-to-zero', 'ON'], /--scale-to-zero must be 'on' or 'off'/],
        [['--scale-max', '0'], /--scale-max must be an integer from 1 to 20/],
        [['--scale-max', '21'], /--scale-max must be an integer from 1 to 20/],
        [['--scale-max', '1.5'], /--scale-max must be an integer from 1 to 20/],
        [['--scale-max', 'two'], /--scale-max must be an integer from 1 to 20/]
    ]) {
        it(`a bad value (${args.join(' ')}) exits 1 before any AWS call, on every target`, () => {
            for (const d of descriptors) {
                setup(d.target);
                const r = deploy(args);
                assert.strictEqual(r.code, 1, `${d.target}: ${r.stdout}${r.stderr}`);
                assert.match(r.stderr, what);
                assert.strictEqual(calls(), '', `${d.target}: no aws call`);
                sb.cleanup();
            }
            sb = null;
        });
    }

    for (const d of enabled) {
        for (const [extra, what] of [
            [['export CAPACITY_RESERVATION_ARN="arn:aws:sagemaker:us-east-1:111122223333:ml-reservation/x"'], /CAPACITY_RESERVATION_ARN is set/],
            [['export INSTANCE_POOLS=\'[{"InstanceType":"ml.g5.xlarge","Priority":1}]\''], /INSTANCE_POOLS is set/],
            [['export ENDPOINT_STRATEGY="existing"'], /ENDPOINT_STRATEGY is 'existing'/],
            [['export ENDPOINT_EXTERNAL=true', 'export ENDPOINT_NAME="someone-elses-ep"'], /ENDPOINT_STRATEGY is/]
        ]) {
            it(`${d.target}: --scale-to-zero on is refused (exit 3) when ${extra[0].replace(/^export /, '').split('=')[0]} is set`, () => {
                setup(d.target, extra);
                const r = deploy(['--scale-to-zero', 'on']);
                assert.strictEqual(r.code, 3, r.stdout + r.stderr);
                assert.match(r.stdout, what);
                assert.strictEqual(calls(), '', 'no aws call');
                assert.ok(!driverRan());
            });
        }
        it(`${d.target}: --scale-to-zero off is not refused by those settings (turning it off is always allowed)`, () => {
            setup(d.target, ['export CAPACITY_RESERVATION_ARN="arn:aws:sagemaker:us-east-1:111122223333:ml-reservation/x"']);
            const r = deploy(['--scale-to-zero', 'off', '--dry-run']);
            assert.strictEqual(r.code, 0, r.stdout + r.stderr);
        });
    }
});
