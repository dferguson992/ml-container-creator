// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Integration (BL117/BL126): the managed-RL evaluator surface ships in a
 * generated project — the native evaluator modules, the `do/register prompt`
 * verb, the reworked `do/register evaluator` (code-based reward function), and
 * the `mtrl` technique on `do/tune`.
 *
 * Offline/generation-level only: no AWS, no RL job. Mirrors the GRPO recipe
 * integration test's generation pattern (runGenerator, --skip-prompts).
 */
import { describe, it, after } from 'mocha';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { runGenerator } from '../helpers/run-generator.js';

describe('Integration: managed-RL evaluators → do/register + do/tune', () => {
    const cleanups = [];
    after(() => {
        for (const c of cleanups) {
            try { c(); } catch { /* best-effort temp cleanup */ }
        }
    });

    // transformers-vllm is a representative training-bearing project (ships
    // do/tune + do/register + the full do/lib/python helper set).
    function generate() {
        const result = runGenerator({
            'deployment-config': 'transformers-vllm',
            'model-name': 'meta-llama/Llama-3.1-8B',
            'instance-type': 'ml.g5.xlarge'
        });
        cleanups.push(result.cleanup);
        return result;
    }

    it('ships the native evaluator Python modules', () => {
        const result = generate();
        assert.ok(existsSync(result.file('do/lib/python/register_evaluator.py')),
            'register_evaluator.py must ship');
        assert.ok(existsSync(result.file('do/lib/python/register_evaluator_sample.py')),
            'register_evaluator_sample.py must ship');
    });

    it('do/register exposes the prompt verb and a code-based evaluator verb', () => {
        const result = generate();
        const src = readFileSync(result.file('do/register'), 'utf8');
        // The subcommand router accepts `prompt`.
        assert.match(src, /prompt\)\s*SUBCOMMAND="prompt"/,
            'do/register must route the `prompt` subcommand');
        // The evaluator verb narrowed to the code-based reward-function path.
        assert.match(src, /register-evaluator/,
            'do/register must invoke the native register-evaluator helper');
        assert.match(src, /register-prompt/,
            'do/register must invoke the native register-prompt helper');
        // The retired local-stub shape must be gone from the exec path.
        assert.doesNotMatch(src, /--type\s+<lambda\|model>\s+--arn\s+<arn>\s+--technique\s+<rlvr\|rlaif>/,
            'the old stub usage line must be replaced');
    });

    it('do/register help lists all four verbs', () => {
        const result = generate();
        const src = readFileSync(result.file('do/register'), 'utf8');
        assert.match(src, /Supports: model \(default\), dataset, evaluator, prompt/,
            'the subcommand routing comment must list all four verbs');
    });

    it('do/tune accepts the mtrl technique and pairs techniques to evaluators', () => {
        const result = generate();
        const src = readFileSync(result.file('do/tune'), 'utf8');
        // The technique validator accepts mtrl.
        assert.match(src, /sft\|dpo\|rlaif\|rlvr\|mtrl/,
            'do/tune _validate_technique must accept mtrl');
        // The two-input RFT pre-flight exists and names the register verbs.
        assert.match(src, /_preflight_rft_inputs/,
            'do/tune must run the RFT two-input pre-flight');
        assert.match(src, /do\/register prompt/,
            'do/tune must point RLAIF users at do/register prompt');
        assert.match(src, /do\/register evaluator/,
            'do/tune must point RLVR/MTRL users at do/register evaluator');
    });

    it('non-RL machinery is unaffected — dataset verb and sft/dpo still present', () => {
        const result = generate();
        const src = readFileSync(result.file('do/register'), 'utf8');
        assert.match(src, /dataset\)\s*SUBCOMMAND="dataset"/,
            'the dataset verb must be unchanged');
        const tune = readFileSync(result.file('do/tune'), 'utf8');
        assert.match(tune, /--technique sft --dataset/,
            'sft dataset-only usage examples must remain');
    });
});
