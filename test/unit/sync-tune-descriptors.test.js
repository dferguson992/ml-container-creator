// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * BL129 — sync-model-families tune-descriptor extension (tasks 5.2, 5.3).
 *
 * Feature: v2-w3-01-bl129
 *
 * Drives syncTuneDescriptors() with an INJECTED override-params fetcher (offline —
 * no S3, stays in the default mocha gate) against a TEMP copy of templates/code/tune.d.
 * Asserts the behaviors the sync must hold:
 *   - TAGGED flavor (sft/dpo/rlvr/rlaif): keep category:hyperparameter, drop system.
 *   - LEANER MTRL flavor (no category): keep all-but-denylist plumbing.
 *   - control_type→control, visibility_tier→tier rename; required defaults to false.
 *   - hp-eks recorded only when the pointer exists (MTRL None → omitted).
 *   - STRUCTURAL layer preserved; re-run idempotent (version unchanged → skipped).
 *   - the output descriptors STILL validate against the schema.
 *   - DENYLIST DRIFT (5.3): a NEW MTRL plumbing key not in the denylist leaks as a
 *     knob — the guard that forces a conscious denylist update.
 */

import assert from 'node:assert';
import { describe, it, beforeEach, afterEach } from 'mocha';
import {
    mkdtempSync, cpSync, readFileSync, rmSync, existsSync
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv from 'ajv/dist/2020.js';
import { syncTuneDescriptors } from '../../scripts/sync-model-families.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../..');
const SRC_TUNE_D = path.join(ROOT, 'templates', 'code', 'tune.d');

const schema = JSON.parse(readFileSync(path.join(SRC_TUNE_D, 'manifest.schema.json'), 'utf8'));
const validate = new Ajv({ allErrors: true, strict: false }).compile(schema);

// A category-TAGGED override-params object (sft/dpo/rlvr/rlaif flavor). Includes a
// category:system knob that MUST be filtered out.
const TAGGED_RLAIF = {
    learning_rate: { category: 'hyperparameter', type: 'number', control_type: 'number_input', visibility_tier: 'primary', default: 1e-5, min: 0, max: 1, required: true },
    judge_model_id: { category: 'hyperparameter', type: 'string', control_type: 'text_input', visibility_tier: 'primary', default: 'bedrock/deepseek.v3-v1:0', required: true },
    global_batch_size: { category: 'hyperparameter', type: 'integer', control_type: 'dropdown', visibility_tier: 'primary', enum: [128, 256, 512], default: 256, required: false },
    data_path: { category: 'system', type: 'string', default: '' },
    mlflow_tracking_uri: { category: 'system', type: 'string', default: '' }
};

// A LEANER MTRL override-params object (NO category on any knob). Includes the four
// denylisted plumbing keys that MUST be excluded.
const LEANER_MTRL = {
    advantage_method: { type: 'string', enum: ['grpo'], default: 'grpo', required: true },
    learning_rate: { type: 'number', min: 0, max: 1, default: 1e-6, required: false },
    group_size: { type: 'integer', min: 1, max: 64, default: 16, required: false },
    data_path: { type: 'string', default: '' },
    model_name_or_path: { type: 'string', default: '' },
    output_path: { type: 'string', default: '' },
    name: { type: 'string', default: '' }
};

// BL180: the fetcher now receives the RECIPE object ({technique, model, peft}) — the
// default fetcher reads the SMTJ SDK's _specs for that (technique, model). The injected
// test fetcher keys off recipe.technique (offline, no SDK/AWS) and returns a fixture
// options-dict in the SAME shape the real _specs/override-params produce.
function makeFetcher(overrides = {}) {
    const mtrl = overrides.mtrl || LEANER_MTRL;
    const rlaif = overrides.rlaif || TAGGED_RLAIF;
    return async (recipe) => ((recipe && recipe.technique === 'mtrl') ? mtrl : rlaif);
}

// Fixture recipe versions use a synthetic, deliberately-high version (v9x) so the
// sync's version-skip NEVER collides with whatever version the SHIPPED descriptors
// carry after a real `npm run sync:models` (BL173 populated them — the earlier
// fixtures pinned v2.2.7 / v1.0.22 and started colliding once the real rlaif
// descriptor also carried v2.2.7, silently skipping the fixture write). A version no
// real recipe will emit keeps these behavior tests independent of the committed data.
const RECIPES = [
    { technique: 'rlaif', peft: 'lora', model: 'test-model-rlaif', smtjOverrideParamsS3Uri: 's3://b/verl-grpo-rlaif-qwen-3-8b-lora_override_params_sm_jobs_v90.0.0.json', hpEksOverrideParamsS3Uri: 's3://b/verl-grpo-rlaif-qwen-3-8b-lora_override_params_k8s_v90.0.0.json' },
    { technique: 'mtrl', peft: 'lora', model: 'test-model-mtrl', smtjOverrideParamsS3Uri: 's3://b/mtrl-gpt-oss-20b-lora_override_params_sm_jobs_v90.0.0.json', hpEksOverrideParamsS3Uri: null }
];

describe('sync-tune-descriptors (tune-descriptor synced layer)', () => {
    let tmp;

    beforeEach(() => {
        tmp = mkdtempSync(path.join(os.tmpdir(), 'mcc-tuned-'));
        cpSync(SRC_TUNE_D, tmp, { recursive: true });
    });

    afterEach(() => {
        if (tmp && existsSync(tmp)) rmSync(tmp, { recursive: true, force: true });
    });

    function readVariant(technique, variant = 'lora') {
        return JSON.parse(readFileSync(path.join(tmp, technique, 'manifest.json'), 'utf8'))
            .variants[variant];
    }

    it('tagged flavor keeps category:hyperparameter and drops system knobs', async () => {
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        const hp = readVariant('rlaif').hyperparameters;
        assert.deepStrictEqual(Object.keys(hp).sort(),
            ['global_batch_size', 'judge_model_id', 'learning_rate']);
        assert.ok(!('data_path' in hp), 'category:system data_path filtered out');
        assert.ok(!('mlflow_tracking_uri' in hp), 'category:system mlflow filtered out');
    });

    it('renames control_type→control and visibility_tier→tier; carries enum/default', async () => {
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        const gbs = readVariant('rlaif').hyperparameters.global_batch_size;
        assert.strictEqual(gbs.control, 'dropdown');
        assert.strictEqual(gbs.tier, 'primary');
        assert.deepStrictEqual(gbs.enum, [128, 256, 512]);
        assert.strictEqual(gbs.default, 256);
        assert.strictEqual(gbs.required, false);
    });

    it('leaner MTRL flavor keeps all-but-denylist plumbing and omits control/tier', async () => {
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        const hp = readVariant('mtrl').hyperparameters;
        assert.deepStrictEqual(Object.keys(hp).sort(),
            ['advantage_method', 'group_size', 'learning_rate']);
        for (const plumb of ['data_path', 'model_name_or_path', 'output_path', 'name']) {
            assert.ok(!(plumb in hp), `${plumb} denylisted`);
        }
        // MTRL knobs carry no control/tier (the source has none; BL128 derives them).
        assert.ok(!('control' in hp.learning_rate));
        assert.ok(!('tier' in hp.learning_rate));
    });

    it('records hp-eks only when the pointer exists (MTRL None → omitted)', async () => {
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        const rlaif = readVariant('rlaif')._synced;
        assert.strictEqual(rlaif.smtj.version, '90.0.0');
        assert.strictEqual(rlaif['hp-eks'].version, '90.0.0');
        const mtrl = readVariant('mtrl')._synced;
        assert.strictEqual(mtrl.smtj.version, '90.0.0');
        assert.ok(!('hp-eks' in mtrl), 'MTRL has no hp-eks backend');
    });

    it('preserves the STRUCTURAL layer and keeps the descriptor schema-valid', async () => {
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        for (const technique of ['rlaif', 'mtrl']) {
            const manifest = JSON.parse(readFileSync(path.join(tmp, technique, 'manifest.json'), 'utf8'));
            assert.ok(validate(manifest), `${technique} invalid after sync: ${JSON.stringify(validate.errors)}`);
            // structural fields untouched
            const v = manifest.variants.lora;
            assert.strictEqual(v.peft, 'lora');
            assert.strictEqual(v.artifact_kind, 'lora_adapter');
            assert.ok(v.dataset_contract && v.dataset_contract.schema);
        }
        assert.strictEqual(
            JSON.parse(readFileSync(path.join(tmp, 'mtrl', 'manifest.json'), 'utf8')).reward_source,
            'agent_env');
    });

    it('is idempotent — a re-run with the same version skips (no rewrite)', async () => {
        const first = await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        assert.strictEqual(first.synced, 2);
        const second = await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        assert.strictEqual(second.synced, 0);
        assert.strictEqual(second.skipped, 2);
    });

    it('leaves the existing synced layer intact on a fetch failure (never regress)', async () => {
        // First a good sync, then a failing fetcher on a BUMPED version — the prior
        // synced block must survive.
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: makeFetcher(), tuneDir: tmp });
        const before = readVariant('rlaif').hyperparameters;
        const bumped = RECIPES.map(r => r.technique === 'rlaif'
            ? { ...r, smtjOverrideParamsS3Uri: r.smtjOverrideParamsS3Uri.replace('v90.0.0', 'v90.0.1') }
            : r);
        const failing = async () => { throw new Error('S3 down'); };
        const res = await syncTuneDescriptors(bumped, { fetchOverrideParams: failing, tuneDir: tmp });
        assert.ok(res.failed >= 1);
        assert.deepStrictEqual(readVariant('rlaif').hyperparameters, before,
            'prior synced layer preserved on fetch failure');
    });

    // ── 5.3 denylist drift guard ────────────────────────────────────────────────
    it('a NEW MTRL plumbing key NOT in the denylist leaks as a knob (drift guard)', async () => {
        // This test documents + guards the denylist: if AWS adds a new plumbing key to
        // the leaner MTRL schema, it will surface as a knob until the denylist is
        // updated. The test asserts the CURRENT behavior (leak) so a maintainer adding
        // the key to the denylist sees this test flip — forcing a conscious decision.
        const withNewPlumbing = {
            ...LEANER_MTRL,
            checkpoint_dir: { type: 'string', default: '' } // hypothetical new plumbing key
        };
        await syncTuneDescriptors(RECIPES, {
            fetchOverrideParams: makeFetcher({ mtrl: withNewPlumbing }),
            tuneDir: tmp
        });
        const hp = readVariant('mtrl').hyperparameters;
        assert.ok('checkpoint_dir' in hp,
            'a new, non-denylisted plumbing key currently LEAKS as a knob — add it to ' +
            'MTRL_PLUMBING_DENYLIST in sync-model-families.js, then update this guard.');
    });

    // BL173 regression: AWS override-params use raw type strings (`float`, `array`,
    // …) that are NOT in the descriptor schema's type enum (string|integer|number|
    // boolean). The sync MUST normalize them, else it writes a schema-invalid
    // descriptor — the exact failure that shipped empty-then-invalid before BL173.
    // (The prior fixtures only used already-normalized types, which is why this
    // slipped through; this fixture uses the raw AWS types on purpose.)
    it('normalizes raw AWS knob types (float to number, array to string) to the schema enum', async () => {
        const RAW_TYPES = {
            lr: { category: 'hyperparameter', type: 'float', control_type: 'number_input', visibility_tier: 'primary', default: 1e-5, required: true },
            epochs: { category: 'hyperparameter', type: 'integer', control_type: 'number_input', visibility_tier: 'primary', default: 3, required: false },
            target_modules: { category: 'hyperparameter', type: 'array', control_type: 'text_input', visibility_tier: 'advanced', required: false },
            use_flash: { category: 'hyperparameter', type: 'boolean', control_type: 'dropdown', visibility_tier: 'advanced', default: true, required: false }
        };
        // The shared RECIPES already use a synthetic v90 version that never collides
        // with the shipped descriptor, so the sync always writes the fixture here.
        await syncTuneDescriptors(RECIPES, {
            fetchOverrideParams: makeFetcher({ rlaif: RAW_TYPES }),
            tuneDir: tmp
        });
        const hp = readVariant('rlaif').hyperparameters;
        assert.strictEqual(hp.lr.type, 'number', 'float must normalize to number');
        assert.strictEqual(hp.target_modules.type, 'string', 'array must normalize to string');
        assert.strictEqual(hp.epochs.type, 'integer', 'integer passes through');
        assert.strictEqual(hp.use_flash.type, 'boolean', 'boolean passes through');
        const allowed = new Set(['string', 'integer', 'number', 'boolean']);
        for (const [kn, ke] of Object.entries(hp)) {
            assert.ok(allowed.has(ke.type), `${kn}.type='${ke.type}' not in the schema enum`);
        }
    });

    // BL180: the fetcher must receive the RECIPE object (technique + model), NOT the
    // raw S3 uri — the authoritative source is the SMTJ SDK's _specs for that
    // (technique, model), read by constructing the trainer. Guard the contract so a
    // regression back to uri-based fetching (the stale-recipe source) is caught.
    it('passes the recipe (technique + model) to the fetcher, not the raw uri', async () => {
        const seen = [];
        const fetcher = async (arg) => {
            seen.push(arg);
            return arg && arg.technique === 'mtrl' ? LEANER_MTRL : TAGGED_RLAIF;
        };
        await syncTuneDescriptors(RECIPES, { fetchOverrideParams: fetcher, tuneDir: tmp });
        assert.ok(seen.length >= 1, 'fetcher was invoked');
        for (const arg of seen) {
            assert.strictEqual(typeof arg, 'object',
                'fetcher received an object (the recipe), not a string uri');
            assert.ok(arg.technique, 'recipe carries technique');
            assert.ok(arg.model, 'recipe carries model (needed to construct the SDK trainer)');
        }
    });
});
