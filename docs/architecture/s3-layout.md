<!--
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
-->
# S3 buckets and layout

MLCC stores everything in a small set of S3 buckets. Each one is owned by a bootstrap
CloudFormation stack, declared once, and reached by consumers only through a key in
your bootstrap profile. This page covers which buckets exist, which stack owns each one,
how a bucket name reaches a `do/` script, and the one-time import that brings existing
buckets under CloudFormation.

## Buckets and owners

| Bucket (physical name pattern) | Owning stack / module | Profile key | `do/` env var | Holds |
|---|---|---|---|---|
| `mlcc-core-<acct>-<region>` | `core` | `coreS3Bucket` (alias `modelsS3Bucket`) | `CORE_S3_BUCKET` (alias `MODELS_S3_BUCKET`) | Shared, servable assets: `models/<org>/<base>/` staged weights, `datasets/<name>/` + sidecar, `reward-prompts/`, `hyperpod-lifecycle/`, `staging-jobs/`, async/batch default I/O |
| `mlcc-tune-<acct>-<region>`, called **artifacts** | `training` | `artifactsS3Bucket` | `ARTIFACTS_S3_BUCKET` (`TUNE_S3_BUCKET` is accepted only as an override alias) | Per-project raw job output: `<project>/tune/<technique>/<job>/`, `<project>/train/`, `<project>/optimize/`, and the MLflow App artifact store (`workspaces/…`) |
| `mlcc-benchmark-results-<acct>-<region>` | `benchmark` | `benchmarkS3Bucket` (alias `ciBenchmarkResultsBucket`) | `BENCHMARK_S3_BUCKET` | `<project>/bmk-*/`, `results/` (Glue table), `athena-query-results/` |
| `mlcc-codebuild-source-<acct>-<region>` | `ci` | `codebuildSourceS3Bucket` | `CODEBUILD_SOURCE_S3_BUCKET` | `do/submit` source archives (`<project>/…zip`) |
| `mlcc-training-<acct>-<region>` (legacy, frozen) | `training` | `trainingS3Bucket` | `S3_BUCKET` | Reward-evaluator code backups; kept so recorded URIs stay valid |
| `mlcc-training-adapters-<acct>-<region>` (legacy, frozen) | `training` | `adapterS3Bucket` | `ADAPTER_S3_BUCKET` | none for new writes (recorded URIs stay readable) |
| `mlcc-hyperpod-tls-<profile>` | `hyperpod` (inference-operator stack) | (none — passed to the add-on) | — | Inference-operator TLS material |

The physical name of the artifacts bucket is `mlcc-tune-…` because it already holds every
recorded tune output and the MLflow App's experiments are pinned to it (S3 buckets can't be
renamed). Everywhere except its stack declaration it is called "artifacts".

`mlcc-models-<acct>-<region>` (pre-BL072) is a legacy relic that no stack owns and nothing
points at; it is left in place. Deleting it is an owner decision.

### Bucket settings

Every bucket is `RemovalPolicy.RETAIN` (CloudFormation `DeletionPolicy: Retain` and
`UpdateReplacePolicy: Retain`), SSE-S3 encrypted, and blocks all public access.

| Bucket | Versioning | Lifecycle |
|---|---|---|
| core | Enabled | Expire noncurrent versions after 30 days; abort incomplete multipart uploads after 7 days |
| artifacts | **Off** (it holds hundreds of GiB of overwritten checkpoint state, and versioning can only be suspended, never removed) | Abort incomplete multipart uploads after 7 days only. Nothing expires job output or resume state |
| benchmark, ci, legacy training buckets | Enabled | none |

## One declared source

Each owning stack declares its buckets once, in a `buckets.json` next to the stack:

```json
{
  "stack": "training",
  "buckets": {
    "ArtifactsBucket": {
      "name": "mlcc-tune-{account}-{region}",
      "adoptContext": "adoptExistingArtifactsBucket",
      "output": "ArtifactsBucketOutput",
      "profileKey": "artifactsS3Bucket",
      "envVar": "ARTIFACTS_S3_BUCKET"
    }
  }
}
```

- The stack (`training/stack.ts`) renders the name with `renderBucketName()` and creates
  the bucket with `declaredBucket()` (`infra/bootstrap-modules/bucket-declarations.ts`).
- `module-runner.cjs` reads the same files through `bucket-declarations.cjs` to compute
  adopt flags and the bucket-import plan. The files are discovered, not listed by hand.
- Other stacks that grant access to a bucket (EKS IRSA and HyperPod node roles) render
  the name from the owning declaration, so the synthesized policy is identical.

### How a name reaches a `do/` script

```
buckets.json  →  CfnOutput (stack)  →  moduleOutputs.<module>.<Output>
              →  _denormalizeModuleOutputs  →  profile key (e.g. artifactsS3Bucket)
              →  do/lib/profile.sh  →  env var (e.g. ARTIFACTS_S3_BUCKET)
```

That chain is the only path. No `do/` script, Python helper, or generated notebook
rebuilds a bucket name from the account ID and region, and none has a literal
`${…:-mlcc-…}` fallback. A script that needs a bucket calls:

```bash
_require_bucket ARTIFACTS_S3_BUCKET || exit 1
```

When the profile key is missing, this fails fast and names the bootstrap module that
provides it (`ml-container-creator bootstrap add-module training`), or tells you to
refresh an already-provisioned module with `bootstrap update --module <m>`. The generated
deploy notebook reads `artifactsS3Bucket` from the active profile when the cell runs,
not when the notebook is generated.

Existing profiles created before the training stack exported `ArtifactsBucket` can get
the key without a deploy: `ml-container-creator bootstrap migrate` reads it from the
MLflow App's recorded `ArtifactStoreUri` (`describe-mlflow-app`). It never guesses the
name from a pattern.

### Drift test

`test/unit/no-literal-bucket-names.test.js` enforces the rules:

- It scans `templates/`, `src/lib/`, `servers/`, `scripts/`, and
  `infra/bootstrap-modules/` (comments included) for anything that builds an MLCC bucket
  name. The only allowed hits are the `"name"` lines in the `buckets.json` files, and
  each must exist exactly once.
- A behavioral check runs `profile.sh`, the `do/tune` output-bucket resolver, the
  notebook generator, and the MLflow provisioner against a profile with deliberately
  unconventional names (`zz-core-x`, `zz-art-y`, …). It then drops each key to prove
  the fail-fast message.
- A conformance check confirms every declared profile key maps from a stack output
  through `_denormalizeModuleOutputs` to a `profile.sh` export. It also confirms that
  `bin/app.ts` and the runner agree on adopt contexts.

## Ownership: adopt only what the stack doesn't own

A bootstrap run decides, for each declared bucket:

| State | Decision |
|---|---|
| The bucket is already a resource of this stack (`describe-stack-resources`) | Keep it as a resource. Never adopt it by reference. |
| The bucket exists but is not in the stack | Import it into the stack (see below). Adopt by reference (`fromBucketName`) only while the stack is first created, then import in the same run. |
| The bucket does not exist | The stack creates it. |

Earlier runners adopted by reference whenever `head-bucket` succeeded, even when the
stack itself owned the bucket. That dropped the bucket from the template, and
CloudFormation orphaned it (`DELETE_SKIPPED`). With the ownership check, a second
`bootstrap` never orphans a bucket. If the runner can't read the stack's resources, it
stops rather than guessing.

## Bringing existing buckets under CloudFormation (automatic)

Buckets created by older MLCC versions, or left behind by a module teardown (they are
`Retain`), exist but belong to no stack. There is no separate command for this:
`bootstrap update` and `bootstrap add-module` import every declared bucket that exists
but isn't a stack resource, as part of the normal deploy.

```bash
# 1. Preview (read-only): per bucket, owned / will be imported / will be created, plus
#    the cdk diff (template comparison). Creates no change set.
ml-container-creator bootstrap update --module training --dry-run

# 2. Import + apply, only for the modules that own buckets.
ml-container-creator bootstrap update --module training,core,benchmark,ci
```

`--module` takes one module or a comma-separated list. The listed modules deploy in
dependency order, and no other stack is touched (the EKS and HyperPod stacks only when
you list `eks` / `hyperpod`). Without `--module`, every provisioned module is updated.

For each stack, the runner:

1. Lists the declared buckets that exist but aren't stack resources, and prints one
   line per bucket (`📥 Bringing existing bucket s3://… under CloudFormation`).
2. Creates an **IMPORT** change set. Its template is the stack's *currently deployed*
   template plus one minimal `AWS::S3::Bucket` (`BucketName`, `Retain`) per bucket, with
   no other diff. Parameters use `UsePreviousValue`.
3. Verifies the change set contains only `Import` actions for exactly those buckets. If
   it contains anything else, it deletes the change set and stops.
4. Executes it and waits for `IMPORT_COMPLETE`.
5. Runs the normal deploy. The deploy applies the configured versioning, encryption,
   public-access block, lifecycle, and tags, plus any other pending template changes for
   that stack, such as removing the unused `mlcc-training-role`.

When the stack doesn't exist yet (a module re-added after teardown), the runner first
creates it referencing the retained bucket, then imports the bucket and deploys again,
so one command ends with the bucket owned. `bootstrap add-module` on a stack that already
exists also imports (then deploys once).

Any failure stops the run with the stack and bucket names, after deleting the change
set. The runner never falls back to adopting the bucket by reference. The step is
idempotent: an owned bucket is skipped, so a second `bootstrap update` shows "nothing to
import". Nothing is moved or deleted: every recorded URI (`TUNE_*` paths, model
packages, MLflow runs, legacy adapter paths) stays valid. Run it in each account and
region that has an MLCC profile.

Expected change-set contents for a typical account:

| Stack | Imported logical IDs |
|---|---|
| `mlcc-<profile>-training` | `ArtifactsBucket2AAC5544`, `TrainingBucketEB7BB5C9`, `AdaptersBucket91CC3455` |
| `mlcc-<profile>-core` | `ModelsBucketResource2D4F0B9A` |
| `mlcc-<profile>-benchmark` | `BenchmarkBucketDD7D1D2F` |
| `mlcc-<profile>-ci` | `SourceBucketResourceB217D9BF` |

The inference-operator stack's TLS bucket follows the same rule when the `hyperpod`
module is updated. The EKS and HyperPod cluster stacks declare no buckets.

## Prefix layout (canonical, Option D)

`templates/do/lib/python/s3_layout.py` is the single module that composes these keys
(shell callers use `do/lib/s3-layout.sh`); `test/unit/s3-layout-path-drift.test.js`
fails when any other file assembles them. Bucket names come only from profile keys.

| Bucket (profile key) | Prefix | What | Writer |
|---|---|---|---|
| core (`coreS3Bucket`) | `models/<org>/<base>/` | staged base weights | `do/stage` (adds only; never `--delete`) |
| core | `models/<org>/<base>/.mlcc-adapters/<project>/<name>/` | canonical LoRA adapter + `mlcc-adapter.json` | promotion in `do/tune` / `do/train`; `do/adapter add` (`--from-hub`, `--from-weights`, `--from-registry`, `--from-tune`, `--from-train`); `do/adapter promote` |
| core | `models/mlcc-<project>/<name>/` | full fine-tune + `mlcc-model.json` | promotion in `do/tune` / `do/train` |
| core | `evaluators/<name>/` | reward-function code backups (the reward-Lambda role reads this prefix) | `do/register evaluator --finalize` |
| artifacts (`artifactsS3Bucket`) | `<project>/tune\|train\|optimize/...` | raw job output, MLflow artifacts | the jobs |

`<org>/<base>` is the model's Hugging Face id; a hub/JumpStart id maps through the tune
catalog (`do/.tune_catalog.json`) and an unmapped id fails fast. The manifest records the
source URI, source job, technique, rank, target modules, base model (+ revision), a
content hash, the creator and a timestamp; promotion is idempotent on the hash and
removes stale files of the previous version of the SAME artifact.

Rules:

- **Promotion.** Jobs keep writing raw output to the artifacts bucket; the promotion step
  copies only the servable part (`checkpoints/hf/` for adapters, the merged/full model)
  server-side with your credentials. `TUNE_ADAPTER_PATH_<T>` / `TUNE_MODEL_PATH_<T>` record
  the canonical URI; `TUNE_OUTPUT_PATH_<T>` / `TUNE_OUTPUT_PATH_LATEST` keep the raw path. A
  failed promotion keeps the raw path and never fails the tune.
- **Backfill.** `./do/adapter promote <name> --from-tune <t>` (or `--from-weights <uri>`)
  publishes an existing output; `--update-config` repoints the do/config var and keeps the
  old value as `TUNE_ADAPTER_SOURCE_<T>` (or `TUNE_MODEL_SOURCE_<T>`).
- **Serving.** Realtime adapter ICs load the canonical prefix directly; the realtime model
  sync excludes `.mlcc-adapters/*`. On Kubernetes the canonical prefix is on the pod's model
  mount (see [LoRA delivery on Kubernetes](k8s-lora-delivery.md)); a serving copy is made
  only on a base-prefix mismatch with the same base and is deleted by `do/adapter remove`.
- **Cleanup.** `./do/clean adapters` deletes only this project's `.mlcc-adapters/<project>/`
  (under every base) and `models/mlcc-<project>/`. Never base weights, never other
  projects' adapters, never the artifacts bucket.
- **Legacy.** The frozen training / adapters buckets get no new writes; recorded URIs stay
  valid (nothing is moved). Already-recorded raw `TUNE_ADAPTER_PATH_*` values keep working:
  `do/adapter add` promotes them on first use.
