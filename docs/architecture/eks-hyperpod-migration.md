# EKS / HyperPod Migration & Live-Rehearsal Findings (BL143 / Wave 2)

This page is the findings-and-migration record for the Wave 2 EKS/HyperPod module
split (BL143). It satisfies the spec's "migration doc" requirement (Req 3.5 — IAM
teardown permissions) and additionally captures everything the **live sim-account
rehearsal** surfaced during bring-up, so the next operator (or a first-time user)
does not rediscover these the hard way.

For the per-role IAM reference and the CDK adopted-role limitation, see
[Bootstrap IAM Roles](../dev/bootstrap-iam-roles.md).

---

## TL;DR for a first-time user

- `bootstrap add-module eks` stands up a vanilla EKS cluster you can deploy to with
  `./do/deploy --target eks`. **Vanilla-EKS users stop here.**
- `bootstrap add-module hyperpod` layers the HyperPod cluster + Inference Operator on
  top. The operator's control plane runs on a **CPU** floor (2 × `ml.c5.4xlarge`) — it
  does **not** need a GPU. GPU compute for model serving is attached on demand as a
  separate instance group.
- A fresh bootstrap on current code includes all the IAM fixes below. The manual
  recovery steps are only for deployments provisioned **before** these fixes landed.

---

## IAM teardown permissions (Req 3.5)

The cutover teardown (`remove-module eks` / `remove-module hyperpod`, or a manual
stack destroy) exercises more IAM than provisioning. A first-time user's role that can
provision may still fail to tear down. The teardown path needs, at minimum:

| Service | Actions (teardown) | Why |
|---|---|---|
| CloudFormation | `cloudformation:DeleteStack`, `DescribeStack*`, `DeleteChangeSet` | Each module is a CFN stack |
| EKS | `eks:DeleteCluster`, `eks:DeleteNodegroup`, `eks:DeleteAddon`, `eks:DeleteFargateProfile`, `eks:Describe*`, `eks:List*` | Cluster + Fargate profile + add-ons |
| SageMaker | `sagemaker:DeleteCluster`, `sagemaker:DescribeCluster`, `sagemaker:UpdateCluster` | HyperPod cluster + instance groups |
| Lambda | `lambda:DeleteFunction`, `lambda:GetFunction`, `lambda:InvokeFunction` | CDK custom-resource providers (e.g. the EKS `KubectlProvider`/`ClusterResourceProvider` nested stacks) |
| IAM | `iam:DeleteRole`, `iam:DeleteRolePolicy`, `iam:DetachRolePolicy`, `iam:DeleteOpenIDConnectProvider` | Roles/OIDC the stacks created (except RETAIN'd service roles) |
| EC2 | `ec2:DeleteVpc`, `ec2:DeleteSubnet`, `ec2:DeleteSecurityGroup`, `ec2:DeleteNatGateway`, `ec2:ReleaseAddress`, ENI deletes | VPC + networking |
| S3 | `s3:DeleteBucket`, `s3:DeleteObject` | Non-RETAIN'd buckets (the TLS bucket is RETAIN'd and is NOT deleted automatically — see Orphan cleanup) |

!!! warning "Cross-account AssumeRole caveat (sim-account finding)"
    In the sim-account run, teardown failed with `AssumeRoleFailed` because the caller
    role in one account could not assume the CDK deploy role
    (`cdk-hnb659fds-deploy-role-...`) in a different account. If you provision and tear
    down across accounts, the caller must be allowed to assume the CDK execution role in
    the target account. This is an environment/trust configuration issue, not a code
    bug — but it blocks teardown silently if missing.

### Orphan cleanup after teardown

`remove-module` always purges the module from the bootstrap profile — even if the CDK
teardown fails — so the profile never carries stale `_PROFILE_*` keys pointing at
destroyed resources. But some resources are intentionally **RETAIN**'d and survive
teardown; clean them up manually if you want a truly empty account:

- RETAIN'd service IAM roles (`eks-cluster-role`, `eks-node-role`,
  `hyperpod-instance-role`, `fargate-pod-exec-role`) — use `remove-module eks
  --force-delete` to also destroy them.
- RETAIN'd buckets: the HyperPod TLS bucket (`mlcc-hyperpod-tls-<profile>`) and the
  core models bucket (`mlcc-core-<account>-<region>`).

---

## Bring-up findings (from the live rehearsal)

These are the issues the live run surfaced and the fixes that shipped. Each is already
in the net-new provision path; the manual steps are recovery for pre-fix deployments.

### 1. Adopted service roles silently drift (the big one)

RETAIN'd service roles are adopted (imported) on re-provision. CDK **cannot** set the
trust policy of an imported role and **cannot** attach a managed policy to one. So an
adopted role froze at whatever it had when first created and drifted out of spec as the
stack evolved. HyperPod then rejected the instance role with **"execution role
permissions are invalid"** — and, confusingly, reported the downstream lifecycle-script
read failure as **"S3 bucket is not owned by your account. Cross-account S3 buckets are
not allowed"** (a same-account bucket). That misleading error cost real debugging time.

**Fix:** the adopt path imports mutably and re-asserts inline policies; the module
runner reconciles the trust policy (`sagemaker` + `ec2` principals) and the managed
policies at provision time (CDK can't). See
[Bootstrap IAM Roles](../dev/bootstrap-iam-roles.md) for the full mechanism and the
manual recovery commands.

!!! tip "Debugging rule"
    If HyperPod reports an invalid execution role OR an S3 "not owned by your account"
    error, check the role's **trust policy and attached policies first** — both symptoms
    trace to the same under-scoped role.

### 2. Inference Operator needs CPU, not GPU

The HyperPod Inference Operator's control plane (gateway controller, controller-manager,
ALB, KEDA — ~13 pods per the AWS operator-install minimum) is CPU/orchestration work.
AWS's documented minimum is `ml.c5.4xlarge`. MLCC's `default-worker` floor is therefore
**2 × `ml.c5.4xlarge` (CPU)**, not GPU. GPU nodes are attached on demand only for
model-serving pods. This decouples "operator healthy" from "GPU capacity available" —
important because GPU capacity is frequently constrained (see finding 5).

### 3. Inference Operator v3.0 S3/ACM tagging permissions

The operator (v3.0+) tags the ALB, S3, and ACM resources it creates, and its startup
validation writes a **tagged** object to its TLS bucket. The custom
`hyperpod-inference-role` therefore needs, beyond `AmazonSageMakerHyperPodInferenceAccess`:

- `s3:PutObject` **and** `s3:PutObjectTagging` on `mlcc-hyperpod-tls-<profile>` (a tagged
  PutObject without the tagging action fails `AccessDenied` on `s3:PutObjectTagging`),
  plus `s3:GetObject` / `s3:ListBucket`.
- `acm:AddTagsToCertificate` on `arn:aws:acm:*:*:certificate/*`.

These are now attached unconditionally (the inference role is an IRSA role, always
freshly created, never adopted — so no reconciler is needed). Pre-fix deployments
self-heal on the next `add-module hyperpod`, or attach the two inline policies manually.

### 4. HyperPod lifecycle script must exist in S3

Every EKS-orchestrated instance group requires a `LifeCycleConfig`. The module uploads a
no-op `on_create.sh` to `s3://mlcc-core-<account>-<region>/hyperpod-lifecycle/` before
cluster create. A manual `update-cluster` that adds an instance group must point
`SourceS3Uri` at that prefix (the directory) with `OnCreate: on_create.sh` (the
filename) — splitting the full object path into the wrong fields fails validation.

### 5. GPU capacity is the gating constraint

GPU instance groups retry until capacity is available in the region; during the
rehearsal, `ml.g5.8xlarge` / `ml.g6.2xlarge` capacity came and went. The CPU operator
floor comes up reliably regardless, so the cluster + operator stay healthy while a GPU
group waits. For model serving, be prepared to wait for capacity or try an alternate GPU
family. This is an AWS capacity reality, not a code limitation.

### 6. The ALB controller runs in the operator's namespace — IRSA trust must allow it

The sharpest find of the rehearsal. The eks module installs the AWS Load Balancer
Controller in `kube-system` (its Helm chart) and binds `alb-controller-role`'s IRSA
trust to `system:serviceaccount:kube-system:aws-load-balancer-controller`. But the
**HyperPod Inference Operator add-on installs its OWN ALB controller** in
`hyperpod-inference-system` (deployment `hyperpod-inference-alb`), under the same SA
name, annotated with that same `alb-controller-role` — and ITS webhook
(`vtargetgroupbinding.elbv2.k8s.aws`, registered as `alb-webhook`) is the one that gates
ingress reconciliation.

That operator-namespace controller presents a token with
`sub = system:serviceaccount:hyperpod-inference-system:aws-load-balancer-controller`,
which a `kube-system`-pinned `StringEquals` trust **rejects**:

```
STS AssumeRoleWithWebIdentity 403 AccessDenied: Not authorized to perform
sts:AssumeRoleWithWebIdentity
```

Result: the ALB never provisions → the ingress (`alb-<endpoint>`) never gets an
`ADDRESS` → the operator's TLS/cert step blocks on "no ingress found in loadbalancer
status" → the `InferenceEndpointConfig` hangs at `DeploymentInProgress` **even though the
model pod is `3/3 Running` and serving 200s on `/ping`**.

!!! warning "Debugging note — this one is deceptive"
    Every individual component verifies correct: OIDC provider registered, role trust
    well-formed, SA annotated, pod IRSA env + token audience (`sts.amazonaws.com`) all
    right. A `kube-system` token even assumes the role fine via the CLI. The ONLY tell is
    that the failing webhook's `clientConfig.service` points at `hyperpod-inference-system`,
    not `kube-system` — i.e. a **second** ALB controller in the operator's namespace. If
    an endpoint hangs at `DeploymentInProgress` with the model pod healthy, check
    `kubectl get validatingwebhookconfiguration alb-webhook -o json` for which namespace
    serves the `elbv2` webhooks.

**Fix:** `alb-controller-role` now trusts the SA **name in any namespace** (a `StringLike`
`system:serviceaccount:*:aws-load-balancer-controller`, via `_createIrsaRole`'s
`saNameAnyNamespace=true`) rather than pinning `kube-system`. This covers whichever
controller — the eks stack's or the operator's — actually serves the ingress. The other
IRSA roles stay exact-namespace-pinned (their namespaces are stack-controlled). Pre-fix
clusters: update the live role's trust to the wildcard sub, then restart the operator's
`hyperpod-inference-alb` deployment so it re-exchanges its token.

---

## Adopted-role reconciliation covers all four service roles

The provision-time reconciler (`_ensureAdoptedServiceRoles` in `module-runner.cjs`)
re-asserts the trust policy **and** managed policies for **every** RETAIN'd service role
(`eks-cluster-role`, `eks-node-role`, `hyperpod-instance-role`, `fargate-pod-exec-role`)
after the eks-cluster stack deploys — not just the HyperPod instance role. The desired
trust principals + managed policies live in one `ADOPTED_SERVICE_ROLES` table, and a
drift test (`hyperpod-cdk-stacks.test.js`) asserts each entry mirrors the stack's
`_createOrAdoptRole(...)` call, so a stack change that isn't reflected in the runner
fails loudly. See [Bootstrap IAM Roles](../dev/bootstrap-iam-roles.md).
