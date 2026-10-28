#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""hyperpod_ops.py — hyperpod-eks rollout capacity, surge node, contract preflight.

Wave 6.1 R4 (owner decision D2: "temporarily add a node; MLCC must not introduce
logic that deliberately causes an outage") and R6 (read-only contract preflight).
Called by do/deploy.d/hyperpod-eks. stdlib only; every cloud/cluster call goes
through the `aws` and `kubectl` CLIs (stubbed on PATH in tests).

Subcommands (all print one JSON object on stdout; human notes on stderr):

  rollout-capacity --cluster C --namespace N --project P --instance-type T --gpus G
      R4.1, read-only. Can a SURGE pod (the operator's RollingUpdate surges 1 and
      keeps the old pod until the new one is Ready) schedule? Looks at every Ready,
      schedulable node of the IEC's instance type and its free nvidia.com/gpu
      (allocatable minus the requests of non-terminal pods). Reports the instance
      group (node label sagemaker.amazonaws.com/instance-group-name) and its
      Current/Target count from DescribeCluster.  exit 0 = schedulable,
      3 = not schedulable, 1 = could not determine.

  surge-up --cluster C --group G
      R4.3 opt-in. UpdateCluster with the group's TargetCount + 1, copying every
      other group field from DescribeCluster. Prints {"original_target": N, ...}.
      A capacity error (InsufficientCapacity...) is printed VERBATIM; exit 1.

  surge-wait --cluster C --group G --count N [--timeout S]
      Wait for the group to reach N InService instances AND N Ready k8s nodes.
      A group failure message (e.g. InsufficientCapacity) is printed verbatim.

  rollout-wait --namespace N --project P --replicas R [--timeout S]
      Wait until app=<P> has exactly R pods, all Ready, none terminating (the old
      pod is gone).

  surge-restore --cluster C --group G --original-count N --namespace NS --project P
      Restore the group to N instances WITHOUT touching a node that hosts a Ready
      serving pod: BatchDeleteClusterNodes on the vacated node (UpdateCluster's
      own scale-down picks instances at random, which could take the new serving
      pod down — R4.2). If the surge instance never joined, UpdateCluster sets the
      target back to N instead. Never goes below N; never scales to zero.

  contract --verb train|tune|add-pod [--group G] [--instance-type T] [--gpus N]
           [--dataset s3://...] [--job J] [--storage-class SC] [--model-bucket B]
           [--template-gpus N --template-efa N] [--adapt-shape] [--allow-surge] [--json]
      Wave 11 R2, read-only: the verb-scoped cluster contract (training-operator,
      pod-identity, scratch-volume, free-gpus, gpu-shape-fit, local-queue,
      s3-csi-role). Prints {"findings": [...], "resolved": {...}, "blocked": b};
      exit 4 on any block finding (the verb stops before any cluster write).

  preflight-env   (same preflight; inputs from do/config env — do/validate; also
                   reports the hp-eks training elements capped at warn)
  preflight --cluster C --namespace NS --project P --model-source S
            [--model-bucket B] [--lora-delivery true|false] [--eks-cluster E]
      R6, read-only: S3 CSI driver present; the model source is S3 when LoRA
      delivery needs the mount; the CSI driver's IAM role can read the model
      bucket (iam simulate-principal-policy) when resolvable. Report, don't mutate:
      always exit 0 with {"findings": [...]}.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
INSTANCE_TYPE_LABEL = "node.kubernetes.io/instance-type"
GPU_RESOURCE = "nvidia.com/gpu"
S3_CSI_DRIVER = "s3.csi.aws.com"
S3_CSI_ADDON = "aws-mountpoint-s3-csi-driver"
# ClusterInstanceGroupSpecification members that DescribeCluster's
# ClusterInstanceGroupDetails also returns (from the botocore sagemaker service
# model): copied verbatim so a surge never changes anything but the count.
_COPY_FIELDS = (
    "InstanceGroupName", "InstanceType", "InstanceRequirements", "LifeCycleConfig",
    "ExecutionRole", "ThreadsPerCore", "InstanceStorageConfigs", "OnStartDeepHealthChecks",
    "TrainingPlanArn", "OverrideVpcConfig", "ScheduledUpdateConfig", "AutoPatchConfig",
    "KubernetesConfig", "SlurmConfig", "CapacityRequirements", "NetworkInterface",
)


# ── CLI plumbing ───────────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd[:4])}"
    return p.returncode, p.stdout or "", p.stderr or ""


def _aws(args: list[str], timeout: int = 120) -> tuple[int, str, str]:
    cmd = ["aws", *args]
    if os.environ.get("AWS_REGION"):
        cmd += ["--region", os.environ["AWS_REGION"]]
    profile = os.environ.get("_PROFILE_awsProfile") or os.environ.get("AWS_PROFILE")
    if profile:
        cmd += ["--profile", profile]
    return _run(cmd, timeout)


def _json(text: str):
    try:
        return json.loads(text or "")
    except ValueError:
        return None


def _kubectl_json(args: list[str]) -> dict | None:
    rc, out, _ = _run(["kubectl", *args, "-o", "json"])
    data = _json(out) if rc == 0 else None
    return data if isinstance(data, dict) else None


def _note(msg: str) -> None:
    print(msg, file=sys.stderr)


def _emit(obj: dict) -> None:
    print(json.dumps(obj))


# ── Cluster model ──────────────────────────────────────────────────────────────


def describe_group(cluster: str, group: str) -> tuple[dict | None, str]:
    rc, out, err = _aws(["sagemaker", "describe-cluster", "--cluster-name", cluster, "--output", "json"])
    if rc != 0:
        return None, err.strip() or out.strip()
    data = _json(out) or {}
    for g in data.get("InstanceGroups") or []:
        if g.get("InstanceGroupName") == group:
            return g, ""
    return None, f"instance group '{group}' not found in cluster '{cluster}'"


def _gpu_qty(resources: dict | None) -> int:
    if not isinstance(resources, dict):
        return 0
    for section in ("requests", "limits"):
        val = (resources.get(section) or {}).get(GPU_RESOURCE)
        if val is not None:
            try:
                return int(str(val))
            except ValueError:
                return 0
    return 0


def _pod_gpus(pod: dict) -> int:
    spec = pod.get("spec") or {}
    main = sum(_gpu_qty(c.get("resources")) for c in spec.get("containers") or [])
    init = max([_gpu_qty(c.get("resources")) for c in spec.get("initContainers") or []] or [0])
    return max(main, init)


def _node_ready(node: dict) -> bool:
    if (node.get("spec") or {}).get("unschedulable"):
        return False
    for cond in (node.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def _pod_ready(pod: dict) -> bool:
    if (pod.get("metadata") or {}).get("deletionTimestamp"):
        return False
    if (pod.get("status") or {}).get("phase") != "Running":
        return False
    for cond in (pod.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def _node_instance_id(node: dict) -> str:
    """HyperPod node → EC2 instance id: providerID ends with ...-i-<hex>, the
    node name is hyperpod-i-<hex>."""
    for src in ((node.get("spec") or {}).get("providerID") or "", (node.get("metadata") or {}).get("name") or ""):
        idx = src.rfind("i-")
        if idx >= 0:
            return src[idx:]
    return ""


def cluster_view(instance_type: str | None = None, group: str | None = None) -> tuple[list[dict], list[dict]] | None:
    selector = []
    if group:
        selector = ["-l", f"{GROUP_LABEL}={group}"]
    elif instance_type:
        selector = ["-l", f"{INSTANCE_TYPE_LABEL}={instance_type}"]
    nodes = _kubectl_json(["get", "nodes", *selector])
    pods = _kubectl_json(["get", "pods", "--all-namespaces"])
    if nodes is None or pods is None:
        return None
    return nodes.get("items") or [], pods.get("items") or []


def free_gpus(nodes: list[dict], pods: list[dict]) -> list[dict]:
    used: dict[str, int] = {}
    for p in pods:
        phase = (p.get("status") or {}).get("phase")
        node = (p.get("spec") or {}).get("nodeName")
        if not node or phase in ("Succeeded", "Failed"):
            continue
        used[node] = used.get(node, 0) + _pod_gpus(p)
    out = []
    for n in nodes:
        name = (n.get("metadata") or {}).get("name", "")
        try:
            alloc = int(str(((n.get("status") or {}).get("allocatable") or {}).get(GPU_RESOURCE, "0")))
        except ValueError:
            alloc = 0
        out.append({
            "name": name,
            "group": ((n.get("metadata") or {}).get("labels") or {}).get(GROUP_LABEL, ""),
            "ready": _node_ready(n),
            "allocatable": alloc,
            "free": alloc - used.get(name, 0),
            "instance_id": _node_instance_id(n),
        })
    return out


def _serving_pods(pods: list[dict], namespace: str, project: str) -> list[dict]:
    return [p for p in pods
            if (p.get("metadata") or {}).get("namespace") == namespace
            and ((p.get("metadata") or {}).get("labels") or {}).get("app") == project]


# ── R4.1 rollout capacity ──────────────────────────────────────────────────────


def rollout_capacity(cluster: str, namespace: str, project: str, instance_type: str, gpus: int) -> tuple[int, dict]:
    view = cluster_view(instance_type=instance_type)
    if view is None:
        return 1, {"schedulable": None, "reason": "could not read nodes/pods with kubectl"}
    nodes, pods = view
    report = free_gpus(nodes, pods)
    eligible = [n for n in report if n["ready"]]
    schedulable = any(n["free"] >= gpus for n in eligible)
    serving_nodes = {(p.get("spec") or {}).get("nodeName") for p in _serving_pods(pods, namespace, project)}
    groups = [n["group"] for n in report if n["name"] in serving_nodes and n["group"]] or \
             [n["group"] for n in report if n["group"]]
    group = groups[0] if groups else ""
    result = {
        "schedulable": schedulable,
        "gpus_needed": gpus,
        "instance_type": instance_type,
        "nodes": report,
        "group": group,
        "current_count": None,
        "target_count": None,
    }
    if group:
        g, err = describe_group(cluster, group)
        if g:
            result["current_count"] = g.get("CurrentCount")
            result["target_count"] = g.get("TargetCount")
        else:
            result["describe_error"] = err
    result["single_node"] = result["current_count"] == 1
    if schedulable:
        result["reason"] = "a node of this instance type has enough free GPUs for the surge pod"
        return 0, result
    result["reason"] = (
        f"no Ready {instance_type} node has {gpus} free GPU(s) for the rollout's surge pod "
        f"(the operator keeps the old pod until the new one is Ready, so it would stay Pending)"
    )
    return 3, result


# ── R4.3 surge node ────────────────────────────────────────────────────────────


def _group_spec(g: dict, count: int) -> dict:
    spec = {k: g[k] for k in _COPY_FIELDS if k in g and g[k] is not None}
    if g.get("MinCount") is not None:
        spec["MinInstanceCount"] = g["MinCount"]
    image = g.get("DesiredImageId")
    if image and image != "default":
        spec["ImageId"] = image
    spec["InstanceCount"] = count
    return spec


def update_group_count(cluster: str, g: dict, count: int) -> tuple[int, str]:
    if count < 1:
        return 2, "refusing to scale an instance group below 1 (MLCC never scales serving capacity to zero)"
    payload = {"ClusterName": cluster, "InstanceGroups": [_group_spec(g, count)]}
    rc, out, err = _aws(["sagemaker", "update-cluster", "--cli-input-json", json.dumps(payload), "--output", "json"])
    return rc, (err.strip() or out.strip())


def surge_up(cluster: str, group: str) -> tuple[int, dict]:
    g, err = describe_group(cluster, group)
    if not g:
        return 1, {"error": err}
    original = int(g.get("TargetCount") or g.get("CurrentCount") or 0)
    rc, msg = update_group_count(cluster, g, original + 1)
    if rc != 0:
        # Verbatim, e.g. "An error occurred (InsufficientCapacity...) ...".
        _note(msg)
        return 1, {"error": msg, "original_target": original}
    return 0, {"original_target": original, "new_target": original + 1, "group": group}


def surge_wait(cluster: str, group: str, count: int, timeout: int, interval: int) -> tuple[int, dict]:
    deadline = time.time() + timeout
    last = {}
    while True:
        g, err = describe_group(cluster, group)
        if g:
            last = {"current": g.get("CurrentCount"), "target": g.get("TargetCount"), "status": g.get("Status")}
            failure = g.get("FailureMessage") or ""
            if failure:
                _note(failure)  # verbatim (e.g. InsufficientCapacity)
                if "InsufficientCapacity" in failure or str(g.get("Status")) in ("Failed", "Degraded"):
                    return 1, {"error": failure, **last}
            view = cluster_view(group=group)
            ready = sum(1 for n in free_gpus(*view) if n["ready"]) if view else 0
            last["ready_nodes"] = ready
            if (g.get("CurrentCount") or 0) >= count and ready >= count:
                return 0, last
        elif err:
            last = {"error": err}
        if time.time() >= deadline:
            return 1, {"error": f"timed out after {timeout}s waiting for {count} Ready node(s) in '{group}'", **last}
        time.sleep(interval)


def rollout_wait(namespace: str, project: str, replicas: int, timeout: int, interval: int) -> tuple[int, dict]:
    deadline = time.time() + timeout
    last = {}
    while True:
        data = _kubectl_json(["get", "pods", "-n", namespace, "-l", f"app={project}"])
        pods = (data or {}).get("items") or []
        ready = [p for p in pods if _pod_ready(p)]
        last = {"pods": len(pods), "ready": len(ready)}
        if data is not None and len(pods) == replicas and len(ready) == replicas:
            return 0, last
        if time.time() >= deadline:
            return 1, {"error": f"timed out after {timeout}s waiting for the old pod to go and {replicas} Ready pod(s)", **last}
        time.sleep(interval)


def surge_restore(cluster: str, group: str, original: int, namespace: str, project: str) -> tuple[int, dict]:
    g, err = describe_group(cluster, group)
    if not g:
        return 1, {"error": err}
    target = int(g.get("TargetCount") or 0)
    if target <= original:
        return 0, {"restored": True, "target": target, "action": "none"}
    view = cluster_view(group=group)
    if view is None:
        return 1, {"error": "could not read the group's nodes; restore manually", "target": target}
    nodes, pods = view
    serving = _serving_pods(pods, namespace, project)
    ready_on = {}
    for p in serving:
        node = (p.get("spec") or {}).get("nodeName")
        if node and _pod_ready(p):
            ready_on[node] = ready_on.get(node, 0) + 1
    report = free_gpus(nodes, pods)
    extra = target - original
    if len(report) <= original:
        # The surge instance never joined: lower the target back, no node to remove.
        rc, msg = update_group_count(cluster, g, original)
        return (0 if rc == 0 else 1), {"restored": rc == 0, "action": "update-cluster", "target": original,
                                      **({} if rc == 0 else {"error": msg})}
    # Remove the node(s) hosting the FEWEST Ready serving pods (the vacated old
    # node after a rollout, or the empty surge node after a failure). Deleting a
    # node with a Ready serving pod is only ever chosen when another Ready one
    # remains elsewhere.
    ranked = sorted(report, key=lambda n: (ready_on.get(n["name"], 0), n["name"]))
    victims = []
    remaining_ready = sum(ready_on.values())
    for n in ranked:
        if len(victims) >= extra:
            break
        hosted = ready_on.get(n["name"], 0)
        if hosted and remaining_ready - hosted < 1:
            continue
        if not n["instance_id"]:
            continue
        victims.append(n)
        remaining_ready -= hosted
    if len(victims) < extra:
        return 1, {"error": "no node can be removed without taking the last Ready serving pod down; "
                            "restore manually once the rollout settles", "target": target}
    ids = [v["instance_id"] for v in victims]
    rc, out, err = _aws(["sagemaker", "batch-delete-cluster-nodes", "--cluster-name", cluster,
                         "--node-ids", *ids, "--output", "json"])
    resp = _json(out) or {}
    failed = resp.get("Failed") or []
    if rc != 0 or failed:
        return 1, {"error": (err.strip() or json.dumps(failed)), "node_ids": ids}
    return 0, {"restored": True, "action": "batch-delete-cluster-nodes", "node_ids": ids,
               "nodes": [v["name"] for v in victims], "target": original}


# ── R6 preflight ───────────────────────────────────────────────────────────────


def _finding(sev: str, check: str, message: str, fix: str = "") -> dict:
    return {"severity": sev, "check": check, "message": message, "fix": fix}


def preflight(cluster: str, namespace: str, project: str, model_source: str,
              model_bucket: str, lora_delivery: bool, eks_cluster: str) -> dict:
    findings = []
    rc, out, err = _run(["kubectl", "get", "csidriver", S3_CSI_DRIVER, "-o", "name"])
    csi_present = rc == 0 and S3_CSI_DRIVER in out
    needs_s3 = model_source == "s3"
    if csi_present:
        findings.append(_finding("info", "s3-csi-driver", f"{S3_CSI_DRIVER} CSI driver is installed"))
    elif rc == 0 or "NotFound" in err or "not found" in err:
        findings.append(_finding(
            "warn" if needs_s3 else "info", "s3-csi-driver",
            f"{S3_CSI_DRIVER} CSI driver not found; the operator mounts S3 model sources with it",
            "Install the Mountpoint for Amazon S3 CSI driver add-on on the cluster (operator / bootstrap step; MLCC does not install it).",
        ))
    else:
        findings.append(_finding("info", "s3-csi-driver", f"could not check CSI drivers: {err.strip()[:200]}"))

    if lora_delivery and not needs_s3:
        findings.append(_finding(
            "warn", "model-source",
            f"LoRA is on but the model source is '{model_source}', not S3: do/adapter can only deliver "
            f"adapters through the S3 model mount",
            "Stage the model to S3 (./do/stage) and redeploy, so adapters can be placed under the model prefix.",
        ))
    elif needs_s3:
        findings.append(_finding("info", "model-source", "model source is S3"))

    role = ""
    if eks_cluster:
        rc, out, _ = _aws(["eks", "describe-addon", "--cluster-name", eks_cluster, "--addon-name", S3_CSI_ADDON,
                           "--query", "addon.serviceAccountRoleArn", "--output", "text"])
        if rc == 0 and out.strip() not in ("", "None"):
            role = out.strip()
    if not role:
        sa = _kubectl_json(["get", "sa", "-n", "kube-system", "s3-csi-driver-sa"])
        role = (((sa or {}).get("metadata") or {}).get("annotations") or {}).get("eks.amazonaws.com/role-arn", "")
    if needs_s3 and model_bucket and role:
        rc, out, err = _aws([
            "iam", "simulate-principal-policy", "--policy-source-arn", role,
            "--action-names", "s3:GetObject", "s3:ListBucket",
            "--resource-arns", f"arn:aws:s3:::{model_bucket}", f"arn:aws:s3:::{model_bucket}/*",
            "--output", "json"])
        data = _json(out) if rc == 0 else None
        if data:
            decisions = {}
            for r in data.get("EvaluationResults") or []:
                action = r.get("EvalActionName")
                res = r.get("EvalResourceName", "")
                ok_res = res.endswith("/*") if action == "s3:GetObject" else not res.endswith("/*")
                if ok_res:
                    decisions[action] = r.get("EvalDecision")
            denied = [a for a, d in decisions.items() if d != "allowed"]
            if denied:
                findings.append(_finding(
                    "warn", "s3-csi-role",
                    f"the S3 CSI role {role} is not allowed {', '.join(sorted(denied))} on s3://{model_bucket}",
                    f"Grant s3:GetObject and s3:ListBucket on {model_bucket} to the CSI driver role (cluster owner).",
                ))
            else:
                findings.append(_finding("info", "s3-csi-role", f"S3 CSI role can read s3://{model_bucket}"))
        else:
            findings.append(_finding("info", "s3-csi-role",
                                     f"could not simulate the S3 CSI role's access: {err.strip()[:200]}"))
    elif needs_s3:
        findings.append(_finding("info", "s3-csi-role", "S3 CSI role not resolvable; bucket access not simulated"))
    return {"findings": findings}


# ── Wave 11 R2: verb-scoped cluster contract ──────────────────────────────────
#
# One read-only entry (`contract --verb train|tune|add-pod`) evaluates exactly the
# elements the verb needs and returns the preflight findings shape plus a
# `resolved` block (EKS cluster, chosen group / node / GPUs, scratch mode) that the
# submit path consumes so nothing is looked up twice. A `block` finding means the
# verb must stop (exit 4) BEFORE any cluster write. MLCC never performs a fix
# (ADR-012): each finding names the owner's command. A transport / permission
# error on a hard element fails CLOSED (block), never open.

HPPJ_CRD = "hyperpodpytorchjobs.sagemaker.amazonaws.com"
TRAINING_OPERATOR_NAMESPACE = "aws-hyperpod"
TRAINING_OPERATOR_ADDON = "amazon-sagemaker-hyperpod-training-operator"
TRAIN_SERVICE_ACCOUNT = "mlcc-train-sa"
FSX_CLAIM = "fsx-claim"
EFA_RESOURCE = "vpc.amazonaws.com/efa"
DEFAULT_SC_ANNOTATION = "storageclass.kubernetes.io/is-default-class"
CONTRACT_ELEMENTS = {
    "train": ("training-operator", "pod-identity", "scratch-volume", "free-gpus", "local-queue"),
    "tune": ("training-operator", "pod-identity", "scratch-volume", "free-gpus", "gpu-shape-fit", "local-queue"),
    "add-pod": ("free-gpus", "local-queue", "s3-csi-role"),
}


def _not_found(err: str) -> bool:
    return "NotFound" in err or "not found" in err


def association_command(eks_cluster: str, namespace: str, role_arn: str, region: str) -> str:
    """The ONE command the owner runs to bind mlcc-train-sa to the training role
    (OQ-11.4). MLCC prints it; it never runs it."""
    return (f"aws eks create-pod-identity-association --cluster-name {eks_cluster or '<eks-cluster>'} "
            f"--namespace {namespace} --service-account {TRAIN_SERVICE_ACCOUNT} "
            f"--role-arn {role_arn or '<MLCC_HP_TRAIN_ROLE_ARN>'} --region {region or '<region>'}")


def eks_cluster_name(cluster: str, env: dict) -> tuple[str, str]:
    """The EKS cluster behind a HyperPod cluster: EKS_CLUSTER_NAME when set, else
    DescribeCluster Orchestrator.Eks.ClusterArn. Returns (name, error)."""
    explicit = (env.get("EKS_CLUSTER_NAME") or "").strip()
    if explicit:
        return explicit, ""
    if not cluster:
        return "", "HP_CLUSTER_NAME is not set"
    rc, out, err = _aws(["sagemaker", "describe-cluster", "--cluster-name", cluster,
                         "--query", "Orchestrator.Eks.ClusterArn", "--output", "text"])
    arn = out.strip()
    if rc != 0 or not arn or arn == "None":
        return "", (err.strip() or f"cluster '{cluster}' has no EKS orchestrator")
    return arn.rsplit("/", 1)[-1], ""


def _node_rows(nodes: list[dict], pods: list[dict]) -> list[dict]:
    """free_gpus() rows plus the instance type and EFA allocatable per node."""
    by_name = {(n.get("metadata") or {}).get("name", ""): n for n in nodes}
    rows = free_gpus(nodes, pods)
    for r in rows:
        n = by_name.get(r["name"]) or {}
        labels = (n.get("metadata") or {}).get("labels") or {}
        r["instance_type"] = labels.get(INSTANCE_TYPE_LABEL, "")
        try:
            r["efa"] = int(str(((n.get("status") or {}).get("allocatable") or {}).get(EFA_RESOURCE, "0")))
        except ValueError:
            r["efa"] = 0
    return rows


def _gpu_pods_on(pods: list[dict], node_names: set) -> list[str]:
    out = []
    for p in pods:
        if (p.get("spec") or {}).get("nodeName") not in node_names:
            continue
        if (p.get("status") or {}).get("phase") in ("Succeeded", "Failed") or _pod_gpus(p) <= 0:
            continue
        md = p.get("metadata") or {}
        out.append(f"{md.get('namespace', '')}/{md.get('name', '')}")
    return sorted(out)


def pick_group(rows: list[dict], group: str = "", instance_type: str = "", gpus: int = 0) -> tuple[dict | None, str]:
    """R4.2 group pick over live nodes: the named group (--instance-group /
    TRAIN_HP_INSTANCE_GROUP) else the first name-sorted Ready GPU group with a
    node that has the requested GPUs free (a WHOLE free node when --gpus is not
    given), filtered by instance type. Never a hardcoded group or SKU."""
    cands = [r for r in rows if r["ready"] and r["allocatable"] > 0 and r["group"]]
    if group:
        cands = [r for r in cands if r["group"] == group]
        if not cands:
            return None, f"instance group '{group}' has no Ready GPU node"
    if instance_type:
        cands = [r for r in cands if r["instance_type"] == instance_type]
        if not cands:
            return None, f"no Ready GPU node of instance type {instance_type}" + (f" in group '{group}'" if group else "")
    for r in sorted(cands, key=lambda x: (x["group"], x["name"])):
        if gpus and gpus > r["allocatable"]:
            continue
        need = gpus or r["allocatable"]
        if r["free"] >= need:
            return dict(r, gpus=need), ""
    if gpus and all(gpus > r["allocatable"] for r in cands):
        return None, f"--gpus {gpus} exceeds every candidate node's allocatable nvidia.com/gpu"
    return None, "no candidate node has " + (f"{gpus} free GPU(s)" if gpus else "a whole free GPU node")


def _check_training_operator(findings: list) -> None:
    rc, out, err = _run(["kubectl", "get", "crd", HPPJ_CRD, "-o", "name"])
    fix = (f"Install the {TRAINING_OPERATOR_ADDON} EKS add-on on the cluster "
           f"(cluster owner; MLCC never installs operators, ADR-012).")
    if rc != 0:
        if _not_found(err):
            findings.append(_finding("block", "training-operator", f"CRD {HPPJ_CRD} is not installed", fix))
        else:
            findings.append(_finding("block", "training-operator",
                                     f"could not read CRD {HPPJ_CRD}: {err.strip()[:200]}",
                                     "Check kubectl access to the cluster (KUBECONFIG / EKS access entry)."))
        return
    deps = _kubectl_json(["get", "deployments", "-n", TRAINING_OPERATOR_NAMESPACE])
    if deps is None:
        findings.append(_finding("block", "training-operator",
                                 f"could not list Deployments in {TRAINING_OPERATOR_NAMESPACE}", fix))
        return
    ops = [d for d in deps.get("items") or [] if "training" in ((d.get("metadata") or {}).get("name") or "")]
    available = [d for d in ops if any(c.get("type") == "Available" and c.get("status") == "True"
                                       for c in (d.get("status") or {}).get("conditions") or [])]
    if available:
        findings.append(_finding("info", "training-operator",
                                 f"HyperPod training operator is Available ({available[0]['metadata']['name']})"))
    else:
        findings.append(_finding("block", "training-operator",
                                 f"no Available HyperPod training-operator Deployment in {TRAINING_OPERATOR_NAMESPACE}", fix))


def _simulate(role: str, action: str, resource: str) -> tuple[str, str]:
    rc, out, err = _aws(["iam", "simulate-principal-policy", "--policy-source-arn", role,
                         "--action-names", action, "--resource-arns", resource, "--output", "json"])
    data = _json(out) if rc == 0 else None
    if not data:
        return "error", err.strip()[:200]
    results = data.get("EvaluationResults") or []
    return (results[0].get("EvalDecision") if results else "error"), ""


def _check_pod_identity(findings: list, resolved: dict, a: dict, env: dict) -> None:
    role = (env.get("MLCC_HP_TRAIN_ROLE_ARN") or "").strip()
    ns, region = a["namespace"], env.get("AWS_REGION", "")
    eks, err = eks_cluster_name(a["cluster"], env)
    resolved["eks_cluster"] = eks
    if not role:
        findings.append(_finding(
            "block", "pod-identity",
            "MLCC_HP_TRAIN_ROLE_ARN is empty: the bootstrap profile has no HyperPod training role",
            "Create it with the training module: ml-container-creator bootstrap update --module training"))
        return
    cmd = association_command(eks, ns, role, region)
    if not eks:
        findings.append(_finding("block", "pod-identity", f"could not resolve the EKS cluster: {err}", cmd))
        return
    rc, out, err = _aws(["eks", "list-pod-identity-associations", "--cluster-name", eks, "--namespace", ns,
                         "--service-account", TRAIN_SERVICE_ACCOUNT, "--output", "json"])
    data = _json(out) if rc == 0 else None
    if data is None:
        findings.append(_finding("block", "pod-identity",
                                 f"could not list pod identity associations: {err.strip()[:200]}", cmd))
        return
    assocs = data.get("associations") or []
    if not assocs:
        findings.append(_finding(
            "block", "pod-identity",
            f"no EKS Pod Identity association for {ns}/{TRAIN_SERVICE_ACCOUNT}: HyperPod training pods "
            f"have no AWS credentials without it (cluster owner runs the command below; ADR-012)", cmd))
        return
    rc, out, err = _aws(["eks", "describe-pod-identity-association", "--cluster-name", eks,
                         "--association-id", str(assocs[0].get("associationId", "")), "--output", "json"])
    got = (((_json(out) or {}).get("association") or {}).get("roleArn") or "") if rc == 0 else ""
    if rc != 0:
        findings.append(_finding("block", "pod-identity",
                                 f"could not describe the pod identity association: {err.strip()[:200]}", cmd))
        return
    if got != role:
        findings.append(_finding(
            "block", "pod-identity",
            f"{ns}/{TRAIN_SERVICE_ACCOUNT} is associated with {got or '<none>'}, not MLCC_HP_TRAIN_ROLE_ARN {role}",
            "Delete the mismatched association (aws eks delete-pod-identity-association --cluster-name "
            f"{eks} --association-id {assocs[0].get('associationId', '')} --region {region or '<region>'}), "
            f"then: {cmd}"))
        return
    probes = []
    dataset = a.get("dataset") or ""
    if dataset.startswith("s3://"):
        key = dataset[len("s3://"):]
        probes.append(("s3:GetObject", f"arn:aws:s3:::{key}{'probe' if key.endswith('/') else ''}", f"read {dataset}"))
    artifacts = (env.get("ARTIFACTS_S3_BUCKET") or "").strip()
    if artifacts and a.get("project"):
        out_key = f"{artifacts}/{a['project']}/{a['kind']}/{a.get('job') or 'mlcc-probe'}/probe"
        probes.append(("s3:PutObject", f"arn:aws:s3:::{out_key}", f"write s3://{out_key.rsplit('/', 1)[0]}/"))
    denied = []
    for action, res, what in probes:
        decision, serr = _simulate(role, action, res)
        if decision != "allowed":
            denied.append(f"{action} ({what}): {decision}{' ' + serr if serr else ''}")
    if denied:
        findings.append(_finding(
            "block", "pod-identity",
            f"the training role {role} is not allowed: {'; '.join(denied)}",
            "Re-run the training module (ml-container-creator bootstrap update --module training); "
            "datasets must live under the core bucket's datasets/ prefix."))
        return
    findings.append(_finding("info", "pod-identity",
                             f"{ns}/{TRAIN_SERVICE_ACCOUNT} → {role}" + (" (dataset read + artifacts write simulated)" if probes else "")))


def _check_scratch(findings: list, resolved: dict, a: dict) -> None:
    ns = a["namespace"]
    rc, out, err = _run(["kubectl", "get", "pvc", FSX_CLAIM, "-n", ns, "-o", "json"])
    pvc = _json(out) if rc == 0 else None
    if rc != 0 and not _not_found(err):
        findings.append(_finding("block", "scratch-volume", f"could not read PVC {ns}/{FSX_CLAIM}: {err.strip()[:200]}",
                                 "Check kubectl access to the cluster."))
        return
    if isinstance(pvc, dict) and ((pvc.get("status") or {}).get("phase") == "Bound"):
        resolved["scratch"] = "fsx"
        findings.append(_finding("info", "scratch-volume",
                                 f"PVC {ns}/{FSX_CLAIM} is Bound: job scratch is its mlcc/<project>/<job>/ subpath"))
        return
    if a["verb"] != "train":
        findings.append(_finding(
            "block", "scratch-volume", f"PVC {ns}/{FSX_CLAIM} is not Bound (the AWS HP-EKS recipes need it)",
            f"Create the FSx for Lustre PV/PVC '{FSX_CLAIM}' in {ns} (cluster owner; AWS recipe prerequisite)."))
        return
    scs = _kubectl_json(["get", "storageclass"])
    if scs is None:
        findings.append(_finding("block", "scratch-volume", "could not list StorageClasses", "Check kubectl access."))
        return
    items = scs.get("items") or []
    want = a.get("storage_class") or ""
    chosen = ""
    if want:
        chosen = want if any((i.get("metadata") or {}).get("name") == want for i in items) else ""
    else:
        for i in items:
            if ((i.get("metadata") or {}).get("annotations") or {}).get(DEFAULT_SC_ANNOTATION) == "true":
                chosen = i["metadata"]["name"]
                break
    if chosen:
        resolved["scratch"] = "ephemeral"
        resolved["storage_class"] = chosen
        findings.append(_finding("info", "scratch-volume",
                                 f"no Bound {FSX_CLAIM}; job scratch is a generic ephemeral volume on StorageClass {chosen}"))
        return
    findings.append(_finding(
        "block", "scratch-volume",
        f"no Bound PVC {ns}/{FSX_CLAIM} and " + (f"StorageClass '{want}' not found" if want else "no default StorageClass"),
        f"Create the '{FSX_CLAIM}' FSx PVC, or a StorageClass (pass it with --storage-class); "
        "MLCC never writes training output to the container overlay."))


def _check_free_gpus(findings: list, resolved: dict, a: dict) -> None:
    view = cluster_view(instance_type=None if a.get("group") else (a.get("instance_type") or None),
                        group=a.get("group") or None)
    if view is None:
        findings.append(_finding("block", "free-gpus", "could not read nodes/pods with kubectl", "Check kubectl access."))
        return
    nodes, pods = view
    rows = _node_rows(nodes, pods)
    if a["verb"] == "add-pod":
        need = int(a.get("gpus") or 1)
        ok = [r for r in rows if r["ready"] and r["free"] >= need
              and (not a.get("instance_type") or r["instance_type"] == a["instance_type"])]
        if ok:
            resolved["node"], resolved["group"] = ok[0]["name"], ok[0]["group"]
            findings.append(_finding("info", "free-gpus", f"{ok[0]['name']} ({ok[0]['group']}) has {ok[0]['free']} free GPU(s)"))
            return
        busy = _gpu_pods_on(pods, {r["name"] for r in rows})
        findings.append(_finding(
            "warn" if a.get("allow_surge") else "block", "free-gpus",
            f"no Ready node has {need} free GPU(s)" + (f"; GPU pods: {', '.join(busy)}" if busy else ""),
            "Free a node, pick another --instance-group, or pass --surge-node to add one."))
        return
    choice, reason = pick_group(rows, a.get("group") or "", a.get("instance_type") or "", int(a.get("gpus") or 0))
    if choice:
        resolved.update({"group": choice["group"], "node": choice["name"], "gpus": choice["gpus"],
                         "allocatable": choice["allocatable"], "efa": choice["efa"],
                         "instance_type": choice["instance_type"]})
        findings.append(_finding("info", "free-gpus",
                                 f"group {choice['group']} node {choice['name']}: {choice['gpus']} of "
                                 f"{choice['allocatable']} GPU(s) free"))
        return
    busy = _gpu_pods_on(pods, {r["name"] for r in rows if r["allocatable"] > 0})
    findings.append(_finding(
        "block", "free-gpus", reason + (f"; GPU pods holding them: {', '.join(busy)}" if busy else ""),
        "Free a GPU node, or name another group with --instance-group (never a serving group you need)."))


def _check_shape_fit(findings: list, resolved: dict, a: dict) -> None:
    t_gpu, t_efa = a.get("template_gpus"), a.get("template_efa") or 0
    if t_gpu is None or "allocatable" not in resolved:
        findings.append(_finding("info", "gpu-shape-fit", "no template shape to compare (render not reached)"))
        return
    g_gpu, g_efa = resolved["allocatable"], resolved.get("efa", 0)
    if t_gpu <= g_gpu and t_efa <= g_efa:
        findings.append(_finding("info", "gpu-shape-fit",
                                 f"template shape ({t_gpu} GPU, {t_efa} EFA) fits group {resolved['group']}"))
        return
    msg = (f"the AWS recipe requests {t_gpu} GPU / {t_efa} EFA per node; group {resolved['group']} "
           f"has {g_gpu} GPU / {g_efa} EFA")
    if a.get("adapt_shape"):
        findings.append(_finding("warn", "gpu-shape-fit",
                                 msg + ": --adapt-shape rewrites it (EXPERIMENTAL, AWS-unsupported)"))
        return
    findings.append(_finding(
        "block", "gpu-shape-fit", msg,
        "Use a p4d/p5 instance group (the recipe's supported shape), or opt in to the EXPERIMENTAL "
        "--adapt-shape rewrite to the group's allocatable."))


def _check_local_queue(findings: list, a: dict, env: dict) -> None:
    q = (env.get("HP_QUEUE") or "").strip()
    if not q:
        return
    rc, out, err = _run(["kubectl", "get", "localqueues.kueue.x-k8s.io", q, "-n", a["namespace"], "-o", "name"])
    if rc == 0 and out.strip():
        findings.append(_finding("info", "local-queue", f"Kueue LocalQueue {a['namespace']}/{q} exists"))
        return
    findings.append(_finding(
        "block", "local-queue",
        f"Kueue LocalQueue {a['namespace']}/{q} " + ("does not exist" if (rc == 0 or _not_found(err)) else f"could not be read: {err.strip()[:200]}")
        + " (a workload naming it would stay Pending forever)",
        f"Create the LocalQueue '{q}' in {a['namespace']} (cluster owner), or unset HP_QUEUE."))


def _check_s3_csi_role(findings: list, a: dict, env: dict) -> None:
    bucket = a.get("model_bucket") or ""
    report = preflight(a["cluster"], a["namespace"], a.get("project", ""), "s3", bucket, False,
                       env.get("EKS_CLUSTER_NAME", ""))
    for f in report["findings"]:
        if f["check"] != "s3-csi-role":
            continue
        if f["severity"] == "warn" and "not allowed" in f["message"]:
            f = dict(f, severity="block")
        findings.append(f)


def contract(verb: str, args: dict, env: dict | None = None) -> dict:
    """Run the verb's contract elements (R2.2). Returns {findings, resolved}."""
    env = dict(os.environ) if env is None else env
    a = dict(args)
    a["verb"] = verb
    a.setdefault("cluster", env.get("HP_CLUSTER_NAME", ""))
    a["namespace"] = a.get("namespace") or env.get("HP_NAMESPACE") or "default"
    a.setdefault("project", env.get("PROJECT_NAME", ""))
    a["kind"] = "tune" if verb == "tune" else "train"
    findings: list[dict] = []
    resolved: dict = {"namespace": a["namespace"], "service_account": TRAIN_SERVICE_ACCOUNT}
    for element in CONTRACT_ELEMENTS[verb]:
        if element == "training-operator":
            _check_training_operator(findings)
        elif element == "pod-identity":
            _check_pod_identity(findings, resolved, a, env)
        elif element == "scratch-volume":
            _check_scratch(findings, resolved, a)
        elif element == "free-gpus":
            _check_free_gpus(findings, resolved, a)
        elif element == "gpu-shape-fit":
            _check_shape_fit(findings, resolved, a)
        elif element == "local-queue":
            _check_local_queue(findings, a, env)
        elif element == "s3-csi-role":
            _check_s3_csi_role(findings, a, env)
    return {"findings": findings, "resolved": resolved,
            "blocked": any(f["severity"] == "block" for f in findings)}


def _cap_warn(findings: list[dict]) -> list[dict]:
    return [dict(f, severity="warn") if f["severity"] == "block" else f for f in findings]


def preflight_from_env(env: dict | None = None) -> dict:
    """The same read-only R6 preflight, with its inputs read from do/config-style
    env (Wave 8 R5.3: do/validate calls it through the hyperpod-eks descriptor's
    cluster_contract). The model source follows the driver's resolution: an
    explicit HP_MODEL_SOURCE, else s3 for an S3-staged model, else huggingface."""
    env = dict(os.environ) if env is None else env
    staged = (env.get("STAGED_MODEL_PATH") or "").strip()
    source = (env.get("HP_MODEL_SOURCE") or "").strip() or ("s3" if staged.startswith("s3://") else "huggingface")
    bucket = staged[len("s3://"):].split("/", 1)[0] if source == "s3" and staged.startswith("s3://") else ""
    report = preflight(env.get("HP_CLUSTER_NAME", ""), env.get("HP_NAMESPACE") or "default",
                       env.get("PROJECT_NAME", ""), source, bucket,
                       (env.get("HP_LORA_ENABLED") or "").lower() == "true", env.get("EKS_CLUSTER_NAME", ""))
    # Wave 11 R2.3: plain do/validate also reports the hp-eks training elements,
    # but never above `warn` (a serving-only project must not fail validate on a
    # training prerequisite; `do/train|tune --backend hp-eks` enforces them).
    findings: list[dict] = []
    a = {"verb": "train", "kind": "train", "cluster": env.get("HP_CLUSTER_NAME", ""),
         "namespace": env.get("HP_NAMESPACE") or "default", "project": env.get("PROJECT_NAME", "")}
    _check_training_operator(findings)
    _check_pod_identity(findings, {}, a, env)
    _check_local_queue(findings, a, env)
    report["findings"].extend(_cap_warn(findings))
    return report


# ── CLI ────────────────────────────────────────────────────────────────────────


def _cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hyperpod_ops.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rollout-capacity")
    a.add_argument("--cluster", required=True)
    a.add_argument("--namespace", required=True)
    a.add_argument("--project", required=True)
    a.add_argument("--instance-type", required=True)
    a.add_argument("--gpus", type=int, required=True)
    b = sub.add_parser("surge-up")
    b.add_argument("--cluster", required=True)
    b.add_argument("--group", required=True)
    c = sub.add_parser("surge-wait")
    c.add_argument("--cluster", required=True)
    c.add_argument("--group", required=True)
    c.add_argument("--count", type=int, required=True)
    c.add_argument("--timeout", type=int, default=2700)
    c.add_argument("--interval", type=int, default=30)
    d = sub.add_parser("rollout-wait")
    d.add_argument("--namespace", required=True)
    d.add_argument("--project", required=True)
    d.add_argument("--replicas", type=int, required=True)
    d.add_argument("--timeout", type=int, default=1800)
    d.add_argument("--interval", type=int, default=15)
    e = sub.add_parser("surge-restore")
    e.add_argument("--cluster", required=True)
    e.add_argument("--group", required=True)
    e.add_argument("--original-count", type=int, required=True)
    e.add_argument("--namespace", required=True)
    e.add_argument("--project", required=True)
    f = sub.add_parser("preflight")
    f.add_argument("--cluster", required=True)
    f.add_argument("--namespace", required=True)
    f.add_argument("--project", required=True)
    f.add_argument("--model-source", required=True)
    f.add_argument("--model-bucket", default="")
    f.add_argument("--lora-delivery", default="false")
    f.add_argument("--eks-cluster", default="")
    sub.add_parser("preflight-env")
    k = sub.add_parser("contract")
    k.add_argument("--verb", required=True, choices=sorted(CONTRACT_ELEMENTS))
    k.add_argument("--cluster", default=None)
    k.add_argument("--namespace", default=None)
    k.add_argument("--project", default=None)
    k.add_argument("--group", default="")
    k.add_argument("--instance-type", default="")
    k.add_argument("--gpus", type=int, default=0)
    k.add_argument("--dataset", default="")
    k.add_argument("--job", default="")
    k.add_argument("--storage-class", default="")
    k.add_argument("--model-bucket", default="")
    k.add_argument("--template-gpus", type=int, default=None)
    k.add_argument("--template-efa", type=int, default=0)
    k.add_argument("--adapt-shape", action="store_true")
    k.add_argument("--allow-surge", action="store_true")
    k.add_argument("--json", action="store_true", help="JSON only (default prints JSON plus a human summary on stderr)")
    args = p.parse_args(argv)

    if args.cmd == "contract":
        cargs = {key: val for key, val in {
            "cluster": args.cluster, "namespace": args.namespace, "project": args.project,
            "group": args.group, "instance_type": args.instance_type, "gpus": args.gpus,
            "dataset": args.dataset, "job": args.job, "storage_class": args.storage_class,
            "model_bucket": args.model_bucket, "template_gpus": args.template_gpus,
            "template_efa": args.template_efa, "adapt_shape": args.adapt_shape,
            "allow_surge": args.allow_surge}.items() if val is not None}
        out = contract(args.verb, cargs)
        if not args.json:
            for f in out["findings"]:
                mark = {"block": "❌", "warn": "⚠️ ", "info": "✅"}.get(f["severity"], "•")
                _note(f"   {mark} {f['check']}: {f['message']}")
                if f["severity"] != "info" and f["fix"]:
                    _note(f"      → {f['fix']}")
        _emit(out)
        return 4 if out["blocked"] else 0

    if args.cmd == "rollout-capacity":
        rc, out = rollout_capacity(args.cluster, args.namespace, args.project, args.instance_type, args.gpus)
    elif args.cmd == "surge-up":
        rc, out = surge_up(args.cluster, args.group)
    elif args.cmd == "surge-wait":
        rc, out = surge_wait(args.cluster, args.group, args.count, args.timeout, args.interval)
    elif args.cmd == "rollout-wait":
        rc, out = rollout_wait(args.namespace, args.project, args.replicas, args.timeout, args.interval)
    elif args.cmd == "surge-restore":
        rc, out = surge_restore(args.cluster, args.group, args.original_count, args.namespace, args.project)
    elif args.cmd == "preflight-env":
        rc, out = 0, preflight_from_env()
    else:
        rc, out = 0, preflight(args.cluster, args.namespace, args.project, args.model_source,
                               args.model_bucket, args.lora_delivery == "true", args.eks_cluster)
    _emit(out)
    return rc


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
