#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""adapter_delivery.py — where a LoRA adapter must live for a k8s pod to load it.

Wave 6.1 R1 (BL189). vLLM loads LoRA adapters from a LOCAL path; the serving
container has no AWS credentials. On hyperpod-eks / eks the only S3-backed path
the pod sees is the model mount, so the target descriptor declares
(targets.d/<t>/manifest.json `adapter_delivery`, read via target_manifest.py):

    {"strategy": "model-prefix", "mount_path": "/opt/ml/model", "subdir": ".mlcc-adapters"}

  s3://<model-bucket>/<model-prefix>/.mlcc-adapters/<project>/<name>/   (in S3)
  /opt/ml/model/.mlcc-adapters/<project>/<name>                         (in the pod)

This module maps an adapter URI to that in-pod path, or refuses with the fix —
it never passes an s3:// path through. Part A scope: an adapter must ALREADY be
under the deployment's model prefix; promoting adapters there (copy from the
tune bucket, `do/adapter promote`) is Wave 6.1 Part B.

CLI (stdlib only):
    adapter_delivery.py model-base --target T [--iec-json FILE]
        -> JSON {"source_type", "bucket", "prefix", "prefetch"} for the deployment
           (hyperpod-eks: the live InferenceEndpointConfig .spec.modelSourceConfig;
            eks: STAGED_MODEL_PATH when EKS_MODEL_MOUNT=s3)
    adapter_delivery.py local-path --target T --uri U --project P --name N
                        --source-type S --bucket B --prefix X --prefetch true|false
        -> prints the in-pod path (exit 0) or the refusal + fix (exit 3)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import target_manifest  # noqa: E402


class DeliveryRefused(Exception):
    """The adapter cannot be delivered to this deployment; .fix says how."""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.fix = fix


def model_base_from_iec(iec: dict | None) -> dict:
    """{source_type, bucket, prefix, prefetch} from a live InferenceEndpointConfig."""
    msc = ((iec or {}).get("spec") or {}).get("modelSourceConfig") or {}
    prefetch = msc.get("prefetchEnabled")
    return {
        "source_type": msc.get("modelSourceType") or "",
        "bucket": (msc.get("s3Storage") or {}).get("bucketName") or "",
        "prefix": str(msc.get("modelLocation") or "").strip("/"),
        "prefetch": "" if prefetch is None else ("true" if prefetch else "false"),
    }


def model_base_from_env(env: dict) -> dict:
    """eks: the S3 model mount (EKS_MODEL_MOUNT=s3) of STAGED_MODEL_PATH."""
    staged = (env.get("STAGED_MODEL_PATH") or "").strip()
    if env.get("EKS_MODEL_MOUNT") != "s3" or not staged.startswith("s3://"):
        return {"source_type": "", "bucket": "", "prefix": "", "prefetch": ""}
    bucket, _, prefix = staged[len("s3://"):].partition("/")
    return {"source_type": "s3", "bucket": bucket, "prefix": prefix.strip("/"), "prefetch": "false"}


def canonical_uri(base: dict, delivery: dict, project: str, name: str) -> str:
    sub = delivery.get("subdir", "")
    prefix = base.get("prefix", "")
    return f"s3://{base['bucket']}/{prefix + '/' if prefix else ''}{sub}/{project}/{name}/"


def local_path(target: str, uri: str, project: str, name: str, base: dict,
               delivery: dict | None = None) -> str:
    """Map an adapter URI to the in-pod path, or raise DeliveryRefused."""
    delivery = delivery if delivery is not None else target_manifest.adapter_delivery(target)
    if not uri.startswith("s3://"):
        return uri.rstrip("/")  # already a path inside the container
    if delivery.get("strategy") != "model-prefix":
        raise DeliveryRefused(
            f"{target} declares no adapter delivery strategy; an s3:// adapter cannot be loaded by path.",
            "Load adapters on this target with its own mechanism.")
    if base.get("source_type") != "s3" or not base.get("bucket"):
        if target in ("hyperpod-eks", "hyperpod"):
            raise DeliveryRefused(
                "this deployment's model is not served from S3 (Hugging Face / model-package source), so "
                "the pod has no S3 mount to read adapters from.",
                "Stage the model to S3 and redeploy: ./do/stage && ./do/deploy --target hyperpod-eks --force")
        raise DeliveryRefused(
            "this eks deployment has no S3 model mount, so the pod cannot read adapters from S3.",
            "Enable the S3 model mount (export EKS_MODEL_MOUNT=s3 with a staged model, ./do/stage) "
            "and redeploy: ./do/deploy --target eks --force")
    if base.get("prefetch") == "true":
        raise DeliveryRefused(
            "the running InferenceEndpointConfig still has prefetchEnabled: true — the worker reads a RAM "
            "copy of the model taken at start, so adapters added under the model prefix are not visible.",
            "Redeploy with LoRA on (MLCC sets prefetchEnabled: false for LoRA + S3): "
            "./do/deploy --target hyperpod-eks --force (add --surge-node on a single-node instance group)")
    sub = delivery.get("subdir", "")
    mount = delivery.get("mount_path", "").rstrip("/")
    root = f"s3://{base['bucket']}/" + (f"{base['prefix']}/" if base.get("prefix") else "") + f"{sub}/"
    norm = uri.rstrip("/") + "/"
    if norm.startswith(root) and len(norm) > len(root):
        rest = norm[len(root):].rstrip("/")
        return f"{mount}/{sub}/{rest}"
    # TODO(W6.1 Part B): promote the adapter into canonical_uri() (s3_layout.py)
    # instead of refusing. Part A never passes an s3:// path to vLLM.
    dest = canonical_uri(base, delivery, project, name)
    raise DeliveryRefused(
        f"{uri} is outside this deployment's model prefix; the pod can only load adapters under {root}",
        f"Copy it there and add it by that URI: aws s3 sync {uri.rstrip('/')}/ {dest} && "
        f"./do/adapter add {name} --from-weights {dest}  (automatic promotion lands in Wave 6.1 Part B)")


def _cli(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="adapter_delivery.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    mb = sub.add_parser("model-base")
    mb.add_argument("--target", required=True)
    mb.add_argument("--iec-json", default="")
    lp = sub.add_parser("local-path")
    for flag in ("--target", "--uri", "--project", "--name"):
        lp.add_argument(flag, required=True)
    for flag in ("--source-type", "--bucket", "--prefix", "--prefetch"):
        lp.add_argument(flag, default="")
    args = p.parse_args(argv)

    if args.cmd == "model-base":
        if target_manifest.resolve_target(args.target) == "hyperpod-eks":
            iec = None
            if args.iec_json:
                try:
                    with open(args.iec_json, encoding="utf-8") as f:
                        iec = json.load(f)
                except (OSError, ValueError):
                    iec = None
            print(json.dumps(model_base_from_iec(iec)))
        else:
            print(json.dumps(model_base_from_env(dict(os.environ))))
        return 0

    base = {"source_type": args.source_type, "bucket": args.bucket, "prefix": args.prefix.strip("/"),
            "prefetch": args.prefetch}
    try:
        print(local_path(args.target, args.uri, args.project, args.name, base))
    except DeliveryRefused as exc:
        print(f"❌ Cannot deliver adapter '{args.name}' to this {args.target} deployment: {exc}")
        if exc.fix:
            print(f"   Fix: {exc.fix}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
