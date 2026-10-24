# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bucket consolidation R3 (Option D) — s3_layout.py, the single layout module,
and the producer <-> consumer round trip (R3.8).

Bucket names are deliberately unconventional fakes (zz-*): the layout never
knows a bucket name, callers pass what the profile says."""
import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

_PY = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "templates", "do", "lib", "python"))
_DO = os.path.normpath(os.path.join(_PY, "..", ".."))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import adapter_delivery as ad  # noqa: E402
import s3_layout as L  # noqa: E402
import target_manifest  # noqa: E402

CORE = "zz-core-x"
ART = "zz-art-y"
CATALOG = {"huggingface-vlm-gemma-4-e4b-it": {"huggingFaceId": "google/gemma-4-E4B-it"},
           "broken-entry": {"displayName": "no hf id"}}
HF = "google/gemma-4-E4B-it"


# ── fake S3 (list/get/put/copy/delete/upload/download/head) ──────────────────
class FakeS3:
    def __init__(self, objects=None):
        self.o = dict(objects or {})  # (bucket, key) -> bytes
        self.calls = []

    def put(self, bucket, key, body):
        self.o[(bucket, key)] = body if isinstance(body, bytes) else body.encode()

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):  # noqa: N803
        keys = sorted(k for (b, k) in self.o if b == Bucket and k.startswith(Prefix))
        import hashlib
        return {"Contents": [{"Key": k, "Size": len(self.o[(Bucket, k)]),
                              "ETag": '"%s"' % hashlib.md5(self.o[(Bucket, k)]).hexdigest()} for k in keys],  # noqa: S324
                "IsTruncated": False}

    def get_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.o:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.o[(Bucket, Key)])}

    def head_object(self, Bucket, Key):  # noqa: N803
        body = self.o[(Bucket, Key)]
        return {"ContentLength": len(body), "ETag": '"e%d"' % len(body)}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.calls.append(("put", Bucket, Key))
        self.put(Bucket, Key, Body)

    def copy(self, src, bucket, key):
        self.calls.append(("copy", src["Bucket"], src["Key"], bucket, key))
        self.o[(bucket, key)] = self.o[(src["Bucket"], src["Key"])]

    def delete_objects(self, Bucket, Delete):  # noqa: N803
        for d in Delete["Objects"]:
            self.calls.append(("delete", Bucket, d["Key"]))
            self.o.pop((Bucket, d["Key"]), None)

    def upload_file(self, path, bucket, key):
        self.calls.append(("upload", bucket, key))
        with open(path, "rb") as f:
            self.o[(bucket, key)] = f.read()

    def download_file(self, bucket, key, path):
        with open(path, "wb") as f:
            f.write(self.o[(bucket, key)])

    def keys(self, bucket):
        return sorted(k for (b, k) in self.o if b == bucket)


RAW = f"s3://{ART}/fresh-ai-ml/tune/sft/job-1/output/model/checkpoints/hf/"
ADAPTER_CFG = json.dumps({"r": 32, "rank_pattern": {"q_proj": 64}, "target_modules": ["q_proj", "v_proj"]})


def _raw_s3():
    s3 = FakeS3()
    root = RAW[len(f"s3://{ART}/"):]
    s3.put(ART, root + "adapter_config.json", ADAPTER_CFG)
    s3.put(ART, root + "adapter_model.safetensors", b"weights-v1")
    return s3


# ── composition ───────────────────────────────────────────────────────────────
def test_paths_compose_the_option_d_layout():
    assert L.base_prefix(HF) == "models/google/gemma-4-E4B-it/"
    assert L.adapter_uri(CORE, HF, "p", "n") == f"s3://{CORE}/models/google/gemma-4-E4B-it/.mlcc-adapters/p/n/"
    assert L.finetune_uri(CORE, "p", "n") == f"s3://{CORE}/models/mlcc-p/n/"
    assert L.artifacts_uri(ART, "p", "tune") == f"s3://{ART}/p/tune/"
    assert L.evaluator_key("math-reward") == "evaluators/math-reward/lambda_function.py"
    assert L.SERVE_SYNC_EXCLUDE == ".mlcc-adapters/*"


@pytest.mark.parametrize("bad", ["", "../x", "a/b/c", "a b", "/abs"])
def test_invalid_segments_are_refused(bad):
    with pytest.raises(L.LayoutError):
        L.adapter_uri(CORE, HF, bad or "", "n") if bad != "/abs" else L.base_prefix("a/b/c/d")


def test_no_bucket_is_refused():
    with pytest.raises(L.LayoutError):
        L.adapter_uri("", HF, "p", "n")


def test_resolve_base_id_maps_catalog_ids_and_never_guesses():
    assert L.resolve_base_id("huggingface-vlm-gemma-4-e4b-it", CATALOG) == HF
    assert L.resolve_base_id(HF, CATALOG) == HF
    assert L.resolve_base_id(f"s3://{CORE}/models/{HF}/", CATALOG) == HF
    for bad in ("huggingface-llm-unknown", "broken-entry", "", "s3://b/not-models/x/"):
        with pytest.raises(L.LayoutError):
            L.resolve_base_id(bad, CATALOG)


def test_real_tune_catalog_maps_the_fresh_ai_ml_base():
    assert L.resolve_base_id("huggingface-vlm-gemma-4-e4b-it", L.load_catalog()) == HF


def test_parse_round_trips_composition():
    uri = L.adapter_uri(CORE, HF, "fresh-ai-ml", "sft-w6")
    p = L.parse_adapter_uri(uri)
    assert p == {"bucket": CORE, "base_prefix": "models/google/gemma-4-E4B-it", "base_model": HF,
                 "project": "fresh-ai-ml", "name": "sft-w6",
                 "prefix": "models/google/gemma-4-E4B-it/.mlcc-adapters/fresh-ai-ml/sft-w6/"}
    assert L.parse_adapter_uri(RAW) is None
    assert L.parse_finetune_uri(L.finetune_uri(CORE, "p", "n"))["project"] == "p"


# ── promotion ─────────────────────────────────────────────────────────────────
def test_promote_copies_payload_writes_manifest_and_is_idempotent():
    s3 = _raw_s3()
    dest = L.adapter_uri(CORE, HF, "fresh-ai-ml", "tuned-sft")
    r1 = L.promote(s3, source=RAW, dest=dest, kind="lora_adapter", project="fresh-ai-ml", name="tuned-sft",
                   base_model=HF, source_job="job-1", technique="sft", created_by="do/tune")
    assert r1["action"] == "copied"
    m = json.loads(s3.o[(CORE, L.split_uri(dest)[1] + L.ADAPTER_MANIFEST)])
    assert m["source_job"] == "job-1" and m["technique"] == "sft" and m["base_model"] == HF
    assert m["rank"] == 64 and m["target_modules"] == ["q_proj", "v_proj"] and m["created_at"]
    assert m["content_hash"].startswith("sha256:")
    n_calls = len(s3.calls)
    r2 = L.promote(s3, source=RAW, dest=dest, kind="lora_adapter", project="fresh-ai-ml", name="tuned-sft",
                   base_model=HF)
    assert r2["action"] == "unchanged" and len(s3.calls) == n_calls  # nothing copied twice
    # New content → copied again, previous hash recorded, stale file removed.
    root = RAW[len(f"s3://{ART}/"):]
    s3.put(ART, root + "adapter_model.safetensors", b"weights-v2")
    s3.o.pop((ART, root + "adapter_config.json"))
    s3.put(ART, root + "adapter_config.json", ADAPTER_CFG)
    s3.put(CORE, L.split_uri(dest)[1] + "leftover.bin", b"old")
    r3 = L.promote(s3, source=RAW, dest=dest, kind="lora_adapter", project="fresh-ai-ml", name="tuned-sft",
                   base_model=HF)
    assert r3["action"] == "copied" and r3["manifest"]["previous_content_hash"] == m["content_hash"]
    assert (CORE, L.split_uri(dest)[1] + "leftover.bin") not in s3.o


def test_promote_full_model_goes_to_its_own_prefix_not_under_the_base():
    s3 = FakeS3({(ART, "p/tune/sft/j/hf_merged/model.safetensors"): b"big"})
    dest = L.finetune_uri(CORE, "p", "tuned-sft")
    L.promote(s3, source=f"s3://{ART}/p/tune/sft/j/hf_merged/", dest=dest, kind="full_model", project="p",
              name="tuned-sft", base_model=HF)
    assert (CORE, "models/mlcc-p/tuned-sft/model.safetensors") in s3.o
    assert (CORE, "models/mlcc-p/tuned-sft/" + L.MODEL_MANIFEST) in s3.o
    assert not any(k.startswith(L.base_prefix(HF)) for k in s3.keys(CORE))


def test_promote_extracts_a_tarball_source(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, body in (("adapter_config.json", ADAPTER_CFG.encode()), ("adapter_model.safetensors", b"w")):
            ti = tarfile.TarInfo(name)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
    s3 = FakeS3({("zz-legacy", "a/adapter.tar.gz"): buf.getvalue()})
    dest = L.adapter_uri(CORE, HF, "p", "n")
    L.promote(s3, source="s3://zz-legacy/a/adapter.tar.gz", dest=dest, kind="lora_adapter", project="p",
              name="n", base_model=HF)
    key = L.split_uri(dest)[1]
    assert (CORE, key + "adapter_model.safetensors") in s3.o
    assert json.loads(s3.o[(CORE, key + L.ADAPTER_MANIFEST)])["rank"] == 64


def test_promote_refuses_non_canonical_destinations_and_empty_sources():
    s3 = _raw_s3()
    with pytest.raises(L.LayoutError):
        L.promote(s3, source=RAW, dest=f"s3://{CORE}/models/{HF}/", kind="lora_adapter", project="p",
                  name="n", base_model=HF)
    with pytest.raises(L.LayoutError):
        L.promote(s3, source=f"s3://{ART}/nothing/", dest=L.adapter_uri(CORE, HF, "p", "n"),
                  kind="lora_adapter", project="p", name="n", base_model=HF)


def test_publish_local_writes_files_and_manifest(tmp_path):
    (tmp_path / "adapter_config.json").write_text(ADAPTER_CFG)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w")
    s3 = FakeS3()
    dest = L.adapter_uri(CORE, HF, "p", "hub-a")
    r = L.publish_local(s3, local_dir=str(tmp_path), dest=dest, kind="lora_adapter", project="p", name="hub-a",
                        base_model=HF, source="hf://org/repo")
    assert r["action"] == "copied" and r["manifest"]["source"] == "hf://org/repo"
    again = L.publish_local(s3, local_dir=str(tmp_path), dest=dest, kind="lora_adapter", project="p",
                            name="hub-a", base_model=HF, source="hf://org/repo")
    assert again["action"] == "unchanged"


# ── clean scope (R3.6 / Wave 6.1 R1.7) ────────────────────────────────────────
def test_clean_plan_only_touches_the_own_projects_artifacts():
    s3 = FakeS3()
    for k in [
        "models/google/gemma-4-E4B-it/config.json",                             # base model
        "models/google/gemma-4-E4B-it/model.safetensors",
        "models/google/gemma-4-E4B-it/.mlcc-adapters/proj/a/adapter_model.safetensors",
        "models/Qwen/Qwen3-8B/.mlcc-adapters/proj/b/adapter_config.json",
        "models/google/gemma-4-E4B-it/.mlcc-adapters/proj-two/a/adapter_config.json",  # other project
        "models/google/gemma-4-E4B-it/.mlcc-adapters/other/a/adapter_config.json",
        "models/mlcc-proj/full/model.safetensors",
        "models/mlcc-proj-two/full/model.safetensors",
        "datasets/proj/x.jsonl",
    ]:
        s3.put(CORE, k, b"x")
    plan = L.clean_plan(s3, CORE, "proj")
    assert sorted(plan["keys"]) == [
        "models/Qwen/Qwen3-8B/.mlcc-adapters/proj/b/adapter_config.json",
        "models/google/gemma-4-E4B-it/.mlcc-adapters/proj/a/adapter_model.safetensors",
        "models/mlcc-proj/full/model.safetensors",
    ]
    assert "models/mlcc-proj/" in plan["prefixes"]
    with pytest.raises(L.LayoutError):
        L.delete_prefix(s3, f"s3://{CORE}/models/google/gemma-4-E4B-it/")


# ── R3.8 producer <-> consumer round trip ─────────────────────────────────────
def _shell_adapter_uri(project, name, model):
    """The shell producers' path (do/lib/s3-layout.sh → s3_layout.py adapter-uri)."""
    script = (f'SCRIPT_DIR="{_DO}"; source "{_DO}/lib/s3-layout.sh"; '
              f'_s3_layout_field "$(_s3_layout adapter-uri --bucket {CORE} --model {model} '
              f'--project {project} --name {name})" uri')
    return subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, check=True).stdout.strip()


def test_round_trip_every_producer_and_consumer_resolves_the_same_key():
    project, name = "fresh-ai-ml", "tuned-sft"
    expected = L.adapter_uri(CORE, HF, project, name)
    # Producers: do/tune + do/train promotion and do/adapter --from-hub / --from-weights /
    # registry / promote all compose through the shell wrapper → s3_layout adapter-uri,
    # with the tune base given as its catalog id or the project's HF id.
    assert _shell_adapter_uri(project, name, "huggingface-vlm-gemma-4-e4b-it") == expected
    assert _shell_adapter_uri(project, name, HF) == expected
    # Producer (k8s add of a non-canonical source) promotes to the same key.
    base = {"source_type": "s3", "bucket": CORE, "prefix": "models/google/gemma-4-E4B-it", "prefetch": "false"}
    plan = ad.plan_delivery("hyperpod-eks", RAW, project, name, base, core_bucket=CORE, adapter_base=HF)
    assert plan["promote_to"] == expected and plan["copy_to"] == ""
    # Producer (SMAI Processing-Job staging) accepts only that key.
    import stage_adapter

    class A:
        output_s3_uri = expected
    assert stage_adapter._canonical_output_uri(A) == expected
    # Consumers: the k8s mount path, the SMAI IC artifact URL (the canonical prefix
    # itself), and do/clean's ownership filter all agree on the same key.
    delivery = target_manifest.adapter_delivery("hyperpod-eks")
    assert plan["path"] == f"{delivery['mount_path']}/{L.ADAPTER_SUBDIR}/{project}/{name}"
    assert ad.plan_delivery("hyperpod-eks", expected, project, name, base)["path"] == plan["path"]
    assert L.parse_adapter_uri(expected)["name"] == name
    assert L.is_project_owned_key(L.split_uri(expected)[1] + "adapter_model.safetensors", project)
    assert not L.is_project_owned_key(L.split_uri(expected)[1] + "x", "fresh-ai")


def test_end_to_end_promote_then_deliver_loads_by_path_without_a_copy():
    s3 = _raw_s3()
    base = {"source_type": "s3", "bucket": CORE, "prefix": "models/google/gemma-4-E4B-it", "prefetch": "false"}
    res = ad.deliver(s3, "hyperpod-eks", RAW, "fresh-ai-ml", "sft-w6", base, core_bucket=CORE,
                     adapter_base="huggingface-vlm-gemma-4-e4b-it")
    assert res["actions"] == ["promote:copied"]
    assert res["path"].endswith("/.mlcc-adapters/fresh-ai-ml/sft-w6")
    again = ad.deliver(s3, "hyperpod-eks", RAW, "fresh-ai-ml", "sft-w6", base, core_bucket=CORE,
                       adapter_base="huggingface-vlm-gemma-4-e4b-it")
    assert again["actions"] == ["promote:unchanged"]
    assert ad.remove_copy(s3, base, "fresh-ai-ml", "sft-w6")["deleted"] == 0  # canonical is kept


def test_remove_deletes_only_a_serving_copy_it_made():
    BASE = {"source_type": "s3", "bucket": "core", "prefix": "models/google/gemma-4-E4B-it", "prefetch": "false"}
    restaged = {**BASE, "prefix": "models/mlcc-fresh-ai-ml/gemma-restaged"}
    s3 = FakeS3({("core", "models/google/gemma-4-E4B-it/.mlcc-adapters/fresh-ai-ml/tuned-sft/adapter_config.json"): b"{}",
                 ("core", "models/google/gemma-4-E4B-it/.mlcc-adapters/fresh-ai-ml/tuned-sft/w.safetensors"): b"w"})
    res = ad.deliver(s3, "hyperpod-eks", "s3://core/models/google/gemma-4-E4B-it/.mlcc-adapters/fresh-ai-ml/tuned-sft/",
                     "fresh-ai-ml", "sft-w6", restaged, deployment_model="google/gemma-4-E4B-it")
    assert res["actions"] == ["copy:copied"]
    copy_key = "models/mlcc-fresh-ai-ml/gemma-restaged/.mlcc-adapters/fresh-ai-ml/sft-w6/"
    assert ("core", copy_key + "w.safetensors") in s3.o
    out = ad.remove_copy(s3, restaged, "fresh-ai-ml", "sft-w6")
    assert out["deleted"] >= 2 and not any(k.startswith(copy_key) for k in s3.keys("core"))
    assert ("core", "models/google/gemma-4-E4B-it/.mlcc-adapters/fresh-ai-ml/tuned-sft/w.safetensors") in s3.o
    # A canonical adapter (no serving_copy_of) is never removed by remove.
    assert ad.remove_copy(s3, BASE, "fresh-ai-ml", "tuned-sft")["deleted"] == 0
