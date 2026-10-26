import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from kernel_optimizer.runtime.worker import started_record


def test_started_proof_preserves_binding_and_references_full_parameters(tmp_path):
    parameters = {"nested": {"data": "payload" * 200000}, "bundles": [{"id": "one"}]}
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"parameters": parameters}))
    context = {"rank": 1, "world_size": 2, "binding": {"pid": 123, "status": "verified"},
               "parameters": parameters, "revision": {"revision": "same-snapshot"}}
    record = started_record(context, request)
    assert len(json.dumps(record)) < 1024
    ref = record["parameters_ref"]
    raw = Path(ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["request_sha256"]
    assert json.loads(raw)["parameters"] == context["parameters"] == parameters
    assert record["binding"] == context["binding"] and record["revision"] == context["revision"]
    assert record["rank"] == 1 and record["world_size"] == 2


@pytest.mark.skipif(os.name == "nt", reason="Archive live-process inspection uses Linux /proc")
def test_worker_archive_roundtrip_skips_live_jobs_and_keeps_results(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/v6_archive_worker_records.py"
    done, live = [tmp_path / "jobs" / name / "run" for name in ["done", "live"]]
    for run in [done, live]:
        run.mkdir(parents=True)
        (run / "request.json").write_bytes(b'{"payload":"' + b"same" * 10000 + b'"}\n')
        (run / "receipt.json").write_text(json.dumps({"worker_pids_reaped": True,
            "ranks": [{"pid": os.getpid()}] if run == live else []}))
    raw = (done / "request.json").read_bytes()
    result = (done / "receipt.json").read_bytes()
    manifest = tmp_path / "journal.jsonl"
    subprocess.run([sys.executable, str(script), str(tmp_path), "--manifest", str(manifest), "--apply"], check=True)
    assert not (done / "request.json").exists()
    assert (live / "request.json").read_bytes() == raw
    assert (done / "receipt.json").read_bytes() == result
    subprocess.run([sys.executable, str(script), "--restore", str(manifest)], check=True)
    assert (done / "request.json").read_bytes() == raw
    (done / "request.json").write_text("new user content")
    completed = subprocess.run([sys.executable, str(script), "--restore", str(manifest)], capture_output=True)
    assert completed.returncode != 0
    assert (done / "request.json").read_text() == "new user content"
