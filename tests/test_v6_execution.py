import json
import os
from pathlib import Path

import pytest

from kernel_optimizer.runtime.execution import DeviceLease, ExecutionRequest, LocalExecutor, resolve_devices


def test_device_placement_uses_current_host_indices_and_preserves_order():
    inventory = [{"index": "0", "uuid": "GPU-new-a"}, {"index": "1", "uuid": "GPU-new-b"}]
    assert resolve_devices(["1", "cuda:0"], inventory=inventory) == ["GPU-new-b", "GPU-new-a"]
    assert resolve_devices(["auto"], inventory=inventory, count=1) == ["GPU-new-a"]
    assert resolve_devices(["auto"], inventory=inventory, count=2) == ["GPU-new-a", "GPU-new-b"]
    assert resolve_devices(["auto"], inventory=[], count=0) == []
    with pytest.raises(ValueError, match="distinct physical GPUs"):
        resolve_devices(["0", "GPU-new-a"], inventory=inventory)
    with pytest.raises(ValueError, match="task needs 3 GPUs"):
        resolve_devices(["auto"], inventory=inventory, count=3)
    with pytest.raises(ValueError, match="--reuse-evaluator"):
        resolve_devices(["GPU-old-host"], inventory=inventory)

def request(tmp_path, source, timeout=20):
    entry = tmp_path / "entry.py"
    entry.write_text(source)
    root = Path(__file__).resolve().parents[1] / "src"
    return ExecutionRequest(entry=entry, cwd=tmp_path, output=tmp_path / "run",
                            lease_dir=tmp_path / "leases", worker_timeout_s=timeout,
                            env={"PYTHONPATH": str(root)})


def test_cpu_worker_executes_fresh_process_and_records_failure(tmp_path):
    job = request(tmp_path, "import os\ndef run(ctx):\n return {'pid':os.getpid(),'rank':ctx['rank']}\n")
    receipt = LocalExecutor().run(job)
    assert receipt["status"] == "passed"
    assert receipt["ranks"][0]["result"]["pid"] != os.getpid()
    assert receipt["worker_pids_reaped"]
    job.entry.write_text("def run(ctx):\n raise ValueError('control failure')\n")
    job.output = tmp_path / "failed"
    receipt = LocalExecutor().run(job)
    assert receipt["status"] == "failed" and receipt["ranks"][0]["error_type"] == "ValueError"


def test_worker_timeout_reaps_process_and_records_receipt(tmp_path):
    job = request(tmp_path, "import time\ndef run(ctx):\n time.sleep(60)\n", timeout=0.5)
    receipt = LocalExecutor().run(job)
    assert receipt["failure"] == "worker_timeout" and receipt["worker_pids_reaped"]
    assert json.loads((job.output / "receipt.json").read_text())["status"] == "failed"


@pytest.mark.skipif(os.name == "nt", reason="flock re-entry test is POSIX specific")
def test_device_lease_is_exclusive_and_released(tmp_path):
    with DeviceLease(tmp_path, "uuid"):
        with pytest.raises(RuntimeError, match="busy"):
            with DeviceLease(tmp_path, "uuid"):
                pass
    with DeviceLease(tmp_path, "uuid"):
        pass
