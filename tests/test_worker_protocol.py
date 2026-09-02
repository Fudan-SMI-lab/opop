"""Worker protocol tests (no GPU): path translation, job shapes, lock semantics."""

import threading
import time

from kernel_optimizer.gpu.jobs import (
    failure_result,
    make_baseline_job,
    make_env_probe_job,
    make_eval_job,
    make_static_check_job,
)
from kernel_optimizer.gpu.worker_client import GpuRwLock, to_wsl_path


def test_to_wsl_path():
    assert to_wsl_path(r"D:\Pyhon_projects\opop\v2\x.py") == "/mnt/d/Pyhon_projects/opop/v2/x.py"
    assert to_wsl_path("C:/Users/me/f.json") == "/mnt/c/Users/me/f.json"


def test_job_shapes_are_json_plain():
    import json

    jobs = [
        make_env_probe_job(),
        make_static_check_job("/tmp/k.py", "triton", "fp32"),
        make_baseline_job("/tmp/ref.py", num_trials=100, timing_method="cuda_event",
                          precision="fp32", use_torch_compile=False),
        make_eval_job("/tmp/ref.py", "/tmp/k.py", measure_performance=True,
                      num_correct_trials=5, num_perf_trials=100,
                      timing_method="cuda_event", backend="triton", precision="fp32",
                      seed=42, build_dir=None, collect_triton_metadata=True),
    ]
    for job in jobs:
        json.dumps(job)  # must be plain JSON

    assert jobs[3]["job_type"] == "eval_perf"
    eval_c = make_eval_job("/r.py", "/k.py", measure_performance=False,
                           num_correct_trials=3, num_perf_trials=0,
                           timing_method="cuda_event", backend="triton",
                           precision="fp32", seed=42, build_dir=None,
                           collect_triton_metadata=False)
    assert eval_c["job_type"] == "eval_correctness"


def test_failure_result_truncates():
    r = failure_result("oom", "x" * 10000)
    assert len(r["log_tail"]) == 4000
    assert r["failure_kind"] == "oom"
    assert not r["ok"]


def test_rwlock_shared_concurrency(tmp_path):
    lock = GpuRwLock(tmp_path / "gpu.lock", max_shared=2)
    active = []
    peak = []

    def shared_worker():
        lock.acquire("shared")
        active.append(1)
        peak.append(len(active))
        time.sleep(0.05)
        active.pop()
        lock.release("shared")

    threads = [threading.Thread(target=shared_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(peak) == 2  # bounded by max_shared, but actually concurrent


def test_rwlock_exclusive_blocks_shared(tmp_path):
    lock = GpuRwLock(tmp_path / "gpu.lock", max_shared=2)
    order = []

    lock.acquire("exclusive")

    def shared_worker():
        lock.acquire("shared")
        order.append("shared")
        lock.release("shared")

    t = threading.Thread(target=shared_worker)
    t.start()
    time.sleep(0.05)
    assert order == []  # shared must wait for the writer
    order.append("exclusive-done")
    lock.release("exclusive")
    t.join(timeout=5)
    assert order == ["exclusive-done", "shared"]


def test_rwlock_exclusive_waits_for_shared(tmp_path):
    lock = GpuRwLock(tmp_path / "gpu.lock", max_shared=2)
    order = []

    lock.acquire("shared")

    def exclusive_worker():
        lock.acquire("exclusive")
        order.append("exclusive")
        lock.release("exclusive")

    t = threading.Thread(target=exclusive_worker)
    t.start()
    time.sleep(0.05)
    assert order == []
    order.append("shared-done")
    lock.release("shared")
    t.join(timeout=5)
    assert order == ["shared-done", "exclusive"]


def test_stale_lockfile_takeover(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    lock_path.write_text('{"pid": 99999, "ts": 0}')  # ancient timestamp
    lock = GpuRwLock(lock_path, max_shared=1, stale_after_s=1.0)
    lock.acquire("exclusive")  # must not deadlock
    lock.release("exclusive")
