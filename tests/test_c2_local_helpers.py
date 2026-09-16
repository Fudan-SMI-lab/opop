"""Fresh CPU subprocess imports, not GPU correctness or performance evidence."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu import worker_client
from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_local_adapter import GpuAdapter
from scripts.experiments.c2_local_inputs import Shared
from tests.test_c2_local_execution import shared as existing_shared

shared_fixture = existing_shared


@pytest.mark.parametrize("full", [False, True], ids=["tuning", "final"])
def test_sibling_imports_are_current_candidate_only_in_fresh_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_fixture: Shared, full: bool,
) -> None:
    # Given: two candidates with identically named, different sibling helpers.
    cfg = AppConfig()
    cfg.wsl.kernelbench_src = str(tmp_path / "kernelbench")
    cfg.wsl.extra_pythonpath = str(Path(__file__).parents[1] / "src")
    original = cfg.wsl.model_dump()
    store = RunStore.create(tmp_path, "run", {})
    adapter = GpuAdapter(shared_fixture, cfg, store)
    adapter.full = full
    candidates = []
    for name, offset in (("a", 100), ("b", 200)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sibling_helper.py").write_text(
            f"def transform(value):\n    return value + {offset}\n", encoding="utf-8")
        candidate = directory / "child.py"
        candidate.write_text(
            "from sibling_helper import transform\nPARAMS = {'x': 0}\nRESULT = transform(PARAMS['x'])\n",
            encoding="utf-8")
        candidates.append(candidate)
    observed: list[tuple[str, list[str]]] = []

    def cpu_worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] == "static_check":
            return {"ok": True}
        _, env = self._build_command(self.jobs_dir / "in.json", self.jobs_dir / "out.json")
        # Exercise the native worker builder; adapt its POSIX separator for this Windows CPU host.
        paths = re.split(r":(?=[A-Za-z]:[\\/])", env["PYTHONPATH"]) if os.name == "nt" else env["PYTHONPATH"].split(":")
        completed = subprocess.run(
            [sys.executable, "-B", "-c",
             "from pathlib import Path; import sys; "
             "exec(compile(Path(sys.argv[1]).read_text(encoding='utf-8'), sys.argv[1], 'exec')); print(RESULT)",
             job["kernel_src_path"]],
            cwd=self.jobs_dir, env={**env, "PYTHONPATH": os.pathsep.join(paths)},
            capture_output=True, text=True, timeout=30, check=False,
        )
        observed.append((completed.stdout.strip(), paths))
        return {"ok": completed.returncode == 0, "log_tail": completed.stderr,
                "failure_kind": "runtime_error" if completed.returncode else None,
                "latency_ms": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "n": 1}}

    monkeypatch.setattr(worker_client, "_wsl_hop_needed", lambda: False)
    monkeypatch.setattr(worker_client.WslGpuWorker, "run_job", cpu_worker)
    # When: the same bridge evaluates successive candidates through fresh CPU workers.
    records = [adapter.measure(candidate, ParamSet(values={"x": 7})) for candidate in candidates]
    # Then: actual helper code uses tuned params, with no earlier sibling directory leaking.
    assert all(record.status == "complete" for record in records), [r.failure_detail for r in records]
    assert [value for value, _ in observed] == ["107", "207"]
    for candidate, (_, paths) in zip(candidates, observed, strict=True):
        assert set(paths) == {cfg.wsl.kernelbench_src, cfg.wsl.extra_pythonpath, str(candidate.parent)}
    assert cfg.wsl.model_dump() == original
    assert adapter.correctness.worker.cfg is cfg.wsl


def test_worker_config_is_restored_when_evaluation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_fixture: Shared,
) -> None:
    # Given: a worker failure while a candidate-specific import path is active.
    cfg = AppConfig()
    store = RunStore.create(tmp_path, "run", {})
    adapter = GpuAdapter(shared_fixture, cfg, store)
    candidate = tmp_path / "child.py"
    candidate.write_text("PARAMS = {'x': 0}\n", encoding="utf-8")
    worker_configs = []

    def failing_worker(self, job, timeout_s, tag, **kwargs):
        worker_configs.append(self.cfg)
        raise RuntimeError("CPU fixture worker failure")

    monkeypatch.setattr(worker_client.WslGpuWorker, "run_job", failing_worker)
    # When: the bridge records the failed attempt.
    result = adapter.measure(candidate, ParamSet(values={"x": 7}))
    # Then: even failure restores the original config object, without mutating it.
    assert result.status == "fail"
    assert worker_configs and worker_configs[0] is not cfg.wsl
    assert adapter.correctness.worker.cfg is cfg.wsl
