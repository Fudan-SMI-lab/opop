"""Real P0 validation. Keep --output private; export only reviewed receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.config import load_config
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, gpu_inventory, write_json
from kernel_optimizer.wiring import Runtime, _sandbox_extra_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    assert cfg.usage.mode == "observe_only" and cfg.opencode.request_timeout_s is None
    devices = [row["uuid"] for row in gpu_inventory()]
    assert len(devices) >= 2, "P0 dual-rank validation needs two available GPUs"
    sb = SandboxFactory(args.output / "sandboxes", _sandbox_extra_config(cfg)).create("gpu-helper")
    job = ExecutionRequest(entry=sb.root / "worker_entry.py", cwd=sb.root,
                           output=sb.root / "gpu-run", lease_dir=args.output.parent / "device-leases",
                           devices=[devices[1]], worker_timeout_s=90)
    sb.write_input("job.json", job.model_dump_json(indent=2))
    with Runtime(cfg, log_dir=args.output) as runtime:
        session = runtime.client.create_session(sb.root, "P0 explicit GPU worker")
        result = runtime.client.prompt(session, (
            "Implement worker_entry.py with run(context). Using torch on context['device'], "
            "create arange(16, dtype=float32), compute the sum of its squares on GPU and "
            "return {'checksum': <the computed float>, 'elements': 16}. This validates a "
            "GPU execution helper, not kernel optimization. Execute the supplied job.json via "
            f"{sys.executable} -B -m kernel_optimizer.runtime.cli job.json. "
            "Use the helper for GPU execution. Read its receipt and report whether execution "
            "and physical device binding passed. If a retry is needed use a fresh output "
            "directory in job.json. Do not change requested devices or the task calculation."
        ), model=cfg.agents.default_model, directory=sb.root)
        assert not result.error, result.error
        usage = runtime.client.collect_usage(session, sb.root)
        repeated = runtime.client.collect_usage(session, sb.root)
        assert usage == repeated and usage["messages"] > 0
    agent_job = ExecutionRequest.model_validate_json((sb.root / "job.json").read_bytes())
    agent_receipt = json.loads((agent_job.output / "receipt.json").read_text())
    assert agent_receipt["status"] == "passed"
    receipts = {}
    for index, device in enumerate(devices[:2]):
        receipt = LocalExecutor().run(job.model_copy(update={"devices": [device], "output": args.output / f"device-{index}"}))
        assert receipt["status"] == "passed"
        assert receipt["ranks"][0]["result"] == {"checksum": 1240.0, "elements": 16}
        assert receipt["ranks"][0]["binding"]["torch_uuid"] == device
        receipts[f"single_{index}"] = receipt
    collective = args.output / "collective.py"
    collective.write_text("""import torch
def run(context):
    if context['parameters'].get('fail_rank') == context['rank']:
        raise RuntimeError('injected rank failure')
    value = torch.tensor([context['rank'] + 1.0], device=context['device'])
    torch.distributed.all_reduce(value)
    assert value.item() == context['world_size'] * (context['world_size'] + 1) / 2
    return {'sum': value.item()}
""")
    distributed = job.model_copy(update={"entry": collective, "devices": devices[:2],
                                         "distributed_backend": "nccl", "worker_timeout_s": 90})
    for name, parameters in (("collective", {}), ("rank_failure", {"fail_rank": 1}), ("recovery", {})):
        receipt = LocalExecutor().run(distributed.model_copy(update={"parameters": parameters, "output": args.output / name}))
        assert receipt["status"] == ("failed" if parameters else "passed"), receipt
        assert receipt["worker_pids_reaped"]
        receipts[name] = receipt
    public = {"schema_version": 1, "status": "passed", "scope": "P0 runtime infrastructure",
              "model": cfg.agents.default_model, "session_id": session, "usage": usage,
              "usage_repeat_is_idempotent": True, "agent_helper_receipt": agent_receipt,
              "agent_source": (sb.root / "worker_entry.py").read_text(), "executions": receipts}
    write_json(args.output / "public-receipt.json", public)
    print(json.dumps({"status": "passed", "messages": usage["messages"], "cost": usage["reported_cost"],
                      "executions": {k: v["status"] for k, v in receipts.items()}}))


if __name__ == "__main__":
    main()
