"""Fresh process boundary: bind devices before importing user code or torch."""
from __future__ import annotations

from datetime import timedelta
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.runtime.execution import ExecutionRequest, write_json


def main() -> int:
    request = ExecutionRequest.model_validate_json(Path(sys.argv[1]).read_bytes())
    rank = int(sys.argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(request.devices)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    context = {"rank": rank, "world_size": len(request.devices) if request.distributed_backend else 1,
               "device": f"cuda:{rank}" if request.devices else "cpu", "output": str(request.output),
               "parameters": request.parameters, "binding": {"pid": os.getpid(), "status": "cpu"}}
    occupancy = None
    try:
        if request.devices and os.name != "nt":
            from kernel_optimizer.runtime.occupancy import OccupancyMonitor
            occupancy = OccupancyMonitor(request.devices, sys.argv[1])
            occupancy.start()
        manifest = request.cwd / ".opop-revision.json"
        if manifest.is_file():
            revision = json.loads(manifest.read_text())
            for name, digest in revision["files"].items():
                assert hashlib.sha256((request.cwd / name).read_bytes()).hexdigest() == digest, name
            context["revision"] = revision
        if request.devices:
            import torch
            assert torch.cuda.device_count() == len(request.devices)
            torch.cuda.set_device(rank)
            proof = torch.ones(1, device=context["device"])
            torch.cuda.synchronize()
            raw_uuid = str(torch.cuda.get_device_properties(rank).uuid)
            # torch versions expose either the UUID alone or NVIDIA's GPU-prefixed form.
            actual = next((device for device in request.devices
                           if device.removeprefix("GPU-").lower() == raw_uuid.removeprefix("GPU-").lower()), raw_uuid)
            assert actual == request.devices[rank], (actual, request.devices[rank])
            query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                   capture_output=True, text=True)
            pid_uuids = [line.split(",", 1)[1].strip() for line in query.stdout.splitlines()
                         if "," in line and line.split(",", 1)[0].strip() == str(os.getpid())]
            context["binding"] = {"pid": os.getpid(), "status": "verified", "torch_uuid": actual,
                                  "requested_uuid": request.devices[rank], "torch_raw_uuid": raw_uuid, "logical_device": rank,
                                  "nvidia_smi_pid_uuids": pid_uuids,
                                  "pid_query_status": "observed" if pid_uuids else "unavailable"}
            if pid_uuids:
                assert actual in pid_uuids
            del proof
            write_json(request.output / f"rank-{rank}-started.json", context)
            if request.distributed_backend:
                torch.distributed.init_process_group(
                    backend=request.distributed_backend, init_method=(request.output / "rendezvous").as_uri(),
                    rank=rank, world_size=context["world_size"],
                    timeout=timedelta(seconds=request.collective_timeout_s))
        sys.path.insert(0, str(request.entry.parent))
        spec = importlib.util.spec_from_file_location("opop_project_entry", request.entry)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        result = getattr(module, request.function)(context)
        if request.devices:
            torch.cuda.synchronize()
        if occupancy:
            observation = occupancy.finish(); occupancy = None
            context["binding"]["occupancy"] = observation
            if observation["external_clients"]:
                raise RuntimeError("unrelated CUDA processes overlapped this worker; measurement discarded")
        write_json(request.output / f"rank-{rank}.json", {"status": "passed", "binding": context["binding"], "result": result})
        return 0
    except BaseException as exc:
        write_json(request.output / f"rank-{rank}.json", {"status": "failed", "binding": context["binding"],
                   "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        return 1
    finally:
        if occupancy:
            context["binding"]["occupancy"] = occupancy.finish()
        if request.devices and "torch" in locals() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
