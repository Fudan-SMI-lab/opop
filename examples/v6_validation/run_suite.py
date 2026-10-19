"""Sequential, bounded readiness runs. This is not the future C3 workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=["qwen_inference", "resnet_inference", "gpt2_ddp_training"], default=["qwen_inference", "resnet_inference", "gpt2_ddp_training"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("GPU readiness suite currently runs on Linux")
    if args.output.exists():
        parser.error("Choose a new output directory; prior run evidence is immutable")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parent
    deployment = json.loads(args.deployment.read_text())
    receipts = []
    for case in args.cases:
        target = args.output / case
        target.mkdir()
        command = [sys.executable]
        devices = deployment["single_gpu_visible_devices"]
        if case == "gpt2_ddp_training":
            command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
            devices = deployment["dual_gpu_visible_devices"]
        command += [str(root / f"{case}.py"), "--output", str(target), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        if case in deployment["assets"]:
            command += ["--assets", deployment["assets"][case]]
        if args.profile:
            command.append("--profile")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        # Dedicated NCCL debug artifacts include the transport actually chosen.
        if case == "gpt2_ddp_training":
            env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH", NCCL_DEBUG_FILE=str(target / "nccl.%h.%p.log"))
        started = time.time()
        print("starting", case, flush=True)
        with (target / "run.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                code = child.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    code = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait(timeout=10)
        receipts.append({"case": case, "command": command, "visible_devices": devices, "started_unix": started, "duration_seconds": time.time() - started, "exit_code": code, "timed_out": timed_out})
        (args.output / "suite-receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print("finished", case, "exit", code, flush=True)
    if any(row["exit_code"] != 0 or row["timed_out"] for row in receipts):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
