"""Agent-accessible execution helper. Job files carry explicit deployment details."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    receipt = LocalExecutor().run(ExecutionRequest.model_validate_json(args.request.read_bytes()))
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
