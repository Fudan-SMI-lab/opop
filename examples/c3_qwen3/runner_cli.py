import argparse
import json
from pathlib import Path

from .runner_prepare import prepare


def main() -> int:
    parser = argparse.ArgumentParser(description="C3 task-local resident runner; GPU campaign admission belongs to T6.")
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("prepare", help="verify the frozen task using CPU-only JSON/hash checks")
    cpu.add_argument("--contract", type=Path, required=True)
    cpu.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare(args.contract, args.assets)
    print(json.dumps({"contract_sha256": prepared.contract_sha256, "corpus_sha256": prepared.corpus_sha256,
                      "prompt_count": len(prepared.prompts_by_id), "model_loaded": False}))
    return 0
