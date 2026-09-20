"""One-time T1 artifact export; source revision is selected by isolated PYTHONPATH."""

import argparse
import json
import platform
import hashlib
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kernel_optimizer import wiring
from kernel_optimizer.config import AppConfig
from tests.c2_contract_capture import FIXTURES, Route, capture_at
from tests.c3_search_fakes import no_sandbox_git


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["cda1130", "8526a13", "worktree"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    routes: tuple[Route, ...] = ("generic", "legacy") if args.label == "cda1130" else ("generic", "legacy", "bundle")
    with TemporaryDirectory(prefix="t1-contract-") as temporary, pytest.MonkeyPatch.context() as patch:
        no_sandbox_git.__wrapped__(patch)
        for route in routes:
            capture = capture_at(Path(temporary) / route, route)
            name = "direct" if args.label == "cda1130" and route == "generic" else route
            stem = f"{args.label}-{name}"
            text = capture.model_dump_json(indent=2) + "\n"
            (output / f"{stem}.json").write_text(text, encoding="utf-8")
            (output / f"{stem}-normalization.json").write_text(
                json.dumps(capture.normalization_map, indent=2) + "\n", encoding="utf-8")
            if args.write_fixtures:
                (FIXTURES / f"{stem}.json").write_text(text, encoding="utf-8")
            print(f"captured {stem}: {len(text.encode())} bytes; {len(capture.files)} seeded files")
    cfg = AppConfig()
    runtime = {"source_root": str(Path(wiring.__file__).parents[2]), "label": args.label,
        "python": platform.python_version(), "pydantic": version("pydantic"), "optuna": version("optuna"),
        "rewriter": cfg.agents.module("rewriter").model_dump(mode="json"), "seed": cfg.run.seed,
        "budgets": cfg.budgets.model_dump(mode="json"),
        "normalization": ["exact sandbox root and session UUID -> <SANDBOX>",
            "exact fixture root -> <CASE>", "exact selected source/src root -> <SOURCE>",
            "same mappings for JSON-escaped Windows and /mnt drive forms; no schema/title/field normalization",
            "exact shlex-quoted fixture PYTHONPATH argument -> canonical quoted source-root placeholder",
            "concrete aliases recorded in each request's separate normalization JSON"],
        "provider_calls": 0, "gpu_jobs": 0}
    if args.label == "worktree":
        runtime["test_support_pure_loc"] = {name: sum(bool(line.strip()) and not line.lstrip().startswith("#")
            for line in (Path(__file__).parent / name).read_text(encoding="utf-8").splitlines())
            for name in ("c2_contract_capture.py", "c2_capture_export.py", "test_c2_agent_compatibility.py", "test_c3_task_routing.py")}
    (output / f"{args.label}-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    if args.label == "worktree":
        comparisons = []
        pairs = [("cda1130-direct", "8526a13-generic"), ("cda1130-legacy", "8526a13-legacy"),
                 *[(f"8526a13-{route}", f"worktree-{route}") for route in routes]]
        for left, right in pairs:
            before = json.loads((output / f"{left}.json").read_text(encoding="utf-8"))
            after = json.loads((output / f"{right}.json").read_text(encoding="utf-8"))
            comparisons.append({"left": left, "right": right,
                "complete_request_equal": before == after,
                "effective_prompt_equal": before["prompt"] == after["prompt"],
                "output_schema_equal": before["output_schema"] == after["output_schema"],
                "seeded_files_equal": before["files"] == after["files"],
                "changed_seeded_files": [name for name in before["files"].keys() | after["files"].keys()
                    if before["files"].get(name) != after["files"].get(name)],
                "prompt_sha256": {label: hashlib.sha256(value["prompt"].encode()).hexdigest()
                    for label, value in ((left, before), (right, after))}})
        (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
