"""Explicit T2 evidence export, separate from immutable T1 goldens; no Git/network."""

import argparse
import ast
import hashlib
import json
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from kernel_optimizer import wiring
from kernel_optimizer.config import AppConfig
from kernel_optimizer.store.run_store import RunStore
from tests.c2_contract_capture import Captured, RequestCapture, capture_at, load_anchor, request_at
from tests.c3_search_fakes import no_sandbox_git
from tests.test_task_agent_profiles import operator_inputs
from tests.c2_contract_capture import RecordingProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    captured: dict[str, RequestCapture] = {}
    with TemporaryDirectory(prefix="t2-profile-") as temporary, pytest.MonkeyPatch.context() as patch:
        no_sandbox_git.__wrapped__(patch)
        root = Path(temporary)
        captured["existing_generic"] = capture_at(root / "generic", "generic")
        captured["legacy"] = capture_at(root / "legacy", "legacy")
        profiles: tuple[wiring.ExecutionProfile, ...] = ("c2_direct_compat", "model_operator")
        for profile in profiles:
            inputs = request_at(root / profile)
            if profile == "model_operator":
                inputs = operator_inputs(inputs)
            with closing(RecordingProvider(inputs.project_root)) as provider:
                runtime = wiring.Runtime(AppConfig())
                runtime.client = provider
                agent = wiring.build_task_rewriter(runtime.cfg, RunStore(inputs.project_root / "run"), runtime,
                                                   execution_profile=profile)
                with pytest.raises(Captured):
                    agent.invoke(inputs)
                captured[profile] = provider.requests[0]
    comparisons = []
    for profile, capture in captured.items():
        (output / f"{profile}-request.json").write_text(capture.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (output / f"{profile}-normalization.json").write_text(json.dumps(capture.normalization_map, indent=2), encoding="utf-8")
    for profile, anchor in (("existing_generic", "8526a13-generic"), ("legacy", "8526a13-legacy"),
                            ("c2_direct_compat", "cda1130-direct")):
        actual, expected = captured[profile], load_anchor(anchor)
        comparisons.append({"profile": profile, "anchor": anchor,
            "effective_prompt_equal": actual.prompt == expected.prompt,
            "schema_equal": actual.output_schema == expected.output_schema,
            "seeded_files_equal": actual.files == expected.files,
            "complete_request_equal": actual.model_dump() == expected.model_dump(),
            "prompt_sha256": hashlib.sha256(actual.prompt.encode()).hexdigest()})
    root = Path(__file__).parents[1]
    protected = []
    for name in ("task_rewriter.py", "modules.py", "method_prompts.py", "base.py"):
        relative = Path("src/kernel_optimizer/agents") / name
        actual = (root / relative).read_text(encoding="utf-8")
        baseline = (args.baseline / relative).read_text(encoding="utf-8")
        protected.append({"path": relative.as_posix(), "text_equal": actual == baseline,
                          "sha256_utf8_lf": hashlib.sha256(actual.encode()).hexdigest()})
    bodies = []
    for source_root in (args.baseline, root):
        source = (source_root / "src/kernel_optimizer/wiring.py").read_text(encoding="utf-8")
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "build_orchestrator")
        bodies.append(ast.get_source_segment(source, node))
    protected.append({"symbol": "wiring.build_orchestrator", "text_equal": bodies[0] == bodies[1]})
    goldens = {path.name: path.read_bytes() == (root / "results/c3-fast-kernel-iteration/setup/compatibility" / path.name).read_bytes()
               for path in (root / "tests/fixtures/c2_contracts").glob("*.json")}
    changed = ("src/kernel_optimizer/models/device_operator.py", "src/kernel_optimizer/agents/c2_direct_compat.py",
        "src/kernel_optimizer/agents/model_operator_rewriter.py", "src/kernel_optimizer/wiring.py",
        "scripts/experiments/c2_local_agents.py", "examples/c3_qwen3/operator_cli.py",
        "tests/test_device_operator.py", "tests/test_task_agent_profiles.py", "tests/test_task_profile_validation.py",
        "tests/test_c3_task_routing.py", "tests/task_profile_evidence.py")
    sizes = {name: sum(bool(line.strip()) and not line.lstrip().startswith("#")
        for line in (root / name).read_text(encoding="utf-8").splitlines()) for name in changed}
    report = {"comparisons": comparisons, "protected": protected,
        "t1_goldens_byte_equal": goldens, "pure_loc": sizes,
        "baseline_source": str(args.baseline), "publication": "173cb9e26e41d0912909d59a24758ff909a7a246"}
    (output / "contract-comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
