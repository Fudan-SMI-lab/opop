"""Real provider smoke for an explicitly unbounded request, not C3 acceptance.

Run with the v6 source on PYTHONPATH. Output contains private OpenCode configs
and messages; publish only receipt.json and the reviewed agent assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from kernel_optimizer.config import load_config
from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.wiring import Runtime, _sandbox_extra_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    assert cfg.opencode.request_timeout_s is None
    assert cfg.opencode.idle_abort_frac == 0
    sandbox = SandboxFactory(args.output / "sandboxes", _sandbox_extra_config(cfg)).create("assessment")
    inputs = {}
    for name in ("common.py", "resnet_inference.py"):
        source = args.repo / "examples" / "v6_validation" / name
        data = source.read_bytes()
        sandbox.write_input(name, data)
        inputs[name] = hashlib.sha256(data).hexdigest()
    prompt = (
        "Read common.py and resnet_inference.py in this directory using your tools. "
        "Write assessment.json with a concise grounded assessment of this inference "
        "workload: entrypoint, measured_metrics, correctness_checks, "
        "possible_optimization_scopes, and missing_for_general_workflow. "
        "Use arrays of strings for the last four fields and a string for entrypoint. "
        "Differentiate observations from optimization hypotheses. This is source "
        "inspection for development; no benchmark or model download is needed. "
        "Do not modify the input files. Return the assessment filename."
    )
    started = time.monotonic()
    with Runtime(cfg, log_dir=args.output) as runtime:
        client = runtime.client
        assert client is not None and client._http.timeout.read is None
        session = client.create_session(sandbox.root, "v6 no-deadline workload inspection")
        result = client.prompt(session, prompt, model=cfg.agents.default_model,
                               agent=cfg.opencode.agent, directory=sandbox.root)
        if result.error:
            raise RuntimeError("Provider error; inspect private server/session logs")
        response = client._http.get(f"/session/{session}/message",
                                    params={"directory": str(sandbox.root)})
        response.raise_for_status()
        messages = response.json()
        assert isinstance(messages, list)
        (args.output / "private-messages.json").write_text(json.dumps(messages, indent=2))
    assessment = json.loads(sandbox.read_output("assessment.json"))
    assert isinstance(assessment.get("entrypoint"), str) and assessment["entrypoint"]
    for field in ("measured_metrics", "correctness_checks", "possible_optimization_scopes",
                  "missing_for_general_workflow"):
        assert isinstance(assessment[field], list) and assessment[field]
        assert all(isinstance(item, str) for item in assessment[field])
    assert all(hashlib.sha256((sandbox.root / name).read_bytes()).hexdigest() == digest
               for name, digest in inputs.items())
    (args.output / "assessment.json").write_text(json.dumps(assessment, indent=2))
    unique = {m["info"]["id"]: m for m in messages if m.get("info", {}).get("role") == "assistant"}
    turns = [{"id": ident, "cost": message["info"].get("cost"),
              "tokens": message["info"].get("tokens", {}),
              "tools": [{"name": p.get("tool"), "status": p.get("state", {}).get("status")}
                        for p in message.get("parts", []) if p.get("type") == "tool"]}
             for ident, message in unique.items()]
    receipt = {
        "status": "passed", "scope": "real request and source inspection, not C3 workflow acceptance",
        "model": cfg.agents.default_model, "session_id": session, "message_id": result.message_id,
        "wall_s": time.monotonic() - started, "request_timeout_s": cfg.opencode.request_timeout_s,
        "idle_abort_frac": cfg.opencode.idle_abort_frac, "inputs_sha256": inputs,
        "usage_policy": "observe_only", "assistant_messages": turns,
        "session_reported_cost": sum(t["cost"] for t in turns if isinstance(t["cost"], (int, float))),
        "final_response_reported_cost": result.cost,
        "cost_scope": "unique assistant messages returned by session API; provider estimates, not invoice",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("status", "model", "wall_s", "session_reported_cost")}), flush=True)


if __name__ == "__main__":
    main()
