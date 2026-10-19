"""Artifact-producing project roles sharing the actual configured provider runtime."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, _sandbox_extra_config


class ProjectAgents:
    def __init__(self, runtime: Runtime, store: RunStore):
        self.runtime, self.store = runtime, store
        self.sandboxes = SandboxFactory(store.run_dir/"sandboxes", _sandbox_extra_config(runtime.cfg))

    def generate(self, role: str, inputs: dict[str, str | bytes], instructions: str,
                 outputs: tuple[str, ...], feedback: str | None = None) -> tuple[Path, dict]:
        ident = role + "-" + uuid4().hex[:12]
        sb = self.sandboxes.create(ident)
        for name, content in inputs.items():
            sb.write_input("inputs/" + name, content)
        session = self.runtime.client.create_session(sb.root, ident)
        prompt = ("Work on the supplied project artifacts. Inputs and framework code are fixed. "
                  "Write only the requested output files in this directory; do not modify external source, "
                  "provider configuration, quality thresholds or workload. The framework will execute and "
                  "validate the output independently.\n" + instructions)
        if feedback:
            sb.write_input("inputs/previous-feedback.txt", feedback)
            prompt += "\nRead inputs/previous-feedback.txt and repair the concrete failure."
        self.store.append("PROJECT_AGENT_STARTED", {"role": role, "session": session, "sandbox": str(sb.root),
                                                     "feedback_provided": feedback is not None})
        response = self.runtime.client.prompt(session, prompt, model=self.runtime.cfg.agents.default_model,
                                               directory=sb.root, agent=self.runtime.cfg.opencode.agent)
        receipt = {"role": role, "session": session, "message": response.message_id,
                   "usage": self.runtime.client.collect_usage(session, sb.root), "sandbox": str(sb.root),
                   "outputs": list(outputs), "feedback_provided": feedback is not None}
        self.store.append("PROJECT_AGENT_FINISHED", receipt)
        if response.error:
            raise RuntimeError(response.error)
        missing = [name for name in outputs if not (sb.root/name).is_file()]
        if missing:
            raise ValueError(f"{role} omitted outputs {missing}; sandbox={sb.root}")
        return sb.root, receipt
