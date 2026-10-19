"""Artifact-producing project roles sharing the actual configured provider runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from uuid import uuid4

from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, _sandbox_extra_config


class ProjectAgents:
    def __init__(self, runtime: Runtime, store: RunStore):
        self.runtime, self.store = runtime, store
        self.sandboxes = SandboxFactory(store.run_dir/"sandboxes", _sandbox_extra_config(runtime.cfg))

    def generate(self, role: str, inputs: dict[str, str | bytes], instructions: str,
                 outputs: tuple[str, ...], feedback: str | None = None,
                 validator: Callable[[Path], None] | None = None) -> tuple[Path, dict]:
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
        return self._complete(role, sb, session, prompt, outputs, validator, feedback is not None)

    def repair(self, role: str, directory: Path, session: str, outputs: tuple[str, ...],
               validator: Callable[[Path], None], failure: str) -> tuple[Path, dict]:
        sb = Sandbox(directory)
        self._save_rejection(role, sb, session, outputs, failure)
        return self._complete(role, sb, session, self._repair_prompt(failure), outputs, validator, True, repairs=1)

    @staticmethod
    def _repair_prompt(failure):
        return ("The framework rejected the output contract before execution. Repair the requested output files, "
                "preserving the valid implementation and frozen task. Do not rerun GPU work or modify framework/configuration. "
                "Follow the supplied JSON schema; move descriptive prose to rationale. Actual validation error:\n" + failure)

    def _save_rejection(self, role, sb, session, outputs, error):
        refs = {name: self.store.put_artifact((sb.root/name).read_bytes()) for name in outputs if (sb.root/name).is_file()}
        self.store.append("PROJECT_OUTPUT_REJECTED", {"role": role, "session": session, "error": error, "artifacts": refs})

    def _complete(self, role, sb, session, prompt, outputs, validator, feedback_provided, repairs=0):
        self.store.append("PROJECT_AGENT_STARTED", {"role": role, "session": session, "sandbox": str(sb.root),
                                                     "feedback_provided": feedback_provided})
        while True:
            response = self.runtime.client.prompt(session, prompt, model=self.runtime.cfg.agents.default_model,
                                               directory=sb.root, agent=self.runtime.cfg.opencode.agent)
            if response.error:
                raise RuntimeError(response.error)
            try:
                missing = [name for name in outputs if not (sb.root/name).is_file()]
                if missing:
                    raise ValueError(f"{role} omitted outputs {missing}")
                if validator:
                    validator(sb.root)
                break
            except (ValueError, SyntaxError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:6000]
                self._save_rejection(role, sb, session, outputs, error)
                prompt = self._repair_prompt(error)
                repairs += 1
        receipt = {"role": role, "session": session, "message": response.message_id,
                   "model": self.runtime.cfg.agents.default_model,
                   "usage": self.runtime.client.collect_usage(session, sb.root), "sandbox": str(sb.root),
                   "outputs": list(outputs), "feedback_provided": feedback_provided or repairs > 0, "contract_repairs": repairs}
        self.store.append("PROJECT_AGENT_FINISHED", receipt)
        return sb.root, receipt
