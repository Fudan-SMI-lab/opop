"""Artifact-producing project roles sharing the actual configured provider runtime."""
from __future__ import annotations

from pathlib import Path
import json
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
        return self._complete(role, sb, session, prompt, outputs, validator, feedback is not None,
                              fresh_session=True)

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

    def _complete(self, role, sb, session, prompt, outputs, validator, feedback_provided, repairs=0,
                  fresh_session=False, prepared_prompt=False):
        from kernel_optimizer.project.provider_queue import ProviderQueue, retryable_status
        from kernel_optimizer.project.runtime_config import worker_environment
        environment = worker_environment()
        if not prepared_prompt:
            sb.write_input("inputs/worker-environment.json", json.dumps(environment, indent=2))
        environment_note = (" Read inputs/worker-environment.json before library/API inspection. "
            f"The measured worker uses Python {environment['python']}; use that executable for CPU checks. "
            "Other system/Conda environments may have different library versions. GPU work remains in leased framework workers.")
        if role in {"selection", "candidate"} and not prepared_prompt:
            sb.write_input("inputs/binding-contract.md", Path(__file__).with_name("binding_contract.md").read_text())
            environment_note += " Read inputs/binding-contract.md before choosing or composing replacements and capture policies."
        if not prepared_prompt:
            prompt += environment_note
        queue = ProviderQueue(self.runtime.cfg)
        self.store.append("PROJECT_AGENT_STARTED", {"role": role, "session": session, "sandbox": str(sb.root),
                                                     "feedback_provided": feedback_provided})
        snapshots = []
        while True:
            from kernel_optimizer.project.generation_snapshot import capture
            history = None
            if not fresh_session:
                # Diagnostic history capture does not impose a model request limit.
                import httpx
                try:
                    with httpx.Client(base_url=self.runtime.client.base_url, timeout=15.0) as probe:
                        response = probe.get(f"/session/{session}/message", params={"directory": str(sb.root)})
                        response.raise_for_status()
                        history = response.json()
                except Exception:
                    pass
            snapshots.append(str(capture(self.store, sb.root, role=role, session=session, prompt=prompt,
                outputs=outputs, config=self.runtime.cfg, fresh=fresh_session, history=history)))
            fresh_session = False
            with queue.admit():
                response = self.runtime.client.prompt(session, prompt, model=self.runtime.cfg.agents.default_model,
                                                   directory=sb.root, agent=self.runtime.cfg.opencode.agent)
            if response.error:
                failure = response.provider_failure
                status = failure.get("status") if failure else retryable_status(response.error)
                status = status if status in {408, 429, 500, 502, 503, 504} else None
                if status:
                    scheduling = queue.overloaded(status, failure)
                    self.store.append("PROJECT_PROVIDER_RETRY", {"session": session, "role": role, **scheduling})
                    prompt = ("The provider interrupted the previous response with a temporary service/rate error. "
                              "Continue the same requested task from this session and saved files. Preserve all original "
                              "instructions, frozen inputs and output schemas; do not restart completed work or run ad hoc GPU tests."
                              + environment_note)
                    continue
                raise RuntimeError(response.error)
            queue.succeeded()
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
                prompt = self._repair_prompt(error) + environment_note
                repairs += 1
        receipt = {"role": role, "session": session, "message": response.message_id,
                   "model": self.runtime.cfg.agents.default_model,
                   "usage": self.runtime.client.collect_usage(session, sb.root), "sandbox": str(sb.root),
                   "outputs": list(outputs), "feedback_provided": feedback_provided or repairs > 0,
                   "contract_repairs": repairs, "generation_snapshots": snapshots}
        self.store.append("PROJECT_AGENT_FINISHED", receipt)
        return sb.root, receipt

    def replay_snapshot(self, snapshot: Path, *, edit=None, validator=None, allow_source_change=False):
        """Fresh generation from a captured role, with at most one explicit change."""
        from kernel_optimizer.project.generation_snapshot import load, edit_one, public_settings
        from kernel_optimizer.runtime.provenance import framework_source
        saved, prompt, files = load(snapshot)
        if not saved["fresh_session"]:
            raise ValueError("fresh replay requires an initial-role snapshot, not a continuation")
        if not allow_source_change and saved["framework_source"] != framework_source():
            raise ValueError("framework source differs; use the frozen source or record an explicit source change")
        if saved["model"] != self.runtime.cfg.agents.default_model or saved["agent"] != self.runtime.cfg.opencode.agent:
            raise ValueError("replay model/agent must match the captured configuration")
        prompt, files = edit_one(prompt, files, edit)
        sb = self.sandboxes.create(saved["role"] + "-replay-" + uuid4().hex[:12])
        if public_settings(json.loads((sb.root/"opencode.json").read_text())) != saved["tool_model_settings"]:
            raise ValueError("replay tool/model settings differ from snapshot")
        for name, raw in files.items():
            sb.write_input(name, raw)
        session = self.runtime.client.create_session(sb.root, "replay " + saved["role"])
        self.store.append("PROJECT_GENERATION_REPLAY", {"snapshot": str(snapshot), "session": session,
            "edit": edit, "source_change": saved["framework_source"] != framework_source(),
            "original_session": saved["session"], "sandbox": str(sb.root)})
        return self._complete(saved["role"], sb, session, prompt, tuple(saved["outputs"]), validator,
                              "inputs/previous-feedback.txt" in files, fresh_session=True, prepared_prompt=True)
