"""AgentModule: the shared contract for every LLM-backed step.

Each call = fresh sandbox + fresh opencode session with the sandbox as its
working directory, structured output validated against the module's pydantic
schema, retries with error feedback, and full event logging.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, PromptResult
from kernel_optimizer.agents.sandbox import Sandbox, SandboxFactory
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.store.run_store import RunStore

TIn = TypeVar("TIn")
TOut = TypeVar("TOut", bound=BaseModel)


@dataclass
class AgentOutcome(Generic[TOut]):
    output: TOut
    sandbox: Sandbox
    session_id: str
    attempts: int
    tokens: dict
    cost: float


class AgentModule(ABC, Generic[TIn, TOut]):
    name: str = "agent"
    output_model: type[BaseModel]

    def __init__(
        self,
        client: OpencodeClient,
        sandboxes: SandboxFactory,
        store: RunStore,
        cfg: AgentModuleConfig,
        agent_name: str = "build",
    ):
        self.client = client
        self.sandboxes = sandboxes
        self.store = store
        self.cfg = cfg
        self.agent_name = agent_name

    @abstractmethod
    def seed_sandbox(self, inputs: TIn, sb: Sandbox) -> None: ...

    @abstractmethod
    def render_prompt(self, inputs: TIn, sb: Sandbox) -> str: ...

    def invoke(self, inputs: TIn) -> AgentOutcome[TOut]:
        call_id = f"{self.name}-{uuid.uuid4().hex[:8]}"
        sb = self.sandboxes.create(call_id)
        self.seed_sandbox(inputs, sb)
        prompt = self.render_prompt(inputs, sb)

        session_id = self.client.create_session(sb.root, title=call_id)
        # Optional subject of the call, read generically so adding it to one Inputs type
        # needs no change here. Without it a call can only be attributed to a candidate
        # by "nearest following *_PRODUCED event", a heuristic that left 2 of 4 observed
        # repair transport timeouts unattributed -- too fragile to support a claim about
        # whether repeat repairs on the same candidate are the ones that hang.
        subject = getattr(inputs, "candidate_id", None)
        payload = {"module": self.name, "call_id": call_id, "session_id": session_id,
                   "model": self.cfg.model}
        if subject:
            payload["candidate_id"] = subject
        self.store.append("AGENT_CALL_STARTED", payload)

        total_tokens: dict = {}
        total_cost = 0.0
        feedback = ""
        last_error = ""
        transport_retries = 0
        soft_retry_used = False  # improvement L: at most one advisory (non-blocking) retry
        for attempt in range(1, self.cfg.max_retries + 2):
            text = prompt if not feedback else (
                f"Your previous response could not be used:\n{feedback}\n\n"
                f"Fix the problem and answer again. Follow the original instructions "
                f"and output format exactly."
            )
            try:
                result: PromptResult = self.client.prompt(
                    session_id,
                    text,
                    model=self.cfg.model or "",
                    agent=self.agent_name,
                    schema=self.output_model.model_json_schema(),
                    directory=sb.root,
                )
            except AgentCallError as exc:
                last_error = str(exc)
                self.store.append(
                    "AGENT_CALL_FAILED",
                    {"module": self.name, "call_id": call_id, "attempt": attempt,
                     "error": last_error[:1000]},
                )
                # A transport timeout means no response ever arrived, so there is nothing
                # for the agent to "fix" — sending corrective feedback would make it
                # apologise for a message it never sent. Keep the ORIGINAL prompt, and
                # move to a fresh session: prompt() already aborted this one, and a
                # second generation queued behind an aborted turn is what made one L3:43
                # repair burn 0.99h (two 20-min ReadTimeouts) before succeeding.
                feedback = ""
                transport_retries += 1
                if transport_retries > self.cfg.max_transport_retries:
                    break
                try:
                    session_id = self.client.create_session(sb.root, title=f"{call_id}-r{attempt}")
                    self.store.append(
                        "AGENT_SESSION_RESET",
                        {"module": self.name, "call_id": call_id, "attempt": attempt,
                         "session_id": session_id, "reason": "transport_timeout"},
                    )
                except Exception as reset_exc:  # keep the old session rather than crash
                    last_error = f"{last_error} (session reset failed: {reset_exc})"
                continue

            total_cost += result.cost
            for key, value in (result.tokens or {}).items():
                if isinstance(value, (int, float)):
                    total_tokens[key] = total_tokens.get(key, 0) + value

            if result.structured is None:
                feedback = ("no parseable JSON found in your response; emit a single "
                            "fenced ```json block matching the required schema")
                last_error = feedback
                continue
            try:
                output = self.output_model.model_validate(result.structured)
            except ValidationError as exc:
                feedback = f"JSON did not match the required schema:\n{exc}"[:2000]
                last_error = feedback
                continue

            problem = self.check_output(output, sb)
            if problem:
                feedback = problem
                last_error = problem
                continue

            # Non-blocking advisory warnings (improvement L). These NEVER reject the
            # output; they are logged for observability and, at most once and only if
            # retry budget remains, fed back so the agent can improve. If the agent
            # ignores them or budget is spent, the output is still accepted.
            warnings = self.soft_check(output, sb)
            if warnings:
                self.store.append(
                    "AGENT_SOFT_WARNING",
                    {"module": self.name, "call_id": call_id, "attempt": attempt,
                     "warnings": warnings},
                )
                if not soft_retry_used and attempt < self.cfg.max_retries + 1:
                    soft_retry_used = True
                    feedback = (
                        "Your output is usable, but consider improving it:\n- "
                        + "\n- ".join(warnings)
                        + "\nIf the current output already handles this, keep it."
                    )
                    continue

            self.store.append(
                "AGENT_CALL_FINISHED",
                {"module": self.name, "call_id": call_id, "attempt": attempt,
                 "tokens": total_tokens, "cost": total_cost},
            )
            return AgentOutcome(
                output=output,  # type: ignore[arg-type]
                sandbox=sb,
                session_id=session_id,
                attempts=attempt,
                tokens=total_tokens,
                cost=total_cost,
            )

        self.store.append(
            "AGENT_CALL_FAILED",
            {"module": self.name, "call_id": call_id, "final": True,
             "error": last_error[:1000]},
        )
        raise AgentCallError(
            f"{self.name} failed after {self.cfg.max_retries + 1} attempts: {last_error[:500]}"
        )

    def check_output(self, output: TOut, sb: Sandbox) -> str | None:
        """Post-validate (e.g. referenced files exist). Return problem text or None."""
        return None

    def soft_check(self, output: TOut, sb: Sandbox) -> list[str]:
        """Non-blocking advisory warnings about an otherwise-usable output. Never
        rejects it; surfaced for observability and (once, budget permitting) fed back
        so the agent can improve. Default: none."""
        return []
