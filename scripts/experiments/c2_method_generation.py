"""Three bounded model chains; P/H parameterize the same immutable proposal."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kernel_optimizer.agents.runtime import AgentCallError
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs
from kernel_optimizer.paramspace.materializer import MaterializeError
from scripts.experiments import c2_local_agents as agents
from scripts.experiments.c2_local_agents import Child, Services
from scripts.experiments.c2_local_inputs import Shared
from scripts.experiments.c2_method_protocol import Arm, Deadline, SchedulingExpired, Status


@dataclass(frozen=True, slots=True)
class GenerationContext:
    shared: Shared
    services: Services
    inputs: TaskRewriteInputs
    deadline: Deadline


@dataclass(frozen=True, slots=True)
class Generated:
    arm: Arm
    child: Child | None = None
    proposal: Path | None = None
    status: Status = "complete"
    error: str | None = None


def generate_chain(chain: Literal["G0", "L", "P"], context: GenerationContext) -> tuple[Generated, ...]:
    arms: tuple[Arm, ...] = ("P", "H") if chain == "P" else (chain,)
    services = context.services
    try:
        context.deadline.check()
        proposal = agents.generate_legacy_proposal(
            context.shared, services, context.inputs,
            reasoning_mode="legacy_local" if chain == "L" else "whole_task",
        )
        proposal_path = services.store.run_dir / "proposal.py"
        proposal_path.write_text(proposal.source, encoding="utf-8")
        (services.store.run_dir / "proposal.json").write_text(json.dumps({
            "source": str(proposal_path), "backend": proposal.backend,
            "change_summary": proposal.change_summary, "hypothesis_id": proposal.hypothesis_id,
            "report": proposal.report.model_dump(mode="json"),
        }, indent=2), encoding="utf-8")
    except SchedulingExpired as exc:
        return tuple(Generated(arm, status="censored", error=str(exc)) for arm in arms)
    except (AgentCallError, MaterializeError, OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        services.store.append("METHOD_PROPOSAL_FAILED", {"chain": chain, "error": error})
        return tuple(Generated(arm, status="dependency_failed" if arm == "H" else "failed", error=error)
                     for arm in arms)
    results: list[Generated] = []
    for arm in arms:
        try:
            context.deadline.check()
            child = agents.parameterize_legacy_proposal(
                context.shared, services, proposal, pass_intent=arm in {"G0", "H"},
            )
            results.append(Generated(arm, child=child, proposal=proposal_path))
        except SchedulingExpired as exc:
            results.append(Generated(arm, proposal=proposal_path, status="censored", error=str(exc)))
        except (AgentCallError, MaterializeError, OSError, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            services.store.append("METHOD_PARAMETERIZATION_FAILED", {"arm": arm, "error": error})
            results.append(Generated(arm, proposal=proposal_path, status="failed", error=error))
    return tuple(results)
