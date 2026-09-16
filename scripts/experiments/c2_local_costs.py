"""Aggregate existing event costs without treating overlapping durations as GPU busy time."""

from pydantic import BaseModel, ConfigDict, Field

from kernel_optimizer.store.run_store import RunStore


class Costs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    worker_attempts: int = 0
    worker_failures: int = 0
    worker_wall_s: float = 0.0
    agent_calls: int = 0
    agent_attempts: int = 0
    agent_wall_s: float = 0.0
    agent_cost: float = 0.0
    agent_tokens: dict[str, float] = Field(default_factory=dict)
    agent_calls_without_final_accounting: int = 0


def costs(store: RunStore) -> Costs:
    events = store.iter_events()
    starts = {e.payload["call_id"]: e.ts for e in events if e.type == "AGENT_CALL_STARTED"}
    ends = {e.payload["call_id"]: e for e in events if e.type == "AGENT_CALL_FINISHED"
            or (e.type == "AGENT_CALL_FAILED" and e.payload.get("final"))}
    done = [e for e in events if e.type == "LOCAL_EVAL_DONE"]
    tokens: dict[str, float] = {}
    for event in ends.values():
        for name, count in event.payload.get("tokens", {}).items():
            if type(count) in (int, float):
                tokens[name] = tokens.get(name, 0.0) + count
    return Costs(
        worker_attempts=sum(e.type == "LOCAL_EVAL_STARTED" for e in events),
        worker_failures=sum(e.payload["trial"]["status"] != "complete" for e in done),
        worker_wall_s=sum(e.payload["wall_s"] for e in done), agent_calls=len(starts),
        agent_attempts=sum(e.payload.get("attempts", e.payload.get("attempt", 0)) for e in ends.values()),
        agent_wall_s=sum(e.ts - starts[cid] for cid, e in ends.items() if cid in starts),
        agent_cost=sum(e.payload.get("cost", 0.0) for e in ends.values()), agent_tokens=tokens,
        agent_calls_without_final_accounting=len(starts.keys() - ends.keys()),
    )
