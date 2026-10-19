"""Transaction controls on real GPUs, not an optimization performance experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kernel_optimizer.agents.sandbox import SandboxFactory
from kernel_optimizer.config import load_config
from kernel_optimizer.models.objective import AggregationSpec, ObjectiveSpec, RequiredMetric
from kernel_optimizer.models.project import (
    CallSite, CandidateBundle, ExecutionPlan, FileEdit, GroupRegistry, GroupSpec,
    ProjectContract, ProjectEvaluation,
)
from kernel_optimizer.project.transactions import RevisionStore, identity
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, gpu_inventory, write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime, _sandbox_extra_config


ENTRY = '''import time
import torch
from a import step as a
from b import step as b
def run(context):
    x = torch.arange(256, device=context['device'], dtype=torch.float32)
    expected = (x + 1) * 2
    for _ in range(2):
        b(a(x))
    times = []
    for _ in range(5):
        torch.cuda.synchronize(); start = time.perf_counter_ns()
        output = b(a(x))
        torch.cuda.synchronize(); times.append((time.perf_counter_ns() - start) / 1e6)
    valid = bool(torch.equal(output, expected))
    return {'valid': valid, 'samples_ms': times, 'calls': ['a', 'b'],
            'bundle_hashes': context['revision']['bundle_hashes']}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    store = RunStore.create(args.output.parent, args.output.name, {"scope": "P1 transaction controls"})
    cfg = load_config(args.config)
    source = args.output / "source"; source.mkdir()
    (source / "a.py").write_text("import time\ndef step(x):\n time.sleep(0.002)\n return x+1\n")
    (source / "b.py").write_text("def step(x):\n return x*2\n")
    (source / "entry.py").write_text(ENTRY)
    objective = ObjectiveSpec(name="latency_ms", unit="ms", direction="minimize",
        required_metrics=(RequiredMetric(name="latency_ms", unit="ms"),), aggregation=AggregationSpec(kind="sum"))
    contract = ProjectContract(id="control", goal="transaction lifecycle control", objective=objective,
        workload={"elements": 256}, quality={"exact": True}, completion="CUDA synchronized return",
        invariants=("output equals (x+1)*2",), state_protocol="fresh process", world_size=2)
    plan = ExecutionPlan(id="plan", calls=tuple(CallSite(id=s, phase="run", kind="function", target=s+".py") for s in ("a", "b")))
    groups = GroupRegistry(id="groups", plan_id=plan.id,
        groups=tuple(GroupSpec(id="g"+s, calls=(s,), semantics="control "+s) for s in ("a", "b")))
    revisions = RevisionStore(store)
    base = revisions.initialize(source, ["a.py", "b.py", "entry.py"], contract, plan, groups)
    devices = [item["uuid"] for item in gpu_inventory()][:2]
    assert len(devices) == 2

    def execute(revision, name):
        root = revisions.materialize(revision, args.output / (name+"-source"))
        return LocalExecutor().run(ExecutionRequest(entry=root/"entry.py", cwd=root, devices=devices,
            distributed_backend="nccl", output=args.output/name, lease_dir=args.output.parent/"device-leases",
            worker_timeout_s=90))

    baseline = execute(base, "baseline")
    assert baseline["status"] == "passed" and all(r["result"]["valid"] for r in baseline["ranks"])
    sb = SandboxFactory(args.output/"sandboxes", _sandbox_extra_config(cfg)).create("joint-control")
    for name in ("a.py", "b.py"):
        sb.write_input("reference/"+name, (source/name).read_bytes())
    with Runtime(cfg, log_dir=args.output) as runtime:
        session = runtime.client.create_session(sb.root, "P1 joint boundary control")
        result = runtime.client.prompt(session,
            "Read reference/a.py and reference/b.py. This is a transaction control fixture, not a performance claim. "
            "Write proposal/a.py and proposal/b.py: a.step must compute (x+1)*2 and b.step must return x unchanged. "
            "Remove the synthetic sleep. The pair preserves b(a(x)); either edit alone must fail exact correctness. "
            "Only edit proposal/ inside this directory. Reference inputs and framework code are fixed. "
            "No GPU run is needed here; the framework independently runs both ranks and the partial-edit negative control. "
            "Return the filenames.", model=cfg.agents.default_model, directory=sb.root)
        assert not result.error
        usage = runtime.client.collect_usage(session, sb.root)
    files = {f.path: f for f in base.files}
    edits = tuple(FileEdit(path=name, before_sha256=files[name].sha256,
                          content_ref=store.put_artifact((sb.root/"proposal"/name).read_bytes())) for name in ("a.py", "b.py"))
    bundle = CandidateBundle(id="joint", parent_revision=base.id, contract_ref=base.contract_ref,
        scope="joint", implementation="composite", groups=("ga", "gb"), calls=("a", "b"), edits=edits,
        read_set=("file:a.py", "file:b.py"), write_set=("file:a.py", "file:b.py"), external_invariants=contract.invariants)
    staged = revisions.stage(bundle)
    full = execute(staged, "joint")
    assert full["status"] == "passed" and all(r["result"]["valid"] for r in full["ranks"])
    import statistics
    metric = lambda receipt: max(statistics.median(r["result"]["samples_ms"]) for r in receipt["ranks"])
    report = ProjectEvaluation(id="joint-report", revision_id=staged.id, parent_id=base.id,
        contract_ref=base.contract_ref, bundle_id=bundle.id, quality="pass", feasible=True,
        objective=metric(full), reference_objective=metric(baseline), required_ranks=(0,1),
        rank_bundle_hashes={str(r["rank"]):r["result"]["bundle_hashes"][bundle.id] for r in full["ranks"]},
        required_calls=bundle.calls, executed_calls=("a","b"), evidence=(str(args.output/"joint/receipt.json"),))
    assert revisions.evaluate(report)["accepted"]  # artificial sleep is part of this explicit control fixture
    revisions.commit(staged.id)
    partial_bundle = bundle.model_copy(update={"id":"partial", "edits": edits[:1]})
    partial = revisions.stage(partial_bundle)
    partial_run = execute(partial, "partial")
    assert partial_run["status"] == "passed" and not any(r["result"]["valid"] for r in partial_run["ranks"])
    revisions.abort(partial.id, "partial boundary edit fails quality")
    assert revisions.current().id == staged.id
    revisions.rollback(base.id, "explicit control rollback")
    restored = execute(revisions.current(), "restored")
    assert all(r["result"]["valid"] for r in restored["ranks"])
    orphan = revisions.stage(bundle.model_copy(update={"id":"interrupted"}))
    recovered = revisions.recover()
    assert any(r["revision"] == orphan.id and r["state"] == "aborted" for r in recovered)
    write_json(args.output/"public-receipt.json", {"status":"passed", "scope":"P1 synthetic transaction controls; not model speedup",
        "session_id":session, "usage":usage, "baseline":baseline, "joint":full, "partial":partial_run,
        "restored":restored, "report":report.model_dump(mode="json"), "recovery":recovered,
        "proposal":{name:(sb.root/"proposal"/name).read_text() for name in ("a.py","b.py")}})
    print(json.dumps({"status":"passed", "joint_quality":True, "partial_quality":False, "rollback":True,
                      "interruption_recovery":True, "usage":usage}))


if __name__ == "__main__":
    main()
