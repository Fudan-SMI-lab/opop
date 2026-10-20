"""Measure legal A, B and A+B versions against the same frozen project parent.

This explicit GPU experiment does not constrain provider usage or infer missing
group effects. The candidates must already have been generated and stored.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.replay import transfer_bundle
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.transactions import RevisionStore
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore


def interaction(results):
    """Gain interaction; positive means more gain than the measured singles."""
    usable = all(
        result.get("quality", result.get("evaluation", {}).get("quality")) == "pass"
        and result.get("measurement", {}).get("valid")
        and not result.get("coverage_errors")
        and not result.get("missing_coverage")
        and all(c["passed"] for c in result["measurement"].get("constraints", []))
        for result in results.values()
    )
    if set(results) != {"A", "B", "AB"} or not usable:
        return {"status": "uninterpretable", "reason": "all three legal, quality-passing measured versions are required"}
    sources = {r["measurement"].get("framework_source") for r in results.values()}
    if len(sources) != 1 or sources & {None, "mixed"}:
        return {"status": "uninterpretable", "reason": "framework source differs between arms"}
    gains = {arm: result["measurement"]["paired_gains"] for arm, result in results.items()}
    bounds = {arm: result["measurement"]["noise_bounds"] for arm, result in results.items()}
    if any(not values for values in (*gains.values(), *bounds.values())):
        return {"status": "uninterpretable", "reason": "paired gain/noise evidence is absent"}
    low = min(gains["AB"])-max(gains["A"])-max(gains["B"])
    high = max(gains["AB"])-min(gains["A"])-min(gains["B"])
    noise = sum(max(values) for values in bounds.values())
    return {"status": "measured", "gain_interaction": median(gains["AB"])-median(gains["A"])-median(gains["B"]),
            "paired_gains": gains, "observed_combination_range": [low, high],
            "range_expanded_by_endpoint_noise": [low-noise, high+noise],
            "interpretation": "Each arm is independently paired with the same parent. Range combines observed pair extrema and then endpoint MAD bounds; it is not a statistical confidence interval. Shared effects are not assigned to individual groups."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--base-run", required=True, type=Path)
    p.add_argument("--a-run", required=True, type=Path)
    p.add_argument("--a-bundle", required=True)
    p.add_argument("--b-run", required=True, type=Path)
    p.add_argument("--b-bundle", required=True)
    p.add_argument("--joint-run", required=True, type=Path)
    p.add_argument("--joint-revision", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    config = load_config(args.config, [])
    base = ProjectSearch(config, args.base_run)
    sources = {arm: RevisionStore(RunStore.open(root)) for arm, root in
               (("A", args.a_run), ("B", args.b_run), ("AB", args.joint_run))}
    a, b = sources["A"].bundle(args.a_bundle), sources["B"].bundle(args.b_bundle)
    joint = sources["AB"].revision(args.joint_revision)
    protocol = {"arguments": {k: str(v) for k, v in vars(args).items()},
                "base": base.revisions.current().id, "anchor": base.prepared["revision"],
                "objective": base.request.objective, "workload": base.request.workload,
                "request_sha256": hashlib.sha256((args.base_run/"request.json").read_bytes()).hexdigest(),
                "framework_source": framework_source(),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "order": ["A", "B", "AB"], "reference": "same frozen parent; AB/BA in fresh workers per arm"}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_file = args.output/"protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("cannot resume interaction with changed protocol, base or code")
    write_json(protocol_file, protocol)
    if (a.requires or b.requires or a.replaces or b.replaces
            or set(a.write_set) & set(b.write_set) or set(joint.bundles) != {a.id, b.id}):
        write_json(args.output/"interaction.json", {"status": "not_independently_legal",
            "reason": "subset dependencies, replacements, conflicting writes or a different joint bundle set"})
        return
    if sources["AB"].bundle(a.id) != a or sources["AB"].bundle(b.id) != b:
        raise ValueError("joint version changed a frozen constituent")
    results = {}
    for arm in protocol["order"]:
        if framework_source() != protocol["framework_source"]:
            raise RuntimeError("framework changed during interaction comparison")
        result_file = args.output/(arm+"-result.json")
        if result_file.exists():
            results[arm] = json.loads(result_file.read_text())
            continue
        out = args.output/arm
        if not out.exists():
            fork_run(args.base_run, out, revision_id=protocol["base"])
        search = ProjectSearch(config, out)
        if search.revisions.current().bundles != base.revisions.current().bundles:
            raise ValueError("interaction arm has changed its parent implementation")
        if arm == "AB":
            result = promote_version(search, args.joint_run, args.joint_revision, publish=False)
        else:
            search.current_discovery()
            root, ident = (args.a_run, a.id) if arm == "A" else (args.b_run, b.id)
            bundle, metadata = transfer_bundle(root, ident, search)
            result = search.validate_bundle(bundle, metadata, publish=False)
        write_json(result_file, result)
        results[arm] = result
        print(json.dumps({"arm": arm, "trial": result["trial"], "decision": result["decision"],
                          "measurement": result["measurement"]}), flush=True)
    write_json(args.output/"interaction.json", {**interaction(results), "protocol": protocol})


if __name__ == "__main__":
    main()
