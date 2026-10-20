"""Compare passive timelines with/without gradient callbacks on fixed revisions."""
import argparse
import hashlib
import json
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import cross_rank
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def summarize(report):
    communication = report["communication"]
    keys = ("communication_kernel_count", "communication_kernel_sum_ms", "communication_union_ms",
            "compute_overlap_ms", "communication_without_compute_ms", "logical_input_bytes")
    return {"rank": report["rank"], "profiled_wall_ms": report["profiled_wall_ms"],
            "trace": report["trace"], "trace_sha256": report["trace_sha256"],
            "gradient_hooks_enabled": report["gradient_hooks_enabled"], "clock": report["clock"],
            "communication": {k: communication[k] for k in keys},
            "collectives": [{k: c[k] for k in ("sequence", "operation", "logical_input_bytes", "cpu_ranges",
                                              "cuda_launches", "cpu_submission", "launch_links", "gpu_input_ready", "consumer_completion")}
                            for c in communication["collectives"]],
            "devices": {d: {k: v for k, v in data.items() if not k.endswith("intervals_us")}
                        for d, data in report["devices"].items()},
            "cuda_synchronization": report["cuda_synchronization"],
            "copies_and_memsets": {"count": len(report["copies_and_memsets"]),
                                  "inclusive_duration_us": sum(v["duration_us"] for v in report["copies_and_memsets"])},
            "gradient_stream_markers": report.get("gradient_stream_markers"),
            "gradient_readiness": {"events": len(report["gradient_readiness"]["events"]),
                "buckets": [{k: v for k, v in bucket.items() if k != "cpu_post_accumulate_times_us"}
                            for bucket in report["gradient_readiness"]["buckets"]]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--parent", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pairs", type=int, default=2)
    args = p.parse_args()
    search = ProjectSearch(load_config(args.config, []), args.run)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"source": framework_source(), "arguments": {k: str(v) for k, v in vars(args).items()},
                "accepted_at_start": search.revisions.current().id,
                "request_sha256": hashlib.sha256((args.run/"request.json").read_bytes()).hexdigest(),
                "scope": "Developer-launched separate timeline diagnostic. No quality/performance acceptance or publication."}
    f = args.output/"protocol.json"
    if f.exists() and json.loads(f.read_text()) != protocol:
        raise ValueError("timeline protocol changed")
    write_json(f, protocol)
    results = []
    for hooks in (False, True):
        for pair in range(args.pairs):
            for arm in (("parent", "candidate") if pair % 2 == 0 else ("candidate", "parent")):
                name = f"{arm}-hooks{int(hooks)}-pair{pair}"
                receipt = args.output/f"{name}.json"
                if receipt.exists():
                    run = json.loads(receipt.read_text())
                else:
                    run = search.execute(search.revisions.revision(getattr(args, arm)), args.output.name+"-"+name,
                                         "execution_timeline", collector_options={"gradient_readiness": hooks, "repeats": 1})
                    write_json(receipt, run)
                if run["status"] != "passed":
                    raise ValueError(f"timeline failed: {name}")
                reports = [json.loads(Path(ref).read_text()) for r in run["ranks"] for ref in r["result"]["reports"]]
                result = {"name": name, "ranks": [summarize(r) for r in reports], "cross_rank": cross_rank(reports)}
                results.append(result)
                write_json(args.output/"summary.json", {"protocol": protocol, "results": results})
                print(json.dumps({"name": name, "ranks": [
                    {"rank": r["rank"], "wall_ms": r["profiled_wall_ms"], **r["communication"],
                     "gradient_events": r["gradient_readiness"]["events"]} for r in result["ranks"]],
                    "cpu_matched": sum(r["status"] == "observed" for r in result["cross_rank"]["collectives"])}), flush=True)
    if search.revisions.current().id != protocol["accepted_at_start"]:
        raise ValueError("accepted revision changed during diagnostic")


if __name__ == "__main__":
    main()
