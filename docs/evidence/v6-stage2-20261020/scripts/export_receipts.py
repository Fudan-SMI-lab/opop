"""Export completed receipts and unchanged raw-file hashes, never live model credentials."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import zipfile

root = Path("/root/autodl-tmp/dev/v6-work")
out = root/"stage2-public-evidence"
out.mkdir(exist_ok=True)
manifest = {}


def raw(path):
    path = Path(path)
    if not path.is_file() or str(path) in manifest:
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(chunk)
    manifest[str(path)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def put(name, value):
    path = out/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+"\n")


def diagnostic_files(value, run):
    if isinstance(value, dict):
        for item in value.values():
            diagnostic_files(item, run)
    elif isinstance(value, list):
        for item in value:
            diagnostic_files(item, run)
    elif isinstance(value, str) and value.startswith(str(run)+"/") and len(value) < 4096:
        path = Path(value)
        if path.suffix in (".json", ".jsonl", ".nsys-rep", ".sqlite", ".prof", ".pstats", ".log"):
            raw(path)


def compact_run(run):
    return {"status": run.get("status"), "failure": run.get("failure"), "ranks": [
        {"rank": rank["rank"], "status": rank.get("status"), "error": rank.get("error"),
         "framework_source": rank.get("framework_source"),
         "result": {k: rank.get("result", {}).get(k) for k in
                    ("valid", "checks", "coverage", "setup", "samples", "metrics", "diagnostics", "metric_samples")}}
        for rank in run.get("ranks", [])]}


def collector_summary(observation):
    ranks = []
    for rank in observation.get("ranks", []):
        communication = rank.get("communication", {})
        gradients = rank.get("gradient_readiness", {})
        collectives = communication.get("collectives")
        payload = (sum(c["logical_input_bytes"] for c in collectives)
                   if collectives and all(isinstance(c.get("logical_input_bytes"), int) for c in collectives) else None)
        ranks.append({k: rank.get(k) for k in
                      ("rank", "scope", "profiled_wall_ms", "workload_window", "clock", "trace", "trace_sha256")}
                     | {"communication_status": communication.get("status"),
                        "collective_count": len(collectives) if collectives is not None else None,
                        "logical_input_bytes_sum": payload,
                        "communication_kernel_sum_ms": communication.get("communication_kernel_sum_ms"),
                        "gradient_callbacks": len(gradients["events"]) if "events" in gradients else None,
                        "buckets": len(gradients["buckets"]) if "buckets" in gradients else None})
    return {k: observation.get(k) for k in ("status", "reason", "provenance", "scope")} | {"ranks": ranks}


def counted_fields(value, fields):
    if not isinstance(value, dict):
        return value
    result = dict(value)
    for key in fields:
        if isinstance(result.get(key), list):
            result[key+"_count"] = len(result.pop(key))
    return result


def compact_trial(trial):
    result = {key: trial.get(key) for key in
              ("trial", "bundle", "selection", "proposal", "selection_agent", "candidate_agent",
               "measurement", "decision", "evaluation", "publication", "missing_coverage", "evaluation_attempt")}
    result["selection"] = counted_fields(result.get("selection"), ("affected_calls",))
    result["proposal"] = counted_fields(result.get("proposal"), ("eliminated_calls",))
    result["bundle"] = counted_fields(result.get("bundle"), ("affected_calls", "read_set", "write_set"))
    result["pairs"] = [{arm: compact_run(run) for arm, run in pair.items()} for pair in trial.get("pairs", [])]
    resources = trial.get("resources", {})
    result["resources"] = {key: resources.get(key) for key in ("status", "reason", "evidence", "raw_discovery")}
    result["resources"]["extra_collectors"] = {
        arm: {name: collector_summary(value) for name, value in values.items()}
        for arm, values in resources.get("extra_collectors", {}).items()}
    result["native_replacement_summary"] = []
    for path in resources.get("raw_discovery", []):
        source = Path(path)
        if not source.exists():
            continue
        raw(source)
        discovery = json.loads(source.read_text())
        bundle = trial.get("bundle", {}).get("id")
        kernels = [k for k in discovery.get("kernels", [])
                   if any(site["bundle"] == bundle for site in k.get("replacement_sites", []))]
        all_bundles=Counter()
        for kernel in discovery.get("kernels", []):
            all_bundles.update({site["bundle"] for site in kernel.get("replacement_sites", [])})
        result["native_replacement_summary"].append({
            "rank": discovery.get("rank"), "bundle": bundle, "raw_discovery": path,
            "launches": len(kernels), "kernel_names": dict(Counter(k["name"] for k in kernels)),
            "launches_by_bundle":dict(all_bundles),
            "scope": "Native launches associated with this bundle's replacement ranges, including any parent calls; separate diagnostic capture, not clean timing."})
    result["raw_trial_scope"] = "Full resource arrays, affected-call closures, eliminated-call lists, and read/write sets remain in the hashed original server receipt. Public counted fields summarize these lists; actual replacement sites, parameters, quality, measurements, and decision are retained."
    search = trial.get("parameter_search", {})
    result["parameter_search"] = {
        k: search.get(k) for k in ("engine", "measurement_scope", "parameter_trials", "space",
                                  "selected", "stats", "responses", "response_interpretation")}
    result["parameter_search"]["observations"] = [
        {"bundle": counted_fields(v.get("bundle"), ("affected_calls", "read_set", "write_set")),
         "revision": v.get("revision"), "result": v.get("result"),
         "constraints": v.get("constraints"), "run": compact_run(v.get("run", {})),
         "diagnostic": v.get("diagnostic"), "candidate_diagnostics": v.get("candidate_diagnostics")}
        for v in search.get("observations", [])]
    return result


for case in ("qwen", "resnet", "ddp"):
    folder = root/"stage2-binding-0566148"/case
    put("binding-"+case+".json", json.loads((folder/"results.json").read_text()))
    for path in folder.glob("*.json"):
        raw(path)
put("binding-summary.json", json.loads((root/"stage2-binding-0566148/summary.json").read_text()))
replica=root/"stage2-qwen-ttft-replication-0e4c751"
if (replica/"result.json").exists():
    for path in replica.glob("*.json"):
        raw(path)
        put(Path(replica.name)/path.name,json.loads(path.read_text()))
    for path in root.glob("stage2-qwen-replication-controller*"):
        raw(path)
        dest=out/replica.name/path.name
        dest.write_bytes(path.read_bytes())
    raw(root/"stage2_qwen_replicate.py")
for name in ("stage2-resnet-binding-recheck", "stage2-resnet-binding-final-21897c1"):
    path = root/name/("final218-backends" if "final-218" in name else "comparison")/"results.json"
    if path.exists():
        raw(path)
        put(name+".json", json.loads(path.read_text()))
    for path in (root/name).glob("*controller*"):
        raw(path)
        dest = out/name/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
for directory in ("stage2-binding-0566148", "stage2-feedback-0e4c751",
                  "stage2-autonomy-be51164", "stage2-final-6179733"):
    path = root/directory/"tests.json"
    if path.exists():
        raw(path)
        put(directory+"-tests.json", json.loads(path.read_text()))

cohorts = [*root.glob("stage2-autonomy-*"),
           *Path("/root/opop-v6-stage2").glob("stage2-autonomy-*")]
summary = []
for cohort in sorted(cohorts):
    if not (cohort/"controllers.json").exists():
        continue
    controllers = json.loads((cohort/"controllers.json").read_text())
    raw(cohort/"controllers.json")
    put(Path(cohort.name)/"controllers.json", controllers)
    if (cohort/"tests.json").exists():
        raw(cohort/"tests.json")
        put(Path(cohort.name)/"tests.json", json.loads((cohort/"tests.json").read_text()))
    for control in controllers:
        run = Path(control["output"])
        result = run/"autonomy-result.json"
        if not result.exists():
            continue
        data = json.loads(result.read_text())
        prefix = Path(cohort.name)/control["name"]
        put(prefix/"result.json", data)
        for path in (run/"request.json", run/"source-fidelity.json", run/"prepared.json",
                     run/"autonomy-protocol.json", run/"autonomy-progress.json"):
            if path.exists():
                raw(path)
                put(prefix/path.name, json.loads(path.read_text()))
        raw(result)
        if (run/"prepared.json").exists():
            prepared = json.loads((run/"prepared.json").read_text())
            for source in ("opop_adapter.py", "opop_evaluate.py"):
                source = Path(prepared["attempt"])/"reference-source"/source
                raw(source)
                if source.exists():
                    dest = out/prefix/"generated-adapter"/source.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(source.read_bytes())
        summary.append({"cohort": cohort.name, "case": control["name"], "status": data["status"],
                        "stage": data.get("stage"), "error": data.get("error"),
                        "source": control["source"], "rounds": data.get("rounds", [])})
        for path in run.glob("trials/*/result.json"):
            trial = json.loads(path.read_text())
            raw(path)
            diagnostic_files(trial.get("resources", {}), run)
            for observation in trial.get("parameter_search", {}).get("observations", []):
                diagnostic_files(observation.get("diagnostic", {}), run)
                diagnostic_files(observation.get("candidate_diagnostics", {}), run)
            put(prefix/"trials"/path.parent.name/"result.json", compact_trial(trial))
            agent = trial.get("candidate_agent") or {}
            if agent.get("sandbox"):
                inputs = Path(agent["sandbox"])/"inputs"
                for input_path in (inputs/"resources.json", inputs/"communication-ownership.json",
                             inputs/"previous-feedback.txt"):
                    raw(input_path)
                ownership = inputs/"communication-ownership.json"
                if ownership.exists():
                    put(prefix/"trials"/trial["trial"]/"communication-ownership.json",
                        json.loads(ownership.read_text()))
                resources = inputs/"resources.json"
                if resources.exists():
                    value = json.loads(resources.read_text())
                    put(prefix/"trials"/trial["trial"]/"observation-index.json", {
                        "source":str(resources), "extra_collectors":value.get("extra_collectors"),
                        "collection_plan":value.get("collection_plan"),
                        "scope":"Index extracted from exact candidate input; full collector payloads remain on box4-2."})
                for name in ("candidate.py", "proposal.json"):
                    source = Path(agent["sandbox"])/name
                    if source.exists():
                        raw(source)
                        dest = out/prefix/"trials"/trial["trial"]/name
                        dest.write_bytes(source.read_bytes())
        if (run/"discovery.json").exists():
            discovery = json.loads((run/"discovery.json").read_text())
            raw(run/"discovery.json")
            diagnostic_files(discovery.get("extra_collectors", {}), run)
            put(prefix/"discovery-summary.json", {
                k: discovery.get(k) for k in ("revision", "plan", "registry", "collection_plan", "profiled_wall_ms")
            } | {"groups": len(discovery["groups"]),
                 "collectors": {name: {"status": item.get("status"), "provenance": item.get("provenance")}
                                for name, item in discovery.get("extra_collectors", {}).items()}})
put("autonomy-summary.json", summary)
matrix = root/"stage2-outcome-matrix.json"
if matrix.exists():
    raw(matrix)
    put("outcome-matrix.json", json.loads(matrix.read_text()))
audit = root/"stage2-session-audit.json"
if audit.exists():
    raw(audit)
    put("agent-evidence-access.json", json.loads(audit.read_text()))
    data=json.loads(audit.read_text())
    roles=[role for run in data["runs"] for role in run["roles"]]
    all_sessions=[sid for role in roles for sid in role["sessions_including_children"]]
    assert len(all_sessions)==len(set(all_sessions)), "costs would duplicate sessions"
    put("usage.json", {"sessions":len(all_sessions), "roles":len(roles),
        "assistant_messages":sum(role["messages"] for role in roles),
        "reported_cost":sum(role["reported_cost"] for role in roles),
        "scope":"Provider-reported estimate for stage 2 autonomous roles and descendants, including failed attempts; excludes outer Codex and prior review sessions. No usage cap."})
put("raw-evidence-manifest.json", {"files": list(manifest.values()),
    "scope": "Completed stage 2 receipts; raw models/tensors/traces remain on box4-2. Source hashes distinguish frozen cohorts."})
archive = root/"stage2-evidence.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(out.rglob("*")):
        if path.is_file():
            bundle.write(path, path.relative_to(out).as_posix())
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                  "completed_runs": len(summary), "raw_files": len(manifest)}))
