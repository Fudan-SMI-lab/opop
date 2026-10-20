from pathlib import Path
import json
from datetime import datetime, timezone
from kernel_optimizer.store.run_store import RunStore

roots=[Path("/root/autodl-tmp/dev/v6-work"),Path("/root/opop-v6-stage2")]
rows=[]
for root in roots:
    for p in sorted(root.glob("stage2-autonomy-*/controllers.json")):
        for c in json.loads(p.read_text()):
            run=Path(c["output"]); result=run/"autonomy-result.json"
            if not result.exists():
                continue
            d=json.loads(result.read_text())
            request=json.loads((run/"request.json").read_text())
            fidelity=run/"source-fidelity.json"
            item={"cohort":p.parent.name,"case":c["name"],"status":d["status"],
                  "framework_unchanged":d.get("framework_unchanged"),
                  "source":c["source"],"metric":request["objective"],
                  "stage":d.get("stage"),"error":d.get("error"),
                  "source_fidelity":json.loads(fidelity.read_text()) if fidelity.exists() else None,
                  "rounds":[],"accepted_revision":d.get("accepted_revision")}
            for r in d.get("rounds",[]):
                row={k:r.get(k) for k in ("round","trial","decision","measurement","revision","error")}
                if r.get("trial"):
                    t=json.loads((run/"trials"/r["trial"]/"result.json").read_text())
                    row.update(scope=t["selection"]["scope"],
                        selected_group_count=len(t["selection"]["groups"]),
                        replacement_call_count=len(t["proposal"]["replacement_calls"]),
                        implementation=t["proposal"]["implementation"],
                        selected_parameters=t.get("bundle",{}).get("parameters"),
                        paired_endpoint_count=sum(len(pair) for pair in t.get("pairs",[])),
                        publication=t.get("publication"),
                        collectors={arm:{name:value.get("status") for name,value in obs.items()}
                                    for arm,obs in t.get("resources",{}).get("extra_collectors",{}).items()})
                    row["screening"]=[]
                    for obs in t.get("parameter_search",{}).get("observations",[]):
                        row["screening"].append({
                            "parameters":obs["bundle"].get("parameters"),
                            "valid":obs["result"].get("valid"),"score":obs["result"].get("score"),
                            "constraints":obs.get("constraints"),
                            "rank_checks":[{"rank":rank["rank"],"worker_status":rank.get("status"),
                                "error":rank.get("error"),"checks":rank.get("result",{}).get("checks")}
                                for rank in obs.get("run",{}).get("ranks",[])]})
                item["rounds"].append(row)
            rows.append(item)
out=roots[0]/"stage2-outcome-matrix.json"
out.write_text(json.dumps({"captured_at":datetime.now(timezone.utc).isoformat(),"runs":rows},indent=2))
print(json.dumps({"completed_receipts":len(rows),"rounds":sum(len(r["rounds"]) for r in rows),
                  "accepted":sum(bool(x.get("decision") and x["decision"].get("accepted")) for r in rows for x in r["rounds"]),
                  "path":str(out)}))
