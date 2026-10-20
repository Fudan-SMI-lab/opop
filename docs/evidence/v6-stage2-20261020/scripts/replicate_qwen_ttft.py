"""Independent unchanged-candidate measurement after all stage-2 Agents finish."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import traceback
from uuid import uuid4
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.runtime.execution import write_json

work=Path("/root/autodl-tmp/dev/v6-work")
out=work/"stage2-qwen-ttft-replication-0e4c751"
out.mkdir(exist_ok=False)
controls=[row for root in (work,Path("/root/opop-v6-stage2"))
          for p in root.glob("stage2-autonomy-*/controllers.json")
          for row in json.loads(p.read_text())]
write_json(out/"waiting.json", {"condition":"All 13 original autonomous controllers have final receipts",
    "runs":[row["output"] for row in controls]})
while any(not (Path(c["output"])/"autonomy-result.json").exists() for c in controls):
    time.sleep(15)
source=framework_source()
pairs=[]
try:
    run=work/"stage2-autonomy-0e4c751/qwen-ttft"
    search=ProjectSearch(load_config(work/"configs/box42-v6.yaml",[]),run)
    result=json.loads((run/"autonomy-result.json").read_text())
    assert source==result["protocol"]["source"]
    trial=result["rounds"][-1]
    assert trial["decision"]["accepted"]
    parent=search.revisions.revision(trial["decision"]["parent"])
    candidate=search.revisions.revision(trial["decision"]["revision"])
    accepted=search.revisions.current().id
    protocol={"source":source,"source_run":str(run),"source_trial":trial["trial"],
        "parent":parent.id,"candidate":candidate.id,"accepted_at_start":accepted,
        "started_at":datetime.now(timezone.utc).isoformat(),"request":search.request.portable(),
        "scope":"Developer-started independent AB/BA on unchanged accepted candidate, after autonomous roles finish; no generation, publication, or intervention in original runs"}
    write_json(out/"protocol.json",protocol)
    prefix="stage2-replication-"+uuid4().hex[:12]
    for index in range(2):
        pair={}
        order=[("parent",parent),("candidate",candidate)]
        for role,revision in order if index==0 else reversed(order):
            assert framework_source()==source
            measured=search.execute(revision,f"{prefix}-{index}-{role}","evaluate")
            write_json(out/f"pair-{index}-{role}.json",measured)
            pair[role]=measured
            if not valid_run(measured):
                raise ValueError("independent endpoint failed quality/coverage/execution")
        pairs.append(pair)
    decision=paired_decision(pairs,search.request.objective,search.prepared["evaluation"])
    assert search.revisions.current().id==accepted
    write_json(out/"result.json",{"status":"completed","protocol":protocol,"measurement":decision,
        "pairs":pairs,"accepted_pointer_unchanged":True,"framework_unchanged":framework_source()==source})
    print(json.dumps({"status":"completed","measurement":decision}),flush=True)
except Exception as exc:
    write_json(out/"result.json",{"status":"failed","error":str(exc),"traceback":traceback.format_exc(),
        "source":source,"pairs":pairs})
    raise
