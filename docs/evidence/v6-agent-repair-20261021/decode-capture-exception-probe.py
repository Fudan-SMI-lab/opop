from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=root/'qwen-throughput/0-treatment-recovered'
saved=json.loads((source/'trials/recovered-treatment/parameter-search.json').read_text())
bundle=CandidateBundle.model_validate_json(json.dumps(saved['observations'][0]['bundle']))
out=root/'decode-capture-exception-probe'
fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(root/'qwen-throughput-recovery-deployment.yaml',[]),out)
edits=[]
for edit in bundle.edits:
 if edit.path==bundle.installer:
  code=search.store.get_artifact(edit.content_ref).decode()
  old='except Exception:\n                        _disable_graphs()'
  new='except Exception as diagnostic_exc:\n                        import traceback\n                        print("DEVELOPER_CAPTURE_EXCEPTION", type(diagnostic_exc).__name__, str(diagnostic_exc), flush=True)\n                        traceback.print_exc()\n                        _disable_graphs()'
  assert code.count(old)==1
  code=code.replace(old,new)
  edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
  (out/'instrumented-candidate.py').write_text(code)
 edits.append(edit)
bundle=bundle.model_copy(update={'id':'decode-exception-diagnostic','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'capture-exception','evaluate')
write_json(out/'result.json',{'scope':'Developer diagnostic: only print the previously swallowed capture exception; no semantic repair or publication; timing is not an optimization result','run':run})
print(run['status'],flush=True)
