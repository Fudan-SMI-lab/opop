from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=root/'fresh-final/qwen-ttft'
saved=json.loads((source/'trials/trial-e262267559/parameter-search.json').read_text())
bundle=CandidateBundle.model_validate_json(json.dumps(saved['observations'][0]['bundle']))
out=root/'fresh-qwen-guard-probe'
fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(root/'qwen-ttft-fresh-final-private.yaml',[]),out)
edits=[]
for edit in bundle.edits:
 if edit.path==bundle.installer:
  code=search.store.get_artifact(edit.content_ref).decode()
  old='return bool(torch.all(diff <= tol).item())'
  new='print("DEVELOPER_GUARD_MEASUREMENT", {"shape": list(a.shape), "max_abs_error": float(diff.max().item()), "violations_internal_001": int((diff > tol).sum().item()), "violations_reference_002": int((diff > (0.02 + 0.02 * rf.abs())).sum().item())}, flush=True)\n        return bool(torch.all(diff <= tol).item())'
  assert code.count(old)==1
  code=code.replace(old,new)
  edit=edit.model_copy(update={'content_ref':search.store.put_artifact(code)})
  (out/'instrumented-candidate.py').write_text(code)
 edits.append(edit)
bundle=bundle.model_copy(update={'id':'fresh-qwen-guard-diagnostic','edits':tuple(edits)})
revision=search.revisions.stage(bundle)
run=search.execute(revision,'guard-measurement','evaluate')
lines=[]
for p in (out/'jobs/guard-measurement/run').glob('rank-*.log'):
 lines.extend(line for line in p.read_text().splitlines() if 'DEVELOPER_GUARD_MEASUREMENT' in line)
write_json(out/'result.json',{'scope':'Developer readout of the existing candidate comparison only. Same arithmetic, same return and fallback policy, no tolerance change, no publication. This does not validate later checks skipped by the original early return; timings are intrusive diagnostics only.','run':run,'guard_measurements':lines})
print(json.dumps(lines),flush=True)
