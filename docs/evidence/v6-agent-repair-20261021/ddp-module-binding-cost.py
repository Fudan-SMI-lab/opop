from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.binding_cost import BindingCost
from kernel_optimizer.models.project import CandidateBundle
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
source=root/'ddp-communication/0-treatment-recovered'
bundle=CandidateBundle.model_validate_json((source/'project/bundles/trial-02e8934b4f-param-540ac4ad.json').read_bytes())
out=root/'ddp-module-binding-cost'
fork_run(source,out,revision_id=bundle.parent_revision)
search=ProjectSearch(load_config(root/'ddp-communication-recovery-deployment.yaml',[]),out)
bundle=bundle.model_copy(update={'id':'ddp-module-cost-diagnostic'})
revision=search.revisions.stage(bundle)
result=BindingCost().collect(search,revision)
write_json(out/'result.json',{'scope':'Developer-started diagnostic of actual Agent-chosen module boundary; same parent/quality/measurement as original functional comparison, no optimization publication','binding_cost':result})
print(json.dumps({'status':result['status'],'pairs':result.get('elapsed_pairs')}),flush=True)
