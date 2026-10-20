from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/opop-v6-stage3');root=Path('/root/autodl-tmp/dev/v6-work/stage3-variants');origin=root/'ddp-step-cross-goal';output=root/'ddp-communication-independent';prepared=json.loads((origin/'prepared.json').read_text());fork_run(origin,output,revision_id=prepared['revision']);result=promote_version(ProjectSearch(load_config(r/'deployment.yaml',[]),output),origin,'c1ddfa6fb146d30eb646f8b49a2a5ae48eadd7741db0a3977cf44c654b48cdff',publish=False);record={'scope':'Developer independent complete-version AB/BA repeat of the frozen DDP compute candidate under the communication objective; no new generation or parameter changes','reason':'First accepted replay had paired communication gains 0.911426 and 19.027553 ms; repeat checks whether that average is stable','trial':result['trial'],'decision':result['decision'],'measurement':result['measurement']};write_json(root/'ddp-communication-independent-result.json',record);print(record,flush=True)
