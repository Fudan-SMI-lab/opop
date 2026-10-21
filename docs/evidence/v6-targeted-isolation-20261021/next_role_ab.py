from pathlib import Path
import ast,hashlib,json,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.request import ProjectUnderstanding
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
experiments={
 'ddp-contract':{'case':'ddp-communication','snapshot':'generation-47e75b19038a.json',
   'addition':"\nThe baseline adapter must reproduce the source implementation. In understanding.invariants, distinguish required task semantics from source implementation choices: preserve request constraints and source-defined correctness, but do not freeze a choice solely because it appears in the reference. Describe such choices in assumptions; for a genuine invariant, cite the request/source requirement that makes it necessary.\n"},
 'qwen-planner':{'case':'qwen-ttft','snapshot':'generation-d0eef7e5923b.json',
   'addition':"\nA binding summary's 'no functional replacements' describes only functional dispatch. Module and project callable replacements can still be installed via direct binding. Determine installed sites from replacement calls and setup/coverage; absence of a functional mode alone is not evidence that no candidate ran.\n"}}
def run(name,arm,offset):
    spec=experiments[name];out=root/(name+'-ab')/arm
    store=RunStore.create(out.parent,out.name,{'kind':'frozen_role_prompt_comparison','experiment':name,'arm':arm})
    snapshot=old/'fresh'/spec['case']/'generation-snapshots'/spec['snapshot']
    manifest,prompt,files=load(snapshot)
    protocol={'snapshot':str(snapshot),'arm':arm,'source_framework':manifest['framework_source'],
       'input_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()},
       'treatment':spec['addition'],'scope':'Developer frozen-role prompt comparison; fresh real sessions, original declared input bytes unchanged, no production prompt changes. Generation only; no optimization claim.'}
    write_json(out/'protocol.json',protocol)
    config=project_agent_config(load_config(old/(spec['case']+'-fresh-private.yaml'),[]))
    config.opencode.port=16820+offset;config.usage.mode='observe_only'
    def validate(directory):
        if name=='qwen-planner':CollectionPlan.model_validate_json((directory/'collection.json').read_bytes())
        else:
            u=ProjectUnderstanding.model_validate_json((directory/'understanding.json').read_bytes())
            request=json.loads(files['inputs/request.json'])
            for field in ['objective','workload','quality']:assert getattr(u,field)==request[field],field
            ast.parse((directory/'adapter.py').read_text())
    try:
        with Runtime(config,log_dir=out) as runtime:
            directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,
               edit={'path':'prompt','old':prompt,'new':prompt+spec['addition']} if arm=='treatment' else None,validator=validate)
        outputs={n:json.loads((directory/n).read_text()) for n in manifest['outputs'] if n.endswith('.json')}
        result={'experiment':name,'arm':arm,'status':'generated','receipt':receipt,'outputs':outputs}
    except Exception as e:
        result={'experiment':name,'arm':arm,'status':'error','error':str(e),'traceback':traceback.format_exc()}
    write_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','receipt']}),flush=True)
    return result
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    tasks=[pool.submit(run,name,arm,offset) for offset,(name,arm) in enumerate(
        (('ddp-contract','control'),('ddp-contract','treatment'),('qwen-planner','control'),('qwen-planner','treatment')))]
    for task in as_completed(tasks):
        rows.append(task.result());write_json(root/'role-ab-result.json',{'runs':rows})
write_json(root/'role-ab-completed.json',{'completed':True,'runs':len(rows)})
