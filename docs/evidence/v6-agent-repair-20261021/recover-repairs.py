"""Resume the frozen experiment after disk exhaustion, retaining candidate bytes."""
import hashlib,json,sys,shutil,time
from pathlib import Path
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.wiring import Runtime

root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
case=sys.argv[1]
base=Path('/root/opop-v6-stage3/final-e2e-187617c')/case
original_trial={'qwen-ttft':'trial-af78dd3283','qwen-throughput':'trial-87dde5b43d',
                'ddp-step':'trial-b0a02885e2','ddp-communication':'trial-d17bd1eb8f'}[case]
old=json.loads((base/'trials'/original_trial/'generated.json').read_text())
config=project_agent_config(load_config(root/(case+'-recovery-deployment.yaml'),[]))
config.usage.mode='observe_only'
old_control=root/case/'0-control'
control_generated=json.loads(next((old_control/'trials').glob('*/generated.json')).read_text())
rows=[]
if case=='ddp-communication':
    rows=json.loads((root/case/'result.json').read_text())['runs']
for arm in ['treatment','control']:
    if any(r['arm']==arm and 'decision' in r for r in rows):continue
    branch=root/case/('0-'+arm+'-recovered')
    parent=fork_run(base,branch,revision_id=old['bundle']['parent_revision'])
    search=ProjectSearch(config,branch)
    evaluate=search._evaluate_generated
    def after_storage_recovery(*args,**kwargs):
        while shutil.disk_usage(root).free<10*1024**3:
            time.sleep(5)
        return evaluate(*args,**kwargs)
    search._evaluate_generated=after_storage_recovery
    try:
        if case=='ddp-communication':
            feedback=json.loads((base/'trials'/original_trial/'result.json').read_text())
            feedback['supplemental_evidence']=json.loads((root/(case+'-evidence.json')).read_text())
            result=search.trial(feedback=json.dumps(feedback))
        else:
            if arm=='control':
                generated=Path(control_generated['candidate_agent']['sandbox'])
                receipt=control_generated['candidate_agent']
            else:
                snapshot=Path(control_generated['candidate_agent']['generation_snapshots'][0])
                _,_,files=load(snapshot)
                previous=files['inputs/previous-feedback.txt'].decode()
                feedback=json.loads(previous)
                feedback['supplemental_evidence']=json.loads((root/(case+'-evidence.json')).read_text())
                edit={'path':'inputs/previous-feedback.txt','old':previous,'new':json.dumps(feedback,indent=2)}
                with Runtime(config,log_dir=branch) as runtime:
                    generated,receipt=ProjectAgents(runtime,search.store).replay_snapshot(snapshot,edit=edit,validator=validate_proposal)
            validate_proposal(generated)
            discovery=search.current_discovery()
            trial='recovered-'+arm
            (branch/'trials'/trial).mkdir(parents=True)
            bundle,proposal=search.bundle_from_outputs(generated,old['selection'],parent,discovery,trial)
            metadata={'trial':trial,'selection':old['selection'],'proposal':proposal,'candidate_agent':receipt,
                'arm':arm,'recovery':{'cause':'ENOSPC interrupted original experiment',
                    'original_control':str(old_control),'reused_generation':arm=='control',
                    'candidate_sha256':hashlib.sha256((generated/'candidate.py').read_bytes()).hexdigest(),
                    'scope':'All interrupted GPU measurements excluded from this decision; full original validation repeated'}}
            write_json(branch/'trials'/trial/'generated.json',{**metadata,'bundle':bundle.model_dump(mode='json')})
            result=search._evaluate_generated(bundle,metadata,discovery)
        row={'repeat':0,'arm':arm,'output':str(branch),'trial':result['trial'],'decision':result['decision'],
             'recovered_after_disk_exhaustion':True}
    except Exception as exc:
        import traceback
        row={'repeat':0,'arm':arm,'output':str(branch),'error':f'{type(exc).__name__}: {exc}',
             'traceback':traceback.format_exc(),'recovered_after_disk_exhaustion':True}
    rows.append(row)
    protocol=json.loads((root/case/'protocol.json').read_text())
    write_json(root/case/'result.json',{'protocol':protocol,'runs':rows})
    print(json.dumps(row),flush=True)
write_json(root/case/'completed.json',{'completed':True,'runs':len(rows),'recovered':True})
