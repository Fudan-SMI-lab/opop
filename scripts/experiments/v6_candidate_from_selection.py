"""Run one ordinary candidate role from a completed selection and original evidence.

Experiment orchestration only: no case-specific optimizer or production modification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
from datetime import datetime, timezone

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.runtime_config import project_agent_config, worker_environment
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_observation_mechanics_repair_20261031 import expected_prompt
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

def now():return datetime.now(timezone.utc).isoformat()
def digest(v):return hashlib.sha256(v.encode()if isinstance(v,str)else v).hexdigest()

def prepare(args):
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    previous=read(args.selection_run/'followup-completed.json');p=read(args.selection_run/'protocol.json')
    assert previous['phase']=='completed'and(args.selection_run/'terminal-audit.json').exists()
    result=previous['results']['selection'];meta,_,files=load(Path(result['initial_snapshot']))
    selection_path=Path(result['agent']['sandbox'])/'selection.json';assert sha(selection_path)==result['selection_sha256']
    search=ProjectSearch(config,Path(p['source']));current=search.revisions.current();assert current.id==p['current']
    assert framework_source()==p['core']==meta['framework_source']
    for f in current.files:assert files['inputs/source/'+f.path]==search.store.get_artifact(f.content_ref)
    discovery=search.current_discovery();selection=read(selection_path);affected=set(selection['affected_calls'])
    while True:
        closure=affected|{c['id']for c in discovery['calls']if set(c['parents'])&affected}
        if closure==affected:break
        affected=closure
    selection['affected_calls']=sorted(affected)
    lookup={c['id']:c for c in discovery['calls']}
    inputs={n.removeprefix('inputs/'):v for n,v in files.items()if n not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    inputs.update({'selection.json':json.dumps(selection,indent=2),
        'targets.json':json.dumps([lookup[c]for c in sorted(affected)],indent=2),
        'target-resources.json':json.dumps({c:discovery.get('call_resources',{}).get(c)for c in sorted(affected)},indent=2)})
    evidence_terminal=read(args.evidence_run/'followup-completed.json')
    assert evidence_terminal['phase']=='completed'and(args.evidence_run/'terminal-audit.json').exists()
    evidence=evidence_terminal['results']['diagnostic-author'];plan=Path(evidence['plan'])
    assert sha(plan)==evidence['plan_sha256']
    constraints=args.user_constraints.read_text()
    index={'scope':'Original generated analysis plan, not a verified causal answer, optimization instruction or certified collector. Its claims/inferences remain hypotheses to reconcile with original artifacts. No generated artifact-reader code, developer terminal audit/correction table or manual optimization supplied. Latest user constraints are explicit user steering, not an unaided/single-variable cohort.',
        'plan':{'input':'inputs/diagnostic-evidence/plan.json','path':str(plan),'sha256':sha(plan),'bytes':plan.stat().st_size},
        'original_artifact_index':{'path':str(args.evidence_run/'evidence-index.json'),'sha256':sha(args.evidence_run/'evidence-index.json')},
        'user_constraints':'inputs/user-constraints.txt'}
    inputs.update({'diagnostic-evidence/plan.json':plan.read_bytes(),'diagnostic-evidence/index.json':json.dumps(index,indent=2),'user-constraints.txt':constraints})
    raw=(args.selection_run/'original-feedback.json').read_bytes();assert raw==(search.output/'latest-feedback.json').read_bytes()
    original=json.loads(raw);augmented=dict(original)
    existing=original.get('supplemental_evidence')
    augmented['supplemental_evidence']={'prior':existing,'diagnostic_artifacts':index,'user_constraints':constraints}
    augmented_text=json.dumps(augmented,indent=2)
    packaged,compact=feedback_artifacts(augmented_text)
    for n,v in packaged.items():
        b=v.encode()if isinstance(v,str)else v
        if n in inputs:assert inputs[n]==b
        else:inputs[n]=b
    feedback=route(compact,current_revision=current.id,registry=search.revisions.load(current.registry_ref,GroupRegistry),
                   selected_groups=selection['groups'],plan=search.revisions.load(current.plan_ref,ExecutionPlan))
    assert json.loads(feedback)['decision']==original['decision']
    expected={'inputs/'+n:v.encode()if isinstance(v,str)else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode()
    expected['inputs/worker-environment.json']=json.dumps(worker_environment(),indent=2).encode()
    # Preserve current task/source and actual supplied evidence. Historical audits are
    # references, not another repeated traversal of every closed run or its tensors.
    paths=[args.selection_run/n for n in ['protocol.json','followup-completed.json','terminal-audit.json','original-feedback.json']]
    paths += [selection_path,Path(result['initial_snapshot']),plan,args.evidence_run/'evidence-index.json',args.evidence_run/'terminal-audit.json',args.user_constraints]
    paths += [search.output/n for n in ['request.json','prepared.json','discovery.json','project/current.json','latest-feedback.json']]
    frozen={str(path):sha(path)for path in paths}
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=8*1024**3+384*1024**2
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    from kernel_optimizer.project import agents
    body=Path(agents.__file__).with_name('candidate_prompt.md').read_text()
    full_prompt=expected_prompt(body+'\nRead inputs/previous-feedback.txt and repair the concrete failure.',json.loads(expected['inputs/worker-environment.json']))
    full_prompt+=' Read inputs/binding-contract.md before choosing or composing replacements and capture policies.'
    details={'source':str(search.output),'parent':current.id,'core':framework_source(),'source_bundles':current.bundles,
        'selection_snapshot':result['initial_snapshot'],'selection_sha256':sha(selection_path),'canonical_selection_sha256':digest(inputs['selection.json']),
        'affected_calls':len(affected),'groups':selection['groups'],'scope_selected':selection['scope'],
        'original_feedback_sha256':digest(raw),'original_feedback_bytes':len(raw),'feedback_sha256':digest(feedback),'feedback_bytes':len(feedback.encode()),
        'evidence_plan_sha256':sha(plan),'user_constraints_sha256':digest(constraints),
        'inputs':len(expected),'input_bytes':sum(map(len,expected.values())),'input_hashes':{n:digest(v)for n,v in expected.items()},
        'critical_frozen_files':len(frozen),'historical_preservation_reference':str(args.evidence_run/'terminal-audit.json'),
        'prompt_sha256':digest(full_prompt),'free_bytes':free,'growth_forecast_bytes':384*1024**2,'warning_free_bytes':8*1024**3,
        'scope':'One ordinary candidate generation from original selection/current source and newly available original analysis plan plus latest user genericity/progress requirements. No reader repair, selector retry, manual algorithm/group/code or production prompt/model/tools change. No GPU/search/parameter/collector/sourcecontrol/referencecopy/publication stage. Schema success is not numerical or performance acceptance.'}
    return config,search,inputs,feedback,expected,full_prompt,frozen,index,details,augmented_text,raw

def main(args):
    os.umask(0o077)
    config,search,inputs,feedback,expected,prompt,frozen,index,details,augmented,raw=prepare(args)
    if args.preflight:print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen})
    write_json(args.output/'preparation-reuse.json',{'source':details['source'],'current':details['parent'],'scope':details['scope']})
    write_json(args.output/'evidence-index.json',index)
    (args.output/'original-feedback.json').write_bytes(raw);(args.output/'diagnostic-feedback.json').write_text(augmented)
    (args.output/'routed-feedback.json').write_text(feedback)
    write_json(args.output/'state.json',{'phase':'running','stage':'candidate-only','at':now()});results={}
    try:
        from kernel_optimizer.project import agents
        body=Path(agents.__file__).with_name('candidate_prompt.md').read_text()
        store=RunStore.create(args.output,'candidate',{'kind':'candidate-from-existing-selection'})
        with Runtime(config,log_dir=store.run_dir)as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('candidate',inputs,body,('candidate.py','proposal.json'),feedback,validator=validate_proposal)
        initial=[]
        for path in(store.run_dir/'generation-snapshots').glob('*.json'):
            m,actual_prompt,actual=load(path)
            if m['fresh_session']:
                assert actual==expected and actual_prompt==prompt and m['framework_source']==details['core']
                initial.append(str(path))
        assert len(initial)==1
        results['candidate']={'status':'completed','agent':receipt,'initial_snapshot':initial[0],
            'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
            'parameter_space':read(directory/'proposal.json').get('parameter_space'),'inputs_verified':True,
            'scope':'Generation/schema only; inspect genericity, dependencies and necessary correctness before original GPU validation. No automatic parameter alternatives or repair chain.'}
    except Exception:results['candidate']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(path)==h for path,h in frozen.items())and search.revisions.current().id==details['parent']
    results['preservation']={'status':'passed'if same else'failed','critical_files':len(frozen),'unchanged':same,'scope':'Current source/task/supplied evidence; closed historical batches referenced, not redundantly revalidated.'}
    terminal={'phase':'completed'if same and results['candidate']['status']=='completed'else'completed_with_errors','finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    for name,value in [('followup-results.json',results),('state.json',terminal),('followup-completed.json',terminal)]:write_json(args.output/name,value)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','selection-run','evidence-run','user-constraints','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');main(parser.parse_args())
