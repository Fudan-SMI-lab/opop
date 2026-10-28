"""Fixed generated-policy validation and one real failed-feedback regression role."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import sha, read
from scripts.experiments.v6_accepted_feedback_replay_20261028 import OLD, NEW
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed(args, config):
    prior = read(args.replay/'followup-results.json')
    source = args.replay.parent/'fsdp-continuation-20261028/fsdp-search'
    branch = args.output/'fixed-validation'
    parent = fork_run(source, branch)
    search = ProjectSearch(config, branch)
    discovery = read(branch/'discovery.json')
    assert parent.id == '799a80d2b4d7810555f73386ec586109159692b93eaf28ddb8dbf6a932b79b1b'
    assert search.prepared['adapter_sha256'] == '8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6'
    base_files = {f.path:f.sha256 for f in parent.files}
    revisions = {'inherited-k3':parent}
    for arm in ['control','treatment']:
        directory=Path(prior[arm]['agent']['sandbox']);validate_proposal(directory)
        assert sha(directory/'candidate.py') == prior[arm]['candidate_sha256']
        metadata, prompt, inputs=load(Path(prior[arm]['agent']['generation_snapshots'][0]))
        assert metadata['framework_source']==framework_source()
        assert json.loads(inputs['inputs/project.json'])['current_revision']==parent.id
        selection=json.loads(inputs['inputs/selection.json'])
        bundle,_=search.bundle_from_outputs(directory,selection,parent,discovery,'fixed-'+arm)
        for depth in [0,1]:
            parameters={'keep_unsharded_blocks':3,'forward_prefetch_depth':depth}
            candidate=bundle.model_copy(update={'id':f'fixed-{arm}-d{depth}','parameters':parameters})
            revision=search.revisions.stage(candidate)
            assert revision.contract_ref==parent.contract_ref
            actual={f.path:f.sha256 for f in revision.files}
            assert all(actual.get(k)==v for k,v in base_files.items())
            revisions[f'{arm}-d{depth}']=revision
    order=['inherited-k3','control-d0','control-d1','treatment-d1','treatment-d0']
    identity={name:{'revision':r.id,'bundles':list(r.bundles)} for name,r in revisions.items()}
    write_json(args.output/'fixed-protocol.json',{'at':now(),'order':order,'revisions':identity,
        'parent':parent.id,'source_request_sha256':sha(source/'request.json'),
        'adapter_sha256':search.prepared['adapter_sha256'],'controls_path':search.prepared['controls_path'],
        'scope':'Developer-selected shared parameter controls, not autonomous selection or acceptance. All original two-update quality, source inputs, math, lifecycle, clean5 and separate3 diagnostic samples remain. No repeat until pass; timings are diagnostic only.'})
    results={}
    for name in order:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'fixed-'+name})
        try:
            run=search.execute(revisions[name],name,'evaluate')
            path=args.output/(name+'.json');write_json(path,run)
            valid=valid_run(run)
            constraints=constraints_for_runs([run],search.request.objective,search.prepared['evaluation']) if valid else []
            status='completed' if valid else 'worker_or_quality_failure'
            if valid and not all(c['passed'] for c in constraints):status='constraint_failure'
            results[name]={'status':status,
                'path':str(path),'sha256':sha(path),'valid_run':valid,'constraints':constraints,
                'identity':identity[name]}
        except Exception:
            results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'fixed-results.json',results)
        print(json.dumps({'endpoint':name,'status':results[name]['status'],'at':now()}),flush=True)
    native=[]
    for name in ['inherited-k3','control-d1','treatment-d1']:
        if not results[name].get('valid_run'):
            native.append({'name':name,'status':'not_run','reason':'Endpoint did not pass full quality/coverage.'});continue
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'native-'+name})
        try:
            observation=Nsys().collect(search,revisions[name]);path=args.output/('native-'+name+'.json');write_json(path,observation)
            native.append({'name':name,'status':observation['status'],'path':str(path),'sha256':sha(path)})
        except Exception:
            native.append({'name':name,'status':'error','traceback':traceback.format_exc()})
    assert search.revisions.current().id==parent.id
    write_json(args.output/'native-results.json',native)
    success=all(r['status']=='completed' for r in results.values()) and all(n['status']!='error' for n in native)
    return {'status':'completed' if success else 'completed_with_errors',
            'endpoints':results,'native':native,'publication':'none; no score comparison or formal acceptance'}


def failed_feedback(args, config):
    original_trial=read(args.replay.parent/'fsdp-search-20261028/fsdp-search/trials/trial-4f5fce6e6b/result.json')
    snapshot=Path(original_trial['candidate_agent']['generation_snapshots'][0])
    saved,prompt,inputs=load(snapshot)
    feedback=json.loads(inputs['inputs/previous-feedback.txt'])
    assert not feedback['decision']['accepted'] and saved['fresh_session'] and saved['role']=='candidate'
    assert saved['framework_source']==framework_source() and prompt.count(OLD)==1
    write_json(args.output/'failed-role-protocol.json',{'at':now(),'snapshot':str(snapshot),
        'original_core':saved['framework_source'],'change':{'old':OLD,'new':NEW},
        'previous_decision':feedback['decision'],
        'scope':'One new real role on an already captured quality-failure input, only the same neutral sentence treatment. No original output, later diagnosis or new evidence is supplied. No GPU validation/performance claim for this separately generated output.'})
    store=RunStore.create(args.output,'failed-feedback-role',{'kind':'neutral-feedback-regression'})
    with Runtime(project_agent_config(config),log_dir=store.run_dir) as runtime:
        directory,agent=ProjectAgents(runtime,store).replay_snapshot(snapshot,
            edit={'path':'prompt','old':OLD,'new':NEW},validator=validate_proposal)
    actual,ap,af=load(Path(agent['generation_snapshots'][0]))
    assert af==inputs and ap==prompt.replace(OLD,NEW)
    assert actual['tool_model_settings']==saved['tool_model_settings']
    return {'status':'completed','agent':agent,'initial_inputs_equal':True,
        'candidate_sha256':sha(directory/'candidate.py'),'proposal':read(directory/'proposal.json'),
        'scope':'Fresh role generation and schema only. Does not retry old GPU endpoints or establish full quality/acceptance.'}


def main(args):
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    config=load_config(args.config,[]);config.usage.mode='observe_only'
    audit=read(args.replay/'terminal-audit.json')
    assert audit['core']==framework_source()
    references=read(args.replay.parent/'fsdp-continuation-followup-20261028/terminal-audit.json')['source_fidelity']['unchanged_references']
    for reference in references:assert sha(reference['path'])==reference['sha256']
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'source_replay':str(args.replay),
        'verified_source_references':references,
        'sequence':['five fixed full-workload endpoints and separate native paths','one fresh neutral feedback role on a historical quality failure'],
        'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'No production adoption, input expansion, sampler change, acceptance or old-result rewrite. Shared source references; raw artifacts server-side. Real API use observed only.'})
    results={}
    for name,operation in [('fixed-gpu',fixed),('failed-feedback-role',failed_feedback)]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        try:results[name]=operation(args,config)
        except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
    terminal={'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','output','replay']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
