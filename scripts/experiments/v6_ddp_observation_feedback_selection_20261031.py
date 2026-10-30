"""One ordinary project selector with exact newly acquired observation artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.experiment_policy import check_selection, eligible_groups
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()


def prepare(args):
    work=args.parent.parent
    observation=work/'ddp-communication-observation-20261031'
    audit=read(observation/'terminal-audit.json')
    assert audit['all_frozen_unchanged'] and audit['all_workers_valid_math_unchanged']
    assert audit['own_raw_traces_recomputed']==12 and audit['core']==framework_source()
    previous=work/'ddp-communication-redecision-20261031'
    prior=read(previous/'followup-completed.json')['results']['selection']
    saved,prompt,files=load(Path(prior['initial_snapshot']))
    inputs={name.removeprefix('inputs/'):data for name,data in files.items() if name not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    oldfeedback=json.loads(files['inputs/previous-feedback.txt'])
    protocol=read(observation/'protocol.json');frozen=protocol['frozen_identities'].copy()
    def item(path):
        path=Path(path);frozen[str(path)]=sha(path)
        return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    index={'scope':'One newly acquired fixed B-observed/C-observed on exact accepted implementation, original evaluator and Agent-authored observer. These are observer-perturbed diagnostics, not a search/independent acceptance or an overhead estimate. Original source/quality and existing assistance boundaries remain. Unknown and unavailable fields are original recorded evidence; no causal solution, group, algorithm, implementation or expected gain supplied. Prior77ac implementation-repair and diagnostic-mechanics repair branches are closed; this is an ordinary project decision, not a request to edit or rerun those implementations.',
        'probe':item(protocol['probe']),'plan':item(protocol['plan']),'endpoints':{},
        'independent_reference':None,'limits':'CPU callbacks are not GPU readiness; raw-trace kernel durations include waiting, not wire time/critical path. Separate captures and clean periods cannot be subtracted. Original Agent plan/code claims remain hypotheses, not developer-certified facts.'}
    inputs['observation-evidence/diagnostic.py']=Path(protocol['probe']).read_bytes()
    inputs['observation-evidence/diagnostic-plan.json']=Path(protocol['plan']).read_bytes()
    for arm,name in [('parent','baseline-observed'),('candidate','candidate-observed')]:
        endpoint={'original_request':item(protocol['source_requests'][arm]),'observed_request':item(observation/name/'request.json'),'ranks':{}}
        for label,path in [('original-request.json',Path(protocol['source_requests'][arm])),('observed-request.json',observation/name/'request.json')]:
            inputs[f'observation-evidence/{arm}/{label}']=path.read_bytes()
        for rank in range(2):
            worker=observation/name/f'rank-{rank}.json';diagnostic=observation/name/f'diagnostic-rank-{rank}.json'
            endpoint['ranks'][str(rank)]={'worker_result':item(worker),'diagnostic':item(diagnostic),
                'raw_traces':[item(p) for p in sorted((observation/name/'secondary-metrics').glob(f'*/metric-rank-{rank}-*.json'))]}
            inputs[f'observation-evidence/{arm}/rank-{rank}.json']=worker.read_bytes()
            inputs[f'observation-evidence/{arm}/diagnostic-rank-{rank}.json']=diagnostic.read_bytes()
        index['endpoints'][arm]=endpoint
    inputs['observation-evidence/index.json']=json.dumps(index,indent=2)
    raw=(args.parent/'ddp-search/latest-feedback.json').read_bytes()
    assert digest(raw)=='a0234920605970500c870b2ec98fc952b264a79f264bd19d6a53c99b796cf459'
    original=json.loads(raw)
    assert 'supplemental_evidence' not in original
    supplement=dict(oldfeedback['supplemental_evidence'])
    supplement['diagnostic_artifacts']={'artifact_index':'inputs/observation-evidence/index.json',
        'sha256':digest(inputs['observation-evidence/index.json'].encode()),'scope':index['scope'],
        'limits':index['limits']}
    augmented={**original,'supplemental_evidence':supplement}
    _,compact=feedback_artifacts(json.dumps(augmented))
    source=ProjectSearch(load_config(args.parent/'private-config.yaml',[]),args.parent/'ddp-search')
    current=source.revisions.current()
    assert current.id=='1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8'
    feedback=route(compact,current_revision=current.id,registry=source.revisions.load(current.registry_ref,GroupRegistry))
    routed=json.loads(feedback)
    assert {k:v for k,v in routed.items() if k!='supplemental_evidence'}=={k:v for k,v in oldfeedback.items() if k!='supplemental_evidence'}
    assert {k:v for k,v in routed['supplemental_evidence'].items() if k!='diagnostic_artifacts'}==oldfeedback['supplemental_evidence']
    for path in [observation/'terminal-audit.json',observation/'public-terminal-audit.json',observation/'protocol.json',observation/'followup-completed.json',observation/'request-identities.json',
        previous/'terminal-audit.json',previous/'selection/events.jsonl',previous/'followup-completed.json',Path(prior['initial_snapshot']),args.parent/'ddp-search/project/current.json']:
        item(path)
    assert all(sha(path)==h for path,h in frozen.items())
    return inputs,feedback,saved,prompt,files,frozen,prior['initial_snapshot'],index


def main(args):
    os.umask(0o077)
    inputs,feedback,saved,prompt,oldfiles,frozen,snapshot,index=prepare(args)
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    assert config.agents.default_model==saved['model'] and config.opencode.agent==saved['agent']
    expected={'inputs/'+n:v.encode() if isinstance(v,str) else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode();expected['inputs/worker-environment.json']=oldfiles['inputs/worker-environment.json']
    details={'core':framework_source(),'frozen_files':len(frozen),'original_snapshot':snapshot,
        'inputs':len(expected),'input_bytes':sum(map(len,expected.values())),
        'raw_feedback_sha256':sha(args.parent/'ddp-search/latest-feedback.json'),'raw_feedback_bytes':(args.parent/'ddp-search/latest-feedback.json').stat().st_size,
        'feedback_sha256':digest(feedback.encode()),'feedback_bytes':len(feedback.encode()),
        'original_feedback_fields_and_existing_supplement_preserved':True,
        'changed_inputs':sorted(k for k in oldfiles.keys()&expected.keys() if oldfiles[k]!=expected[k]),
        'added_inputs':sorted(expected.keys()-oldfiles.keys()),'removed_inputs':sorted(oldfiles.keys()-expected.keys()),
        'parent':'1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8',
        'source':str(args.parent/'ddp-search'),'growth_forecast_bytes':256*1024**2,'warning_free_bytes':8*1024**3,
        'scope':'One ordinary selection from unchanged accepted current code/discovery with original feedback plus newly acquired original observation artifacts. No terminal audit/causal table/manual group or implementation supplied. Prior CPU/source/mapping assistance preserved; not unaided or single-variableAB. No candidate/GPU/collector/search/acceptance/pointer stage; no diagnostic-repair/77ac/Qwen/FSDP restart.'}
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1]);assert free>=8*1024**3+details['growth_forecast_bytes']
    details['free_bytes']=free
    if args.preflight:print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen})
    write_json(args.output/'transport-preflight.json',details);write_json(args.output/'evidence-index.json',index)
    (args.output/'original-feedback.json').write_bytes((args.parent/'ddp-search/latest-feedback.json').read_bytes())
    (args.output/'routed-feedback.json').write_text(feedback)
    write_json(args.output/'preparation-reuse.json',{'source':details['source'],'parent':details['parent'],
        'request_sha256':sha(args.parent/'ddp-search/request.json'),'prepared_sha256':sha(args.parent/'ddp-search/prepared.json'),
        'discovery_sha256':sha(args.parent/'ddp-search/discovery.json'),'scope':'Read-only current accepted code/discovery/control reuse; no source/model/control/reference copy or pointer mutation.'})
    write_json(args.output/'state.json',{'phase':'running','stage':'selection-only','at':now()});results={}
    try:
        store=RunStore.create(args.output,'selection',{'kind':'ddp-observation-feedback-selection'})
        source=ProjectSearch(config,args.parent/'ddp-search');discovery=source.current_discovery()
        policy=json.loads(inputs['search.json']);eligible=eligible_groups(discovery,leaf_only=False)
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('selection',inputs,
                Path(__import__('kernel_optimizer.project.agents',fromlist=['__file__']).__file__).with_name('selection_prompt.md').read_text(),
                ('selection.json',),feedback,validator=lambda path:check_selection(path,discovery,policy['allowed_scopes'],eligible))
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            meta,actual_prompt,actual=load(path)
            if meta['fresh_session']:
                assert actual==expected and actual_prompt==prompt
                assert meta['framework_source']==saved['framework_source'] and meta['tool_model_settings']==saved['tool_model_settings']
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['selection']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'inputs_verified':True,'prompt_model_tools_core_unchanged':True,'selection_sha256':sha(directory/'selection.json'),
            'scope':'Selection/schema only; full actual tools and reasoning/code evidence need terminal review before any candidate/GPU stage.'}
    except Exception:results['selection']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items());results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['selection']['status']=='completed' else 'completed_with_errors','finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');main(parser.parse_args())
