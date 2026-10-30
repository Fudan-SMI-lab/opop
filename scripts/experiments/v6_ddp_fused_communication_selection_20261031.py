"""One ordinary project selection on the independently accepted fused version."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import textwrap
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.experiment_policy import check_selection
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config, worker_environment
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

CURRENT='10992e0d329f45c941eb2edcd31f60fc1e7f17c7a6bba8241bad93edde56a979'
CODE='508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a'
AUDIT='7cd9fa02d28a60b4fec48a69cb1c4365842e079cc928dd03db409d3f91da3ab9'

def now(): return datetime.now(timezone.utc).isoformat()
def digest(raw): return hashlib.sha256(raw).hexdigest()

def ordinary_inputs(search, feedback):
    """Execute only the existing read-only ordinary trial input-assembly prefix.

    The prefix stops before trial ID/directory/runtime creation. No selector,
    candidate, trial, collector or worker is run here. This retains the exact
    production default policy and common-input packaging without hand routing.
    """
    from kernel_optimizer.project import search as module
    assert read(search.output/'discovery.json')['revision']==search.revisions.current().id
    source=textwrap.dedent(inspect.getsource(ProjectSearch._trial))
    method=ast.parse(source).body[0]
    stop=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='prompt_root' for t in n.targets))
    statements=method.body[:stop+1]
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in {'mkdir','generate','execute','validate_bundle','discover','write_text','write_bytes'}
                   for s in statements for n in ast.walk(s))
    namespace=dict(module.__dict__)
    namespace.update(self=search,feedback=feedback,allowed_scopes=('group','joint','region','execution_plan'),
                     leaf_groups_only=False,selection_strategy='agent',feedback_view='full')
    exec(compile(ast.Module(body=statements,type_ignores=[]),'<ordinary-input-assembly>','exec'),namespace)
    prefix='\n'.join(source.splitlines()[:statements[-1].end_lineno])
    return namespace,digest(prefix.encode())

def prepare(args,config):
    root=args.parent; protocol=read(root/'protocol.json')
    assert sha(root/'terminal-audit.json')==AUDIT
    audit=read(root/'terminal-audit.json')
    assert audit['all_frozen_unchanged'] and audit['current']==CURRENT and len(audit['jobs'])==5
    assert read(root/'followup-completed.json')['phase']=='completed'
    assert protocol['core']==framework_source()
    search=ProjectSearch(config,root/'ddp-evidence');current=search.revisions.current()
    assert current.id==CURRENT and len(current.bundles)==1
    bundle=search.revisions.bundle(current.bundles[0]);assert bundle.parameters=={}
    assert {f.path:f.sha256 for f in current.files}[bundle.installer]==CODE
    raw=(search.output/'latest-feedback.json').read_bytes()
    original=root.parent/'ddp-observation-route-independent-20261031/ddp-independent/version-trials/version-aefa021e172f/result.json'
    assert raw==original.read_bytes() and json.loads(raw)['decision']['accepted']
    assert 'supplemental_evidence' not in json.loads(raw)
    data,assembly_sha=ordinary_inputs(search,raw.decode())
    feedback=route(data['feedback'],current_revision=current.id,registry=search.revisions.load(current.registry_ref,GroupRegistry))
    routed=json.loads(feedback)
    assert routed['decision']==json.loads(raw)['decision'] and routed['complete_version']['revision']==json.loads(raw)['revision']
    assert routed['supplemental_evidence'] is None
    frozen=dict(protocol['frozen_identities'])
    for name in ['protocol.json','terminal-audit.json','public-terminal-audit.json','followup-completed.json']:
        frozen[str(root/name)]=sha(root/name)
    for name in ['request.json','prepared.json','project/current.json','discovery.json','latest-feedback.json','events.jsonl']:
        frozen[str(search.output/name)]=sha(search.output/name)
    for directory in ['jobs','project/revisions','project/transactions','generation-snapshots']:
        for path in (search.output/directory).rglob('*'):
            if path.is_file() and not path.is_symlink():frozen[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in frozen.items())
    inputs=data['common'];expected={'inputs/'+n:v.encode() if isinstance(v,str) else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode()
    from kernel_optimizer.project import agents
    expected['inputs/binding-contract.md']=Path(agents.__file__).with_name('binding_contract.md').read_bytes()
    expected['inputs/worker-environment.json']=json.dumps(worker_environment(),indent=2).encode()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=8*1024**3+384*1024**2
    details={'core':framework_source(),'source':str(search.output),'current':current.id,'code_sha256':CODE,
        'frozen_files':len(frozen),'inputs':len(expected),'input_bytes':sum(map(len,expected.values())),
        'input_hashes':{k:digest(v) for k,v in expected.items()},'ordinary_assembly_sha256':assembly_sha,
        'raw_feedback_sha256':digest(raw),'raw_feedback_bytes':len(raw),
        'routed_feedback_sha256':digest(feedback.encode()),'routed_feedback_bytes':len(feedback.encode()),
        'supplemental_evidence':None,'default_policy':data['policy'],'devices':search.request.devices,
        'request_sha256':sha(search.output/'request.json'),'prepared_sha256':sha(search.output/'prepared.json'),
        'discovery_sha256':sha(search.output/'discovery.json'),'free_bytes':free,
        'additional_budget_bytes':384*1024**2,'warning_floor_bytes':8*1024**3,
        'scope':'One ordinary project selection on exact independently accepted508a and its fresh9call8group discovery. Original complete-version feedback and normal current collector artifacts only; no developer terminal audit/cause/group/implementation or new supplement. Prior source/metric/CPU assistance and actual planner external historical-plan access remain provenance boundaries. No candidate/GPU/trial/collector/sourcecontrol/referencecopy/pointer/production change or closed-chain restart.'}
    return search,data,feedback,expected,frozen,details

def main(args):
    os.umask(0o077);config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    search,data,feedback,expected,frozen,details=prepare(args,config)
    if args.preflight:print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen})
    write_json(args.output/'transport-preflight.json',details)
    (args.output/'original-feedback.json').write_bytes((search.output/'latest-feedback.json').read_bytes())
    (args.output/'routed-feedback.json').write_text(feedback)
    write_json(args.output/'preparation-reuse.json',{k:details[k] for k in ['source','current','request_sha256','prepared_sha256','discovery_sha256','scope']})
    write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'selection-only'});results={}
    try:
        store=RunStore.create(args.output,'selection',{'kind':'ddp-fused-communication-selection'})
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('selection',data['common'],
                (data['prompt_root']/'selection_prompt.md').read_text(),('selection.json',),feedback,
                validator=lambda path:check_selection(path,data['discovery'],data['policy']['allowed_scopes'],data['eligible']))
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            meta,prompt,actual=load(path)
            if meta['fresh_session']:
                assert actual==expected and meta['framework_source']==details['core']
                old,oldprompt,_=load(args.parent.parent/'ddp-observation-feedback-selection-20261031/selection/generation-snapshots/generation-999c08dfa37c.json')
                assert prompt==oldprompt and meta['tool_model_settings']==old['tool_model_settings']
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['selection']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'selection_sha256':sha(directory/'selection.json'),'inputs_verified':True,
            'scope':'Selection/schema only. Complete access/reasoning audit required before any implementation or GPU work.'}
    except Exception:results['selection']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items()) and search.revisions.current().id==CURRENT
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['selection']['status']=='completed' else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    for name,value in [('followup-results.json',results),('state.json',terminal),('followup-completed.json',terminal)]:write_json(args.output/name,value)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');main(parser.parse_args())
