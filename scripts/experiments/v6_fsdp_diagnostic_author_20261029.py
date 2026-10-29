"""Second-task diagnostic author feasibility: original FSDP quality rejection."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_diagnostic_author_20261029 import INSTRUCTIONS,INTERFACE,validate


ADDITIONAL = """
The supplied original worker may finish normally while returning a failed quality
decision. Preserve that original returned decision just as you preserve exceptions;
neither a completed worker nor an evidence file means the project passed quality.
Derive missing observations from the actual available record, not a predetermined
failure phase or cause. Respect the supplied multi-process execution semantics:
use the actual rank/world_size and existing shard layouts, avoid extra collectives
or rank-asymmetric work that changes execution/deadlock behavior. Do not assume
replicas when observations are shards. Explain any additional CPU/device work and
retained owners. Compare only compatible tensors and state what was not observed.
No GPU work in this author session; no source/reference/model copies, large raw
exports or new traces. Existing reference paths are read-only; metadata or CPU
checks are allowed. Plan selective raw persistence and small indexed summaries
according to the demonstrated evidence gap, not blanket dumps of all model states.
The source request and failure are a different task, not a replay of a successful
diagnostic implementation. Do not inspect prior diagnostic-author experiments,
developer observers/audits, subsequent accepted versions or later trial results.
"""


def now():return datetime.now(timezone.utc).isoformat()


def build_inputs(work):
    job=work/'fsdp-search-20261028/fsdp-search/jobs/trial-86e14487d0-param-b3a4c7d3-screen'
    path=job/'run/request.json';request=read(path);p=request['parameters'];assert p['action']=='evaluate'
    adapter=Path(p['adapter']);candidate=Path(p['bundles'][0]['file']);assert len(p['bundles'])==1
    assert p['bundles'][0]['parameters']=={'keep_unsharded_blocks':6}
    assert sha(adapter)=='8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6'
    failures={};expected={str(path):sha(path)};source_files={}
    manifest=read(Path(request['cwd'])/'.opop-revision.json')
    for name,h in manifest['files'].items():
        file=Path(request['cwd'])/name;assert sha(file)==h;expected[str(file)]=h
        source_files[name]={'path':str(file),'sha256':h}
    manifest_path=Path(request['cwd'])/'.opop-revision.json';expected[str(manifest_path)]=sha(manifest_path)
    references={}
    for rank in [0,1]:
        worker_path=job/f'run/rank-{rank}.json';worker=read(worker_path)
        assert worker['status']=='passed' and worker['result']['valid'] is False
        assert len(worker['result']['checks'])==2 and worker['result']['checks'][0]['passed'] and not worker['result']['checks'][1]['passed']
        assert worker['result']['samples']==[] and worker['result']['math_policy_unchanged']
        failures[str(rank)]={'path':str(worker_path),'sha256':sha(worker_path),'record':worker}
        expected[str(worker_path)]=sha(worker_path)
        ref=Path(p['reference_dir'])/f'reference-rank-{rank}.pt';expected[str(ref)]=sha(ref)
        references[str(rank)]={'path':str(ref),'bytes':ref.stat().st_size,'sha256':sha(ref)}
    index={'original_execution_request':{'path':str(path),'sha256':sha(path)},
        'source_files':source_files,'frozen_rank_references':references,
        'original_entry':{'path':request['entry'],'function':request['function'],'sha256':sha(request['entry'])},
        'baseline_control':'No new baseline endpoint has been run. A prospective matched control may use this exact request and source with parameters.bundles=[]; this is a declared derived control, not a historical measured endpoint.',
        'independent_reference':None,
        'scope':'Only original failed-quality parameter observation, source/adapter/candidate and frozen controls. No accepted FSDP policy, Qwen diagnostic code, later audit or causal answer supplied.'}
    inputs={'task.json':json.dumps(p['request'],indent=2),'failure.json':json.dumps(failures,indent=2),
        'adapter.py':adapter.read_bytes(),'candidate.py':candidate.read_bytes(),
        'evidence-index.json':json.dumps(index,indent=2),'interface.md':INTERFACE}
    return inputs,expected,index


def main(args):
    os.umask(0o077);inputs,expected,index=build_inputs(args.work)
    if args.preflight:
        print(json.dumps({'status':'passed','input_sizes':{k:len(v.encode()if isinstance(v,str)else v)for k,v in inputs.items()},'frozen_files':len(expected)}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'frozen_identities':expected,'input_index':index,
        'scope':'One fresh FSDP diagnostic-author role. No GPU probe/search/formal acceptance/production adoption. Original first parameter quality rejection only, no later answers. Prospective empty-bundle control explicitly derived.',
        'instructions_base':'v6_diagnostic_author_20261029.INSTRUCTIONS unchanged; one explicit generic multi-rank/invalid-return/resource paragraph appended.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    config=load_config(args.config,[]);config.usage.mode='observe_only';results={}
    write_json(args.output/'state.json',{'phase':'running','stage':'diagnostic-author','at':now()})
    try:
        root=args.output/'diagnostic-author';store=RunStore.create(root.parent,root.name,{'kind':'isolated_diagnostic_author'})
        with Runtime(project_agent_config(config),root)as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS+ADDITIONAL,
                ('diagnostic.py','diagnostic-plan.json'),validator=validate)
        results['diagnostic-author']={'status':'completed','agent':receipt,'probe':str(directory/'diagnostic.py'),
            'probe_sha256':sha(directory/'diagnostic.py'),'plan':str(directory/'diagnostic-plan.json'),'plan_sha256':sha(directory/'diagnostic-plan.json'),
            'validation':'Syntax/schema only; full tools/code review and separate real execution still required.'}
    except Exception:results['diagnostic-author']={'status':'error','traceback':traceback.format_exc()}
    unchanged=all(sha(p)==h for p,h in expected.items())
    results['preservation']={'status':'completed'if unchanged else'error','frozen_files_unchanged':unchanged}
    write_json(args.output/'followup-results.json',results)
    terminal={'phase':'completed'if all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)
    print(json.dumps({'phase':terminal['phase'],'finished_at':terminal['finished_at']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','work']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
