"""One artifact-only evidence-reader author; no live collector or optimization."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config, worker_environment
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_pipeline_communication_selection_20261031 import CURRENT, CODE, now
from scripts.experiments.v6_ddp_observation_mechanics_repair_20261031 import expected_prompt
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

AUDIT='1797e5de55b0f6c816a5c83f5fc1990fe0a26efdd156378cf23b8a16aef7e098'
QUESTION='''For the unchanged current accepted DDP implementation, what can the existing
original artifacts and relevant installed library source establish about the actual
protocol identity of steady workload AllReduce operations and the scope/timing of
protocol configuration and communicator creation? What remains unknown?
The supplied original project selection is a hypothesis, not an instruction to
implement its proposed protocol, groups, arithmetic, segmentation or predicted gain.
Determine which facts can be joined within a capture and which cannot, with precise
provenance and units. Identify any concrete missing fact that blocks a communication
decision. No desired protocol, algorithm, group, collector or causal answer is given.
This is one bounded evidence question, not another selector, candidate correction,
runtime observer repair, reviewer stack or optimization acceptance.
'''
INTERFACE='''Write artifact-reader.py exporting run(context), a plain CPU artifact-analysis
entry point. context is a dict with artifact_index (path to the supplied index JSON)
and output_dir (a new designated report directory). Return a JSON-serializable report
and persist it under output_dir. The artifact index supplies input paths and hashes.
The reader may use read-only JSON, text, SQLite and installed source inspection;
it must never invoke the training adapter/evaluator, model, GPU, distributed process
group, collective, profiler or live workload. Do not alter runtime environment,
protocol settings, inputs, source, configuration or artifacts. Open SQLite read-only.
Use bounded metadata and exact evidence references rather than copying raw traces,
weights, tensors, credentials or entire source trees. Unknown or failed reads must
remain explicit. No fixed experiment paths, expected answers or output hashes in code.
This interface is independent of the closed runtime observer and its worker hooks.
Only authoring and syntax/schema checks are scheduled now. The generated reader must
be reviewed before any separately declared CPU artifact-analysis execution.
'''
INSTRUCTIONS='''Answer the supplied bounded evidence question by authoring a small read-only
artifact reader and its evidence plan, using the actual original artifacts and
relevant installed source. Read evidence-question.md, interface.md and evidence-index.json.
Write ONLY artifact-reader.py and evidence-plan.json as final outputs. Do not author
an optimized workload, runtime collector, patched evaluator or alternative candidate.
Choose the analysis yourself; do not treat the supplied selection as established fact.
Distinguish direct observations, source inference, unverified hypotheses and missing
evidence. Every conclusion must reference actual artifact identity, field/line/query,
capture/rank/phase and units as applicable. Independent captures do not share sequence
identities or clocks without evidence. Kernel names, host launch times, configuration
logs, collective completion and physical wire transfer are different evidence types.
Do not infer causal critical-path savings or protocol-specific overhead from a
subtraction across different profiles/calibrations. Preserve tails, errors, absent
tables and ambiguous joins; an explicit unknown is a useful result.

No GPU initialization, model/tensor computation, training, process-group creation,
real collective (including world-size-one Gloo), profiler, live collector, subprocess
workload, environment/configuration change or new capture in this author session or
generated reader. CPU JSON/SQLite/source parsing, syntax/API metadata and inert mocks
are allowed. No standalone performance tests or parameter alternatives. Read only
supplied inputs, indexed original artifacts and relevant installed library/framework
source; do not inspect private provider configs, later audits, unrelated experiments,
reviewer answers or closed repair fixtures. The external filesystem is not hermetic;
all actual tool inputs and outputs will be audited. Do not print credentials.

The plan must explain answerability with existing evidence, limitations, bounded
output, validation using inert data and the concrete missing observation if one is
necessary. Do not prescribe an optimization or execute further measurement. No result
is automatically adopted as a production collector, prompt change or acceptance.
evidence-plan.json must contain evidence_question(string), source_mappings(array),
claims(array), unknowns(array), validation(array), limitations(array), storage(object),
interface(object). Each claim must distinguish its evidence from an inference.
Generation success is syntax/schema only. There is no automatic further repair chain.
'''

def validate(directory):
    tree=ast.parse((directory/'artifact-reader.py').read_text())
    if not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in tree.body):
        raise ValueError('artifact-reader.py must export run(context)')
    try: plan=read(directory/'evidence-plan.json')
    except (json.JSONDecodeError,OSError) as exc: raise ValueError(str(exc)) from exc
    for key,kind in {'evidence_question':str,'source_mappings':list,'claims':list,'unknowns':list,
                     'validation':list,'limitations':list,'storage':dict,'interface':dict}.items():
        if not isinstance(plan.get(key),kind):raise ValueError(f'{key} must be {kind.__name__}')

def prepare(args):
    parent=args.parent; protocol=read(parent/'protocol.json')
    assert sha(parent/'terminal-audit.json')==AUDIT
    audit=read(parent/'terminal-audit.json')
    assert audit['all_frozen_unchanged'] and audit['source_current']==CURRENT
    terminal=read(parent/'followup-completed.json');assert terminal['phase']=='completed'
    meta,_,files=load(Path(terminal['results']['selection']['initial_snapshot']))
    sandbox=Path(meta['sandbox']);frozen=dict(protocol['frozen_identities'])
    def item(path):
        path=Path(path);h=sha(path);frozen[str(path)]=h
        return {'path':str(path),'sha256':h,'bytes':path.stat().st_size}
    source=Path(protocol['source']);current=read(source/'project/current.json')
    assert CURRENT in json.dumps(current)
    inputs={};index={'scope':'Existing exact current accepted artifacts and original selector hypothesis only. No audit conclusions or corrected causal/algorithm table. No live workload/collector, optimization, closed observer/candidate repair or production adoption.',
                     'source_current':CURRENT,'inputs':{},'original_artifacts':{},'selection_status':'Unvalidated project hypothesis; no candidate generated from it.'}
    names=['request.json','adapter.py','search.json','communication-ownership.json',
           'source/README.md','source/common.py','source/gpt2_ddp_training.py',
           'source/opop_candidates/fixed-fused-pipeline.py']
    names += [n.removeprefix('inputs/') for n in files if n.startswith('inputs/observations/current/')]
    assert len(names)==12
    for name in names:
        inputs[name]=files['inputs/'+name];index['inputs'][name]=item(sandbox/'inputs'/name)
    assert hashlib.sha256(inputs['source/opop_candidates/fixed-fused-pipeline.py']).hexdigest()==CODE
    inputs['selection.json']=(sandbox/'selection.json').read_bytes()
    index['inputs']['selection.json']=item(sandbox/'selection.json')
    assert index['inputs']['selection.json']['sha256']==audit['selection_sha256']
    # Reference the original five current-code captures, never copy raw traces or controls.
    for path in sorted((source/'jobs').rglob('*')):
        if path.is_file() and not path.is_symlink() and path.suffix in {'.json','.sqlite','.log'}:
            index['original_artifacts'][str(path.relative_to(source/'jobs'))]=item(path)
    for name in ['protocol.json','followup-completed.json','terminal-audit.json','public-terminal-audit.json',
                 'startup-verification.json','native-arithmetic-check.json']:
        item(parent/name)
    item(Path(terminal['results']['selection']['initial_snapshot']));item(parent/'selection/events.jsonl')
    inputs['evidence-question.md']=QUESTION;inputs['interface.md']=INTERFACE
    inputs['evidence-index.json']=json.dumps(index,indent=2)
    expected={'inputs/'+n:v.encode() if isinstance(v,str)else v for n,v in inputs.items()}
    expected['inputs/worker-environment.json']=json.dumps(worker_environment(),indent=2).encode()
    assert all(sha(p)==h for p,h in frozen.items())
    assert framework_source()==protocol['core']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=8*1024**3+256*1024**2
    details={'core':framework_source(),'source':str(source),'current':CURRENT,'frozen_files':len(frozen),
             'inputs':len(expected),'input_bytes':sum(map(len,expected.values())),
             'input_hashes':{n:hashlib.sha256(v).hexdigest()for n,v in expected.items()},
             'author_body_sha256':hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
             'free_bytes':free,'additional_budget_bytes':256*1024**2,'warning_floor_bytes':8*1024**3,
             'scope':'One artifact-only protocol-attribution diagnostic_author. Original selection is hypothesis; no prescribed answer, optimization code or runtime collector. No GPU/model/collective, source/control run/copy, production prompt/model/tools/core changes or automatic reader execution. Experiment question/interface only; prior assistance boundaries persist.'}
    return inputs,expected,frozen,index,details

def main(args):
    os.umask(0o077);inputs,expected,frozen,index,details=prepare(args)
    if args.preflight:print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen})
    write_json(args.output/'evidence-index.json',index)
    write_json(args.output/'preparation-reuse.json',{'source':details['source'],'current':CURRENT,'scope':details['scope']})
    write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'diagnostic-author'});results={}
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    try:
        store=RunStore.create(args.output,'diagnostic-author',{'kind':'ddp-protocol-attribution-author'})
        with Runtime(config,log_dir=store.run_dir)as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS,
                ('artifact-reader.py','evidence-plan.json'),validator=validate)
        initial=[]
        old,_,_=load(Path(read(args.parent/'followup-completed.json')['results']['selection']['initial_snapshot']))
        for path in(store.run_dir/'generation-snapshots').glob('*.json'):
            meta,prompt,actual=load(path)
            if meta['fresh_session']:
                assert actual==expected and prompt==expected_prompt(INSTRUCTIONS,json.loads(actual['inputs/worker-environment.json']))
                assert meta['framework_source']==details['core'] and meta['model']==old['model'] and meta['tool_model_settings']==old['tool_model_settings']
                initial.append(str(path))
        assert len(initial)==1
        results['diagnostic-author']={'status':'completed','agent':receipt,'initial_snapshot':initial[0],
            'reader':str(directory/'artifact-reader.py'),'reader_sha256':sha(directory/'artifact-reader.py'),
            'plan':str(directory/'evidence-plan.json'),'plan_sha256':sha(directory/'evidence-plan.json'),
            'inputs_verified':True,'validation':'Syntax/schema only. Full tool/code audit required; no automatic execution or acceptance.'}
    except Exception:results['diagnostic-author']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items())
    results['preservation']={'status':'passed'if same else'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed'if same and results['diagnostic-author']['status']=='completed'else'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    for name,value in [('followup-results.json',results),('state.json',terminal),('followup-completed.json',terminal)]:write_json(args.output/name,value)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');main(parser.parse_args())
