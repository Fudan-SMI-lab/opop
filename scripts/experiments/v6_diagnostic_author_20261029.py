"""One real diagnostic-author role from a frozen failure, without later answers.

No generated probe executes here. Review and fixed controls are a separate stage.
This experiment uses existing role/snapshot APIs, not a production routing change.
"""
import argparse
import ast
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


INSTRUCTIONS = """Design a small diagnostic collector for the supplied failed project evaluation.
Your task is to acquire missing evidence, not optimize or repair the project/candidate.
Use inputs/evidence-index.json and the original failed worker record. Read the actual
adapter, candidate and evaluator code to identify what the recorded failure proves,
what is unknown, and the smallest observations that would resolve those unknowns.
Do not assume that a successful run-only diagnostic established complete quality.
Distinguish original observations, code hypotheses and proposed measurements.

Write diagnostic.py and diagnostic-plan.json. diagnostic.py exports run(context)
for the existing LocalExecutor worker interface. The caller will use each original
baseline/candidate ExecutionRequest, changing only entry/function/output/wait settings
and adding parameters.diagnostic_context as documented in inputs/interface.md.
Derive any task-specific instrumentation and field mapping from the supplied code,
document the source locations, and keep the entry independent of fixed experiment
paths, candidate hashes and device IDs. Use current context and indexed metadata.
The original computation, bindings, continuous quality protocol, precision, reset,
completion boundaries, quality thresholds and clean repetitions must be preserved.
Do not replace the measured workload with a toy or reconstruct an alternate fast path.
Instrumentation must be removable/restored even on errors. Persist evidence before
propagating original evaluation errors; do not convert a failed evaluation to valid.
Report absent observations as absent. Do not manufacture data after a failed stage.

Store raw diagnostic values on the server only when needed to support exact comparisons;
produce a small JSON index with paths/hashes, phase/invocation identity and limitations.
Explain allocation/synchronization/lifetime perturbations and what a matching baseline
control can and cannot establish. Diagnostic timings never become acceptance scores.
Propose independent controls and negative probes before interpreting a claimed cause.
No GPU work in this author session. CPU/library/API/syntax checks are allowed, with
their failures retained; CPU mocks do not establish full-model quality. The output
will be reviewed before any leased GPU execution. Do not write an optimized candidate.

This experiment supplies an original failure without subsequent diagnostic answers.
Use only supplied inputs, indexed original source/worker/reference files and relevant
installed framework/library source. Do not inspect later experiment results, existing
developer diagnostic scripts, other candidates or accepted implementations. The
filesystem is not hermetic; actual tool accesses will be audited for contamination.
No production prompt, collector, quality or search changes are adopted from this output.

diagnostic-plan.json must contain: evidence_gap (string), source_mappings (array),
observations (array), controls (array), perturbations (array), limitations (array),
unresolved (array), and interface (object). Each source mapping must explain its
meaning and code evidence. Keep hypotheses conditional; generation is not validation.
"""


INTERFACE = """Worker entry contract:
def run(context): ...
context has rank, world_size, device (runtime logical string), output (new directory),
parameters, binding, and optional revision. Original parameters are unchanged:
request, adapter, action='evaluate', collector_options, reference_dir, bundles.
An additional parameters['diagnostic_context'] will contain:
  original_entry: path to original evaluator entry; original_function: its function;
  independent_reference: optional path to source reference tensors;
  arm: a descriptive label only, never a reason to change computation or assert results.
Use only context['output'] for generated artifacts. The executor imports this file
with cwd set to the original materialized source. Do not hardcode any experiment path.
Run the original evaluator exactly once per supplied worker; it owns steps/warmup/
repeats and binding. Do not silently rerun a failure, suppress assertions, change
acceptance, force GC, alter precision/backends or modify external files. Original
exceptions must propagate after evidence has been persisted. A successful return
must preserve original evaluator validity/quality result, with an optional diagnostic
index field. Use actual code review to choose observational hooks, not a special
failure-message router. Keep server evidence indexed, not large tensor dumps in replies.
"""


def now():return datetime.now(timezone.utc).isoformat()


def build_inputs(work):
    c=work/'qwen-constraint-continuation-20261029/qwen-continuation/jobs/trial-a254520daa-param-fc711840-screen/run/request.json'
    b=work/'qwen-end-to-end-original-followup-20261029/qwen-original/jobs/trial-b8ecdd4254-evaluation-762e158a-pair-0-parent/run/request.json'
    requests={name:read(p)for name,p in [('baseline',b),('candidate',c)]}
    assert requests['baseline']['parameters']['request']==requests['candidate']['parameters']['request']
    cp=requests['candidate']['parameters'];assert len(cp['bundles'])==1 and cp['bundles'][0]['parameters']=={}
    candidate=Path(cp['bundles'][0]['file']);assert sha(candidate)=='a86694e792df3baac1ea6d93e5de574fc43fa1818ebee4ce452d4f4557295178'
    failure=read(c.parent/'rank-0.json');assert failure['status']=='failed'
    assert failure['error']=='teacher-forced replay token sequence differs from greedy run'
    adapter=Path(cp['adapter']);assert sha(adapter)=='498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7'
    # Only original source/reference metadata, never the subsequent reviewed mapping,
    # raw observer output, successful implementation, or developer causal diagnosis.
    original=read(work/'qwen-end-to-end-original-20261029/protocol.json')
    ref=original['independent_source_reference'];assert sha(ref['path'])==ref['sha256']
    expected={str(p):sha(p)for p in [b,c,c.parent/'rank-0.json',candidate,adapter,Path(ref['path'])]}
    source_files={}
    for arm,path in [('baseline',b),('candidate',c)]:
        request=requests[arm];manifest=read(Path(request['cwd'])/'.opop-revision.json')
        expected[str(Path(request['cwd'])/'.opop-revision.json')]=sha(Path(request['cwd'])/'.opop-revision.json')
        for name,h in manifest['files'].items():
            p=Path(request['cwd'])/name;assert sha(p)==h;expected[str(p)]=h
        source_files[arm]={name:{'path':str(Path(request['cwd'])/name),'sha256':h}for name,h in manifest['files'].items()}
    controls=Path(cp['reference_dir'])/'reference-rank-0.pt';expected[str(controls)]=sha(controls)
    index={'original_execution_requests':{k:{'path':str(p),'sha256':sha(p)}for k,p in [('baseline',b),('candidate',c)]},
        'source_files':source_files,'failed_worker':{'path':str(c.parent/'rank-0.json'),'sha256':sha(c.parent/'rank-0.json')},
        'independent_source_reference':ref,'controls_reference':{'path':str(controls),'sha256':sha(controls)},
        'original_entries':{k:{'path':r['entry'],'function':r['function'],'sha256':sha(r['entry'])}for k,r in requests.items()},
        'scope':'Original failed evaluation and prior baseline only. No later developer diagnosis, phase index, observed values or repair implementation supplied.'}
    inputs={'task.json':json.dumps(cp['request'],indent=2),
        'failure.json':json.dumps({k:failure[k]for k in ['status','error_type','error','traceback','framework_source']},indent=2),
        'adapter.py':adapter.read_bytes(),'candidate.py':candidate.read_bytes(),
        'evidence-index.json':json.dumps(index,indent=2),'interface.md':INTERFACE}
    return inputs,expected,index


def validate(directory):
    text=(directory/'diagnostic.py').read_text();tree=ast.parse(text);compile(text,'diagnostic.py','exec')
    if not any(isinstance(n,ast.FunctionDef)and n.name=='run'for n in tree.body):raise ValueError('diagnostic.py must export run(context)')
    data=read(directory/'diagnostic-plan.json')
    if not isinstance(data.get('evidence_gap'),str)or not isinstance(data.get('interface'),dict):raise ValueError('evidence_gap string and interface object required')
    for key in ['source_mappings','observations','controls','perturbations','limitations','unresolved']:
        if not isinstance(data.get(key),list):raise ValueError(key+' must be an array')


def main(args):
    os.umask(0o077);inputs,expected,index=build_inputs(args.work)
    if args.preflight:
        print(json.dumps({'status':'passed','input_sizes':{k:len(v.encode()if isinstance(v,str)else v)for k,v in inputs.items()},'frozen_files':len(expected)}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'scope':'Single fresh real diagnostic-author generation; no GPU execution/search/acceptance or production adoption. Inputs original a866 failure only, no later outcomes. External filesystem access audited, not hermetic.',
        'frozen_identities':expected,'input_index':index,'sequence':['one real diagnostic_author session','syntax/schema only','preservation check'],
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    config=load_config(args.config,[]);config.usage.mode='observe_only';results={}
    write_json(args.output/'state.json',{'phase':'running','stage':'diagnostic-author','at':now()})
    try:
        root=args.output/'diagnostic-author';store=RunStore.create(root.parent,root.name,{'kind':'isolated_diagnostic_author'})
        with Runtime(project_agent_config(config),root)as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS,
                ('diagnostic.py','diagnostic-plan.json'),validator=validate)
        results['diagnostic-author']={'status':'completed','agent':receipt,'probe':str(directory/'diagnostic.py'),
            'probe_sha256':sha(directory/'diagnostic.py'),'plan':str(directory/'diagnostic-plan.json'),'plan_sha256':sha(directory/'diagnostic-plan.json'),
            'validation':'Syntax/schema only; original generated bytes unexecuted on GPU. Requires code and actual tool/probe review.'}
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
