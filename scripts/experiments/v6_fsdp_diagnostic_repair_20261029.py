"""One fresh real diagnostic author repair from a concrete CPU interface failure."""
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
from scripts.experiments.v6_fsdp_diagnostic_author_20261029 import build_inputs as original_inputs,INSTRUCTIONS,ADDITIONAL,validate


REPAIR = """
This is one declared repair of your earlier diagnostic output, not a new project
optimization or a request to repeat until successful. inputs/previous-diagnostic.py
and inputs/previous-diagnostic-plan.json are exact prior outputs. A new small CPU
fixture used real framework compare and your exact script; it did not run a model,
GPU, or distributed collective. The finite case preserved original valid=false.
The NaN and Inf cases in execution-failure.json had an original returned valid=false
at output.logits, followed by nonfinite data in a later observation. Your entry
raised a new JSON ValueError and saved no diagnostic evidence instead of preserving
that failed-quality result. This is a demonstrated diagnostic persistence defect,
NOT a claim that the actual FSDP run has nonfinite observations.
The small CPU fixture source named in that failure record may be inspected to
understand the reproduction; this does not authorize reading later experiments.

Make the smallest correction that preserves original failed returned decisions
and original exceptions while retaining truthful serializable evidence when a
diagnostic observation includes nonfinite values. Do not hide or label nonfinite
observations as finite/within tolerance, invent data, weaken original quality or
alter model/collective/control flow. Preserve finite-case measurements and traversal
behavior. Keep the previous probe's entry and output contract; document the change
and its limits in diagnostic-plan.json. Do not redesign instrumentation, add a new
collector, optimize the candidate, or act on unrelated proposed parameter/rerun
controls. Any unaffected limitation stays explicit, not silently reinterpreted.
Run appropriate small CPU checks on finite, NaN and Inf values, including return
and exception propagation and hook restoration, retaining failures and artifacts
under your sandbox. No GPU/model/reference copies, no new FSDP experiments. We will
review exact output before any later fixed GPU execution. No later Qwen diagnostics,
FSDP successful implementations or developer numerical causes have been supplied;
do not inspect them. API usage is observed only, never capped by this experiment.
"""


def now():return datetime.now(timezone.utc).isoformat()


def build_inputs(parent):
    inputs,expected,index=original_inputs(parent.parent)
    a=read(parent/'terminal-audit.json');assert not a['review']['execute_original_on_gpu']
    generated=a['generated'];failure=Path(a['persistence_cpu']['path'])
    for key in ['probe','plan']:assert sha(generated[key])==generated[key+'_sha256'];expected[generated[key]]=generated[key+'_sha256']
    assert sha(failure)==a['persistence_cpu']['sha256'];expected[str(failure)]=sha(failure)
    inputs.update({'previous-diagnostic.py':Path(generated['probe']).read_bytes(),
        'previous-diagnostic-plan.json':Path(generated['plan']).read_bytes(),
        'execution-failure.json':json.dumps({k:v for k,v in read(failure).items()if k!='top_mismatch_label_fixture'},indent=2)})
    return inputs,expected,index


def main(args):
    os.umask(0o077);inputs,expected,index=build_inputs(args.parent)
    if args.preflight:
        print(json.dumps({'status':'passed','input_sizes':{k:len(v.encode()if isinstance(v,str)else v)for k,v in inputs.items()},'frozen_files':len(expected)}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent':str(args.parent),
        'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),'frozen_identities':expected,
        'scope':'One fresh real author request, narrow persistence defect repair. Original six task inputs unchanged, prior exact probe/plan and CPU failure added. No FSDP numerical diagnosis, GPU/search/acceptance/production adoption.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    config=load_config(args.config,[]);config.usage.mode='observe_only';results={}
    write_json(args.output/'state.json',{'phase':'running','stage':'diagnostic-author-repair','at':now()})
    try:
        root=args.output/'diagnostic-author';store=RunStore.create(root.parent,root.name,{'kind':'isolated_diagnostic_author_repair'})
        with Runtime(project_agent_config(config),root)as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS+ADDITIONAL+REPAIR,
                ('diagnostic.py','diagnostic-plan.json'),validator=validate)
        results['diagnostic-author']={'status':'completed','agent':receipt,'probe':str(directory/'diagnostic.py'),
            'probe_sha256':sha(directory/'diagnostic.py'),'plan':str(directory/'diagnostic-plan.json'),'plan_sha256':sha(directory/'diagnostic-plan.json'),
            'validation':'Syntax/schema only. Audit actual code/tools and independent CPU controls before fixed GPU validation.'}
    except Exception:results['diagnostic-author']={'status':'error','traceback':traceback.format_exc()}
    unchanged=all(sha(p)==h for p,h in expected.items());results['preservation']={'status':'completed'if unchanged else'error','frozen_files_unchanged':unchanged}
    write_json(args.output/'followup-results.json',results)
    terminal={'phase':'completed'if all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal);print(json.dumps({'phase':terminal['phase'],'finished_at':terminal['finished_at']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
