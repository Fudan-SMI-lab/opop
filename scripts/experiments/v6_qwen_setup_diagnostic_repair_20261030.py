"""One real narrow repair of demonstrated diagnostic retention, no GPU execution."""
import argparse
import json
import os
from pathlib import Path
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_setup_author_20261030 import INSTRUCTIONS, EXTRA, now, validate

REPAIR = """
Repair only the concrete evidence-retention defect in the previous generated probe.
The supplied CPU execution record shows that successful comparisons exhaust the
record cap and a later real failing comparison loses its operands and outcome.
This is a diagnostic interface defect, not a model numerical result or repair answer.
Preserve bounded useful evidence for late failures after many successes, with truthful
counts, omitted/overflow scope and retained comparison provenance. Do not remove
all resource bounds, use fixed failure-message/outcome routing, dump model/cache
state, change quality thresholds, modify candidate code or rerun model computation.
Keep original once-only invocation, returns, internally caught failures, uncaught
exceptions and restoration behavior. Make a narrow change to diagnostic.py and
update diagnostic-plan.json to describe actual bounds and remaining limitations.
Use inputs/previous-diagnostic.py and previous-diagnostic-plan.json as the exact
generated-before state. The named CPU fixture source and its recorded outputs in
inputs/execution-failure.json may be read; developer terminal audits, later answers
and other experiment outputs remain outside scope. CPU tests of the repair are
allowed; preserve failures and distinguish those tests from full GPU execution.
"""


def build_inputs(work):
    parent = work/'qwen-setup-author-20261030'
    terminal = read(parent/'followup-completed.json'); assert terminal['phase'] == 'completed'
    result = terminal['results']['diagnostic-author']; a = result['agent']
    meta, _, files = load(Path(a['generation_snapshots'][0]))
    inputs = {name[len('inputs/'):]:data for name,data in files.items() if name != 'inputs/worker-environment.json'}
    assert len(inputs) == 6 and sum(map(len,inputs.values())) == 75682
    probe = Path(result['probe']); plan = Path(result['plan'])
    assert sha(probe) == result['probe_sha256'] == '8ab9078e71390b240d8a471cc765084a58b9681d3170f6a8a4050a7221fb2ad4'
    assert sha(plan) == result['plan_sha256'] == 'c0dc33c969b45437517d2f2282138a650a4f62f2014c4b06b0dc992decf097c8'
    cpu_root = parent/'audit/independent-cpu-cap'; cpu = read(cpu_root/'result.json')
    assert not cpu['cuda_initialized'] and not cpu['late_failure']['failure_operands_or_outcome_saved']
    fixture = cpu_root/'fixture.py'
    assert sha(fixture) == '88c4be346306845accb5c802c8b886fd55107deb6a1ad4b74a40989f22d6e4f2'
    failure = {'scope':cpu['scope'],'probe_sha256':cpu['probe_sha256'], 'observed_late_failure':cpu['late_failure'],
        'fixture_source':{'path':str(fixture),'sha256':sha(fixture)},
        'recorded_result':{'path':str(cpu_root/'result.json'),'sha256':sha(cpu_root/'result.json')},
        'recorded_evidence':{'path':str(cpu_root/'late-failure-evidence.json'),'sha256':sha(cpu_root/'late-failure-evidence.json')},
        'limits':'Synthetic CPU comparison sequence using the exact generated probe and real torch comparator. No candidate/model/GPU run; not a numerical model finding.'}
    inputs.update({'previous-diagnostic.py':probe.read_bytes(),'previous-diagnostic-plan.json':plan.read_bytes(),
        'execution-failure.json':json.dumps(failure,indent=2)})
    frozen = read(parent/'protocol.json')['frozen_identities'].copy()
    for p in [probe,plan,parent/'followup-completed.json',Path(a['generation_snapshots'][0]),fixture,
              cpu_root/'result.json',cpu_root/'late-failure-evidence.json',cpu_root/'nonfinite-evidence.json']:
        frozen[str(p)] = sha(p)
    return inputs,frozen,failure


def main(args):
    os.umask(0o077); inputs,frozen,failure = build_inputs(args.work)
    assert all(sha(p)==h for p,h in frozen.items())
    sizes={k:len(v.encode() if isinstance(v,str) else v) for k,v in inputs.items()}
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'frozen_files':len(frozen),'input_sizes':sizes})); return
    args.output.mkdir(parents=True,exist_ok=True); assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'frozen_identities':frozen,
        'input_sizes':sizes,'execution_failure':failure,
        'scope':'One real diagnostic_author repair of CPU-demonstrated late-failure retention. Original six inputs unchanged, exact previous output plus concrete CPU evidence. No handwritten fix, candidate/GPU/model/search/acceptance or production adoption.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    config=load_config(args.config,[]);config.usage.mode='observe_only';results={}
    write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'diagnostic-author-repair'})
    try:
        directory=args.output/'diagnostic-author';store=RunStore.create(directory.parent,directory.name,{'kind':'setup_diagnostic_retention_repair'})
        with Runtime(project_agent_config(config),directory) as runtime:
            output,agent=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS+EXTRA+REPAIR,
                ('diagnostic.py','diagnostic-plan.json'),validator=validate)
        results['diagnostic-author']={'status':'completed','agent':agent,'probe':str(output/'diagnostic.py'),
            'probe_sha256':sha(output/'diagnostic.py'),'plan':str(output/'diagnostic-plan.json'),'plan_sha256':sha(output/'diagnostic-plan.json'),
            'validation':'Syntax/schema only; independent CPU boundary and actual code/tool audit required before any separate fixed observational workers.'}
    except Exception:results['diagnostic-author']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items());results['preservation']={'status':'completed' if same else 'error','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if all(x['status']=='completed'for x in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)
    print(json.dumps({'phase':terminal['phase'],'finished_at':terminal['finished_at']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','output','work']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
