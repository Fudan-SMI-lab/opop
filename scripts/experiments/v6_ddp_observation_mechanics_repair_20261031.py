"""One CPU-fact-assisted diagnostic-author repair; generation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
from scripts.experiments.v6_diagnostic_author_20261029 import now, validate
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_communication_observation_author_20261031 import INSTRUCTIONS

REPAIR = """
This is one bounded repair of the previous diagnostic's observation contract,
not an optimization candidate or an iterative prompt study. Original inputs are
unchanged. Read inputs/repair/previous-diagnostic.py, previous-plan.json,
cpu-observations.json and index.json. The CPU record was acquired once from the
exact previous bytes with synthetic events and a Python entry, without a model,
CUDA, process group or real collective. It establishes conditional reporting and
restoration counterexamples, not historical GPU/NCCL failures or a causal answer.
Use those actual observations to reassess evidence joins, invocation/phase scope,
bounded retention and original-entry/return/exception/restoration preservation.
Unknown or unavailable evidence is an acceptable result. Choose the smallest
appropriate implementation yourself; no repair code or collector choice is given.
Preserve the original evidence question, evaluator authority and execution count.
Only metadata/API inspection and inert synthetic Python mocks are permitted in
this generation: do not instantiate/train a model, initialize any process group,
invoke a real collective (including world_size=1 Gloo), or initialize/use CUDA.
Retain all failed checks. Write the two requested diagnostic outputs once this
bounded task is complete. Generated code will be reviewed independently; no GPU
execution, further repair, performance acceptance or production adoption follows
automatically.
"""


def expected_prompt(body, environment):
    return ('Work on the supplied project artifacts. Inputs and framework code are fixed. '
            'Write only the requested output files in this directory; do not modify external source, '
            'provider configuration, quality thresholds or workload. The framework will execute and '
            'validate the output independently.\n' + body +
            ' Read inputs/worker-environment.json before library/API inspection. '
            f"The measured worker uses Python {environment['python']}; use that executable for CPU checks. "
            'Other system/Conda environments may have different library versions. GPU work remains in leased framework workers.')


def prepare(work):
    parent = work/'ddp-communication-observation-author-20261031'
    audit = read(parent/'terminal-audit.json')
    assert audit['real_author_finished'] and audit['all_frozen_unchanged']
    assert audit['probe_sha256'] == '024441f7d80bddaa51590b3630fbcc3969ac31b05548546ad2089106a10b5e98'
    meta, prompt, files = load(Path(audit['snapshot']))
    inputs = {k.removeprefix('inputs/'): v for k,v in files.items() if k != 'inputs/worker-environment.json'}
    assert prompt == expected_prompt(INSTRUCTIONS, json.loads(files['inputs/worker-environment.json']))
    frozen = read(parent/'protocol.json')['frozen_identities'].copy()
    def item(path):
        path=Path(path); h=sha(path); frozen[str(path)]=h
        return {'path':str(path),'sha256':h,'bytes':path.stat().st_size}
    sandbox = Path(meta['sandbox']); cpu = parent/'audit/independent-cpu-mechanics'
    index={'scope':'One diagnostic mechanics repair, explicit developer CPU-fact assistance. Exact prior probe/plan and one recorded synthetic CPU fixture result only; not a historical GPU failure, communication optimization/cause answer, manual implementation, or production change. Original question/source/current remain unchanged. No terminal audit supplied.',
           'probe':item(sandbox/'diagnostic.py'),'plan':item(sandbox/'diagnostic-plan.json'),
           'cpu_result':item(cpu/'result.json'),'fixture':item(cpu/'fixture.py'),
           'execution':item(cpu/'execution.json')}
    inputs.update({'repair/previous-diagnostic.py':(sandbox/'diagnostic.py').read_bytes(),
                   'repair/previous-plan.json':(sandbox/'diagnostic-plan.json').read_bytes(),
                   'repair/cpu-observations.json':(cpu/'result.json').read_bytes(),
                   'repair/index.json':json.dumps(index,indent=2)})
    for name in ['protocol.json','followup-completed.json','followup-results.json','terminal-audit.json',
                 'public-terminal-audit.json','startup-verification.json','preparation-reuse.json',
                 'diagnostic-author/events.jsonl','terminal-private-history/messages.json',
                 'terminal-private-history/tools.json','terminal-private-history/complete-access-index.json']:
        item(parent/name)
    item(Path(audit['snapshot']))
    assert all(sha(path)==h for path,h in frozen.items())
    return inputs, frozen, index, meta, files, audit


def main(args):
    os.umask(0o077)
    inputs, frozen, index, oldmeta, oldfiles, audit = prepare(args.work)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert not gpu.strip() and free>=8*1024**3+128*1024**2
    record={'status':'passed','core':framework_source(),'frozen_files':len(frozen),
            'input_sizes':{k:len(v.encode() if isinstance(v,str) else v) for k,v in inputs.items()},
            'source_current':audit['source_current'],'previous_snapshot':audit['snapshot'],
            'original_initial_files':len(oldfiles),'growth_forecast_bytes':128*1024**2,
            'warning_free_bytes':8*1024**3,'free_bytes':free,
            'repair_paragraph_sha256':hashlib.sha256(REPAIR.encode()).hexdigest()}
    if args.preflight: print(json.dumps(record)); return
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{**record,'at':now(),'scope':'One diagnostic_author mechanics repair using exact developer CPU evidence. No optimization candidate/77ac repair/Qwen/FSDP restart or automatic execution/acceptance. Original inputs remain byte-equal; production unchanged. A new failure is preserved, no automatic further repair.',
        'frozen_identities':frozen,'evidence_index':index})
    write_json(args.output/'evidence-index.json',index)
    write_json(args.output/'preparation-reuse.json',read(args.work/'ddp-communication-observation-author-20261031/preparation-reuse.json'))
    config=load_config(args.config,[]); config.usage.mode='observe_only'; results={}
    write_json(args.output/'state.json',{'phase':'running','stage':'diagnostic-author','at':now()})
    try:
        directory=args.output/'diagnostic-author'
        store=RunStore.create(directory.parent,directory.name,{'kind':'ddp_observation_mechanics_repair'})
        with Runtime(project_agent_config(config),directory) as runtime:
            output,receipt=ProjectAgents(runtime,store).generate('diagnostic_author',inputs,INSTRUCTIONS+REPAIR,
                ('diagnostic.py','diagnostic-plan.json'),validator=validate)
        expected={'inputs/'+k:v.encode() if isinstance(v,str) else v for k,v in inputs.items()}
        snapshots=[]
        for path in (directory/'generation-snapshots').glob('*.json'):
            meta,prompt,actual=load(path)
            if not meta['fresh_session']: continue
            assert set(actual)==set(expected)|{'inputs/worker-environment.json'}
            assert all(actual[n]==v for n,v in expected.items())
            assert all(actual[n]==v for n,v in oldfiles.items())
            assert prompt==expected_prompt(INSTRUCTIONS+REPAIR,json.loads(actual['inputs/worker-environment.json']))
            assert meta['framework_source']==record['core']
            assert meta['model']==oldmeta['model'] and meta['tool_model_settings']==oldmeta['tool_model_settings']
            snapshots.append(str(path))
        assert len(snapshots)==1
        results['diagnostic-author']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'inputs_verified':True,'probe':str(output/'diagnostic.py'),'probe_sha256':sha(output/'diagnostic.py'),
            'plan':str(output/'diagnostic-plan.json'),'plan_sha256':sha(output/'diagnostic-plan.json'),
            'validation':'Schema/syntax only, not correct measurement or original full GPU quality. No automatic execution.'}
    except Exception:
        results['diagnostic-author']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(path)==h for path,h in frozen.items())
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['diagnostic-author']['status']=='completed' else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','output','work'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
