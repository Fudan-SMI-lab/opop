"""One concrete CPU-interface-failure repair through existing candidate role only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

REPAIR = """
Bounded implementation repair: the supplied prior generated candidate/proposal and CPU interface observations document concrete contract failures. Review their exact bytes and the named CPU fixture and limits. Address these failures while preserving the supplied selection scope, original complete workload, quality, constraints and framework interfaces. The CPU observations are developer-assisted evidence, not GPU/NCCL/model quality or a proposed implementation. Choose implementation changes yourself; do not infer a required algorithm or measured benefit. Preserve truthful limitations and failure behavior. Do not execute a GPU/model/real collective probe in this generation; bounded non-model CPU interface tests are permitted. Output candidate.py and proposal.json under the ordinary contract. No automatic execution or acceptance follows generation.
"""


def now(): return datetime.now(timezone.utc).isoformat()
def digest(v): return hashlib.sha256(v.encode() if isinstance(v,str) else v).hexdigest()


def prepare(args):
    p=read(args.parent/'protocol.json');terminal=read(args.parent/'followup-completed.json')
    assert terminal['phase']=='completed' and framework_source()==p['core']
    audit=read(args.parent/'terminal-audit.json')
    assert audit['all_frozen_unchanged'] and audit['candidate_sha256']=='890659f73c7c88e704e21682afb584b548dd181447307ef18c4bcfefa54f150f'
    result=terminal['results']['candidate'];meta,prompt,files=load(Path(result['initial_snapshot']))
    sandbox=Path(meta['sandbox']);feedback=files['inputs/previous-feedback.txt'].decode()
    inputs={n.removeprefix('inputs/'):v for n,v in files.items()
            if n not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    cpu=args.parent/'audit/cpu-contract/result.json'
    assert read(cpu)['candidate_sha256']==sha(sandbox/'candidate.py')
    fixture=Path(__file__).with_name('v6_ddp_candidate_cpu_contract_20261030.py')
    assert sha(fixture)==read(cpu)['fixture_sha256']
    evidence=[cpu,fixture,args.parent/'audit/cpu-contract-execution.json']
    index={'scope':'Exact CPU interface/failure observations and fixture. No CUDA/model/real collective or full quality execution. Synthetic equal-peer SUM and injected failures are explicitly bounded; no historical NCCL failure claimed.',
           'artifacts':[{'path':str(x),'sha256':sha(x),'bytes':x.stat().st_size} for x in evidence]}
    inputs.update({'repair/previous-candidate.py':(sandbox/'candidate.py').read_bytes(),
        'repair/previous-proposal.json':(sandbox/'proposal.json').read_bytes(),
        'repair/cpu-observations.json':cpu.read_bytes(),
        'repair/evidence-index.json':json.dumps(index,indent=2)})
    frozen=p['frozen_identities'].copy()
    paths=[args.parent/n for n in ['protocol.json','state.json','followup-completed.json','followup-results.json',
        'terminal-audit.json','public-terminal-audit.json','startup-verification.json','candidate/events.jsonl']]
    paths += [sandbox/'candidate.py',sandbox/'proposal.json',*evidence,
              *(Path(x) for x in result['agent']['generation_snapshots'])]
    for path in paths:frozen[str(path)]=sha(path)
    assert all(sha(path)==h for path,h in frozen.items())
    expected={'inputs/'+n:v.encode() if isinstance(v,str) else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode()
    expected['inputs/worker-environment.json']=files['inputs/worker-environment.json']
    assert all(expected[n]==v for n,v in files.items())
    detail={'source':p['source'],'parent':p['parent'],'core':p['core'],
        'previous_snapshot':result['initial_snapshot'],'previous_candidate_sha256':sha(sandbox/'candidate.py'),
        'previous_proposal_sha256':sha(sandbox/'proposal.json'),'cpu_result_sha256':sha(cpu),
        'original_input_files':len(files),'original_inputs_byteequal':True,'inputs':len(expected),
        'input_bytes':sum(map(len,expected.values())),'added_inputs':sorted(set(expected)-set(files)),
        'feedback_sha256':digest(feedback),'feedback_unchanged':True,'repair_paragraph_sha256':digest(REPAIR),
        'repair_paragraph_bytes':len(REPAIR.encode()),'frozen_files':len(frozen),
        'request_sha256':p['request_sha256'],'prepared_sha256':p['prepared_sha256'],'discovery_sha256':p['discovery_sha256'],
        'growth_forecast_bytes':256*1024**2,'warning_free_bytes':8*1024**3,
        'forecast_basis':'Prior generation launch-to-terminal usage grew about86MB including32MB inputs; this role adds about35KB, no worker/control/reference/native stage.256MiB is a forecast, not API/output cap.'}
    return inputs,feedback,meta,prompt,expected,frozen,detail


def main(args):
    os.umask(0o077)
    inputs,feedback,oldmeta,oldprompt,expected,frozen,detail=prepare(args)
    config=project_agent_config(load_config(args.config,[]));config.usage.mode='observe_only'
    assert config.agents.default_model==oldmeta['model'] and config.opencode.agent==oldmeta['agent']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=detail['warning_free_bytes']+detail['growth_forecast_bytes'],free
    detail['free_bytes']=free
    if args.preflight:print(json.dumps({'status':'passed',**detail}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**detail,'frozen_identities':frozen,
        'scope':'ONE existing candidate generation for concrete CPU-proven interface failures. Original40 inputs unchanged plus exact prior output code/proposal and CPU result/index. Explicit developer CPU-fact assistance, not ordinary unassisted cohort/same-prompt AB. Same production core/prompt file/model/tools; narrow experiment repair paragraph only. No hand implementation, group/bucket/algorithm/collector selection, GPU/model/real collective probes, parameter/search/acceptance/pointer stage. No automatic execution; preserve new failures and review exact output. Qwen closed/FSDP rejection preserved.'})
    write_json(args.output/'transport-preflight.json',detail)
    write_json(args.output/'preparation-reuse.json',{k:detail[k] for k in ['source','parent','request_sha256','prepared_sha256','discovery_sha256','feedback_sha256']})
    write_json(args.output/'state.json',{'phase':'running','stage':'candidate-contract-repair','at':now()})
    results={}
    try:
        store=RunStore.create(args.output,'candidate',{'kind':'ddp-communication-contract-repair'})
        prompt_path=Path(__import__('kernel_optimizer.project.agents',fromlist=['__file__']).__file__).with_name('candidate_prompt.md')
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('candidate',inputs,prompt_path.read_text()+REPAIR,
                ('candidate.py','proposal.json'),feedback,validator=validate_proposal)
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            actual,prompt,files=load(path)
            if actual['fresh_session']:
                assert files==expected and actual['framework_source']==oldmeta['framework_source']
                assert actual['model']==oldmeta['model'] and actual['tool_model_settings']==oldmeta['tool_model_settings']
                assert REPAIR in prompt
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['candidate']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
            'parameter_space':read(directory/'proposal.json').get('parameter_space'),'scope':'Generation only; review before any execution.'}
    except Exception:results['candidate']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(path)==h for path,h in frozen.items())
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['candidate']['status']=='completed' else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal)
    write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
