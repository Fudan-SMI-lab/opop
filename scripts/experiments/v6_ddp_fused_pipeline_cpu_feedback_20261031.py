"""One CPU-fact-assisted ordinary candidate correction; no hand repair or GPU stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generated import validate_proposal
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
def digest(data): return hashlib.sha256(data.encode() if isinstance(data,str) else data).hexdigest()


def assisted_feedback(raw, cpu):
    value=json.loads(raw)
    assert 'supplemental_evidence' not in value
    value['supplemental_evidence']={'cpu_contract_observations':cpu,
        'evidence_index':'inputs/cpu-contract-evidence/index.json',
        'scope':'Explicit developer CPU-fact assistance for the immediately preceding generated candidate. Conditional synthetic stream/Work example and small CPU autograd dependency example, not original GPU failure/fullquality/optimization code. One bounded candidate correction; no supplied repair algorithm or implementation.'}
    return json.dumps(value,indent=2)


def prepare(args):
    p=read(args.parent/'protocol.json');terminal=read(args.parent/'followup-completed.json')
    assert terminal['phase']=='completed' and framework_source()==p['core']
    assert sha(args.parent/'terminal-audit.json')=='87cddc2c69cb2287cafd9b566a18badb9eeec0883fc6b01c47c925f108cdc137'
    result=terminal['results']['candidate'];box=Path(result['agent']['sandbox'])
    oldmeta,oldprompt,files=load(Path(result['initial_snapshot']))
    assert sha(box/'candidate.py')=='eef828816dd138cb69b0d12c960012b1970a0724b426a258ec6f8fcb6900abd7'
    assert sha(box/'proposal.json')=='0f0a977afdfcb94e9a46f9c950bc4efc587ac72e16cec9fbfed750255234da94'
    source=ProjectSearch(load_config(args.parent/'private-config.yaml',[]),Path(p['source']));current=source.revisions.current()
    assert current.id==p['parent'] and len(current.bundles)==1
    bundle=source.revisions.bundle(current.bundles[0]);assert bundle.parameters=={}
    assert {f.path:f.sha256 for f in current.files}[bundle.installer]==p['source_code_sha256']
    selection=json.loads(files['inputs/selection.json']);cpu_root=args.parent/'audit/cpu-stream-consumer'
    result_path=cpu_root/'result/result.json';cpu=read(result_path)
    assert sha(result_path)=='0c40b6ef9f05bdf6f19525c134142cfb124875bf0f120309e25aa6fb3fa0d477'
    assert sha(cpu_root/'fixture.py')==cpu['fixture_sha256']=='42a253121a48d097ebf74478d6b41baf432b114b5f8e19c0cc45b671c114a70a'
    assert read(cpu_root/'execution.json')['returncode']==0
    index={'scope':'Exact preceding generated candidate and once-recorded CPU interface/dependency observations. Original selection/task/current source remain unchanged. Produce one candidate using original ordinary rules and the evidence; no manual implementation is provided. The synthetic scheduler, mock graph/Work and small CPU model are fixtures only, never transplant into candidate or call them original GPU/quality evidence. No automatic GPU execution or further repair-until-pass.',
        'previous_candidate_sha256':sha(box/'candidate.py'),
        'artifacts':{name:{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}for name,path in {
            'previous_candidate':box/'candidate.py','previous_proposal':box/'proposal.json','cpu_result':result_path,
            'executed_fixture':cpu_root/'fixture.py','execution':cpu_root/'execution.json','installed_api_source':cpu_root/'result/installed-all-reduce-source.txt'}.items()}}
    inputs={n.removeprefix('inputs/'):v for n,v in files.items()if n not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    inputs.update({'cpu-contract-evidence/candidate.py':(box/'candidate.py').read_bytes(),
        'cpu-contract-evidence/proposal.json':(box/'proposal.json').read_bytes(),
        'cpu-contract-evidence/cpu-observations.json':result_path.read_bytes(),
        'cpu-contract-evidence/index.json':json.dumps(index,indent=2)})
    raw=(args.parent/'original-feedback.json').read_text();assert digest(raw)==p['original_feedback_sha256']
    assert raw.encode()==(source.output/'latest-feedback.json').read_bytes()
    augmented=assisted_feedback(raw,cpu);packaged,compact=feedback_artifacts(augmented)
    assert all(inputs[n]==(v.encode()if isinstance(v,str)else v)for n,v in packaged.items())
    feedback=route(compact,current_revision=current.id,registry=source.revisions.load(current.registry_ref,GroupRegistry),selected_groups=selection['groups'],plan=source.revisions.load(current.plan_ref,ExecutionPlan))
    routed=json.loads(feedback);previous=json.loads(files['inputs/previous-feedback.txt'])
    assert {k:v for k,v in routed.items()if k!='supplemental_evidence'}=={k:v for k,v in previous.items()if k!='supplemental_evidence'}
    assert routed['supplemental_evidence']['cpu_contract_observations']==cpu and routed['decision']['accepted']
    frozen=p['frozen_identities'].copy()
    names=['protocol.json','followup-completed.json','followup-results.json','state.json','original-feedback.json','routed-feedback.json','transport-preflight.json','preparation-reuse.json','startup-verification.json','startup-tool-review.json','terminal-audit.json','public-terminal-audit.json']
    paths=[args.parent/n for n in names]+[box/'candidate.py',box/'proposal.json',Path(result['initial_snapshot']),args.parent/'candidate/events.jsonl']
    paths += [Path(v['path'])for v in index['artifacts'].values()]
    for path in paths:frozen[str(path)]=sha(path)
    assert all(sha(path)==h for path,h in frozen.items())
    details={'source':p['source'],'parent':current.id,'selection_snapshot':p['selection_snapshot'],'selection_sha256':p['selection_sha256'],
        'canonical_selection_sha256':p['canonical_selection_sha256'],'selection_semantics_unchanged':True,'selected_scope':selection['scope'],
        'selected_groups':selection['groups'],'affected_calls':len(selection['affected_calls']),
        'original_feedback_sha256':digest(raw),'original_feedback_bytes':len(raw.encode()),'augmented_feedback_sha256':digest(augmented),
        'feedback_sha256':digest(feedback),'feedback_bytes':len(feedback.encode()),'supplemental_evidence':'Exact new CPU observations plus index/scope only; original routed fields unchanged',
        'cpu_result_sha256':sha(result_path),'executed_fixture_sha256':sha(cpu_root/'fixture.py'),
        'previous_generated_candidate_sha256':sha(box/'candidate.py'),'previous_generated_proposal_sha256':sha(box/'proposal.json'),
        'prior_candidate_snapshot':result['initial_snapshot'],'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'source_code_sha256':p['source_code_sha256'],'original_input_files':len(files),'original_nonfeedback_inputs_byte_equal':True,
        'input_hashes':{'inputs/'+n:digest(v)for n,v in inputs.items()},
        'request_sha256':sha(source.output/'request.json'),'prepared_sha256':sha(source.output/'prepared.json'),'discovery_sha256':sha(source.output/'discovery.json')}
    return inputs,feedback,oldmeta,oldprompt,files,files,frozen,details


def main(args):
    os.umask(0o077)
    inputs,feedback,oldmeta,oldprompt,oldfiles,selector_files,frozen,details = prepare(args)
    config = project_agent_config(load_config(args.config, [])); config.usage.mode='observe_only'
    assert config.agents.default_model == oldmeta['model'] and config.opencode.agent == oldmeta['agent']
    expected = {'inputs/'+n:v.encode() if isinstance(v,str) else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode()
    expected['inputs/worker-environment.json']=selector_files['inputs/worker-environment.json']
    details.update({'core':framework_source(),'frozen_files':len(frozen),'inputs':len(expected),
        'input_bytes':sum(map(len,expected.values())), 'growth_forecast_bytes':384*1024**2,'warning_free_bytes':8*1024**3,
        'old_candidate_changed_inputs':sorted(k for k in expected.keys()&oldfiles.keys() if expected[k]!=oldfiles[k]),
        'old_candidate_added_inputs':sorted(expected.keys()-oldfiles.keys()),'old_candidate_removed_inputs':sorted(oldfiles.keys()-expected.keys())})
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=details['warning_free_bytes']+details['growth_forecast_bytes'],free
    details['free_bytes']=free
    if args.preflight:
        print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen,
        'scope':'ONE explicit developer CPU-fact-assisted candidate correction for exact preceding eef828. Original33 nonfeedback inputs and all original routed fields remain unchanged; append exact recorded CPU observations/scope/index and four previous-code/proposal/result/index inputs. Same ordinary production prompt plus existingfeedbacksuffix/model/tools, no experiment repair paragraph or manual code/group/algorithm/terminalaudit/cause table. API observe-only/no caps; normal schema repairs retained. No selector/model/GPU/controller collective/search/parameters/collector/sourcecontrols/referencecopy/pointer/acceptance stage. Inspect all actual tools and exact output before later execution. Conditional CPU evidence is not historical NCCL/GPU numerical rejection. One bounded correction, not repeated until pass or reopening77ac/observer/Qwen/FSDP. Existing source/metric/CPU assistance persists, not unaided or same-snapshot AB.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output/'transport-preflight.json',details)
    write_json(args.output/'preparation-reuse.json',{k:details[k] for k in ['source','parent','request_sha256','prepared_sha256','discovery_sha256','selection_sha256','canonical_selection_sha256','original_feedback_sha256','feedback_sha256']})
    (args.output/'original-feedback.json').write_bytes((args.parent/'original-feedback.json').read_bytes())
    (args.output/'routed-feedback.json').write_text(feedback)
    (args.output/'evidence-index.json').write_text(inputs['cpu-contract-evidence/index.json'])
    augmented=assisted_feedback((args.parent/'original-feedback.json').read_text(),json.loads(inputs['cpu-contract-evidence/cpu-observations.json']))
    assert digest(augmented)==details['augmented_feedback_sha256']
    (args.output/'diagnostic-feedback.json').write_text(augmented)
    write_json(args.output/'state.json',{'phase':'running','stage':'candidate-only','at':now()})
    results={}
    try:
        store=RunStore.create(args.output,'candidate',{'kind':'ddp-fused-pipeline-cpu-feedback'})
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt=ProjectAgents(runtime,store).generate('candidate',inputs,
                Path(__import__('kernel_optimizer.project.agents',fromlist=['__file__']).__file__).with_name('candidate_prompt.md').read_text(),
                ('candidate.py','proposal.json'),feedback,validator=validate_proposal)
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            actual,prompt,actual_files=load(path)
            if actual['fresh_session']:
                assert actual_files==expected and prompt==oldprompt
                assert actual['framework_source']==oldmeta['framework_source'] and actual['tool_model_settings']==oldmeta['tool_model_settings']
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['candidate']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'inputs_verified':True,'prompt_model_tools_core_unchanged':True,
            'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
            'parameter_space':read(directory/'proposal.json').get('parameter_space'),
            'scope':'Generation/schema only; actual code/tools/parameters require audit, no execution/quality/acceptance claim.'}
    except Exception:
        results['candidate']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(path)==h for path,h in frozen.items())
    same=same and ProjectSearch(config,Path(details['source'])).revisions.current().id==details['parent']
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['candidate']['status']=='completed' else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
