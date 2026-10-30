"""One bounded project selection after a closed CPU-contract repair chain; no candidate/GPU stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import candidate_artifacts, route
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

CODE = '5e3a9a001ba4c793c602f2b066cb12789550cf09e900cd6518ba821deaefef26'


def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()


def prepare(args):
    work = args.parent.parent
    previous = work/'ddp-communication-contract-repair-20261030'
    base = work/'ddp-communication-selection-20261030'
    protocol = read(previous/'protocol.json')
    assert framework_source() == protocol['core']
    assert read(previous/'followup-completed.json')['phase'] == 'completed'
    closed = read(previous/'terminal-audit.json')
    assert closed['candidate_sha256'] == '77ac04e8eb86f5b7662d1a96e1152a7c59ea7c6120de5db7a8a2a9ee085b672e'
    prior = read(base/'followup-completed.json')['results']['selection']
    saved, prompt, files = load(Path(prior['initial_snapshot']))
    inputs = {n.removeprefix('inputs/'):v for n,v in files.items() if n not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    result = read(previous/'followup-completed.json')['results']['candidate']
    directory = Path(result['agent']['sandbox'])
    cpu = previous/'audit/cpu-contract/result.json'
    assert read(cpu)['candidate_sha256'] == result['candidate_sha256'] and read(cpu)['model_runs'] == read(cpu)['real_collectives'] == 0
    fixture = Path('/root/autodl-tmp/dev/opop-v6/scripts/experiments/v6_ddp_repaired_cpu_contract_20261030.py')
    assert sha(fixture) == read(cpu)['fixture_sha256']
    items = [directory/'candidate.py',directory/'proposal.json',cpu,fixture,previous/'audit/cpu-contract-execution.json']
    index = {'scope':'Exact generated code/proposal and completed synthetic CPU-contract observations. Not a search trial, historical NCCL fault or GPU quality result. The previous implementation-repair branch is closed; this is a new project-level evidence decision, not a request to edit that implementation. No algorithm, group, bucket, predicted gain or causal solution is prescribed.',
        'artifacts':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in items]}
    inputs.update({'decision-evidence/candidate.py':items[0].read_bytes(),'decision-evidence/proposal.json':items[1].read_bytes(),
        'decision-evidence/cpu-observations.json':cpu.read_bytes(),'decision-evidence/index.json':json.dumps(index,indent=2)})
    raw = (args.parent/'ddp-search/latest-feedback.json').read_text()
    original = json.loads(raw)
    assert sha(args.parent/'ddp-search/latest-feedback.json') == 'a0234920605970500c870b2ec98fc952b264a79f264bd19d6a53c99b796cf459'
    assert 'supplemental_evidence' not in original
    supplement = {'cpu_contract_observations':{'scope':index['scope'],'assistance':'Explicit developer CPU-fact assistance, not unaided original-entry cohort or same-snapshotAB.',
        'candidate_sha256':result['candidate_sha256'],'proposal_sha256':result['proposal_sha256'],
        'observations':read(cpu),'artifact_index':'inputs/decision-evidence/index.json'}}
    pos = raw.rfind('}'); augmented = raw[:pos]+',\n  "supplemental_evidence": '+json.dumps(supplement)+raw[pos:]
    assert {k:v for k,v in json.loads(augmented).items() if k!='supplemental_evidence'} == original
    source = ProjectSearch(load_config(args.parent/'private-config.yaml',[]),args.parent/'ddp-search')
    current = source.revisions.current(); assert current.id == protocol['parent']
    _, compact = feedback_artifacts(augmented)
    feedback = route(compact,current_revision=current.id,registry=source.revisions.load(current.registry_ref,GroupRegistry))
    oldfeedback=json.loads(files['inputs/previous-feedback.txt'])
    assert {k:v for k,v in json.loads(feedback).items() if k!='supplemental_evidence'} == {k:v for k,v in oldfeedback.items() if k!='supplemental_evidence'}
    frozen = protocol['frozen_identities'].copy()
    for p in [*items,previous/'terminal-audit.json',previous/'public-terminal-audit.json',previous/'followup-completed.json',previous/'candidate/events.jsonl',Path(result['initial_snapshot']),base/'terminal-audit.json',Path(prior['initial_snapshot']),args.parent/'ddp-search/project/current.json']:
        frozen[str(p)] = sha(p)
    assert all(sha(p)==h for p,h in frozen.items())
    return inputs, feedback, saved, prompt, files, frozen, Path(prior['initial_snapshot'])

def main(args):
    os.umask(0o077)
    inputs,feedback,saved,prompt,oldfiles,frozen,snapshot = prepare(args)
    config = project_agent_config(load_config(args.config, [])); config.usage.mode = 'observe_only'
    assert config.agents.default_model == saved['model'] and config.opencode.agent == saved['agent']
    expected = {'inputs/'+name:(data.encode() if isinstance(data,str) else data) for name,data in inputs.items()}
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    expected['inputs/worker-environment.json'] = oldfiles['inputs/worker-environment.json']
    details = {'core':framework_source(), 'frozen_files':len(frozen), 'original_snapshot':str(snapshot),
        'original_snapshot_sha256':sha(snapshot), 'inputs':len(expected), 'input_bytes':sum(map(len,expected.values())),
        'raw_feedback_bytes':(args.parent/'ddp-search/latest-feedback.json').stat().st_size,
        'raw_feedback_sha256':sha(args.parent/'ddp-search/latest-feedback.json'), 'original_feedback_bytes_unchanged':True,
        'feedback_bytes':len(feedback.encode()), 'feedback_sha256':digest(feedback.encode()),
        'historical_candidate_sha256':CODE, 'scope':saved['role'],
        'changed_inputs':sorted(k for k in oldfiles.keys() & expected.keys() if oldfiles[k] != expected[k]),
        'removed_inputs':sorted(oldfiles.keys()-expected.keys()),'added_inputs':sorted(expected.keys()-oldfiles.keys()),
        'explicit_developer_cpu_facts':True, 'new_task_or_single_variable_AB':False, 'no_gpu_stage':True,
        'parent':read(args.parent/'protocol.json')['parent'], 'source':str(args.parent/'ddp-search'),
        'prior_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'growth_forecast_bytes':256*1024**2, 'warning_free_bytes':8*1024**3}
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free >= 8*1024**3 + details['growth_forecast_bytes'], free
    details['free_bytes']=free
    if args.preflight:
        print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen,
        'scope':'User explicitly requested continued progress on2026-10-31. ONE existing selector from unchanged accepted current source/discovery. Prior37 initial files unchanged except feedback supplemental CPU-contract observations;4 exact code/proposal/result/index inputs added. Explicit developer CPU-fact assistance. Previous77ac repair chain closed, not reopened or executed. No candidate/search/GPU/collector/native/acceptance/pointer stage, manual group/bucket/algorithm/code, production prompt/model/tool changes. Review selection; only a genuinely justified new route may proceed in a separately declared step. Qwen closed/FSDP rejection preserved. No endless retry or waiting for a new external task by default.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    (args.output/'original-feedback.json').write_bytes((args.parent/'ddp-search/latest-feedback.json').read_bytes())
    (args.output/'routed-feedback.json').write_text(feedback)
    write_json(args.output/'evidence-index.json',json.loads(inputs['decision-evidence/index.json']))
    write_json(args.output/'transport-preflight.json',details)
    write_json(args.output/'preparation-reuse.json',{'source':details['source'],'parent':details['parent'],
        'request_sha256':sha(args.parent/'ddp-search/request.json'),'prepared_sha256':sha(args.parent/'ddp-search/prepared.json'),
        'discovery_sha256':sha(args.parent/'ddp-search/discovery.json'),
        'scope':'Original accepted files/bundle/current discovery from exact prior selector snapshot. Only new CPU evidence supplemental and4files. No branch/pointer mutation, source/model/control rerun or reference copies.'})
    write_json(args.output/'state.json',{'phase':'running','stage':'project-redecision-only','at':now()})
    results={}
    try:
        store = RunStore.create(args.output,'selection',{'kind':'ddp-communication-redecision'})
        source=ProjectSearch(config,args.parent/'ddp-search')
        discovery=source.current_discovery()
        policy=json.loads(inputs['search.json'])
        eligible=eligible_groups(discovery,leaf_only=False)
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt = ProjectAgents(runtime,store).generate('selection', inputs,
                Path(__import__('kernel_optimizer.project.agents',fromlist=['__file__']).__file__).with_name('selection_prompt.md').read_text(),
                ('selection.json',),feedback,validator=lambda path: check_selection(path,discovery,policy['allowed_scopes'],eligible))
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            actual,actual_prompt,actual_files=load(path)
            if actual['fresh_session']:
                assert actual_files == expected and actual_prompt == prompt
                assert actual['framework_source'] == saved['framework_source'] and actual['tool_model_settings'] == saved['tool_model_settings']
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['selection']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'inputs_verified':True,'prompt_model_tools_core_unchanged':True,
            'selection_sha256':sha(directory/'selection.json'),
            'selected_scope':read(directory/'selection.json')['scope'],
            'scope':'Selection/schema only. Full actual tools and evidence/assumptions need audit before any candidate or GPU stage.'}
    except Exception:
        results['selection']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items())
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['selection']['status']=='completed' else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
