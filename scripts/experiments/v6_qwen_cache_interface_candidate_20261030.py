"""One existing candidate role from a new execution failure; generation only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.feedback import candidate_artifacts, route
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

CODE = '6ec4e97a4bd39a7d16c82109b49b841f0258c039d56de15eeb6f80696e4906fa'


def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()


def prepare(args):
    source = args.parent/'qwen-continuation'
    protocol = read(args.parent/'protocol.json')
    terminal = read(args.parent/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    search = ProjectSearch(load_config(args.parent/'private-config.yaml', []), source)
    current = search.revisions.current()
    assert current.id == protocol['parent_revision'] and not current.bundles
    trial = read(source/'trials/trial-2a7d20b4d9/result.json')
    assert trial['decision'] == {'accepted':False,'reasons':['no_valid_parameter_candidate']}
    snapshot = Path(trial['candidate_agent']['generation_snapshots'][0])
    saved, prompt, files = load(snapshot)
    assert saved['fresh_session'] and saved['role'] == 'candidate'
    raw = (source/'latest-feedback.json').read_text()
    assert json.loads(raw) == trial
    observations, compact = feedback_artifacts(raw)
    historical = candidate_artifacts(raw, search.store)
    assert digest(historical['previous-candidate/candidate.py']) == CODE
    feedback = route(compact, current_revision=current.id,
        registry=search.revisions.load(current.registry_ref, GroupRegistry),
        selected_groups=trial['selection']['groups'],
        plan=search.revisions.load(current.plan_ref, ExecutionPlan))
    assert json.loads(feedback)['previous_trial'] == 'trial-2a7d20b4d9'
    assert json.loads(feedback)['supplemental_evidence'] is None
    inputs = {name.removeprefix('inputs/'):data for name,data in files.items()
              if name not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']
              and not name.startswith(('inputs/previous-candidate/','inputs/observations/previous-'))}
    inputs.update(observations); inputs.update(historical)
    frozen = protocol['frozen_identities'].copy()
    for path in source.rglob('*'):
        if path.is_file() and any(part in ['jobs','trials','generation-snapshots'] for part in path.relative_to(source).parts):
            frozen[str(path)] = sha(path)
    for path in [source/'latest-feedback.json', source/'prepared.json', source/'discovery.json', snapshot,
                 args.parent/'terminal-audit.json', args.parent/'public-terminal-audit.json']:
        frozen[str(path)] = sha(path)
    assert all(sha(p) == h for p,h in frozen.items())
    return inputs, feedback, saved, prompt, files, frozen, snapshot


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
        'feedback_bytes':len(feedback.encode()), 'feedback_sha256':digest(feedback.encode()),
        'historical_candidate_sha256':CODE, 'scope':saved['role'],
        'changed_inputs':sorted(k for k in oldfiles.keys() & expected.keys() if oldfiles[k] != expected[k]),
        'removed_inputs':sorted(oldfiles.keys()-expected.keys()),'added_inputs':sorted(expected.keys()-oldfiles.keys()),
        'no_supplemental_evidence':True, 'no_gpu_stage':True}
    if args.preflight:
        print(json.dumps({'status':'passed',**details}));return
    args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**details,'frozen_identities':frozen,
        'scope':'One new ordinary candidate-role generation on the exact prior selected scope, with existing routed framework failure/collector outputs and historical6ec4 code. Original task/current discovery/selection/prompt/model/tools stay; previous feedback/code/collector packets advance to the new failed trial. No new selector/search/GPU worker/parameter execution/acceptance/pointer publication. No developer CPU binding result, cause table or repair implementation supplied. Earlier assistance remains; not unaided or same-snapshot A/B. Generated code must be reviewed before any later execution.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output/'state.json',{'phase':'running','stage':'candidate','at':now()})
    results={}
    try:
        store = RunStore.create(args.output,'candidate',{'kind':'qwen-cache-interface-candidate'})
        with Runtime(config,log_dir=store.run_dir) as runtime:
            directory,receipt = ProjectAgents(runtime,store).generate('candidate', inputs,
                Path(__import__('kernel_optimizer.project.agents',fromlist=['__file__']).__file__).with_name('candidate_prompt.md').read_text(),
                ('candidate.py','proposal.json'),feedback,validator=validate_proposal)
        snapshots=[]
        for path in (store.run_dir/'generation-snapshots').glob('*.json'):
            actual,actual_prompt,actual_files=load(path)
            if actual['fresh_session']:
                assert actual_files == expected and actual_prompt == prompt
                assert actual['framework_source'] == saved['framework_source'] and actual['tool_model_settings'] == saved['tool_model_settings']
                snapshots.append(str(path))
        assert len(snapshots)==1
        results['candidate']={'status':'completed','agent':receipt,'initial_snapshot':snapshots[0],
            'inputs_verified':True,'prompt_model_tools_core_unchanged':True,
            'candidate_sha256':sha(directory/'candidate.py'),'proposal_sha256':sha(directory/'proposal.json'),
            'parameter_space':read(directory/'proposal.json').get('parameter_space'),
            'scope':'Generation/schema only. Actual tools/code/parameters require audit; no execution or quality claim.'}
    except Exception:
        results['candidate']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items())
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['candidate']['status']=='completed' else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
