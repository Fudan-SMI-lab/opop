"""One isolated ordinary selection from exact rejected accepted-version feedback; no candidate or GPU stage."""
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
    source = args.parent/'ddp-search'
    protocol = read(args.parent/'protocol.json')
    terminal = read(args.parent/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    search = ProjectSearch(load_config(args.parent/'private-config.yaml', []), source)
    current = search.revisions.current()
    assert current.id == protocol['parent'] and len(current.bundles)==1
    assert read(args.parent/'terminal-audit.json')['all_frozen_unchanged']
    trial = read(source/'trials/trial-02ed5436b4/result.json')
    assert not trial['decision']['accepted'] and trial['decision']['reasons']==['paired_gain_not_above_noise','objective_not_improved']
    snapshot = Path(trial['selection_agent']['generation_snapshots'][0])
    saved, prompt, files = load(snapshot)
    assert saved['fresh_session'] and saved['role'] == 'selection'
    raw = (source/'latest-feedback.json').read_text()
    assert json.loads(raw) == trial
    assert (source/'latest-feedback.json').read_bytes()==(source/'trials/trial-02ed5436b4/result.json').read_bytes()
    observations, compact = feedback_artifacts(raw)
    historical = candidate_artifacts(raw, search.store)
    assert digest(historical['previous-candidate/candidate.py']) == CODE
    feedback = route(compact, current_revision=current.id,
        registry=search.revisions.load(current.registry_ref, GroupRegistry))
    assert json.loads(feedback)['previous_trial'] == 'trial-02ed5436b4'
    assert json.loads(feedback)['supplemental_evidence'] is None
    inputs = {name.removeprefix('inputs/'):data for name,data in files.items()
              if name not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']
              and not name.startswith(('inputs/previous-candidate/','inputs/observations/previous-'))}
    inputs.update(observations); inputs.update(historical)
    frozen = protocol['frozen_identities'].copy()
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and any(part in ['jobs','trials','generation-snapshots'] for part in path.relative_to(source).parts):
            frozen[str(path)] = sha(path)
    for path in list((source/'sandboxes').glob('*/candidate.py'))+list((source/'sandboxes').glob('*/proposal.json'))+list((source/'sandboxes').glob('*/selection.json'))+list((source/'sandboxes').glob('*/collection.json'))+[source/'request.json',source/'project/current.json',source/'events.jsonl',source/'latest-feedback.json', source/'prepared.json', source/'discovery.json', snapshot,
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
        'raw_feedback_bytes':(args.parent/'ddp-search/latest-feedback.json').stat().st_size,
        'raw_feedback_sha256':sha(args.parent/'ddp-search/latest-feedback.json'), 'added_feedback_bytes':0,
        'feedback_bytes':len(feedback.encode()), 'feedback_sha256':digest(feedback.encode()),
        'historical_candidate_sha256':CODE, 'scope':saved['role'],
        'changed_inputs':sorted(k for k in oldfiles.keys() & expected.keys() if oldfiles[k] != expected[k]),
        'removed_inputs':sorted(oldfiles.keys()-expected.keys()),'added_inputs':sorted(expected.keys()-oldfiles.keys()),
        'no_supplemental_evidence':True, 'no_gpu_stage':True,
        'parent':read(args.parent/'protocol.json')['parent'], 'source':str(args.parent/'ddp-search'),
        'prior_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'growth_forecast_bytes':384*1024**2, 'warning_free_bytes':8*1024**3}
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
        'scope':'One isolated existing ordinary selector on exact accepted current code/discovery with the newly rejected original framework feedback and historical5e3a code/actual collectors. No developer audit/cause/repair, group/bucket/algorithm selection, production prompt/model/tool changes, candidate generation, search, GPU/controller experiment, new collector or acceptance. API observe-only. Existing source/mapping assistance remains; not unaided or same-snapshot AB. Review real selection evidence before any later implementation. This bounded decision stage is not a reviewer/prompt retry or automatic candidate chain; Qwen closed/FSDP rejection preserved.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    (args.output/'original-feedback.json').write_bytes((args.parent/'ddp-search/latest-feedback.json').read_bytes())
    write_json(args.output/'transport-preflight.json',details)
    write_json(args.output/'preparation-reuse.json',{'source':details['source'],'parent':details['parent'],
        'request_sha256':sha(args.parent/'ddp-search/request.json'),'prepared_sha256':sha(args.parent/'ddp-search/prepared.json'),
        'discovery_sha256':sha(args.parent/'ddp-search/discovery.json'),
        'scope':'Original accepted files/bundle/current discovery from exact initial selector snapshot. No branch/pointer mutation, source/model/control rerun or reference copies.'})
    write_json(args.output/'state.json',{'phase':'running','stage':'selection-only','at':now()})
    results={}
    try:
        store = RunStore.create(args.output,'selection',{'kind':'ddp-communication-selection'})
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
