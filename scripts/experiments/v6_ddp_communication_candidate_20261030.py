"""One ordinary candidate role after an isolated selector; no controller execution stage."""
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


def prepare(args):
    p = read(args.parent/'protocol.json')
    terminal = read(args.parent/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == p['core']
    assert read(args.parent/'terminal-audit.json')['all_frozen_unchanged']
    selected = terminal['results']['selection']
    selection_path = Path(selected['agent']['sandbox'])/'selection.json'
    assert sha(selection_path) == selected['selection_sha256']
    saved, selector_prompt, files = load(Path(selected['initial_snapshot']))
    source = ProjectSearch(load_config(args.parent/'private-config.yaml', []), Path(p['source']))
    current = source.revisions.current()
    assert current.id == p['parent'] and len(current.bundles) == 1
    discovery = source.current_discovery()
    selection = read(selection_path)
    call_lookup = {c['id']:c for c in discovery['calls']}
    affected = set(selection['affected_calls'])
    # Exactly the normal ProjectSearch.generate_candidate containment closure and canonicalization.
    while True:
        closure = affected | {c['id'] for c in discovery['calls'] if set(c['parents']) & affected}
        if closure == affected: break
        affected = closure
    assert affected == set(selection['affected_calls'])
    selection['affected_calls'] = sorted(affected)
    inputs = {n.removeprefix('inputs/'):v for n,v in files.items()
              if n not in ['inputs/previous-feedback.txt','inputs/worker-environment.json']}
    inputs.update({'selection.json':json.dumps(selection,indent=2),
                   'targets.json':json.dumps([call_lookup[c] for c in sorted(affected)],indent=2),
                   'target-resources.json':json.dumps({c:discovery.get('call_resources',{}).get(c) for c in sorted(affected)},indent=2)})
    raw = (args.parent/'original-feedback.json').read_text()
    assert digest(raw) == p['raw_feedback_sha256']
    assert raw.encode() == (Path(p['source'])/'latest-feedback.json').read_bytes()
    packaged, compact = feedback_artifacts(raw)
    assert all(inputs[n] == (v.encode() if isinstance(v,str) else v) for n,v in packaged.items())
    feedback = route(compact, current_revision=current.id,
        registry=source.revisions.load(current.registry_ref,GroupRegistry),selected_groups=selection['groups'],
        plan=source.revisions.load(current.plan_ref,ExecutionPlan))
    assert json.loads(feedback)['supplemental_evidence'] is None
    assert json.loads(feedback)['previous_trial'] == 'trial-02ed5436b4'
    old_trial = json.loads(raw)
    previous_candidate_snapshot = Path(old_trial['candidate_agent']['generation_snapshots'][0])
    oldmeta, oldprompt, oldfiles = load(previous_candidate_snapshot)
    assert oldmeta['role'] == 'candidate' and oldmeta['fresh_session']
    frozen = p['frozen_identities'].copy()
    paths = [args.parent/n for n in ['protocol.json','followup-completed.json','followup-results.json','state.json',
             'original-feedback.json','transport-preflight.json','preparation-reuse.json','startup-verification.json',
             'terminal-audit.json','public-terminal-audit.json']]
    paths += [selection_path,Path(selected['initial_snapshot']),args.parent/'selection/events.jsonl']
    for path in paths: frozen[str(path)] = sha(path)
    assert all(sha(path) == h for path,h in frozen.items())
    details = {'source':p['source'],'parent':current.id,'selection_snapshot':selected['initial_snapshot'],
        'selection_sha256':sha(selection_path),'canonical_selection_sha256':digest(inputs['selection.json']),
        'selection_semantics_unchanged':True,'canonicalization':'Normal affected_calls sort; target-resources JSON uses stable key order, same mapping as existing API.',
        'selected_scope':selection['scope'],'selected_groups':selection['groups'],'affected_calls':len(affected),
        'original_feedback_sha256':p['raw_feedback_sha256'],'original_feedback_bytes':len(raw.encode()),
        'feedback_sha256':digest(feedback),'feedback_bytes':len(feedback.encode()),'supplemental_evidence':None,
        'prior_candidate_snapshot':str(previous_candidate_snapshot),'selector_snapshot_sha256':sha(selected['initial_snapshot']),
        'selection_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'request_sha256':sha(Path(p['source'])/'request.json'),'prepared_sha256':sha(Path(p['source'])/'prepared.json'),
        'discovery_sha256':sha(Path(p['source'])/'discovery.json')}
    return inputs,feedback,oldmeta,oldprompt,oldfiles,files,frozen,details


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
        'scope':'One existing ordinary candidate generation from the actual isolated selection and original framework rejection/history/current accepted source and registry. Selection hypotheses remain unedited; only ordinary containment/canonicalization and selected-group feedback routing. No developer audit/Tensor-index correction/cause table/handwritten implementation, new selector/reviewer/prompt retry or extra context. Controller schedules no GPU/model/collective/parameter/search/collector/acceptance/pointer publication. Agent tools remain subject to full actual audit. Review exact generated bytes before any later execution. API observe-only/no caps. Existing manual source/metric mapping assistance remains; not unaided or single-variable AB. Qwen closed/FSDP rejection preserved.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output/'transport-preflight.json',details)
    write_json(args.output/'preparation-reuse.json',{k:details[k] for k in ['source','parent','request_sha256','prepared_sha256','discovery_sha256','selection_sha256','canonical_selection_sha256','original_feedback_sha256','feedback_sha256']})
    (args.output/'original-feedback.json').write_bytes((args.parent/'original-feedback.json').read_bytes())
    write_json(args.output/'state.json',{'phase':'running','stage':'candidate-only','at':now()})
    results={}
    try:
        store=RunStore.create(args.output,'candidate',{'kind':'ddp-communication-candidate'})
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
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and results['candidate']['status']=='completed' else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
