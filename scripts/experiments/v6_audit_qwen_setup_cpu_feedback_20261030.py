"""Read-only terminal audit of the current-device ordinary trial; no GPU rerun."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

CODE = '6ec4e97a4bd39a7d16c82109b49b841f0258c039d56de15eeb6f80696e4906fa'
OLD = '4693d1472b012cfa786209e8ce2ede85a84ae443d7cc1c2212d2680ff42dcd38'


def digest(x):
    return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()


def main(root):
    protocol = read(root/'protocol.json'); terminal = read(root/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    q = root/'qwen-continuation'; search = ProjectSearch(load_config(root/'private-config.yaml', []), q)
    current = search.revisions.current()
    assert current.id == terminal['results']['search']['current'] and not current.bundles
    assert read(root/'preparation-reuse.json')['controls_path'] == search.prepared['controls_path']
    trialpath = q/'trials/trial-2a7d20b4d9/result.json'; trial = read(trialpath)
    assert trial['decision'] == {'accepted': False, 'reasons': ['no_valid_parameter_candidate']}
    assert trial['parameter_search']['space'] == {} and len(trial['parameter_search']['observations']) == 1
    events = [json.loads(x) for x in (q/'events.jsonl').read_text().splitlines()]
    agents = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    assert len(agents) == 3 and all(a['contract_repairs'] == 0 for a in agents)
    private = root/'terminal-private-history'; inventory = []; roles = []
    packet, compact = feedback_artifacts((root/'diagnostic-feedback.json').read_text())
    registry = search.revisions.load(current.registry_ref, GroupRegistry)
    plan = search.revisions.load(current.plan_ref, ExecutionPlan)
    for agent in agents:
        snap = Path(agent['generation_snapshots'][0]); meta, prompt, files = load(snap)
        assert meta['framework_source'] == protocol['core'] and meta['fresh_session']
        sandbox = Path(meta['sandbox'])
        assert all((sandbox/name).read_bytes() == data for name, data in files.items())
        delivered = None
        if 'inputs/previous-feedback.txt' in files:
            actual = json.loads(files['inputs/previous-feedback.txt'])
            candidate_role = meta['role'] == 'candidate'
            routed = json.loads(route(compact, current_revision=current.id, registry=registry,
                selected_groups=trial['selection']['groups'] if candidate_role else None,
                plan=plan if candidate_role else None))
            assert actual == routed
            assert digest(files['inputs/previous-candidate/candidate.py']) == OLD
            for name, data in packet.items():
                assert files['inputs/'+name] == (data.encode() if isinstance(data, str) else data)
            delivered = {'feedback_sha256': digest(files['inputs/previous-feedback.txt']),
                'previous_trial': actual['previous_trial'], 'historical_code_sha256': OLD,
                'supplemental_evidence_equal': True, 'full_diagnostic_packaged_equal': True}
        history = read(private/(agent['session']+'.json'))
        tools = read(private/(agent['session']+'-tools.json'))
        toolrows = []
        for i, t in enumerate(tools, 1):
            row = {'number': i, 'part': t['part'], 'tool': t['tool'], 'status': t['status'],
                'input': t['input'], 'exit': t.get('metadata', {}).get('exit'),
                'output_sha256': digest(t.get('output', '')), 'output_bytes': len(t.get('output', '').encode()),
                'traceback': 'Traceback (most recent call last)' in t.get('output', '')}
            toolrows.append(row)
            if t['tool'] in ['write', 'edit'] or "python - <<" in t['input'].get('command', ''):
                target = private/(agent['session']+'-tool-'+str(i)+'.json')
                if not target.exists(): write_json(target, t)
        inventory.append({'session': agent['session'], 'tools': toolrows})
        roles.append({'role': agent['role'], 'session': agent['session'], 'snapshot': str(snap),
            'snapshot_sha256': sha(snap), 'inputs': len(files), 'input_bytes': sum(map(len, files.values())),
            'prompt_sha256': digest(prompt), 'settings_sha256': digest(json.dumps(meta['tool_model_settings'], sort_keys=True)),
            'input_manifest': meta['files'], 'delivery': delivered, 'repairs': agent['contract_repairs'],
            'assistant_messages': sum(x['data'].get('role') == 'assistant' for x in history['message']),
            'tools': dict(Counter(t['tool'] for t in tools)), 'usage': agent['usage'],
            'errors': [{k: t[k] for k in ['number','part','tool','exit','traceback','output_sha256']}
                       for t in toolrows if t['traceback'] or t['exit'] not in [0, None]],
            'private_history_sha256': sha(private/(agent['session']+'.json'))})
    write_json(private/'review-inventory.json', inventory)
    jobs = {}
    for job in sorted((q/'jobs').iterdir()):
        folder = job/'run'; receipt = read(folder/'receipt.json'); request = read(folder/'request.json')
        assert receipt['status'] == 'failed' and len(receipt['ranks']) == 1
        rank = receipt['ranks'][0]
        assert rank['status'] == 'failed' and rank['framework_source'] == protocol['core']
        assert rank['binding']['torch_uuid'] == protocol['devices'][0]
        assert sha(request['entry']) == receipt['entry_sha256']
        assert sha(request['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert request['parameters']['request'] == search.request.portable()
        manifest = read(Path(request['cwd'])/'.opop-revision.json')
        assert all(sha(Path(request['cwd'])/p) == h for p,h in manifest['files'].items())
        bundles = [{'id': b['id'], 'sha256': sha(b['file']), 'parameters': b['parameters']} for b in request['parameters']['bundles']]
        assert len(bundles) == 1 and bundles[0]['sha256'] == CODE and bundles[0]['parameters'] == {}
        assert 'result' not in rank and rank['error_type'] == 'AttributeError'
        assert rank['error'] == "'_CacheHandle' object has no attribute 'get_seq_length'"
        assert 'return _original(*args, **kwargs)' in rank['traceback']
        artifacts = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in folder.iterdir() if p.is_file()}
        assert not any(p.suffix in ['.sqlite', '.nsys-rep', '.gz'] for p in folder.iterdir())
        assert not list(folder.glob('timeline-*.json'))
        jobs[job.name] = {'artifacts': artifacts, 'entry_sha256': receipt['entry_sha256'],
            'adapter_sha256': sha(request['parameters']['adapter']),
            'manifest_sha256': sha(Path(request['cwd'])/'.opop-revision.json'),
            'action': request['parameters'].get('action'), 'worker_status': rank['status'],
            'error_type': rank['error_type'], 'error': rank['error'], 'traceback': rank['traceback'],
            'bundles': bundles, 'occupancy': rank['binding'].get('occupancy'),
            'quality_checks': 'not_returned', 'clean_samples': 0, 'native_trace': 'not_produced'}
    assert len(jobs) == 4
    original = Path(protocol['source'])
    for name in ['request.json','discovery.json','prepared.json']:
        assert (original/name).read_bytes() == (q/name).read_bytes()
    assert ProjectSearch(load_config(root/'private-config.yaml', []), original).revisions.current().id == current.id
    dp = q/'jobs/candidate-diagnostics-7f6ae8983f/run/candidate-diagnostics-rank-0.json'
    diagnostic = read(dp)
    assert len(diagnostic['events']) == 1
    event = diagnostic['events'][0]
    assert event['event'] == 'return' and event['type'] == 'bool' and event['count'] == 252
    assert event['message'] == 'False' and sha(event['file']) == CODE
    candidate_agent = next(a for a in agents if a['role'] == 'candidate')
    meta, _, _ = load(Path(candidate_agent['generation_snapshots'][0]))
    sandbox = Path(meta['sandbox']); candidate = sandbox/'candidate.py'
    assert sha(candidate) == CODE
    tools = read(private/(candidate_agent['session']+'-tools.json'))
    initial = tools[41]['input']['content']
    write_json(private/'generated-before.json', {'tool':42, 'sha256':digest(initial), 'content':initial})
    rebuilt = initial
    for n in range(50,56):
        edit = tools[n-1]['input']; old,new = edit['oldString'],edit['newString']
        assert rebuilt.count(old) == 1, n
        rebuilt = rebuilt.replace(old,new,1)
    assert rebuilt.encode() == candidate.read_bytes()
    extra_outputs = []
    for agent in agents:
        for t in read(private/(agent['session']+'-tools.json')):
            path = t.get('metadata',{}).get('outputPath')
            if path:
                source = Path(path); target = private/source.name
                if not target.exists(): target.write_bytes(source.read_bytes())
                assert sha(target) == sha(source)
                extra_outputs.append({'path':str(source),'sha256':sha(source),'bytes':source.stat().st_size})
    cpu = read(root/'audit/cpu-binding-interface/result.json')
    assert cpu['candidate_sha256'] == CODE and not cpu['cuda_initialized'] and cpu['restored']
    assert cpu['core'] == protocol['core']
    proposal = read(sandbox/'proposal.json')
    assert proposal['parameter_space'] == {} and len(proposal['eliminated_calls']) == 23225
    original_feedback = (original/'latest-feedback.json').read_bytes()
    new_feedback = (root/'diagnostic-feedback.json').read_bytes()
    before, after = json.loads(original_feedback), json.loads(new_feedback)
    assert new_feedback.startswith(original_feedback[:original_feedback.rfind(b'}')])
    assert {k:v for k,v in after.items() if k != 'supplemental_evidence'} == before
    assert after['supplemental_evidence']['cpu_interface_observations']['result'] == read(Path(protocol['cpu_observation'])/'audit/cpu-control-flow/result.json')
    assert len(protocol['frozen_identities']) == 422
    assert all(sha(p) == h for p,h in protocol['frozen_identities'].items())
    report = {'at':datetime.now(timezone.utc).isoformat(), 'terminal_at':terminal['finished_at'],
        'terminal_reason':'batch_finished', 'controller_status':'completed', 'core':framework_source(),
        'frozen_files':422, 'all_frozen_unchanged':True, 'current':current.id, 'bundles':[],
        'trial_sha256':sha(trialpath), 'decision':trial['decision'], 'parameter_space':{},
        'parameter_observations':1, 'paired_acceptance':'not_run', 'independent':'not_run',
        'samples':0, 'full_checks_returned':0, 'roles':roles, 'jobs':jobs,
        'candidate_sha256':CODE, 'candidate_bytes':candidate.stat().st_size,
        'proposal_sha256':sha(sandbox/'proposal.json'), 'initial_code_sha256':digest(initial),
        'six_edits_reconstructed_exactly':True, 'extra_private_outputs':extra_outputs,
        'diagnostic':{'sha256':sha(dp),'events':diagnostic['events'], 'interpretation':'One normal False generator return aggregated252; no candidate-frame exception does not erase original-parent worker failure.'},
        'cpu_interface':{'sha256':sha(root/'audit/cpu-binding-interface/result.json'),'result':cpu},
        'findings':{
            'failure':'All4workers fail in original parent get_seq_length after binding._original; no returned quality checks, samples, constraints or noise acceptance. Nsys/timeline fail during pre-capture invocation and produce no native trace.',
            'delivery':'Selector30 and candidate33 initialfiles independently replayed through actual registry/plan route; prior4693 bytes and packaged artifacts exact. New inline CPU facts are developer assistance, not unaided discovery.',
            'consumption':'Selector tool18 exposes nested names/limits but not nested numeric facts; candidate tool6 prints complete CPU result and tool30 explicitly prints stride/cache guard results. Neither opens new index/fixture/result files; they read inline facts plus original sidecars/code/library. No observed later accepted/reviewer answer reads in complete recorded tools; external filesystem not hermetic.',
            'code':'New opaque _CacheHandle has no parent cache ABI; framework signature retains DynamicCache type and stride1,1. Candidate internal stride-agnostic guards/direct selfchecks cannot override dispatch. CPU real BindingSession with device-only adapted signature corroborates type/stride fallback, not historical GPU per-call state or numerical quality.',
            'tests':'Candidate CPU guard fails on CPU cache, wrong4x1 shape, and immutableTensor.device. Final passing fast-tail test manually copies implementation rather than invoking replacement and contains assert(... or True). Syntax/schema do not validate binding or GPU quality.',
            'limits':'No new kernel/native evidence. Code success flag changed, but verify exception after fast_enabled=True does not clear it; latent catch issue not established as actual error. Setup checks bypass frameworkdispatch; no unique old numerical/root cause or lifetime/TTFT guarantee.',
            'claims_not_adopted':'Selector opaque-handle legality, stride guarantee, zero-overhead empty installation and profile-gap critical-path/reclaimable-host claims are unsupported. Planner recognized parent/binding alternatives but collectors fail before capture; old parent discovery does not certify currentnative.',
            'assistance':'Current-device controls/source reused read-only, no new full-source comparison; manual source mapping and prior assistance remain. No production/closed-study adoption, no replay until pass.'},
        'audit_reader_error':"Read-only inventory initially guessed collection-plan.json; actual filename collection.json. First audit expected occupancy in failed worker binding; absent field now recorded as null. Original errors retained; no model/fixture rerun or candidate edit.",
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json', report)
    public = {k:v for k,v in report.items() if k not in ['roles','jobs','diagnostic','extra_private_outputs']}
    public['roles'] = [{k:a[k] for k in ['role','snapshot_sha256','inputs','input_bytes','prompt_sha256','settings_sha256','repairs','assistant_messages','tools','errors','private_history_sha256']} |
                       {'provider_estimated_cost':a['usage']['reported_cost'], 'delivery':a['delivery']} for a in roles]
    public['jobs'] = {k:{n:v for n,v in j.items() if n not in ['traceback','occupancy']} for k,j in jobs.items()}
    public['diagnostic'] = {'sha256':sha(dp),'normal_false_count':252,'candidate_exception_events':0}
    public['terminal_audit_sha256'] = sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'], 'public_sha256':sha(root/'public-terminal-audit.json'),
        'public_bytes':(root/'public-terminal-audit.json').stat().st_size,
        'roles':[(a['role'],a['inputs'],a['input_bytes'],a['assistant_messages'],sum(a['tools'].values()),a['usage']['reported_cost'])for a in roles]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
