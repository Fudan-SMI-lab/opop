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

CODE = '4693d1472b012cfa786209e8ce2ede85a84ae443d7cc1c2212d2680ff42dcd38'
OLD = '04dfc6c2c9165560ae7412bbdbd00be6e79044e7f52276d096dd48728c6583f8'


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
    trialpath = q/'trials/trial-bf84d454d6/result.json'; trial = read(trialpath)
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
    endpoints = {}; jobs = {}; samples = checks = 0
    for job in sorted((q/'jobs').iterdir()):
        folder = job/'run'; receipt = read(folder/'receipt.json'); request = read(folder/'request.json')
        assert receipt['status'] == 'passed' and len(receipt['ranks']) == 1
        rank = receipt['ranks'][0]; result = rank['result']
        assert rank['status'] == 'passed' and rank['framework_source'] == protocol['core']
        assert rank['binding']['torch_uuid'] == protocol['devices'][0]
        assert sha(request['entry']) == receipt['entry_sha256']
        assert sha(request['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert request['parameters']['request'] == search.request.portable()
        manifest = read(Path(request['cwd'])/'.opop-revision.json')
        assert all(sha(Path(request['cwd'])/p) == h for p, h in manifest['files'].items())
        bundles = [{'id': b['id'], 'sha256': sha(b['file']), 'parameters': b['parameters']} for b in request['parameters'].get('bundles', [])]
        jobs[job.name] = {'request_sha256': sha(folder/'request.json'), 'receipt_sha256': sha(folder/'receipt.json'),
            'entry_sha256': receipt['entry_sha256'], 'adapter_sha256': sha(request['parameters']['adapter']),
            'manifest_sha256': sha(Path(request['cwd'])/'.opop-revision.json'), 'bundles': bundles,
            'action': request['parameters'].get('action'), 'worker_status': rank['status'], 'valid': result.get('valid'),
            'occupancy': rank['binding']['occupancy'], 'log_sha256': sha(folder/'rank-0.log')}
        if 'samples' in result:
            assert result['valid'] and result['math_policy_unchanged'] and len(result['samples']) == 5
            assert len(result['checks']) == 2 and all(c['passed'] and c['max_abs_error'] == 0 for c in result['checks'])
            for name, value in result['metrics'].items():
                assert statistics.median(s[name] for s in result['samples']) == value
            endpoints[job.name] = {k: result[k] for k in ['checks', 'samples', 'metrics', 'coverage', 'setup']}
            endpoints[job.name]['constraints'] = constraints_for_runs([receipt], search.request.portable()['objective'], search.prepared['evaluation'])
            samples += 5; checks += 2
    assert len(jobs) == 8 and samples == 25 and checks == 10
    screen = endpoints['trial-bf84d454d6-param-8c71d6a8-screen']
    assert screen['metrics']['ttft_ms'] == 26.232951
    assert screen['constraints'][0]['limit'] == 24.45004275 and not screen['constraints'][0]['passed']
    assert jobs['trial-bf84d454d6-param-8c71d6a8-screen']['bundles'][0]['sha256'] == CODE
    original = Path(protocol['source'])
    for name in ['request.json', 'discovery.json', 'prepared.json']:
        assert (original/name).read_bytes() == (q/name).read_bytes()
    source_search = ProjectSearch(load_config(root/'private-config.yaml', []), original)
    assert source_search.revisions.current().id == current.id
    diagnostics = {}
    for path in q.glob('jobs/candidate-diagnostics-*/run/candidate-diagnostics-rank-0.json'):
        d = read(path)
        errors = [e for e in d['events'] if e['event'] == 'exception']
        assert Counter((e['type'], e['message']) for e in errors) == {
            ('AssertionError', 'chain_E_absorb exceeded task tolerance (measured max_abs=22.0312)'): 3,
            ('_GuardFailure', 'input_ids'): 1}
        assert next(e for e in errors if e['type'] == '_GuardFailure')['count'] == 63
        assert all(sha(e['file']) == CODE for e in d['events'])
        diagnostics[path.parent.parent.name] = {'path': str(path), 'sha256': sha(path), 'events': d['events']}
    assert len(diagnostics) == 2
    path = q/'jobs/timeline-d2554d838c/run/timeline-analysis-rank-0-0.json'
    stored = read(path)
    assert sha(stored['trace']) == stored['trace_sha256']
    recalculated = analyze(read(stored['trace']), rank=0, anchors=stored['clock_anchors'], domain=stored['clock']['domain'])
    for key in ['devices','communication','cuda_synchronization','clock','workload_window']:
        assert recalculated[key] == stored[key], key
    timeline = {'sha256': sha(path), 'trace_sha256': stored['trace_sha256'], 'profiled_wall_ms': stored['profiled_wall_ms'],
        'devices': {k: {n:v for n,v in d.items() if not isinstance(v, (list, dict))} for k,d in stored['devices'].items()},
        'synchronization_count': len(stored['cuda_synchronization']), 'clock_status': stored['clock']['status']}
    candidate_agent = next(a for a in agents if a['role'] == 'candidate')
    meta, _, _ = load(Path(candidate_agent['generation_snapshots'][0]))
    sandbox = Path(meta['sandbox']); candidate = sandbox/'candidate.py'
    assert sha(candidate) == CODE
    tools = read(private/(candidate_agent['session']+'-tools.json'))
    initial = tools[47]['input']['content']
    write_json(private/'generated-before.json', {'tool':48, 'sha256':digest(initial), 'content':initial})
    rebuilt = initial
    for n in [50,51,53]:
        edit = tools[n-1]['input']
        old, new = edit['oldString'].replace('\\n','\n'), edit['newString'].replace('\\n','\n')
        assert rebuilt.count(old) == 1, n
        rebuilt = rebuilt.replace(old,new,1)
    assert rebuilt.encode() == candidate.read_bytes()
    extra_outputs = []
    for agent in agents:
        for t in read(private/(agent['session']+'-tools.json')):
            path = t['metadata'].get('outputPath')
            if path:
                source = Path(path); target = private/source.name
                if not target.exists(): target.write_bytes(source.read_bytes())
                assert sha(target) == sha(source)
                extra_outputs.append({'path':str(source),'sha256':sha(source),'bytes':source.stat().st_size})
    cpu = read(root/'audit/cpu-control-flow/result.json')
    assert cpu['candidate_sha256'] == CODE and not cpu['cuda_initialized']
    proposal = read(sandbox/'proposal.json')
    report = {'at':datetime.now(timezone.utc).isoformat(), 'finished_at':terminal['finished_at'], 'core':protocol['core'],
        'frozen_files':len(protocol['frozen_identities']), 'current':current.id, 'bundles':list(current.bundles),
        'trial_sha256':sha(trialpath), 'decision':trial['decision'], 'parameter_observations':1,
        'revision':trial['parameter_search']['observations'][0]['revision'],
        'candidate_sha256':CODE, 'candidate_bytes':candidate.stat().st_size, 'proposal_sha256':sha(sandbox/'proposal.json'),
        'scope':trial['selection']['scope'], 'groups':trial['selection']['groups'],
        'affected_calls':len(trial['selection']['affected_calls']), 'eliminated_declaration':len(proposal['eliminated_calls']),
        'requires':proposal['requires'], 'replaces':proposal['replaces'],
        'roles':roles, 'jobs':jobs, 'endpoints':endpoints, 'samples':samples, 'full_checks':checks,
        'diagnostics':diagnostics, 'timeline':timeline, 'cpu':cpu, 'cpu_result_sha256':sha(root/'audit/cpu-control-flow/result.json'),
        'initial_code_sha256':digest(initial), 'final_reconstructed_exact':True, 'extra_private_outputs':extra_outputs,
        'review':{
          'reads':'Complete recorded commands/paths/outputs inspected: selector40 and candidate63 tools never opened the new observation index/probe/plan/JSON. Selector tool10 printed its pointer. Both used old packaged sidecars and code. No later accepted implementation/reviewer answer reads observed; external filesystem not hermetic. Input delivery is not evidence use.',
          'role_errors':'Selector wrong source/ directory exit2. Candidate evaluation null AttributeError, missing transformers.modeling_attention_interface, CPU DynamicCache layers TypeError. Later CPU cache/inference operations, syntax and schema pass; no model/GPU numerical role probe. Agent removed own __pycache__; scripts/initial code/three edits/history and full truncated search output retained.',
          'code':'Decode-only reconstructed stock module operations and GQA no-mask SDPA, 31 graphs; no Triton or new prefill binding. Prebuilt cache mirrors reused. Guard checks shape/dtype/device/stride; shared masters/refresh and retained owners not arbitrary-alias/concurrency proof. Gate FP32 .02/.02 differs from original BF16 outer relation arithmetic. Conditional second graph replay on bitwise mismatch is generated setup logic, not an external retry.',
          'actual_failure':'Each sidecar has63 input_ids guard failures and one chain_E assertion propagated through3frames;252 normal generator False and1 master-view False are not errors. Run-only valid does not compare quality. Full evaluator quality passes only on62 parent setup-fallback routes. No accepted bundle/paired or independent stage.',
          'cpu_boundary':'Exact singleton .contiguous keeps stride32,1 and same object; exact replacement guard then parent mutates same CPU cache258->259->260. Supports a code/order explanation for63 B/R/E guards and E mismatch, not a historical per-call GPU cache trace or unique numerical cause. Separately stubbed successful setup+verify still ends disabled/not_attempted: latent build-success defect, not actual E failure cause.',
          'claims_not_adopted':'Prefill unbound does not guarantee TTFT, every-return-fresh-cache claim false, fused old failure not merely tighter tolerance, setup selfchecks are not outer full quality, exclusive host/bandwidth/root-cause claims unsupported. Planner no-graph-question/quality-only-divergence framing and profile subtraction do not establish path or cause.',
          'measurements':'5 screen samples/2 checks plus20 binding diagnostic samples/8 checks retained. Fixed original TTFT limit24.45004275; no reanchor, outlier removal or observer subtraction. Timeline is a separate capture, binding passthrough differences are not negative physical cost.',
          'assistance':'Current-device controls/source hashes reused read-only, no new source comparison; manual source field mapping and earlier assistance remain. No production changes or closed-study reopening.'},
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True)}
    assert all(sha(p) == h for p,h in protocol['frozen_identities'].items())
    write_json(root/'terminal-audit.json', report)
    public = {k:v for k,v in report.items() if k not in ['roles','diagnostics','endpoints','extra_private_outputs']}
    public['roles'] = [{k:v for k,v in a.items() if k not in ['input_manifest','usage','snapshot','session','delivery']} |
        {'provider_estimated_cost':a['usage']['reported_cost'], 'delivery':a['delivery']} for a in roles]
    public['endpoints'] = {k:{n:v for n,v in e.items() if n != 'coverage'} |
        {'coverage':{n:e['coverage'].get(n) for n in ['counts','routes','reference_invocations','fallbacks']}} for k,e in endpoints.items()}
    public['diagnostics'] = {k:{n:v for n,v in d.items() if n != 'events'} for k,d in diagnostics.items()}
    public['terminal_audit_sha256'] = sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'roles':[(a['role'],a['inputs'],a['input_bytes'],a['assistant_messages'],sum(a['tools'].values()),a['usage']['reported_cost'])for a in roles],
        'samples':samples,'checks':checks,'timeline':timeline,'candidate_sha256':CODE,'public_bytes':(root/'public-terminal-audit.json').stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
