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

CODE = '04dfc6c2c9165560ae7412bbdbd00be6e79044e7f52276d096dd48728c6583f8'
OLD = 'f5b76ebde100c039d8ad9264e316dbade5acdb74c02f40308e958673e802be7e'


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
    trialpath = q/'trials/trial-0db61e9938/result.json'; trial = read(trialpath)
    assert trial['decision'] == {'accepted': False, 'reasons': ['no_valid_parameter_candidate']}
    assert trial['parameter_search']['space'] == {} and len(trial['parameter_search']['observations']) == 1
    events = [json.loads(x) for x in (q/'events.jsonl').read_text().splitlines()]
    agents = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    assert len(agents) == 4 and all(a['contract_repairs'] == 0 for a in agents)
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
    assert len(jobs) == 11 and samples == 25 and checks == 10
    screen = endpoints['trial-0db61e9938-param-4be0257f-screen']
    assert screen['metrics']['ttft_ms'] == 24.679412
    assert screen['constraints'][0]['limit'] == 24.45004275 and not screen['constraints'][0]['passed']
    assert jobs['trial-0db61e9938-param-4be0257f-screen']['bundles'][0]['sha256'] == CODE
    diagnostics = {}
    for path in q.glob('jobs/candidate-diagnostics-*/run/candidate-diagnostics-rank-0.json'):
        d = read(path); exceptions = [e for e in d['events'] if e['event'] == 'exception']
        assert len(exceptions) == 6
        assert Counter(e['message'] for e in exceptions) == {
            'decode logits mismatch (fused_eager_vs_parent_step_1)': 3,
            'decode logits mismatch (exact_chain_E_absorb)': 3}
        diagnostics[path.parent.parent.name] = {'path': str(path), 'sha256': sha(path), 'events': d['events']}
    assert len(diagnostics) == 2
    path = q/'jobs/timeline-40edd908a1/run/timeline-analysis-rank-0-0.json'; stored = read(path)
    assert sha(stored['trace']) == stored['trace_sha256']
    recalculated = analyze(read(stored['trace']), rank=0, anchors=stored['clock_anchors'], domain=stored['clock']['domain'])
    for key in ['devices','communication','cuda_synchronization','clock','workload_window']:
        assert recalculated[key] == stored[key], key
    timeline = {'sha256': sha(path), 'trace_sha256': stored['trace_sha256'], 'profiled_wall_ms': stored['profiled_wall_ms'],
        'devices': {k: {n:v for n,v in d.items() if not isinstance(v, (list, dict))} for k,d in stored['devices'].items()},
        'synchronization_count': len(stored['cuda_synchronization']), 'clock_status': stored['clock']['status']}
    python = []
    for path in q.glob('jobs/python-runtime-*/run/python-rank-0.json'):
        d = read(path); stats = pstats.Stats(d['pstats'])
        for name in ['by_self_time', 'by_cumulative_time']:
            for row in d[name]:
                cc, nc, tt, ct, callers = stats.stats[(row['file'], row['line'], row['function'])]
                assert row['calls'] == nc and row['primitive_calls'] == cc
                assert row['self_ms'] == tt*1000 and row['cumulative_ms'] == ct*1000
        python.append({'path': str(path), 'sha256': sha(path), 'pstats_sha256': sha(d['pstats']),
            'profiled_wall_ms': d['profiled_wall_ms'], 'total_calls': stats.total_calls, 'primitive_calls': stats.prim_calls})
    discovery = read(q/'jobs/discover-524d35b658/run/rank-0.json')['result']
    discovery['raw_sha256'] = sha(discovery['discovery'])
    raw_discovery = read(discovery['discovery'])
    assert sha(raw_discovery['trace']) == raw_discovery['trace_sha256']
    trace = read(raw_discovery['trace'])
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    assert len(kernels) == len(raw_discovery['kernels']) == discovery['coverage']['cuda_kernels'] == 66441
    assert len(raw_discovery['calls']) == discovery['call_sites'] == 26910
    assert len(raw_discovery['groups']) == discovery['groups'] == 17852
    discovery['trace_sha256'] = raw_discovery['trace_sha256']
    review = {
        'outcome': 'One empty-parameter screen, outer quality passes on parent fallback; TTFT24.679412 exceeds fixed24.45004275. No paired acceptance, accepted version, rediscovery, coexistence, independent or native candidate certification.',
        'reads': 'Selector tool14 read original setup sidecar; candidate tool16 read packaged full diagnostic after wrong dev-v6-work paths. Both read historical f5b and empty installed bundles. No later accepted implementation/reviewer answer access observed in complete recorded commands/outputs; filesystem is not hermetic.',
        'cpu': 'Selector local schema validation initially omitted affected_calls and agent corrected it; framework repair count remains zero. Candidate CPU four-layer cache fixtures failed missing method, inconsistent lengths, torch.Size equality, bad slice expectation, mixed ownership, unpacking and mock parent positional signature. Last full guard probe still exits1 after partial checks; only final syntax/import/schema smoke passes. No full CUDA or Triton numerical probe by the role.',
        'code': 'Three Triton kernels plus parent linear/SDPA fused mode; reconstructed exact-mode decomposition, 31 decode graphs, prefill passthrough. Fresh mirror objects share master tensors. Actual fast_enabled staysFalse until setup ends, so setup replacement chains call parent, not replay. ChainE calls parent then passes same cache to another call; mutation/input equivalence was not recorded. Code inference is not unique historical cause.',
        'gates': 'Internal FP32 .012/.012 check differs from original full quality .02/.02. Both diagnostics report two assertions with three propagated frames each; no raw compared values/max/cache states. Normal False504 is not error. Cannot infer outer violation or successful new fast path.',
        'profiles': 'Timeline/Python and two run-only candidate diagnostics are distinct captures without observe/compare. Inspect actual request bundles in jobs: labels do not certify candidate path. No clean-wall minus profile host-cost attribution. Binding parent/passthrough differences change sign across workers; not physical negative cost.',
        'limits': 'Manual source mapping and developer diagnostic assistance remain; source comparison not rerun. Candidate setup/extra owners affect resources; no memory gain, arbitrary alias/lifetime/concurrency or same-op bitwise guarantee. No production changes.',
        'audit_error': 'First CPU audit route replay omitted actual registry/plan and failed equality on routing only. Retained audit-attempt-1.txt; corrected replay includes actual registry and selected plan and matches both complete snapshots. No Agent/GPU rerun or production change.',
        'next': 'A bounded original-failure diagnostic-author request may address missing internal comparison values/input-state provenance. No manual candidate fix, threshold weakening, performance retry or additional reviewer.'}
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'], 'core': protocol['core'],
        'frozen_files': len(protocol['frozen_identities']), 'current': current.id, 'bundles': list(current.bundles),
        'trial_sha256': sha(trialpath), 'decision': trial['decision'], 'parameter_observations': 1,
        'roles': roles, 'jobs': jobs, 'endpoints': endpoints, 'samples': samples, 'full_checks': checks,
        'discovery': discovery, 'diagnostics': diagnostics, 'timeline': timeline, 'python': python, 'review': review,
        'df': subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'], text=True)}
    write_json(root/'terminal-audit.json', report)
    public = {k:v for k,v in report.items() if k not in ['roles','diagnostics','endpoints']}
    public['roles'] = [{k:v for k,v in a.items() if k not in ['input_manifest']} for a in roles]
    public['endpoints'] = endpoints
    public['diagnostics'] = {k:{n:v for n,v in d.items() if n != 'events'} for k,d in diagnostics.items()}
    public['terminal_audit_sha256'] = sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json', public)
    print(json.dumps({'audit_sha256': public['terminal_audit_sha256'], 'samples': samples, 'checks': checks,
        'roles': [(a['role'],a['assistant_messages'],sum(a['tools'].values()),a['usage']['reported_cost']) for a in roles],
        'job_bundles': {k:v['bundles'] for k,v in jobs.items()}, 'timeline':timeline}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
