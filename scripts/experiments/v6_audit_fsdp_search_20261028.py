"""Read-only audit of the frozen FSDP search, raw metrics and native paths."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def endpoint(run, request):
    assert valid_run(run)
    rows = []
    for rank in run['ranks']:
        result = rank['result']
        assert result['math_policy_unchanged'] and len(result['samples']) == 5
        assert len(result['checks']) == 2 and all(c['passed'] for c in result['checks'])
        traces = []
        assert set(result['diagnostic_metrics']) == set(request['diagnostic_metrics'])
        for name, diagnostic in result['diagnostic_metrics'].items():
            assert len(diagnostic['values']) == len(diagnostic['evidence']) == 3
            field = diagnostic['specification']['source']['field']
            for path, value in zip(diagnostic['evidence'], diagnostic['values'], strict=True):
                trace, window = trace_window(read(path), 'opop.communication_metric_workload')
                assert window
                recalc = normalize(trace, {'kernels': []}, rank['rank'])
                assert abs(recalc[field] - value) < 1e-8
                traces.append({'metric': name, 'path': path, 'sha256': sha(path), 'value': value,
                               'nccl_kernel_count': recalc['communication_kernel_count']})
        rows.append({'rank': rank['rank'], 'framework_source': rank['framework_source'],
                     **{k: result[k] for k in ['checks', 'coverage', 'samples', 'metrics', 'metric_samples', 'bundle_hashes']},
                     'traces': traces})
    aligned = {}
    for name in ['step_ms', 'peak_allocated_bytes', 'allocated_bytes', *request['diagnostic_metrics']]:
        series = [r['result']['metric_samples'][name] if name in request['diagnostic_metrics']
                  else [s[name] for s in r['result']['samples']] for r in run['ranks']]
        samples = [max(values) for values in zip(*series, strict=True)]
        aligned[name] = {'samples': samples, 'median': statistics.median(samples)}
    return {'ranks': rows, 'max_rank_then_median': aligned}


def agent_record(agent, database):
    metadata, prompt, files = load(Path(agent['generation_snapshots'][0]))
    tools = []
    for ident, data in database.execute('select id,data from part where session_id=? order by time_created,id', (agent['session'],)):
        part = json.loads(data)
        if part.get('type') != 'tool':
            continue
        state = part.get('state', {}); inp = state.get('input', {})
        serialized = json.dumps(inp, sort_keys=True)
        row = {'part': ident, 'tool': part['tool'], 'status': state.get('status'),
               'input_sha256': hashlib.sha256(serialized.encode()).hexdigest(), 'path': inp.get('filePath')}
        if 'feedback' in serialized:
            row.update(feedback_command=inp.get('command'), output_sha256=hashlib.sha256(str(state.get('output', '')).encode()).hexdigest(),
                       output_preview=str(state.get('output', ''))[:1800])
        tools.append(row)
    return {'session': agent['session'], 'role': agent['role'], 'usage': agent.get('usage'),
            'contract_repairs': agent.get('contract_repairs'), 'snapshot': agent['generation_snapshots'][0],
            'framework_source': metadata['framework_source'], 'model': metadata.get('model'),
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'settings_sha256': hashlib.sha256(json.dumps(metadata['tool_model_settings'], sort_keys=True).encode()).hexdigest(),
            'inputs': {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in files.items()}, 'tools': tools}


def audit(root, output):
    terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed'
    search = ProjectSearch(load_config(root / 'private-config.yaml', []), root / 'fsdp-search')
    prepared = search.prepared; request = read(search.output / 'request.json')
    source_check = root.parent / 'fsdp-owner-repair-20261028/source-check.json'
    prior = read(source_check); assert prior['status'] == 'passed'
    adapter = Path(prepared['controls_path']).parent / 'reference-source/opop_adapter.py'
    assert sha(adapter) == prepared['adapter_sha256'] == prior['adapter_sha256']
    assert request == read(root.parent / 'fsdp-owner-repair-20261028/fsdp-repaired/request.json')
    verified = []
    for rank in prior['ranks']:
        for path, expected in [(rank['adapter_reference'], rank['adapter_reference_sha256']),
                               *((c['source'], c['source_sha256']) for c in rank['checks'])]:
            assert sha(path) == expected
            verified.append({'path': path, 'sha256': expected})
    events = [json.loads(line) for line in (search.output / 'events.jsonl').read_text().splitlines()]
    order = [e['payload']['session'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED' and e['payload']['role'] == 'candidate']
    trials = sorted([read(p) for p in (search.output / 'trials').glob('*/result.json')], key=lambda t: order.index(t['candidate_agent']['session']))
    report = {'finished_at': terminal['finished_at'], 'terminal_sha256': sha(root / 'followup-completed.json'),
              'adapter': {'path': str(adapter), 'sha256': prepared['adapter_sha256']}, 'request': request,
              'source_fidelity': {'prior_check': str(source_check), 'sha256': sha(source_check), 'unchanged_references': verified,
                                  'scope': 'Prior independent full source comparison retained; actual reference hashes rechecked. Candidate comparisons use these source-verified baseline references.'},
              'contract_invariants': search.revisions.load(search.revisions.current().contract_ref, ProjectContract).invariants,
              'trials': [], 'native': [], 'event_counts': dict(Counter(e['type'] for e in events)),
              'discoveries': [e for e in events if e['type'] == 'PROJECT_DISCOVERED']}
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    for t in trials:
        candidate = Path(t['candidate_agent']['sandbox']) / 'candidate.py'
        row = {'trial': t['trial'], 'decision': t['decision'], 'scope': t['selection']['scope'],
               'group_count': len(t['selection']['groups']), 'affected_calls': len(t['selection']['affected_calls']),
               'rationale': t['selection']['rationale'], 'candidate': str(candidate), 'candidate_sha256': sha(candidate),
               'parameters': t['bundle'].get('parameters'), 'selected': t['parameter_search']['selected'],
               'agents': [agent_record(t[k], db) for k in ['selection_agent', 'candidate_agent']], 'parameter_observations': []}
        for o in t['parameter_search']['observations']:
            observation = {'bundle': o['bundle']['id'], 'parameters': o['bundle']['parameters'], 'status': o['run']['status'],
                           'failure': o['run'].get('failure'), 'valid': o['result']['valid'], 'score': o['result']['score'],
                           'checks': [{'rank': r['rank'], 'checks': (r.get('result') or {}).get('checks'),
                                       'status': r.get('status')} for r in o['run']['ranks']]}
            if valid_run(o['run']):
                observation['endpoint'] = endpoint(o['run'], request)
            row['parameter_observations'].append(observation)
        if t.get('pairs'):
            assert paired_decision(t['pairs'], request['objective'], prepared['evaluation']) == t['measurement']
            row['measurement'] = t['measurement']
            row['pairs'] = [{arm: endpoint(ep, request) for arm, ep in pair.items()} for pair in t['pairs']]
        report['trials'].append(row)
        print(json.dumps({'audited_trial': t['trial']}), flush=True)
    db.close()
    independent_path = next((root / 'fsdp-independent/version-trials').glob('*/result.json'))
    independent = read(independent_path)
    assert paired_decision(independent['pairs'], request['objective'], prepared['evaluation']) == independent['measurement']
    report['independent'] = {'path': str(independent_path), 'sha256': sha(independent_path), 'decision': independent['decision'],
                             'measurement': independent['measurement'],
                             'pairs': [{arm: endpoint(ep, request) for arm, ep in pair.items()} for pair in independent['pairs']]}
    for label in ['baseline', 'candidate']:
        path = root / f'native-{label}.json'; native = read(path); rows = []
        for rank in native['ranks']:
            ns = sqlite3.connect('file:' + rank['evidence'] + '?mode=ro', uri=True)
            counts = list(ns.execute('select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id group by s.value'))
            ns.close()
            assert sum(v[1] for v in counts) == len(rank['kernels'])
            nccl = [v for v in counts if 'nccl' in v[0].lower()]
            assert sum(v[1] for v in nccl) == rank['nccl_kernels']
            assert abs(sum(v[2] for v in nccl)/1e6 - rank['nccl_kernel_sum_ms']) < 1e-8
            rows.append({'rank': rank['rank'], 'path': rank['evidence'], 'sha256': sha(rank['evidence']),
                         'kernels': sum(v[1] for v in counts), 'nccl': [{'name': name, 'count': count, 'sum_ms': duration/1e6} for name, count, duration in nccl]})
        report['native'].append({'arm': label, 'path': str(path), 'sha256': sha(path), 'ranks': rows})
    current = search.revisions.current()
    report['current'] = {'revision': current.id, 'bundles': list(current.bundles)}
    report['limits'] = ['Developer-assisted evaluator baseline; subsequent search autonomous, not a fresh unaided original-entry cohort.',
                       'Final-round acceptance: no post-accept rediscovery or bundle composition in this batch.',
                       'P2P ordering and reassociated GEMM are plausible failure mechanisms, not independently isolated unique causes.',
                       'NCCL kernel count reduction is observed; kernel durations/logical payload are not wire time, bandwidth or exposed critical path.',
                       'Four diagnostic metrics use distinct profile invocations; do not subtract cross-field medians.',
                       'Historical rejections and every original sample remain unchanged.']
    write_json(output, report)
    m = independent['measurement']
    print(json.dumps({'audit': str(output), 'bytes': output.stat().st_size, 'independent_accepted': independent['decision']['accepted'],
                      'gain_percent': 100*(1-m['objective']/m['reference'])}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.output)
