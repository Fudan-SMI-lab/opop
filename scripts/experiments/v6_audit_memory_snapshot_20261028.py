"""Read frozen allocator experiments; keep raw samples and GC evidence on server."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_ddp_tp_20261028 import normalize_parameters
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import endpoint_summary


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke_tree(source):
    tree = ast.parse(source)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_worker_entry')
    node = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'invoke')
    worker.body.remove(node)
    return ast.dump(tree), node


def gc_intervals(data):
    pending, result = {}, []
    for event in data['events']:
        key = event['thread'], event['generation']
        if event['phase'] == 'start':
            assert key not in pending
            pending[key] = event
        else:
            first = pending.pop(key)
            result.append({**first, 'stop_ns': event['at_ns'], 'collected': event['collected'],
                           'duration_ms': (event['at_ns'] - first['at_ns']) / 1e6})
    assert not pending
    return result


def audit(root, output):
    initial = root.parent / 'memory-snapshot-diagnostics-20261028'
    old_end, end = read(initial/'followup-completed.json'), read(root/'followup-completed.json')
    assert old_end['phase'] == 'completed_with_errors' and end['phase'] == 'completed'
    assert old_end['results']['ddp']['status'] == 'skipped_failed_controls'
    assert not list((initial/'controls').rglob('rank-*-started.json'))
    old_framework = next((initial/'framework').iterdir())
    framework = next((root/'framework').iterdir())
    rel = Path('scripts/experiments/v6_memory_snapshot_batch_20261028.py')
    before, after = (old_framework/rel).read_text(), (framework/rel).read_text()
    assert before.replace('output=output,', "output=output/'run',") == after
    report = {'root': str(root), 'initial_attempt': {'terminal': old_end['phase'],
        'cuda_workers_launched': False, 'ddp': 'skipped_failed_controls',
        'correction_only_worker_output_subdirectory': True,
        'error': old_end['results']['cuda-controls']['traceback']}, 'terminal': end['phase'],
        'controls': [], 'endpoints': [], 'generation2_measure': [], 'reviews': []}
    expected = read(root/'protocol.json')
    original_file = framework/'src/kernel_optimizer/project/evaluation.py'
    assert sha(original_file) == expected['production_evaluator_sha256']
    original_ast, _ = invoke_tree(original_file.read_text())
    for lifetime in ['legacy', 'invocation']:
        variants = {}
        for policy in ['original', 'snapshot']:
            path = root/'controls'/f'{lifetime}-{policy}'/'run/receipt.json'
            raw = read(path); assert raw['status'] == 'passed'
            v = raw['ranks'][0]['result']; assert v['valid']
            ev = v['evaluation_source']
            assert sha(ev['path']) == ev['sha256'] == expected[
                'isolated_evaluator_sha256' if policy == 'snapshot' else 'production_evaluator_sha256']
            assert all(c['passed'] for c in v['controls']['reset'])
            assert all(v['controls'][k] for k in ['identity_covered', 'wrong_output_rejected', 'missing_coverage_rejected'])
            assert v['controls']['identity']['passed']
            variants[policy] = {'fields': [{k: x for k, x in s.items() if k.startswith('framework.') and k != 'framework.elapsed_ms'} for s in v['samples']],
                'calls': v['memory_stats_calls'], 'evaluation_source': v['evaluation_source'],
                'receipt': str(path), 'sha256': sha(path)}
        assert variants['original']['fields'] == variants['snapshot']['fields']
        assert [variants[p]['calls'] for p in ['original', 'snapshot']] == [74, 20]
        assert all(s['framework.lifecycle_peak_allocated_bytes'] == 8912896 for s in variants['original']['fields'])
        report['controls'].append({'lifetime': lifetime, 'all_fields_equal': True, 'variants': variants})
    normalized, sources = [], []
    for e in read(root/'ddp/result.json')['endpoints']:
        path = Path(e['receipt']); job = path.parents[1]; raw = read(path); computed = endpoint_summary(raw)
        assert all(e[k] == v for k, v in computed.items())
        assert computed['valid'] and len(computed['ranks']) == 2
        p = normalize_parameters(read(job/'protocol.json')['parameters'])
        policy, passive = p.pop('memory_read_policy'), p.pop('passive_gc')
        assert (policy, passive) == {'L': ('original', False), 'S': ('snapshot', False),
                                    'P': ('original', True), 'Q': ('snapshot', True)}[e['arm']]
        normalized.append(p)
        inventory = {f.relative_to(job/'source').as_posix(): sha(f) for f in (job/'source').rglob('*')
                     if f.is_file() and '__pycache__' not in f.parts}
        sources.append(inventory)
        evaluations = []
        for rank, r in zip(raw['ranks'], computed['ranks']):
            assert len(r['samples']) == 20 and r['checks_passed'] == [True, True] and r['math_policy_unchanged']
            for c in r['coverage'].values():
                assert not c['fallbacks'] and all(c['counts'].get(k, 0) > 0 for k in c['required'])
            ev = rank['result']['evaluation_source']; assert sha(ev['path']) == ev['sha256']
            assert ev['sha256'] == expected['isolated_evaluator_sha256' if policy == 'snapshot' else 'production_evaluator_sha256']
            if policy == 'snapshot':
                assert invoke_tree(Path(ev['path']).read_text())[0] == original_ast
            evaluations.append(ev)
        row = {'index': e['index'], 'arm': e['arm'], 'valid': True, 'receipt': str(path), 'receipt_sha256': sha(path),
            'median_ms': e['median_max_rank_step_ms'], 'mad_ms': e['mad_max_rank_step_ms'],
            'max_rank_samples_ms': e['max_rank_step_samples_ms'],
            'rank_samples_ms': [[s['step_ms'] for s in r['samples']] for r in computed['ranks']],
            'large_sample_indexes': [i for i, x in enumerate(e['max_rank_step_samples_ms']) if x > 200],
            'evaluation_sources': evaluations, 'gc_artifacts': []}
        paths = sorted((job/'run').glob('gc-passive-rank-*.json'))
        assert len(paths) == (2 if passive else 0)
        observers = [read(f) for f in paths]
        if observers:
            assert observers[0]['clock'] == observers[1]['clock']
        for f, obs in zip(paths, observers):
            assert obs['gc_enabled']
            timing = [t for t in obs['timing'] if t[0] == 'measure']
            assert [t[1] for t in timing] == list(range(22))
            assert [t[3] for t in timing[2:]] == row['rank_samples_ms'][obs['rank']]
            intervals = gc_intervals(obs)
            row['gc_artifacts'].append({'path': str(f), 'sha256': sha(f), 'rank': obs['rank'],
                                       'clock': obs['clock'], 'interval_count': len(intervals)})
            for interval in intervals:
                if interval['mode'] != 'measure' or interval['generation'] != 2:
                    continue
                overlaps = []
                for peer in observers:
                    for mode, index, entry, elapsed in peer['timing']:
                        if mode != 'measure':
                            continue
                        approximate_start = entry - elapsed * 1e6
                        overlap = max(0, min(interval['stop_ns'], entry) - max(interval['at_ns'], approximate_start)) / 1e6
                        if overlap:
                            overlaps.append({'rank': peer['rank'], 'sample_index': index-2, 'warmup': index < 2,
                                'elapsed_ms': elapsed, 'approximate_overlap_ms': overlap})
                report['generation2_measure'].append({'endpoint': e['index'], 'rank': obs['rank'],
                    **interval, 'overlaps': overlaps,
                    'scope': 'Metrics-entry timestamp is later than timer end by unmeasured call latency. '
                             'These intervals are approximate; shared host/boot/CLOCK_MONOTONIC verified. '
                             'Callbacks can shift GC. No causal transfer to unobserved arms.'})
        report['endpoints'].append(row)
    assert all(p == normalized[0] for p in normalized)
    assert all(s == sources[0] for s in sources)
    report['same_task_parameters_and_source'] = True
    report['source_inventory_sha256'] = hashlib.sha256(json.dumps(sources[0], sort_keys=True).encode()).hexdigest()
    report['candidate_parameters'] = normalized[0]['bundles'][0]['parameters']
    for name, directory in [('initial_failed_attempt', initial), ('corrected', root)]:
        rr = read(directory/'measurement-review/result.json')
        m, prompt, files = load(Path(rr['agent']['generation_snapshots'][0]))
        for key, data in files.items():
            assert data == (Path(m['sandbox'])/key).read_bytes()
        assert files['inputs/invoke-before.py'].decode() == expected['before']
        assert files['inputs/invoke-after.py'].decode() == expected['after']
        report['reviews'].append({'attempt': name, 'session': rr['agent']['session'], 'snapshot': rr['agent']['generation_snapshots'][0],
            'input_bytes': sum(map(len, files.values())), 'inputs': {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k,v in files.items()},
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'review': rr['review'], 'review_sha256': sha(rr['review']),
            'usage': rr['agent']['usage'], 'framework_source': m['framework_source'], 'fresh': m['fresh_session']})
    report['scope'] = 'Read-only audit. All 160 samples retained. No formal acceptance or production adoption. All old failures remain.'
    write_json(output, report)
    print(json.dumps({'audit': str(output), 'bytes': output.stat().st_size,
                      'endpoints': len(report['endpoints']), 'generation2_measure': len(report['generation2_measure'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.output)
