"""Audit the completed GC experiment without editing its frozen evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import endpoint_summary
from scripts.experiments.v6_audit_ddp_tp_20261028 import normalize_parameters
from scripts.experiments.v6_gc_review_batch_20261028 import gc_summary, compact_tp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, output):
    results = json.loads((root/'ddp-gc/result.json').read_text())
    report = {'root': str(root), 'endpoints': [], 'generation2_measure': [], 'review_replay': []}
    normalized, inventories = [], []
    for e in results['endpoints']:
        receipt = Path(e['receipt']); raw = json.loads(receipt.read_text()); computed = endpoint_summary(raw)
        assert all(e[k] == computed[k] for k in computed)
        assert computed['valid'] and len(computed['ranks']) == 2
        assert all(len(r['samples']) == 20 and r['checks_passed'] == [True, True] and
                   r['math_policy_unchanged'] for r in computed['ranks'])
        for rank in computed['ranks']:
            for coverage in rank['coverage'].values():
                assert not coverage['fallbacks'] and all(coverage['counts'].get(c, 0) > 0 for c in coverage['required'])
        job = receipt.parents[1]
        p = normalize_parameters(json.loads((job/'protocol.json').read_text())['parameters'])
        mode = p.pop('gc_probe'); normalized.append(p)
        inventories.append({f.relative_to(job/'source').as_posix(): sha(f) for f in (job/'source').rglob('*')
                            if f.is_file() and '__pycache__' not in f.parts})
        row = {'index': e['index'], 'arm': e['arm'], 'mode': mode, 'valid': True,
            'median_ms': e['median_max_rank_step_ms'], 'mad_ms': e['mad_max_rank_step_ms'],
            'max_rank_samples_ms': e['max_rank_step_samples_ms'],
            'per_rank_samples_ms': [[s['step_ms'] for s in r['samples']] for r in computed['ranks']],
            'large_sample_indexes': [i for i, v in enumerate(e['max_rank_step_samples_ms']) if v > 200],
            'receipt': str(receipt), 'receipt_sha256': sha(receipt), 'gc': []}
        for recorded in e['gc']:
            path = Path(recorded['artifact']); assert sha(path) == recorded['sha256']
            assert gc_summary(path, 2) == recorded
            data = json.loads(path.read_text())
            assert len(recorded['samples']) == 20 and not recorded['unclosed_gc_intervals']
            assert all(s['gc_enabled'] == (e['arm'] == 'O') for s in recorded['samples'])
            row['gc'].append({'rank': recorded['rank'], 'artifact': str(path), 'sha256': sha(path),
                'measured_nr_throttled': sum(s['cgroup_cpu_delta']['nr_throttled'] for s in recorded['samples']),
                'measured_throttled_usec': sum(s['cgroup_cpu_delta']['throttled_usec'] for s in recorded['samples']),
                'sample_gc_overlap_ms': [sum(v['bracket_overlap_ms'] for v in s['gc_overlaps']) for s in recorded['samples']]})
            pending = {}
            for event in data['events']:
                key = event['thread'], event['generation']
                if event['phase'] == 'start':
                    pending[key] = event
                    continue
                first = pending.pop(key)
                if first['generation'] != 2 or first['mode'] != 'measure':
                    continue
                overlaps = []
                for other in (job/'run').glob('gc-rank-*.json'):
                    peer = json.loads(other.read_text())
                    for invocation in peer['invocations']:
                        if invocation['mode'] != 'measure' or 'end_ns' not in invocation:
                            continue
                        overlap = max(0, min(event['at_ns'], invocation['end_ns'])-
                                      max(first['at_ns'], invocation['start_ns']))/1e6
                        if overlap:
                            overlaps.append({'rank': peer['rank'], 'sample_index': invocation['index']-2,
                                'warmup': invocation['index'] < 2, 'elapsed_ms': invocation['evaluator_elapsed_ms'],
                                'bracket_overlap_ms': overlap})
                report['generation2_measure'].append({'endpoint': e['index'], 'rank': data['rank'],
                    'start_ns': first['at_ns'], 'stop_ns': event['at_ns'],
                    'duration_ms': (event['at_ns']-first['at_ns'])/1e6,
                    'invocation_at_start': first['invocation'], 'trigger_stack': first['trigger_stack'],
                    'collected': event['collected'], 'overlaps': overlaps,
                    'scope': 'Host monotonic timestamps from the same local dual-rank executor; brackets include '
                             'boundary overhead. Cross-rank temporal overlap alone is not full causal proof.'})
        report['endpoints'].append(row)
    assert all(p == normalized[0] for p in normalized)
    assert all(p == inventories[0] for p in inventories)
    report['same_task_and_source'] = True
    report['source_inventory_sha256'] = hashlib.sha256(json.dumps(inventories[0], sort_keys=True).encode()).hexdigest()
    old = root.parent/'ddp-tp-diagnostics-20261028/diagnostic-review/result.json'
    initial = json.loads(old.read_text())
    m, prompt, files = load(Path(initial['agent']['generation_snapshots'][0]))
    expected = compact_tp(files['inputs/tp-summary.json']).encode()
    for arm in json.loads((root/'review-ab/result.json').read_text())['runs']:
        n, actual_prompt, actual_files = load(Path(arm['agent']['generation_snapshots'][0]))
        changed = [k for k in sorted(set(files)|set(actual_files)) if files.get(k) != actual_files.get(k)]
        assert prompt == actual_prompt
        assert all(m[k] == n[k] for k in ['model', 'agent', 'framework_source', 'tool_model_settings'])
        assert changed == ([] if arm['arm'] == 'control' else ['inputs/tp-summary.json'])
        if arm['arm'] == 'treatment':
            assert actual_files['inputs/tp-summary.json'] == expected
        report['review_replay'].append({'arm': arm['arm'], 'session': arm['agent']['session'],
            'snapshot': arm['agent']['generation_snapshots'][0], 'changed_inputs': changed,
            'same_prompt_model_settings_core': True, 'fresh': n['fresh_session'],
            'review': arm['review'], 'review_sha256': sha(arm['review']), 'usage': arm['agent']['usage'],
            'input_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in actual_files.items()}})
    report['scope'] = ('Full frozen run audit. Diagnostic GC disablement is not an optimized candidate. '
                       'No samples removed and all prior independent failures remain.')
    write_json(output, report)
    print(json.dumps({'audit': str(output), 'bytes': output.stat().st_size,
                      'verified_endpoints': len(report['endpoints']), 'review_arms': len(report['review_replay'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.root, a.output)
