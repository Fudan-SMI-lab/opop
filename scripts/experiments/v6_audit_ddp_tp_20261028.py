"""Read-only audit of the frozen diagnostic batch, producing a small evidence index."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics

from kernel_optimizer.project.communication import trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import endpoint_summary
from scripts.experiments.v6_tp_block_audit_20261027 import difference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_parameters(parameters):
    result = copy.deepcopy(parameters)
    result['adapter'] = Path(result['adapter']).name
    for bundle in result['bundles']:
        for field in ['file', 'specs_path']:
            bundle[field] = Path(bundle[field]).name
    return result


def audit(root, output):
    import torch
    report = {'root': str(root), 'terminal': json.loads((root/'followup-completed.json').read_text())['phase'],
              'ddp': [], 'timelines': [], 'tp': [], 'artifacts': []}
    protocols = []
    frozen = json.loads((root/'ddp-fixed/protocol.json').read_text())
    ddp = json.loads((root/'ddp-fixed/result.json').read_text())
    for endpoint in ddp['clean']:
        receipt = Path(endpoint['receipt'])
        raw = json.loads(receipt.read_text())
        computed = endpoint_summary(raw)
        assert all(computed[k] == endpoint[k] for k in computed)
        assert len(computed['ranks']) == 2 and computed['valid']
        assert all(len(r['samples']) == 20 and r['checks_passed'] == [True, True]
                   and r['math_policy_unchanged'] for r in computed['ranks'])
        job = receipt.parents[1]
        protocol = json.loads((job/'protocol.json').read_text())
        protocols.append(normalize_parameters(protocol['parameters']))
        source = job/'source'
        manifest = json.loads((source/'.opop-revision.json').read_text())
        # Actual source inventory is indexed independently of path-normalized requests.
        source_hashes = {p.relative_to(source).as_posix(): sha(p)
                         for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        comm_samples = [r['metric_samples']['communication_kernel_sum_ms'] for r in computed['ranks']]
        assert all(len(v) == 3 for v in comm_samples)
        comm = [max(v[i] for v in comm_samples) for i in range(3)]
        report['ddp'].append({'index': endpoint['index'], 'arm': endpoint['arm'], 'valid': True,
            'step_samples_max_rank_ms': computed['max_rank_step_samples_ms'],
            'step_samples_per_rank_ms': [[s['step_ms'] for s in r['samples']] for r in computed['ranks']],
            'step_median_ms': computed['median_max_rank_step_ms'], 'step_mad_ms': computed['mad_max_rank_step_ms'],
            'comm_profile_samples_max_rank_ms': comm, 'comm_median_ms': statistics.median(comm),
            'over_200ms_sample_indexes': [i for i, x in enumerate(computed['max_rank_step_samples_ms']) if x > 200],
            'coverage': [r['coverage'] for r in computed['ranks']],
            'receipt': str(receipt), 'receipt_sha256': sha(receipt), 'source_hashes': source_hashes})
    c, n = protocols[1], copy.deepcopy(protocols[2])
    assert c['bundles'][0]['parameters'] == {'per_collective_align': 'barrier', 'pre_backward_align': 'none'}
    assert n['bundles'][0]['parameters'] == {'per_collective_align': 'none', 'pre_backward_align': 'none'}
    n['bundles'][0]['parameters']['per_collective_align'] = 'barrier'
    assert c == n and protocols[1] == protocols[4] and protocols[2] == protocols[3] and protocols[0] == protocols[5]
    request = copy.deepcopy(c['request']); request['repeats'] = frozen['request']['repeats']
    assert request == frozen['request']
    assert report['ddp'][1]['source_hashes'] == report['ddp'][2]['source_hashes']
    report['request_check'] = {'C_N_only_difference': 'per_collective_align',
        'request_only_change': 'repeats 5 -> 20', 'C_N_source_bytes_equal': True,
        'quality_and_coverage': 'all ranks, all endpoints passed; two continuous checks each',
        'scope': 'Diagnostic ablation, not a new candidate version or formal acceptance'}
    for profile in ddp['profiles']:
        for item in profile['ranks']:
            trace_path = Path(item['trace']); assert sha(trace_path) == item['trace_sha256']
            raw = json.loads(trace_path.read_text()); trace, window = trace_window(raw, 'opop.timeline_workload')
            ranges = [v for v in trace['traceEvents'] if v.get('ph') == 'X' and
                      v.get('cat') != 'gpu_user_annotation' and
                      any(s in v.get('name', '').lower() for s in ['c10d', 'gloo'])]
            assert dict(Counter(v['name'] for v in ranges)) == item['range_inventory']
            detailed = json.loads(Path(item['report']).read_text())
            comm = detailed['communication']['collectives']
            max_gloo = max((v for v in ranges if v['name'] == 'gloo:barrier'),
                           key=lambda v: v.get('dur', 0), default=None)
            record = {'arm': profile['arm'], 'rank': item['rank'], 'index': item['index'],
                'trace': str(trace_path), 'sha256': sha(trace_path), 'range_inventory': item['range_inventory'],
                'max_gloo': max_gloo, 'cpu_distributed_ranges': item['cpu_distributed_ranges'],
                'collectives': len(comm), 'clock': detailed['clock'], 'profiled_wall_ms': item['profiled_wall_ms']}
            report['timelines'].append(record)
        report.setdefault('cross_rank', {})[profile['arm']] = profile['cross_rank']
    tp = json.loads((root/'tp-result.json').read_text())
    for rank in tp['ranks']:
        value = rank['result']; output_path = Path(value['artifact']['path'])
        assert sha(output_path) == value['artifact']['sha256']
        saved = torch.load(output_path, map_location='cpu', weights_only=True)
        observations = []
        for source in value['source_observations']:
            assert sha(source['path']) == source['sha256']
            observations.append(torch.load(source['path'], map_location='cpu', weights_only=True, mmap=True))
        for projection in value['projections']:
            name = projection['name']; assert projection['status'] == 'compared'
            assert projection['same_input']['exact_equal'] and projection['checkpoint_row_slice']['exact_equal']
            refs = {'against_recorded_single_slice': observations[0]['tensors'][name+'.output'].chunk(2, -1)[rank['rank']],
                    'against_recorded_tp': observations[1]['tensors'][name+'.output'],
                    'against_official': saved[name+'.official_module']}
            for ref_name, expected in refs.items():
                for variant, claimed in projection[ref_name].items():
                    actual = difference(saved[name+'.'+variant], expected)
                    # CPU reduction order may change tiny RMS round-off; exact/max are direct checks.
                    assert all(actual[k] == claimed[k] for k in ['status', 'exact_equal', 'elements', 'max_abs_error'])
            report['tp'].append({'rank': rank['rank'], 'name': name, 'weight_mapping_exact': True,
                'input_mapping_exact': True, 'raw_output_sha256': sha(output_path),
                **{ref: {k: {'exact': v['exact_equal'], 'max_abs': v['max_abs_error']}
                         for k, v in projection[ref].items()} for ref in refs}})
    review = json.loads((root/'diagnostic-review/result.json').read_text())
    manifest, prompt, files = load(Path(review['agent']['generation_snapshots'][0]))
    report['agent'] = {'session': review['agent']['session'], 'usage': review['agent']['usage'],
        'snapshot': review['agent']['generation_snapshots'][0], 'fresh': manifest['fresh_session'],
        'framework_source': manifest['framework_source'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'inputs': {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k, v in files.items()},
        'total_input_file_bytes': sum(map(len, files.values())), 'review': review['review'], 'review_sha256': sha(review['review'])}
    report['limitations'] = ['Original independent failures unchanged. No formal acceptance in this batch.',
        'Gloo async range is not the Python blocking-call duration; keep tids and raw events.',
        'CPU/GPU clocks, relative CUDA event origins and profiler timestamps require evidenced mappings.',
        'The reviewer output requires human audit; exact local TP replay does not mean exact source full GEMM.']
    write_json(output, report)
    print(json.dumps({'audit': str(output), 'bytes': output.stat().st_size, 'endpoints': len(report['ddp']),
                      'timeline_traces': len(report['timelines']), 'tp_projections': len(report['tp']),
                      'input_bytes': report['agent']['total_input_file_bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.output)
