"""Validate adjacent allocator reads before any production adoption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import environment, now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import ddp_request, endpoint_summary
from scripts.experiments.v6_memory_snapshot_probe_20261028 import patched_source


def controls(args):
    rows = []
    for lifetime in ['legacy', 'invocation']:
        pair = {}
        for policy in ['original', 'snapshot']:
            output = args.output/'controls'/f'{lifetime}-{policy}'
            task = {'quality': {'atol': 0, 'rtol': 0}, 'continuous_steps': 2, 'warmup': 1, 'repeats': 2,
                    'objective': {'name': 'latency_ms', 'direction': 'minimize'}, 'measurement_lifetime': lifetime}
            request = ExecutionRequest(entry=Path(__file__).with_name('v6_memory_snapshot_probe_20261028.py'),
                function='control', cwd=Path(__file__).resolve().parents[2], devices=['0'], output=output,
                lease_dir=args.source.parent/'device-leases', worker_timeout_s=180, wait_for_devices=True,
                parameters={'request': task, 'adapter': str(output/'fixture-adapter'), 'action': 'controls',
                            'memory_read_policy': policy})
            write_json(output/'protocol.json', request.model_dump(mode='json'))
            result = LocalExecutor().run(request)
            write_json(output/'result.json', result)
            assert result['status'] == 'passed' and result['ranks'][0]['result']['valid'], result
            pair[policy] = result['ranks'][0]['result']
        def fields(value):
            return [{k: v for k, v in s.items() if k.startswith('framework.') and k != 'framework.elapsed_ms'}
                    for s in value['samples']]
        assert fields(pair['original']) == fields(pair['snapshot'])
        assert pair['snapshot']['memory_stats_calls'] < pair['original']['memory_stats_calls']
        assert all(s['framework.lifecycle_peak_allocated_bytes'] > s['framework.peak_allocated_bytes']
                   for s in pair['snapshot']['samples'])
        rows.append({'lifetime': lifetime, 'all_memory_fields_equal': True, 'memory_fields': fields(pair['snapshot']),
                     'memory_stats_calls': {k: v['memory_stats_calls'] for k, v in pair.items()},
                     'controls': {k: v['controls'] for k, v in pair.items()}})
        write_json(args.output/'controls-result.json', {'status': 'completed', 'rows': rows})
    return {'status': 'completed', 'result': str(args.output/'controls-result.json')}


def ddp(args, config):
    search = ProjectSearch(config, args.source/'ddp-communication')
    candidate = search.revisions.current()
    assert candidate.id == '75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d'
    arms = {'L': ('original', False), 'S': ('snapshot', False),
            'P': ('original', True), 'Q': ('snapshot', True)}
    order = ['L', 'S', 'P', 'Q', 'Q', 'P', 'S', 'L']
    root = args.output/'ddp'
    write_json(root/'protocol.json', {'at': now(), 'order': order, 'arms': arms, 'repeats': 20,
        'candidate': candidate.id, 'scope': 'Fixed candidate; no GC disablement or before_run instrumentation. '
        'L/S are unobserved original/snapshot evaluator copies; P/Q add identical passive GC callbacks. '
        'All samples retained; no formal acceptance, performance-score adjustment or production adoption.'})
    rows = []
    for index, arm in enumerate(order):
        job = root/f'{index}-{arm}'
        req = ddp_request(search, candidate, job, arm='C', action='evaluate')
        req = req.model_copy(update={
            'entry': Path(__file__).with_name('v6_memory_snapshot_probe_20261028.py'), 'function': 'run',
            'parameters': {**req.parameters, 'memory_read_policy': arms[arm][0], 'passive_gc': arms[arm][1]}})
        write_json(job/'protocol.json', req.model_dump(mode='json'))
        before = environment(search.request.devices)
        result = LocalExecutor().run(req)
        row = {'index': index, 'arm': arm, 'receipt': str(job/'run/receipt.json'),
               'before': before, 'after': environment(search.request.devices), **endpoint_summary(result)}
        rows.append(row); write_json(root/'result.json', {'endpoints': rows})
        print(json.dumps({'stage': 'ddp', 'index': index, 'arm': arm, 'valid': row['valid']}), flush=True)
    return {'status': 'completed' if all(r['valid'] for r in rows) else 'completed_with_errors',
            'result': str(root/'result.json')}


def review(args, config, results):
    root = args.output/'measurement-review'
    store = RunStore.create(root.parent, root.name, {'kind': 'measurement_semantics_review'})
    _, _, before, after = patched_source()
    inputs = {'invoke-before.py': before, 'invoke-after.py': after, 'stage-index.json': json.dumps(results)}
    if (args.output/'controls-result.json').exists():
        inputs['cuda-controls.json'] = (args.output/'controls-result.json').read_text()
    if (args.output/'ddp/result.json').exists():
        rows = json.loads((args.output/'ddp/result.json').read_text())['endpoints']
        inputs['ddp-index.json'] = json.dumps([{'arm': r['arm'], 'index': r['index'], 'valid': r['valid'],
            'receipt': r['receipt'], 'all_max_rank_step_samples_ms': r['max_rank_step_samples_ms']} for r in rows], indent=2)
    instructions = ('Review only the proposed coalescing of adjacent CUDA allocator statistics reads. '
        'Compare before/after code, real CUDA lifetime/quality controls and the indexed fixed-candidate endpoints. '
        'Identify any changed field meaning, timer boundary, lifecycle ownership or asynchronous-allocation caveat. '
        'Read indexed raw receipts only for a concrete uncertainty, not unrelated history. '
        'Do not run GPU work or modify source/configuration. No GC disablement, no before_run observer, no '
        'formal candidate acceptance and no automatic production adoption occur in this experiment. '
        'Do not infer original long-tail causes from another endpoint or subtract GC time from scores. '
        'Write review.json with findings, limitations and recommendation; distinguish checked facts from hypotheses. '
        'The existing DDP independent rejection and TP full-model failure remain unchanged.')
    def validate(path):
        value = json.loads((path/'review.json').read_text())
        if not all(k in value for k in ['findings', 'limitations', 'recommendation']):
            raise ValueError('review requires findings, limitations, recommendation')
    with Runtime(project_agent_config(config), root) as runtime:
        directory, receipt = ProjectAgents(runtime, store).generate(
            'measurement_review', inputs, instructions, ('review.json',), validator=validate)
    result = {'status': 'completed', 'agent': receipt, 'review': str(directory/'review.json')}
    write_json(root/'result.json', result); return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'source']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077); args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    old, new, before, after = patched_source()
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'production_evaluator_sha256': hashlib.sha256(old.encode()).hexdigest(),
        'isolated_evaluator_sha256': hashlib.sha256(new.encode()).hexdigest(),
        'before': before, 'after': after, 'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Only isolated worker evaluator copies change; production core and original experiments unchanged. '
                 'No bulk local transfer, Goal, API caps or formal acceptance.'})
    results = {}
    for name, operation in [('cuda-controls', lambda: controls(args)), ('ddp', lambda: ddp(args, config)),
                            ('measurement-review', lambda: review(args, config, results))]:
        write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': name})
        if name == 'ddp' and results['cuda-controls']['status'] != 'completed':
            results[name] = {'status': 'skipped_failed_controls'}
        else:
            try:
                results[name] = operation()
            except Exception:
                results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'followup-results.json', results)
    result = {'finished_at': now(), 'phase': 'completed' if all(v['status'] == 'completed' for v in results.values())
              else 'completed_with_errors', 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json', result); write_json(args.output/'followup-completed.json', result)


if __name__ == '__main__':
    main()
