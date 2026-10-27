"""Fixed-version diagnostics after storage maintenance; never publish candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, ProjectContract
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.aggregation import rank_requirements
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import environment, now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def evaluator(context):
    # Identical production evaluator; the experiment changes only the explicit
    # diagnostic repeat count and N's already-supported candidate parameter.
    from kernel_optimizer.project.evaluation import worker_entry
    return worker_entry(context)


def ddp_request(search, revision, destination, *, arm, action):
    source = search.revisions.materialize(revision, destination/'source')
    contract = search.revisions.load(revision.contract_ref, ProjectContract)
    bundles = []
    for ident in revision.bundles:
        bundle = search.revisions.bundle(ident)
        parameters = dict(bundle.parameters)
        if arm == 'N':
            assert parameters == {'pre_backward_align': 'none', 'per_collective_align': 'barrier'}
            parameters['per_collective_align'] = 'none'
        bundles.append({'id': ident, 'file': str(source/bundle.installer), 'parameters': parameters,
            'specs_path': str(source/Path(bundle.installer).with_suffix('.targets.json')),
            'expected_calls': list(bundle.calls), 'expected_calls_by_rank': rank_requirements(
                bundle, search.revisions.load(search.revisions.revision(bundle.parent_revision).plan_ref,
                                             ExecutionPlan), contract.world_size)})
    request = search.request.portable()
    request['repeats'] = 20
    return ExecutionRequest(entry=Path(__file__), function='evaluator', cwd=source,
        output=destination/'run', devices=search.request.devices,
        distributed_backend=search.request.distributed_backend, lease_dir=search.request.lease_dir,
        worker_timeout_s=search.request.worker_timeout_s, wait_for_devices=True,
        parameters={'request': request, 'adapter': str(source/'opop_adapter.py'), 'action': action,
                    'reference_dir': search.prepared['controls_path'], 'bundles': bundles,
                    'collector_options': {'repeats': 2, 'gradient_readiness': False}})


def endpoint_summary(execution):
    ranks = []
    for rank in execution.get('ranks', []):
        result = rank.get('result') or {}
        ranks.append({'rank': rank['rank'], 'valid': result.get('valid'), 'metrics': result.get('metrics'),
            'samples': result.get('samples', []), 'metric_samples': result.get('metric_samples', {}),
            'checks_passed': [c.get('passed') for c in result.get('checks', [])],
            'coverage': result.get('coverage', {}), 'math_policy_unchanged': result.get('math_policy_unchanged')})
    aligned = []
    if valid_run(execution):
        samples = [r['samples'] for r in ranks]
        if len({len(s) for s in samples}) != 1:
            raise ValueError('rank samples must remain aligned')
        aligned = [max(r[i]['step_ms'] for r in samples) for i in range(len(samples[0]))]
    median = statistics.median(aligned) if aligned else None
    return {'status': execution['status'], 'valid': valid_run(execution), 'ranks': ranks,
            'max_rank_step_samples_ms': aligned, 'median_max_rank_step_ms': median,
            'mad_max_rank_step_ms': statistics.median(abs(x-median) for x in aligned) if aligned else None}


def timeline_summary(execution):
    from kernel_optimizer.project.communication import trace_window
    from kernel_optimizer.project.timeline import cross_rank
    rows, by_index = [], {}
    for rank in execution.get('ranks', []):
        for index, name in enumerate((rank.get('result') or {}).get('reports', [])):
            report = json.loads(Path(name).read_text())
            raw = json.loads(Path(report['trace']).read_text())
            trace, window = trace_window(raw, 'opop.timeline_workload')
            waits = [r for r in trace.get('traceEvents', []) if r.get('ph') == 'X'
                     and r.get('cat') != 'gpu_user_annotation'
                     and any(s in r.get('name', '').lower() for s in ('c10d', 'gloo'))]
            groups = {}
            for item in waits:
                key = item['name']
                entry = groups.setdefault(key, {'count': 0, 'sum_duration_us': 0, 'max_duration_us': 0})
                entry['count'] += 1
                entry['sum_duration_us'] += item.get('dur', 0)
                entry['max_duration_us'] = max(entry['max_duration_us'], item.get('dur', 0))
            by_index.setdefault(index, []).append(report)
            rows.append({'rank': rank['rank'], 'index': index, 'report': name,
                'trace': report['trace'], 'trace_sha256': report['trace_sha256'], 'window': window,
                'profiled_wall_ms': report['profiled_wall_ms'], 'cpu_distributed_ranges': groups,
                'range_inventory': dict(Counter(r.get('name') for r in waits)),
                'cuda_synchronization': report['cuda_synchronization'],
                'devices': {k: {a: b for a, b in v.items() if a not in
                    {'communication_intervals_us', 'compute_intervals_us'}} for k, v in report['devices'].items()}})
    return {'status': execution['status'], 'ranks': rows,
            'cross_rank': {str(i): cross_rank(v) for i, v in by_index.items()},
            'scope': 'Separate profiler runs. CPU ranges can nest/overlap: sums are not additive costs. '
                     'Empty ranges do not prove absence. NCCL duration is not wire time or causal critical path.'}


def ddp(args, config):
    search = ProjectSearch(config, args.previous/'ddp-communication')
    candidate = search.revisions.current()
    parent = search.revisions.revision(candidate.parent_id)
    assert candidate.id == '75e28d3d2ccd5fab06175ef1b827a325ed2159d011758c7b2deccda6173b9e2d'
    assert not parent.bundles and len(candidate.bundles) == 1
    versions = {'B': parent, 'C': candidate, 'N': candidate}
    root = args.output/'ddp-fixed'
    protocol = {'at': now(), 'source': str(search.output), 'versions': {k: v.id for k, v in versions.items()},
        'order': ['B', 'C', 'N', 'N', 'C', 'B'], 'repeats': 20, 'warmup': search.request.warmup,
        'parameter_change_N_only': {'per_collective_align': ['barrier', 'none']},
        'request': search.request.portable(), 'original_evaluation': search.prepared['evaluation'],
        'references': search.prepared['controls_path'], 'candidate_parameters':
            search.revisions.bundle(candidate.bundles[0]).parameters,
        'profile_order': ['B', 'C', 'N'], 'profile_repeats': 2,
        'scope': 'Fixed developer diagnostics; original quality, continuous steps and full work retained. '
                 'N is a parameter ablation, not the frozen accepted version. No publication, '
                 'new formal acceptance, sample removal or until-pass repeats.'}
    write_json(root/'protocol.json', protocol)
    rows, profiles = [], []
    valid_arms = {arm: True for arm in versions}
    for index, arm in enumerate(protocol['order']):
        job = root/f'clean-{index}-{arm}'
        req = ddp_request(search, versions[arm], job, arm=arm, action='evaluate')
        write_json(job/'protocol.json', req.model_dump(mode='json'))
        before = environment(search.request.devices)
        execution = LocalExecutor().run(req)
        row = {'index': index, 'arm': arm, 'before': before, 'after': environment(search.request.devices),
               'receipt': str(job/'run/receipt.json'), **endpoint_summary(execution)}
        rows.append(row)
        valid_arms[arm] &= row['valid']
        write_json(root/'clean-result.json', {'endpoints': rows, 'scope': protocol['scope']})
        print(json.dumps({'stage': 'ddp_clean', 'index': index, 'arm': arm, 'valid': row['valid']}), flush=True)
    for arm in protocol['profile_order']:
        if not valid_arms[arm]:
            profiles.append({'arm': arm, 'status': 'skipped_quality_or_execution_failed'})
            continue
        job = root/f'timeline-{arm}'
        req = ddp_request(search, versions[arm], job, arm=arm, action='execution_timeline')
        write_json(job/'protocol.json', req.model_dump(mode='json'))
        execution = LocalExecutor().run(req)
        profiles.append({'arm': arm, 'receipt': str(job/'run/receipt.json'), **timeline_summary(execution)})
        write_json(root/'timeline-result.json', {'profiles': profiles})
    result = {'status': 'completed' if all(valid_arms.values()) and all(
        p['status'] == 'passed' for p in profiles) else 'completed_with_errors',
        'clean': rows, 'profiles': profiles, 'scope': protocol['scope']}
    write_json(root/'result.json', result)
    return {'status': result['status'], 'result': str(root/'result.json')}


def tp(args):
    old = json.loads((args.previous/'tp-block-2-protocol.json').read_text())
    framework = Path(__file__).resolve().parents[2]
    overlay = args.previous.parent/'goal-switch-audit-20261027/tp-dependencies'
    req = ExecutionRequest.model_validate(old).model_copy(update={
        'entry': Path(__file__).with_name('v6_tp_colwise_diagnostic_20261028.py'), 'function': 'run',
        'cwd': framework, 'output': args.output/'tp-colwise',
        'env': {**old['env'], 'PYTHONPATH': os.pathsep.join(map(str, [overlay, framework/'src', framework]))},
        'parameters': {**old['parameters'], 'prior_tp': str(args.previous/'tp-block-2')}})
    write_json(args.output/'tp-protocol.json', req.model_dump(mode='json'))
    result = LocalExecutor().run(req)
    write_json(args.output/'tp-result.json', result)
    return {'status': result['status'], 'result': str(args.output/'tp-result.json')}


def review(args, config, results):
    """A real evidence-review role; no candidate generation or production adoption."""
    root = args.output/'diagnostic-review'
    store = RunStore.create(root.parent, root.name, {'kind': 'developer_diagnostic_review'})
    inputs = {'stage-index.json': json.dumps(results, indent=2),
              'scope.txt': 'Prior DDP independent rejection and TP full 32-token logits failure remain. '
                           'This is a diagnostic review, not autonomous optimization or acceptance.'}
    ddp_root = args.output/'ddp-fixed'
    for filename in ['protocol.json', 'clean-result.json', 'timeline-result.json']:
        path = ddp_root/filename
        if path.exists():
            data = json.loads(path.read_text())
            if filename == 'clean-result.json':
                # Keep all rank samples but leave large coverage maps in indexed receipts.
                for row in data['endpoints']:
                    for rank in row['ranks']:
                        rank.pop('coverage', None)
            inputs['ddp-'+filename] = json.dumps(data, indent=2)
    tp_path = args.output/'tp-result.json'
    if tp_path.exists():
        raw = json.loads(tp_path.read_text())
        inputs['tp-summary.json'] = json.dumps({'status': raw['status'],
            'ranks': [{'rank': r['rank'], 'result': r.get('result'), 'error': r.get('error')}
                      for r in raw.get('ranks', [])]}, indent=2)
    search = ProjectSearch(config, args.previous/'ddp-communication')
    revision = search.revisions.current()
    installer = search.revisions.bundle(revision.bundles[0]).installer
    for name in ['opop_adapter.py', installer]:
        file = next(f for f in revision.files if f.path == name)
        inputs[Path(name).name] = search.store.get_artifact(file.content_ref)
    inputs['timeline-collector.py'] = (Path(__file__).resolve().parents[2]/
                                     'src/kernel_optimizer/project/timeline.py').read_bytes()
    instructions = (
        'Review the fixed DDP B/C/N diagnostic endpoints and TP same-input projection evidence. '
        'Read the supplied scope and indexed protocols first. Use existing evidence to distinguish '
        'observations, hypotheses and unresolved causes. You may inspect the indexed raw SERVER files '
        'if the summary leaves a specific uncertainty; do not preload unrelated experiments. '
        'Do not run GPU work, generate a candidate, modify files outside this sandbox, or change any '
        'prompt, workload, metric or threshold. Write review.json with findings (a list, each with '
        'claim, evidence_paths, limitations), unresolved (list), and next_minimal_experiment (object). '
        'State whether the existing collector exposes the information actually needed by a candidate '
        'agent; propose only a justified small gap, with a falsifiable test, if one remains. '
        'Do not infer Gloo wait absence from missing ranges, sum nested CPU ranges as additive cost, '
        'claim NCCL wire time, call a parameter ablation a frozen full version, or turn local BF16/FP32 '
        'arithmetic into full-model quality acceptance. Preserve all prior independent failures. '
        'No production change is adopted automatically from your answer.')

    def validate(directory):
        data = json.loads((directory/'review.json').read_text())
        if not isinstance(data.get('findings'), list) or not isinstance(data.get('unresolved'), list) \
                or not isinstance(data.get('next_minimal_experiment'), dict):
            raise ValueError('review must contain findings/unresolved lists and next_minimal_experiment object')

    with Runtime(project_agent_config(config), root) as runtime:
        directory, receipt = ProjectAgents(runtime, store).generate(
            'diagnostic_review', inputs, instructions, ('review.json',), validator=validate)
    result = {'status': 'completed', 'agent': receipt, 'review': str(directory/'review.json')}
    write_json(root/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'previous']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'previous': str(args.previous), 'stages': ['ddp-fixed', 'tp-colwise', 'real-diagnostic-review'],
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Server-only evidence; serial leased GPU execution, fixed endpoints, no publication. '
                 'No old raw evidence deletion, bulk local transfer, API caps or production prompt changes.'})
    results = {}
    for name, operation in [('ddp-fixed', lambda: ddp(args, config)), ('tp-colwise', lambda: tp(args)),
                            ('real-diagnostic-review', lambda: review(args, config, results))]:
        write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': name})
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'followup-results.json', results)
        print(json.dumps({'stage': name, 'status': results[name]['status']}), flush=True)
    errors = {k: v['status'] for k, v in results.items() if v['status'] not in {'passed', 'completed'}}
    terminal = {'finished_at': now(), 'phase': 'completed_with_errors' if errors else 'completed',
                'errors': errors, 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json', terminal)
    write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
