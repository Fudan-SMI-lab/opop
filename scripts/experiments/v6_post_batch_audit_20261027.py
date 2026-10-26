"""Fixed-candidate sampling and source-specific evaluator audit, outside core routing.

The perturbation is an experiment on this recorded source's exact replica
requirement, not an optimizer rule. The role replay changes one prompt paragraph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.request import ProjectUnderstanding
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import environment, now


RELATION_NOTE = (
    "Per-rank tensor tolerance comparisons do not enforce relationships between outputs or participants. "
    "Implement the source-defined relational correctness checks in observe(), outside timed run(), "
    "and expose exact boolean/integer observations or raise on violation. Preserve the full required "
    "observations. Declaring a required relationship only in understanding.json is insufficient."
)
PROMPT_ANCHOR = "Do not put elapsed times or performance counters into correctness observations."


def negative_probe(context):
    """Run actual DDP, then change one last-gradient element on rank 1 only."""
    import torch
    import torch.distributed as dist
    from kernel_optimizer.project.evaluation import compare, load_adapter, snapshot

    params = context['parameters']
    request = params['request']
    adapter = load_adapter(Path(params['adapter']))
    state = adapter.prepare(context, request)
    if callable(getattr(adapter, 'set_mode', None)):
        adapter.set_mode(state, 'quality')
    adapter.reset(state)
    if callable(getattr(adapter, 'before_run', None)):
        adapter.before_run(state)
    result = adapter.run(state)
    torch.cuda.synchronize(context['device'])
    baseline = snapshot(adapter.observe(state, result))
    seen, gradients = set(), []
    for root, model in adapter.models(state).items():
        for name, parameter in model.named_parameters():
            if id(parameter) not in seen and parameter.grad is not None and parameter.grad.is_floating_point():
                seen.add(id(parameter))
                gradients.append((root + '.' + name, parameter.grad))
    if not gradients:
        raise ValueError('No actual model gradients exposed for this source audit')
    name, grad = gradients[-1]
    # Last parameter avoids leaving peers in a subsequent collective if observe
    # implements the source's sequential broadcast-and-assert replica check.
    index = (0,) * grad.ndim
    before = grad[index].detach().clone()
    replicated = before.clone()
    dist.broadcast(replicated, src=0)
    if not torch.equal(before, replicated):
        raise ValueError('Probe requires initially equal selected replica gradients')
    delta = float(request['quality']['atol']) / 4
    if delta <= 0:
        raise ValueError('This probe requires a positive source reference tolerance')
    if context['rank'] == 1:
        with torch.no_grad():
            grad[index].add_(delta)
    after = grad[index].detach().clone()
    change = float((after-before).abs())
    tolerance_check = compare(after.cpu(), before.cpu(), request['quality'])
    if context['rank'] == 1 and (change == 0 or not tolerance_check['passed']):
        raise ValueError('Perturbation must be nonzero and within the original scalar tolerance')
    dist.barrier()
    report = {'probe_executed': True, 'rank': context['rank'], 'parameter': name,
              'delta_requested': delta, 'actual_abs_change': change,
              'scalar_within_tolerance': tolerance_check['passed'],
              'exact_replica_equal_after': bool(torch.equal(after, replicated)),
              'scope': 'Mutation only in this isolated worker after a complete step, outside timing'}
    try:
        perturbed = snapshot(adapter.observe(state, result))
        check = compare(perturbed, baseline, request['quality'])
        report.update(detected=not check['passed'], detection='observation_comparison', comparison=check)
    except AssertionError as exc:
        report.update(detected=True, detection='observe_assertion', error=str(exc)[:2000])
    except Exception:
        report.update(detected=None, detection='observe_execution_error', traceback=traceback.format_exc())
    return report


def execute_probe(source, request, output):
    run = LocalExecutor().run(ExecutionRequest(
        entry=Path(__file__).resolve(), function='negative_probe', cwd=source,
        output=output, devices=request['devices'], distributed_backend=request['distributed_backend'],
        lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'],
        wait_for_devices=True, parameters={'adapter': str(source/'opop_adapter.py'), 'request': request}))
    rows = [rank.get('result') or {} for rank in run.get('ranks', [])]
    valid = (run['status'] == 'passed' and len(rows) == len(request['devices'])
             and all(row.get('probe_executed') and row.get('detected') is not None for row in rows)
             and any(row.get('actual_abs_change', 0) > 0 and not row['exact_replica_equal_after'] for row in rows))
    return {'status': 'completed' if valid else 'error', 'probe_valid': valid,
            'detected': any(row.get('detected') is True for row in rows) if valid else None,
            'run': run}


def baseline_probe(args):
    prepared = json.loads((args.prior/'ddp-raw/prepared.json').read_text())
    request = json.loads((args.prior/'ddp-raw/request.json').read_text())
    source = Path(prepared['controls_path']).parent/'reference-source'
    write_json(args.output/'probe-protocol.json', {
        'at': now(), 'framework_source': framework_source(), 'source': str(source),
        'adapter_sha256': hashlib.sha256((source/'opop_adapter.py').read_bytes()).hexdigest(),
        'scope': 'Source-specific negative audit. Original experiment and controls remain unchanged.',
        'perturbation': 'After full step and valid observation, rank 1 last floating gradient first element += original atol/4; call observe again and compare all observations.',
        'decision': 'Only a successfully executed, within-tolerance, nonzero exact replica mismatch can establish missing detection. Execution errors are not detection evidence.'})
    return execute_probe(source, request, args.output/'baseline-negative-worker')


def qwen_samples(args, config):
    old = args.prior/'qwen-fixed'
    protocol = json.loads((old/'protocol.json').read_text())
    output = args.output/'qwen-sampling'
    fork_run(old, output)
    search = ProjectSearch(config, output)
    original = search.request.portable()
    search.request = search.request.model_copy(update={'repeats': 20})
    order = ['B', 'C', 'C', 'B']
    write_json(output/'sampling-protocol.json', {
        'at': now(), 'framework_source': framework_source(), 'original_protocol': str(old/'protocol.json'),
        'versions': protocol['versions'], 'order': order, 'original_request': original,
        'diagnostic_request': search.request.portable(), 'difference': 'In-memory repeats 5 -> 20 only; frozen on-disk request unchanged.',
        'scope': 'Developer fixed-version temporal diagnostic. No publication, sample deletion, score-dependent repetitions or revision of the earlier failed acceptance.'})
    rows = []
    for position, arm in enumerate(order):
        revision = search.revisions.revision(protocol['versions'][arm])
        before = environment(search.request.devices)
        run = search.execute(revision, f'sampling-{position}-{arm}', 'evaluate')
        row = {'position': position, 'arm': arm, 'valid': valid_run(run), 'before': before,
               'after': environment(search.request.devices), 'run': run}
        if row['valid']:
            values = run['ranks'][0]['result']['samples']
            if len(values) != 20:
                raise ValueError(f'Expected 20 actual samples, received {len(values)}')
            row['ttft_blocks_ms'] = [statistics.median(v['ttft_ms'] for v in values[i:i+5]) for i in range(0, 20, 5)]
        rows.append(row)
        write_json(output/'result.json', {'endpoints': rows})
        print(json.dumps({'stage': 'qwen_sampling', 'position': position, 'arm': arm, 'valid': row['valid']}), flush=True)
        if not row['valid']:
            return {'status': 'endpoint_failed', 'output': str(output)}
    return {'status': 'completed', 'output': str(output),
            'blocks': [{'arm': row['arm'], 'position': row['position'], 'ttft_blocks_ms': row['ttft_blocks_ms']} for row in rows]}


def source_audit(context):
    from scripts.experiments.v6_box43_followup_20261027 import ddp_reference
    return ddp_reference(context)


def evaluator_ab(args, config):
    original_probe = json.loads((args.output/'probe-result.json').read_text())
    if not original_probe['probe_valid'] or original_probe['detected']:
        return {'status': 'not_run', 'reason': 'The original negative probe did not establish a missing check.'}
    snapshot_path = args.prior/'ddp-raw/generation-snapshots/generation-7265bc838390.json'
    saved, prompt, inputs = load(snapshot_path)
    request = json.loads(inputs['inputs/request.json'])
    if prompt.count(PROMPT_ANCHOR) != 1:
        raise ValueError('Expected one observe paragraph in the initial snapshot')
    changed = prompt.replace(PROMPT_ANCHOR, PROMPT_ANCHOR + '\n\n' + RELATION_NOTE)
    output = args.output/'evaluator-ab'
    output.mkdir()
    write_json(output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'original_snapshot': str(snapshot_path), 'original_framework': saved['framework_source'],
        'order': ['control', 'treatment'], 'difference': 'One generic paragraph appended to the observe contract; all captured files and other prompt text unchanged.',
        'note': RELATION_NOTE, 'model': saved['model'],
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        'prompt_sha256': {arm: hashlib.sha256(value.encode()).hexdigest() for arm, value in [('control', prompt), ('treatment', changed)]},
        'scope': 'One fresh-session replay per arm with existing automatic contract repair and no API/cost budget. Controls, source fidelity and negative probes are separate real GPU checks, not autonomous optimization.'})
    runtime_config = project_agent_config(config)
    runtime_config.usage.mode = 'observe_only'

    def validate(directory):
        understanding = ProjectUnderstanding.model_validate_json((directory/'understanding.json').read_bytes()).model_dump()
        for field in ('objective', 'workload', 'quality'):
            if understanding[field] != request[field]:
                raise ValueError(f'Agent changed frozen {field}')
        compile((directory/'adapter.py').read_bytes(), 'adapter.py', 'exec')

    rows = []
    for arm in ('control', 'treatment'):
        store = RunStore.create(output, arm, {'kind': 'source-relationship-evaluator-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': 'prompt', 'old': PROMPT_ANCHOR, 'new': PROMPT_ANCHOR+'\n\n'+RELATION_NOTE}
            with Runtime(runtime_config, log_dir=store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(snapshot_path,
                    edit=edit, validator=validate, allow_source_change=True)
            understanding = json.loads((directory/'understanding.json').read_text())
            row = {'arm': arm, 'status': 'generated', 'agent': receipt, 'understanding': understanding}
            if understanding['missing_information']:
                row.update(status='needs_information')
            else:
                source = store.run_dir/'reference-source'
                source.mkdir()
                for name, raw in inputs.items():
                    if name.startswith('inputs/source/'):
                        path = source/name.removeprefix('inputs/source/')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(raw)
                (source/'opop_adapter.py').write_bytes((directory/'adapter.py').read_bytes())
                (source/'opop_evaluate.py').write_text('from kernel_optimizer.project.evaluation import worker_entry as run\n')
                controls = store.run_dir/'controls'
                row['controls'] = LocalExecutor().run(ExecutionRequest(
                    entry=source/'opop_evaluate.py', cwd=source, devices=request['devices'],
                    distributed_backend=request['distributed_backend'], output=controls,
                    lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'],
                    wait_for_devices=True, parameters={'adapter': str(source/'opop_adapter.py'), 'request': request, 'action': 'controls'}))
                row['controls_valid'] = valid_run(row['controls'])
                if row['controls_valid']:
                    row['negative_probe'] = execute_probe(source, request, store.run_dir/'negative-worker')
                    row['source_fidelity'] = LocalExecutor().run(ExecutionRequest(
                        entry=Path(__file__).resolve(), function='source_audit', cwd=source,
                        output=store.run_dir/'source-audit', devices=request['devices'][:1],
                        lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'],
                        wait_for_devices=True, parameters={'source': str(source), 'controls': str(controls), 'request': request}))
                    row['source_fidelity_valid'] = valid_run(row['source_fidelity'])
                row['status'] = 'completed' if row['controls_valid'] else 'invalid_evaluator'
        except Exception:
            row = {**(row if 'row' in locals() and row.get('arm') == arm else {}),
                   'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()}
        rows.append(row)
        write_json(output/'result.json', {'runs': rows})
        print(json.dumps({'stage': 'evaluator_ab', 'arm': arm, 'status': row['status']}), flush=True)
    return {'status': 'completed', 'output': str(output), 'runs': rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['probe', 'followup'], required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.stage == 'probe':
        try:
            results['baseline_negative'] = baseline_probe(args)
        except Exception:
            results['baseline_negative'] = {'status': 'error', 'probe_valid': False, 'detected': None,
                                            'traceback': traceback.format_exc()}
        write_json(args.output/'probe-result.json', results['baseline_negative'])
        write_json(args.output/'controller-completed.json', {'at': now(), 'results': results})
    else:
        config = load_config(args.config, [])
        for name, operation in [('qwen_sampling', qwen_samples), ('evaluator_ab', evaluator_ab)]:
            try:
                results[name] = operation(args, config)
            except Exception:
                results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
            write_json(args.output/'followup-result.json', results)
        write_json(args.output/'followup-completed.json', {'at': now(), 'results': results})
    print(json.dumps({'at': now(), 'stages': {key: value['status'] for key, value in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
