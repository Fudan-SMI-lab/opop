"""One intake prompt A/B, TP baseline readiness and an original-entry goal switch."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.intake import IntakeProposal, resolve_request
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_autonomous_cohort_20261027 import autonomous, independent
from scripts.experiments.v6_box43_next_stage_20261027 import now


INTAKE_ANCHOR = 'statements. Record assumptions separately.'
INTAKE_NOTE = (
    "Separate required task semantics from the source's implementation choices. "
    "The baseline evaluator must reproduce the original implementation, but a choice is not a frozen "
    "requirement merely because it appears in the source. In description fields (including notes and "
    "invocation), preserve user/source-required workload, numerical behavior, outputs and completion "
    "boundaries; cite the requirement that makes a restriction necessary. Record incidental implementation "
    "choices as baseline behavior in assumptions, without prohibiting semantics-preserving alternatives "
    "or proposing an optimization."
)


def intake_ab(args, config, *, note=INTAKE_NOTE):
    snapshot = args.cohort/'ddp-communication.intake/generation-snapshots/generation-4b715484e927.json'
    saved, prompt, inputs = load(snapshot)
    assert prompt.count(INTAKE_ANCHOR) == 1
    environment = json.loads(inputs['inputs/deployment.json'])
    source = Path(json.loads(inputs['inputs/repository.json'])['root'])
    goal = json.loads(inputs['inputs/intent.json'])['goal']
    output = args.output/'intake-ab'
    output.mkdir()
    write_json(output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'snapshot': str(snapshot), 'original_framework': saved['framework_source'],
        'order': ['control', 'treatment'], 'difference': note,
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        'scope': 'One fresh real intake session per arm; one added generic paragraph only. No production prompt adoption and no artifacts injected into the independent goal-switch run.'})

    def validate(directory):
        proposal = IntakeProposal.model_validate_json((directory/'intake.json').read_bytes())
        resolve_request(proposal, source, goal, environment)

    runtime_config = project_agent_config(config)
    runtime_config.usage.mode = 'observe_only'
    rows = []
    for arm in ('control', 'treatment'):
        store = RunStore.create(output, arm, {'kind': 'intake-semantics-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': 'prompt', 'old': INTAKE_ANCHOR,
                                                 'new': INTAKE_ANCHOR+'\n\n'+note}
            with Runtime(runtime_config, log_dir=store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(
                    snapshot, edit=edit, validator=validate)
            proposal = IntakeProposal.model_validate_json((directory/'intake.json').read_bytes())
            request = resolve_request(proposal, source, goal, environment)
            row = {'arm': arm, 'status': proposal.status, 'proposal': proposal.model_dump(mode='json'),
                   'request': request.portable() if request else None, 'agent': receipt}
        except Exception:
            row = {'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()}
        rows.append(row)
        write_json(output/'result.json', {'runs': rows})
        print(json.dumps({'stage': 'intake_ab', 'arm': arm, 'status': row['status'], 'at': now()}), flush=True)
    if all('agent' in row for row in rows):
        snapshots = [load(Path(row['agent']['generation_snapshots'][0])) for row in rows]
        (left, lp, lf), (right, rp, rf) = snapshots
        write_json(output/'input-comparison.json', {
            'same_framework': left['framework_source'] == right['framework_source'],
            'same_model': left['model'] == right['model'],
            'same_tool_settings': left['tool_model_settings'] == right['tool_model_settings'],
            'changed_inputs': [name for name in sorted(set(lf)|set(rf)) if lf.get(name) != rf.get(name)],
            'prompt_diff': list(difflib.unified_diff(lp.splitlines(), rp.splitlines(), lineterm=''))})
    return {'status': 'completed' if all(row['status'] != 'error' for row in rows) else 'completed_with_errors', 'runs': rows}


def tp_readiness(args):
    request = json.loads((args.cohort/'ddp-communication/request.json').read_text())
    rows = []
    for count in (1, 2):
        result = LocalExecutor().run(ExecutionRequest(
            entry=Path(__file__).with_name('v6_tp_qwen_readiness_20261027.py'), function='run',
            cwd=args.project, output=args.output/f'qwen-tp{count}-readiness',
            devices=[str(i) for i in range(count)], distributed_backend='nccl' if count == 2 else None,
            lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
            parameters={'source': str(args.project), 'assets': str(args.qwen_assets),
                        'reference': str(args.readiness/'qwen_inference/interactive-reference.pt')}))
        rows.append({'world_size': count, 'source_fidelity_valid': valid_run(result), 'execution': result})
        write_json(args.output/'tp-readiness.json', {'runs': rows})
    return {'status': 'completed' if all(row['execution']['status'] == 'passed' for row in rows) else 'completed_with_errors',
            'ready_for_optimization': all(row['source_fidelity_valid'] for row in rows), 'runs': rows}


def cross_goals(args):
    """Rescore already fixed endpoints; never alter or publish their contracts."""
    rows = []
    for name, run in [('latency_selected', args.cohort/'resnet-latency'),
                      ('memory_selected', args.output/'resnet-memory')]:
        directory = run.with_name(run.name+'-independent')/'version-trials'
        paths = list(directory.glob('*/result.json'))
        if len(paths) != 1:
            rows.append({'version': name, 'status': 'unavailable', 'reason': 'No unique independent final-version result'})
            continue
        result = json.loads(paths[0].read_text())
        prepared = json.loads((run/'prepared.json').read_text())
        scores = {}
        for key, objective in {
            'latency': {'name': 'framework.elapsed_ms', 'direction': 'minimize'},
            'memory_with_throughput': {'name': 'framework.peak_allocated_bytes', 'direction': 'minimize',
                'constraints': [{'metric': 'images_per_second', 'op': 'ge', 'baseline_ratio': .95}]},
        }.items():
            try:
                scores[key] = paired_decision(result['pairs'], objective, prepared['evaluation'])
            except (KeyError, ValueError) as exc:
                scores[key] = {'status': 'unavailable', 'reason': str(exc)}
        rows.append({'version': name, 'source_result': str(paths[0]),
                     'source_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                     'actual_objective': json.loads((run/'request.json').read_text())['objective'],
                     'original_decision': result['decision'], 'cross_metric_scores': scores,
                     'lifecycle_memory': [{'pair': i, 'arm': arm,
                         'ranks': [{'rank': r['rank'], 'metrics': {k: v for k, v in r['result']['metrics'].items() if 'bytes' in k}}
                                   for r in endpoint['ranks']]}
                         for i, pair in enumerate(result['pairs']) for arm, endpoint in pair.items()]})
    result = {'status': 'completed', 'scope': 'Retrospective cross-goal diagnostic from all original independent samples, not a new acceptance/publication. Separate runs may have different adapters/anchors; audit comparability and source fidelity before conclusions. No cross-contract bundle import.', 'versions': rows}
    write_json(args.output/'cross-goals.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'cohort', 'qwen-assets', 'resnet-assets', 'readiness']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    args.project = Path(__file__).resolve().parents[2]/'examples/v6_validation'
    config = load_config(args.config, [])
    spec = {'name': 'resnet-memory', 'case': 'resnet', 'devices': ['0'],
        'command': f'python resnet_inference.py --assets {shlex.quote(str(args.resnet_assets))} --output /tmp/original-resnet-memory',
        'goal': "Minimize the source's peak_allocated_bytes for the offline batch-32 workload while retaining images_per_second >= 0.95 of the original baseline, preserving its pretrained model, real image inputs, FP32 numerical policies, full logits and exact predicted classes. Keep the original workload and completion boundary.",
        'reference': str(args.readiness/'resnet_inference/offline-reference.pt')}
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['intake A/B', 'single GPU and official TP=2 baseline readiness', 'ResNet memory three original-entry rounds', 'independent source/full-version checks', 'cross-goal diagnostic'],
        'resnet_specification': spec, 'intake_note_not_in_production': INTAKE_NOTE,
        'tp_protocol': 'Original Qwen interactive B1/S256/N32 loop, BF16/SDPA, exact tokens and original full logits tolerance; 2 quality invocations, 2 warmups, 5 samples, one separate full-workload profile per rank; single GPU then official TP=2. Quality failures retained, not optimized.',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.project.iterdir() if p.is_file()},
        'free_data_bytes': shutil.disk_usage(args.output).free,
        'scope': 'No production prompt mutation, no score-dependent extra runs, no model/API budget. Frozen old experiments untouched. New snapshot archives source/scripts/examples only to avoid duplicating historical evidence.'})
    results = {}

    def stage(name, operation):
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'controller-result.json', results)
        print(json.dumps({'stage': name, 'status': results[name]['status'], 'at': now()}), flush=True)

    try:
        stage('intake_ab', lambda: intake_ab(args, config))
        stage('tp_readiness', lambda: tp_readiness(args))
        stage('resnet-memory', lambda: autonomous(args, config, spec, args.project))
        stage('resnet-memory-independent', lambda: independent(args, config, spec))
        stage('cross_goals', lambda: cross_goals(args))
    finally:
        terminal = {'at': now(), 'results': results, 'free_data_bytes': shutil.disk_usage(args.output).free}
        write_json(args.output/'controller-completed.json', terminal)
        write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
