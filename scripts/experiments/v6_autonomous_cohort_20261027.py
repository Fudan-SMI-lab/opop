"""One frozen source, three original-entry projects, independent follow-up audits.

Case choices and source checks live in this experiment recipe, not task routing.
Role replays never supply artifacts or feedback to the autonomous project runs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.generated import CollectionPlan
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import now


OLD_CAPABILITY = (
    'candidate_diagnostics records executed candidate Python calls, handled exceptions with propagated '
    'library traceback locations, and limited scalar returns in a separate instrumented run. '
)


def collection_replay(args, config):
    snapshot = args.previous/'ddp-raw/generation-snapshots/generation-40fef4adc280.json'
    saved, prompt, inputs = load(snapshot)
    # Obtain the exact production wording, rather than maintaining a second copy.
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(ProjectSearch.plan_collectors)))
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    production = next(value for value in constants if value.startswith('Choose useful diagnostic collectors'))
    start = production.index('candidate_diagnostics records ')
    stop = production.index('It does not compare intermediate tensors', start)
    new = production[start:stop]
    assert prompt.count(OLD_CAPABILITY) == 1
    output = args.output/'collection-ab'
    output.mkdir()
    write_json(output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'snapshot': str(snapshot), 'original_framework': saved['framework_source'],
        'order': ['control', 'treatment'], 'change': {'old': OLD_CAPABILITY, 'new': new},
        'scope': 'One fresh real role session per arm; only the capability sentence changes. No collector execution or developer feedback is injected into autonomous project runs.',
        'inputs': {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()}})
    names = set(json.loads(inputs['inputs/available-collectors.json']))
    choices = {item['revision'] for item in json.loads(inputs['inputs/revisions.json'])['candidates']}

    def validate(directory):
        plan = CollectionPlan.model_validate_json((directory/'collection.json').read_bytes())
        if set(plan.collectors)-names:
            raise ValueError('collector plan refers to an unregistered collector')
        if plan.revision is not None and plan.revision not in choices:
            raise ValueError('collection revision must belong to supplied candidates')
        if plan.collectors and plan.revision not in choices:
            raise ValueError('select one supplied revision for post-evaluation collection')

    runtime_config = project_agent_config(config)
    runtime_config.usage.mode = 'observe_only'
    rows = []
    for arm in ('control', 'treatment'):
        store = RunStore.create(output, arm, {'kind': 'collection-capability-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': 'prompt', 'old': OLD_CAPABILITY, 'new': new}
            with Runtime(runtime_config, log_dir=store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(
                    snapshot, edit=edit, validator=validate, allow_source_change=True)
            row = {'arm': arm, 'status': 'completed', 'agent': receipt,
                   'plan': json.loads((directory/'collection.json').read_text())}
        except Exception:
            row = {'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()}
        rows.append(row)
        write_json(output/'result.json', {'runs': rows})
        print(json.dumps({'stage': 'collection_ab', 'arm': arm, 'status': row['status']}), flush=True)
    return {'status': 'completed' if all(row['status'] == 'completed' for row in rows) else 'completed_with_errors', 'runs': rows}


def specifications(args):
    project = Path(__file__).resolve().parents[2]/'examples/v6_validation'
    return [
        {'name': 'qwen-ttft', 'case': 'qwen', 'devices': ['0'],
         'command': f'python qwen_inference.py --assets {shlex.quote(str(args.qwen_assets))} --output /tmp/original-qwen',
         'goal': "Minimize first-token latency for the source's interactive workload (batch 1, prompt 256 tokens, full greedy generation of 32 new tokens), preserving the source model, inputs, completion semantics and full quality requirements.",
         'reference': str(args.readiness/'qwen_inference/interactive-reference.pt')},
        {'name': 'resnet-latency', 'case': 'resnet', 'devices': ['0'],
         'command': f'python resnet_inference.py --assets {shlex.quote(str(args.resnet_assets))} --output /tmp/original-resnet',
         'goal': "Minimize full inference latency for the source's offline batch-32 workload, preserving its pretrained model, real image inputs, FP32 numerical policies, full logits and exact predicted classes.",
         'reference': str(args.readiness/'resnet_inference/offline-reference.pt')},
        {'name': 'ddp-communication', 'case': 'ddp', 'devices': ['auto'],
         'command': 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp --profile',
         'goal': "Reduce the source's diagnostic maximum-rank summed NCCL kernel duration for the accumulated training step, retaining its step-time constraint and full correctness checks."},
    ], project


def autonomous(args, config, spec, project):
    output = args.output/spec['name']
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config),
            'optimize-project', '--project', str(project), '--command', spec['command'], '--goal', spec['goal'],
            '--devices', *spec['devices'], '--output', str(output), '--rounds', '3']
    write_json(args.output/(spec['name']+'-protocol.json'), {'at': now(), 'framework_source': framework_source(),
        'argv': argv, 'specification': spec, 'scope': 'Three complete original-entry autonomous rounds; no supplied adapter, groups, candidates, replay output or developer diagnosis. Agent/API usage is observe-only, not capped.',
        'free_data_bytes': shutil.disk_usage(args.output).free})
    with (args.output/(spec['name']+'-controller.log')).open('wb') as log:
        process = subprocess.run(argv, cwd=project, stdout=log, stderr=subprocess.STDOUT)
    row = {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode,
           'output': str(output), 'free_data_bytes': shutil.disk_usage(args.output).free}
    if (output/'prepared.json').exists():
        prepared = json.loads((output/'prepared.json').read_text())
        row['preparation_status'] = prepared['status']
        if prepared['status'] == 'prepared':
            search = ProjectSearch(config, output)
            row.update(current=search.revisions.current().id, bundles=list(search.revisions.current().bundles), trials=[])
            for path in sorted((output/'trials').glob('*/result.json')):
                trial = json.loads(path.read_text())
                row['trials'].append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'decision': trial['decision'], 'selection': trial.get('selection')})
    return row


def source_reference(context):
    """Independent checks of these validation sources; never an optimization rule."""
    params = context['parameters']
    if params['case'] == 'ddp':
        from scripts.experiments.v6_box43_followup_20261027 import ddp_reference
        return ddp_reference(context)
    import torch
    from kernel_optimizer.project.evaluation import compare
    reference = torch.load(params['reference'], map_location='cpu', weights_only=True, mmap=True)
    actual = torch.load(Path(params['controls'])/'reference-rank-0.pt', map_location='cpu', weights_only=True, mmap=True)
    expected = ({'tokens': reference['tokens'], 'logits': reference['logits']} if params['case'] == 'qwen'
                else {'logits': reference['logits'], 'top1': reference['logits'].argmax(-1)})
    checks = []
    for step, observed in enumerate(actual):
        missing = sorted(set(expected)-set(observed))
        if missing:
            return {'valid': False, 'status': 'observation_schema_requires_manual_audit', 'missing_names': missing,
                    'observed_names': sorted(observed), 'scope': 'A naming mismatch does not establish numerical infidelity.'}
        checks.append({'step': step, 'additional_observations': sorted(set(observed)-set(expected)),
                       **compare({name: observed[name] for name in expected}, expected, params['request']['quality'])})
    return {'valid': len(checks) == params['request']['continuous_steps'] and all(x['passed'] for x in checks),
            'checks': checks, 'reference': params['reference'],
            'reference_sha256': hashlib.sha256(Path(params['reference']).read_bytes()).hexdigest(),
            'scope': 'Full source logits and exact discrete outputs; any additional adapter observations are reported, not independently certified by this source fixture.'}


def independent(args, config, spec):
    output = args.output/spec['name']
    if not (output/'prepared.json').exists():
        return {'status': 'not_run', 'reason': 'No completed evaluator'}
    prepared = json.loads((output/'prepared.json').read_text())
    if prepared['status'] != 'prepared':
        return {'status': 'not_run', 'reason': 'Evaluator controls did not pass'}
    request = json.loads((output/'request.json').read_text())
    source = Path(prepared['controls_path']).parent/'reference-source'
    audit_dir = args.output/(spec['name']+'-audit')
    audit = LocalExecutor().run(ExecutionRequest(
        entry=Path(__file__).resolve(), function='source_reference', cwd=source, output=audit_dir,
        devices=request['devices'][:1], lease_dir=Path(request['lease_dir']),
        worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
        parameters={'case': spec['case'], 'request': request, 'source': str(source),
                    'controls': prepared['controls_path'], 'reference': spec.get('reference')}))
    row = {'source_fidelity': audit, 'source_fidelity_valid': valid_run(audit), 'status': 'completed'}
    if spec['case'] == 'ddp':
        from scripts.experiments.v6_post_batch_audit_20261027 import execute_probe
        row['negative_probe'] = execute_probe(source, request, args.output/'ddp-new-evaluator-negative')
    search = ProjectSearch(config, output)
    if search.revisions.current().bundles:
        target = args.output/(spec['name']+'-independent')
        fork_run(output, target, revision_id=search.prepared['revision'])
        comparison = promote_version(ProjectSearch(config, target), output, publish=False)
        row['independent_version'] = {name: comparison[name] for name in ('revision', 'parent', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication')}
    else:
        row['independent_version'] = {'status': 'not_run', 'reason': 'No accepted optimized current version'}
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--qwen-assets', type=Path, required=True)
    parser.add_argument('--resnet-assets', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    specs, project = specifications(args)
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['collection-planner one-sentence A/B', *[spec['name'] for spec in specs], 'independent source audits and full-version AB/BA where available'],
        'rounds_per_project': 3, 'specifications': specs,
        'source_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(project.iterdir()) if p.is_file()},
        'scope': 'Single frozen framework. GPU work is serial. Role replays and independent checks are not injected into autonomous cohorts. No model/API cost limits; no score-dependent repetitions.',
        'free_data_bytes': shutil.disk_usage(args.output).free})
    results = {}

    def stage(name, operation):
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'controller-result.json', results)
        print(json.dumps({'stage': name, 'status': results[name]['status'], 'at': now()}), flush=True)

    try:
        stage('collection_ab', lambda: collection_replay(args, config))
        for spec in specs:
            stage(spec['name'], lambda: autonomous(args, config, spec, project))
        for spec in specs:
            stage(spec['name']+'-independent', lambda: independent(args, config, spec))
    finally:
        terminal = {'at': now(), 'results': results, 'free_data_bytes': shutil.disk_usage(args.output).free}
        write_json(args.output/'controller-completed.json', terminal)
        # Both records belong to this single controller; existing watcher protocol.
        write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
