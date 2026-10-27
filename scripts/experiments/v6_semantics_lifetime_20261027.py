"""Small producer replays and a frozen-evaluator goal contrast, plus TP diagnostics."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.request import ProjectUnderstanding
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_autonomous_cohort_20261027 import independent
from scripts.experiments.v6_goal_switch_20261027 import intake_ab, cross_goals
from scripts.experiments.v6_box43_next_stage_20261027 import now


INTAKE_NOTE = (
    "The downstream evaluator receives the resolved description, not this proposal's top-level assumptions. "
    "Make the description self-contained: distinguish source-required task semantics from baseline implementation "
    "choices within any notes or invocation text that mentions them. An implementation choice is not a frozen "
    "requirement merely because the source uses it; identify it explicitly as baseline behavior, preserving "
    "semantics-equivalent alternatives unless a cited user/source requirement forbids them. Keep the measured "
    "invocation about the full required work, outputs and completion boundary; keep source-excluded setup/reset "
    "outside it. Do not resolve this distinction only in assumptions, propose optimizations, or omit numerical, "
    "state or relational correctness requirements."
)
LIFETIME_ANCHOR = 'Trace live Python references at the source timer, including loaded tensors unused by the forward: preserve allocations still held by source-local variables at that point.'
LIFETIME_NOTE = (
    "In understanding.state_protocol, give a concise ownership mapping from source objects that hold live "
    "device allocations at the timer to their owners in the adapter. Check that each mapped allocation "
    "actually survives prepare() in returned state or another explicit owner and remains live through run(). "
    "A comment or a temporary prepare-local variable does not retain an allocation. Report unresolved "
    "source lifetime facts without guessing them."
)


def compare_inputs(rows, output):
    if not all('agent' in row for row in rows):
        return
    (left, lp, lf), (right, rp, rf) = [load(Path(row['agent']['generation_snapshots'][0])) for row in rows]
    write_json(output/'input-comparison.json', {'same_framework': left['framework_source'] == right['framework_source'],
        'same_model': left['model'] == right['model'], 'same_tool_settings': left['tool_model_settings'] == right['tool_model_settings'],
        'changed_inputs': [k for k in sorted(set(lf)|set(rf)) if lf.get(k) != rf.get(k)],
        'prompt_diff': list(difflib.unified_diff(lp.splitlines(), rp.splitlines(), lineterm=''))})


def state_inventory(context):
    import torch
    from kernel_optimizer.project.evaluation import load_adapter
    adapter = load_adapter(Path(context['parameters']['adapter']))
    state = adapter.prepare(context, context['parameters']['request'])
    records, visited = [], set()

    def visit(value, path):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, torch.Tensor):
            records.append({'path': path, 'shape': list(value.shape), 'dtype': str(value.dtype),
                            'device': str(value.device), 'bytes': value.numel()*value.element_size()})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, path+'.'+str(key))
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, path+f'[{i}]')
    visit(state, 'state')
    return {'tensors': records, 'scope': 'Diagnostic of direct nested returned-state tensors after prepare; does not traverse arbitrary objects, model parameters or globals. Other owners require code review.'}


def evaluator_ab(args, config):
    snapshot = args.previous/'resnet-memory/generation-snapshots/generation-2ce38bd93f38.json'
    saved, prompt, inputs = load(snapshot)
    request = json.loads(inputs['inputs/request.json'])
    assert prompt.count(LIFETIME_ANCHOR) == 1
    output = args.output/'evaluator-ab'
    output.mkdir()
    write_json(output/'protocol.json', {'at': now(), 'snapshot': str(snapshot), 'framework_source': framework_source(),
        'original_framework': saved['framework_source'], 'order': ['control', 'treatment'], 'addition': LIFETIME_NOTE,
        'input_sha256': {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        'scope': 'One real fresh session per arm, one generic paragraph only; generated controls and separate state inventory. Source observation mapping remains subject to manual audit. No production prompt adoption.'})

    def validate(directory):
        understanding = ProjectUnderstanding.model_validate_json((directory/'understanding.json').read_bytes()).model_dump()
        for key in ['objective', 'workload', 'quality']:
            if understanding[key] != request[key]:
                raise ValueError('Changed frozen '+key)
        compile((directory/'adapter.py').read_bytes(), 'adapter.py', 'exec')

    rows = []
    for arm in ['control', 'treatment']:
        row = {'arm': arm}
        store = RunStore.create(output, arm, {'kind': 'evaluator-lifetime-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': 'prompt', 'old': LIFETIME_ANCHOR,
                                                 'new': LIFETIME_ANCHOR+'\n\n'+LIFETIME_NOTE}
            with Runtime(project_agent_config(config), log_dir=store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(snapshot, edit=edit, validator=validate)
            row.update(agent=receipt, understanding=json.loads((directory/'understanding.json').read_text()))
            if row['understanding']['missing_information']:
                row['status'] = 'needs_information'
            else:
                source = store.run_dir/'reference-source'; source.mkdir()
                for name, raw in inputs.items():
                    if name.startswith('inputs/source/'):
                        p = source/name.removeprefix('inputs/source/'); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
                (source/'opop_adapter.py').write_bytes((directory/'adapter.py').read_bytes())
                (source/'opop_evaluate.py').write_text('from kernel_optimizer.project.evaluation import worker_entry as run\n')
                base = dict(cwd=source, devices=['0'], lease_dir=Path(request['lease_dir']),
                            worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True)
                row['controls'] = LocalExecutor().run(ExecutionRequest(entry=source/'opop_evaluate.py', output=store.run_dir/'controls',
                    parameters={'adapter': str(source/'opop_adapter.py'), 'request': request, 'action': 'controls'}, **base))
                row['controls_valid'] = valid_run(row['controls'])
                row['state_inventory'] = LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(), function='state_inventory',
                    output=store.run_dir/'state-inventory', parameters={'adapter': str(source/'opop_adapter.py'), 'request': request}, **base))
                row['status'] = 'completed' if row['controls_valid'] else 'invalid_evaluator'
        except Exception:
            row.update(status='error', traceback=traceback.format_exc())
        rows.append(row); write_json(output/'result.json', {'runs': rows})
        print(json.dumps({'stage': 'evaluator_ab', 'arm': arm, 'status': row['status'], 'at': now()}), flush=True)
    compare_inputs(rows, output)
    return {'status': 'completed' if all(r['status'] == 'completed' for r in rows) else 'completed_with_errors', 'runs': rows}


def tp_diagnostic(args):
    previous = json.loads((args.audit/'tp-retry-protocol.json').read_text())['new_request']
    parameters = previous['parameters']
    rows = []
    for count in [1, 2]:
        request = ExecutionRequest.model_validate(previous).model_copy(update={
            'entry': Path(__file__).with_name('v6_tp_prefill_audit_20261027.py'), 'function': 'run',
            'devices': [str(i) for i in range(count)], 'distributed_backend': 'nccl' if count == 2 else None,
            'cwd': Path(__file__).resolve().parents[2], 'output': args.output/f'tp-prefill-{count}',
            'parameters': {**parameters, 'single_observations': str(args.output/'single-prefill-observations.pt')}})
        result = LocalExecutor().run(request)
        rows.append({'world_size': count, 'execution': result})
        write_json(args.output/'tp-prefill-result.json', {'runs': rows})
        if result['status'] != 'passed':
            break
    return {'status': 'completed' if len(rows) == 2 and all(r['execution']['status'] == 'passed' for r in rows) else 'completed_with_errors', 'runs': rows}


def memory_variant(args, config):
    parent = args.cohort/'resnet-latency'
    request = json.loads((parent/'request.json').read_text())
    request['goal'] = "Minimize peak_allocated_bytes for the same offline batch-32 workload, subject to images_per_second >= 0.95 of the original baseline. Preserve the frozen evaluator, original source workload, assets, quality, numerical policies and completion boundary."
    request['objective'] = {'name': 'peak_allocated_bytes', 'unit': 'bytes', 'direction': 'minimize',
                            'constraints': [{'metric': 'images_per_second', 'op': 'ge', 'baseline_ratio': .95}]}
    assert request['notes'][1].startswith('User goal selects the offline batch-32 row')
    old_goal_note = request['notes'][1]
    request['notes'][1] = ("This explicit goal variant selects the source's offline batch-32 memory goal: "
        "minimize peak_allocated_bytes with images_per_second >= 0.95 of the original baseline. "
        "latency_ms remains recorded; all input work, precision, quality and completion boundaries are unchanged.")
    request['devices'] = ['0']
    request_file = args.output/'memory-variant-request.json'; write_json(request_file, request)
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
            '--request', str(request_file), '--reuse-evaluator', str(parent), '--output', str(args.output/'resnet-memory'), '--rounds', '3']
    write_json(args.output/'variant-protocol.json', {'argv': argv, 'parent': str(parent), 'request': request,
        'replaced_goal_note': old_goal_note,
        'scope': 'Explicit goal variant of a source-audited baseline evaluator, with real contract-only adaptation and fresh controls. No old optimized bundle or developer diagnostic supplied to search. Not a fresh original-entry intake/evaluator cohort.'})
    with (args.output/'variant-controller.log').open('wb') as log:
        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    return {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode}


def variant_independent(args, config):
    result = independent(args, config, {'name': 'resnet-memory', 'case': 'resnet',
        'reference': str(args.readiness/'resnet_inference/offline-reference.pt')})
    if result['status'] != 'completed':
        return result
    run = args.output/'resnet-memory'
    prepared = json.loads((run/'prepared.json').read_text())
    parent = json.loads((args.cohort/'resnet-latency/prepared.json').read_text())
    assert prepared['adapter_sha256'] == parent['adapter_sha256'], 'Frozen evaluator bytes changed'
    request = json.loads((run/'request.json').read_text())
    source = Path(prepared['controls_path']).parent/'reference-source'
    # The reused evaluator's exact predicted_top1 mapping was manually reviewed
    # in the cohort audit; verify unchanged identity before using that recipe.
    audit = LocalExecutor().run(ExecutionRequest(entry=Path(__file__).with_name('v6_cohort_schema_audit_20261027.py'),
        function='check', cwd=source, output=args.output/'variant-reviewed-source-audit', devices=[],
        lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
        parameters={'case': 'resnet', 'request': request, 'controls': prepared['controls_path'],
                    'source_reference': str(args.readiness/'resnet_inference/offline-reference.pt')}))
    result.update(reviewed_source_audit=audit, reviewed_source_fidelity_valid=valid_run(audit),
                  reused_adapter_sha256=prepared['adapter_sha256'])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'cohort', 'previous', 'audit', 'readiness']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['intake v2 A/B', 'evaluator lifetime A/B with controls', 'TP prefill/weight diagnostics', 'frozen-evaluator memory goal variant, three rounds', 'independent full-version check', 'cross metrics'],
        'intake_addition': INTAKE_NOTE, 'evaluator_addition': LIFETIME_NOTE,
        'scope': 'No production prompt changes, API budgets or old-result edits. GPU work serial. Hypothesis replays and TP diagnostics never feed the variant search.',
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
        stage('intake_ab', lambda: intake_ab(args, config, note=INTAKE_NOTE))
        stage('evaluator_ab', lambda: evaluator_ab(args, config))
        stage('tp_prefill', lambda: tp_diagnostic(args))
        stage('resnet-memory', lambda: memory_variant(args, config))
        stage('resnet-memory-independent', lambda: variant_independent(args, config))
        stage('cross_goals', lambda: cross_goals(args))
    finally:
        terminal = {'at': now(), 'results': results, 'free_data_bytes': shutil.disk_usage(args.output).free}
        write_json(args.output/'controller-completed.json', terminal)
        write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
