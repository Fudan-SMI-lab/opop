"""Small evidence-index replay plus an explicit same-adapter DDP goal variant."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_post_batch_audit_20261027 import execute_probe


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed_evidence(old):
    endpoints = json.loads(old)
    for row in endpoints:
        # Index already collected evidence; no interpretation or GC values added.
        row['passive_gc_artifacts'] = [{'path': str(p), 'sha256': sha(p)}
            for p in sorted(Path(row['receipt']).parent.glob('gc-passive-rank-*.json'))]
    return json.dumps({'endpoints': endpoints, 'artifact_format': {
        'events': 'GC start/stop: phase, at_ns (monotonic), thread, mode, generation; Gen2 start has trigger_stack.',
        'timing': '[mode, phase_index, metrics_entry_monotonic_ns, evaluator_elapsed_ms]; measure indices 0/1 are warmup, >=2 map to sample index-2.',
        'clock': 'Host, boot_id, monotonic and perf_counter clock implementation are stored in each artifact.',
        'limits': 'Metrics entry follows timer end by unmeasured call latency. Callbacks/records can shift GC. Empty artifact lists mean no passive observer file was recorded, not no GC. No pause subtraction.'}}, indent=2)


def review_ab(args, config):
    prior = read(args.previous/'measurement-review/result.json')
    snapshot = Path(prior['agent']['generation_snapshots'][0])
    saved, prompt, inputs = load(snapshot)
    filename = 'inputs/ddp-index.json'; old = inputs[filename].decode(); new = indexed_evidence(old)
    root = args.output/'review-ab'; root.mkdir(parents=True)
    write_json(root/'protocol.json', {'at': now(), 'snapshot': str(snapshot), 'only_changed_file': filename,
        'old_bytes': len(old.encode()), 'new_bytes': len(new.encode()),
        'old_sha256': hashlib.sha256(old.encode()).hexdigest(), 'new_sha256': hashlib.sha256(new.encode()).hexdigest(),
        'scope': 'Fresh control and treatment from the same pre-generation snapshot. Only existing passive GC artifact locators/format are added. '
                 'All original endpoint values retained; no causal answer or manual audit supplied. Same prompt/model/tools/core. '
                 'External files and provider state are accessible, not exhaustively frozen. No production adoption.'})
    (root/'treatment-ddp-index.json').write_text(new)
    def validate(path):
        value = read(path/'review.json')
        if not all(k in value for k in ['findings', 'limitations', 'recommendation']):
            raise ValueError('review requires findings, limitations, recommendation')
    rows = []
    for arm in ['control', 'treatment']:
        store = RunStore.create(root, arm, {'kind': 'measurement-artifact-index-replay', 'arm': arm})
        try:
            edit = None if arm == 'control' else {'path': filename, 'old': old, 'new': new}
            with Runtime(project_agent_config(config), store.run_dir) as runtime:
                directory, receipt = ProjectAgents(runtime, store).replay_snapshot(snapshot, edit=edit, validator=validate)
            m, actual_prompt, actual_inputs = load(Path(receipt['generation_snapshots'][0]))
            changed = [k for k in sorted(set(inputs)|set(actual_inputs)) if inputs.get(k) != actual_inputs.get(k)]
            assert changed == ([] if arm == 'control' else [filename]) and actual_prompt == prompt
            assert all(m[k] == saved[k] for k in ['model', 'agent', 'tool_model_settings', 'framework_source'])
            rows.append({'arm': arm, 'status': 'completed', 'agent': receipt, 'changed_inputs': changed,
                         'same_prompt_model_settings_core': True, 'review': str(directory/'review.json')})
        except Exception:
            rows.append({'arm': arm, 'status': 'error', 'traceback': traceback.format_exc()})
        write_json(root/'result.json', {'runs': rows})
    return {'status': 'completed' if all(r['status'] == 'completed' for r in rows) else 'completed_with_errors',
            'result': str(root/'result.json')}


def step_request(parent):
    old = read(parent/'request.json'); request = copy.deepcopy(old)
    request['goal'] = ('Minimize maximum-rank clean step_ms for the same complete accumulated GPT-2 training step, '
        'aggregating aligned ranks before taking the median of repeats. Keep the existing step_ms <= 1.05x baseline constraint, '
        'all source workload, numerical/quality/replica requirements and completion boundary. '
        'Continue recording communication_kernel_sum_ms as a separate diagnostic, not wire time or the primary score.')
    request['objective'] = {'name': 'step_ms', 'unit': 'ms', 'direction': 'minimize',
        'constraints': copy.deepcopy(old['objective']['constraints']), 'rank_aggregation': 'max',
        'aggregation_order': 'ranks_then_samples'}
    request['devices'] = ['0', '1']
    assert request['notes'][6].startswith('Objective definition (')
    assert request['notes'][7].startswith('Step-time constraint semantics:')
    request['notes'][6] = request['notes'][6].replace('Objective definition (', 'Retained communication diagnostic definition (', 1)
    request['notes'][7] = request['notes'][7].replace('Step-time constraint semantics:', 'Primary step-time objective and retained constraint semantics:', 1)
    assert {k for k in old if old[k] != request[k]} == {'goal', 'objective', 'devices', 'notes'}
    assert old['workload'] == request['workload'] and old['quality'] == request['quality']
    return request


def variant(args, config):
    request = step_request(args.parent)
    request_file = args.output/'step-variant-request.json'; write_json(request_file, request)
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
        '--request', str(request_file), '--reuse-evaluator', str(args.parent),
        '--output', str(args.output/'ddp-step'), '--rounds', '3']
    write_json(args.output/'variant-protocol.json', {'at': now(), 'argv': argv, 'parent': str(args.parent),
        'request': request, 'original_request_sha256': sha(args.parent/'request.json'),
        'scope': 'Explicit same-adapter goal variant, not fresh original-entry cohort. Real contract-only adaptation and new baseline controls. '
                 'Only baseline revision is reused; no old candidate/bundle or developer diagnosis enters search. '
                 'Three rounds fixed in advance, no score-dependent retries. Quality/workload/measurement/GC policies unchanged.'})
    with (args.output/'variant-controller.log').open('wb') as log:
        p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    return {'status': 'completed' if p.returncode == 0 else 'error', 'returncode': p.returncode}


def independent(args, config):
    run = args.output/'ddp-step'
    if not (run/'prepared.json').exists() or read(run/'prepared.json')['status'] != 'prepared':
        return {'status': 'not_run', 'reason': 'No validated baseline evaluator'}
    prepared = read(run/'prepared.json'); original = read(args.parent/'prepared.json'); request = read(run/'request.json')
    source = Path(prepared['controls_path']).parent/'reference-source'
    # This mapping was manually reviewed for precisely this adapter. It is an
    # experiment source-fidelity recipe, not a production schema heuristic.
    expected = 'a4056384d718eebc668c897357ba3468306756aefa07fa77236ffb8feea3ec2d'
    assert prepared['adapter_sha256'] == original['adapter_sha256'] == sha(source/'opop_adapter.py') == expected
    audit = LocalExecutor().run(ExecutionRequest(entry=Path(__file__).with_name('v6_ddp_source_mapping_20261027.py'),
        function='check', cwd=source, output=args.output/'ddp-source-audit', devices=request['devices'][:1],
        lease_dir=Path(request['lease_dir']), worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
        parameters={'request': request, 'source': str(source), 'controls': prepared['controls_path']}))
    negative = execute_probe(source, request, args.output/'ddp-relational-negative')
    row = {'status': 'completed', 'source_fidelity': audit, 'source_fidelity_valid': valid_run(audit),
           'negative_probe': negative, 'adapter_sha256': expected}
    write_json(args.output/'independent-result.json', row)
    search = ProjectSearch(config, run)
    if search.revisions.current().bundles:
        target = args.output/'ddp-step-independent'
        fork_run(run, target, revision_id=prepared['revision'])
        comparison = promote_version(ProjectSearch(config, target), run, publish=False)
        row['independent_version'] = {k: comparison[k] for k in ['revision', 'parent', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']}
    else:
        row['independent_version'] = {'status': 'not_run', 'reason': 'No accepted optimized current version'}
    write_json(args.output/'independent-result.json', row)
    return row


def cross_goals(args):
    rows = []
    for label, run in [('communication_selected', args.parent), ('step_selected', args.output/'ddp-step')]:
        files = list(run.with_name(run.name+'-independent').glob('version-trials/*/result.json'))
        if len(files) != 1:
            rows.append({'selection_goal': label, 'status': 'unavailable', 'reason': 'No unique independent version result'})
            continue
        result, prepared = read(files[0]), read(run/'prepared.json')
        scores = {}
        for name, request in [('communication', read(args.parent/'request.json')),
                              ('step', read(args.output/'step-variant-request.json'))]:
            try:
                scores[name] = paired_decision(result['pairs'], request['objective'], prepared['evaluation'])
            except (KeyError, ValueError) as exc:
                scores[name] = {'status': 'unavailable', 'reason': str(exc)}
        rows.append({'selection_goal': label, 'source_result': str(files[0]), 'sha256': sha(files[0]),
            'original_decision': result['decision'], 'cross_scores': scores})
    result = {'status': 'completed', 'versions': rows, 'scope': 'Diagnostic cross-metric evaluation only. Different measurement periods, '
        'each with its own baseline. Never replaces original acceptance/rejection and never imports cross-contract bundles.'}
    write_json(args.output/'cross-goals.json', result); return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'parent', 'previous']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077); args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['real review evidence-index A/B', 'same-adapter DDP step goal, three rounds', 'source/negative/full-version independent checks', 'cross-goal diagnostics'],
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Production evaluator/prompts unchanged. Snapshot coalescing not adopted. No GC changes, raw local transfers, Goal or API caps. '
                 'GPU work serial, existing frozen experiments/results preserved.'})
    results = {}
    for name, operation in [('review-ab', lambda: review_ab(args, config)), ('ddp-step', lambda: variant(args, config)),
                            ('ddp-independent', lambda: independent(args, config)), ('cross-goals', lambda: cross_goals(args))]:
        write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': name})
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'followup-results.json', results)
        print(json.dumps({'stage': name, 'status': results[name]['status']}), flush=True)
    terminal = {'finished_at': now(), 'phase': 'completed' if all(r['status'] == 'completed' for r in results.values()) else 'completed_with_errors',
                'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json', terminal); write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
