"""Reviewed external observation mapping, then the unstarted original-entry search.

This does not change the generated adapter, request, contract or production core.
The source comparison is a developer audit fixture, never a candidate input.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_original_end_to_end_20261029 import search, independent, native

# Identity of the code whose observe() mapping was manually reviewed. This is
# provenance of this external audit, not an allowlist in the optimizer.
ADAPTER = '498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7'
MAPPING = {'tokens': 'generated_tokens', 'logits': 'step_logits'}


def now():
    return datetime.now(timezone.utc).isoformat()


def reviewed_source(context):
    import torch
    from kernel_optimizer.project.evaluation import compare
    p = context['parameters']
    assert sha(p['adapter']) == ADAPTER
    reference = torch.load(p['reference'], map_location='cpu', weights_only=True, mmap=True)
    actual = torch.load(p['controls_reference'], map_location='cpu', weights_only=True, mmap=True)
    expected = {key: reference[key] for key in MAPPING}
    checks = []
    for index, observation in enumerate(actual):
        mapped = {key: observation[name] for key, name in MAPPING.items()}
        checks.append({'step': index, **compare(mapped, expected, p['quality']),
                       'exact_tokens': bool(torch.equal(mapped['tokens'], expected['tokens'])),
                       'finite_logits': bool(torch.isfinite(mapped['logits']).all()),
                       'additional_observations': sorted(set(observation) - set(MAPPING.values()))})
    token_negative = {**expected, 'tokens': expected['tokens'].clone()}
    token_negative['tokens'].view(-1)[0] += 1
    logits_negative = {**expected, 'logits': expected['logits'].clone()}
    element = logits_negative['logits'].view(-1)
    element[0] += element[0].abs() * 10 + 1
    negatives = {'integer_token_plus_one': compare(token_negative, expected, p['quality']),
                 'logit_outside_tolerance': compare(logits_negative, expected, p['quality'])}
    valid = (len(checks) == p['continuous_steps'] and
             all(c['passed'] and c['exact_tokens'] and c['finite_logits'] for c in checks) and
             all(not c['passed'] for c in negatives.values()))
    return {'valid': valid, 'mapping': MAPPING, 'adapter_sha256': ADAPTER,
            'source_reference_sha256': sha(p['reference']),
            'controls_reference_sha256': sha(p['controls_reference']), 'checks': checks,
            'negative_checks': negatives,
            'scope': 'CPU full comparison of fresh real CUDA controls to independent source tensors; no GPU rerun, sample selection or tolerance change. Additional adapter observations and ownership are separate code review.'}


def source_gate(args, config):
    parent = ProjectSearch(config, args.parent / 'qwen-original')
    controls = Path(parent.prepared['controls_path'])
    source = controls.parent / 'reference-source'
    reference = read(args.parent / 'protocol.json')['independent_source_reference']
    assert sha(reference['path']) == reference['sha256']
    assert parent.prepared['adapter_sha256'] == sha(source / 'opop_adapter.py') == ADAPTER
    assert not parent.revisions.current().bundles
    run = LocalExecutor().run(ExecutionRequest(
        entry=Path(__file__).resolve(), function='reviewed_source', cwd=source,
        output=args.output / 'reviewed-source-run', devices=['cpu'],
        lease_dir=Path(parent.request.lease_dir),
        worker_timeout_s=parent.request.worker_timeout_s,
        parameters={'adapter': str(source / 'opop_adapter.py'), 'reference': reference['path'],
                    'controls_reference': str(controls / 'reference-rank-0.pt'),
                    'quality': parent.request.quality, 'continuous_steps': parent.request.continuous_steps}))
    write_json(args.output / 'reviewed-source-check.json', run)
    return {'status': 'completed' if valid_run(run) else 'source_check_failure',
            'path': str(args.output / 'reviewed-source-check.json')}


def branch(args, config):
    source = args.parent / 'qwen-original'
    parent = ProjectSearch(config, source)
    destination = args.output / 'qwen-original'
    revision = fork_run(source, destination)
    fresh = ProjectSearch(config, destination)
    assert not revision.bundles and fresh.revisions.current().id == revision.id == parent.prepared['revision']
    assert fresh.request == parent.request
    assert fresh.prepared == parent.prepared
    assert not (destination / 'latest-feedback.json').exists()
    identities = {f.path: f.sha256 for f in revision.files}
    for f in revision.files:
        import hashlib
        assert hashlib.sha256(fresh.store.get_artifact(f.content_ref)).hexdigest() == f.sha256
    row = {'status': 'completed', 'revision': revision.id, 'bundles': [], 'files': identities,
           'contract_ref': revision.contract_ref, 'request_sha256': sha(destination / 'request.json'),
           'shared_controls': fresh.prepared['controls_path'], 'parent_pointer_unchanged': parent.revisions.current().id,
           'scope': 'Existing APIs fork the completed fresh original-entry preparation only. No new evaluator, controls, source copies, discovery or developer feedback supplied. Previous failed source audit remains frozen.'}
    write_json(args.output / 'preparation-reuse.json', row)
    return row


def main(args):
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    prior = read(args.parent / 'followup-completed.json')
    assert prior['results']['prepare']['status'] == 'completed'
    assert all(prior['results'][k]['status'] == 'not_run' for k in ['search', 'independent', 'native'])
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(),
        'parent': str(args.parent), 'parent_terminal_sha256': sha(args.parent / 'followup-completed.json'),
        'adapter_sha256': ADAPTER, 'reviewed_observation_mapping': MAPPING,
        'sequence': ['reviewed external source check', 'fork completed preparation',
                     'previously unstarted three autonomous rounds', 'conditional independent AB/BA', 'conditional native'],
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Manual external fixture mapping only; no adapter/request/contract/core/prompt edits or GPU controls rerun. Continuation of fresh original-entry task, not a second independent evaluator cohort or retry of rejected candidates.'})
    results = {}
    for name, operation in [('source', source_gate), ('branch', branch), ('search', search),
                            ('independent', independent), ('native', native)]:
        write_json(args.output / 'state.json', {'phase': 'running', 'stage': name, 'at': now()})
        if any(results.get(k, {}).get('status') not in [None, 'completed'] for k in ['source', 'branch', 'search']):
            results[name] = {'status': 'not_run', 'reason': 'Required stage failed; preserve failure, no automatic retry.'}
        elif name == 'native' and results.get('independent', {}).get('quality') != 'pass':
            results[name] = {'status': 'not_run', 'reason': 'No independent full-quality-passing active version.'}
        else:
            try:
                results[name] = operation(args, config)
            except Exception:
                results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output / 'followup-results.json', results)
        print(json.dumps({'stage': name, 'status': results[name]['status'], 'at': now()}), flush=True)
    terminal = {'phase': 'completed' if all(r['status'] in ['completed', 'not_run'] for r in results.values())
                else 'completed_with_errors', 'finished_at': now(), 'results': results,
                'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'parent']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
