"""Explicit device migration: fresh baseline controls/source gate, then fixed exact-byte B/C."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.project.versioning import stage_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments import v6_qwen_original_followup_20261029 as source_fixture

REVISION = 'f211410ebc62bba2546b3e6cc2dee2cff9a6085a3cfc802dce4231c4c8000511'
CODE = 'f5b76ebde100c039d8ad9264e316dbade5acdb74c02f40308e958673e802be7e'


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(args, config):
    prior = read(args.recovery / 'terminal-audit.json')
    recovery = read(args.recovery / 'recovery-audit.json')
    assert prior['terminal'] == 'completed_with_errors' and prior['second_candidate']['gpu_worker_started'] is False
    assert framework_source() == prior['core'] == recovery['core']
    original = ProjectSearch(config, Path(recovery['branch']))
    candidate = original.revisions.revision(REVISION)
    bundle = original.revisions.bundle(candidate.bundles[0])
    assert len(candidate.bundles) == 1 and bundle.parameters == {}
    assert next(f.sha256 for f in candidate.files if f.path == bundle.installer) == CODE
    devices = resolve_devices([args.device])
    assert len(devices) == 1 and devices != original.request.devices
    request = original.request.model_copy(update={'devices': devices})
    before, after = original.request.portable(), request.portable()
    assert {k for k in set(before) | set(after) if before.get(k) != after.get(k)} == {'devices'}
    assert request.continuous_steps == 2 and request.warmup == 2 and request.repeats == 5
    assert not request.diagnostic_metrics
    frozen = recovery['preserved_files'].copy()
    for p in [args.recovery / 'followup-completed.json', args.recovery / 'terminal-audit.json',
              original.output / 'project/current.json', original.output / 'latest-feedback.json',
              original.output / 'trials/trial-recovered-4f3d7f11aa/generated.json']:
        frozen[str(p)] = sha(p)
    for f in candidate.files:
        artifact = original.store.artifacts_dir / f.sha256
        assert sha(artifact) == f.sha256
        frozen[str(artifact)] = f.sha256
    assert all(sha(p) == h for p, h in frozen.items())
    reference = read(args.recovery.parent / 'qwen-end-to-end-original-20261029/protocol.json')['independent_source_reference']
    assert sha(reference['path']) == reference['sha256']
    frozen[reference['path']] = reference['sha256']
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,driver_version', '--format=csv,noheader'], text=True)
    assert devices[0] in inventory and original.request.devices[0] not in inventory
    return original, candidate, request, reference, frozen, inventory


def main(args):
    os.umask(0o077)
    config = load_config(args.config, []); config.usage.mode = 'observe_only'
    original, incoming, request, reference, frozen, inventory = preflight(args, config)
    definitions = {'at': now(), 'core': framework_source(), 'source': str(original.output),
        'source_current': original.revisions.current().id, 'incoming_revision': incoming.id,
        'code_sha256': CODE, 'parameters': {}, 'old_devices': original.request.devices,
        'new_devices': request.devices, 'inventory': inventory, 'request_changed_fields': ['devices'],
        'old_baseline': original.prepared['revision'], 'old_controls': original.prepared['controls_path'],
        'source_reference': reference, 'frozen_files': len(frozen), 'disk': disk_observation(Path('/root/autodl-tmp'))}
    if args.preflight:
        print(json.dumps({'status': 'passed', **definitions})); return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {**definitions, 'frozen_identities': frozen,
        'sequence': ['new-device original-baseline controls', 'CPU full source gate', 'fixed B', 'exact generated C'],
        'scope': 'Declared new-device measurement variant after UUID mismatch, not restart of either frozen batch. Existing prepare_variant changes only placement and generates fresh original-baseline controls/reference/timing anchor; no new Agent or goal adaptation. Reuse reviewed manual external observation mapping. Fixed exact generated second candidate, no manual code/parameter edits, search, formal ABBA acceptance, Nsight or pointer publication. Old timing anchors/acceptances remain frozen; no cross-period incremental gain. New controls are generated, not copied old references.'})
    results = {}; branch = args.output / 'qwen-fixed'
    try:
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'fresh_controls'})
        prepared = prepare_variant(original.output, request, branch, config=config)
        write_json(args.output / 'preparation-reuse.json', {k: prepared[k] for k in ['status', 'revision', 'contract_ref', 'adapter_sha256', 'controls_path', 'reuse']})
        results['prepare'] = {'status': prepared['status'], 'controls': prepared['controls_path']}
        if prepared['status'] != 'prepared': raise RuntimeError('Fresh controls invalid; no dependent source/candidate execution.')
        search = ProjectSearch(config, branch); baseline = search.revisions.current()
        assert not baseline.bundles and baseline.id == prepared['revision']
        assert baseline.anchor_id == incoming.anchor_id and baseline.contract_ref == incoming.contract_ref
        assert prepared['adapter_sha256'] == source_fixture.ADAPTER
        assert prepared['reuse']['contract_adaptation'] is None
        controls = Path(prepared['controls_path']); source = controls.parent / 'reference-source'
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'source_gate'})
        gate = LocalExecutor().run(ExecutionRequest(entry=Path(source_fixture.__file__).resolve(),
            function='reviewed_source', cwd=source, output=args.output / 'source-run', devices=['cpu'],
            lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s,
            parameters={'adapter': str(source / 'opop_adapter.py'), 'reference': reference['path'],
                'controls_reference': str(controls / 'reference-rank-0.pt'),
                'quality': request.quality, 'continuous_steps': request.continuous_steps}))
        write_json(args.output / 'source-check.json', gate)
        results['source'] = {'status': 'passed' if valid_run(gate) else 'failed', 'path': str(args.output / 'source-check.json')}
        if not valid_run(gate): raise RuntimeError('Full independent source gate failed; no B/C execution.')
        candidate = stage_version(search.revisions, original.output, revision_id=incoming.id)
        assert candidate.files == incoming.files and candidate.bundles == incoming.bundles
        assert search.revisions.current().id == baseline.id
        write_json(args.output / 'staged-identities.json', {'baseline': baseline.model_dump(mode='json'),
            'candidate': candidate.model_dump(mode='json'), 'incoming': incoming.id,
            'bundle': search.revisions.bundle(candidate.bundles[0]).model_dump(mode='json'),
            'new_controls_sha256': sha(controls / 'reference-rank-0.pt'),
            'controls_constraint_anchor': constraints_for_runs([prepared['evaluation']], request.objective, prepared['evaluation']),
            'scope': 'Byte-exact full version rebased on same frozen contract/source anchor. Fresh timing baseline belongs solely to current device variant.'})
        baseline_valid = False
        for name, revision in [('B', baseline), ('C', candidate)]:
            if name == 'C' and not baseline_valid:
                results[name] = {'status': 'not_run', 'reason': 'Fresh B quality/coverage failed.'}; continue
            write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': name})
            run = search.execute(revision, name, 'evaluate')
            path = args.output / (name + '-result.json'); write_json(path, run)
            valid = valid_run(run)
            if name == 'B': baseline_valid = valid
            results[name] = {'status': 'attempt_completed', 'worker_status': run['status'],
                'valid_run': valid, 'result': str(path), 'sha256': sha(path),
                'constraints': constraints_for_runs([run], request.objective, prepared['evaluation']) if valid else [],
                'scope': 'Fixed diagnostic only, not search or independent performance acceptance.'}
            write_json(args.output / 'followup-results.json', results)
        assert search.revisions.current().id == baseline.id
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(p) == h for p, h in frozen.items())
    assert original.revisions.current().id == definitions['source_current']
    results['preservation'] = {'status': 'passed' if same else 'failed', 'frozen_files': len(frozen)}
    terminal = {'phase': 'completed_with_errors' if 'controller' in results or not same else 'completed',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Controller completion differs from worker validity, source fidelity, constraints and formal acceptance; no formal acceptance in this batch.'}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal); write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'recovery']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', required=True); parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
