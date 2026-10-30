"""One independent complete-version B/C/C/B for the exact accepted DDP output."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

CODE = 'ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b'
ACCEPTED = '4541830b6c81c1c0c2c2fef6dfb5f2566e312483900238abf1e7ac900a1758e6'
BASELINE = '7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
AUDIT = '07fc3b7b3fa2f535204582bc95f7aa027724b4efa9bfdc8d2adfc71d4358d8df'


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    a = read(args.parent / 'terminal-audit.json'); p = read(args.parent / 'protocol.json')
    assert sha(args.parent / 'terminal-audit.json') == AUDIT
    assert read(args.parent / 'followup-completed.json')['phase'] == 'completed'
    assert a['core'] == p['core'] == framework_source()
    assert a['decision']['accepted'] and a['candidate_sha256'] == CODE
    source = ProjectSearch(config, args.parent / 'ddp-search')
    current = source.revisions.current()
    assert current.id == ACCEPTED and source.prepared['revision'] == BASELINE
    assert len(current.bundles) == 1 and not source.revisions.revision(BASELINE).bundles
    bundle = source.revisions.bundle(current.bundles[0])
    assert not bundle.parameters and {f.path: f.sha256 for f in current.files}[bundle.installer] == CODE
    assert current.contract_ref == source.revisions.revision(BASELINE).contract_ref
    assert resolve_devices(source.request.devices) == p['devices']
    assert (source.request.continuous_steps, source.request.warmup, source.request.repeats) == (2, 2, 5)
    assert len(source.request.diagnostic_metrics) == 1
    assert a['measurement']['constraints'][0]['original_reference'] == 133.379621
    assert a['measurement']['constraints'][0]['limit'] == 140.04860205

    prior = args.parent.parent / 'ddp-communication-device-20261030'
    check = read(prior / 'source-check.json'); neg = read(prior / 'rank-negative-check.json')
    assert sha(prior / 'source-check.json') == 'cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c'
    assert check['status'] == 'passed' and check['run']['ranks'][0]['result']['valid']
    assert check['adapter_sha256'] == source.prepared['adapter_sha256']
    assert neg['probe_valid'] and neg['detected']
    sr = check['run']['ranks'][0]['result']
    assert len(sr['checks']) == 6 and all(v['passed'] for c in sr['checks'] for k, v in c.items() if isinstance(v, dict))
    assert len(sr['exact_replica_gradients']) == 2 and all(c['passed'] for c in sr['exact_replica_gradients'])
    sq = read(prior / 'source-worker/request.json')['parameters']
    assert sq['controls'] == source.prepared['controls_path'] and sq['request'] == source.request.portable()
    files = {str(prior / n): sha(prior / n) for n in ['source-check.json', 'rank-negative-check.json', 'source-worker/request.json', 'rank-negative-worker/request.json', 'terminal-audit.json']}
    for path in Path(sq['source']).glob('*'):
        if path.is_file(): files[str(path)] = sha(path)
    for rank, expected in [(0, '1b1c2ddc768e6d1651dd2575f4506616fd1126a838bab680566c88e9b67ba49f'), (1, 'b1a93181d73466f7c0b75765578ad201e11e1f90cb985ed5e85731d95fdbe80a')]:
        path = Path(sq['controls']) / f'reference-rank-{rank}.pt'
        assert sha(path) == expected; files[str(path)] = expected
    reuse = {'at': now(), 'status': 'passed', 'files': files, 'prior_checks': sr['checks'],
             'exact_replica_gradients': sr['exact_replica_gradients'], 'rank_negative_detected': True,
             'scope': 'Exact-hash reuse of prior fresh-current-device full source/controls and exact-replica negative evidence only; no fresh tensor comparison, source/model/control rerun or reference copy. Reviewed manual source/metric mappings remain assistance; full source raw tensors were not saved.'}
    native = []
    for n in a['native']:
        assert sha(n['path']) == n['sha256']
        native.append({k: n[k] for k in ['rank', 'path', 'sha256', 'normalized_sha256', 'kernels', 'nccl']})
    norm = args.parent / 'ddp-search/jobs/nsys-e69b7ae138/run/nsys-normalized.json'
    assert all(sha(norm) == n['normalized_sha256'] for n in native)
    frozen = dict(p['frozen_identities']); frozen.update(files)
    for name in ['protocol.json', 'terminal-audit.json', 'terminal-numerical-audit.json', 'public-terminal-audit.json', 'followup-completed.json']:
        frozen[str(args.parent / name)] = sha(args.parent / name)
    for name in ['request.json', 'prepared.json', 'discovery.json', 'latest-feedback.json', 'project/current.json']:
        frozen[str(source.output / name)] = sha(source.output / name)
    for path in (source.output / 'jobs').glob('*/run/**/*'):
        if path.is_file() and not path.is_symlink(): frozen[str(path)] = sha(path)
    for path in (source.output / 'sandboxes').glob('*/candidate.py'): frozen[str(path)] = sha(path)
    for path in (source.output / 'sandboxes').glob('*/proposal.json'): frozen[str(path)] = sha(path)
    assert all(sha(path) == h for path, h in frozen.items())
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', '/root/autodl-tmp'], text=True).splitlines()[-1])
    assert free > 9 * 1024**3
    return source, frozen, reuse, {'normalized_path': str(norm), 'ranks': native, 'graph': a['graph_native'],
        'scope': 'Reuse exact accepted-code current-device native records, no new Nsys. 4103 kernels, 2981 Graph kernels, two versioned cudaGraphLaunch APIs/rank; 13 AllReduce remain eager. This is compute scheduling evidence, not communication-volume reduction or clean timing attribution.'}, free


def main(args):
    os.umask(0o077)
    cfg = load_config(args.config, []); cfg.usage.mode = 'observe_only'
    source, frozen, reuse, native, free = prepare(args, cfg)
    info = {'core': framework_source(), 'source': str(source.output), 'accepted_revision': ACCEPTED,
            'baseline': BASELINE, 'candidate_sha256': CODE, 'bundles': list(source.revisions.current().bundles),
            'parameters': {}, 'devices': source.request.devices, 'frozen_files': len(frozen),
            'free_bytes': free, 'additional_budget_bytes': 1024**3, 'warning_floor_bytes': 8 * 1024**3,
            'original_step_anchor_ms': 133.379621, 'step_limit_ms': 140.04860205}
    if args.preflight: print(json.dumps({'status': 'passed', **info})); return
    args.output.mkdir(parents=True, exist_ok=True); assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {'at': now(), **info, 'frozen_identities': frozen,
        'sequence': ['hash-only source evidence reuse', 'one exact complete-version B/C/C/B'], 'publish': False,
        'scope': 'Necessary independent closure of genuinely new search acceptance. No new Agent/search/parameter choice, manual edit, source/controls rerun, dedicated native collection or production adoption. Original full work/quality/step constraint/rank aggregation/diagnostic profiles preserved. Compute Graph success does not automatically count as communication acceptance; keep priority on evidence-led communication after closure. Preserve any rejection, no retry until pass.'})
    write_json(args.output / 'source-evidence-reuse.json', reuse)
    write_json(args.output / 'native-candidate-reuse.json', native)
    write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'independent'})
    results = {}
    try:
        target = args.output / 'ddp-independent'
        base = fork_run(source.output, target, revision_id=BASELINE)
        check = ProjectSearch(cfg, target)
        assert not base.bundles and check.prepared == source.prepared and check.request == source.request
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'accepted_revision': ACCEPTED,
            'baseline': base.id, 'contract_ref': base.contract_ref, 'controls_path': source.prepared['controls_path'],
            'request_sha256': sha(target / 'request.json'), 'prepared_sha256': sha(target / 'prepared.json'),
            'adapter_sha256': source.prepared['adapter_sha256'], 'publish': False,
            'scope': 'Fresh B/C/C/B leased workers; each rank two continuous full checks, two warmups, all five clean samples and three separate communication metric traces. Shared read-only current controls; no large model/reference copies.'})
        r = promote_version(check, source.output, revision_id=ACCEPTED, publish=False)
        assert check.revisions.current().id == BASELINE
        results['independent'] = {'status': 'completed', **{k: r[k] for k in ['trial', 'revision', 'parent', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']}}
    except Exception:
        results['independent'] = {'status': 'error', 'traceback': traceback.format_exc()}
    try:
        assert source.revisions.current().id == ACCEPTED
        assert all(sha(path) == h for path, h in frozen.items())
        results['preservation'] = {'status': 'completed', 'files': len(frozen), 'source_current': ACCEPTED}
    except Exception:
        results['preservation'] = {'status': 'error', 'traceback': traceback.format_exc()}
    terminal = {'phase': 'completed' if all(v['status'] == 'completed' for v in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    for name, value in [('followup-results.json', results), ('state.json', terminal), ('followup-completed.json', terminal)]:
        write_json(args.output / name, value)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output']: ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--preflight', action='store_true'); main(ap.parse_args())
