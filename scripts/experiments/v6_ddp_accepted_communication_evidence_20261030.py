"""One accepted-version rediscovery and ordinary collection plan; no candidate trial."""
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
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_feedback_independent_20261030 import CODE, ACCEPTED, BASELINE


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(args, cfg):
    prior = read(args.parent / 'protocol.json'); audit = read(args.parent / 'terminal-audit.json')
    assert sha(args.parent / 'terminal-audit.json') == '9d85463cf952cf3dce386d9b39e604be293fb141a861cc0a2f94cd51d6958c38'
    assert read(args.parent / 'followup-completed.json')['phase'] == 'completed'
    assert audit['trial']['decision']['accepted'] and audit['trial']['quality'] == 'pass'
    assert audit['trial']['publication'] == 'exploration' and audit['candidate_sha256'] == CODE
    assert prior['core'] == audit['core'] == framework_source()
    source = ProjectSearch(cfg, Path(prior['source']))
    independent = ProjectSearch(cfg, args.parent / 'ddp-independent')
    assert source.revisions.current().id == ACCEPTED and independent.revisions.current().id == BASELINE
    assert source.request == independent.request and source.prepared == independent.prepared
    assert resolve_devices(source.request.devices) == prior['devices']
    # Existing discovery is genuinely stale for the accepted version; the prior
    # post-evaluation profile did not attach its registry/plan to this current.
    assert read(source.output / 'discovery.json')['revision'] != ACCEPTED
    current = source.revisions.current(); assert len(current.bundles) == 1
    bundle = source.revisions.bundle(current.bundles[0])
    assert bundle.parameters == {} and {f.path: f.sha256 for f in current.files}[bundle.installer] == CODE
    frozen = dict(prior['frozen_identities'])
    for name in ['protocol.json', 'terminal-audit.json', 'public-terminal-audit.json', 'followup-completed.json', 'source-evidence-reuse.json', 'native-candidate-reuse.json']:
        frozen[str(args.parent / name)] = sha(args.parent / name)
    for name in ['request.json', 'prepared.json', 'project/current.json']:
        frozen[str(independent.output / name)] = sha(independent.output / name)
    for path in (independent.output / 'jobs').glob('*/run/**/*'):
        if path.is_file() and not path.is_symlink(): frozen[str(path)] = sha(path)
    assert all(sha(path) == h for path, h in frozen.items())
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', '/root/autodl-tmp'], text=True).splitlines()[-1])
    assert free > 9 * 1024**3
    return source, frozen, free


def main(args):
    os.umask(0o077)
    cfg = load_config(args.config, []); cfg.usage.mode = 'observe_only'
    source, frozen, free = prepare(args, cfg)
    info = {'core': framework_source(), 'source': str(source.output), 'source_revision': ACCEPTED,
        'independent_root': str(args.parent), 'baseline': BASELINE, 'candidate_sha256': CODE,
        'bundles': list(source.revisions.current().bundles), 'devices': source.request.devices,
        'frozen_files': len(frozen), 'free_bytes': free, 'additional_budget_bytes': 1024**3,
        'warning_floor_bytes': 8 * 1024**3, 'no_candidate_search': True,
        'budget_basis': 'Prior exact-version profile 45,445,120B; all six diagnostic jobs together 39,698,432B by actual du. Reserve 1GiB for fresh discovery, existing planner/session, and its selected collectors; no controls/reference/model copies.'}
    if args.preflight: print(json.dumps({'status': 'passed', **info})); return
    args.output.mkdir(parents=True, exist_ok=True); assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {'at': now(), **info, 'frozen_identities': frozen,
        'sequence': ['fork exact accepted version', 'one existing discover including ordinary collection planner and its chosen collectors'],
        'scope': 'Bounded evidence acquisition after independently accepted compute scheduling. Refresh stale empty-baseline registry on actual accepted code before deciding any next communication candidate. No selector/candidate/trial, hand group/collector/algorithm selection, new independent validation, controls/source rerun, production change or automatic next candidate. Existing planner instructions/model/tools unchanged; no developer causal/factual answer supplied. Profiles/run-only diagnostics are not clean scores or new quality/acceptance.'})
    write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'accepted-discovery'})
    results = {}
    try:
        target = args.output / 'ddp-evidence'; parent = fork_run(source.output, target, revision_id=ACCEPTED)
        search = ProjectSearch(cfg, target)
        assert search.request == source.request and search.prepared == source.prepared
        assert not (target / 'discovery.json').exists()
        write_json(args.output / 'preparation-reuse.json', {'parent': parent.id, 'source': str(source.output),
            'request_sha256': sha(target / 'request.json'), 'prepared_sha256': sha(target / 'prepared.json'),
            'controls_path': search.prepared['controls_path'], 'adapter_sha256': search.prepared['adapter_sha256'],
            'scope': 'Exact accepted source/bundle/contract and current-device controls shared; old baseline discovery not copied. No feedback/audit facts injected into collection-planner inputs.'})
        d = search.discover(); new = search.revisions.current()
        assert new.files == parent.files and new.bundles == parent.bundles and new.contract_ref == parent.contract_ref
        assert new.parent_id == parent.id and d['revision'] == new.id
        assert not list((target / 'trials').glob('*/result.json'))
        results['discovery'] = {'status': 'completed', 'revision': new.id, 'parent': parent.id,
            'discovery_sha256': sha(target / 'discovery.json'), 'calls': len(d['calls']), 'groups': len(d['groups']),
            'collectors': d.get('collection_plan', {}).get('plan', {}).get('collectors', []),
            'scope': 'Discovery attached only in this isolated branch; exact code/bundle/contract retained, not optimization acceptance.'}
    except Exception:
        results['discovery'] = {'status': 'error', 'traceback': traceback.format_exc()}
    try:
        assert source.revisions.current().id == ACCEPTED
        assert all(sha(path) == h for path, h in frozen.items())
        results['preservation'] = {'status': 'completed', 'files': len(frozen), 'source_current': ACCEPTED}
    except Exception:
        results['preservation'] = {'status': 'error', 'traceback': traceback.format_exc()}
    terminal = {'phase': 'completed' if all(v['status'] == 'completed' for v in results.values()) else 'completed_with_errors',
        'finished_at': now(), 'results': results,
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True)}
    for name, value in [('followup-results.json', results), ('state.json', terminal), ('followup-completed.json', terminal)]:
        write_json(args.output / name, value)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output']: ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--preflight', action='store_true'); main(ap.parse_args())
