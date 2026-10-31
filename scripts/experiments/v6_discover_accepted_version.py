"""Refresh ordinary discovery on one independently accepted complete version."""
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
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    os.umask(0o077)
    cfg = load_config(args.config, [])
    cfg.usage.mode = 'observe_only'
    audit = read(args.parent/'terminal-audit.json')
    assert read(args.parent/'followup-completed.json')['phase'] == 'completed'
    trial = audit['trial']
    assert trial['decision']['accepted'] and trial['quality'] == 'pass'
    source = ProjectSearch(cfg, args.parent/'ddp-independent')
    current = source.revisions.current()
    accepted = source.revisions.revision(trial['revision'])
    paths = [args.parent/n for n in ['protocol.json', 'terminal-audit.json', 'followup-completed.json']]
    paths += [source.output/n for n in ['request.json', 'prepared.json', 'project/current.json']]
    frozen = {str(p): sha(p) for p in paths}
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(args.output.parent)], text=True).splitlines()[-1])
    assert free >= args.warning_free_bytes + args.growth_forecast_bytes
    info = {'source': str(source.output), 'source_current': current.id, 'source_revision': accepted.id,
            'core': framework_source(), 'candidate_sha256': audit['candidate_sha256'],
            'bundles': list(accepted.bundles), 'devices': source.request.devices,
            'critical_files': len(frozen), 'free_bytes': free,
            'growth_forecast_bytes': args.growth_forecast_bytes, 'warning_free_bytes': args.warning_free_bytes}
    if args.preflight:
        print(json.dumps({'status': 'passed', **info}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    scope = ('One existing ProjectSearch.discover on exact independently accepted exploration; '
             'ordinary real collection planner chooses existing collectors. Attach current-code registry/plan only. '
             'No selector/candidate/parameter/trial/independent rerun/source-control/reference copy/pointer publication. '
             'No developer audit answers or manual collector/algorithm supplied. Preserve all errors; no retry until pass.')
    write_json(args.output/'protocol.json', {'at': now(), **info, 'frozen_identities': frozen, 'scope': scope})
    write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': 'accepted_discovery'})
    results = {}
    try:
        target = args.output/'ddp-evidence'
        parent = fork_run(source.output, target, revision_id=accepted.id)
        search = ProjectSearch(cfg, target)
        assert search.request == source.request and search.prepared == source.prepared
        write_json(args.output/'preparation-reuse.json', {'source': str(source.output), 'revision': parent.id,
                   'request_sha256': sha(target/'request.json'), 'prepared_sha256': sha(target/'prepared.json'),
                   'controls_path': search.prepared['controls_path'], 'adapter_sha256': search.prepared['adapter_sha256'],
                   'scope': 'Exact accepted files/bundles/contract and shared read-only controls; no source/reference copies.'})
        discovery = search.discover()
        new = search.revisions.current()
        assert new.files == parent.files and new.bundles == parent.bundles and new.contract_ref == parent.contract_ref
        assert new.parent_id == parent.id and discovery['revision'] == new.id
        results['discovery'] = {'status': 'completed', 'revision': new.id, 'parent': parent.id,
               'discovery_sha256': sha(target/'discovery.json'), 'calls': len(discovery['calls']), 'groups': len(discovery['groups']),
               'collectors': discovery.get('collection_plan', {}).get('plan', {}).get('collectors', [])}
    except Exception:
        results['discovery'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = source.revisions.current().id == current.id and all(sha(p) == h for p, h in frozen.items())
    results['preservation'] = {'status': 'completed' if same else 'error', 'critical_files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if all(v['status'] == 'completed' for v in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(args.output.parent), 'scope': scope}
    for name, value in [('followup-results.json', results), ('state.json', terminal), ('followup-completed.json', terminal)]:
        write_json(args.output/name, value)
    print(json.dumps({'at': now(), 'phase': terminal['phase'], 'results': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--growth-forecast-bytes', type=int, required=True)
    parser.add_argument('--warning-free-bytes', type=int, default=8*1024**3)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
