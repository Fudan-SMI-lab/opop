"""Run one original independent comparison of an accepted complete version."""
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
    protocol = read(args.parent/'protocol.json')
    audit = read(args.parent/'terminal-audit.json')
    assert read(args.parent/'followup-completed.json')['phase'] == 'completed'
    assert audit['decision']['accepted'] and audit['publication'] == 'exploration'
    source = ProjectSearch(cfg, args.parent/'ddp-validation')
    accepted = source.revisions.revision(audit['candidate'])
    current = source.revisions.current()
    baseline = source.revisions.revision(source.prepared['revision'])
    assert current.id == protocol['parent'] and accepted.contract_ref == baseline.contract_ref
    bundles = [source.revisions.bundle(b) for b in accepted.bundles]
    assert accepted.id == protocol['candidate']
    native_record = read(args.parent/'public-terminal-audit.json')
    native = native_record.get('native', [])
    for row in native:
        assert sha(row['path']) == row['sha256']
    # These are references to already reviewed evidence, not new source/model runs.
    source_reuse = {'at': now(), 'record': str(args.source_evidence),
                    'sha256': sha(args.source_evidence),
                    'scope': 'Prior source/controls evidence record reuse only. No fresh tensor/source comparison, model execution or reference copy.'}
    paths = [args.parent/n for n in ['protocol.json', 'terminal-audit.json', 'public-terminal-audit.json', 'followup-completed.json']]
    paths += [source.output/n for n in ['request.json', 'prepared.json', 'project/current.json']]
    paths.append(args.source_evidence)
    frozen = {str(p): sha(p) for p in paths}
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(args.output.parent)], text=True).splitlines()[-1])
    assert free >= args.warning_free_bytes + args.growth_forecast_bytes
    info = {'core': framework_source(), 'source': str(source.output), 'source_current': current.id,
            'accepted_revision': accepted.id, 'baseline': baseline.id, 'bundles': list(accepted.bundles),
            'bundle_parameters': {b.id: b.parameters for b in bundles},
            'candidate_sha256': protocol['candidate_sha256'], 'proposal_sha256': protocol['proposal_sha256'],
            'devices': source.request.devices, 'shared_controls': source.prepared['controls_path'],
            'critical_files': len(frozen), 'free_bytes': free,
            'growth_forecast_bytes': args.growth_forecast_bytes, 'warning_free_bytes': args.warning_free_bytes,
            'formal_audit_sha256': sha(args.parent/'terminal-audit.json')}
    if args.preflight:
        print(json.dumps({'status': 'passed', **info}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    scope = ('One existing promote_version(publish=False) of the exact formally accepted complete version, '
             'against the original prepared baseline. Fresh original B/C/C/B, quality, samples, constraints and noise unchanged. '
             'No Agent/search/parameter/collector/new native/source-control/reference-copy/pointer publication. '
             'Reuse existing native/source evidence; retain every failure/tail, no retry until pass.')
    write_json(args.output/'protocol.json', {'at': now(), **info, 'frozen_identities': frozen, 'scope': scope})
    write_json(args.output/'source-evidence-reuse.json', source_reuse)
    write_json(args.output/'native-candidate-reuse.json', {'at': now(), 'ranks': native,
               'formal_public_audit_sha256': sha(args.parent/'public-terminal-audit.json'),
               'normalized_sha256': native_record.get('normalized_sha256'),
               'scope': 'Exact existing formal native evidence only; no additional native capture, no wire/critical-path attribution.'})
    write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': 'independent'})
    results = {}
    try:
        target = args.output/'ddp-independent'
        base = fork_run(source.output, target, revision_id=baseline.id)
        search = ProjectSearch(cfg, target)
        assert base == baseline and search.prepared == source.prepared and search.request == source.request
        write_json(args.output/'preparation-reuse.json', {'source': str(source.output), 'baseline': base.id,
                   'accepted_revision': accepted.id, 'shared_controls': search.prepared['controls_path'],
                   'request_sha256': sha(target/'request.json'), 'prepared_sha256': sha(target/'prepared.json'),
                   'scope': 'Shared read-only original controls, no initial discovery or source/reference copy.'})
        result = promote_version(search, source.output, revision_id=accepted.id, publish=False)
        assert search.revisions.current().id == baseline.id
        results['independent'] = {'status': 'completed', **{k: result[k] for k in [
            'trial', 'revision', 'parent', 'source_revision', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']}}
    except Exception:
        results['independent'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = source.revisions.current().id == current.id and all(sha(p) == h for p, h in frozen.items())
    results['preservation'] = {'status': 'completed' if same else 'error', 'critical_files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if all(v['status'] == 'completed' for v in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(args.output.parent), 'scope': scope}
    for name, value in [('followup-results.json', results), ('state.json', terminal), ('followup-completed.json', terminal)]:
        write_json(args.output/name, value)
    print(json.dumps({'at': now(), 'phase': terminal['phase'], 'results': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output', 'source-evidence']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--growth-forecast-bytes', type=int, required=True)
    parser.add_argument('--warning-free-bytes', type=int, default=8*1024**3)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
