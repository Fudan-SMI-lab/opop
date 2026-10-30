"""One ordinary current-device communication-priority FSDP trial."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

BASELINE = '7231e858ad370240b7e86ac29c1745205ae8616501af53539423e1671003cca1'
CORE = 'ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f'
AUDIT = '23be91898d179965dfa74656518ec32616edba2b37bc22caad9052c5a1ce7ce6'
SOURCE_CHECK = '30db47da3d1cc770408b9a7ba45eedd4df8fdb62aeb54c3fd173348a27c2a07b'


def now():
    return datetime.now(timezone.utc).isoformat()


def preflight(args, config):
    assert sha(args.parent / 'terminal-audit.json') == AUDIT
    assert sha(args.parent / 'source-check.json') == SOURCE_CHECK
    audit = read(args.parent / 'terminal-audit.json')
    done = read(args.parent / 'followup-completed.json')
    assert done['phase'] == 'completed' and audit['all_frozen_unchanged']
    source = ProjectSearch(config, args.parent / 'fsdp-current')
    assert source.revisions.current().id == source.prepared['revision'] == BASELINE
    assert not source.revisions.current().bundles and not (source.output / 'discovery.json').exists()
    assert source.prepared['adapter_sha256'] == '8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6'
    assert resolve_devices(source.request.devices) == source.request.devices
    assert source.request.devices == ['GPU-20e8003c-f531-d50c-084c-a9a97fe94004', 'GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321']
    assert source.request.continuous_steps == 2 and source.request.warmup == 2 and source.request.repeats == 5
    assert len(source.request.diagnostic_metrics) == 4
    assert source.request.objective['aggregation_order'] == 'ranks_then_samples'
    assert source.request.objective['rank_aggregation'] == 'max'
    assert framework_source()['sha256'] == CORE
    assert audit['aggregates']['step_ms']['median'] == 175.753368
    assert audit['constraints'][0]['limit'] == 2351207577.6 and audit['constraints'][0]['passed']
    assert audit['source_check']['status'] == 'passed'
    for rank in audit['source_check']['ranks']:
        for check in rank['checks']:
            assert check['comparison']['passed'] and check['comparison']['max_abs_error'] == 0
            assert not check['full_logits_plus_one_rejected']['passed']
            assert not check['false_tied_observation_rejected']['passed']
    frozen = dict(read(args.parent / 'protocol.json')['frozen_identities'])
    for name in ['terminal-audit.json', 'public-terminal-audit.json', 'source-check.json', 'followup-completed.json', 'protocol.json']:
        path = args.parent / name
        frozen[str(path)] = sha(path)
    for path in source.output.rglob('*'):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            frozen[str(path)] = sha(path)
    assert all(sha(path) == value for path, value in frozen.items())
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', '/root/autodl-tmp'], text=True).splitlines()[-1])
    assert free >= 10 * 1024**3, ('8GiB warning floor plus 2GiB trial budget', free)
    return source, frozen, free


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    source, frozen, free = preflight(args, config)
    summary = {'core': framework_source(), 'source': str(source.output), 'parent': BASELINE,
        'devices': source.request.devices, 'frozen_files': len(frozen), 'rounds': 1,
        'baseline_step_ms': 175.753368, 'peak_limit_bytes': 2351207577.6,
        'source_check_sha256': SOURCE_CHECK, 'baseline_audit_sha256': AUDIT,
        'free_bytes': free, 'planned_growth_bytes': 2 * 1024**3, 'warning_free_bytes': 8 * 1024**3}
    if args.preflight:
        print(json.dumps({'status': 'passed', **summary}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {'at': now(), **summary, 'frozen_identities': frozen,
        'scope': 'One ordinary real trial on the original full FSDP training task with its existing communication diagnostics. Fresh current-device empty baseline and actual source-verified controls shared read-only. Fresh baseline discovery occurs through the existing CLI. No developer-selected group/K/implementation, historical optimized bundle, feedback fact table, source/model/control rerun, reference copy or production change. Default scopes and parameter exploration; API observe-only. Existing developer-assisted source-owner adapter remains an explicit boundary. No independent/native stage scheduled by this controller; the ordinary planner may choose collectors. Not unaided original-entry or a retry-until-pass batch.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    results = {}
    try:
        branch = args.output / 'fsdp-search'
        parent = fork_run(source.output, branch)
        search = ProjectSearch(config, branch)
        assert parent.id == BASELINE and not parent.bundles
        assert search.request == source.request and search.prepared == source.prepared
        assert not (branch / 'discovery.json').exists() and not (branch / 'latest-feedback.json').exists()
        assert search.prepared['controls_path'] == source.prepared['controls_path']
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'parent': BASELINE,
            'request_sha256': sha(branch / 'request.json'), 'prepared_sha256': sha(branch / 'prepared.json'),
            'controls_path': search.prepared['controls_path'], 'source_check_sha256': SOURCE_CHECK,
            'initial_bundles': [], 'fresh_baseline_discovery_required': True,
            'scope': 'Exact current-device controls/contract reused read-only; no old-device timing or K3 installed. No new source comparison claimed in this trial.'})
        argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config),
            'optimize-project', '--request', str(branch / 'request.json'), '--output', str(branch), '--resume', '--rounds', '1']
        write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': argv,
            'scope': 'Existing discovery, collector planning, selector, candidate, default parameter exploration and acceptance. One ordinary trial. No manually supplied selection/candidate/feedback or changed production prompt.'})
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'fresh-discovery-and-one-ordinary-trial'})
        with (args.output / 'search-controller.log').open('wb') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        current = ProjectSearch(config, branch).revisions.current()
        trials = [{'path': str(path), 'sha256': sha(path), 'decision': read(path).get('decision')}
            for path in sorted((branch / 'trials').glob('*/result.json'))]
        results['search'] = {'status': 'completed' if process.returncode == 0 else 'error',
            'returncode': process.returncode, 'current': current.id, 'bundles': list(current.bundles),
            'trials': trials, 'independent_acceptance': 'not_run_in_this_batch'}
        assert len(trials) <= 1
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(path) == value for path, value in frozen.items()) and source.revisions.current().id == BASELINE
    results['preservation'] = {'status': 'passed' if same else 'failed', 'files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if same and results.get('search', {}).get('status') == 'completed' and 'controller' not in results else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp')),
        'next': 'Audit all real role reads, parameters, quality/constraints/noise and native evidence. If genuinely new accepted version, declare independent whole-version closure; otherwise retain failure. DDP next, no automatic Qwen extension or retry until pass.'}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
