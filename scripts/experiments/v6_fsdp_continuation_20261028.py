"""Continue a verified FSDP version with ordinary feedback and fresh discovery."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_fsdp_search_20261028 import native


def search_rounds(args, config):
    parent = ProjectSearch(config, args.parent)
    audit = read(args.parent.parent / 'terminal-audit.json')
    assert audit['independent']['decision']['accepted']
    assert parent.revisions.current().id == audit['current']['revision']
    feedback = args.parent / 'latest-feedback.json'
    assert read(feedback)['decision']['accepted']
    assert read(feedback)['decision']['revision'] == parent.revisions.current().id
    command = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
               '--request', str(args.parent / 'request.json'), '--branch-from', str(args.parent),
               '--feedback', str(feedback), '--output', str(args.output / 'fsdp-search'), '--rounds', '3']
    if args.discovery_from:
        # Preserve the failed attempt and reuse its completed GPU discovery.
        # Complete the exact previously unstarted tail of discover() in a new branch.
        failed = ProjectSearch(config, args.discovery_from)
        assert failed.revisions.current().bundles == parent.revisions.current().bundles
        assert failed.revisions.current().files == parent.revisions.current().files
        assert failed.revisions.current().contract_ref == parent.revisions.current().contract_ref
        discovery = read(args.discovery_from / 'discovery.json')
        assert discovery['revision'] == failed.revisions.current().id and 'collection_plan' not in discovery
        for path in discovery['raw_discovery']:
            raw = read(path)
            assert sha(raw['trace']) == raw['trace_sha256']
        fork_run(args.discovery_from, args.output / 'fsdp-search')
        resumed = ProjectSearch(config, args.output / 'fsdp-search')
        write_json(args.output / 'discovery-reuse.json', {'source': str(args.discovery_from),
            'source_discovery_sha256': sha(args.discovery_from / 'discovery.json'),
            'source_revision': failed.revisions.current().id, 'raw_discovery': discovery['raw_discovery'],
            'scope': 'Unmodified completed GPU discovery reused; real collector planner/collectors were not previously executed.'})
        discovery['collection_plan'] = resumed.plan_collectors(discovery)
        discovery['extra_collectors'] = resumed.collect_extra(resumed.revisions.current(), discovery['collection_plan']['plan']['collectors'])
        write_json(resumed.output / 'discovery.json', discovery)
        index = command.index('--branch-from'); command[index:index+2] = ['--resume']
    write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': command,
        'parent_revision': parent.revisions.current().id, 'parent_bundles': list(parent.revisions.current().bundles),
        'feedback_sha256': sha(feedback), 'prior_audit_sha256': sha(args.parent.parent / 'terminal-audit.json'),
        'scope': 'New branch from an independently accepted same-contract version. Existing unmodified framework feedback only; no developer diagnosis, manual groups or new candidate code. Fresh discovery required because prior discovery predates acceptance. Three new autonomous rounds; no rerun of old failed endpoints.'})
    with (args.output / 'search-controller.log').open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    current = ProjectSearch(config, args.output / 'fsdp-search').revisions.current()
    return {'status': 'completed' if result.returncode == 0 else 'error', 'returncode': result.returncode,
            'revision': current.id, 'bundles': list(current.bundles),
            'changed_bundles': list(current.bundles) != list(parent.revisions.current().bundles)}


def new_version(args, config):
    current = ProjectSearch(config, args.output / 'fsdp-search').revisions.current()
    previous = ProjectSearch(config, args.parent).revisions.current()
    return list(current.bundles) != list(previous.bundles)


def independent(args, config):
    if not new_version(args, config):
        return {'status': 'not_run', 'reason': 'No newly accepted bundle set; prior independent acceptance retained without repeat.'}
    target = args.output / 'fsdp-independent'
    fork_run(args.baseline, target)
    result = promote_version(ProjectSearch(config, target), args.output / 'fsdp-search', publish=False)
    return {'status': 'completed', **{k: result[k] for k in ['revision', 'parent', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']}}


def native_if_new(args, config):
    if not new_version(args, config):
        return {'status': 'not_run', 'reason': 'No newly accepted bundle set; no repeated native profiles.'}
    return native(args, config)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'parent', 'baseline']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--discovery-from', type=Path, help='Reuse completed GPU discovery from an attempt that failed before agent planning.')
    args = parser.parse_args(); os.umask(0o077); args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    assert read(args.parent / 'request.json') == read(args.baseline / 'request.json')
    assert read(args.parent / 'prepared.json')['controls_path'] == read(args.baseline / 'prepared.json')['controls_path']
    write_json(args.output / 'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'disk': disk_observation(Path('/root/autodl-tmp')), 'baseline': str(args.baseline), 'parent': str(args.parent),
        'discovery_from': str(args.discovery_from) if args.discovery_from else None,
        'sequence': ['accepted-version rediscovery and three autonomous rounds', 'conditional new full-version AB/BA', 'conditional separate B/C native paths'],
        'scope': 'Same corrected adapter, source, workload, full quality, original memory constraint and core; shared read-only controls. Historical results untouched. No API caps, new optimization hints, bulk local transfer, Goal or model polling.'})
    results = {}
    for name, operation in [('search', search_rounds), ('independent', independent), ('native', native_if_new)]:
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': name})
        try:
            results[name] = operation(args, config)
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output / 'followup-results.json', results)
        print(name, results[name]['status'], now(), flush=True)
    terminal = {'phase': 'completed' if all(r['status'] in ['completed', 'not_run'] for r in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'state.json', terminal); write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
