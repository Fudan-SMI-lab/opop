"""Fresh original-entry constrained full-generation search; no evaluator reuse."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

GOAL = (
    "Minimize end_to_end_ms for the source's complete interactive Qwen3-4B generation "
    "(batch 1, prompt 256 tokens, exactly 32 new greedy tokens), while keeping ttft_ms <= 1.05 "
    "times this task's freshly measured ORIGINAL baseline TTFT. Preserve the source model, "
    "inputs, full workload, BF16/SDPA/noTF32 policies, full logits at the source 0.02/0.02 "
    "tolerance and exact tokens, both completion synchronization boundaries, and state lifetimes. "
    "The primary metric covers the complete generation; the first-token metric is a constraint. "
    "The source's listed goals are provenance, not a requirement to optimize only one phase."
)


def now():
    return datetime.now(timezone.utc).isoformat()


def cli(args, stage, options):
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config),
            'optimize-project', *options]
    write_json(args.output / (stage + '-protocol.json'), {'at': now(), 'argv': argv})
    with (args.output / (stage + '-controller.log')).open('wb') as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode


def prepare(args, config):
    project = Path(__file__).resolve().parents[2] / 'examples/v6_validation'
    command = f'python qwen_inference.py --assets {shlex.quote(str(args.assets))} --output /tmp/original-qwen'
    run = args.output / 'qwen-original'
    code = cli(args, 'prepare', ['--project', str(project), '--command', command, '--goal', GOAL,
                               '--devices', '0', '--output', str(run), '--prepare-only'])
    if code:
        return {'status': 'error', 'returncode': code}
    prepared = read(run / 'prepared.json')
    if prepared['status'] != 'prepared':
        return {'status': 'preparation_failed', 'prepared_status': prepared['status']}
    search = ProjectSearch(config, run)
    current = search.revisions.current()
    assert not current.bundles
    return {'status': 'completed', 'revision': current.id,
            'adapter_sha256': prepared['adapter_sha256'],
            'request_sha256': sha(run / 'request.json'),
            'scope': 'New real intake, evaluator and controls; no supplied request, adapter, groups or candidate.'}


def source_check(args, config):
    search = ProjectSearch(config, args.output / 'qwen-original')
    source = Path(search.prepared['controls_path']).parent / 'reference-source'
    result = LocalExecutor().run(ExecutionRequest(
        entry=Path(__file__).with_name('v6_autonomous_cohort_20261027.py'), function='source_reference',
        cwd=source, output=args.output / 'source-check-run', devices=['0'],
        lease_dir=Path(search.request.lease_dir), worker_timeout_s=search.request.worker_timeout_s,
        wait_for_devices=True, parameters={'case': 'qwen', 'reference': str(args.reference),
            'controls': search.prepared['controls_path'], 'request': search.request.portable()}))
    write_json(args.output / 'source-check.json', result)
    # Unknown names/containers and worker failures are retained, not numerical verdicts.
    statuses = [r.get('result', {}).get('status') for r in result.get('ranks', [])]
    return {'status': 'completed' if valid_run(result) else 'source_check_requires_audit',
            'worker_status': result.get('status'), 'observation_statuses': statuses,
            'path': str(args.output / 'source-check.json'), 'source_reference_sha256': sha(args.reference),
            'controls_reference_sha256': sha(Path(search.prepared['controls_path']) / 'reference-rank-0.pt'),
            'scope': 'Independent full source logits/exact tokens; additional observations and source ownership require code review. No result passed to candidate agents.'}


def search(args, config):
    run = args.output / 'qwen-original'
    code = cli(args, 'search', ['--request', str(run / 'request.json'), '--output', str(run),
                              '--resume', '--rounds', '3'])
    current = ProjectSearch(config, run).revisions.current()
    return {'status': 'completed' if code == 0 else 'error', 'returncode': code,
            'current': current.id, 'bundles': list(current.bundles)}


def independent(args, config):
    run = args.output / 'qwen-original'
    search = ProjectSearch(config, run)
    if not search.revisions.current().bundles:
        return {'status': 'not_run', 'reason': 'No autonomous accepted active bundle.'}
    target = args.output / 'qwen-independent'
    fork_run(run, target, revision_id=search.prepared['revision'])
    result = promote_version(ProjectSearch(config, target), run, publish=False)
    return {'status': 'completed', **{k: result[k] for k in
        ['revision', 'parent', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']}}


def native(args, config):
    search = ProjectSearch(config, args.output / 'qwen-original')
    rows = []
    for name, revision in [('baseline', search.revisions.revision(search.prepared['revision'])),
                           ('candidate', search.revisions.current())]:
        result = Nsys().collect(search, revision)
        path = args.output / ('native-' + name + '.json')
        write_json(path, result)
        rows.append({'arm': name, 'status': result['status'], 'path': str(path), 'sha256': sha(path)})
    return {'status': 'completed', 'runs': rows, 'scope': 'Separate perturbed path evidence, never clean acceptance scores.'}


def main(args):
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    assert args.reference.is_file() and args.assets.is_dir()
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    project = Path(__file__).resolve().parents[2] / 'examples/v6_validation'
    write_json(args.output / 'protocol.json', {
        'at': now(), 'core': framework_source(), 'goal': GOAL,
        'source_files': {p.name: sha(p) for p in sorted(project.iterdir()) if p.is_file()},
        'independent_source_reference': {'path': str(args.reference), 'sha256': sha(args.reference)},
        'sequence': ['fresh original-entry prepare', 'external full source gate', 'three autonomous rounds',
                     'conditional independent full-version AB/BA', 'conditional separate native paths'],
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'No adapter/request/groups/candidates/old bundles/developer diagnoses supplied. Unknown source schema stops dependent work for review; no automatic mapping or retries. API observe-only. No production changes or bulk transfers.'})
    results = {}
    for name, operation in [('prepare', prepare), ('source', source_check), ('search', search),
                            ('independent', independent), ('native', native)]:
        write_json(args.output / 'state.json', {'phase': 'running', 'stage': name, 'at': now()})
        if any(results.get(k, {}).get('status') not in [None, 'completed'] for k in ['prepare', 'source', 'search']):
            results[name] = {'status': 'not_run', 'reason': 'Required prior stage failed; retained for audit, no retry.'}
        elif name == 'native' and results.get('independent', {}).get('quality') != 'pass':
            results[name] = {'status': 'not_run', 'reason': 'No complete independent quality-passing active version.'}
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
    for name in ['config', 'output', 'assets', 'reference']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
