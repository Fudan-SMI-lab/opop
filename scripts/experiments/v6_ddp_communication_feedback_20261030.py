"""One ordinary DDP continuation using exact original framework failure feedback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts, route
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

CORE = 'ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f'
PARENT = 'e925da5fa07cb2c7ac12e0c19e1c297b62ee28feaf39fdf4494a00cfe4f31d29'
BASELINE = '7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
AUDIT = '70b28f9d3d09ed8078461a325a40e0ce021a5c00e3b13d12e4bb5fff467746bf'
FEEDBACK = '3e2ddf153aeb14cb79861d0abf413d75147164e5fc754f039e35cd637c38f5ab'
CANDIDATE = 'cf5e5fa7572e159a2a03828ee83bbbded6f730dd68d5a8488d46327c0af15d05'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def prepare(args, config):
    old = read(args.parent / 'protocol.json')
    assert sha(args.parent / 'terminal-audit.json') == AUDIT
    assert read(args.parent / 'followup-completed.json')['phase'] == 'completed'
    assert framework_source()['sha256'] == CORE and framework_source() == old['core']
    source = ProjectSearch(config, args.parent / 'ddp-search')
    current = source.revisions.current()
    assert current.id == PARENT and not current.bundles and source.prepared['revision'] == BASELINE
    assert source.prepared['adapter_sha256'] == 'b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193'
    assert resolve_devices(source.request.devices) == source.request.devices == old['devices']
    assert (source.request.continuous_steps, source.request.warmup, source.request.repeats) == (2, 2, 5)
    assert len(source.request.diagnostic_metrics) == 1
    raw = (source.output / 'latest-feedback.json').read_bytes()
    assert digest(raw) == FEEDBACK and raw == (source.output / 'trials/trial-df2eb7acd2/result.json').read_bytes()
    feedback = json.loads(raw)
    assert feedback['trial'] == 'trial-df2eb7acd2'
    assert feedback['decision'] == {'accepted': False, 'reasons': ['no_valid_parameter_candidate']}
    assert 'supplemental_evidence' not in feedback
    files, compact = feedback_artifacts(raw.decode())
    historical = candidate_artifacts(raw.decode(), source.store)
    assert digest(historical['previous-candidate/candidate.py']) == CANDIDATE
    registry = source.revisions.load(current.registry_ref, GroupRegistry)
    plan = source.revisions.load(current.plan_ref, ExecutionPlan)
    routes = []
    for label, kw in [('selector', {}), ('historical_selected_candidate', {'selected_groups': feedback['selection']['groups'], 'plan': plan})]:
        routed = route(compact, current_revision=current.id, registry=registry, **kw)
        parsed = json.loads(routed)
        assert parsed['supplemental_evidence'] is None
        routes.append({'role': label, 'bytes': len(routed.encode()), 'sha256': digest(routed),
                       'supplemental_evidence': None, 'scope': 'Existing registry/plan route, not actual later role consumption.'})
    transport = {'feedback_bytes': len(raw), 'feedback_sha256': digest(raw), 'bytes_added': 0,
                 'original_framework_feedback_unchanged': True, 'candidate_sha256': CANDIDATE,
                 'packaged_files': {n: {'bytes': len(v.encode() if isinstance(v, str) else v), 'sha256': digest(v)} for n, v in files.items()},
                 'routes': routes,
                 'scope': 'No developer terminal audit, control-loss read, causal table, repair implementation or new supplemental facts supplied. Original failure and sidecars only; historical candidate is not installed.'}
    frozen = dict(old['frozen_identities'])
    paths = [args.parent / n for n in ['protocol.json', 'followup-completed.json', 'terminal-audit.json', 'public-terminal-audit.json']]
    paths += [source.output / n for n in ['request.json', 'prepared.json', 'discovery.json', 'latest-feedback.json', 'events.jsonl', 'project/current.json', 'trials/trial-df2eb7acd2/result.json']]
    paths += list((source.output / 'generation-snapshots').glob('*.json'))
    paths += list((source.output / 'sandboxes').glob('*/candidate.py')) + list((source.output / 'sandboxes').glob('*/proposal.json'))
    for job in (source.output / 'jobs').iterdir():
        paths += [p for p in (job / 'run').iterdir() if p.is_file() and not p.is_symlink()]
        paths += [p for p in (job / 'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in paths:
        frozen[str(p)] = sha(p)
    assert all(sha(p) == h for p, h in frozen.items())
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', '/root/autodl-tmp'], text=True).splitlines()[-1])
    assert free >= 10 * 1024**3, free
    return source, raw, frozen, transport, free


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    source, raw, frozen, transport, free = prepare(args, config)
    info = {'core': framework_source(), 'source': str(source.output), 'parent': PARENT, 'baseline': BASELINE,
            'devices': source.request.devices, 'frozen_files': len(frozen), 'rounds': 1,
            'baseline_step_ms': 133.379621, 'step_limit_ms': 140.04860205,
            'free_bytes': free, 'planned_growth_bytes': 2 * 1024**3, 'warning_free_bytes': 8 * 1024**3,
            'transport': transport}
    if args.preflight:
        print(json.dumps({'status': 'passed', **info})); return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    (args.output / 'original-feedback.json').write_bytes(raw)
    write_json(args.output / 'transport-preflight.json', transport)
    write_json(args.output / 'protocol.json', {'at': now(), **info, 'frozen_identities': frozen,
        'scope': 'One newly declared ordinary current-device DDP feedback trial after user requested continued progress. Exact original framework failure feedback, existing default selector/candidate/scopes/parameter exploration and acceptance. Existing current-device empty-parent discovery/controls shared read-only; no fresh baseline or initial GPU rediscovery. No manual group/bucket/algorithm/repair, developer audit conclusions, optimized historical bundle, production prompt/sampler/acceptance changes, or manually scheduled fixed reruns of the old candidate. Actual generated bytes still require audit. Not unaided original-entry or same-snapshot A/B. No independent/native stage scheduled; planner may select existing collectors. API observe-only; Qwen closed and prior FSDP rejection unchanged.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    results = {}
    try:
        branch = args.output / 'ddp-search'
        parent = fork_run(source.output, branch)
        search = ProjectSearch(config, branch)
        assert parent.id == PARENT and not parent.bundles
        assert search.request == source.request and search.prepared == source.prepared
        assert sha(branch / 'discovery.json') == sha(source.output / 'discovery.json')
        assert not (branch / 'latest-feedback.json').exists()
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'parent': PARENT,
            'baseline': BASELINE, 'initial_bundles': [], 'request_sha256': sha(branch / 'request.json'),
            'prepared_sha256': sha(branch / 'prepared.json'), 'discovery_sha256': sha(branch / 'discovery.json'),
            'controls_path': search.prepared['controls_path'], 'feedback_sha256': digest(raw),
            'scope': 'Reuse completed current-device empty-baseline discovery and exact original controls/contract. No source/control rerun or reference copies; original source mapping assistance persists.'})
        argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config),
                'optimize-project', '--request', str(branch / 'request.json'), '--output', str(branch),
                '--resume', '--rounds', '1', '--feedback', str(args.output / 'original-feedback.json')]
        write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': argv, 'feedback_sha256': digest(raw),
            'scope': 'Ordinary original-feedback continuation; actual new selection/candidate inputs and full tool use require audit. Only one trial, no repeat until pass.'})
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'one-ordinary-feedback-trial'})
        with (args.output / 'search-controller.log').open('wb') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        current = ProjectSearch(config, branch).revisions.current()
        trials = [{'path': str(p), 'sha256': sha(p), 'decision': read(p).get('decision')} for p in sorted((branch / 'trials').glob('*/result.json'))]
        assert len(trials) <= 1
        results['search'] = {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode,
                             'current': current.id, 'bundles': list(current.bundles), 'trials': trials,
                             'independent_acceptance': 'not_run_in_this_batch'}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(p) == h for p, h in frozen.items()) and source.revisions.current().id == PARENT
    results['preservation'] = {'status': 'passed' if same else 'failed', 'files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if same and results.get('search', {}).get('status') == 'completed' and 'controller' not in results else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp')),
                'next': 'Audit all actual snapshots/tools/parameters/quality/constraints/noise/native. Only genuinely new acceptance justifies declared independent whole-version B/C/C/B. Preserve rejection; do not relaunch old batches or retry until pass. Continue evidence-led communication research with each next bounded step explicit.'}
    for name, value in [('followup-results.json', results), ('state.json', terminal), ('followup-completed.json', terminal)]:
        write_json(args.output / name, value)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for n in ['config', 'parent', 'output']: p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    main(p.parse_args())
