"""One ordinary trial with original rejection feedback and an unchanged probe-artifact index."""
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


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(args, config):
    observation = read(args.observation / 'protocol.json')
    completed = read(args.observation / 'followup-completed.json')
    assert completed['phase'] == 'completed'
    assert completed['results']['preservation']['unchanged']
    assert framework_source() == observation['core']
    assert sha(args.observation / 'terminal-audit.json') == 'c45883919600b7bdfddb6683a670ff1835a502f1f4cd04f463a3fd4a866b509f'
    source = ProjectSearch(config, args.parent / 'qwen-continuation')
    current = source.revisions.current()
    assert current.id == '2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1'
    assert not current.bundles and read(source.output / 'discovery.json')['revision'] == current.id
    assert resolve_devices(source.request.devices) == source.request.devices
    raw = (source.output / 'latest-feedback.json').read_bytes()
    assert digest(raw) == '2cf4eb2a2476f398bcbe0cd045b0790374d51852e31bea86d64416480e87b25c'
    original = json.loads(raw)
    assert original['trial'] == 'trial-0db61e9938' and not original['decision']['accepted']
    assert 'supplemental_evidence' not in original
    paths = [('probe', Path(observation['probe'])), ('plan', Path(observation['plan']))]
    for arm in ['baseline', 'candidate']:
        run = args.observation / (arm + '-observed')
        paths.extend([(arm + '_original_request', Path(observation['source_requests'][arm])),
                      (arm + '_observed_request', run / 'request.json'),
                      (arm + '_worker_result', run / 'rank-0.json'),
                      (arm + '_observation', run / 'diagnostic-evidence-rank-0.json')])
    index = {'schema_version': 'setup-observation-artifacts.v1',
        'scope': 'One completed fixed baseline/candidate observation of the historical implementation. These are existing artifacts, not a proposed implementation or causal conclusion.',
        'limits': 'The probe records bounded installation comparisons and partial parent-call metadata. Full comparison operands are not saved. CPU summaries and observed timing are distinct from original evaluator decisions and clean acceptance. The probe may perturb synchronization, allocations and lifetimes. Read its code and original outcomes to assess the scope of any inference.',
        'independent_reference': None,
        'artifacts': [{'kind': label, 'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for label, path in paths]}
    index_raw = (json.dumps(index, indent=2) + '\n').encode()
    supplemental = {'diagnostic_artifacts': {
        'scope': 'Existing bounded setup observation artifacts for the previous candidate; inspect original outcomes and probe scope. Observed timing is not acceptance, and no implementation or unique cause is supplied.',
        'index': {'path': str(args.output / 'evidence-index.json'), 'sha256': digest(index_raw), 'bytes': len(index_raw)}}}
    end = raw.rfind(b'}')
    augmented = raw[:end] + b',\n  "supplemental_evidence": ' + json.dumps(supplemental, indent=2).encode() + b'\n' + raw[end:]
    parsed = json.loads(augmented)
    assert {k: v for k, v in parsed.items() if k != 'supplemental_evidence'} == original
    old_files, old_compact = feedback_artifacts(raw.decode())
    new_files, new_compact = feedback_artifacts(augmented.decode())
    assert old_files == new_files
    registry = source.revisions.load(current.registry_ref, GroupRegistry)
    plan = source.revisions.load(current.plan_ref, ExecutionPlan)
    routing = []
    for label, kwargs in [('selector', {}), ('historical_selected_candidate', {'selected_groups': original['selection']['groups'], 'plan': plan})]:
        before = json.loads(route(old_compact, current_revision=current.id, registry=registry, **kwargs))
        after = json.loads(route(new_compact, current_revision=current.id, registry=registry, **kwargs))
        assert after.pop('supplemental_evidence') == supplemental
        assert before.pop('supplemental_evidence') is None and before == after
        routing.append({'role': label, 'only_supplemental_evidence_differs': True})
    historical = candidate_artifacts(augmented.decode(), source.store)
    code = historical['previous-candidate/candidate.py']
    assert digest(code if isinstance(code, bytes) else code.encode()) == '04dfc6c2c9165560ae7412bbdbd00be6e79044e7f52276d096dd48728c6583f8'
    frozen = observation['frozen_identities'].copy()
    extra = [path for _, path in paths] + [args.observation / name for name in ['protocol.json', 'terminal-audit.json', 'followup-completed.json']]
    extra += [source.output / name for name in ['latest-feedback.json', 'prepared.json', 'request.json', 'discovery.json', 'project/current.json', 'trials/trial-0db61e9938/result.json']]
    for path in extra:
        frozen[str(path)] = sha(path)
    assert all(sha(path) == value for path, value in frozen.items())
    transport = {'original_bytes': len(raw), 'original_sha256': digest(raw), 'new_bytes': len(augmented), 'new_sha256': digest(augmented),
        'bytes_added': len(augmented) - len(raw), 'original_prefix_before_closing_brace_preserved': True,
        'all_original_fields_equal': True, 'packaged_original_artifacts_equal': True, 'route_checks': routing,
        'historical_candidate_sha256': digest(code if isinstance(code, bytes) else code.encode()),
        'scope': 'CPU interface check only. Actual new selector/candidate snapshots, reads and decisions remain to be audited; no whole-role single-variable A/B claim.'}
    return source, index_raw, augmented, frozen, transport


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    source, index_raw, feedback_raw, frozen, transport = prepare(args, config)
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'frozen_files': len(frozen),
            'parent': source.revisions.current().id, 'devices': source.request.devices, 'transport': transport,
            'index_bytes': len(index_raw), 'rounds': 1, 'discovery': 'reuse completed current-device empty-parent discovery'}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    (args.output / 'evidence-index.json').write_bytes(index_raw)
    (args.output / 'diagnostic-feedback.json').write_bytes(feedback_raw)
    write_json(args.output / 'transport-preflight.json', transport)
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(), 'parent': str(args.parent),
        'observation': str(args.observation), 'source': str(source.output), 'parent_revision': source.revisions.current().id,
        'baseline': source.prepared['revision'], 'devices': source.request.devices, 'rounds': 1, 'frozen_identities': frozen,
        'index_sha256': digest(index_raw), 'feedback_sha256': digest(feedback_raw),
        'current_device_ttft_anchor': 23.285755, 'constraint_limit': 24.45004275,
        'scope': 'One newly declared ordinary trial with original failure feedback plus a metadata index to unchanged Agent-authored observation artifacts. Reuse completed current-device discovery/controls, no source/reference rerun/copy. No developer factual table/causal answer/manual implementation or prompt change. Not unaided original-entry or same-snapshot A/B. No independent/native stage scheduled; planner may request existing collectors.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    results = {}
    try:
        branch = args.output / 'qwen-continuation'
        parent = fork_run(source.output, branch)
        search = ProjectSearch(config, branch)
        assert not parent.bundles and parent.id == source.revisions.current().id
        assert search.prepared == source.prepared and search.request == source.request
        assert sha(branch / 'discovery.json') == sha(source.output / 'discovery.json')
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'parent': parent.id,
            'baseline': search.prepared['revision'], 'controls_path': search.prepared['controls_path'], 'initial_bundles': [],
            'request_sha256': sha(branch / 'request.json'), 'discovery_sha256': sha(branch / 'discovery.json'),
            'discovery': 'Completed current-device original empty-parent discovery reused, no initial GPU rediscovery or new controls.'})
        argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
            '--request', str(branch / 'request.json'), '--output', str(branch), '--resume', '--rounds', '1',
            '--feedback', str(args.output / 'diagnostic-feedback.json')]
        write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': argv, 'feedback_sha256': digest(feedback_raw),
            'scope': 'Existing default selector/candidate/scopes/parameter exploration and ordinary feedback routing. API observe-only, no new cap. Preserve every failure, no retry-until-pass or automatically installed probe.'})
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'one-ordinary-feedback-trial'})
        with (args.output / 'search-controller.log').open('wb') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        current = ProjectSearch(config, branch).revisions.current()
        results['search'] = {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode,
            'current': current.id, 'bundles': list(current.bundles),
            'trials': [{'path': str(p), 'sha256': sha(p), 'decision': read(p).get('decision')} for p in sorted((branch / 'trials').glob('*/result.json'))],
            'independent_acceptance': 'not_run_in_this_batch'}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(path) == value for path, value in frozen.items())
    results['preservation'] = {'status': 'passed' if same else 'failed', 'files': len(frozen), 'unchanged': same}
    terminal = {'phase': 'completed' if same and results.get('search', {}).get('status') == 'completed' else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'observation', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
