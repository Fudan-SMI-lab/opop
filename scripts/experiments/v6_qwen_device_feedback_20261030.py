"""One ordinary new-device trial after audited fixed fallback and setup evidence."""
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
from kernel_optimizer.project.feedback import candidate_artifacts, route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    protocol = read(args.parent / 'protocol.json'); audit = read(args.parent / 'terminal-audit.json')
    source = ProjectSearch(config, args.parent / 'qwen-fixed')
    assert framework_source() == protocol['core'] == audit['core']
    assert resolve_devices(source.request.devices) == source.request.devices
    assert source.revisions.current().id == source.prepared['revision'] and not source.revisions.current().bundles
    completed = read(args.parent / 'followup-completed.json')
    assert completed['results']['B']['valid_run'] and completed['results']['C']['valid_run']
    diagnostic = read(args.diagnostic / 'diagnostic-result.json')
    assert diagnostic['status'] == 'passed'
    evidence = args.diagnostic / 'qwen-diagnostic/jobs/setup-diagnostic/run/candidate-diagnostics-rank-0.json'
    events = read(evidence)
    assert any(e['event'] == 'exception' and e['phase'] == 'setup' for e in events['events'])
    candidate = source.revisions.revision(read(args.parent / 'staged-identities.json')['candidate']['id'])
    bundle = source.revisions.bundle(candidate.bundles[0])
    original = Path(protocol['source'])
    snapshot = original / 'generation-snapshots/generation-cb08f1ca8ed2.json'
    meta, _, files = load(snapshot)
    proposal_path = Path(meta['sandbox']) / 'proposal.json'
    assert sha(Path(meta['sandbox']) / 'candidate.py') == protocol['code_sha256']
    fixed = read(args.parent / 'C-result.json'); baseline = read(args.parent / 'B-result.json')
    packet = {'status': 'observed', 'run': diagnostic, 'evidence': [str(evidence)],
        'ranks': [{'file': str(evidence), **events}],
        'provenance': {'revision': candidate.id, 'devices': source.request.devices, 'framework_source': framework_source(),
            'scope': 'Existing run-only candidate_diagnostics executed once separately after fixed B/C; no observe/compare or clean acceptance.'}}
    feedback = {'trial': 'external-new-device-fixed-f5b', 'bundle': bundle.model_dump(mode='json'),
        'selection': json.loads(files['inputs/selection.json']), 'proposal': read(proposal_path),
        'baseline': baseline, 'candidate': fixed,
        'decision': {'accepted': False, 'reasons': ['fixed_diagnostic_only_fast_path_disabled'],
            'scope': 'Developer diagnostic continuation marker, not a formal paired-noise rejection or search result.'},
        'evaluation': {'formal_acceptance_performed': False, 'original_constraints': completed['results']['C']['constraints']},
        'measurement': {'scope': 'One fixed B/C, all samples retained; not paired-noise acceptance. Current-device original controls anchor applies.',
            'baseline_metrics': baseline['ranks'][0]['result']['metrics'], 'candidate_metrics': fixed['ranks'][0]['result']['metrics'],
            'baseline_constraints': completed['results']['B']['constraints'], 'candidate_constraints': completed['results']['C']['constraints']},
        'resources': {'groups': [], 'extra_collectors': {'after': {'candidate_diagnostics': packet}}},
        'supplemental_evidence': {'scope': 'Measured fixed execution and separately collected setup evidence only; no proposed implementation or unique cause.',
            'quality_scope': 'Fixed full evaluator checks passed while quality telemetry recorded 62 parent_fallback_setup_failure and 2 prefill_parent_passthrough calls. Run-only diagnostic valid is not an additional quality check.',
            'limits': 'The captured setup assertion reports eager_vs_parent_step_1; it supplies no full logits or numerical maximum. Candidate internal 0.008/0.008 check is stricter than unchanged task 0.02/0.02. This does not establish an outer-quality violation or unique numerical cause. No threshold or acceptance rule is changed.',
            'placement': 'New device variant has freshly measured original baseline controls. Earlier device timing anchors/acceptances remain historical; current request and prepared baseline govern this trial.',
            'diagnostic': {'path': str(evidence), 'sha256': sha(evidence)}}}
    raw = json.dumps(feedback, indent=2)
    packaged, compact = feedback_artifacts(raw)
    routed = json.loads(route(compact, current_revision=source.revisions.current().id))
    assert routed['supplemental_evidence'] == feedback['supplemental_evidence']
    assert packaged and routed['resources']['extra_collectors']['after']['candidate_diagnostics']['input'] in {'inputs/' + n for n in packaged} | set(packaged)
    history = candidate_artifacts(raw, source.store)
    assert history['previous-candidate/candidate.py'] == (Path(meta['sandbox']) / 'candidate.py').read_bytes()
    frozen = protocol['frozen_identities'].copy()
    for p in [args.parent / 'terminal-audit.json', args.parent / 'B-result.json', args.parent / 'C-result.json',
              args.parent / 'source-check.json', args.parent / 'followup-completed.json', evidence,
              args.diagnostic / 'diagnostic-result.json', args.diagnostic / 'followup-completed.json',
              source.output / 'project/current.json', source.output / 'prepared.json', source.output / 'request.json', snapshot, proposal_path]:
        frozen[str(p)] = sha(p)
    assert all(sha(p) == h for p, h in frozen.items())
    return source, feedback, frozen


def main(args):
    os.umask(0o077); config = load_config(args.config, []); config.usage.mode = 'observe_only'
    source, feedback, frozen = prepare(args, config)
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'baseline': source.prepared['revision'],
            'devices': source.request.devices, 'controls': source.prepared['controls_path'], 'frozen_files': len(frozen),
            'transport': 'supplemental facts and existing full diagnostic artifact survive CPU route', 'rounds': 1})); return
    args.output.mkdir(parents=True, exist_ok=True); assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(), 'parent': str(args.parent),
        'diagnostic': str(args.diagnostic), 'source': str(source.output), 'baseline': source.prepared['revision'],
        'devices': source.request.devices, 'rounds': 1, 'frozen_identities': frozen,
        'scope': 'One ordinary current-device trial from empty original baseline after one fresh discovery. Shared freshly measured controls, no new controls/reference copies. Historical exact f5b as previous candidate only; diagnostic accepted=false marker is not paired acceptance. Developer factual assistance explicit, not unaided original-entry or same-snapshot A/B. No manual selection/candidate/repair, production change or independent version acceptance in this batch.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    results = {}
    try:
        branch = args.output / 'qwen-continuation'; baseline = fork_run(source.output, branch)
        search = ProjectSearch(config, branch)
        assert not baseline.bundles and not (branch / 'discovery.json').exists()
        assert search.prepared == source.prepared and search.request == source.request
        write_json(args.output / 'preparation-reuse.json', {'source': str(source.output), 'baseline': baseline.id,
            'controls_path': search.prepared['controls_path'], 'request_sha256': sha(branch / 'request.json'),
            'initial_bundles': [], 'discovery': 'One fresh current-device original-baseline discovery required; no old accepted-version discovery reused.'})
        feedback_path = args.output / 'diagnostic-feedback.json'; write_json(feedback_path, feedback)
        argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config), 'optimize-project',
            '--request', str(branch / 'request.json'), '--output', str(branch), '--resume', '--rounds', '1', '--feedback', str(feedback_path)]
        write_json(args.output / 'search-protocol.json', {'at': now(), 'argv': argv, 'feedback_sha256': sha(feedback_path),
            'scope': 'Original production roles/allowed scopes/default parameter exploration; API observe-only. No additional trial retry or fixed native scheduled; real planner may request collectors.'})
        write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': 'fresh-discovery-and-one-trial'})
        with (args.output / 'search-controller.log').open('wb') as log: process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        current = ProjectSearch(config, branch).revisions.current()
        results['search'] = {'status': 'completed' if process.returncode == 0 else 'error', 'returncode': process.returncode,
            'current': current.id, 'bundles': list(current.bundles),
            'trials': [{'path': str(p), 'sha256': sha(p), 'decision': read(p).get('decision')} for p in sorted((branch / 'trials').glob('*/result.json'))],
            'independent_acceptance': 'not_run_in_this_batch'}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    same = all(sha(p) == h for p, h in frozen.items())
    results['preservation'] = {'status': 'passed' if same else 'failed', 'files': len(frozen)}
    terminal = {'phase': 'completed' if same and results.get('search', {}).get('status') == 'completed' else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'followup-results.json', results)
    write_json(args.output / 'state.json', terminal); write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for n in ['config', 'parent', 'diagnostic', 'output']: p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--preflight', action='store_true'); main(p.parse_args())
