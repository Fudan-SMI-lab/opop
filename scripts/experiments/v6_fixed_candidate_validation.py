"""One unchanged generated candidate against its current parent, using original evaluation.

Experiment orchestration only; workload, devices, bindings and quality rules come
from the source request and generation artifacts, never case-specific constants.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    done = read(args.generation / 'followup-completed.json')
    audit = read(args.generation / 'terminal-audit.json')
    assert done['phase'] == 'completed' and audit['core'] == framework_source()
    assert audit['code_review']['fixed_execution_justified']
    result = done['results']['candidate']
    meta, _, files = load(Path(result['initial_snapshot']))
    directory = Path(meta['sandbox'])
    validate_proposal(directory)
    for name in ['candidate', 'proposal']:
        suffix = '.py' if name == 'candidate' else '.json'
        assert sha(directory / (name + suffix)) == result[name + '_sha256']
    source = Path(read(args.generation / 'protocol.json')['source'])
    original = ProjectSearch(config, source)
    parent = original.revisions.current()
    assert parent.id == audit['source_current'] == json.loads(files['inputs/project.json'])['current_revision']
    assert original.request.portable() == json.loads(files['inputs/request.json'])
    discovery = original.current_discovery()
    assert discovery['revision'] == parent.id
    actual_devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    assert all(device in actual_devices for device in original.request.devices)
    paths = [directory/'candidate.py', directory/'proposal.json', Path(result['initial_snapshot']),
             args.generation/'terminal-audit.json', args.generation/'followup-completed.json']
    paths += [source/name for name in ['project/current.json', 'request.json', 'prepared.json', 'discovery.json']]
    identities = {str(path): sha(path) for path in paths}
    return original, parent, directory, files, discovery, identities, result


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    original, parent, directory, files, discovery, identities, generated = prepare(args, config)
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    free = int(subprocess.check_output(['df', '-B1', '--output=avail', str(args.output.parent)], text=True).splitlines()[-1])
    assert free >= args.warning_free_bytes + args.growth_forecast_bytes
    scope = ('One current accepted parent B then exact generated C using unchanged original evaluator and empty parameters. '
             'Original full checks/warmup/repeats/diagnostic profiles/constraints retained. Baseline quality gate only, not timing. '
             'No Agent/search/parameter alternatives/collector selection/native/source-control rerun/reference copy/publication. '
             'All failures/tails retained; fixed B/C is not formal paired-noise or independent acceptance.')
    definitions = {'source': str(original.output), 'parent': parent.id, 'core': framework_source(),
        'candidate_sha256': generated['candidate_sha256'], 'proposal_sha256': generated['proposal_sha256'],
        'parameters': {}, 'devices': original.request.devices, 'original_request': original.request.portable(),
        'adapter_sha256': original.prepared['adapter_sha256'], 'controls_path': original.prepared['controls_path'],
        'original_prepared': original.prepared['revision'], 'critical_files': len(identities),
        'growth_forecast_bytes': args.growth_forecast_bytes, 'warning_free_bytes': args.warning_free_bytes,
        'free_bytes': free, 'scope': scope}
    if args.preflight:
        print(json.dumps({'status': 'passed', **definitions}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json', {'at': now(), **definitions, 'frozen_identities': identities, 'sequence': ['B', 'C']})
    results = {}
    try:
        branch = args.output/'ddp-fixed'
        forked = fork_run(original.output, branch)
        search = ProjectSearch(config, branch)
        assert forked.id == parent.id and search.prepared == original.prepared
        for name in ['request.json', 'prepared.json', 'discovery.json']:
            assert (branch/name).read_bytes() == (original.output/name).read_bytes()
        bundle, _ = search.bundle_from_outputs(directory, json.loads(files['inputs/selection.json']), forked, discovery, args.bundle_id)
        assert bundle.parameters == {}
        candidate = search.revisions.stage(bundle)
        assert candidate.contract_ref == forked.contract_ref
        assert {f.path:f.sha256 for f in candidate.files}[bundle.installer] == generated['candidate_sha256']
        write_json(args.output/'staged-identities.json', {'baseline': forked.id, 'candidate': candidate.id, 'bundle': bundle.model_dump(mode='json'), 'contract_ref': forked.contract_ref})
        write_json(args.output/'preparation-reuse.json', {'source': str(original.output), 'parent': parent.id, 'request_prepared_discovery_byte_equal': True, 'controls_shared': search.prepared['controls_path'], 'no_source_or_reference_rerun_copy': True})
        baseline_valid = False
        for arm, revision in [('B', forked), ('C', candidate)]:
            if arm == 'C' and not baseline_valid:
                results[arm] = {'status': 'not_run', 'reason': 'Original baseline quality/coverage/math gate failed.'}
                continue
            write_json(args.output/'state.json', {'at': now(), 'phase': 'running', 'stage': arm})
            try:
                run = search.execute(revision, arm, 'evaluate')
                path = args.output/(arm+'-result.json')
                write_json(path, run)
                valid = valid_run(run)
                full = valid and len(run['ranks']) == len(search.request.devices) and all(
                    rank['result']['math_policy_unchanged']
                    and len(rank['result']['checks']) == search.request.continuous_steps
                    and all(check['passed'] for check in rank['result']['checks'])
                    and len(rank['result']['samples']) == search.request.repeats for rank in run['ranks'])
                if arm == 'B':
                    baseline_valid = full
                results[arm] = {'status': 'attempt_completed' if run.get('worker_pids_reaped') else 'execution_error',
                    'worker_status': run['status'], 'valid_run': valid, 'full_gate': full,
                    'constraints': constraints_for_runs([run], search.request.objective, search.prepared['evaluation']) if valid else [],
                    'result': str(path), 'result_sha256': sha(path), 'request': str(branch/'jobs'/arm/'run/request.json'), 'revision': revision.id}
            except Exception:
                results[arm] = {'status': 'execution_error', 'traceback': traceback.format_exc()}
            write_json(args.output/'followup-results.json', results)
            print(json.dumps({'at': now(), 'endpoint': arm, **results[arm]}), flush=True)
        assert search.revisions.current().id == original.revisions.current().id == parent.id
        results['pointers'] = {'status': 'passed', 'source_and_branch_current': parent.id}
    except Exception:
        results['controller'] = {'status': 'execution_error', 'traceback': traceback.format_exc()}
    same = all(sha(path) == digest for path, digest in identities.items())
    results['preservation'] = {'status': 'passed' if same else 'failed', 'critical_files': len(identities), 'unchanged': same}
    terminal = {'phase': 'completed' if all(r['status'] in ['passed', 'attempt_completed'] for r in results.values()) else 'completed_with_errors',
                'finished_at': now(), 'results': results, 'disk': disk_observation(args.output.parent), 'scope': scope}
    write_json(args.output/'followup-results.json', results)
    write_json(args.output/'state.json', terminal)
    write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'generation', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bundle-id', required=True)
    parser.add_argument('--warning-free-bytes', type=int, default=8*1024**3)
    parser.add_argument('--growth-forecast-bytes', type=int, default=512*1024**2)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
