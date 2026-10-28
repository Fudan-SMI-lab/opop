"""Fixed generated Qwen candidate validation; no search, publication or acceptance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, ProjectContract
from kernel_optimizer.project.aggregation import rank_requirements
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs(args):
    terminal = read(args.replay / 'followup-completed.json'); audit = read(args.replay / 'terminal-audit.json')
    assert terminal['phase'] == 'completed' and audit['core'] == framework_source()
    prior = args.replay.parent / 'qwen-relation-diagnostic-20261029'
    relation = read(prior / 'protocol.json')
    template_path = Path(next(iter(relation['source_requests'])))
    template = ExecutionRequest.model_validate(read(template_path))
    observer = Path(read(prior / '0-B-observed/request.json')['entry'])
    assert sha(observer) == 'a06b6c0e14495bffb68aa498c845ce66378223e9c95579b0b66a7bd463cc566c'
    assert template.parameters['action'] == 'evaluate' and template.parameters['bundles'] == []
    assert template.parameters['request']['repeats'] == 5 and template.parameters['request']['continuous_steps'] == 2
    for arm in ['control', 'treatment']:
        row = terminal['results'][arm]; directory = Path(row['agent']['sandbox'])
        validate_proposal(directory)
        assert sha(directory / 'candidate.py') == row['candidate_sha256'] == audit['roles'][arm]['candidate_sha256']
        assert read(directory / 'proposal.json')['parameter_space'] == {}
        metadata, _, files = load(Path(row['agent']['generation_snapshots'][0]))
        assert metadata['framework_source'] == framework_source()
        assert all((directory / n).read_bytes() == raw for n, raw in files.items())
    expected = audit['unchanged_frozen_source_controls_requests']
    assert all(sha(p) == h for p, h in expected.items())
    return terminal, audit, template, observer, relation['source_reference'], expected


def main(args):
    os.umask(0o077)
    terminal, audit, template, observer, reference, expected = inputs(args)
    config = load_config(args.config, []); config.usage.mode = 'observe_only'
    source = args.replay.parent / 'qwen-end-to-end-original-followup-20261029/qwen-original'
    source_search = ProjectSearch(config, source)
    parent = source_search.revisions.current()
    assert parent.id == 'b61996e1a998de911fa0412b595ef484b75dc79765d404aed835051e885b8a3c' and not parent.bundles
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'parent': parent.id,
            'arms': {a: terminal['results'][a]['candidate_sha256'] for a in ['control', 'treatment']},
            'parameters': {}, 'source_controls_hashes_verified': len(expected), 'observer_sha256': sha(observer)}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists()
    sequence = [('B', True), ('control', False), ('control', True), ('treatment', False), ('treatment', True)]
    write_json(args.output / 'protocol.json', {'at': now(), 'core': framework_source(), 'source': str(source),
        'replay': str(args.replay), 'replay_audit_sha256': sha(args.replay / 'terminal-audit.json'),
        'source_revision': parent.id, 'sequence': sequence, 'observer': str(observer), 'observer_sha256': sha(observer),
        'source_reference': reference, 'frozen_identities': expected, 'template': template.model_dump(mode='json'),
        'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Each endpoint once, B observed and each exact generated candidate unobserved/observed. Empty generated parameter spaces. No code edits, search, formal acceptance, native trace, GPU reference copies or source-pointer publication. Original quality/TTFT constraint and all samples retained. Observed CPU copies perturb synchronization/host work/lifetimes; timings diagnostic only.'})
    results = {}
    try:
        branch = args.output / 'fixed-validation'
        forked = fork_run(source, branch); search = ProjectSearch(config, branch)
        assert forked.id == parent.id
        assert (branch / 'request.json').read_bytes() == (source / 'request.json').read_bytes()
        assert search.prepared['adapter_sha256'] == '498c90f3f6f91b993ac8a586ac3230ffe2e6901752165b26948142d978f7cff7'
        discovery = read(branch / 'discovery.json')
        revisions = {'B': forked}; base_files = {f.path: f.sha256 for f in parent.files}
        for arm in ['control', 'treatment']:
            row = terminal['results'][arm]; directory = Path(row['agent']['sandbox'])
            _, _, files = load(Path(row['agent']['generation_snapshots'][0]))
            assert json.loads(files['inputs/project.json'])['current_revision'] == parent.id
            bundle, _ = search.bundle_from_outputs(directory, json.loads(files['inputs/selection.json']), parent, discovery, 'fixed-feedback-' + arm)
            assert bundle.parameters == {}
            revision = search.revisions.stage(bundle)
            assert revision.contract_ref == parent.contract_ref
            actual = {f.path: f.sha256 for f in revision.files}
            assert all(actual.get(k) == v for k, v in base_files.items())
            assert actual[bundle.installer] == row['candidate_sha256']
            revisions[arm] = revision
        write_json(args.output / 'staged-identities.json', {name: {'revision': r.id, 'bundles': list(r.bundles),
            'contract_ref': r.contract_ref, 'files': {f.path: f.sha256 for f in r.files}} for name, r in revisions.items()})
        for index, (arm, observed) in enumerate(sequence):
            name = f'{index}-{arm}-' + ('observed' if observed else 'unobserved')
            write_json(args.output / 'state.json', {'at': now(), 'phase': 'running', 'stage': name})
            try:
                revision = revisions[arm]; job = branch / 'jobs' / name
                materialized = search.revisions.materialize(revision, job / 'source')
                contract = search.revisions.load(revision.contract_ref, ProjectContract)
                bundles = []
                for ident in revision.bundles:
                    bundle = search.revisions.bundle(ident)
                    bundles.append({'id': ident, 'file': str(materialized / bundle.installer), 'parameters': bundle.parameters,
                        'specs_path': str(materialized / Path(bundle.installer).with_suffix('.targets.json')),
                        'expected_calls': list(bundle.calls), 'expected_calls_by_rank': rank_requirements(bundle,
                            search.revisions.load(search.revisions.revision(bundle.parent_revision).plan_ref, ExecutionPlan), contract.world_size)})
                parameters = {**template.parameters, 'adapter': str(materialized / 'opop_adapter.py'), 'bundles': bundles}
                if observed: parameters['diagnostic_source_reference'] = reference['path']
                request = template.model_copy(update={'cwd': materialized, 'entry': observer if observed else materialized / 'opop_evaluate.py',
                    'function': 'observed_worker' if observed else template.function, 'parameters': parameters,
                    'output': job / 'run', 'wait_for_devices': True})
                run = LocalExecutor().run(request)
                path = args.output / (name + '-result.json'); write_json(path, run)
                valid = valid_run(run)
                constraints = constraints_for_runs([run], search.request.objective, search.prepared['evaluation']) if valid else []
                sidecar = job / 'run/relation-observations.json'
                results[name] = {'status': 'attempt_completed' if run.get('worker_pids_reaped') else 'execution_error',
                    'worker_status': run['status'], 'valid_run': valid, 'constraints': constraints,
                    'result': str(path), 'result_sha256': sha(path), 'request': str(job / 'run/request.json'),
                    'observer_summary': str(sidecar) if sidecar.exists() else None,
                    'revision': revision.id, 'bundles': list(revision.bundles)}
                if observed and not sidecar.exists(): results[name]['status'] = 'execution_error'
            except Exception:
                results[name] = {'status': 'execution_error', 'traceback': traceback.format_exc()}
            write_json(args.output / 'followup-results.json', results)
            print(json.dumps({'at': now(), 'endpoint': name, 'status': results[name]['status'], 'worker_status': results[name].get('worker_status')}), flush=True)
        assert search.revisions.current().id == parent.id == source_search.revisions.current().id
        assert all(sha(p) == h for p, h in expected.items())
        assert all(sha(Path(terminal['results'][arm]['agent']['sandbox']) / 'candidate.py') == terminal['results'][arm]['candidate_sha256']
                   for arm in ['control', 'treatment'])
        results['frozen-identities'] = {'status': 'passed', 'source_and_staged_pointers_unchanged': True,
                                      'all_frozen_hashes_unchanged': True}
    except Exception:
        results['controller'] = {'status': 'execution_error', 'traceback': traceback.format_exc()}
    completion = {'phase': 'completed' if all(r['status'] in ['passed', 'attempt_completed'] for r in results.values()) else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp')),
        'scope': 'Attempt completion does not imply quality, constraints or optimization acceptance.'}
    write_json(args.output / 'state.json', completion); write_json(args.output / 'followup-completed.json', completion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'replay', 'output']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args())
