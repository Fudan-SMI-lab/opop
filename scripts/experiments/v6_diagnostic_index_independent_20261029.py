"""One independent AB/BA per frozen generated version, without publishing or agents."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    protocol = read(args.parent/'protocol.json')
    audit = read(args.parent/'terminal-audit.json')
    done = read(args.parent/'followup-completed.json')
    staged = read(args.parent/'staged-identities.json')
    assert done['phase'] == 'completed' and done['frozen_files_unchanged']
    assert audit['core'] == protocol['core'] == framework_source()
    assert (audit['clean_rank_samples'], audit['full_zero_error_checks'], audit['recomputed_traces']) == (30, 12, 48)
    frozen = dict(protocol['frozen_identities'])
    for name in ['protocol.json', 'terminal-audit.json', 'followup-completed.json', 'staged-identities.json']:
        frozen[str(args.parent/name)] = sha(args.parent/name)
    definitions = {}
    for case, d in protocol['cases'].items():
        search = ProjectSearch(config, args.parent/(case+'-fixed'))
        parent = search.revisions.revision(search.prepared['revision'])
        candidate = search.revisions.revision(staged[case]['candidate'])
        assert not parent.bundles and len(candidate.bundles) == 1
        assert parent.contract_ref == candidate.contract_ref
        assert search.revisions.current().id == d['parent']
        assert ProjectSearch(config, Path(d['source'])).revisions.current().id == d['source_current']
        assert search.prepared['adapter_sha256'] == d['adapter_sha256']
        assert search.prepared['controls_path'] == d['controls_path']
        bundle = search.revisions.bundle(candidate.bundles[0])
        assert bundle.parameters == d['parameters']
        assert {f.path:f.sha256 for f in candidate.files}[bundle.installer] == d['candidate_sha256']
        assert (search.request.continuous_steps, search.request.warmup, search.request.repeats) == (2, 2, 5)
        assert search.request.objective.get('paired_repeats', 2) == 2
        assert len(search.request.diagnostic_metrics) == (4 if case == 'fsdp' else 0)
        assert all(valid_run(read(args.parent/f'{case}-{arm}-result.json')) for arm in ['B','C'])
        for name in ['project/current.json', 'prepared.json', 'request.json']:
            frozen[str(search.output/name)] = sha(search.output/name)
        definitions[case] = {'source':str(search.output), 'baseline':parent.id, 'candidate':candidate.id,
            'fixed_current':search.revisions.current().id, 'original_source':d['source'],
            'original_source_current':d['source_current'], 'candidate_sha256':d['candidate_sha256'],
            'parameters':d['parameters'], 'bundles':list(candidate.bundles), 'contract_ref':candidate.contract_ref,
            'adapter_sha256':d['adapter_sha256'], 'controls_path':d['controls_path'],
            'request_sha256':sha(search.output/'request.json')}

    # Existing independent full-source checks are reused only with their exact
    # controls/source hashes. This is identity preservation, not a new source run.
    work = args.parent.parent
    qpath = work/'qwen-routed-independent-20261029/source-check.json'
    qcheck = read(qpath)
    assert valid_run(qcheck)
    qr = qcheck['ranks'][0]['result']
    assert qr['adapter_sha256'] == definitions['qwen']['adapter_sha256']
    assert len(qr['checks']) == 2 and all(c['passed'] and c['exact_tokens'] and c['finite_logits'] and c['max_abs_error'] == 0 for c in qr['checks'])
    assert all(not c['passed'] for c in qr['negative_checks'].values())
    qreference = read(work/'qwen-end-to-end-original-20261029/protocol.json')['independent_source_reference']
    qcontrol = str(Path(definitions['qwen']['controls_path'])/'reference-rank-0.pt')
    assert qreference['sha256'] == qr['source_reference_sha256']
    assert frozen[qcontrol] == qr['controls_reference_sha256']
    frozen[qreference['path']] = qr['source_reference_sha256']
    frozen[str(qpath)] = sha(qpath)
    fprior = audit['reused_fsdp_source_check']
    fpath = Path(fprior['prior_check']); fcheck = read(fpath)
    assert sha(fpath) == fprior['sha256'] and fcheck['status'] == 'passed'
    assert fcheck['adapter_sha256'] == definitions['fsdp']['adapter_sha256']
    assert all(r['same_allocated_and_peak'] for r in fcheck['memory'])
    for rank in fcheck['ranks']:
        assert len(rank['checks']) == 2
        assert all(c['comparison']['passed'] and c['comparison']['max_abs_error'] == 0 for c in rank['checks'])
    frozen[str(fpath)] = fprior['sha256']
    for r in fprior['unchanged_references']:
        frozen[r['path']] = r['sha256']
    assert all(sha(path) == h for path, h in frozen.items())
    reuse = {'qwen':{'report':str(qpath), 'sha256':sha(qpath), 'mapping':qr['mapping'],
                     'source_reference':qreference, 'controls_reference':qcontrol},
             'fsdp':fprior,
             'scope':'Reused independent source/control comparisons with exact hashes. No new numerical source fixture, model run or reference copy. Qwen manual mapping and FSDP developer-assisted 512-byte owner repair remain explicit; arbitrary extra owners are not certified.'}
    return definitions, frozen, reuse


def independent(args, config, case, d):
    source = ProjectSearch(config, Path(d['source']))
    destination = args.output/(case+'-independent')
    baseline = fork_run(source.output, destination, revision_id=d['baseline'])
    search = ProjectSearch(config, destination)
    assert not baseline.bundles and search.prepared == source.prepared and search.request == source.request
    result = promote_version(search, source.output, revision_id=d['candidate'], publish=False)
    assert search.revisions.current().id == baseline.id
    assert source.revisions.current().id == d['fixed_current']
    path = destination/'version-trials'/result['trial']/'result.json'
    return {'status':'completed', 'result':str(path), 'result_sha256':sha(path),
            **{k:result[k] for k in ['trial','revision','parent','quality','coverage_errors','measurement','decision','publication']}}


def main(args):
    os.umask(0o077)
    config = load_config(args.config, []); config.usage.mode = 'observe_only'
    definitions, frozen, reuse = prepare(args, config)
    if args.preflight:
        print(json.dumps({'status':'passed', 'cases':definitions, 'frozen_files':len(frozen),
                          'core':framework_source(), 'disk':disk_observation(Path('/root/autodl-tmp'))}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json', {'at':now(), 'core':framework_source(), 'cases':definitions,
        'frozen_identities':frozen, 'fixed_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'sequence':['qwen independent B/C/C/B', 'fsdp independent B/C/C/B K2', 'conditional one Qwen candidate Nsys'],
        'scope':'Exact frozen generated versions, no search, parameter selection, new roles or adoption. Existing full-version rules, publish=False, one AB/BA per case. All original clean5, two continuous quality, warmup2 and FSDP four separate diagnostic metrics remain. No repeated-until-pass or new reference copies. Accepted/rejected decisions do not rewrite prior fixed diagnostics or historical acceptances.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output/'source-evidence-reuse.json', reuse)
    write_json(args.output/'preparation-reuse.json', {'cases':definitions, 'publish':False,
        'scope':'Fork original prepared empty baseline, import entire exact staged candidate through existing stage_version. Shared immutable controls. All source/fixed/destination current pointers retained.'})
    results = {}
    for case, d in definitions.items():
        name = case+'-independent'
        write_json(args.output/'state.json', {'at':now(), 'phase':'running', 'stage':name})
        try:
            results[name] = independent(args, config, case, d)
        except Exception:
            results[name] = {'status':'error', 'traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json', results)
        print(json.dumps({'at':now(), 'stage':name, 'status':results[name]['status']}), flush=True)
    name = 'qwen-native'
    write_json(args.output/'state.json', {'at':now(), 'phase':'running', 'stage':name})
    if results['qwen-independent'].get('quality') != 'pass' or results['qwen-independent'].get('coverage_errors'):
        results[name] = {'status':'not_run', 'reason':'Independent Qwen quality/coverage did not pass.'}
    else:
        try:
            search = ProjectSearch(config, args.output/'qwen-independent')
            result = Nsys().collect(search, search.revisions.revision(results['qwen-independent']['revision']))
            path = args.output/'native-qwen-candidate.json'; write_json(path, result)
            results[name] = {'status':'completed' if result['status'] == 'observed' else 'collection_error',
                'path':str(path), 'sha256':sha(path),
                'scope':'One exact-current-candidate perturbed native profile, not clean timing or acceptance. No baseline rerun and no reuse of another candidate Graph trace. Existing FSDP metric traces supply collective-path observations.'}
        except Exception:
            results[name] = {'status':'error', 'traceback':traceback.format_exc()}
    try:
        assert all(sha(path) == h for path, h in frozen.items())
        for case, d in definitions.items():
            assert ProjectSearch(config, Path(d['source'])).revisions.current().id == d['fixed_current']
            assert ProjectSearch(config, Path(d['original_source'])).revisions.current().id == d['original_source_current']
            target = args.output/(case+'-independent')
            if (target/'project/current.json').exists():
                assert read(target/'project/current.json')['revision'] == d['baseline']
        results['preservation'] = {'status':'completed', 'frozen_files':len(frozen), 'all_current_pointers_unchanged':True}
    except Exception:
        results['preservation'] = {'status':'error', 'traceback':traceback.format_exc()}
    terminal = {'phase':'completed' if all(r['status'] in ['completed','not_run'] for r in results.values()) else 'completed_with_errors',
                'finished_at':now(), 'results':results, 'disk':disk_observation(Path('/root/autodl-tmp')),
                'scope':'Completion, quality, source evidence reuse and independent performance decisions are separate. No publication or production prompt change.'}
    write_json(args.output/'followup-results.json', results)
    write_json(args.output/'state.json', terminal)
    write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    main(p.parse_args())
