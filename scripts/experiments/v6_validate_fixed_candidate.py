"""Original paired validation of an unchanged, previously evaluated bundle."""
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
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    os.umask(0o077)
    config = load_config(args.config, [])
    config.usage.mode = 'observe_only'
    previous = read(args.parent/'protocol.json')
    audit = read(args.parent/'terminal-audit.json')
    terminal = read(args.parent/'followup-completed.json')
    assert terminal['phase'] == 'completed' and audit['all_frozen_unchanged']
    assert terminal['results']['B']['full_gate'] and terminal['results']['C']['full_gate']
    assert previous['core'] == framework_source()
    source = args.parent/'ddp-fixed'
    original = ProjectSearch(config, source)
    parent = original.revisions.current()
    identities = read(args.parent/'staged-identities.json')
    candidate = original.revisions.revision(identities['candidate'])
    bundle = original.revisions.bundle(identities['bundle']['id'])
    assert parent.id == previous['parent'] == audit['source_current']
    assert candidate.id == audit['candidate'] and bundle.parameters == {}
    assert next(f.sha256 for f in candidate.files if f.path == bundle.installer) == previous['candidate_sha256']
    paths = [args.parent/name for name in ['protocol.json', 'followup-completed.json', 'terminal-audit.json', 'staged-identities.json']]
    paths += [source/name for name in ['project/current.json', 'request.json', 'prepared.json', 'discovery.json']]
    frozen = {str(path):sha(path) for path in paths}
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free = int(subprocess.check_output(['df','-B1','--output=avail',str(args.output.parent)],text=True).splitlines()[-1])
    assert free >= args.warning_free_bytes + args.growth_forecast_bytes
    scope = ('One existing validate_bundle(publish=False), exact staged bundle and empty parameters, original fresh AB then BA. '
             'Original quality, workload, anchor, noise, constraints and diagnostic profiles unchanged; fixed samples not pooled. '
             'Normal resource_feedback profile and real collection-planner choose existing collectors without controller selection. '
             'No candidate/selector/parameter search/source-control rerun/reference copy/independent stage/publication. '
             'Preserve any rejection/error; no retry until pass.')
    definitions = {'source':str(source), 'parent':parent.id, 'candidate':candidate.id, 'bundle':bundle.id,
        'candidate_sha256':previous['candidate_sha256'], 'proposal_sha256':previous['proposal_sha256'],
        'parameters':{}, 'core':framework_source(), 'devices':original.request.devices,
        'prepared':original.prepared['revision'], 'shared_controls':original.prepared['controls_path'],
        'critical_files':len(frozen), 'free_bytes':free, 'warning_free_bytes':args.warning_free_bytes,
        'growth_forecast_bytes':args.growth_forecast_bytes, 'fixed_audit_sha256':sha(args.parent/'terminal-audit.json'), 'scope':scope}
    if args.preflight:
        print(json.dumps({'status':'passed', **definitions}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json', {'at':now(), **definitions, 'frozen_identities':frozen})
    results = {}
    try:
        branch = args.output/'ddp-validation'
        forked = fork_run(source, branch)
        search = ProjectSearch(config, branch)
        assert forked.id == parent.id and search.prepared == original.prepared and search.request == original.request
        assert (branch/'discovery.json').read_bytes() == (source/'discovery.json').read_bytes()
        staged = search.revisions.stage(bundle)
        assert staged == candidate
        write_json(args.output/'preparation-reuse.json', {'source':str(source), 'parent':parent.id, 'candidate':staged.id,
            'same_exact_bundle_files_contract':True, 'shared_controls':search.prepared['controls_path'],
            'request_prepared_discovery_equal':True, 'no_initial_discovery_source_controls_reference_copy':True})
        write_json(args.output/'state.json', {'at':now(), 'phase':'running', 'stage':'original_paired_validation'})
        result = search.validate_bundle(bundle, {'trial':args.trial_id, 'scope':scope}, discovery=search.current_discovery(), candidate=staged, publish=False)
        result_path = branch/'trials'/args.trial_id/'result.json'
        results['validation'] = {'status':'completed', 'result':str(result_path), 'result_sha256':sha(result_path),
            'decision':result['decision'], 'publication':result['publication'], 'measurement':result['measurement']}
        assert search.revisions.current().id == original.revisions.current().id == parent.id
        results['pointers'] = {'status':'passed', 'source_and_newbranch_current':parent.id}
    except Exception:
        results['controller'] = {'status':'error', 'traceback':traceback.format_exc()}
    same = all(sha(path) == digest for path,digest in frozen.items())
    results['preservation'] = {'status':'passed' if same else 'failed', 'critical_files':len(frozen), 'unchanged':same}
    terminal = {'phase':'completed' if same and 'validation' in results and 'controller' not in results else 'completed_with_errors',
                'finished_at':now(), 'results':results, 'disk':disk_observation(args.output.parent), 'scope':scope}
    write_json(args.output/'followup-results.json', results)
    write_json(args.output/'state.json', terminal)
    write_json(args.output/'followup-completed.json', terminal)
    print(json.dumps({'at':now(), 'phase':terminal['phase'], 'results':results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'parent', 'output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--trial-id',required=True)
    parser.add_argument('--warning-free-bytes',type=int,default=8*1024**3)
    parser.add_argument('--growth-forecast-bytes',type=int,default=512*1024**2)
    parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
