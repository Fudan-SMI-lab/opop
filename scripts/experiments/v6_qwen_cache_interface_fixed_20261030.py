"""One exact-byte B/C execution, original full work; no search or acceptance."""
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
from scripts.experiments.v6_audit_qwen_cache_interface_candidate_20261030 import CODE, PROPOSAL


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    done = read(args.replay/'followup-completed.json'); audit = read(args.replay/'terminal-audit.json')
    assert done['phase'] == 'completed' and audit['core'] == framework_source()
    assert audit['code_review']['fixed_execution_justified']
    row = done['results']['candidate']; meta, _, files = load(Path(row['initial_snapshot']))
    directory = Path(meta['sandbox']); validate_proposal(directory)
    assert sha(directory/'candidate.py') == CODE and sha(directory/'proposal.json') == PROPOSAL
    assert read(directory/'proposal.json')['parameter_space'] == {}
    source = args.replay.parent/'qwen-setup-cpu-feedback-20261030/qwen-continuation'
    search = ProjectSearch(config, source); parent = search.revisions.current()
    assert parent.id == audit['source_current'] == json.loads(files['inputs/project.json'])['current_revision']
    assert not parent.bundles and search.request.portable() == json.loads(files['inputs/request.json'])
    assert search.request.continuous_steps == 2 and search.request.warmup == 2 and search.request.repeats == 5
    assert not search.request.diagnostic_metrics
    assert search.prepared['evaluation']['ranks'][0]['result']['metrics']['ttft_ms'] == 23.285755
    discovery = read(source/'discovery.json')
    assert discovery['revision'] == parent.id
    assert discovery['calls'] == json.loads(files['inputs/calls.json'])
    assert discovery['groups'] == json.loads(files['inputs/groups.json'])
    devices = search.request.devices
    assert devices == ['GPU-20e8003c-f531-d50c-084c-a9a97fe94004']
    available = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    assert all(d in available for d in devices)
    frozen = audit['frozen_identities'].copy()
    for path in [directory/'candidate.py',directory/'proposal.json',Path(row['initial_snapshot']),
                 args.replay/'terminal-audit.json',args.replay/'public-terminal-audit.json',
                 args.replay/'followup-completed.json',source/'project/current.json',
                 source/'prepared.json',source/'discovery.json',source/'request.json']:
        frozen[str(path)] = sha(path)
    assert all(sha(p) == h for p,h in frozen.items())
    return source, search, parent, directory, files, discovery, frozen


def main(args):
    os.umask(0o077); config = load_config(args.config,[]); config.usage.mode = 'observe_only'
    source, original, parent, directory, files, discovery, frozen = prepare(args,config)
    definitions = {'source':str(source),'parent':parent.id,'candidate_sha256':CODE,'proposal_sha256':PROPOSAL,
        'parameters':{},'devices':original.request.devices,'adapter_sha256':original.prepared['adapter_sha256'],
        'controls_path':original.prepared['controls_path'],'frozen_files':len(frozen),
        'original_ttft_ms':23.285755,'ttft_limit_ms':24.45004275,
        'core':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp'))}
    if args.preflight:
        print(json.dumps({'status':'passed',**definitions})); return
    args.output.mkdir(parents=True,exist_ok=True); assert not (args.output/'protocol.json').exists()
    scope = ('One exact-byte B then C, original evaluator/full two continuous checks/two warmup/five clean samples. '
             'Baseline quality/coverage gate, not TTFT gate; failed baseline skips candidate. Empty parameters. '
             'Generated setup verify-failure flag and first-layer ownership limitations retained; outer quality is authority. '
             'No hand repair, observer, Agent/search/tuning/formal acceptance/Nsight/source-reference copy/pointer publishing. '
             'Current-device original TTFT23.285755*1.05=24.45004275; all timings diagnostic, no reanchor or retries.')
    write_json(args.output/'protocol.json',{'at':now(),**definitions,'frozen_identities':frozen,
        'sequence':['B','C'],'scope':scope,'replay_audit_sha256':sha(args.replay/'terminal-audit.json')})
    results = {}
    try:
        branch = args.output/'qwen-fixed'; forked = fork_run(source,branch); search = ProjectSearch(config,branch)
        assert forked.id == parent.id and search.prepared == original.prepared
        assert (branch/'request.json').read_bytes() == (source/'request.json').read_bytes()
        assert (branch/'discovery.json').read_bytes() == (source/'discovery.json').read_bytes()
        bundle,_ = search.bundle_from_outputs(directory,json.loads(files['inputs/selection.json']),forked,discovery,'fixed-cache-interface')
        assert bundle.parameters == {}
        candidate = search.revisions.stage(bundle)
        old = {f.path:f.sha256 for f in forked.files}; new = {f.path:f.sha256 for f in candidate.files}
        assert all(new[k] == v for k,v in old.items()) and new[bundle.installer] == CODE
        assert candidate.contract_ref == forked.contract_ref
        write_json(args.output/'preparation-reuse.json',{'source':str(source),'parent':parent.id,
            'shared_controls':search.prepared['controls_path'],'prepared_byte_equal':(branch/'prepared.json').read_bytes()==(source/'prepared.json').read_bytes(),
            'request_discovery_byte_equal':True,'no_gpu_discovery_or_reference_copy':True})
        write_json(args.output/'staged-identities.json',{'baseline':forked.id,'candidate':candidate.id,
            'bundle':bundle.model_dump(mode='json'),'original_files':old,'candidate_files':new,'contract_ref':forked.contract_ref})
        baseline_valid = False
        for arm, revision in [('B',forked),('C',candidate)]:
            if arm == 'C' and not baseline_valid:
                results[arm] = {'status':'not_run','reason':'Fixed baseline quality/coverage invalid.'}; continue
            write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':arm})
            try:
                run = search.execute(revision,arm,'evaluate'); path = args.output/(arm+'-result.json'); write_json(path,run)
                valid = valid_run(run)
                if arm == 'B': baseline_valid = valid
                constraints = constraints_for_runs([run],search.request.objective,search.prepared['evaluation']) if valid else []
                results[arm] = {'status':'attempt_completed' if run.get('worker_pids_reaped') else 'execution_error',
                    'worker_status':run['status'],'valid_run':valid,'constraints':constraints,'result':str(path),
                    'result_sha256':sha(path),'request':str(branch/'jobs'/arm/'run/request.json'),'revision':revision.id}
            except Exception:
                results[arm] = {'status':'execution_error','traceback':traceback.format_exc()}
            write_json(args.output/'followup-results.json',results)
            print(json.dumps({'at':now(),'endpoint':arm,**results[arm]}),flush=True)
        assert search.revisions.current().id == original.revisions.current().id == parent.id
        results['pointers'] = {'status':'passed','source_and_branch_current':parent.id}
    except Exception:
        results['controller'] = {'status':'execution_error','traceback':traceback.format_exc()}
    same = all(sha(p) == h for p,h in frozen.items())
    results['preservation'] = {'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal = {'phase':'completed' if all(r['status'] in ['passed','attempt_completed'] for r in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp')),'scope':scope}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal); write_json(args.output/'followup-completed.json',terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config','replay','output']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true'); main(parser.parse_args())
