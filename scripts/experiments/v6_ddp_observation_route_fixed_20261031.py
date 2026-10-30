"""One exact generated candidate B/C validation; no search, acceptance or publication."""
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
from kernel_optimizer.project.measurement import metric, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_audit_ddp_observation_route_candidate_20261031 import CODE, PROPOSAL


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(args, config):
    done = read(args.generation/'followup-completed.json'); audit = read(args.generation/'terminal-audit.json')
    assert done['phase'] == 'completed' and audit['core'] == framework_source()
    assert audit['code_review']['fixed_execution_justified'] and audit['all_frozen_unchanged']
    result = done['results']['candidate']; meta,_,files = load(Path(result['initial_snapshot']))
    directory = Path(meta['sandbox']); validate_proposal(directory)
    assert sha(directory/'candidate.py') == CODE and sha(directory/'proposal.json') == PROPOSAL
    assert read(directory/'proposal.json')['parameter_space'] == {}
    source = Path(read(args.generation/'protocol.json')['source'])
    original = ProjectSearch(config,source); parent = original.revisions.current()
    assert parent.id == audit['source_current'] == json.loads(files['inputs/project.json'])['current_revision']
    assert len(parent.bundles) == 1 and list(parent.bundles) == ['trial-d06aa22e2d-param-a72285d2']
    assert original.request.portable() == json.loads(files['inputs/request.json'])
    assert original.request.continuous_steps == 2 and original.request.warmup == 2 and original.request.repeats == 5
    assert metric(original.prepared['evaluation'],'step_ms','max','ranks_then_samples') == 133.379621
    assert original.prepared['adapter_sha256'] == 'b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193'
    assert list(original.request.diagnostic_metrics) == ['diagnostic.max_rank_nccl_kernel_sum_ms']
    discovery = read(source/'discovery.json'); assert discovery['revision'] == parent.id
    assert discovery['calls'] == json.loads(files['inputs/calls.json']) and discovery['groups'] == json.loads(files['inputs/groups.json'])
    devices = ['GPU-20e8003c-f531-d50c-084c-a9a97fe94004','GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321']
    assert original.request.devices == devices
    actual = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    assert all(d in actual for d in devices)
    frozen = audit['frozen_identities'].copy()
    for path in [directory/'candidate.py',directory/'proposal.json',Path(result['initial_snapshot']),
                 args.generation/'terminal-audit.json',args.generation/'public-terminal-audit.json',
                 args.generation/'followup-completed.json',args.generation/'followup-results.json',
                 args.generation/'candidate/events.jsonl',source/'project/current.json',
                 source/'prepared.json',source/'discovery.json',source/'request.json']:
        frozen[str(path)] = sha(path)
    assert all(sha(path) == h for path,h in frozen.items())
    return source,original,parent,directory,files,discovery,frozen


def main(args):
    os.umask(0o077); config = load_config(args.config,[]); config.usage.mode = 'observe_only'
    source,original,parent,directory,files,discovery,frozen = prepare(args,config)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free = int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free >= 8*1024**3 + 512*1024**2,free
    definitions = {'source':str(source),'parent':parent.id,'candidate_sha256':CODE,'proposal_sha256':PROPOSAL,
        'parameters':{},'devices':original.request.devices,'adapter_sha256':original.prepared['adapter_sha256'],
        'controls_path':original.prepared['controls_path'],'original_prepared':original.prepared['revision'],
        'frozen_files':len(frozen),'original_step_ms':133.379621,'step_limit_ms':140.04860205,
        'core':framework_source(),'growth_forecast_bytes':512*1024**2,'warning_free_bytes':8*1024**3,
        'free_bytes':free,'disk':disk_observation(Path('/root/autodl-tmp'))}
    if args.preflight:
        print(json.dumps({'status':'passed',**definitions})); return
    args.output.mkdir(parents=True,exist_ok=True); assert not (args.output/'protocol.json').exists()
    scope = ('One exact-byte fixed current acceptedccd90 B then generated508a C, original evaluator/two continuous full checks/two warmup/five clean/rank and original three postclean communication profiles/rank. '
             'Baseline validity/full-check/math gate, not timing gate; invalid baseline skips C. Empty parameters. No Agent/search/tuning/new collector choice/Nsight/observer/source-control rerun/reference copy/formal acceptance/pointer publishing. '
             'Original133.379621*1.05=140.04860205 constraint; fixed B/C is not paired-noise or independent acceptance. '
             'Generated copied-input/partial guards/after-update fallback/rank-asymmetric/global override limitations retained; no manual fixes or retry. All tails/failures retained.')
    write_json(args.output/'protocol.json',{'at':now(),**definitions,'frozen_identities':frozen,
        'sequence':['B','C'],'scope':scope,'generation_audit_sha256':sha(args.generation/'terminal-audit.json')})
    results = {}
    try:
        branch = args.output/'ddp-fixed'; forked = fork_run(source,branch); search = ProjectSearch(config,branch)
        assert forked.id == parent.id and search.prepared == original.prepared
        assert (branch/'request.json').read_bytes() == (source/'request.json').read_bytes()
        assert (branch/'discovery.json').read_bytes() == (source/'discovery.json').read_bytes()
        bundle,_ = search.bundle_from_outputs(directory,json.loads(files['inputs/selection.json']),forked,discovery,'fixed-observation-route')
        assert bundle.parameters == {} and list(bundle.replaces) == list(forked.bundles)
        assert not bundle.requires
        candidate = search.revisions.stage(bundle)
        old = {f.path:f.sha256 for f in forked.files}; new = {f.path:f.sha256 for f in candidate.files}
        assert all(new[k] == v for k,v in old.items()) and new[bundle.installer] == CODE
        assert candidate.contract_ref == forked.contract_ref and candidate.bundles == (bundle.id,)
        write_json(args.output/'preparation-reuse.json',{'source':str(source),'parent':parent.id,
            'shared_controls':search.prepared['controls_path'],'prepared_byte_equal':(branch/'prepared.json').read_bytes()==(source/'prepared.json').read_bytes(),
            'request_discovery_byte_equal':True,'no_gpu_discovery_or_reference_copy':True})
        write_json(args.output/'staged-identities.json',{'baseline':forked.id,'candidate':candidate.id,
            'bundle':bundle.model_dump(mode='json'),'original_files':old,'candidate_files':new,'contract_ref':forked.contract_ref})
        baseline_valid = False
        for arm,revision in [('B',forked),('C',candidate)]:
            if arm == 'C' and not baseline_valid:
                results[arm] = {'status':'not_run','reason':'Fixed accepted-parent quality/coverage/math gate invalid.'}; continue
            write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':arm})
            try:
                run = search.execute(revision,arm,'evaluate'); path = args.output/(arm+'-result.json'); write_json(path,run)
                valid = valid_run(run)
                full = valid and len(run['ranks']) == 2 and all(
                    r['result']['math_policy_unchanged'] and len(r['result']['checks']) == 2
                    and all(c['passed'] for c in r['result']['checks']) and len(r['result']['samples']) == 5
                    for r in run['ranks'])
                if arm == 'B': baseline_valid = full
                constraints = constraints_for_runs([run],search.request.objective,search.prepared['evaluation']) if valid else []
                results[arm] = {'status':'attempt_completed' if run.get('worker_pids_reaped') else 'execution_error',
                    'worker_status':run['status'],'valid_run':valid,'full_gate':full,'constraints':constraints,
                    'result':str(path),'result_sha256':sha(path),'request':str(branch/'jobs'/arm/'run/request.json'),'revision':revision.id}
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
    for name in ['config','generation','output']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true'); main(parser.parse_args())
