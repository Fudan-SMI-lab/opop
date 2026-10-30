"""One fixed exact-Agent B/C observation; no search, tuning or acceptance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

PROBE = '92279e28dc1e548efe86ef44467a3b0d09f7d94586da3dc16c0a3f55b773db96'
PLAN = '0a82e46248377b94a7ec21954eb34d2bdf556d65796557556923ebf9eeefcf13'
DEVICES = ['GPU-20e8003c-f531-d50c-084c-a9a97fe94004','GPU-a24af2e6-99bb-9729-5d14-50fa10a2b321']
FORECAST = 512*1024**2


def now(): return datetime.now(timezone.utc).isoformat()


def prepare(parent):
    audit = read(parent/'terminal-audit.json')
    assert audit['all_frozen_unchanged'] and audit['core'] == framework_source()
    assert audit['decision'].startswith('Eligible for ONE declared exact-byte fixed')
    result = read(parent/'followup-completed.json')['results']['diagnostic-author']
    probe = Path(result['probe']); plan = Path(result['plan'])
    assert sha(probe) == PROBE and sha(plan) == PLAN
    index = read(Path(result['agent']['sandbox'])/'inputs/evidence-index.json')
    expected = read(parent/'protocol.json')['frozen_identities'].copy()
    for path in [probe,plan,parent/'terminal-audit.json',parent/'public-terminal-audit.json',
                 parent/'followup-completed.json',parent/'diagnostic-author/events.jsonl',
                 parent/'audit/independent-cpu-mechanics/result.json',parent/'audit/independent-cpu-mechanics/fixture.py',
                 parent/'audit/independent-cpu-mechanics/execution.json',parent/'terminal-private-history/complete-access-index.json']:
        expected[str(path)] = sha(path)
    requests = {}; identities = {}
    for arm in ('parent','candidate'):
        item = index['original_execution_requests'][arm]; path = Path(item['path'])
        assert sha(path) == item['sha256']; request = read(path); task = request['parameters']['request']
        assert task['continuous_steps'] == 2 and task['warmup'] == 2 and task['repeats'] == 5
        assert task['quality']['atol'] == 1e-6 and task['quality']['rtol'] == 1e-4
        assert request['parameters']['action'] == 'evaluate' and request['devices'] == DEVICES
        assert task['workload']['world_size'] == 2 and task['workload']['accumulation_steps'] == 4
        assert task['workload']['sequence_length'] == 128 and task['workload']['micro_batch'] == 1
        assert task['profiling'].get('metric_repeats',3) == 3
        assert list(task['diagnostic_metrics']) == ['diagnostic.max_rank_nccl_kernel_sum_ms']
        bundles = request['parameters']['bundles']
        if arm == 'parent': assert bundles == []
        else:
            assert len(bundles) == 1 and bundles[0]['parameters'] == {}
            assert sha(bundles[0]['file']) == 'ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b'
        assert sha(request['parameters']['adapter']) == 'b276a7ed5524fc0c47d6080a6349a0a65b9e14bda8270cbaf87d6cd81bceb193'
        for item in index['source_files'][arm].values(): assert sha(item['path']) == item['sha256']
        for item in index['controls_references'].values(): assert sha(item['path']) == item['sha256']
        requests[arm] = path
        identities[arm] = {'request':str(path),'request_sha256':sha(path),'entry_sha256':sha(request['entry']),
            'adapter_sha256':sha(request['parameters']['adapter']),'source_files':index['source_files'][arm],
            'bundles':bundles,'controls':index['controls_references'],'devices':request['devices']}
    assert index['independent_reference'] is None
    assert all(sha(path) == h for path,h in expected.items())
    available = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    assert all(d in available for d in DEVICES)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free = int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free >= 8*1024**3+FORECAST
    return probe,plan,requests,expected,identities,audit


def baseline_gate(worker, evidence):
    result = worker.get('result',{})
    return (worker.get('status') == 'passed' and result.get('valid') is True
        and result.get('math_policy_unchanged') is True and len(result.get('checks',[])) == 2
        and len(result.get('samples',[])) == 5
        and evidence.get('outcome',{}).get('status') == 'returned'
        and evidence.get('stages_reached',{}).get('original_invoked') is True
        and evidence.get('stages_reached',{}).get('original_returned') is True
        and len(evidence.get('patch_restore',[])) == 6
        and all(row.get('status') == 'restored' for row in evidence['patch_restore'])
        and evidence.get('observer_errors') == [])


def main(args):
    os.umask(0o077)
    probe,plan,requests,expected,identities,audit = prepare(args.parent)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'frozen_files':len(expected),
            'probe_sha256':PROBE,'plan_sha256':PLAN,'sequence':['baseline-observed','candidate-observed'],
            'growth_forecast_bytes':FORECAST,'warning_free_bytes':8*1024**3,
            'original_step_anchor':133.379621,'constraint_limit':140.04860205,'independent_reference':None})); return
    args.output.mkdir(parents=True,exist_ok=True); assert not (args.output/'protocol.json').exists()
    write_json(args.output/'request-identities.json',identities)
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'probe':str(probe),'probe_sha256':PROBE,
        'plan':str(plan),'plan_sha256':PLAN,'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'source_requests':{k:str(v) for k,v in requests.items()},'frozen_identities':expected,
        'independent_reference':None,'sequence':['baseline-observed','candidate-observed'],
        'scope':'Exact Agent probe, each original evaluator once per leased rank. Original sources/requests/bundles/controls/math/precision/quality/all samples/metrics retained. Only entry/function/output/wait/diagnostic_context wiring changes. No Agent/search/parameter/native/observer-off repeat/source rerun/reference copies/pointer publication or acceptance.',
        'baseline_gate':'Both original workers passed/valid, math unchanged, two checks/five samples, original returned, six patches restored and no observer error. Timing and join completeness are not acceptance gates. Otherwise C not_run.',
        'limits':audit['code_limits'],'original_step_anchor':133.379621,'constraint_limit':140.04860205,
        'timing_scope':'Observed/perturbed diagnostic only; retain all values, no score/observer-cost subtraction or reanchor.',
        'growth_forecast_bytes':FORECAST,'disk':disk_observation(Path('/root/autodl-tmp'))})
    results = {}; baseline_ok = False
    for arm,name in [('parent','baseline-observed'),('candidate','candidate-observed')]:
        if arm == 'candidate' and not baseline_ok:
            results[name] = {'status':'not_run','reason':'Baseline failed the declared original-quality/observer-return/restoration gate'}; break
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        original = ExecutionRequest.model_validate(read(requests[arm]))
        updated = original.model_copy(update={'entry':probe,'function':'run','output':args.output/name,'wait_for_devices':True,
            'parameters':{**original.parameters,'diagnostic_context':{'original_entry':str(original.entry),
                'original_function':original.function,'independent_reference':None,'arm':name}}})
        try:
            run = LocalExecutor().run(updated); result_path = args.output/(name+'-result.json'); write_json(result_path,run)
            rows = []; gates = []
            for rank in range(2):
                worker_path = args.output/name/f'rank-{rank}.json'; evidence_path = args.output/name/f'diagnostic-rank-{rank}.json'
                worker = read(worker_path) if worker_path.exists() else {}
                evidence = read(evidence_path) if evidence_path.exists() else {}
                rows.append({'rank':rank,'worker':str(worker_path),'worker_sha256':sha(worker_path) if worker_path.exists() else None,
                    'worker_status':worker.get('status'),'valid':worker.get('result',{}).get('valid'),
                    'evidence':str(evidence_path),'evidence_sha256':sha(evidence_path) if evidence_path.exists() else None,
                    'outcome':evidence.get('outcome'),'observer_errors':len(evidence.get('observer_errors',[]))})
                gates.append(baseline_gate(worker,evidence))
            results[name] = {'status':'attempt_completed' if run.get('worker_pids_reaped') else 'execution_error',
                'result':str(result_path),'result_sha256':sha(result_path),'ranks':rows,
                'interpretation':'Attempt completed is not correct observation/full quality/optimization acceptance.'}
            if arm == 'parent':
                baseline_ok = all(gates); results[name]['baseline_gate_passed'] = baseline_ok
        except Exception:
            results[name] = {'status':'execution_error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'at':now(),'endpoint':name,**results[name]}),flush=True)
    same = all(sha(p) == h for p,h in expected.items())
    results['preservation'] = {'status':'completed' if same else 'error','files':len(expected),'unchanged':same}
    terminal = {'phase':'completed' if same and all(v['status'] in ['attempt_completed','completed'] for v in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results); write_json(args.output/'state.json',terminal)
    write_json(args.output/'followup-completed.json',terminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ['parent','output']: parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true'); main(parser.parse_args())
