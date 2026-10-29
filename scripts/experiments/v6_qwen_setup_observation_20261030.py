"""One exact repaired-probe B/C observation, no search or acceptance."""
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


def now():return datetime.now(timezone.utc).isoformat()


def prepare(parent):
    audit=read(parent/'terminal-audit.json');assert audit['review']['execute_exact_bytes_once']
    assert audit['core']==framework_source() and audit['independent_cpu']['status']=='passed'
    result=read(parent/'followup-completed.json')['results']['diagnostic-author']
    index=read(Path(result['agent']['sandbox'])/'inputs/evidence-index.json')
    probe=Path(result['probe']);plan=Path(result['plan'])
    assert sha(probe)=='bc4751ca9b77f9edaf5269c10dd053298e287d50452f51d113f1555a33265ff5'
    assert sha(plan)=='33fdb9239bc1420293fd89fe74a139a081276e2c955aca411db4b9c1bdef63a6'
    expected=read(parent/'protocol.json')['frozen_identities'].copy()
    for p in [probe,plan,parent/'terminal-audit.json',parent/'followup-completed.json',parent/'audit/independent-cpu/result.json']:
        expected[str(p)]=sha(p)
    requests={k:Path(v['path'])for k,v in index['original_execution_requests'].items()}
    identities={}
    for arm,path in requests.items():
        assert sha(path)==index['original_execution_requests'][arm]['sha256']
        request=read(path);task=request['parameters']['request']
        assert task['warmup']==2 and task['repeats']==5 and task['continuous_steps']==2
        assert task['quality']=={'atol':.02,'rtol':.02} and request['parameters']['action']=='evaluate'
        assert request['devices']==['GPU-20e8003c-f531-d50c-084c-a9a97fe94004']
        assert task['workload']['batch']==1 and task['workload']['prompt_tokens']==256 and task['workload']['new_tokens']==32
        bundles=request['parameters']['bundles']
        if arm=='baseline':assert bundles==[]
        else:
            assert len(bundles)==1 and bundles[0]['parameters']=={}
            assert sha(bundles[0]['file'])=='04dfc6c2c9165560ae7412bbdbd00be6e79044e7f52276d096dd48728c6583f8'
        for item in index['source_files'][arm].values():assert sha(item['path'])==item['sha256']
        identities[arm]={'request':str(path),'request_sha256':sha(path),'entry_sha256':sha(request['entry']),
            'adapter_sha256':sha(request['parameters']['adapter']),'bundles':bundles,
            'reference_dir':request['parameters']['reference_dir'],'source':index['source_files'][arm]}
    assert index['independent_reference'] is None and all(sha(p)==h for p,h in expected.items())
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    assert 'GPU-20e8003c-f531-d50c-084c-a9a97fe94004' in devices
    assert int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])>8*1024**3
    return probe,plan,requests,expected,identities,audit


def main(args):
    os.umask(0o077);probe,plan,requests,expected,identities,audit=prepare(args.parent)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'frozen_files':len(expected),
            'probe_sha256':sha(probe),'plan_sha256':sha(plan),'sequence':['baseline-observed','candidate-observed'],
            'current_device_ttft_anchor':23.285755,'constraint_limit':24.45004275,'independent_reference':None}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'request-identities.json',identities)
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'probe':str(probe),'probe_sha256':sha(probe),
        'plan':str(plan),'plan_sha256':sha(plan),'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),
        'source_requests':{k:str(v)for k,v in requests.items()},'frozen_identities':expected,'independent_reference':None,
        'sequence':['baseline-observed','candidate-observed'],'limits':audit['review']['limitations'],
        'scope':'Each original evaluator once through exact generated observer. Original failure and all work/samples/quality/bindings/math retained. No manual code fix, Agent/search/acceptance/Nsight/reference copies/pointer publication. CPU summaries and timings diagnostic, no native or unique-cause proof.',
        'baseline_gate':'Original worker passed/valid and empty install windows; no TTFT acceptance gate or reanchor.',
        'current_device_ttft_anchor':23.285755,'constraint_limit':24.45004275,'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={};baseline_ok=False
    for arm in ['baseline','candidate']:
        name=arm+'-observed'
        if arm=='candidate' and not baseline_ok:
            results[name]={'status':'not_run','reason':'baseline observation did not pass the declared gate'};break
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        original=ExecutionRequest.model_validate(read(requests[arm]))
        updated=original.model_copy(update={'entry':probe,'function':'run','output':args.output/name,'wait_for_devices':True,
            'parameters':{**original.parameters,'diagnostic_context':{'original_entry':str(original.entry),
                'original_function':original.function,'independent_reference':None,'arm':name}}})
        try:
            run=LocalExecutor().run(updated);path=args.output/(name+'-result.json');write_json(path,run)
            worker_path=args.output/name/'rank-0.json';worker=read(worker_path)
            evidence=args.output/name/'diagnostic-evidence-rank-0.json'
            results[name]={'status':'attempt_completed'if run.get('worker_pids_reaped')and evidence.exists()else'execution_error',
                'worker_status':worker.get('status'),'valid':worker.get('result',{}).get('valid'),
                'result':str(path),'result_sha256':sha(path),'worker':str(worker_path),
                'evidence':str(evidence)if evidence.exists()else None,'evidence_sha256':sha(evidence)if evidence.exists()else None,
                'interpretation':'Attempt completion is not diagnostic correctness, fast-path quality or optimization acceptance.'}
            if arm=='baseline':
                d=read(evidence)if evidence.exists()else{}
                baseline_ok=(worker.get('status')=='passed' and worker.get('result',{}).get('valid')is True
                    and worker['result'].get('math_policy_unchanged')is True and len(worker['result'].get('checks',[]))==2
                    and d.get('install_windows')==[] and d.get('original_outcome',{}).get('original_returned')is True)
                results[name]['baseline_gate_passed']=baseline_ok
        except Exception:results[name]={'status':'execution_error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results);print(json.dumps({'at':now(),'endpoint':name,**results[name]}),flush=True)
    same=all(sha(p)==h for p,h in expected.items())
    results['preservation']={'status':'completed'if same else'error','files':len(expected),'unchanged':same}
    terminal={'phase':'completed'if same and all(v['status']in ['attempt_completed','completed']for v in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
