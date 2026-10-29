"""One derived baseline and exact failed FSDP candidate with the repaired probe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_fsdp_diagnostic_author_20261029 import build_inputs


def now():return datetime.now(timezone.utc).isoformat()


def main(args):
    os.umask(0o077)
    audit=read(args.parent/'terminal-audit.json')
    assert audit['review']['execute_exact_bytes_once'] and audit['core']==framework_source()
    _,original,index=build_inputs(args.parent.parent)
    expected={**original,**audit['frozen_identities']}
    probe=Path(audit['generated']['probe'])
    for key in ['probe','plan']:
        p=audit['generated'][key];h=audit['generated'][key+'_sha256'];assert sha(p)==h;expected[p]=h
    source=Path(index['original_execution_request']['path']);request=ExecutionRequest.model_validate(read(source))
    expected[str(request.entry)]=sha(request.entry)
    project=source.parents[3]/'project'
    for name in ['current.json']:
        p=project/name;assert p.is_file();expected[str(p)]=sha(p)
    task=request.parameters['request']
    assert task['continuous_steps']==2 and task['warmup']==2 and task['repeats']==5
    assert len(request.parameters['bundles'])==1 and request.parameters['bundles'][0]['parameters']=={'keep_unsharded_blocks':6}
    assert request.parameters['action']=='evaluate' and len(task['diagnostic_metrics'])==4
    assert all(sha(p)==h for p,h in expected.items())
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'probe_sha256':sha(probe),
            'frozen_files':len(expected),'sequence':['derived-baseline-observed','candidate-observed'],
            'original_request':str(source),'diagnostic_metrics':task['diagnostic_metrics'],
            'disk':disk_observation(Path('/root/autodl-tmp'))}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent':str(args.parent),
        'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),'generated':audit['generated'],
        'original_request':str(source),'original_request_sha256':sha(source),'frozen_identities':expected,
        'sequence':['derived-baseline-observed','candidate-observed'],'independent_reference':None,
        'baseline_definition':'Derived from exact original candidate request by parameters.bundles=[] only. Materialized source/manifest remains original staged revision; no historical baseline measurement claimed.',
        'changes':'Both arms replace entry/function/output/wait and add diagnostic_context. Only baseline clears bundles. No other request/precision/quality/order/warmup/repeats/metrics changes.',
        'limitations':audit['review']['limitations'],
        'scope':'Exact real-agent repaired diagnostic, each evaluator once in two leased workers. No new agent/search/acceptance/Nsight/control/reference copies/pointer publish. Original rejection preserved; all observed timings diagnostic.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    for arm in ['derived-baseline-observed','candidate-observed']:
        write_json(args.output/'state.json',{'phase':'running','stage':arm,'at':now()})
        parameters={**request.parameters,'diagnostic_context':{'original_entry':str(request.entry),
            'original_function':request.function,'independent_reference':None,'arm':arm}}
        if arm=='derived-baseline-observed':parameters['bundles']=[]
        updated=request.model_copy(update={'entry':probe,'function':'run','output':args.output/arm,
            'wait_for_devices':True,'parameters':parameters})
        try:
            run=LocalExecutor().run(updated);result_path=args.output/(arm+'-result.json');write_json(result_path,run)
            sides=[args.output/arm/f'diagnostic-rank-{rank}.json'for rank in [0,1]]
            results[arm]={'status':'attempt_completed'if run.get('worker_pids_reaped')and all(p.is_file()for p in sides)else'execution_error',
                'worker_status':run['status'],'result':str(result_path),'result_sha256':sha(result_path),
                'diagnostics':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in sides if p.is_file()],
                'interpretation':'Attempt completion is not quality, source fidelity or performance acceptance; inspect each rank original returned result and diagnostic validity.'}
        except Exception:results[arm]={'status':'execution_error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'at':now(),'endpoint':arm,**results[arm]}),flush=True)
    same=all(sha(p)==h for p,h in expected.items())
    terminal={'phase':'completed'if same and all(r['status']=='attempt_completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'all_frozen_hashes_unchanged':same,
        'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
