"""Same-workload Qwen whole-generation goal with an explicit TTFT constraint."""
import argparse
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_goal_contrast_20261028 import ADAPTER


def now():return datetime.now(timezone.utc).isoformat()


def goal_request(parent):
    old=read(parent/'request.json');request=copy.deepcopy(old)
    request['goal']=('Minimize end_to_end_ms for the same complete interactive Qwen3-4B generation '
        '(batch 1, prompt 256 tokens, exactly 32 new greedy tokens), while keeping ttft_ms <= 1.05 times '
        'this goal variant\'s newly measured original baseline TTFT. Keep the complete source workload, '
        'BF16/SDPA/noTF32 policies, full logits tolerance, exact tokens, state lifetimes and both completion boundaries. '
        'This is an explicit new goal and constraint; the source workload goals remain provenance, not limits on optimization scope.')
    request['objective']={'name':'end_to_end_ms','unit':'ms','direction':'minimize',
        'constraints':[{'metric':'ttft_ms','op':'le','baseline_ratio':1.05}]}
    request['devices']=['0']
    assert 'this request optimizes only ttft_ms' in request['notes'][1]
    request['notes'][1]=("workload is the verbatim interactive row from workloads.json; its source-declared TTFT/decode goals remain provenance. "
        "This explicit variant minimizes complete end_to_end_ms and adds ttft_ms <= 1.05x the newly measured original baseline. "
        "No previous optimized version or historical timing is the constraint anchor.")
    request['notes'].append('End-to-end objective uses source infer() end_to_end_ms from prefill start through the final synchronization after all 32 new tokens. '
        'The TTFT constraint uses the original first-token synchronization boundary, median of five clean samples, against this variant\'s original controls anchor. '
        'Decode throughput and allocation metrics remain diagnostics; quality is unchanged.')
    assert {k for k in old if old[k]!=request[k]}=={'goal','objective','devices','notes'}
    ProjectRequest.model_validate(request)
    return request


def prepare(args,config):
    request=goal_request(args.parent);path=args.output/'end-to-end-request.json';write_json(path,request)
    parent=ProjectSearch(config,args.parent)
    baseline=parent.revisions.revision(parent.prepared['revision'])
    assert parent.prepared['adapter_sha256']==ADAPTER and not baseline.bundles
    argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
        '--request',str(path),'--reuse-evaluator',str(args.parent),'--output',str(args.output/'qwen-end-to-end'),'--prepare-only']
    write_json(args.output/'variant-protocol.json',{'at':now(),'argv':argv,'parent':str(args.parent),
        'reused_baseline':baseline.id,'reused_files':{f.path:f.sha256 for f in baseline.files},
        'adapter_sha256':ADAPTER,'request':request,
        'scope':'Explicit same-adapter new objective/constraint, not fresh original-entry evaluator. Only original baseline bytes reused. Real contract adaptation and fresh controls; no old bundles, replay outputs, contrast results or developer optimization hints supplied.'})
    with (args.output/'prepare-controller.log').open('wb')as log:
        result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode!=0:return {'status':'error','returncode':result.returncode}
    search=ProjectSearch(config,args.output/'qwen-end-to-end');current=search.revisions.current()
    assert search.prepared['adapter_sha256']==ADAPTER and not current.bundles
    assert {f.path:f.sha256 for f in current.files}=={f.path:f.sha256 for f in baseline.files}
    return {'status':'completed','revision':current.id,'adapter_sha256':ADAPTER,'contract_adaptation':search.prepared['reuse']['contract_adaptation']}


def source_check(args,config):
    search=ProjectSearch(config,args.output/'qwen-end-to-end');p=search.prepared
    source=Path(p['controls_path']).parent/'reference-source'
    reference=read(args.parent.parent/'protocol.json')['specifications'][0]['reference']
    assert sha(source/'opop_adapter.py')==ADAPTER
    run=LocalExecutor().run(ExecutionRequest(entry=Path(__file__).with_name('v6_autonomous_cohort_20261027.py'),
        function='source_reference',cwd=source,output=args.output/'qwen-source-audit',devices=['0'],
        lease_dir=Path(search.request.lease_dir),worker_timeout_s=search.request.worker_timeout_s,wait_for_devices=True,
        parameters={'case':'qwen','reference':reference,'controls':p['controls_path'],'request':search.request.portable()}))
    write_json(args.output/'source-check.json',run)
    return {'status':'completed' if valid_run(run) else 'source_fidelity_failure','path':str(args.output/'source-check.json'),
        'adapter_sha256':ADAPTER,'source_reference_sha256':sha(reference),
        'controls_reference_sha256':sha(Path(p['controls_path'])/'reference-rank-0.pt')}


def search(args,config):
    run=args.output/'qwen-end-to-end'
    argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
        '--request',str(run/'request.json'),'--output',str(run),'--resume','--rounds','3']
    write_json(args.output/'search-protocol.json',{'at':now(),'argv':argv,
        'scope':'Three autonomous rounds after real contract preparation and independent full source check. No manual groups/candidates or developer optimization feedback. Source check is an external gate, not input to the agents. API use observed only.'})
    with (args.output/'search-controller.log').open('wb')as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    current=ProjectSearch(config,run).revisions.current()
    return {'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode,
        'current':current.id,'bundles':list(current.bundles)}


def independent(args,config):
    run=args.output/'qwen-end-to-end';search=ProjectSearch(config,run)
    if not search.revisions.current().bundles:return {'status':'not_run','reason':'No autonomous accepted current bundle set.'}
    target=args.output/'qwen-end-to-end-independent';fork_run(run,target,revision_id=search.prepared['revision'])
    result=promote_version(ProjectSearch(config,target),run,publish=False)
    return {'status':'completed',**{k:result[k]for k in ['revision','parent','quality','coverage_errors','measurement','decision','publication']}}


def native(args,config):
    search=ProjectSearch(config,args.output/'qwen-end-to-end');rows=[]
    if not search.revisions.current().bundles:return {'status':'not_run','reason':'No autonomous accepted bundle set.'}
    for name,revision in [('baseline',search.revisions.revision(search.prepared['revision'])),('candidate',search.revisions.current())]:
        result=Nsys().collect(search,revision);path=args.output/('native-'+name+'.json');write_json(path,result)
        rows.append({'arm':name,'status':result['status'],'path':str(path),'sha256':sha(path)})
    return {'status':'completed','runs':rows,'scope':'Separate perturbed native paths, not clean timing or formal acceptance.'}


def main(args):
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'protocol.json').exists()
    config=load_config(args.config,[]);config.usage.mode='observe_only'
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent':str(args.parent),
        'sequence':['real variant contract and fresh controls','independent full source check','three autonomous rounds','conditional full-version AB/BA','conditional separate native profiles'],
        'disk':disk_observation(Path('/root/autodl-tmp')),'scope':'Same workload and frozen adapter, explicit end_to_end_ms goal with TTFT<=1.05 baseline. No old optimized bundle imports. No production change, API caps, raw local transfer, Goal or model polling.'})
    results={}
    for name,operation in [('prepare',prepare),('source',source_check),('search',search),('independent',independent),('native',native)]:
        write_json(args.output/'state.json',{'phase':'running','stage':name,'at':now()})
        if any(results.get(k,{}).get('status') not in [None,'completed']for k in ['prepare','source','search']):
            results[name]={'status':'not_run','reason':'Earlier required stage failed; preserve failure, no automatic retry.'}
        elif name=='native' and results.get('independent',{}).get('quality')!='pass':
            results[name]={'status':'not_run','reason':'No complete independent quality-passing version.'}
        else:
            try:results[name]=operation(args,config)
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'stage':name,'status':results[name]['status'],'at':now()}),flush=True)
    terminal={'phase':'completed' if all(r['status']in['completed','not_run']for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
