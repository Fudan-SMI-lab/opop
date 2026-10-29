"""Independent full-version validation of the frozen, fact-assisted Qwen trial."""
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
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_qwen_original_followup_20261029 import ADAPTER


def now():
    return datetime.now(timezone.utc).isoformat()


def identities(args, config):
    prior=read(args.parent/'terminal-audit.json')
    search=ProjectSearch(config,args.parent/'qwen-continuation')
    current=search.revisions.current()
    assert prior['trial']['decision']['accepted']
    assert current.id==prior['pointers']['current']=='a3766b5026d1e6eb137962285bd5300e2e3f579ea09069f5ad18da43cb4771de'
    assert list(current.bundles)==prior['pointers']['bundles'] and len(current.bundles)==1
    assert framework_source()==prior['core']
    assert all(sha(p)==h for p,h in prior['frozen_identities_verified'].items())
    assert sha(prior['trial']['candidate_path'])==prior['trial']['candidate_sha256']
    native=prior['native']
    assert sha(native['sqlite'])==native['sqlite_sha256']
    assert sha(native['normalized_path'])==native['normalized_sha256']
    assert search.prepared['adapter_sha256']==ADAPTER
    return search,prior


def source(args, config):
    search,prior=identities(args,config)
    controls=Path(search.prepared['controls_path']);source=controls.parent/'reference-source'
    original=read(args.parent.parent/'qwen-end-to-end-original-20261029/protocol.json')
    reference=original['independent_source_reference'];assert sha(reference['path'])==reference['sha256']
    run=LocalExecutor().run(ExecutionRequest(
        entry=Path(__file__).with_name('v6_qwen_original_followup_20261029.py'),function='reviewed_source',
        cwd=source,output=args.output/'source-check-run',devices=['cpu'],
        lease_dir=Path(search.request.lease_dir),worker_timeout_s=search.request.worker_timeout_s,
        parameters={'adapter':str(source/'opop_adapter.py'),'reference':reference['path'],
            'controls_reference':str(controls/'reference-rank-0.pt'),'quality':search.request.quality,
            'continuous_steps':search.request.continuous_steps}))
    path=args.output/'source-check.json';write_json(path,run)
    return {'status':'completed'if valid_run(run)else'source_check_failed','path':str(path),'sha256':sha(path),
        'scope':'Reviewed external mapping, CPU comparison of unchanged real CUDA controls to full independent tokens/logits. No new controls or GPU reference copy, not certification of arbitrary owners.'}


def independent(args, config):
    search,prior=identities(args,config);target=args.output/'qwen-independent'
    revision=fork_run(search.output,target,revision_id=search.prepared['revision'])
    check=ProjectSearch(config,target)
    assert not revision.bundles and check.prepared==search.prepared and check.request==search.request
    write_json(args.output/'preparation-reuse.json',{'source':str(search.output),'source_revision':search.revisions.current().id,
        'baseline':revision.id,'contract_ref':revision.contract_ref,'shared_controls':search.prepared['controls_path'],
        'request_sha256':sha(target/'request.json'),'adapter_sha256':search.prepared['adapter_sha256'],
        'publish':False,'sequence':'AB then BA, fresh workers, original5 samples/2 continuous full-quality checks per endpoint'})
    result=promote_version(check,search.output,revision_id=search.revisions.current().id,publish=False)
    assert check.revisions.current().id==revision.id
    identities(args,config)
    return {'status':'completed',**{k:result[k]for k in ['trial','revision','parent','quality','coverage_errors','measurement','decision','publication']}}


def native(args, config):
    search,prior=identities(args,config)
    # The real planner already collected the exact accepted candidate. Reuse
    # its hashed raw SQLite rather than duplicating that perturbed invocation.
    target=ProjectSearch(config,args.output/'qwen-independent')
    result=Nsys().collect(target,target.revisions.revision(target.prepared['revision']))
    path=args.output/'native-baseline.json';write_json(path,result)
    return {'status':'completed'if result['status']=='observed'else'collection_error','baseline':{'path':str(path),'sha256':sha(path)},
        'candidate_reused':prior['native'],
        'scope':'Separate baseline profile and existing exact-candidate native capture. Distinct periods; neither trace contributes acceptance timing or isolates critical-path cost.'}


def main(args):
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    assert not(args.output/'protocol.json').exists()
    config=load_config(args.config,[]);config.usage.mode='observe_only'
    search,prior=identities(args,config)
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent':str(args.parent),
        'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),'source_revision':search.revisions.current().id,
        'original_baseline':search.prepared['revision'],'original_ttft_limit_ms':25.2536046,
        'candidate_sha256':prior['trial']['candidate_sha256'],'sequence':['CPU full-source identity/negative checks','independent full-version AB/BA','conditional baseline native; reuse current-candidate native'],
        'scope':'Fact-assisted fresh-adapter continuation, not unaided original-entry success or new agent/search. Unmodified candidate and production rules; no repeats until pass, no new references, no bulk transfers. Independent decision never replaces historical search decision.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    for name,op in [('source',source),('independent',independent),('native',native)]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        if name!='source'and results['source']['status']!='completed':
            results[name]={'status':'not_run','reason':'Source gate failed; preserve error, no retry.'}
        elif name=='native'and results.get('independent',{}).get('quality')!='pass':
            results[name]={'status':'not_run','reason':'Independent full-quality version unavailable.'}
        else:
            try:results[name]=op(args,config)
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'stage':name,'status':results[name]['status'],'at':now()}),flush=True)
    try:
        final,unused=identities(args,config)
        results['preservation']={'status':'completed','source_current':final.revisions.current().id}
    except Exception:results['preservation']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed'if all(r['status']in ['completed','not_run']for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
