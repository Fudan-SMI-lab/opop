"""Three autonomous rounds from a reviewed source-faithful FSDP evaluator."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.collectors import Nsys
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def autonomous(args, config):
    check=read(args.parent.parent/'source-check.json');prepared=read(args.parent/'prepared.json')
    assert check['status']=='passed' and all(r['same_allocated_and_peak'] for r in check['memory'])
    assert prepared['adapter_sha256']==check['adapter_sha256']=='8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6'
    command=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
        '--request',str(args.parent/'request.json'),'--branch-from',str(args.parent),'--output',str(args.output/'fsdp-search'),'--rounds','3']
    write_json(args.output/'search-protocol.json',{'at':now(),'argv':command,'parent':str(args.parent),
        'parent_request_sha256':sha(args.parent/'request.json'),'adapter_sha256':prepared['adapter_sha256'],
        'source_check_sha256':sha(args.parent.parent/'source-check.json'),
        'scope':'Autonomous discovery/selection/candidate/feedback over three rounds from the reviewed corrected baseline. No old optimized bundle, manual group, candidate or developer optimization diagnosis. Evaluator correction was developer-assisted; this is not a fresh unaided original-entry cohort. Shared read-only source controls avoid duplicate reference tensors.'})
    with (args.output/'search-controller.log').open('wb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    row={'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode}
    if (args.output/'fsdp-search/prepared.json').exists():
        search=ProjectSearch(config,args.output/'fsdp-search');row.update(current=search.revisions.current().id,bundles=list(search.revisions.current().bundles),
            trials=[{'path':str(p),'sha256':sha(p),'decision':read(p)['decision']} for p in sorted((search.output/'trials').glob('*/result.json'))])
    return row


def independent(args, config):
    search=ProjectSearch(config,args.output/'fsdp-search')
    if not search.revisions.current().bundles:return {'status':'not_run','reason':'No accepted candidate version'}
    target=args.output/'fsdp-independent';fork_run(args.parent,target)
    result=promote_version(ProjectSearch(config,target),search.output,publish=False)
    return {'status':'completed',**{k:result[k] for k in ['revision','parent','quality','coverage_errors','measurement','decision','publication']}}


def native(args, config):
    search=ProjectSearch(config,args.output/'fsdp-search')
    current=search.revisions.current()
    if not current.bundles:return {'status':'not_run','reason':'No accepted candidate version'}
    rows=[]
    for label,revision in [('baseline',search.revisions.revision(search.prepared['revision'])),('candidate',current)]:
        result=Nsys().collect(search,revision);path=args.output/f'native-{label}.json';write_json(path,result)
        rows.append({'arm':label,'revision':revision.id,'status':result['status'],'path':str(path),'sha256':sha(path)})
    return {'status':'completed' if all(r['status']=='observed' for r in rows) else 'completed_with_unavailable',
        'runs':rows,'scope':'Separate fixed native profiles, not clean timing or acceptance. Includes original and current full-version paths.'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True);config=load_config(args.config,[])
    write_json(args.output/'protocol.json',{'at':now(),'framework_source':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp')),
        'sequence':['three autonomous FSDP rounds','conditional independent full-version AB/BA','conditional fixed native B/C profiles'],
        'scope':'Frozen repaired adapter/source/request/quality/memory constraint. No API caps, no score-based retries, all samples retained. '
                'Explicit developer-assisted baseline preparation followed by autonomous search. Raw stays server-side; no Goal, polling model calls, delivery probes or bulk local transfer.'})
    results={}
    for name,op in [('search',lambda:autonomous(args,config)),('independent',lambda:independent(args,config)),('native',lambda:native(args,config))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        try:results[name]=op()
        except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(name,results[name]['status'],now(),flush=True)
    terminal={'phase':'completed' if all(x['status'] in ['completed','not_run'] for x in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':main()
