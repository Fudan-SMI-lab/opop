"""Fresh current-device FSDP controls before the communication-priority trial."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.variants import prepare_variant
from kernel_optimizer.runtime.execution import resolve_devices,write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha,independent_source
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

ADAPTER='8b014f6ff6ca1305ee0399eeaa6136653fe3f347f1accbefd83c37bdff67fba6'
BASELINE='7231e858ad370240b7e86ac29c1745205ae8616501af53539423e1671003cca1'
OLD_CURRENT='3007940f06c3a77c32c40b9c49b53e2b7a93af0fd66ba33f1cac90ed37cdf05f'


def now():return datetime.now(timezone.utc).isoformat()


def preflight(args,config):
    original=ProjectSearch(config,args.parent);baseline=original.revisions.revision(original.prepared['revision'])
    assert baseline.id==BASELINE and not baseline.bundles and original.revisions.current().id==OLD_CURRENT
    assert original.prepared['adapter_sha256']==ADAPTER
    devices=resolve_devices(args.devices);assert len(devices)==2 and len(set(devices))==2
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id,driver_version','--format=csv,noheader'],text=True)
    assert all(d in inventory for d in devices) and all(d not in inventory for d in original.request.devices)
    request=original.request.model_copy(update={'devices':devices})
    before,after=original.request.portable(),request.portable()
    assert {k for k in before if before[k]!=after[k]}=={'devices'}
    assert request.continuous_steps==2 and request.warmup==2 and request.repeats==5 and len(request.diagnostic_metrics)==4
    assert request.objective['aggregation_order']=='ranks_then_samples' and request.objective['rank_aggregation']=='max'
    assert framework_source()['sha256']=='ed24af7acc8caa15578e0b859c85ef8a901be51354448ed39094606ed9e10a2f'
    old_check=args.parent.parent.parent/'fsdp-owner-repair-20261028/source-check.json'
    assert sha(old_check)=='6f83d6ea80c602561220fea6e4686e2bd3664de672f5fb506457295ff62c1ba8'
    source_check=read(old_check);assert source_check['status']=='passed' and source_check['adapter_sha256']==ADAPTER
    protected={str(p):sha(p)for p in [args.parent/'prepared.json',args.parent/'request.json',original.revisions.pointer,old_check]}
    for f in baseline.files:
        p=original.store.artifacts_dir/f.sha256;assert sha(p)==f.sha256;protected[str(p)]=f.sha256
    refs=[]
    for rank in range(2):
        old_control=Path(original.prepared['controls_path'])/f'reference-rank-{rank}.pt'
        refs.append(old_control)
        for step in range(2):refs.append(args.source/f'source-step-{step}-rank-{rank}.pt')
    for p in refs:protected[str(p)]=sha(p)
    for rank in source_check['ranks']:
        assert protected[rank['adapter_reference']]==rank['adapter_reference_sha256']
        for check in rank['checks']:
            assert protected[check['source']]==check['source_sha256'] and check['comparison']['passed']
    controls_size=sum((Path(original.prepared['controls_path'])/f'reference-rank-{r}.pt').stat().st_size for r in range(2))
    return original,request,protected,inventory,controls_size


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    original,request,frozen,inventory,controls_size=preflight(args,config)
    definitions={'at':now(),'core':framework_source(),'parent':str(args.parent),'source_current':OLD_CURRENT,
        'original_baseline':BASELINE,'adapter_sha256':ADAPTER,'old_devices':original.request.devices,'new_devices':request.devices,
        'request_changed_fields':['devices'],'inventory':inventory,'independent_source':str(args.source),
        'frozen_identities':frozen,'known_controls_bytes':controls_size,
        'space_budget':{'warning_floor_bytes':8*1024**3,'fresh_controls_bytes':controls_size,'additional_profiles_and_metadata_bytes':1024**3},
        'scope':'User communication-priority task preparation only. Exact original source/owner-repaired adapter and full workload; new placement only. Empty original prepared baseline, no historical K3 installed, no old timing anchor. Fresh real two-rank controls followed by full independent-source CPU comparison; references reused read-only without copies. No Agent/search/acceptance in this preparation batch. One ordinary real trial follows only after evidence review; no selected group/K/implementation supplied.'}
    if args.preflight:print(json.dumps({'status':'passed',**definitions}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>8*1024**3+controls_size+1024**3
    write_json(args.output/'protocol.json',{**definitions,'disk':disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output/'variant-request.json',request.portable());results={};branch=args.output/'fsdp-current'
    try:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'fresh_controls'})
        prepared=prepare_variant(original.output,request,branch,config=config)
        write_json(args.output/'preparation-reuse.json',{k:prepared[k]for k in ['status','revision','contract_ref','adapter_sha256','controls_path','reuse']})
        results['prepare']={'status':prepared['status'],'controls':prepared['controls_path']}
        if prepared['status']!='prepared':raise RuntimeError('Fresh baseline controls invalid; retain failure, do not repair or search.')
        s=ProjectSearch(config,branch);assert not s.revisions.current().bundles and s.revisions.current().id==prepared['revision']
        assert prepared['adapter_sha256']==ADAPTER and prepared['reuse']['contract_adaptation'] is None
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'full_source_comparison'})
        source=independent_source(branch,args.source)
        source.update(adapter_sha256=ADAPTER,scope='Full loss/logits/named per-rank DTensor shard gradients and updated parameters over two continuous updates, compared to existing independent source. Tied observation and +1logit negatives. Current controls are freshly executed, not copied old controls; old timing/memory samples are not current anchors. Prior developer-assisted 512B source owner and reviewed mapping remain explicit.')
        write_json(args.output/'source-check.json',source);assert source['status']=='passed'
        results['source']={'status':'passed','path':str(args.output/'source-check.json'),'sha256':sha(args.output/'source-check.json')}
        results['baseline']={'status':'prepared','current':s.revisions.current().id,'bundles':[],
            'constraints':constraints_for_runs([prepared['evaluation']],request.objective,prepared['evaluation']),
            'reference_hashes':{str(Path(prepared['controls_path'])/f'reference-rank-{r}.pt'):sha(Path(prepared['controls_path'])/f'reference-rank-{r}.pt')for r in range(2)}}
    except Exception:results['error']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items()) and original.revisions.current().id==OLD_CURRENT
    results['preservation']={'status':'passed'if same else'failed','frozen_files':len(frozen)}
    terminal={'phase':'completed'if 'error'not in results and same else'completed_with_errors','finished_at':now(),
        'results':results,'disk':disk_observation(Path('/root/autodl-tmp')),
        'next':'Audit original controls and all24 separate metric traces, then one ordinary real communication-focused trial from this new empty baseline; DDP next. Do not continue Qwen.'}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)
    print(json.dumps(terminal),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','parent','source']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--devices',nargs=2,required=True);p.add_argument('--preflight',action='store_true');main(p.parse_args())
