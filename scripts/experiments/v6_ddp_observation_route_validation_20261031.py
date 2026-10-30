"""One original paired validation of exact generated DDP bytes, publish=False."""
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
from scripts.experiments.v6_audit_ddp_observation_route_candidate_20261031 import CODE, PROPOSAL


def now():return datetime.now(timezone.utc).isoformat()


def prepare(args,config):
    p=read(args.parent/'protocol.json');audit=read(args.parent/'terminal-audit.json')
    t=read(args.parent/'followup-completed.json')
    assert t['phase']=='completed' and audit['all_frozen_unchanged'] and p['core']==framework_source()
    assert t['results']['B']['full_gate'] and t['results']['C']['full_gate']
    source=args.parent/'ddp-fixed';s=ProjectSearch(config,source);parent=s.revisions.current()
    assert parent.id==p['parent']==audit['source_current']
    ids=read(args.parent/'staged-identities.json');c=s.revisions.revision(ids['candidate']);b=s.revisions.bundle('fixed-observation-route')
    assert c.id==audit['candidate'] and b.parameters=={} and b.requires==() and b.replaces==parent.bundles
    assert c.bundles==(b.id,) and c.contract_ref==parent.contract_ref
    assert next(f.sha256 for f in c.files if f.path==b.installer)==CODE==p['candidate_sha256']
    assert p['proposal_sha256']==PROPOSAL
    assert len(audit['traces'])==12
    assert all(len(x['payloads'])==(13 if x['arm']=='B' else 1) and x['logical_bytes']==497759232 for x in audit['traces'])
    assert s.request.continuous_steps==2 and s.request.warmup==2 and s.request.repeats==5
    assert s.request.profiling=={'agent_plan':True,'feedback_agent_plan':True}
    frozen=p['frozen_identities'].copy()
    paths=[args.parent/n for n in ['protocol.json','followup-completed.json','followup-results.json','state.json',
        'B-result.json','C-result.json','terminal-audit.json','public-terminal-audit.json','staged-identities.json','preparation-reuse.json','launch.json']]
    paths += [source/n for n in ['request.json','prepared.json','discovery.json','project/current.json']]
    paths += list((source/'jobs').glob('*/run/*.json'))+list((source/'jobs').glob('*/run/*.log'))
    paths += [Path(x['path']) for x in audit['traces']]
    paths += list((source/'project/revisions').glob(c.id+'.json'))+list((source/'project/bundles').glob(b.id+'.json'))
    for path in paths:frozen[str(path)]=sha(path)
    assert all(sha(path)==h for path,h in frozen.items())
    return p,audit,source,s,parent,c,b,frozen


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    p,audit,source,original,parent,candidate,bundle,frozen=prepare(args,config)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=9*1024**3,free
    defs={'source':str(source),'parent':parent.id,'candidate':candidate.id,'bundle':bundle.id,
          'candidate_sha256':CODE,'proposal_sha256':PROPOSAL,'parameters':{},'core':framework_source(),
          'devices':original.request.devices,'prepared':original.prepared['revision'],'shared_controls':original.prepared['controls_path'],
          'frozen_files':len(frozen),'free_bytes':free,'growth_forecast_bytes':1024**3,'warning_free_bytes':8*1024**3,
          'original_step_ms':133.379621,'step_limit_ms':140.04860205,'fixed_audit_sha256':sha(args.parent/'terminal-audit.json')}
    if args.preflight:print(json.dumps({'status':'passed',**defs}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'protocol.json').exists()
    scope=('ONE existing ProjectSearch.validate_bundle(publish=False) exact508a against current acceptedccd90. '
           'Original AB then BA in fresh workers, fixed original anchor/quality/math/2warmup/5clean/postclean metrics. '
           'Normal resource_feedback profile and existing real collection-planner chooses existing collectors; controller does not choose them. '
           'No selector/candidate/parameter tuning/new implementation/source-control rerun/reference copy/independent stage/pointer publication. '
           'Fixed B/C samples are not pooled; no retry after rejection. Audit actual full role/tool/native history and preserve all failures. '
           'Only genuinely new formal acceptance can justify later separately declared independent complete-version closure.')
    write_json(args.output/'protocol.json',{'at':now(),**defs,'frozen_identities':frozen,'scope':scope})
    results={}
    try:
        branch=args.output/'ddp-validation';forked=fork_run(source,branch);search=ProjectSearch(config,branch)
        assert forked.id==parent.id and search.prepared==original.prepared and search.request==original.request
        assert (branch/'discovery.json').read_bytes()==(source/'discovery.json').read_bytes()
        staged=search.revisions.stage(bundle)
        assert staged==candidate
        write_json(args.output/'preparation-reuse.json',{'source':str(source),'parent':parent.id,'candidate':staged.id,
            'same_exact_bundle_files_contract':True,'shared_controls':search.prepared['controls_path'],
            'request_prepared_discovery_equal':True,'no_initial_discovery_source_controls_reference_copy':True})
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'original_paired_validation'})
        result=search.validate_bundle(bundle,{'trial':'trial-observation-route-validation','scope':scope},
            discovery=search.current_discovery(),candidate=staged,publish=False)
        result_path=branch/'trials/trial-observation-route-validation/result.json'
        results['validation']={'status':'completed','result':str(result_path),'result_sha256':sha(result_path),
            'decision':result['decision'],'publication':result['publication'],'measurement':result['measurement']}
        assert search.revisions.current().id==original.revisions.current().id==parent.id
        results['pointers']={'status':'passed','source_and_newbranch_current':parent.id}
    except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(path)==h for path,h in frozen.items())
    results['preservation']={'status':'passed' if same else 'failed','files':len(frozen),'unchanged':same}
    terminal={'phase':'completed' if same and 'validation' in results and 'controller' not in results else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp')),'scope':scope}
    write_json(args.output/'followup-results.json',results)
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)
    print(json.dumps({'at':now(),'phase':terminal['phase'],'results':results}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');main(parser.parse_args())
