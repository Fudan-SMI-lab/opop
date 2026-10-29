"""One ordinary continuation from exact fixed quality/constraint evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts, route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

CODE='7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92'


def now(): return datetime.now(timezone.utc).isoformat()
def digest(raw): return hashlib.sha256(raw).hexdigest()


def prepare(args,config):
    protocol=read(args.parent/'protocol.json');done=read(args.parent/'followup-completed.json')
    audit=read(args.parent/'terminal-audit.json')
    assert done['phase']=='completed' and audit['core']==protocol['core']==framework_source()
    assert sha(args.parent/'terminal-audit.json')=='c16a1b85af24a69fb09a7ab437a7f33f73c20d8a89d462dcab52ddd62c5242e4'
    source=ProjectSearch(config,args.parent/'qwen-fixed');current=source.revisions.current()
    assert current.id=='2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1' and not current.bundles
    assert read(source.output/'discovery.json')['revision']==current.id
    assert resolve_devices(source.request.devices)==source.request.devices==['GPU-20e8003c-f531-d50c-084c-a9a97fe94004']
    assert source.prepared['evaluation']['ranks'][0]['result']['metrics']['ttft_ms']==23.285755
    stage=read(args.parent/'staged-identities.json');candidate=source.revisions.revision(stage['candidate'])
    assert len(candidate.bundles)==1;bundle=source.revisions.bundle(candidate.bundles[0]);assert bundle.parameters=={}
    role_root=args.parent.parent/'qwen-cache-interface-candidate-20261030'
    role=read(role_root/'followup-completed.json')['results']['candidate']
    meta,_,files=load(Path(role['initial_snapshot']));directory=Path(meta['sandbox'])
    assert sha(directory/'candidate.py')==CODE
    baseline=read(args.parent/'B-result.json');result=read(args.parent/'C-result.json')
    assert valid_run(baseline) and valid_run(result)
    assert not done['results']['B']['constraints'][0]['passed'] and not done['results']['C']['constraints'][0]['passed']
    paths=[]
    for arm in ['B','C']:
        paths += [(arm+'_request',source.output/'jobs'/arm/'run/request.json'),(arm+'_result',args.parent/(arm+'-result.json'))]
    index={'schema_version':'fixed-quality-constraint-artifacts.v1',
        'scope':'Original unobserved fixed B/C evaluator artifacts for exact historical candidate. No paired-noise/independent acceptance or native profile in this batch.',
        'limits':'Python route labels are candidate telemetry, not independent native proof. No full output tensors or new source comparison in this fixed batch. Existing full quality comparisons and every sample remain authoritative for their actual scope.',
        'artifacts':[{'kind':kind,'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}for kind,path in paths]}
    index_raw=(json.dumps(index,indent=2)+'\n').encode()
    feedback={'trial':'external-fixed-cache-interface-7e78','bundle':bundle.model_dump(mode='json'),
        'selection':json.loads(files['inputs/selection.json']),'proposal':read(directory/'proposal.json'),
        'baseline':baseline,'candidate':result,
        'decision':{'accepted':False,'reasons':['fixed_original_ttft_constraint_failed'],
            'scope':'Developer-assembled diagnostic continuation marker from one fixed B/C. Not a formal paired-noise rejection, autonomous-search result or independent acceptance.'},
        'evaluation':{'formal_acceptance_performed':False,'original_constraints':done['results']['C']['constraints']},
        'measurement':{'scope':'Unobserved original evaluator, one fixed B then C. All five samples each retained; not a paired performance decision. Original prepared anchor remains frozen.',
            'baseline_metrics':baseline['ranks'][0]['result']['metrics'],'candidate_metrics':result['ranks'][0]['result']['metrics'],
            'baseline_samples':baseline['ranks'][0]['result']['samples'],'candidate_samples':result['ranks'][0]['result']['samples'],
            'baseline_constraints':done['results']['B']['constraints'],'candidate_constraints':done['results']['C']['constraints']},
        'resources':{'groups':[]},
        'supplemental_evidence':{'fixed_execution_artifacts':{'scope':index['scope'],'limits':index['limits'],
            'index':{'path':str(args.output/'evidence-index.json'),'sha256':digest(index_raw),'bytes':len(index_raw)}}}}
    raw=(json.dumps(feedback,indent=2)+'\n').encode();packaged,compact=feedback_artifacts(raw.decode())
    registry=source.revisions.load(current.registry_ref,GroupRegistry);plan=source.revisions.load(current.plan_ref,ExecutionPlan)
    routing={}
    for label,kwargs in [('selector',{}),('historical_candidate',{'selected_groups':feedback['selection']['groups'],'plan':plan})]:
        routed=route(compact,current_revision=current.id,registry=registry,**kwargs);d=json.loads(routed)
        assert d['supplemental_evidence']==feedback['supplemental_evidence']
        assert d['project_measurement']==feedback['measurement']
        assert d['quality_and_coverage'][0]['checks']==result['ranks'][0]['result']['checks']
        routing[label]={'bytes':len(routed.encode()),'sha256':digest(routed.encode()),'facts_and_all_samples_survive':True}
    historical=candidate_artifacts(raw.decode(),source.store)
    assert historical['previous-candidate/candidate.py']==(directory/'candidate.py').read_bytes()
    frozen=protocol['frozen_identities'].copy()
    extra=[path for _,path in paths]+[args.parent/n for n in ['terminal-audit.json','public-terminal-audit.json','followup-completed.json','staged-identities.json']]
    extra += [source.output/n for n in ['project/current.json','request.json','prepared.json','discovery.json']]
    for arm in ['B','C']:
        job=source.output/'jobs'/arm
        extra += [p for p in job.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for path in extra:frozen[str(path)]=sha(path)
    assert all(sha(p)==h for p,h in frozen.items())
    transport={'feedback_bytes':len(raw),'feedback_sha256':digest(raw),'index_bytes':len(index_raw),'index_sha256':digest(index_raw),
        'route_checks':routing,'historical_candidate_sha256':CODE,'historical_code_exact':True,
        'packaged_artifact_count':len(packaged),'scope':'CPU routing only; actual new role snapshots/reads remain to audit. No hand-written code/cause, setup-gate audit, new group or prompt supplied. Prior developer assistance remains; not unaided or same-snapshot A/B.'}
    return source,raw,index_raw,frozen,transport


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    source,raw,index_raw,frozen,transport=prepare(args,config)
    summary={'core':framework_source(),'source':str(source.output),'parent':source.revisions.current().id,
        'devices':source.request.devices,'frozen_files':len(frozen),'transport':transport,
        'current_ttft_anchor':23.285755,'ttft_limit':24.45004275,'rounds':1}
    if args.preflight:print(json.dumps({'status':'passed',**summary}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    (args.output/'diagnostic-feedback.json').write_bytes(raw);(args.output/'evidence-index.json').write_bytes(index_raw)
    write_json(args.output/'transport-preflight.json',transport)
    write_json(args.output/'protocol.json',{'at':now(),**summary,'frozen_identities':frozen,
        'scope':'One newly declared ordinary selector/candidate trial from current-device empty parent. Developer-assembled fixed quality/constraint feedback, exact7e78 historical only. No manual group/candidate/cause or prompt change. Reuse completed discovery/controls without model/source/reference rerun/copy. Default scopes/parameter exploration, API observe-only. Not unaided, same-snapshot A/B, prior batch restart or retry-until-pass. No dedicated native or independent stage; actual planner may request collectors.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    try:
        branch=args.output/'qwen-continuation';parent=fork_run(source.output,branch);search=ProjectSearch(config,branch)
        assert parent.id==source.revisions.current().id and not parent.bundles
        assert search.prepared==source.prepared and search.request==source.request
        assert (branch/'discovery.json').read_bytes()==(source.output/'discovery.json').read_bytes()
        write_json(args.output/'preparation-reuse.json',{'source':str(source.output),'parent':parent.id,
            'prepared':source.prepared['revision'],'controls_path':search.prepared['controls_path'],
            'request_sha256':sha(branch/'request.json'),'discovery_sha256':sha(branch/'discovery.json'),
            'initial_bundles':[],'scope':'Completed current-device discovery and original controls reused; no initial GPU rediscovery or reference copies.'})
        argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
            '--request',str(branch/'request.json'),'--output',str(branch),'--resume','--rounds','1','--feedback',str(args.output/'diagnostic-feedback.json')]
        write_json(args.output/'search-protocol.json',{'at':now(),'argv':argv,'feedback_sha256':digest(raw),'scope':'Existing ordinary roles/scopes/default parameter exploration; no independent/native stage scheduled by controller.'})
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'one-ordinary-constraint-feedback-trial'})
        with(args.output/'search-controller.log').open('wb')as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        current=ProjectSearch(config,branch).revisions.current()
        results['search']={'status':'completed'if p.returncode==0 else'error','returncode':p.returncode,
            'current':current.id,'bundles':list(current.bundles),
            'trials':[{'path':str(p),'sha256':sha(p),'decision':read(p).get('decision')}for p in sorted((branch/'trials').glob('*/result.json'))],
            'independent_acceptance':'not_run_in_this_batch'}
        assert source.revisions.current().id==parent.id
    except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
    same=all(sha(p)==h for p,h in frozen.items());results['preservation']={'status':'passed'if same else'failed','files':len(frozen),'unchanged':same}
    done={'phase':'completed'if same and results.get('search',{}).get('status')=='completed'else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',done);write_json(args.output/'followup-completed.json',done)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','parent','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
