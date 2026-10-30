"""One exact complete-version independent closure after new formal acceptance."""
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
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import resolve_devices, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_audit_ddp_observation_route_candidate_20261031 import CODE

ACCEPTED='bbe28c40885890ef6d2c1e69bb75f1a6934178494d9d704ab56b0887e76301e6'
BASELINE='7a49c6370b52d881a52550766b279df22684adc16cf3c91be56275f8fe2e1df9'
SOURCE_CURRENT='1fc24545217087adefa1248704619fcdc882a19057336b231d8e02918bf4d0d8'
AUDIT='e431bb807d9c2a3fb3fee3fa1d4f7e4e1a46c9688cd94a6d0731166cccdea900'


def now():return datetime.now(timezone.utc).isoformat()


def prepare(args,cfg):
    a=read(args.parent/'terminal-audit.json');p=read(args.parent/'protocol.json')
    assert sha(args.parent/'terminal-audit.json')==AUDIT
    assert read(args.parent/'followup-completed.json')['phase']=='completed'
    assert a['decision']['accepted'] and a['candidate']==ACCEPTED and a['candidate_sha256']==CODE
    assert a['core']==p['core']==framework_source() and a['counts']['full_checks']==16
    s=ProjectSearch(cfg,args.parent/'ddp-validation');current=s.revisions.current();c=s.revisions.revision(ACCEPTED)
    assert current.id==SOURCE_CURRENT and s.prepared['revision']==BASELINE
    assert c.bundles==('fixed-observation-route',) and not s.revisions.revision(BASELINE).bundles
    b=s.revisions.bundle(c.bundles[0]);assert not b.parameters and not b.requires
    assert {f.path:f.sha256 for f in c.files}[b.installer]==CODE
    assert c.contract_ref==s.revisions.revision(BASELINE).contract_ref
    assert resolve_devices(s.request.devices)==p['devices']
    assert (s.request.continuous_steps,s.request.warmup,s.request.repeats)==(2,2,5)
    assert a['measurement']['constraints'][0]['original_reference']==133.379621
    assert a['measurement']['constraints'][0]['limit']==140.04860205
    prior=args.parent.parent/'ddp-feedback-independent-20261030/source-evidence-reuse.json'
    old=read(prior);assert old['status']=='passed' and old['rank_negative_detected']
    assert all(sha(path)==h for path,h in old['files'].items())
    original=args.parent.parent/'ddp-communication-device-20261030'
    assert sha(original/'source-check.json')=='cbf30d10062ab7a4bce98e8003d0829ed91268e2f4ea0c9685e385bad268611c'
    original_request=read(original/'source-worker/request.json')['parameters']
    assert original_request['controls']==s.prepared['controls_path'] and original_request['request']==s.request.portable()
    reuse={**old,'at':now(),'inherited_record':str(prior),'inherited_sha256':sha(prior),
        'scope':'Exact-hash reuse only of prior fresh-current-device full source/controls comparison and replica negative. No fresh tensor comparison/model/source/control run/reference copy. Prior tiny nonzero errors within original tolerance and manual source/metric mappings remain; no arbitrary-owner proof.'}
    normalized=args.parent/'ddp-validation/jobs/nsys-6db4c3619f/run/nsys-normalized.json'
    for n in a['native']:
        assert sha(n['path'])==n['sha256'] and sha(normalized)==n['normalized_sha256']
        assert n['kernels']==1105 and n['graph_kernels']==1103 and sum(n['launches'].values())==2
        assert len(n['nccl'])==1 and not n['nccl'][0]['graph_node']
    native={'at':now(),'normalized_path':str(normalized),'ranks':a['native'],
        'scope':'Reuse exact508a current-device native collected by prior normal planner.1105 kernels/1103 Graph/two versioned launches and one eager AllReduce/rank, unchanged497759232B. No new Nsys; joint batching and communication fusion, not isolated comm contribution/wire/critical-path proof.'}
    frozen=dict(p['frozen_identities']);frozen.update(old['files']);frozen[str(prior)]=sha(prior)
    for name in ['protocol.json','terminal-audit.json','public-terminal-audit.json','followup-completed.json','followup-results.json',
                 'state.json','launch.json','preparation-reuse.json','startup-verification.json','planner-startup-identities.json']:
        frozen[str(args.parent/name)]=sha(args.parent/name)
    for name in ['request.json','prepared.json','discovery.json','project/current.json','events.jsonl']:
        frozen[str(s.output/name)]=sha(s.output/name)
    for path in (s.output/'jobs').glob('*/run/**/*'):
        if path.is_file() and not path.is_symlink():frozen[str(path)]=sha(path)
    for path in (s.output/'trials').glob('*/result.json'):frozen[str(path)]=sha(path)
    for path in (s.output/'generation-snapshots').glob('*.json'):frozen[str(path)]=sha(path)
    for path in (s.output/'sandboxes').glob('*/collection.json'):frozen[str(path)]=sha(path)
    for path in (args.parent/'terminal-private-history').glob('*.json'):frozen[str(path)]=sha(path)
    assert all(sha(path)==h for path,h in frozen.items())
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    free=int(subprocess.check_output(['df','-B1','--output=avail','/root/autodl-tmp'],text=True).splitlines()[-1])
    assert free>=9*1024**3
    return s,c,frozen,reuse,native,free


def main(args):
    os.umask(0o077);cfg=load_config(args.config,[]);cfg.usage.mode='observe_only'
    source,candidate,frozen,reuse,native,free=prepare(args,cfg)
    info={'core':framework_source(),'source':str(source.output),'accepted_revision':ACCEPTED,'source_current':SOURCE_CURRENT,
        'baseline':BASELINE,'candidate_sha256':CODE,'bundles':list(candidate.bundles),'parameters':{},'devices':source.request.devices,
        'frozen_files':len(frozen),'free_bytes':free,'additional_budget_bytes':1024**3,'warning_floor_bytes':8*1024**3,
        'original_step_anchor_ms':133.379621,'step_limit_ms':140.04860205}
    if args.preflight:print(json.dumps({'status':'passed',**info}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),**info,'frozen_identities':frozen,'publish':False,
        'sequence':['source/native hash-only reuse','one exact complete-version B/C/C/B'],
        'scope':'Necessary independent closure after new formal acceptance, exact508a versus original current-device empty prepared7a49. No new Agent/search/parameters/collector/native/source-control run/reference copy/pointer publication. Original two continuous full checks/two warmups/five samples and postclean diagnostic profiles. All errors/tails/rejections retained; no retry until pass. Joint batching/communication-fusion attribution only.'})
    write_json(args.output/'source-evidence-reuse.json',reuse);write_json(args.output/'native-candidate-reuse.json',native)
    write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'independent'})
    results={}
    try:
        target=args.output/'ddp-independent';base=fork_run(source.output,target,revision_id=BASELINE)
        search=ProjectSearch(cfg,target);assert not base.bundles and search.prepared==source.prepared and search.request==source.request
        write_json(args.output/'preparation-reuse.json',{'source':str(source.output),'accepted_revision':ACCEPTED,
            'source_current':SOURCE_CURRENT,'baseline':base.id,'contract_ref':base.contract_ref,'controls_path':source.prepared['controls_path'],
            'request_sha256':sha(target/'request.json'),'prepared_sha256':sha(target/'prepared.json'),
            'scope':'Fresh leased B/C/C/B workers; shared read-only original current-device controls. No source/reference copies or initial discovery.'})
        result=promote_version(search,source.output,revision_id=ACCEPTED,publish=False)
        assert search.revisions.current().id==BASELINE
        results['independent']={'status':'completed',**{k:result[k] for k in ['trial','revision','parent','source_revision','quality','coverage_errors','measurement','decision','publication']}}
    except Exception:results['independent']={'status':'error','traceback':traceback.format_exc()}
    try:
        assert source.revisions.current().id==SOURCE_CURRENT
        assert all(sha(path)==h for path,h in frozen.items())
        results['preservation']={'status':'completed','files':len(frozen),'source_current':SOURCE_CURRENT}
    except Exception:results['preservation']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed' if all(v['status']=='completed' for v in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    for name,value in [('followup-results.json',results),('state.json',terminal),('followup-completed.json',terminal)]:write_json(args.output/name,value)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--preflight',action='store_true');main(ap.parse_args())
