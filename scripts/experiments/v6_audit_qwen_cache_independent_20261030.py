"""Read-only full-version closure audit; retain rejections without retry."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.measurement import paired_decision
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint
from scripts.experiments.v6_audit_qwen_independent_20261029 import native
from scripts.experiments.v6_audit_qwen_cache_constraint_20261030 import CODE,ACCEPTED


def main(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json')
    assert terminal['phase']=='completed' and framework_source()==protocol['core']
    frozen=protocol['frozen_identities'];assert all(sha(p)==h for p,h in frozen.items())
    config=load_config(root/'private-config.yaml',[]);s=ProjectSearch(config,root/'qwen-independent')
    parent=ProjectSearch(config,Path(protocol['parent'])/'qwen-continuation')
    old=read(Path(protocol['parent'])/'terminal-audit.json')
    assert sha(Path(protocol['parent'])/'terminal-audit.json')==protocol['parent_audit_sha256']
    assert parent.revisions.current().id==ACCEPTED and s.revisions.current().id==s.prepared['revision']==protocol['original_baseline']
    assert s.prepared==parent.prepared and s.request==parent.request
    paths=list((s.output/'version-trials').glob('*/result.json'));assert len(paths)==1;t=read(paths[0])
    assert t['source_revision']==ACCEPTED and t['parent']==s.prepared['revision']
    assert t['bundles']==list(parent.revisions.current().bundles)
    assert paired_decision(t['pairs'],s.request.objective,s.prepared['evaluation'])==t['measurement']
    jobs=[];samples=0;checks=0
    for path in sorted((s.output/'jobs').glob('*/run/receipt.json')):
        req=read(path.parent/'request.json');rec=read(path);started=read(path.parent/'rank-0-started.json')
        assert req['parameters']['request']==s.request.portable() and rec['entry_sha256']==sha(req['entry'])
        assert started['parameters_ref']['request_sha256']==sha(path.parent/'request.json')
        assert sha(req['parameters']['adapter'])==s.prepared['adapter_sha256']
        assert rec['ranks'][0]['framework_source']==protocol['core'] and rec['ranks'][0]['binding']['torch_uuid']==protocol['devices'][0]
        manifest=read(Path(req['cwd'])/'.opop-revision.json')
        assert all(sha(Path(req['cwd'])/p)==h for p,h in manifest['files'].items())
        assert all(started['revision']['files'][p]==h for p,h in manifest['files'].items())
        for b in req['parameters']['bundles']:assert sha(b['file'])==CODE and b['parameters']=={}
        ep=endpoint(rec,protocol['core']) if rec['status']=='passed' and rec['ranks'][0].get('result',{}).get('valid') else rec
        if 'samples'in ep:
            samples+=len(ep['samples']);checks+=len(ep['checks'])
            for x in ep['samples']:
                assert x['generated_tokens']==32 and x['decode_tokens']==31
                assert abs(x['end_to_end_ms']-x['ttft_ms']-x['decode_ms'])<1e-6
        jobs.append({'job':path.parents[1].name,'request_sha256':sha(path.parent/'request.json'),'receipt_sha256':sha(path),
            'entry_sha256':rec['entry_sha256'],'source_manifest_sha256':sha(Path(req['cwd'])/'.opop-revision.json'),
            'source_revision':started['revision']['revision'],'endpoint':ep})
    recomputed=[]
    for i,pair in enumerate(t['pairs']):
        if len(pair)!=2 or not t['measurement']['valid']:continue
        vv={k:[x['end_to_end_ms']for x in v['ranks'][0]['result']['samples']]for k,v in pair.items()}
        med={k:statistics.median(v)for k,v in vv.items()};mad={k:statistics.median(abs(x-med[k])for x in v)for k,v in vv.items()}
        gain=med['parent']-med['candidate'];noise=2*max(mad.values())
        assert abs(gain-t['measurement']['paired_gains'][i])<1e-8 and abs(noise-t['measurement']['noise_bounds'][i])<1e-8
        recomputed.append({'values':vv,'medians':med,'MAD':mad,'gain':gain,'noise':noise})
    reuse=read(root/'source-evidence-reuse.json');assert all(sha(p)==h for p,h in reuse['files'].items())
    assert sha(reuse['prior_check'])==reuse['prior_check_sha256']
    nr=read(root/'native-candidate-reuse.json')['native'];assert sha(nr['sqlite'])==nr['sqlite_sha256'] and sha(nr['normalized_path'])==nr['normalized_sha256']
    actual_native=native(read(nr['normalized_path'])[0]);assert actual_native['kernel_count']==nr['kernel_count'] and actual_native['graph_kernel_count']==nr['graph_kernel_count']
    assert all(sha(p)==h for p,h in frozen.items())
    m=t['measurement'];report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':protocol['core'],
        'terminal_reason':'batch_finished','frozen_files':len(frozen),'all_frozen_unchanged':True,'trial_sha256':sha(paths[0]),
        'trial':{k:t[k]for k in ['trial','revision','parent','source_revision','bundles','quality','coverage_errors','publication','measurement','decision']},
        'manual_pairs':recomputed,'gain_percent':100*(1-m['objective']/m['reference']) if m['valid'] else None,
        'candidate_sha256':CODE,'jobs':jobs,'counts':{'workers':len(jobs),'samples':samples,'full_checks':checks},
        'source_reuse':reuse,'native_reused':actual_native,'native_scope':'Current candidate capture from prior search, no new trace; perturbed period not acceptance score/critical path/host cost.',
        'pointers':{'source_accepted':parent.revisions.current().id,'independent_current':s.revisions.current().id,'original_empty':old['pointers']['source_current']},
        'scope':'Declared developer-feedback-assisted current-device continuation; independent quality/performance separate from search48.0946416%, not unaided original-entry, production adoption or increment above historical device47.325%. Manual source mapping and cache-owner limits retained.',
        'next':'Qwen closure complete regardless independent decision. Current-device FSDP baseline/source/controls and ONE ordinary communication-focused trial; DDP next. No Qwen chain extension.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',report)
    public=dict(report);public['terminal_audit_sha256']=sha(root/'terminal-audit.json');write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'public_sha256':sha(root/'public-terminal-audit.json'),
        'public_bytes':(root/'public-terminal-audit.json').stat().st_size,'counts':report['counts'],'decision':t['decision'],'measurement':m,'gain_percent':report['gain_percent']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
