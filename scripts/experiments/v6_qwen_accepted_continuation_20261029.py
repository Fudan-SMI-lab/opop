"""Two ordinary rounds after independent Qwen acceptance, with untouched feedback."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs(args,config):
    p=read(args.parent/'protocol.json');audit=read(args.parent/'terminal-audit.json')
    done=read(args.parent/'followup-completed.json')
    assert done['phase']=='completed' and p['core']==audit['core']==framework_source()
    result=read(Path(done['results']['qwen-independent']['result']))
    assert result['decision']['accepted'] and result['quality']=='pass' and not result['coverage_errors']
    assert result['publication']=='exploration' and result['decision']==audit['cases']['qwen']['decision']
    assert not audit['cases']['fsdp']['decision']['accepted']
    source=ProjectSearch(config,args.parent/'qwen-independent')
    assert source.revisions.current().id==source.prepared['revision']==p['cases']['qwen']['baseline']
    chosen=source.revisions.revision(result['revision'])
    assert len(chosen.bundles)==1 and list(chosen.bundles)==result['bundles']
    assert chosen.contract_ref==source.revisions.current().contract_ref
    feedback=source.output/'latest-feedback.json'
    assert feedback.read_bytes()==Path(done['results']['qwen-independent']['result']).read_bytes()
    assert read(feedback)['revision']==chosen.id
    frozen=dict(p['frozen_identities'])
    for path in [args.parent/'terminal-audit.json',args.parent/'followup-completed.json',feedback,
                 source.output/'project/current.json',source.output/'prepared.json',source.output/'request.json']:
        frozen[str(path)]=sha(path)
    assert all(sha(path)==h for path,h in frozen.items())
    return source,chosen,feedback,frozen


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    source,chosen,feedback,frozen=inputs(args,config)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'accepted_revision':chosen.id,
            'accepted_bundles':list(chosen.bundles),'source_current':source.revisions.current().id,
            'baseline':source.prepared['revision'],'feedback_sha256':sha(feedback),'feedback_bytes':feedback.stat().st_size,
            'frozen_files':len(frozen),'rounds':2,'controls':source.prepared['controls_path'],
            'scope':'Original complete-version feedback only. Fresh accepted-version discovery; no developer diagnostic or manual candidate.'}))
        return
    args.output.mkdir(parents=True,exist_ok=True);assert not (args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'source':str(source.output),
        'source_current':source.revisions.current().id,'accepted_revision':chosen.id,'accepted_bundles':list(chosen.bundles),
        'baseline':source.prepared['revision'],'contract_ref':chosen.contract_ref,'rounds':2,
        'feedback':str(feedback),'feedback_sha256':sha(feedback),'frozen_identities':frozen,
        'sequence':['fork accepted complete version and share controls','fresh discovery and two ordinary real trials','conditional new bundle set independent AB/BA'],
        'scope':'Accepted-version continuation of externally mapped and diagnostically assisted task, not unaided original-entry cohort. Unmodified framework feedback only, no manual group/candidate/developer optimization answer. Original core/prompts/quality/TTFT/acceptance preserved. No additional fixed Nsys scheduled. API observe-only; no retries until pass.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    try:
        branch=args.output/'qwen-continuation'
        revision=fork_run(source.output,branch,revision_id=chosen.id);search=ProjectSearch(config,branch)
        assert revision==chosen and search.prepared==source.prepared and search.request==source.request
        assert (branch/'latest-feedback.json').read_bytes()==feedback.read_bytes()
        assert not (branch/'discovery.json').exists()
        write_json(args.output/'preparation-reuse.json',{'source':str(source.output),'source_current':source.revisions.current().id,
            'branch':str(branch),'revision':chosen.id,'bundles':list(chosen.bundles),'contract_ref':chosen.contract_ref,
            'request_sha256':sha(branch/'request.json'),'adapter_sha256':search.prepared['adapter_sha256'],
            'shared_controls':search.prepared['controls_path'],'feedback_sha256':sha(branch/'latest-feedback.json'),
            'scope':'Existing fork_run accepts a quality-passing feasible evaluated exploration. Source current stays baseline; new branch starts accepted version. No reference copying or reused obsolete discovery.'})
        argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
            '--request',str(branch/'request.json'),'--output',str(branch),'--resume','--rounds','2','--feedback',str(feedback)]
        write_json(args.output/'search-protocol.json',{'at':now(),'argv':argv,'initial_revision':chosen.id,
            'initial_bundles':list(chosen.bundles),'feedback_sha256':sha(feedback),'feedback_bytes':feedback.stat().st_size,
            'scope':'Existing CLI automatically discovers the accepted version before two trials. No new prompt paragraph, selector override, supplied groups or developer diagnostics.'})
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'search'})
        with (args.output/'search-controller.log').open('wb') as log:
            process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        search=ProjectSearch(config,branch);current=search.revisions.current()
        changed=list(current.bundles)!=list(chosen.bundles)
        results['search']={'status':'completed' if process.returncode==0 else'error','returncode':process.returncode,
            'current':current.id,'bundles':list(current.bundles),'changed_bundles':changed,
            'trials':[{'path':str(p),'sha256':sha(p),'decision':read(p).get('decision')}for p in sorted((branch/'trials').glob('*/result.json'))]}
        write_json(args.output/'followup-results.json',results)
        if process.returncode!=0 or not changed:
            results['independent']={'status':'not_run','reason':'Search execution failed or no newly accepted active bundle set; preserve outcome without retrying old version.'}
        else:
            write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':'independent'})
            target=args.output/'qwen-independent'
            fork_run(source.output,target,revision_id=source.prepared['revision'])
            independent=ProjectSearch(config,target)
            result=promote_version(independent,branch,revision_id=current.id,publish=False)
            assert independent.revisions.current().id==source.prepared['revision']
            path=target/'version-trials'/result['trial']/'result.json'
            results['independent']={'status':'completed','path':str(path),'sha256':sha(path),
                **{k:result[k] for k in ['trial','revision','parent','quality','coverage_errors','measurement','decision','publication']}}
    except Exception:
        results['controller']={'status':'error','traceback':traceback.format_exc()}
    try:
        assert all(sha(path)==h for path,h in frozen.items())
        assert source.revisions.current().id==source.prepared['revision']
        results['preservation']={'status':'completed','frozen_files':len(frozen),'source_current':source.revisions.current().id}
    except Exception:
        results['preservation']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed'if all(r['status']in ['completed','not_run']for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'Worker/controller completion, autonomous search acceptance and independent complete-version acceptance are separate. IDs alone do not prove changed effective implementation or multiple-bundle coexistence.'}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ['config','output','parent']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
