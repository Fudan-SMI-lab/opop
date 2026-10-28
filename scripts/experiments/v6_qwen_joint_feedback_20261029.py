"""One ordinary trial using existing failed feedback plus observed full-output facts."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.feedback import candidate_artifacts,route
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.models.project import GroupRegistry,ExecutionPlan
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():return datetime.now(timezone.utc).isoformat()


def inputs(args,config):
    protocol=read(args.parent/'protocol.json');audit=read(args.parent/'terminal-audit.json')
    assert read(args.parent/'followup-completed.json')['phase']=='completed'
    assert protocol['core']==audit['core']==framework_source()
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    origin=Path(protocol['parent']);source=ProjectSearch(config,origin/'qwen-continuation');parent=source.revisions.current()
    assert not parent.bundles and parent.id=='b61996e1a998de911fa0412b595ef484b75dc79765d404aed835051e885b8a3c'
    feedback=source.output/'latest-feedback.json';old=read(feedback)
    assert old['trial']=='trial-a254520daa' and old['decision']=={'accepted':False,'reasons':['no_valid_parameter_candidate']}
    assert audit['endpoints']['1-C-unobserved']['worker_status']=='failed'
    observed=audit['endpoints']['2-C-observed']['observed'];rows=observed['observations']
    assert len(rows)==6 and all(r['exact_full_source']==(i!=4)for i,r in enumerate(rows))
    assert all(sha(r['path'])==r['sha256']for r in rows)
    fact={'scope':'Developer-supplied fixed execution observations of the failed trial-a254520daa, not a root-cause answer or performance acceptance. Original framework feedback fields/bytes retained.',
        'candidate_sha256':protocol['candidate_sha256'],
        'unobserved':'Original evaluator repeated the same teacher-forced token AssertionError before clean measurement. No timing or paired acceptance.',
        'observed_continuous_steps':[
            {'step':0,'greedy_source':'full tokens/logits exactly equal','teacher_source':'full tokens/logits exactly equal','greedy_before_after_observe':'bitwise equal'},
            {'step':1,'greedy_source':'full tokens/logits exactly equal','teacher_source':'12 of32 tokens differ; full finite logits first differ at decode position1, max_abs31.125,4596832 elements exceed CPU float32 .02+.02*abs(source)',
             'teacher_prefill':'position0 full logits exactly equal source','greedy_before_after_observe':'bitwise equal'}],
        'routes':observed['routes'],
        'limits':'Observer CPU copies perturb host/synchronization/lifetimes. Callbacks are not native traces; missing routes do not prove absence of fallback. Facts do not identify unique shared-pool/cache/alias cause. Original full work/quality/TTFT anchor remain; no relaxed checks or old acceptance overwrite.',
        'audit':{'path':str(args.parent/'terminal-audit.json'),'sha256':sha(args.parent/'terminal-audit.json')},
        'raw_observations':{'path':str(args.parent/'2-C-observed/relation-observations.json'),'sha256':sha(args.parent/'2-C-observed/relation-observations.json')}}
    raw=feedback.read_bytes();index=raw.rfind(b'}');assert index>=0
    key='fixed_continuous_full_output_diagnostic';assert key not in old and 'supplemental_evidence'not in old
    # The preserved first attempt put this at the top level, where route()
    # intentionally ignores unknown fields. Use its existing evidence channel.
    prior=read(args.prior_attempt/'protocol.json')
    assert read(args.prior_attempt/'followup-completed.json')['phase']=='completed'
    assert fact==read(args.prior_attempt/'supplemental-facts.json')
    assert prior['source_feedback_sha256']==sha(feedback) and prior['parent_revision']==parent.id
    addition=b',\n  "supplemental_evidence": '+json.dumps({key:fact},ensure_ascii=False,indent=2).encode()+b'\n'
    augmented=raw[:index]+addition+raw[index:]
    rebuilt=json.loads(augmented);assert rebuilt.pop('supplemental_evidence')=={key:fact} and rebuilt==old
    assert augmented[:index]==raw[:index] and augmented[index+len(addition):]==raw[index:]
    history=candidate_artifacts(augmented.decode(),source.store)
    from hashlib import sha256
    assert sha256(history['previous-candidate/candidate.py']).hexdigest()==protocol['candidate_sha256']
    assert all(r['status']=='available'for r in json.loads(history['previous-candidate/manifest.json'])['files'])
    registry=source.revisions.load(parent.registry_ref,GroupRegistry);plan=source.revisions.load(parent.plan_ref,ExecutionPlan)
    _,control=feedback_artifacts(raw.decode());_,treatment=feedback_artifacts(augmented.decode())
    for groups in [None,old['selection']['groups']]:
        kwargs={'current_revision':parent.id,'registry':registry}
        if groups is not None:kwargs.update(selected_groups=groups,plan=plan)
        before=json.loads(route(control,**kwargs));after=json.loads(route(treatment,**kwargs))
        assert before['supplemental_evidence']is None and after['supplemental_evidence']=={key:fact}
        after['supplemental_evidence']=None;assert after==before
    return protocol,audit,source,parent,feedback,raw,augmented,fact


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    protocol,audit,source,parent,feedback,raw,augmented,fact=inputs(args,config)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'parent_revision':parent.id,
            'original_feedback_sha256':sha(feedback),'original_feedback_bytes':len(raw),'augmented_feedback_bytes':len(augmented),
            'added_bytes':len(augmented)-len(raw),'historical_candidate_sha256':protocol['candidate_sha256'],
            'adapter_sha256':source.prepared['adapter_sha256'],'controls_shared_read_only':source.prepared['controls_path'],
            'prior_attempt':str(args.prior_attempt),'facts_identical_to_prior_attempt':True,
            'full_selector_and_candidate_route_checks':'passed; only supplemental_evidence differs'}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    frozen={**protocol['frozen_identities'],str(feedback):sha(feedback),str(args.parent/'terminal-audit.json'):sha(args.parent/'terminal-audit.json')}
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent_batch':str(args.parent),
        'source':str(source.output),'parent_revision':parent.id,'installed_bundles':[],
        'source_feedback':str(feedback),'source_feedback_sha256':sha(feedback),'source_feedback_bytes':len(raw),
        'augmented_feedback_bytes':len(augmented),'added_bytes':len(augmented)-len(raw),'frozen_identities':frozen,
        'prior_attempt':str(args.prior_attempt),'prior_attempt_audit_sha256':sha(args.prior_attempt/'terminal-audit.json'),
        'facts_identical_to_prior_attempt':True,'input_channel':'existing supplemental_evidence',
        'historical_candidate_sha256':protocol['candidate_sha256'],'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'One corrected-interface ordinary trial from the same original pre-trial empty baseline/feedback as the completed omitted-field attempt. Same supplemental facts, only the experimental container changed to existing supplemental_evidence. No ce85 implementation/result or new developer answer supplied. Not latest-ce85 repair, fresh unaided cohort or full candidate-role single-variable A/B. Preserve prior failed attempt. No production changes/duplicate discovery/references/independent acceptance.'})
    write_json(args.output/'supplemental-facts.json',fact)
    path=args.output/'diagnostic-feedback.json';path.write_bytes(augmented)
    results={}
    try:
        branch=args.output/'qwen-continuation';forked=fork_run(source.output,branch);search=ProjectSearch(config,branch)
        assert forked.id==parent.id and not forked.bundles
        assert (branch/'request.json').read_bytes()==(source.output/'request.json').read_bytes()
        assert (branch/'discovery.json').read_bytes()==(source.output/'discovery.json').read_bytes()
        assert search.prepared['controls_path']==source.prepared['controls_path']
        write_json(args.output/'preparation-reuse.json',{'source':str(source.output),'branch':str(branch),'revision':parent.id,
            'request_sha256':sha(branch/'request.json'),'discovery_sha256':sha(branch/'discovery.json'),
            'adapter_sha256':search.prepared['adapter_sha256'],'controls_path':search.prepared['controls_path'],
            'installed_bundles':[],'scope':'Completed original preparation/discovery reused. Failed candidate bytes available only through normal historical feedback artifacts.'})
        argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
            '--request',str(branch/'request.json'),'--output',str(branch),'--resume','--rounds','1','--feedback',str(path)]
        write_json(args.output/'search-protocol.json',{'at':now(),'argv':argv,'feedback_sha256':sha(path),
            'source_feedback_sha256':sha(feedback),'original_bytes_preserved':True,'added_bytes':len(augmented)-len(raw),
            'scope':'One ordinary trial, default parameter exploration, original allowed scopes/work/quality/TTFT. Existing schema repair only, no repeat until pass. API observe-only. Any accepted version needs later independent full-version AB/BA/native audit.'})
        write_json(args.output/'state.json',{'phase':'running','stage':'one-autonomous-trial','at':now()})
        with(args.output/'search-controller.log').open('wb')as log:process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        current=ProjectSearch(config,branch).revisions.current()
        results['search']={'status':'completed'if process.returncode==0 else'error','returncode':process.returncode,
            'current':current.id,'bundles':list(current.bundles),'independent_acceptance':'not_run_in_this_batch'}
        assert source.revisions.current().id==parent.id
        assert all(sha(p)==h for p,h in frozen.items())
    except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed'if all(r['status']=='completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','parent','output','prior-attempt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
