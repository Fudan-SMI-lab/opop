"""Audit the frozen constrained Qwen goal search without rerunning GPU work."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision,valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import agent_record
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def endpoint(run,core,*,controls=False):
    assert valid_run(run) and len(run['ranks'])==1
    rank=run['ranks'][0];result=rank['result']
    assert rank['framework_source']==core and len(result['samples'])==5
    if not controls:
        assert result['math_policy_unchanged'] and len(result['checks'])==2
        assert all(c['passed'] and c['max_abs_error']==0 for c in result['checks'])
    else:
        control=result['controls']
        assert control['identity']['passed'] and all(c['passed']for c in control['reset'])
        assert all(control[k]for k in ['identity_covered','wrong_output_rejected','missing_coverage_rejected'])
    for key,value in result['metrics'].items():assert abs(statistics.median(s[key]for s in result['samples'])-value)<1e-7,key
    return {'pid':rank['pid'],'core':rank['framework_source'],'checks':result.get('checks',[]),
        **{k:result[k]for k in ['controls','samples','metrics','coverage','setup','bundle_hashes']if k in result}}


def audit(root):
    terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    protocol=read(root/'protocol.json');core=protocol['core'];config=load_config(root/'private-config.yaml',[])
    s=ProjectSearch(config,root/'qwen-end-to-end');request=read(s.output/'request.json');baseline=s.revisions.revision(s.prepared['revision'])
    contract=s.revisions.load(baseline.contract_ref,ProjectContract)
    before=read(root/'variant-protocol.json');assert not baseline.bundles
    base_files={f.path:f.sha256 for f in baseline.files};assert base_files==before['reused_files']
    source=Path(s.prepared['controls_path']).parent/'reference-source'
    assert all(sha(source/name)==value for name,value in base_files.items())
    source_run=read(root/'source-check.json');assert valid_run(source_run)
    source_result=source_run['ranks'][0]['result']
    assert sha(source_result['reference'])==source_result['reference_sha256']
    assert all(c['passed'] and c['max_abs_error']==0 for c in source_result['checks'])
    assert sha(Path(s.prepared['controls_path'])/'reference-rank-0.pt')==terminal['results']['source']['controls_reference_sha256']
    events=[json.loads(line)for line in(s.output/'events.jsonl').read_text().splitlines()]
    agents=[e['payload']for e in events if e['type']=='PROJECT_AGENT_FINISHED']
    order=[a['session']for a in agents if a['role']=='candidate']
    trials=sorted([read(p)for p in(s.output/'trials').glob('*/result.json')],key=lambda t:order.index(t['candidate_agent']['session']))
    report={'at':datetime.now(timezone.utc).isoformat(),'completed_at':terminal['finished_at'],'terminal_sha256':sha(root/'followup-completed.json'),
        'core':core,'request':request,'contract':contract.model_dump(mode='json'),'baseline':baseline.id,
        'baseline_source_files':base_files,'source_fidelity':source_result,'controls_reference_sha256':sha(Path(s.prepared['controls_path'])/'reference-rank-0.pt'),
        'controls':endpoint(s.prepared['evaluation'],core,controls=True),'trials':[],'agents':[],
        'event_counts':dict(Counter(e['type']for e in events)),
        'discoveries':[{k:v for k,v in e['payload'].items()if not isinstance(v,(dict,list))}for e in events if e['type']=='PROJECT_DISCOVERED']}
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    for a in agents:
        receipt=agent_record(a,db);m,p,files=load(Path(a['generation_snapshots'][0]));assert m['framework_source']==core
        assert m['fresh_session']
        feedback=files.get('inputs/previous-feedback.txt')
        receipt['feedback_input_bytes']=len(feedback)if feedback else 0
        if feedback:
            try:receipt['previous_decision']=json.loads(feedback)['decision']
            except (ValueError,KeyError):receipt['previous_decision']='not a trial-decision object'
        if 'inputs/project.json'in files:
            project=json.loads(files['inputs/project.json']);receipt['project_revision']=project.get('current_revision')
            receipt['project_keys']=list(project)
        receipt['old_diagnostic_path_mentions']=[]
        for ident,raw in db.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],)):
            part=json.loads(raw)
            if part.get('type')!='tool':continue
            inp=part.get('state',{}).get('input',{});text=json.dumps(inp)
            hits=[word for word in ['qwen-goal-contrast-20261028','metric-mapping-decode-20261028','qwen-ttft/trials','accepted-feedback-replay-20261028']if word in text]
            if hits:receipt['old_diagnostic_path_mentions'].append({'part':ident,'matched':hits,'input_sha256':hashlib.sha256(text.encode()).hexdigest()})
        report['agents'].append(receipt)
    db.close()
    for t in trials:
        candidate=Path(t['candidate_agent']['sandbox'])/'candidate.py'
        row={'trial':t['trial'],'decision':t['decision'],'scope':t['selection']['scope'],'groups':t['selection']['groups'],
            'affected_call_count':len(t['selection']['affected_calls']),'selection_rationale':t['selection']['rationale'],
            'candidate':str(candidate),'candidate_sha256':sha(candidate),'candidate_agent':t['candidate_agent']['session'],
            'bundle':{k:t['bundle'].get(k)for k in ['id','parameters','requires','replaces','scope','installer']},'parameter_observations':[]}
        for o in t['parameter_search']['observations']:
            run=o['run'];assert valid_run(run)
            constraints=constraints_for_runs([run],s.request.objective,s.prepared['evaluation'])
            row['parameter_observations'].append({'parameters':o['bundle']['parameters'],'bundle':o['bundle']['id'],
                'parameter_valid':o['result']['valid'],'score':o['result']['score'],'constraints':constraints,'endpoint':endpoint(run,core)})
        if t.get('pairs'):
            assert paired_decision(t['pairs'],s.request.objective,s.prepared['evaluation'])==t['measurement']
            row['measurement']=t['measurement'];row['pairs']=[{k:endpoint(v,core)for k,v in pair.items()}for pair in t['pairs']]
        report['trials'].append(row)
    independent_file=next((root/'qwen-end-to-end-independent/version-trials').glob('*/result.json'))
    independent=read(independent_file)
    assert paired_decision(independent['pairs'],s.request.objective,s.prepared['evaluation'])==independent['measurement']
    report['independent']={'path':str(independent_file),'sha256':sha(independent_file),'decision':independent['decision'],
        'measurement':independent['measurement'],'pairs':[{k:endpoint(v,core)for k,v in pair.items()}for pair in independent['pairs']]}
    report['native']={}
    for label in ['baseline','candidate']:
        path=root/('native-'+label+'.json');native=read(path);rank=native['ranks'][0]
        db=sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True)
        count,graph=db.execute('select count(*),coalesce(sum(graphNodeId!=0),0) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
        assert count==len(rank['kernels'])
        calls=list(db.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like '%Graph%' group by s.value"))
        db.close();report['native'][label]={'path':rank['evidence'],'sha256':sha(rank['evidence']),
            'kernel_count':count,'graph_kernel_count':graph,'graph_runtime_calls':calls,'nccl_kernels':rank['nccl_kernels']}
    # Every materialized stage keeps the complete baseline source, and records its
    # actual candidate bytes. No identity claim is based only on bundle labels.
    jobs=[]
    for branch in [s.output,root/'qwen-end-to-end-independent']:
        for p in sorted((branch/'jobs').glob('*/run/rank-0-started.json')):
            started=read(p);job=p.parents[1];execution=read(p.parent/'request.json')
            files=started.get('revision',{}).get('files',{})
            assert all(files.get(k)==v for k,v in base_files.items()),str(p)
            assert all(sha(job/'source'/k)==v for k,v in files.items())
            assert started['parameters_ref']['request_sha256']==sha(p.parent/'request.json')
            assert execution['parameters']['request']==s.request.portable()
            jobs.append({'path':str(p),'sha256':sha(p),'revision':started['revision']['revision'],
                'bundle_hashes':started['revision']['bundle_hashes'],'source_files':files})
    report['job_identities']=jobs
    current=s.revisions.current();report['current']={'revision':current.id,'bundles':list(current.bundles)}
    endpoints=[report['controls']]
    for t in report['trials']:
        endpoints.extend(o['endpoint']for o in t['parameter_observations'])
        endpoints.extend(v for pair in t.get('pairs',[])for v in pair.values())
    endpoints.extend(v for pair in report['independent']['pairs']for v in pair.values())
    report['counts']={'evaluated_clean_endpoints':len(endpoints),'clean_samples':sum(len(x['samples'])for x in endpoints),
        'full_quality_checks':sum(len(x['checks'])for x in endpoints),'real_roles':len(report['agents'])}
    report['review']={'contract':'New invariants and role rationales use end_to_end_ms and the fixed TTFT anchor; inherited completion prose still says ttft_ms objective. Execution semantics preserved; not automatically rewritten.',
        'first_trial':'Full quality passed; first paired TTFT25.856583 exceeds25.49943795, so rejected despite full-step gain.',
        'second_trial':'One new joint bundle owns prefill and decode via shared static cache and real CUDA graph replay; no old optimized bundle import. Full independent acceptance passes.',
        'third_trial':'Full quality passed, but screening TTFT27.404051 fails original constraint. Prefill delegates parent (setup_unavailable), decode uses eager-op graph, no fused route. Unique internal setup/validation cause not recorded; do not call this a full numerical failure or successful fusion.',
        'composition':'Accepted second round followed by actual rediscovery and third-round feedback. One active bundle across two groups, not proof of multiple active bundles.',
        'limits':['Same-adapter goal variant, not fresh original-entry evaluator cohort.','Profiles are separate perturbed runs, not score inputs.','Agent host-gap/bandwidth explanations are hypotheses, not isolated causal decompositions.','All samples and historical decisions retained.']}
    report['disk']=disk_observation(Path('/root/autodl-tmp'));write_json(root/'terminal-audit.json',report)
    m=independent['measurement'];print(json.dumps({'audit':str(root/'terminal-audit.json'),'bytes':(root/'terminal-audit.json').stat().st_size,
        'counts':report['counts'],'gain_percent':100*(1-m['objective']/m['reference']),'native':report['native'],
        'external_mentions':sum(len(a['old_diagnostic_path_mentions'])for a in report['agents'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
