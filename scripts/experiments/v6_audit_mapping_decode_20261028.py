"""Audit frozen metric-schema replay and complete Qwen goal-variant evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.communication import available_fields, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_step_goal_20261028 import read, sha
from scripts.experiments.v6_metric_mapping_decode_20261028 import NOTE


def endpoint(run):
    assert valid_run(run)
    rows=[]
    for rank in run['ranks']:
        result=rank['result'];assert result['math_policy_unchanged']
        assert len(result['checks'])==2 and all(c['passed'] for c in result['checks'])
        assert len(result['samples'])==5
        rows.append({'rank':rank['rank'],'core':rank['framework_source'],
            **{k:result[k] for k in ['checks','coverage','samples','metrics','metric_samples','bundle_hashes']}})
    return rows


def audit(root, output):
    terminal=read(root/'followup-completed.json')
    report={'terminal':terminal['phase'],'finished_at':terminal['finished_at'],'intake':[],'collection':[],'trials':[]}
    arms=read(root/'intake-ab/result.json')['runs'];snapshots=[load(Path(a['agent']['generation_snapshots'][0])) for a in arms]
    (a,ap,af),(b,bp,bf)=snapshots
    assert ap==bp and all(a[k]==b[k] for k in ['model','agent','tool_model_settings','framework_source'])
    assert [k for k in sorted(set(af)|set(bf)) if af.get(k)!=bf.get(k)]==['inputs/intake.schema.json']
    left=json.loads(af['inputs/intake.schema.json']);right=json.loads(bf['inputs/intake.schema.json'])
    prop=right['$defs']['Description']['properties']['diagnostic_metrics']
    assert prop['description'].endswith(' '+NOTE);prop['description']=prop['description'].removesuffix(' '+NOTE)
    assert right==left
    old=load(Path(read(root/'intake-ab/protocol.json')['snapshot']))
    assert af==old[2] and ap==old[1]
    report['input_comparison']={'only_added_schema_paragraph':True,'same_prompt_model_tools_between_arms':True,
        'same_core_between_arms':True,'original_core':old[0]['framework_source'],'replay_core':a['framework_source'],
        'bytes':[len(f['inputs/intake.schema.json']) for _,_,f in snapshots]}
    for arm in arms:
        q=arm['request'];assert q['objective']['name']=='step_ms'
        assert q['objective']['aggregation_order']=='ranks_then_samples'
        assert q['objective']['constraints'][0]['baseline_ratio']==1.05
        assert q['quality']['atol']==1e-6 and q['quality']['rtol']==1e-4
        for k,v in {'world_size':2,'micro_batch':1,'sequence_length':128,'accumulation_steps':4}.items():assert q['workload'][k]==v
        report['intake'].append(arm)
    for row in read(root/'collection-integration.json')['runs']:
        receipt=read(row['receipt']);record={'arm':row['arm'],'action':row['action'],'status':row['status'],'receipt_sha256':sha(row['receipt'])}
        protocol=read(Path(row['receipt']).parent.parent/'protocol.json');task=protocol['parameters']['request']
        if row['action']=='profile':
            ranks=[]
            for r in receipt['ranks']:
                d=read(r['result']['discovery']);assert sha(d['trace'])==d['trace_sha256']
                available=available_fields(d['communication'])|{f['name'] for f in d['capabilities']['fields'] if f['status']=='observed'}
                assert not set(task['required_observations'])-available
                ranks.append({'rank':r['rank'],'path':r['result']['discovery'],'sha256':sha(r['result']['discovery']),
                    'trace_sha256':d['trace_sha256'],'available_fields':sorted(available),'required':task['required_observations']})
            record.update(ranks=ranks,policy=task['communication'],worker_required_gate_executed=task['communication']=='required')
        else:
            record['endpoints']=endpoint(receipt);traces=[];aligned={}
            for name,spec in task['diagnostic_metrics'].items():
                series=[]
                for r in receipt['ranks']:
                    diagnostic=r['result']['diagnostic_metrics'][name];assert len(diagnostic['values'])==3
                    series.append(diagnostic['values'])
                    assert set(spec['source'])=={'collector','field'}
                    for p,value in zip(diagnostic['evidence'],diagnostic['values']):
                        trace,window=trace_window(read(p),'opop.communication_metric_workload');assert window
                        actual=sum(e['dur'] for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X' and ('nccl' in e.get('name','').lower() or 'Collective name' in e.get('args',{})))/1000
                        assert abs(actual-value)<1e-8
                        traces.append({'path':p,'sha256':sha(p),'rank':r['rank'],'sum_ms':value})
                aligned[name]=[max(x) for x in zip(*series)]
                assert statistics.median(aligned[name])==row['secondary'][name]
            record.update(traces=traces,all_aligned_secondary_samples=aligned)
        report['collection'].append(record)
    run=root/'qwen-decode';config=load_config(root/'private-config.yaml',[]);search=ProjectSearch(config,run)
    prepared=search.prepared;parent=Path(prepared['reuse']['parent_output']);oldq=read(parent/'request.json');q=read(run/'request.json')
    assert {k for k in oldq if oldq[k]!=q[k]}=={'goal','objective','devices','notes'}
    source=Path(prepared['controls_path']).parent/'reference-source'
    assert sha(source/'opop_adapter.py')==prepared['adapter_sha256']==read(parent/'prepared.json')['adapter_sha256']
    current=search.revisions.current();baseline=search.revisions.revision(prepared['revision'])
    assert not baseline.bundles and len(current.bundles)==1
    contract=search.revisions.load(current.contract_ref,ProjectContract)
    adapt=prepared['reuse']['contract_adaptation'];am,ap,files=load(Path(adapt['generation_snapshots'][0]))
    assert set(files)=={'inputs/original-contract.json','inputs/request.json','inputs/invariants.schema.json','inputs/adapter.py','inputs/worker-environment.json'}
    report['variant']={'adapter_sha256':prepared['adapter_sha256'],'baseline_revision':baseline.id,'baseline_bundles':list(baseline.bundles),
        'parent_baseline_revision':prepared['reuse']['revision'],'request':q,'adaptation':adapt,'invariants':contract.invariants,
        'controls':[{'rank':r['rank'],'valid':r['result']['valid'],'controls':r['result']['controls']} for r in prepared['evaluation']['ranks']]}
    events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    order=[e['payload']['session'] for e in events if e['type']=='PROJECT_AGENT_FINISHED' and e['payload']['role']=='candidate']
    trials=sorted([read(p) for p in (run/'trials').glob('*/result.json')],key=lambda t:order.index(t['candidate_agent']['session']))
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    for t in trials:
        agent=t['candidate_agent'];file=Path(agent['sandbox'])/'candidate.py';m,p,inputs=load(Path(agent['generation_snapshots'][0]))
        tools=[]
        for ident,data in db.execute('select id,data from part where session_id=? order by time_created,id',(agent['session'],)):
            z=json.loads(data)
            if z.get('type')!='tool':continue
            inp=z.get('state',{}).get('input',{});serialized=json.dumps(inp,sort_keys=True)
            tools.append({'part':ident,'tool':z['tool'],'status':z.get('state',{}).get('status'),'read_path':inp.get('filePath'),
                'input_sha256':hashlib.sha256(serialized.encode()).hexdigest(),'mentions_feedback':'feedback' in serialized})
        obs=[]
        for o in t['parameter_search']['observations']:
            obs.append({'status':o['run']['status'],'result':o['result'],'rank_errors':[r.get('error') for r in o['run']['ranks']]})
        report['trials'].append({'trial':t['trial'],'decision':t['decision'],'measurement':t.get('measurement'),
            'scope':t['selection']['scope'],'group_count':len(t['selection']['groups']),'affected_calls':len(t['selection']['affected_calls']),
            'rationale':t['selection']['rationale'],'candidate':str(file),'sha256':sha(file),'agent':agent,
            'feedback_files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for k,v in inputs.items() if 'feedback' in k},
            'tools':tools,'screening':obs,'pairs':[{arm:endpoint(ep) for arm,ep in pair.items()} for pair in t.get('pairs',[])]})
    db.close()
    native=trials[-1]['resources']['extra_collectors']['after']['nsys']['ranks'][0]
    ns=sqlite3.connect('file:'+native['evidence']+'?mode=ro',uri=True)
    api=list(ns.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like '%Graph%' group by s.value"));ns.close()
    report['native']={'path':native['evidence'],'sha256':sha(native['evidence']),'kernel_count':len(native['kernels']),
        'graph_kernels':sum(bool(k.get('graphNodeId')) for k in native['kernels']),'graph_api_counts':api}
    result_path=next((root/'qwen-decode-independent/version-trials').glob('*/result.json'));ind=read(result_path)
    assert paired_decision(ind['pairs'],q['objective'],prepared['evaluation'])==ind['measurement']
    report['independent']={'path':str(result_path),'sha256':sha(result_path),'decision':ind['decision'],'measurement':ind['measurement'],
        'pairs':[{arm:endpoint(ep) for arm,ep in pair.items()} for pair in ind['pairs']]}
    report['source_fidelity']=terminal['results']['qwen-independent']['source_fidelity']
    report['cross_goals']=read(root/'cross-goals.json')
    report['events']={'counts':dict(Counter(e['type'] for e in events)),
        'discoveries':[{k:e['payload'][k] for k in ['parent','revision']} for e in events if e['type']=='PROJECT_DISCOVERED']}
    report['current']={'revision':current.id,'bundles':list(current.bundles),'no_post_accept_rediscovery':True}
    report['limits']=['Both fresh intake arms chose valid names; no causal success-rate estimate for the added paragraph.',
        'Both chose communication=auto: the required-mode worker gate did not run; the actual observed field sets were independently checked here.',
        'Collection integration reuses an audited adapter; it is not a new original-entry evaluator cohort.',
        'Cross scores are diagnostics from different measurement periods and framework versions, not a paired ranking of the two candidates.',
        'Native tracing is separate from clean timing. Agent host-gap percentages are hypotheses, not causal decomposition.',
        'Frozen historical failures and all samples remain unchanged.']
    write_json(output,report)
    print(json.dumps({'audit':str(output),'bytes':output.stat().st_size,'decode_gain_percent':100*(ind['measurement']['objective']/ind['measurement']['reference']-1)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
