"""Audit one corrected-interface trial; no new generation or GPU execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry, ExecutionPlan
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import agent_record
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint


def audit(root):
    terminal=read(root/'followup-completed.json');protocol=read(root/'protocol.json');reuse=read(root/'preparation-reuse.json')
    assert terminal['phase']=='completed'
    config=load_config(root/'private-config.yaml',[]);s=ProjectSearch(config,root/'qwen-continuation');core=protocol['core']
    source=ProjectSearch(config,Path(reuse['source']));parent=source.revisions.current();current=s.revisions.current()
    assert parent.id==reuse['revision']==protocol['parent_revision'] and not parent.bundles
    assert len(current.bundles)==1 and s.prepared==source.prepared
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    assert sha(s.output/'request.json')==reuse['request_sha256'] and sha(s.output/'discovery.json')==reuse['discovery_sha256']
    assert sha(root/'diagnostic-feedback.json')==read(root/'search-protocol.json')['feedback_sha256']
    baseline=s.revisions.revision(s.prepared['revision']);base_files={f.path:f.sha256 for f in baseline.files}
    for revision in [baseline,current]:
        assert revision.contract_ref==baseline.contract_ref
        assert all(hashlib.sha256(s.store.get_artifact(f.content_ref)).hexdigest()==f.sha256 for f in revision.files)
    trials=[read(p)for p in(s.output/'trials').glob('*/result.json')];assert len(trials)==1;t=trials[0]
    assert t['trial']=='trial-9523ac3f5f' and t['decision']['accepted'] and not t['decision']['reasons']
    assert t['proposal']['parameter_space']=={} and len(t['parameter_search']['observations'])==1
    o=t['parameter_search']['observations'][0];assert o['bundle']['parameters']=={} and o['revision']==current.id
    code=Path(t['candidate_agent']['sandbox'])/'candidate.py';assert sha(code)=='d5bd32d47d6b4d8165f617810a4dbeabed5c5c850932e6f74aaafd9adaa4bf2b'
    screen=endpoint(o['run'],core);pairs=[{k:endpoint(v,core)for k,v in pair.items()}for pair in t['pairs']]
    for result in [screen,*[v for pair in pairs for v in pair.values()]]:
        for sample in result['samples']:
            assert sample['generated_tokens']==32 and sample['decode_tokens']==31
            assert abs(sample['end_to_end_ms']-sample['ttft_ms']-sample['decode_ms'])<1e-6
            assert abs(sample['decode_tokens_per_second']-31000/sample['decode_ms'])<1e-8
            assert abs(sample['generated_tokens_per_second']-32000/sample['end_to_end_ms'])<1e-8
    assert paired_decision(t['pairs'],s.request.objective,s.prepared['evaluation'])==t['measurement']
    events=[json.loads(l)for l in(s.output/'events.jsonl').read_text().splitlines()]
    agents=[e['payload']for e in events if e['type']=='PROJECT_AGENT_FINISHED']
    old=root.parent/'qwen-joint-feedback-20261029/qwen-continuation'
    old_agents={e['payload']['role']:e['payload']for e in (json.loads(l)for l in(old/'events.jsonl').read_text().splitlines())if e['type']=='PROJECT_AGENT_FINISHED'}
    private=root/'audit';private.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    registry=s.revisions.load(parent.registry_ref,GroupRegistry);plan=s.revisions.load(parent.plan_ref,ExecutionPlan)
    _,original=feedback_artifacts(Path(protocol['source_feedback']).read_text())
    _,augmented=feedback_artifacts((root/'diagnostic-feedback.json').read_text())
    rows=[]
    for a in agents:
        row=agent_record(a,db);meta,prompt,files=load(Path(a['generation_snapshots'][0]));assert meta['framework_source']==core
        assert all((Path(a['sandbox'])/name).read_bytes()==data for name,data in files.items())
        if a['role']in ['selection','candidate']:
            kwargs={'current_revision':parent.id,'registry':registry}
            if a['role']=='candidate':kwargs.update(selected_groups=t['selection']['groups'],plan=plan)
            before=json.loads(route(original,**kwargs));after=json.loads(route(augmented,**kwargs))
            assert files['inputs/previous-feedback.txt']==route(augmented,**kwargs).encode()
            assert [k for k in before if before[k]!=after[k]]==['supplemental_evidence']
            assert after['supplemental_evidence']['fixed_continuous_full_output_diagnostic']==read(root/'supplemental-facts.json')
            assert after['previous_trial']=='trial-a254520daa'
            assert hashlib.sha256(files['inputs/previous-candidate/candidate.py']).hexdigest()==protocol['historical_candidate_sha256']
            om,op,of=load(Path(old_agents[a['role']]['generation_snapshots'][0]))
            diff=[k for k in sorted(set(files)|set(of))if files.get(k)!=of.get(k)]
            row.update(initial_fact_received=True,feedback_bytes=len(files['inputs/previous-feedback.txt']),
                differing_inputs_vs_omitted_attempt=diff,prompt_equal_vs_omitted=prompt==op,
                settings_equal_vs_omitted=meta['tool_model_settings']==om['tool_model_settings'])
            if a['role']=='selection':assert diff==['inputs/previous-feedback.txt'] and prompt==op and meta['tool_model_settings']==om['tool_model_settings']
        parts=[{'part_id':i,'data':json.loads(v)}for i,v in db.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],))]
        path=private/(a['role']+'-session-parts.json');write_json(path,parts)
        tools=[];facts=[];probes=[]
        for part in parts:
            p=part['data']
            if p.get('type')!='tool':continue
            st=p.get('state',{});inp=st.get('input',{});cmd=inp.get('command','');out=str(st.get('output',''))
            item={'part':part['part_id'],'tool':p['tool'],'status':st.get('status'),'command_preview':cmd[:600],
                'path':inp.get('filePath'),'exit':st.get('metadata',{}).get('exit'),'output_sha256':hashlib.sha256(out.encode()).hexdigest(),
                'output_bytes':len(out.encode()),'traceback':'Traceback (most recent call last)'in out}
            tools.append(item)
            if 'fixed_continuous_full_output_diagnostic'in out:facts.append(item)
            if '/tmp/opencode/candtest'in cmd:
                pp=private/(part['part_id']+'-cpu-probe.json');write_json(pp,{'input':inp,'output':out,'exit':item['exit']})
                probes.append({'path':str(pp),'sha256':sha(pp),'exit':item['exit']})
        if a['role']in ['selection','candidate']:assert facts
        row.update(private_parts={'path':str(path),'sha256':sha(path)},tools_bounded=tools,fact_reads=facts,
            cpu_probes=probes,tool_inventory=dict(Counter(x['tool']for x in tools)))
        rows.append(row)
    db.close()
    probe=Path('/tmp/opencode/candtest/check.py')
    if probe.exists():
        target=private/'final-cpu-probe.py';target.write_bytes(probe.read_bytes())
    jobs=[];binding=[];diagnostics=[];native=None;timeline=None
    for p in sorted((s.output/'jobs').glob('*/run/rank-0-started.json')):
        started=read(p);req=read(p.parent/'request.json');worker=read(p.parent/'rank-0.json');job=p.parents[1];action=req['parameters']['action']
        assert worker['framework_source']==core and req['parameters']['request']==s.request.portable() and worker['status']=='passed'
        assert started['parameters_ref']['request_sha256']==sha(p.parent/'request.json')
        actual=started['revision']['files'];assert all(actual.get(k)==v for k,v in base_files.items())
        assert all(sha(job/'source'/n)==h for n,h in actual.items())
        jobs.append({'request':str(p.parent/'request.json'),'request_sha256':sha(p.parent/'request.json'),'entry_sha256':sha(req['entry']),
            'action':action,'status':worker['status'],'source_files':actual,'revision':started['revision']['revision']})
        if action=='binding_ablation':
            binding.append({'job':job.name,'ablation':worker['result']['diagnostic_ablation'],
                'endpoint':endpoint({'status':'passed','ranks':[{**worker,'pid':started.get('pid')}]},core)})
        if action=='candidate_diagnostics':
            side=p.parent/'candidate-diagnostics-rank-0.json';d=read(side)
            diagnostics.append({'path':str(side),'sha256':sha(side),'events':d['events'],
                'scope':'run-only; never observe/compare; normal False predicate/context exit is not failure or proof of quality.'})
        if action=='nsys_profile':
            norm=p.parent/'nsys-normalized.json';rank=read(norm)[0]
            conn=sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True);conn.row_factory=sqlite3.Row
            kernels=[dict(r)for r in conn.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k left join StringIds s on k.demangledName=s.id')]
            assert kernels==rank['kernels'] and len(kernels)==rank['cuda_kernels']
            native={'normalized_path':str(norm),'normalized_sha256':sha(norm),'sqlite':rank['evidence'],'sqlite_sha256':sha(rank['evidence']),
                'kernel_count':len(kernels),'graph_kernel_count':sum(bool(r.get('graphNodeId'))for r in kernels),
                'graph_runtime_calls':[list(r)for r in conn.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like '%Graph%' group by s.value")],
                'graph_kernel_counts_per_correlation':dict(Counter(r['correlationId']for r in kernels if r.get('graphNodeId'))),
                'flash_graph_kernels':sum('flash_fwd'in r['name']and bool(r.get('graphNodeId'))for r in kernels),
                'flash_other_kernels':sum('flash_fwd'in r['name']and not r.get('graphNodeId')for r in kernels)}
            conn.close()
        if action=='execution_timeline':
            path=next(p.parent.glob('timeline-analysis-*.json'));d=read(path);assert sha(d['trace'])==d['trace_sha256']
            recalc=analyze(read(d['trace']),rank=0,anchors=d['clock_anchors'],domain=d['clock']['domain'])
            assert recalc['devices']==d['devices'] and recalc['cuda_synchronization']==d['cuda_synchronization'] and recalc['clock']==d['clock']
            timeline={'path':str(path),'sha256':sha(path),'trace':d['trace'],'trace_sha256':d['trace_sha256'],
                'devices':{k:{n:v[n]for n in ['kernel_count','kernel_span_us','kernel_active_union_us','gaps_between_kernels_us']}for k,v in d['devices'].items()},
                'clock':d['clock'],'profiled_wall_ms':d['profiled_wall_ms'],'scope':d['scope']}
    m=t['measurement']
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':core,
        'frozen_identities_verified':protocol['frozen_identities'],'reuse':reuse,'baseline_source_files':base_files,
        'trial':{'id':t['trial'],'candidate_path':str(code),'candidate_sha256':sha(code),'decision':t['decision'],
            'selection':{k:v for k,v in t['selection'].items()if k!='affected_calls'},'affected_calls':len(t['selection']['affected_calls']),
            'parameters':{},'revision':current.id,'screen':screen,'measurement':m,'pairs':pairs,'gain_percent':100*(1-m['objective']/m['reference'])},
        'agents':rows,'jobs':jobs,'binding_diagnostics':binding,'candidate_diagnostics':diagnostics,'native':native,'timeline':timeline,
        'collection_plan':t['resources']['collection_plan']['plan'],'event_counts':dict(Counter(e['type']for e in events)),
        'pointers':{'current':current.id,'source':parent.id,'bundles':list(current.bundles)},
        'counts':{'formal_clean_endpoints':5,'formal_clean_samples':25,'formal_quality_checks':10,'binding_endpoints':len(binding),
            'binding_samples':sum(len(x['endpoint']['samples'])for x in binding),'binding_quality_checks':sum(len(x['endpoint']['checks'])for x in binding)},
        'review':{
            'delivery':'Exact selector snapshot changes only routed supplemental_evidence. Both selector and candidate read facts; downstream selection differs, so candidate is not a same-snapshot one-variable A/B. One success is not stable causal superiority. External filesystem is accessible.',
            'code':'Fresh cache/layer objects per return, request reset, value/length-aware prefix refresh, separate prefill pool and direct replacement greedy/teacher self-checks. These jointly differ from a866; no unique cause isolated. Direct self-checks bypass framework signature dispatch; actual external evaluator passes, not self-check assertions, establish tested quality.',
            'scope':'One joint bundle covers prefill and decode, not multiple-bundle coexistence. One round means no post-accept rediscovery. Corrected-interface developer-factual continuation with prior manual external source mapping, not unaided original-entry success.',
            'probes':'Two CPU fixture errors (cache length then assertion shape) retained, final fake-cache/value/reset guards passed. No full CUDA quality from CPU probes. Selector timeline inspection had a type error. All roles zero schema repairs.',
            'collectors':'Native SQLite independently recomputed for current code. Separate profile, never acceptance timing. Timeline unions/gaps cannot be subtracted from clean scores as isolated host cost. Binding ablation compares whole parent/passthrough workers without candidate build/computation, and yielded negative deltas: not negative physical binding cost or group-specific attribution. Run-only diagnostics do not certify quality.',
            'limits':'Temporary global capture patch/concurrent composition and arbitrary ownership are not generically certified. Source numeric mapping only covers full tokens/logits. Lifetime memory increased; not a memory gain. Original TTFT anchor is25.2536046. Preserve old failures.',
            'next':'Independent full-version AB/BA against original controls/baseline, publish=False, reviewed source identity check, and a separate baseline native profile. Reuse already collected current-candidate native evidence; no new search or repeated acceptance.'},
        'df':subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True)}
    write_json(root/'terminal-audit.json',report)
    public={k:report[k]for k in ['at','finished_at','core','trial','native','timeline','counts','pointers','event_counts','review','df']}
    public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    public['frozen_files_verified']=len(protocol['frozen_identities'])
    public['agents']=[{k:a[k]for k in ['role','session','usage','contract_repairs','prompt_sha256','settings_sha256','inputs','tool_inventory','fact_reads','cpu_probes','differing_inputs_vs_omitted_attempt','prompt_equal_vs_omitted','settings_equal_vs_omitted']if k in a}for a in rows]
    public['binding_diagnostic_medians']=[{'job':x['job'],'full_ms':x['endpoint']['metrics']['end_to_end_ms']}for x in binding]
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'public_bytes':(root/'public-audit.json').stat().st_size,
        'gain_percent':report['trial']['gain_percent'],'native':native,'counts':report['counts'],
        'roles':[{'role':a['role'],'tools':a['tool_inventory'],'usage':a['usage']}for a in rows]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
