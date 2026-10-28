"""Audit ordinary failed-feedback trial and the experimental field omission."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import GroupRegistry,ExecutionPlan
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import agent_record


def audit(root):
    terminal=read(root/'followup-completed.json');protocol=read(root/'protocol.json');reuse=read(root/'preparation-reuse.json')
    assert terminal['phase']=='completed'
    config=load_config(root/'private-config.yaml',[]);s=ProjectSearch(config,root/'qwen-continuation');parent=s.revisions.current()
    source=ProjectSearch(config,Path(reuse['source']));core=protocol['core']
    assert parent.id==source.revisions.current().id==reuse['revision']==protocol['parent_revision'] and not parent.bundles
    assert s.prepared==source.prepared and all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    assert sha(s.output/'request.json')==reuse['request_sha256'] and sha(s.output/'discovery.json')==reuse['discovery_sha256']
    assert sha(root/'diagnostic-feedback.json')==read(root/'search-protocol.json')['feedback_sha256']
    baseline=s.revisions.revision(s.prepared['revision']);base_files={f.path:f.sha256 for f in baseline.files}
    assert all(hashlib.sha256(s.store.get_artifact(f.content_ref)).hexdigest()==f.sha256 for f in baseline.files)
    trials=[read(p)for p in(s.output/'trials').glob('*/result.json')];assert len(trials)==1;t=trials[0]
    assert t['trial']=='trial-b46364b259' and t['decision']=={'accepted':False,'reasons':['no_valid_parameter_candidate']}
    assert not t.get('pairs') and t['proposal']['parameter_space']=={} and len(t['parameter_search']['observations'])==1
    observation=t['parameter_search']['observations'][0];failed=observation['run']
    assert failed['status']=='failed' and observation['bundle']['parameters']=={}
    assert all(r['framework_source']==core and r['error']=='teacher-forced replay token sequence differs from greedy run'for r in failed['ranks'])
    assert all(not(r.get('result')or{}).get('samples')for r in failed['ranks'])
    code=Path(t['candidate_agent']['sandbox'])/'candidate.py'
    assert sha(code)=='ce85acc72901aa6bfe77ca88b28977a8a28d83ff2dbd94dd566076e95d4ceba3'
    events=[json.loads(l)for l in(s.output/'events.jsonl').read_text().splitlines()]
    agents=[e['payload']for e in events if e['type']=='PROJECT_AGENT_FINISHED']
    private=root/'audit';private.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    agent_rows=[];registry=s.revisions.load(parent.registry_ref,GroupRegistry);plan=s.revisions.load(parent.plan_ref,ExecutionPlan)
    original_raw=Path(protocol['source_feedback']).read_text();augmented_raw=(root/'diagnostic-feedback.json').read_text()
    _,original=feedback_artifacts(original_raw);_,augmented=feedback_artifacts(augmented_raw)
    needles=['qwen-joint-relation-20261029','fixed_continuous_full_output_diagnostic','supplemental-facts.json','diagnostic-feedback.json','31.125','4596832']
    for a in agents:
        row=agent_record(a,db);meta,prompt,files=load(Path(a['generation_snapshots'][0]));assert meta['framework_source']==core
        assert all((Path(a['sandbox'])/name).read_bytes()==data for name,data in files.items())
        if a['role']in ['selection','candidate']:
            groups=t['selection']['groups']if a['role']=='candidate'else None
            kwargs={'current_revision':parent.id,'registry':registry}
            if groups is not None:kwargs.update(selected_groups=groups,plan=plan)
            routed=route(augmented,**kwargs);assert route(original,**kwargs)==routed
            assert files['inputs/previous-feedback.txt']==routed.encode()
            feedback=json.loads(routed);assert feedback['supplemental_evidence']is None
            assert feedback['previous_trial']=='trial-a254520daa' and feedback['decision']['accepted']is False
            assert hashlib.sha256(files['inputs/previous-candidate/candidate.py']).hexdigest()==protocol['historical_candidate_sha256']
            assert json.loads(files['inputs/request.json'])==s.request.portable()
            row.update(feedback_bytes=len(routed.encode()),initial_added_fact_received=False,
                original_and_augmented_full_routing_equal=True,historical_candidate_sha256=protocol['historical_candidate_sha256'])
        parts=[{'part_id':i,'data':json.loads(v)}for i,v in db.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],))]
        path=private/(a['role']+'-session-parts.json')
        if path.exists():assert read(path)==parts
        else:write_json(path,parts)
        bounded=[];hits=[];probes=[]
        for part in parts:
            p=part['data']
            if p.get('type')!='tool':continue
            state=p.get('state',{});inp=state.get('input',{});cmd=inp.get('command','');out=str(state.get('output',''));serialized=json.dumps(inp)
            matched=[n for n in needles if n in serialized or n in out]
            if matched:hits.append({'part':part['part_id'],'matched':matched})
            bounded.append({'part':part['part_id'],'tool':p.get('tool'),'status':state.get('status'),
                'path':inp.get('filePath'),'command_preview':cmd[:500],'exit':state.get('metadata',{}).get('exit'),
                'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'output_bytes':len(out.encode()),
                'contains_traceback_text':'Traceback (most recent call last)'in out})
            if a['role']=='candidate' and 'candidate.py'in cmd and ('python'in cmd):
                pp=private/(part['part_id']+'-probe.json');write_json(pp,{'input':inp,'output':out,'exit':state.get('metadata',{}).get('exit')})
                probes.append({'part':part['part_id'],'path':str(pp),'sha256':sha(pp),'exit':state.get('metadata',{}).get('exit')})
        assert not hits
        row.update(private_session_parts={'path':str(path),'sha256':sha(path)},tools_bounded=bounded,probes=probes,
            new_evidence_path_or_scalar_hits=hits,tool_inventory=dict(Counter(x['tool']for x in bounded)),
            external_access_limit='No explicit new diagnostic paths/field/scalars in complete tool inputs/outputs. Commands were reviewed; this is recorded tool-access evidence, not a hermetic filesystem or proof of all possible provider-internal knowledge.')
        agent_rows.append(row)
    db.close();jobs=[];diagnostics=[]
    for path in sorted((s.output/'jobs').glob('*/run/rank-0-started.json')):
        started=read(path);req=read(path.parent/'request.json');worker=read(path.parent/'rank-0.json');job=path.parents[1]
        assert worker['framework_source']==core and req['parameters']['request']==s.request.portable()
        assert started['parameters_ref']['request_sha256']==sha(path.parent/'request.json')
        actual=started['revision']['files'];assert all(actual.get(k)==v for k,v in base_files.items())
        assert all(sha(job/'source'/n)==h for n,h in actual.items())
        jobs.append({'request':str(path.parent/'request.json'),'request_sha256':sha(path.parent/'request.json'),
            'entry_sha256':sha(req['entry']),'action':req['parameters']['action'],'worker_status':worker['status'],
            'source_files':actual,'revision':started['revision']['revision']})
        if req['parameters']['action']=='candidate_diagnostics':
            side=path.parent/'candidate-diagnostics-rank-0.json';d=read(side)
            diagnostics.append({'path':str(side),'sha256':sha(side),'events':d['events'],'worker_result':worker.get('result'),
                'scope':'Separate run-only diagnostic, never observe/compare. False predicate/context-exit is normal; valid=true does not overturn failed quality or establish native paths.'})
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':core,
        'frozen_identities_verified':protocol['frozen_identities'],'reuse':reuse,
        'trial':{'id':t['trial'],'candidate_path':str(code),'candidate_sha256':sha(code),'decision':t['decision'],
            'selection':{k:v for k,v in t['selection'].items()if k!='affected_calls'},'affected_calls':len(t['selection']['affected_calls']),
            'parameter_space':t['proposal']['parameter_space'],'revision':observation['revision'],'screen_run':failed,
            'parameter_result':observation['result'],'clean_samples':0,'pairs':0},
        'agents':agent_rows,'jobs':jobs,'diagnostics':diagnostics,'collection_plan':t['resources']['collection_plan'],
        'event_counts':dict(Counter(e['type']for e in events)),
        'pointers':{'current':parent.id,'source':source.revisions.current().id,'bundles':[]},
        'review':{
            'input':'Both selection and candidate initial routed inputs lack intended facts; full route replay matches snapshots exactly. Both read original feedback and a866 historical code. Candidate explicitly printed supplemental_evidence=null. No recorded tools read new diagnostic paths/raw values. Do not attribute result to omitted facts.',
            'code':'ce85 derives decode step from current tensor length/data_ptr instead of mutable step counter, allocates separate prefill pool, and adds eager-vs-Graph build checks. It still returns reused mirror objects. Build checks replay graphs directly with an independent eager cache and greedy/greedy/teacher order; they do not execute framework signature dispatch or actual run/observe/run/observe sequence. Self-checks are not equivalent to the failed full evaluator.',
            'probes':'CPU import/view/step-resolution/mock-route/schema checks only. Initial mock runner omitted graph fields, fell back to parent string, then AttributeError past_key_values (exit1); later fuller mock passed routes. This is not full GPU validation or established runtime failure cause. Historical traceback printed by selector is not a new tool failure.',
            'failure':'One new candidate, one empty-parameter screen, same token relation AssertionError before clean metrics. Which step/values fail for ce85 is unavailable; do not copy a866 second-teacher tensor facts to this new code.',
            'unsupported':'Selector says FIRST observe though original failed record did not identify step; old a866 observer establishes second observe. Profile-window subtraction is not clean host-cost attribution, setup can influence TTFT despite decode timed after it, separate pool/self-verification does not prove safe lifetimes or guarantee quality. No adoption.',
            'next':'Correct only experimental input container to existing supplemental_evidence and execute one fixed corrected-interface trial from same original pre-trial baseline/feedback. Preserve ce85 failure and all old output; no production change, manual candidate or added root-cause answer, and no repeat-until-pass.'},
        'df':subprocess.check_output(['df','-B1','/root/autodl-tmp'],text=True)}
    write_json(root/'terminal-audit.json',report)
    print(json.dumps({'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json'),'bytes':(root/'terminal-audit.json').stat().st_size,
        'trial':t['trial'],'candidate_sha256':sha(code),'roles':[{'role':a['role'],'repairs':a['contract_repairs'],'tools':a['tool_inventory'],'usage':a['usage']}for a in agent_rows],
        'jobs':len(jobs),'diagnostics':len(diagnostics),'df':report['df']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
