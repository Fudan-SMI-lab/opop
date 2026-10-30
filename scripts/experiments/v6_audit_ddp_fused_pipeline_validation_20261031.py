"""Read-only audit of exact DDP paired validation and actual normal collectors."""
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.nccl_tuning import read_logs
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze, cross_rank
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_ddp_fused_pipeline_cpu_feedback_20261031 import CODE
from scripts.experiments.v6_audit_ddp_fused_pipeline_fixed_20261031 import OLD, union_duration

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-validation-20261031')
FIELDS = ['communication_kernel_count','communication_kernel_sum_ms','communication_union_ms',
          'compute_overlap_ms','communication_without_compute_ms','logical_input_bytes']


def digest(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


def main():
    assert not (ROOT/'terminal-audit.json').exists()
    p=read(ROOT/'protocol.json'); terminal=read(ROOT/'followup-completed.json')
    assert terminal==read(ROOT/'state.json') and terminal['results']==read(ROOT/'followup-results.json')
    assert terminal['phase']=='completed' and p['core']==framework_source()
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    cfg=load_config(ROOT/'private-config.yaml',[])
    s=ProjectSearch(cfg,ROOT/'ddp-validation'); source=ProjectSearch(cfg,Path(p['source']))
    assert s.revisions.current().id==source.revisions.current().id==p['parent']
    assert s.request==source.request and s.prepared==source.prepared
    assert (s.output/'discovery.json').read_bytes()==(source.output/'discovery.json').read_bytes()
    paths=list((s.output/'trials').glob('*/result.json')); assert len(paths)==1
    t=read(paths[0]); assert t['decision']['accepted'] and t['publication']=='exploration'
    assert paired_decision(t['pairs'],s.request.objective,s.prepared['evaluation'])==t['measurement']
    assert t['measurement']['constraints'][0]['limit']==140.04860205
    assert t['measurement']['constraints'][0]['original_reference']==133.379621
    candidate=s.revisions.revision(p['candidate']); original=source.revisions.revision(p['candidate'])
    assert candidate==original and candidate.bundles==(p['bundle'],)
    bundle=s.revisions.bundle(p['bundle']); assert not bundle.parameters and not bundle.requires
    assert bundle.replaces==s.revisions.current().bundles
    assert next(f.sha256 for f in candidate.files if f.path==bundle.installer)==CODE
    jobs=[]; endpoints=[]; traces=[]; profiles=[]; natives=[]; timelines=[]; timeline_raw=[]
    calibrations=[]; diagnostics=[]; transports=[]; receipts={}
    for folder in sorted((s.output/'jobs').glob('*/run')):
        req=read(folder/'request.json'); rec=read(folder/'receipt.json'); action=req['parameters']['action']
        assert req['parameters']['request']==s.request.portable() and req['devices']==p['devices']
        assert req['parameters']['reference_dir']==s.prepared['controls_path']
        assert req['function']=='run' and req['wait_for_devices']
        assert sha(req['entry'])==rec['entry_sha256'] and sha(req['parameters']['adapter'])==s.prepared['adapter_sha256']
        manifest=read(Path(req['cwd'])/'.opop-revision.json')
        assert all(sha(Path(req['cwd'])/n)==h for n,h in manifest['files'].items())
        parent=folder.parent.name.endswith('-parent'); expected=OLD if parent else CODE
        assert len(req['parameters']['bundles'])==1
        b=req['parameters']['bundles'][0]; assert sha(b['file'])==expected and not b['parameters']
        assert rec['status']=='passed' and len(rec['ranks'])==2 and rec['worker_pids_reaped']
        job={'job':folder.parent.name,'action':action,'request_sha256':sha(folder/'request.json'),
             'receipt_sha256':sha(folder/'receipt.json'),'entry_sha256':rec['entry_sha256'],
             'manifest_sha256':sha(Path(req['cwd'])/'.opop-revision.json'),'code':expected,'ranks':[]}
        for row in rec['ranks']:
            k=row['rank']; z=row['result']; start=read(folder/f'rank-{k}-started.json')
            assert read(folder/f'rank-{k}.json')['result']==z
            assert row['status']=='passed' and z['valid'] and row['framework_source']==p['core']
            assert row['binding']['torch_uuid']==p['devices'][k]
            assert start['parameters_ref']['request_sha256']==sha(folder/'request.json')
            assert all(start['revision']['files'][n]==h for n,h in manifest['files'].items())
            log=(folder/f'rank-{k}.log').read_text()
            job['ranks'].append({'rank':k,'result_sha256':sha(folder/f'rank-{k}.json'),
                'started_sha256':sha(folder/f'rank-{k}-started.json'),'log_sha256':sha(folder/f'rank-{k}.log'),
                'traceback':'Traceback (most recent call last)' in log,'stream_warning':'stream does not match' in log,
                'result_keys':list(z)})
            assert not job['ranks'][-1]['traceback']
            if action=='profile':
                raw=read(z['discovery']); assert sha(raw['trace'])==raw['trace_sha256']
                trace=read(raw['trace']); calc=normalize(trace,raw,k)
                assert all(abs(calc[n]-raw['communication'][n])<1e-8 for n in FIELDS)
                profiles.append({'rank':k,'path':z['discovery'],'sha256':sha(z['discovery']),
                    'trace_sha256':raw['trace_sha256'],'kernels':len(raw['kernels']),'calls':len(raw['calls']),
                    'groups':len(raw['groups']),'profiled_wall_ms':raw['profiled_wall_ms'],
                    'communication':{n:calc[n] for n in FIELDS},'payloads':[c['logical_input_bytes'] for c in calc['collectives']]})
            if action=='collective_calibration':
                c=z['calibration']; assert (c['warmup'],c['repeats'])==(2,5)
                for v in c['rows']:
                    assert v['quality'] and len(v['samples_ms'])==5
                    assert statistics.median(v['samples_ms'])==v['median_completion_ms']
                    assert abs(v['logical_bytes']/(v['median_completion_ms']/1000)-v['logical_payload_bytes_per_second'])<1e-4
                calibrations.append({'rank':k,'data':c})
        jobs.append(job); receipts[job['job']]=rec
        if action=='evaluate':
            assert valid_run(rec)
            endpoint={'job':job['job'],'arm':'parent' if parent else 'candidate','ranks':[],'aggregates':{}}
            for row in rec['ranks']:
                k=row['rank']; z=row['result']; assert z['math_policy_unchanged']
                assert len(z['checks'])==2 and len(z['samples'])==5
                assert all(c['passed'] and c['compared_elements']==248879620 for c in z['checks'])
                cov=z['coverage'][b['id']]
                assert {r['route']:r['count'] for r in cov['candidate_declared_routes']}=={'parent_eager_baseline':1,'graph_setup_capture':1}
                assert cov['reference_invocations']=={'5d7d6bb13691b02f533f62b0':1}
                endpoint['ranks'].append({'rank':k,**{n:z[n] for n in ['checks','samples','metrics','metric_samples','coverage','setup','diagnostics','bundle_hashes']}})
                for name,d in z['diagnostic_metrics'].items():
                    assert d['status']=='observed' and len(d['values'])==len(d['evidence'])==3
                    assert statistics.median(d['values'])==z['metrics'][name]
                    for repeat,(path,v) in enumerate(zip(d['evidence'],d['values'],strict=True)):
                        trace,window=trace_window(read(path),'opop.communication_metric_workload'); assert window
                        calc=normalize(trace,{'kernels':[]},k); kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
                        comm=[e for e in kernels if 'nccl' in e.get('name','').lower()]
                        payloads=[e['args']['In msg nelems']*4 for e in comm]
                        assert len(comm)==(1 if parent else 8) and sum(payloads)==497759232
                        assert sorted(payloads)==([497759232] if parent else [6144]+[56702976]*6+[157535232])
                        assert abs(calc[d['specification']['source']['field']]-v)<1e-8
                        assert abs(sum(e['dur'] for e in comm)/1000-v)<1e-8
                        assert abs(union_duration(comm)/1000-calc['communication_union_ms'])<1e-8
                        traces.append({'job':job['job'],'rank':k,'repeat':repeat,'path':path,'sha256':sha(path),
                            'window':window,'sum_ms':v,'union_ms':calc['communication_union_ms'],'compute_overlap_ms':calc['compute_overlap_ms'],'without_compute_ms':calc['communication_without_compute_ms'],'kernels':len(kernels),
                            'graph_kernels':sum(bool(e.get('args',{}).get('graph node id')) for e in kernels),
                            'graph_launches':dict(Counter(e['name'] for e in trace['traceEvents'] if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'cudaGraphLaunch' in e.get('name',''))),
                            'nccl_names':dict(Counter(e['name'] for e in comm)),'payloads':payloads,
                            'nccl_graph_nodes':[e['args'].get('graph node id') for e in comm],'nccl_streams':[e['args'].get('stream') for e in comm],
                            'profiled_wall_ms':d['profiled_wall_ms'][repeat]})
            for name in rec['ranks'][0]['result']['metrics']:
                series=[r['result']['metric_samples'].get(name) or [v[name] for v in r['result']['samples']] for r in rec['ranks']]
                values=[max(v) for v in zip(*series,strict=True)]; median=statistics.median(values)
                endpoint['aggregates'][name]={'aligned_rank_max':values,'median':median,'MAD':statistics.median(abs(v-median) for v in values)}
            endpoints.append(endpoint)
        normalized=folder/'nsys-normalized.json'
        if normalized.exists():
            for row in read(normalized):
                con=sqlite3.connect('file:'+row['evidence']+'?mode=ro',uri=True)
                kernels=list(con.execute('select s.value,k.graphNodeId,k.end-k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName'))
                apis=list(con.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId group by s.value'))
                joins=list(con.execute("select r.correlationId,count(k.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId left join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=r.correlationId where s.value like 'cudaGraphLaunch%' group by r.correlationId order by r.start"));con.close()
                comm=[k for k in kernels if 'nccl' in k[0].lower()]
                assert len(kernels)==row['cuda_kernels']==len(row['kernels']) and len(comm)==row['nccl_kernels']==8
                assert abs(sum(v[2] for v in comm)/1e6-row['nccl_kernel_sum_ms'])<1e-8
                natives.append({'rank':row['rank'],'path':row['evidence'],'sha256':sha(row['evidence']),
                    'normalized_sha256':sha(normalized),'kernels':len(kernels),'graph_kernels':sum(bool(v[1]) for v in kernels),
                    'launches':{n:c for n,c in apis if n.startswith('cudaGraphLaunch')},'launch_correlations':joins,
                    'nccl':[{'name':n,'graph_node':g,'sum_ms':d/1e6} for n,g,d in comm],
                    'hardware_traffic_status':row['hardware_traffic_status']})
        for path in folder.glob('nccl*.log'):
            transports.append({'path':str(path),'sha256':sha(path),'channel_lines':[v for v in path.read_text().splitlines() if 'Channel ' in v and ' via ' in v]})
        for path in folder.glob('candidate-diagnostics-rank-*.json'):
            d=read(path); diagnostics.append({'path':str(path),'sha256':sha(path),'data':d})
    assert len(jobs)==9 and len(endpoints)==4 and len(traces)==24 and len(natives)==len(profiles)==2
    pairs=[]
    for i,pair in enumerate(t['pairs']):
        ep={}
        for arm,rec in pair.items():
            e=next(e for e in endpoints if e['job'].endswith(f'pair-{i}-{arm}'))
            assert rec==receipts[e['job']];ep[arm]=e['aggregates']['step_ms']
        gain=ep['parent']['median']-ep['candidate']['median'];noise=2*max(v['MAD'] for v in ep.values())
        assert abs(gain-t['measurement']['paired_gains'][i])<1e-8 and abs(noise-t['measurement']['noise_bounds'][i])<1e-8
        pairs.append({'endpoints':ep,'gain':gain,'noise':noise})
    for path in sorted((s.output/'jobs').glob('timeline-*/run/timeline-analysis-rank-*.json')):
        z=read(path);trace=read(z['trace']);assert sha(z['trace'])==z['trace_sha256']
        gradients={e['name']:e['parameters'] for e in z['gradient_readiness']['events']}
        inventory={'buckets':[{k:v for k,v in b.items() if k not in ['observed_parameter_count','cpu_post_accumulate_times_us','last_cpu_post_accumulate_us','timed_collective_identity','interpretation']} for b in z['gradient_readiness']['buckets']]}
        calc=analyze(trace,rank=z['rank'],anchors=z['clock_anchors'],domain=z['clock']['domain'],gradients=gradients,inventory=inventory)
        for k in ['devices','clock','cuda_synchronization','copies_and_memsets','workload_window','gradient_readiness','communication']: assert calc[k]==z[k],k
        timeline_raw.append(z)
        timelines.append({'rank':z['rank'],'path':str(path),'sha256':sha(path),'trace_sha256':z['trace_sha256'],
            'profiled_wall_ms':z['profiled_wall_ms'],'communication':{n:z['communication'][n] for n in FIELDS},
            'sync_count':len(z['cuda_synchronization']),'gradient_events':len(z['gradient_readiness']['events']),
            'bucket_count':len(z['gradient_readiness']['buckets']),'gradient_hooks_enabled':z['gradient_hooks_enabled'],
            'gradient_hook_missing':z['gradient_hook_missing']})
    after=t['resources']['extra_collectors']['after']
    assert cross_rank(timeline_raw)==after['execution_timeline']['cross_rank']
    tuning=read_logs(sorted((s.output/'jobs').glob('nccl-tuning-*/run/nccl*.log')))
    assert all(tuning[k]==after['nccl_tuning'][k] for k in tuning)
    a=t['resources']['collection_plan']['agent']; private=ROOT/'terminal-private-history'
    meta,prompt,files=load(Path(a['generation_snapshots'][0])); sandbox=Path(meta['sandbox'])
    assert meta['framework_source']==p['core'] and all((sandbox/n).read_bytes()==v for n,v in files.items())
    assert len(files)==8 and sum(map(len,files.values()))==82263
    old_events=[json.loads(x) for x in (ROOT.parent/'ddp-accepted-communication-trial-20261030/ddp-search/events.jsonl').read_text().splitlines()]
    old=next(e['payload'] for e in old_events if e['type']=='PROJECT_AGENT_FINISHED' and e['payload']['role']=='collection-planner')
    om,op,_=load(Path(old['generation_snapshots'][0]));assert prompt==op and meta['tool_model_settings']==om['tool_model_settings'] and meta['model']==om['model']
    tools=read(private/(a['session']+'-tools.json')); history=read(private/(a['session']+'.json'))
    access=[];lines=0;truncated=0
    for n,tool in enumerate(tools,1):
        inp,out=tool['input'],tool['output'];md=tool['metadata'] or {}
        assert tool['status']=='completed' and not md.get('outputPath')
        if tool['tool']=='read' and Path(inp['filePath']).is_file():
            rel=Path(inp['filePath']).relative_to(sandbox).as_posix(); source_lines=files[rel].decode().splitlines()
            for number,value in re.findall(r'(?m)^(\d+): (.*)$',out):
                original=source_lines[int(number)-1]
                if '... (line truncated to ' in value:
                    assert original.startswith(value.split('... (line truncated to ')[0]);truncated+=1
                else: assert value==original,(n,number)
                lines+=1
        if n==13:
            script=inp['command'].split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0];tree=ast.parse(script)
            assert all(not isinstance(x,(ast.ImportFrom,ast.AsyncFunctionDef)) for x in ast.walk(tree))
            assert all(all(y.name in ['json','jsonschema'] for y in x.names) for x in ast.walk(tree) if isinstance(x,ast.Import))
            def safe_open(name,mode='r',*args,**kwargs):
                assert mode=='r' and name in ['inputs/collection.schema.json','collection.json','inputs/available-collectors.json','inputs/revisions.json']
                return io.StringIO((sandbox/name).read_text())
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(tree,'reviewed-schema-only','exec'),{'open':safe_open})
            assert stream.getvalue()==out
        if tool['tool']=='write':
            assert n==12 and Path(inp['filePath'])==sandbox/'collection.json'
            assert inp['content'].encode()==(sandbox/'collection.json').read_bytes()
        access.append({'ordinal':n,'tool':tool['tool'],'input':inp,'status':tool['status'],'exit':md.get('exit'),
            'output_sha256':digest(out),'output_bytes':len(out.encode()),'truncated':md.get('truncated')})
    assert len(tools)==13 and a['contract_repairs']==0
    observed={}
    for tool in tools:
        if tool['tool']=='read' and Path(tool['input']['filePath']).is_file():
            name=Path(tool['input']['filePath']).relative_to(sandbox).as_posix()
            observed.setdefault(name,set()).update(int(n) for n in re.findall(r'(?m)^(\d+): ',tool['output']))
    complete={name:{'read_numbered_lines':len(v),'source_lines':len(files[name].decode().splitlines()),'all_lines_exposed':v==set(range(1,len(files[name].decode().splitlines())+1))} for name,v in observed.items()}
    write_json(private/'complete-access-review.json',{'tools':access,'read_lines_verified':lines,'truncated_prefixes':truncated,'input_line_coverage':complete,
        'review':'All13 inputs/outputs inspected and every numbered read line checked against exact initial files; tool8 byte-truncated observations followed by9 offset continuation, coverage explicitly measured, not assumed. Tool12 one plan write0edits and13 pure schema/revision validation exactly reconstructed. All8 initial inputs opened, directory listings only otherwise; no recorded model/tensor/CUDA/processgroup/collective/privateconfig/lateranswer content accesses. Nonhermetic FS. Observations are parent508a discovery, not new02b933 profile/implementation. No childsessions or manual collector choice. No production change.'})
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'reason':'batch_finished',
        'terminal_sha256':sha(ROOT/'followup-completed.json'),'core':p['core'],'frozen_files':len(p['frozen_identities']),
        'all_frozen_unchanged':True,'candidate_sha256':CODE,'candidate':candidate.id,'bundle':bundle.id,'parent':p['parent'],
        'newbranch_current':s.revisions.current().id,'source_current':source.revisions.current().id,
        'trial_sha256':sha(paths[0]),'decision':t['decision'],'measurement':t['measurement'],'publication':t['publication'],
        'gain_percent':100*(1-t['measurement']['objective']/t['measurement']['reference']),
        'manual_pairs':pairs,'jobs':jobs,'endpoints':endpoints,'metric_traces':traces,'profiles':profiles,'native':natives,
        'timelines':timelines,'tuning':tuning,'calibration':calibrations,'diagnostics':diagnostics,'transport_logs':transports,
        'collection_plan':t['resources']['collection_plan']['plan'],
        'agent':{'generation':meta['id'],'session':a['session'],'inputs':len(files),'bytes':sum(map(len,files.values())),
            'prompt_model_tools_equal':True,'assistant':sum(m['data'].get('role')=='assistant' for m in history['message']),
            'tools':len(tools),'contract_repairs':a['contract_repairs'],'usage':a['usage'],'read_lines':lines,'truncated':truncated,'input_line_coverage':complete,
            'history_sha256':sha(private/(a['session']+'.json')),'access_sha256':sha(private/'complete-access-review.json')},
        'counts':{'jobs':9,'ranks':18,'clean_rank_samples':40,'full_checks':16,'metric_traces':24,'native_sqlite':2},
        'limits':['All16 full quality checks passed: firstcheck0, secondrank0max9.5367431640625e-7/rank1max1.1920928955078125e-7 for bothB/C. Exactreplica observe passed, no full raw tensors or fresh independent source comparison. Qualityfirstparent/secondsetup, not two optimized checks. All40samples/tails retained.',
            'Actual new02b933 rawtraces/Nsys show eight eager AllReduce kernels, generateddefaultfine8 payload6144+6x56702976+157535232=497759232B versus accepted508a single497MB. SeventeenGraphlaunches and actual same-trace overlap support scheduling change, not isolated communication contribution or volume reduction.',
            'NCCLsum includeswaiting, Nsys/profile/timeline are different intrusive windows, not clean wall/wire/criticalpath algebra. Native worker channel logs support observedSHM/direct/direct only. No topology-only inference or calibration performed this batch.',
            'Normalplanner reads parent508a1AR discovery and candidateevaluation/revision, not candidate source/newprofileasregistry. Inputobservations firstbyte-truncated then continuation; line coverage separately verified. No observed externalanswer/privateconfig or modelprobes.13tools/10assistant/schemaonly/no repairs; provider estimate not invoice.',
            'Planner rawrationale parentcollective fullyexposed is not causalcriticalpath;36.5-37.4ms separateNCCLprofile dividedby48msclean is not78percentstep cost. Effectivebindingdirect/no functionalreplacements does not mean no whole-step replacement. Bucketmembership/consumerbounds remain limited. Originalplan retained, not correctedinputs.',
            'Run-only candidate diagnostics show six normalFalse generatorlocations125/147/157/176/177/198 totaling458returns perrank, no exceptionevents; neverobserve/compare. Timeline zeroCPUgradientcallbacks underGraph means readinessunknown, not no gradients. One reducerbucket cannot uniquely attribute candidate eightchunks.',
            'Runtime/lifecycle/reserved memory increased versusaccepted508a and varies slightly byworker; no newmemoryconstraint/gain/unique cause. Setup6-12msinstallation excludes laterquality-call capture. Conditionalsetuphooks/copiedtokens/guards/postupdatefallback/rankasymmetric/sharedpool/globaloverride limits remain.',
            'Full controller/protocol/launch/preparation/sessionhealth andworkerlogs retained; AccumulateGrad warnings not uncaughtfailures. No experimentrestart/manualcode/parameterretry.'],
        'next_decision':'Declare ONE independent completeversion B/C/C/B promote_version publishFalse exactaccepted928db266/02b933 versusoriginalCURRENTempty7a49. Reuse exactnew native/sourcehash evidence; no newAgent/search/params/collector/Nsys/sourcecontrol/referencecopy/publication. Preserveanyindependentreject withoutoverwriting formalaccepted. No further automatic correction.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)}
    write_json(ROOT/'terminal-audit.json',report)
    public={k:v for k,v in report.items() if k not in ['transport_logs','df','gpu']}
    public['agent']={k:v for k,v in report['agent'].items() if k not in ['session','history_sha256','access_sha256']}
    public['endpoints']=[{**{k:v for k,v in e.items() if k!='ranks'},'ranks':[{k:v for k,v in r.items() if k not in ['coverage','bundle_hashes','diagnostics']} for r in e['ranks']]} for e in endpoints]
    public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json');write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),
        'gain_percent':report['gain_percent'],'counts':report['counts'],'pairs':pairs,'native':natives,'profiles':profiles,
        'timelines':timelines,'agent':public['agent'],'df':report['df'],'gpu':report['gpu']}))


if __name__=='__main__':main()
