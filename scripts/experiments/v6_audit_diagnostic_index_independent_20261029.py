"""Audit all original independent samples and traces without GPU/agent reruns."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import coverage_errors
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_qwen_independent_20261029 import native


def aligned(run, name):
    values = [r['result'].get('metric_samples',{}).get(name) or [s[name] for s in r['result']['samples']] for r in run['ranks']]
    series = [max(x) for x in zip(*values, strict=True)]
    center = statistics.median(series)
    return {'samples':series, 'median':center, 'mad':statistics.median(abs(x-center) for x in series)}


def audit(root):
    p=read(root/'protocol.json');done=read(root/'followup-completed.json')
    assert done['phase']=='completed' and done==read(root/'state.json')
    assert p['core']==framework_source()
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    config=load_config(root/'private-config.yaml',[])
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':done['finished_at'],
            'core':p['core'],'terminal_sha256':sha(root/'followup-completed.json'),
            'frozen_files_verified':len(p['frozen_identities']),'cases':{},'logs':[]}
    sample_count=check_count=trace_count=0
    for case,d in p['cases'].items():
        search=ProjectSearch(config,root/(case+'-independent'))
        source=ProjectSearch(config,Path(d['source']))
        original=ProjectSearch(config,Path(d['original_source']))
        assert search.prepared==source.prepared and search.request==source.request
        assert search.revisions.current().id==d['baseline'] and source.revisions.current().id==d['fixed_current']
        assert original.revisions.current().id==d['original_source_current']
        files=list((search.output/'version-trials').glob('*/result.json'));assert len(files)==1
        path=files[0];t=read(path)
        assert sha(path)==done['results'][case+'-independent']['result_sha256']
        assert t['source_revision']==d['candidate'] and t['bundles']==d['bundles']
        incoming=source.revisions.revision(d['candidate']);candidate=search.revisions.revision(t['revision'])
        baseline=search.revisions.revision(d['baseline'])
        for field in ['files','bundles','contract_ref','plan_ref','registry_ref']:
            assert getattr(incoming,field)==getattr(candidate,field)
        assert candidate.contract_ref==baseline.contract_ref==d['contract_ref']
        assert not baseline.bundles and len(candidate.bundles)==1
        assert coverage_errors(search.revisions,candidate,[pair['candidate'] for pair in t['pairs']])==t['coverage_errors']==[]
        assert len(t['pairs'])==2 and t['quality']=='pass'
        endpoints=[]
        for pair_index,pair in enumerate(t['pairs']):
            assert set(pair)=={'parent','candidate'}
            for role in (['parent','candidate'] if pair_index==0 else ['candidate','parent']):
                run=pair[role];assert valid_run(run) and run['worker_pids_reaped']
                job=search.output/'jobs'/f"{t['trial']}-pair-{pair_index}-{role}"
                out=job/'run';req=read(out/'request.json');assert run==read(out/'receipt.json')
                revision=baseline if role=='parent' else candidate
                revision_files={f.path:f.sha256 for f in revision.files}
                assert req['parameters']['request']==search.request.portable()
                assert req['parameters']['action']=='evaluate' and req['parameters']['reference_dir']==d['controls_path']
                assert sha(req['parameters']['adapter'])==d['adapter_sha256']
                assert req['entry']==str(Path(req['cwd'])/'opop_evaluate.py')
                assert run['entry_sha256']==sha(req['entry'])==revision_files['opop_evaluate.py']
                assert all(sha(Path(req['cwd'])/f)==h for f,h in revision_files.items())
                bundles=req['parameters']['bundles'];assert len(bundles)==int(role=='candidate')
                if bundles:
                    assert sha(bundles[0]['file'])==d['candidate_sha256'] and bundles[0]['parameters']==d['parameters']
                ranks=[]
                for rr in run['ranks']:
                    ri=rr['rank'];v=rr['result'];worker=read(out/f'rank-{ri}.json');started=read(out/f'rank-{ri}-started.json')
                    assert worker['status']=='passed' and worker['result']==v and worker['framework_source']==p['core']
                    assert started['revision']['files']==revision_files and started['parameters_ref']['request_sha256']==sha(out/'request.json')
                    assert v['valid'] and v['math_policy_unchanged'] and len(v['samples'])==5 and len(v['checks'])==2
                    assert [c['continuous_step'] for c in v['checks']]==[0,1]
                    elements=4861988 if case=='qwen' else (361257090 if ri==0 else 361255042)
                    assert all(c['passed'] and c['max_abs_error']==0 and c['compared_elements']==elements for c in v['checks'])
                    sample_count+=5;check_count+=2
                    for metric in v['samples'][0]:
                        assert v['metrics'][metric]==statistics.median(s[metric] for s in v['samples'])
                    if case=='qwen':
                        for s in v['samples']:
                            assert s['generated_tokens']==32 and s['decode_tokens']==31
                            assert abs(s['ttft_ms']+s['decode_ms']-s['end_to_end_ms'])<1e-9
                            assert abs(s['decode_tokens_per_second']-31000/s['decode_ms'])<1e-9
                    traces=[]
                    assert set(v.get('diagnostic_metrics',{}))==set(search.request.diagnostic_metrics)
                    for metric,diag in v.get('diagnostic_metrics',{}).items():
                        assert diag['specification']==req['parameters']['request']['diagnostic_metrics'][metric]
                        assert len(diag['values'])==len(diag['evidence'])==3
                        assert v['metric_samples'][metric]==diag['values'] and v['metrics'][metric]==statistics.median(diag['values'])
                        field=diag['specification']['source']['field']
                        for f,value in zip(diag['evidence'],diag['values'],strict=True):
                            trace,window=trace_window(read(f),'opop.communication_metric_workload');assert window
                            recalc=normalize(trace,{'kernels':[]},ri)
                            assert abs(recalc[field]-value)<1e-8
                            kernels=Counter(e['name'] for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'nccl'in e.get('name','').lower())
                            assert sum(kernels.values())==recalc['communication_kernel_count']
                            traces.append({'path':f,'sha256':sha(f),'metric':metric,'value':value,'nccl_kernels':dict(kernels)})
                            trace_count+=1
                    ranks.append({'rank':ri,'worker_sha256':sha(out/f'rank-{ri}.json'),
                                  **{k:v[k] for k in ['checks','samples','metrics','coverage','setup','bundle_hashes']},'traces':traces})
                primary='end_to_end_ms'if case=='qwen'else'step_ms'
                constraint='ttft_ms'if case=='qwen'else'peak_allocated_bytes'
                metrics=[primary,constraint,'framework.lifecycle_peak_allocated_bytes',*search.request.diagnostic_metrics]
                endpoints.append({'pair':pair_index,'role':role,'request':str(out/'request.json'),
                    'request_sha256':sha(out/'request.json'),'entry_sha256':sha(req['entry']),
                    'aligned':{m:aligned(run,m)for m in metrics},'ranks':ranks})
        m=t['measurement'];assert paired_decision(t['pairs'],search.request.objective,search.prepared['evaluation'])==m
        primary=search.request.objective['name'];manual_pairs=[]
        for i in range(2):
            a=next(e for e in endpoints if e['pair']==i and e['role']=='parent')['aligned'][primary]
            b=next(e for e in endpoints if e['pair']==i and e['role']=='candidate')['aligned'][primary]
            gain=a['median']-b['median'];noise=search.request.objective.get('noise_factor',2)*max(a['mad'],b['mad'])
            assert a['median']==m['parent_values'][i] and b['median']==m['candidate_values'][i]
            assert abs(gain-m['paired_gains'][i])<1e-9 and abs(noise-m['noise_bounds'][i])<1e-9
            manual_pairs.append({'gain':gain,'noise':noise,'gain_above_noise':gain>max(search.request.objective.get('improvement_threshold',0),noise)})
        accepted=not m['reasons'] and not t['coverage_errors'] and t['quality']=='pass'
        assert t['decision']['accepted']==accepted
        assert t['publication']==('exploration'if accepted else'rejected')
        report['cases'][case]={'trial':t['trial'],'path':str(path),'sha256':sha(path),'source_candidate':d['candidate'],
            'revision':t['revision'],'parameters':d['parameters'],'candidate_sha256':d['candidate_sha256'],
            'decision':t['decision'],'publication':t['publication'],'quality':t['quality'],'measurement':m,
            'manual_pairs':manual_pairs,'descriptive_percent':100*(1-m['objective']/m['reference']),
            'endpoints':endpoints,'pointers':{'independent':search.revisions.current().id,'fixed':source.revisions.current().id,'original':original.revisions.current().id}}
        print(json.dumps({'case':case,'accepted':accepted,'measurement':m}),flush=True)
    assert (sample_count,check_count,trace_count)==(60,24,96)
    ns=read(root/'native-qwen-candidate.json');assert ns['status']=='observed'and ns['run']['status']=='passed'
    assert len(ns['ranks'])==1
    report['native']=native(ns['ranks'][0])
    np=Path(ns['ranks'][0]['evidence']).parent;req=read(np/'request.json');started=read(np/'rank-0-started.json');worker=read(np/'rank-0.json')
    assert req['parameters']['action']=='nsys_profile'and worker['status']=='passed'and worker['framework_source']==p['core']
    assert req['parameters']['request']==read(root/'qwen-independent/request.json')
    assert all(sha(Path(req['cwd'])/f)==h for f,h in started['revision']['files'].items())
    assert sha(req['parameters']['bundles'][0]['file'])==p['cases']['qwen']['candidate_sha256']
    assert req['parameters']['bundles'][0]['parameters']=={}
    report['native'].update(request_sha256=sha(np/'request.json'),entry_sha256=sha(req['entry']),normalized_sha256=sha(np/'nsys-normalized.json'),
                            scope='Exact candidate, separately perturbed native profile. Not clean timing or isolated critical-path/host-cost evidence.')
    for case in p['cases']:
        for f in sorted((root/(case+'-independent')/'jobs').rglob('*.log')):
            text=f.read_text(errors='replace')
            report['logs'].append({'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size,
                'traceback': 'Traceback (most recent call last)'in text,
                'warnings':dict(Counter(line for line in text.splitlines() if 'Warning'in line or 'ERROR'in line))})
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    report.update(counts={'clean_rank_samples':sample_count,'full_zero_error_checks':check_count,'recomputed_traces':trace_count},
        source_evidence_reuse=read(root/'source-evidence-reuse.json'),
        limits=['Source comparisons reused by exact identity, not a new source numerical rerun; manual mapping and owner repair boundaries remain.',
                'Generated fixed versions have independent decisions but no preceding autonomous search acceptance or unaided original-entry claim.',
                'No prompt/sampler/allocator adoption; bounded prompt/verifier studies remain closed.',
                'All samples retained. Different communication metrics use distinct invocations, not composable physical costs.',
                'One new Qwen candidate native only; no paired native baseline or unique causal attribution.'],
        df=subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True))
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k not in ['cases','logs']}
    public['audit_sha256']=sha(root/'terminal-audit.json')
    public['cases']={case:{**{k:v for k,v in d.items()if k!='endpoints'},'endpoints':[
        {**{k:v for k,v in e.items()if k!='ranks'},'ranks':[{'rank':r['rank'],'checks':r['checks'],'coverage':r['coverage'],'setup':r['setup'],
        'metrics':r['metrics'],'trace_count':len(r['traces']),'nccl_counts':[dict(x)for x in {tuple(sorted(t['nccl_kernels'].items())) for t in r['traces']}]} for r in e['ranks']]} for e in d['endpoints']]} for case,d in report['cases'].items()}
    public['log_summary']={'files':len(report['logs']),'traceback_files':[r['path']for r in report['logs']if r['traceback']]}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':public['audit_sha256'],'counts':report['counts'],'native':report['native'],'public_bytes':(root/'public-audit.json').stat().st_size}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
