"""Read-only audit of exact generated DDP fixed B/C; no workload rerun."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_ddp_observation_route_candidate_20261031 import CODE

ROOT=Path('/root/autodl-tmp/dev/v6-work/ddp-observation-route-fixed-20261031')
OLD='ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b'
METRIC='diagnostic.max_rank_nccl_kernel_sum_ms'


def union_duration(events):
    end=None;total=0
    for e in sorted(events,key=lambda x:x['ts']):
        a,b=e['ts'],e['ts']+e['dur']
        total+=max(0,b-max(a,end if end is not None else a));end=max(b,end if end is not None else b)
    return total


def main():
    assert not (ROOT/'terminal-audit.json').exists()
    p=read(ROOT/'protocol.json');t=read(ROOT/'followup-completed.json')
    assert t==read(ROOT/'state.json') and t['results']==read(ROOT/'followup-results.json')
    assert t['phase']=='completed' and p['core']==framework_source()
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    config=load_config(ROOT.parent/'ddp-observation-route-candidate-20261031/private-config.yaml',[])
    search=ProjectSearch(config,ROOT/'ddp-fixed');source=ProjectSearch(config,Path(p['source']))
    assert search.revisions.current().id==source.revisions.current().id==p['parent']
    assert search.prepared==source.prepared and search.request==source.request
    ids=read(ROOT/'staged-identities.json');candidate=search.revisions.revision(ids['candidate'])
    assert candidate.bundles==('fixed-observation-route',) and candidate.contract_ref==ids['contract_ref']
    assert candidate.retired_bundles[-1]=='trial-d06aa22e2d-param-a72285d2'
    assert candidate.id==t['results']['C']['revision']
    jobs=[];traces=[]
    for arm in ['B','C']:
        folder=ROOT/'ddp-fixed/jobs'/arm/'run'; req=read(folder/'request.json');run=read(ROOT/(arm+'-result.json'))
        assert run==read(folder/'receipt.json') and valid_run(run) and run['worker_pids_reaped']
        assert req['parameters']['request']==search.request.portable() and req['devices']==p['devices']
        assert req['function']=='run' and req['wait_for_devices'] and req['parameters']['action']=='evaluate'
        assert req['parameters']['reference_dir']==search.prepared['controls_path'] and req['parameters']['collector_options']=={}
        assert sha(req['parameters']['adapter'])==p['adapter_sha256'] and sha(req['entry'])==run['entry_sha256']
        assert len(req['parameters']['bundles'])==1
        b=req['parameters']['bundles'][0];assert sha(b['file'])==(OLD if arm=='B' else CODE) and b['parameters']=={}
        manifest=read(Path(req['cwd'])/'.opop-revision.json')
        assert all(sha(Path(req['cwd'])/n)==h for n,h in manifest['files'].items())
        assert constraints_for_runs([run],search.request.objective,search.prepared['evaluation'])==t['results'][arm]['constraints']
        job={'arm':arm,'request_sha256':sha(folder/'request.json'),'receipt_sha256':sha(folder/'receipt.json'),
             'entry_sha256':sha(req['entry']),'manifest_sha256':sha(Path(req['cwd'])/'.opop-revision.json'),
             'bundle':b['id'],'candidate_sha256':sha(b['file']),'ranks':[],'aggregates':{}}
        for rank in run['ranks']:
            k=rank['rank'];z=rank['result'];w=read(folder/f'rank-{k}.json');started=read(folder/f'rank-{k}-started.json')
            assert w['result']==z and w['status']==rank['status']=='passed' and z['valid'] and z['math_policy_unchanged']
            assert rank['framework_source']==p['core'] and rank['binding']['torch_uuid']==p['devices'][k]
            assert rank['binding']['occupancy']['status']=='sampled_clear'
            assert started['parameters_ref']['request_sha256']==sha(folder/'request.json')
            assert all(started['revision']['files'][n]==h for n,h in manifest['files'].items())
            assert len(z['samples'])==5 and len(z['checks'])==2
            assert all(c['passed'] and c['compared_elements']==248879620 for c in z['checks'])
            coverage=z['coverage'][b['id']]
            assert coverage['reference_invocations']=={'5d7d6bb13691b02f533f62b0':1}
            assert {x['route']:x['count'] for x in coverage['candidate_declared_routes']}=={'graph_setup_capture':1,'parent_eager_baseline':1}
            log=(folder/f'rank-{k}.log').read_text();assert 'Traceback (most recent call last)' not in log
            job['ranks'].append({'rank':k,'result_sha256':sha(folder/f'rank-{k}.json'),'log_sha256':sha(folder/f'rank-{k}.log'),
                'accumulate_grad_stream_warning':'stream does not match' in log,
                **{name:z[name] for name in ['samples','checks','metrics','metric_samples','coverage','setup','bundle_hashes']}})
            d=z['diagnostic_metrics'][METRIC];assert d['status']=='observed' and len(d['values'])==len(d['evidence'])==3
            for repeat,(path,value) in enumerate(zip(d['evidence'],d['values'],strict=True)):
                raw=read(path);trace,window=trace_window(raw,'opop.communication_metric_workload');assert window
                calc=normalize(trace,{'kernels':[]},k)
                kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
                comm=[e for e in kernels if 'nccl' in e.get('name','').lower()]
                payloads=[e['args']['In msg nelems']*4 for e in comm]
                assert len(comm)==(13 if arm=='B' else 1)
                assert sorted(payloads)==([28351488]*12+[157541376] if arm=='B' else [497759232])
                assert all(e['args']['Collective name']=='allreduce' for e in comm)
                assert abs(sum(e['dur'] for e in comm)/1000-value)<1e-8
                assert abs(calc['communication_kernel_sum_ms']-value)<1e-8
                assert abs(calc['communication_union_ms']-union_duration(comm)/1000)<1e-8
                assert calc['logical_input_bytes']==sum(payloads)==497759232
                apis=[e for e in trace['traceEvents'] if e.get('cat') in ['cuda_runtime','cuda_driver']]
                traces.append({'arm':arm,'rank':k,'repeat':repeat,'path':path,'sha256':sha(path),'window':window,
                    'kernels':len(kernels),'graph_kernels':sum(bool(e.get('args',{}).get('graph node id')) for e in kernels),
                    'graph_launches':dict(Counter(e['name'] for e in apis if 'graphlaunch' in e.get('name','').lower())),
                    'nccl_names':dict(Counter(e['name'] for e in comm)),'payloads':payloads,
                    'logical_bytes':sum(payloads),'sum_ms':value,'union_ms':union_duration(comm)/1000,
                    'sequences':[e['args'].get('Seq') for e in comm],
                    'nccl_graph_nodes':[e['args'].get('graph node id') for e in comm],
                    'profiled_wall_ms':d['profiled_wall_ms'][repeat]})
        for name in run['ranks'][0]['result']['metrics']:
            series=[r['result']['metric_samples'].get(name) or [v[name] for v in r['result']['samples']] for r in run['ranks']]
            values=[max(v) for v in zip(*series,strict=True)];median=statistics.median(values)
            job['aggregates'][name]={'aligned_rank_max':values,'median':median,'MAD':statistics.median(abs(v-median) for v in values)}
        assert job['aggregates']['step_ms']['median']==t['results'][arm]['constraints'][0]['values'][0]
        jobs.append(job)
    assert len(traces)==12
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':t['finished_at'],'reason':'batch_finished',
        'terminal_sha256':sha(ROOT/'followup-completed.json'),'core':p['core'],'all_frozen_unchanged':True,
        'frozen_files':len(p['frozen_identities']),'jobs':jobs,'traces':traces,'candidate':candidate.id,'code':CODE,
        'source_current':p['parent'],'newbranch_current':search.revisions.current().id,
        'counts':{'jobs':2,'ranks':4,'samples':20,'full_checks':8,'raw_metric_traces':12},
        'claims':['C second full checks nonzero max9.5367431640625e-7/rank0 and1.1920928955078125e-7/rank1, passed original tolerance plus exact-replica observe. No raw full tensors saved or fresh independent source comparison.',
            'Both quality routes first parent then graph_setup_capture; not two optimized checks. Both arms retain AccumulateGrad stream warnings, no uncaught traceback. Installation about8-10ms is not full second-invocation setup/capture cost.',
            'Actual separate workload-bounded raw traces show B13 AR versus C1 AR, unchanged497759232 logical bytes, two graph launches. Preparation barrier outside workload excluded, not deleted or counted as model AR. No new Nsys/physical transport proof. Fused reduction and batching are joint mechanism, not isolated communication causal contribution.',
            'NCCL sums/profile wall are separate intrusive periods, not wire/critical path/clean subtraction. C runtime and lifecycle peaks/reserved grow relative accepted B; no memory constraint or memory-gain claim.',
            'Original fixed B71.369506/C54.459087ms and140.04860205 constraint pass; this single B/C has no paired-noise/search/independent acceptance. All samples retained. Conditional fallback/guard/owner limitations from generation audit remain.'],
        'audit_reader_error':'Attempt1 expected native Collective name all_reduce; actual producer emits allreduce. Exact raw label inspected and reader corrected only. Error retained, no model/GPU rerun.',
        'next_decision':'Declare ONE existing validate_bundle publish=False on exact staged candidate, unchanged current accepted parent/controls/empty parameters. Normal original AB then BA acceptance rules and automatic normal resource-feedback planner/collectors, no manual collector choice or new candidate. Preserve any rejection; only a new genuine acceptance can justify subsequent independent whole-version closure. Fixed B/C samples are not pooled with formal pairs.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
            'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',report)
    public={k:v for k,v in report.items() if k not in ['machine','jobs']}
    public['jobs']=[{**{k:v for k,v in j.items() if k!='ranks'},'ranks':[{k:v for k,v in r.items() if k not in ['coverage','bundle_hashes']} for r in j['ranks']]} for j in jobs]
    public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json');write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),
        'B':jobs[0]['aggregates']['step_ms'],'C':jobs[1]['aggregates']['step_ms'],
        'traces':[{'arm':x['arm'],'rank':x['rank'],'kernels':x['kernels'],'graph_kernels':x['graph_kernels'],'launches':x['graph_launches'],'nccl':x['nccl_names'],'sum':x['sum_ms']} for x in traces],
        'machine':report['machine']}))


if __name__=='__main__':main()
