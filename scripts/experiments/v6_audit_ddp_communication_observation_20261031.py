"""Read-only independent arithmetic/identity audit of exact B/C observation."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import subprocess
from kernel_optimizer.runtime.execution import ExecutionRequest, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-observation-20261031')
METRIC = 'diagnostic.max_rank_nccl_kernel_sum_ms'


def union(intervals):
    merged = []
    for start,end in sorted(intervals):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(merged[-1][1],end)
        else: merged.append([start,end])
    return merged


def length(intervals): return sum(b-a for a,b in intervals)
def close(a,b): assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-5),(a,b)


def trace_audit(path, saved):
    events = read(path)['traceEvents']; assert sha(path) == saved['sha256']
    windows = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation' and e.get('name') == 'opop.communication_metric_workload']
    assert len(windows) == 1
    start = windows[0]['ts']; end = start+windows[0]['dur']
    inside = [e for e in events if e.get('ph') == 'X' and start <= e.get('ts',-1) <= end]
    kernels = [e for e in inside if e.get('cat') == 'kernel']
    comm = sorted([e for e in kernels if 'nccl' in e['name'].lower()],key=lambda e:e['ts'])
    compute = [e for e in kernels if 'nccl' not in e['name'].lower()]
    assert len(comm) == len(saved['collectives']) == 13
    assert all('AllReduce_Sum_f32_RING_LL' in e['name'] for e in comm)
    names = Counter(e['name'] for e in comm)
    launches = [e for e in inside if e.get('cat') in ('cuda_runtime','cuda_driver') and 'Launch' in e.get('name','')]
    ranges = [e for e in inside if e.get('cat') == 'cpu_op' and e.get('name','').startswith('c10d::')]
    for raw,reported in zip(comm,saved['collectives']):
        k=reported['kernel'];args=raw['args']
        assert k == {'name':raw['name'],'start_us':raw['ts'],'duration_us':raw['dur'],'stream':args.get('stream'),'correlation':args.get('correlation')}
        identity=reported['collective_identity']
        for key,rkey in [('process_group','Process Group Name'),('sequence','Seq'),('operation','Collective name'),('participants','Process Group Ranks'),('dtype','dtype'),('input_elements','In msg nelems'),('output_elements','Out msg nelems')]:
            assert identity[key] == args.get(rkey)
        assert identity['logical_input_bytes'] == args['In msg nelems']*4
        launch=[e for e in launches if e.get('args',{}).get('correlation') == args['correlation']]
        assert len(launch) == 1; launch=launch[0]
        candidates=[e for e in ranges if e.get('pid') == launch.get('pid') and e.get('tid') == launch.get('tid') and e['ts'] <= launch['ts'] and e['ts']+e['dur'] >= launch['ts']+launch['dur']]
        inners=[a for a in candidates if not any(a is not b and a['ts']<=b['ts'] and a['ts']+a['dur']>=b['ts']+b['dur'] and (a['ts']<b['ts'] or a['ts']+a['dur']>b['ts']+b['dur']) for b in candidates)]
        assert len(inners) == 1 and reported['cpu_submission']['status'] == 'observed'
        assert reported['cpu_submission']['ranges'] == [{'name':e['name'],'start_us':e['ts'],'duration_us':e['dur']} for e in candidates]
        assert reported['launch_link']['status'] == 'observed'
        close(reported['launch_link']['launch_to_kernel_start_us'],raw['ts']-launch['ts'])
    ci=union([(e['ts'],e['ts']+e['dur']) for e in comm]);co=union([(e['ts'],e['ts']+e['dur']) for e in compute])
    overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in ci for c,d in co)
    actual={'kernel_count':len(kernels),'communication_kernel_count':len(comm),
        'communication_kernel_sum_us':sum(e['dur'] for e in comm),'communication_union_us':length(ci),
        'compute_union_us':length(co),'compute_overlap_us':overlap}
    for key,value in actual.items():close(value,saved['device_activity'][key])
    sync=Counter(e['name'] for e in inside if e.get('cat') in ('cuda_runtime','cuda_driver') and 'Synchronize' in e.get('name',''))
    assert dict(sync) == saved['synchronization_calls']
    assert sum(e.get('cat') in ('gpu_memcpy','gpu_memset') for e in inside) == saved['gpu_copy_count']
    assert all(saved['unknowns'].get(key) for key in ['gpu_input_ready','consumer_completion','work_future_completion','wire_transfer'])
    payloads=sorted(e['args']['In msg nelems']*4 for e in comm)
    assert payloads == [28351488]*12+[157541376]
    return {'path':str(path),'sha256':sha(path),'window_start_us':start,'window_end_us':end,
        **actual,'logical_input_bytes':sum(payloads),'payloads':payloads,'kernel_names':dict(names),
        'sequences':[e['args']['Seq'] for e in comm],
        'graph_launches':sum('graphlaunch' in e.get('name','').lower() for e in launches),
        'synchronization_calls':dict(sync),'all_13_identity_submission_launch_rows_verified':True}


def main():
    assert not (ROOT/'terminal-audit.json').exists()
    p=read(ROOT/'protocol.json');terminal=read(ROOT/'followup-completed.json')
    assert terminal == read(ROOT/'state.json') and terminal['results'] == read(ROOT/'followup-results.json')
    assert terminal['phase'] == 'completed' and p['core'] == framework_source()
    assert all(sha(path) == h for path,h in p['frozen_identities'].items())
    records={};trace_rows=[];ranks=[]
    for arm,name in [('parent','baseline-observed'),('candidate','candidate-observed')]:
        original=ExecutionRequest.model_validate(read(p['source_requests'][arm])).model_dump(mode='json')
        actual=ExecutionRequest.model_validate(read(ROOT/name/'request.json')).model_dump(mode='json')
        expected={**original,'entry':p['probe'],'function':'run','output':str(ROOT/name),'wait_for_devices':True,
            'parameters':{**original['parameters'],'diagnostic_context':{'original_entry':original['entry'],'original_function':original['function'],'independent_reference':None,'arm':name}}}
        assert actual == expected and sha(actual['entry']) == p['probe_sha256']
        run=read(ROOT/(name+'-result.json'));assert run['worker_pids_reaped']
        workers=[]
        for rank in range(2):
            path=ROOT/name/f'rank-{rank}.json';w=read(path);res=w['result'];workers.append(res)
            epath=ROOT/name/f'diagnostic-rank-{rank}.json';e=read(epath)
            assert w['status'] == 'passed' and w['framework_source'] == p['core'] and res['valid'] and res['math_policy_unchanged']
            assert len(res['checks']) == 2 and all(c['passed'] and c['compared_elements'] == 248879620 and c['max_abs_error'] == 0 for c in res['checks'])
            assert len(res['samples']) == 5 and len(res['metric_samples'][METRIC]) == 3
            assert res['diagnostic_index']['sha256'] == sha(epath)
            assert e['outcome']['status'] == 'returned' and all(e['stages_reached'].values())
            assert len(e['patch_restore']) == 6 and all(x['status'] == 'restored' for x in e['patch_restore'])
            assert len(e['patch_installation']) == 6 and all(x['status'] == 'installed' for x in e['patch_installation'])
            assert e['observer_errors'] == [] and e['gradient_hooks_removed'] == 148
            assert e['hook_events_dropped'] == e['broadcast_calls']['dropped'] == e['trace_files_truncated'] == 0
            inventory=e['passive_bucket_inventory'];assert len(inventory) == 1 and len(inventory[0]['buckets']) == 13
            assert sorted(b['logical_bytes'] for b in inventory[0]['buckets']) == [28351488]*12+[157541376]
            assert sorted(slot for b in inventory[0]['buckets'] for slot in b['param_slots']) == list(range(148))
            for i,saved in enumerate(e['same_invocation_metric_traces']):
                raw=ROOT/name/'secondary-metrics'/saved['directory']/saved['file']
                row=trace_audit(raw,saved);row.update(endpoint=name,rank=rank,repeat=i);trace_rows.append(row)
                close(row['communication_kernel_sum_us']/1000,res['metric_samples'][METRIC][i])
            phases=e['phases']['validation'];join=e['invocation_trace_join']
            if arm == 'parent':
                assert phases['quality_invocations'] == 2 and phases['profile_invocations'] == 3 and not phases['quality_count_mismatch']
                assert join['mapping_complete'] and [x['invocation_ordinal'] for x in join['traces']] == [9,10,11]
                assert res['diagnostic_index']['hook_events'] == 7104 and e['unassigned_hook_events']['count'] == 0
            else:
                assert phases['quality_invocations'] == 1 and phases['quality_count_mismatch'] and phases['profile_invocations'] == 0
                assert join['status'] == 'unknown' and not join['mapping_complete']
                assert all(x['mapping_status'] == 'unknown' and x['invocation_ordinal'] is None for x in join['traces'])
                assert res['diagnostic_index']['hook_events'] == 3848 and e['unassigned_hook_events']['count'] == 3256
                coverage=next(iter(res['coverage'].values()))
                assert {x['route']:x['count'] for x in coverage['candidate_declared_routes']} == {'parent_eager_baseline':1,'graph_setup_capture':1}
            log=ROOT/name/f'rank-{rank}.log';text=log.read_text()
            assert 'Traceback (most recent call last)' not in text
            ranks.append({'endpoint':name,'rank':rank,'worker_sha256':sha(path),'evidence_sha256':sha(epath),
                'artifact_bytes':epath.stat().st_size,'checks':res['checks'],'samples':res['samples'],
                'coverage':res['coverage'],'setup':res['setup'],'binding':w['binding'],
                'phase_validation':phases,'join':join,'broadcast_calls':e['broadcast_calls'],
                'hooks':res['diagnostic_index'],'passive_inventory_logical_bytes':497759232,
                'log_sha256':sha(log),'log_scope':'Deprecation/barrier/loss/profiler messages; candidate stale AccumulateGrad-stream warning retained, not an observed uncaught error.'})
        samples=[max(a['step_ms'],b['step_ms']) for a,b in zip(workers[0]['samples'],workers[1]['samples'])]
        metrics=[max(a,b) for a,b in zip(workers[0]['metric_samples'][METRIC],workers[1]['metric_samples'][METRIC])]
        records[name]={'aligned_maxrank_samples_ms':samples,'median_ms':median(samples),'mad_ms':median(abs(v-median(samples)) for v in samples),
            'original_constraint_limit_ms':140.04860205,'descriptive_within_original_limit':median(samples)<=140.04860205,
            'metric_aligned_maxrank_ms':metrics,'metric_median_ms':median(metrics),'timing_scope':'Observer-perturbed diagnostic; not clean acceptance/no observer-cost subtraction.'}
    assert len(trace_rows) == 12 and sum(len(x['checks']) for x in ranks) == 8
    payload={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':p['core'],
        'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'all_requests_exact_declared_wiring':True,
        'probe_sha256':p['probe_sha256'],'plan_sha256':p['plan_sha256'],'rank_workers':4,'quality_checks':8,'sample_count':20,
        'checks_compared_elements_each':248879620,'all_checks_zero_error':True,'all_workers_valid_math_unchanged':True,
        'ranks':ranks,'endpoints':records,'own_raw_traces_recomputed':12,'traces':trace_rows,
        'findings':[
            'Both observed endpoints original full two-update quality passed; first candidate quality is eager parent, second graph setup/capture. No two-optimized-check claim, new source proof, search/independent acceptance or observer overhead estimate.',
            'B eachrank12 Python-anchored records:2quality/2warmup/5measure/3profile;7104 retained callbacks and three structural trace joins. C eachrank one anchored quality record/592 anchored plus3256unassigned callbacks,296 broadcasts with148 flagged unanchored precedent; quality_count_mismatch and all3profile joins explicitly unknown. This is expected limited observability, not failed original quality or permission to fabricate phase IDs.',
            'All12rawtrace windows have13AllReduceSumf32RING_LL kernels and497759232 logical input bytes, payload12x28351488+157541376. B3999kernels/0GraphLaunch, C4106/2GraphLaunch. This corroborates exact existing compute scheduling, not communication reduction. No new Nsys/physical transport evidence; oldSHM logs not relabeled as new capture.',
            'All156 reported kernel identity/unique same-thread innermost c10d/launch associations and eachtrace sum/union/overlap/sync counts recomputed. No crosscapture clock alignment or subtraction. CPUcallback timestamps and payload/order do not prove GPUready/WorkFuture/wire/consumer or critical path.',
            'Artifact sizes136KB/94KB, no cap loss or observer errors,148 hook removals and six restored labels eachrank. Runtime hook/patch identity is code plus CPUfixture evidence, not an independent GPU identity logger. Full raw callback arrays were not persisted; aggregate summaries cannot all be independently recomputed.',
            'Post-entry bucket getter adds roughly497759232B zeros-like allocation and retained owners affect lifetime; neither step/lifecycle peak before teardown nor baselinepass bounds observer overhead or universal memory/concurrency safety. Plan no-kernel/universal2MB claims remain unadopted.'
        ],
        'decision':'Close this fixed observation and mechanics repair. Next bounded project selection may consume exact new raw B/C results/Agent evidence plus original existing feedback via supplemental index; no developer causal table/manual implementation or repeated observer repair. No automatic GPU/candidate stage.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
            'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',payload)
    public={k:v for k,v in payload.items() if k not in ['ranks','traces','machine']}
    public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json')
    public['rank_summaries']=[{k:v for k,v in x.items() if k in ['endpoint','rank','evidence_sha256','artifact_bytes','phase_validation','hooks','log_scope']} for x in ranks]
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),'endpoints':records,
        'trace_kernel_counts':sorted(set(x['kernel_count'] for x in trace_rows)),'graph_launch_counts':sorted(set(x['graph_launches'] for x in trace_rows))}))


if __name__ == '__main__':main()
