"""Read-only FSDP source audit and isolated source-owner memory experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_compare(actual, expected, quality):
    from kernel_optimizer.project.evaluation import compare
    assert set(actual)==set(expected)|{'tied_embedding'}
    assert actual['tied_embedding'] is True
    # Code-reviewed mapping: all four source names/containers are identical.
    return compare({k:actual[k] for k in expected},expected,quality)


def independent_source(run, source):
    import torch
    torch.set_num_threads(4)
    prepared=read(run/'prepared.json');q=read(run/'request.json');controls=Path(prepared['controls_path']);rows=[]
    for rank in range(2):
        path=controls/f'reference-rank-{rank}.pt'
        actual=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert len(actual)==2
        checks=[]
        for index,a in enumerate(actual):
            f=source/f'source-step-{index}-rank-{rank}.pt';expected=torch.load(f,map_location='cpu',weights_only=False,mmap=True)
            result=source_compare(a,expected,q['quality'])
            from kernel_optimizer.project.evaluation import compare
            bad={**a,'logits':a['logits']+1.}
            negative=compare(bad,a,q['quality']);del bad
            untied=compare({**a,'tied_embedding':False},a,q['quality'])
            assert result['passed'] and not negative['passed'] and negative['observation_path']=='output.logits'
            assert not untied['passed'] and untied['observation_path']=='output.tied_embedding'
            checks.append({'step':index,'source':str(f),'source_sha256':sha(f),'comparison':result,
                'full_logits_plus_one_rejected':negative,'false_tied_observation_rejected':untied})
            del expected
        rows.append({'rank':rank,'adapter_reference':str(path),'adapter_reference_sha256':sha(path),'checks':checks});del actual
    return {'status':'passed','mapping':'Same loss/logits/gradients/parameters names, full per-rank tensors. Additional tied_embedding=True checked separately; negative probes mutate saved observations and exercise the actual comparator, not live model injection.', 'ranks':rows}


def worker_entry(context):
    import torch
    import torch.distributed as dist
    from kernel_optimizer.project.evaluation import load_adapter
    params=context['parameters'];adapter=load_adapter(Path(params['adapter']));state=adapter.prepare(context,params['request'])
    expected=torch.load(Path(params['source'])/f"source-step-0-rank-{context['rank']}.pt",map_location='cpu',weights_only=False,mmap=True)
    flag=None;records=[]
    for label,retain in [('A',False),('H',True),('H',True),('A',False)]:
        # Recreate the actual original main() owner outside timing, not a score correction.
        flag=None;torch.cuda.synchronize();before=torch.cuda.memory_allocated()
        if retain:
            flag=torch.tensor(1,device=context['device']);dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        torch.cuda.synchronize();footprint=torch.cuda.memory_allocated()-before
        samples=[]
        for index in range(3):
            adapter.reset(state);adapter.before_run(state);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter_ns();result=adapter.run(state);torch.cuda.synchronize()
            row=adapter.metrics(state,result,(time.perf_counter_ns()-begin)/1e6)
            if index==2:
                observation=adapter.observe(state,result);quality=source_compare(observation,expected,params['request']['quality']);del observation
                assert quality['passed']
            samples.append(row);del result
        records.append({'arm':label,'retained_source_flag':retain,'flag_allocated_bytes':footprint,
            'samples':samples,'final_quality':quality})
    return {'valid':True,'records':records,'adapter_sha256':sha(params['adapter']),
        'scope':'Developer source-owner ablation A/H/H/A, three samples each including warmup effects; no acceptance, no subtraction from frozen metrics, no production adapter edit.'}


def audit(run, source, output):
    from kernel_optimizer.config import load_config
    from kernel_optimizer.project.search import ProjectSearch
    from kernel_optimizer.models.project import ProjectContract
    from kernel_optimizer.project.communication import trace_window, normalize
    from kernel_optimizer.project.generation_snapshot import load
    prepared=read(run/'prepared.json');request=read(run/'request.json');search=ProjectSearch(load_config(run.parent/'private-config.yaml',[]),run)
    contract=search.revisions.load(search.revisions.current().contract_ref,ProjectContract)
    controls=prepared['evaluation'];report={'source_fidelity':independent_source(run,source),'request':request,
        'adapter_sha256':prepared['adapter_sha256'],'evaluator_agent':prepared['agent'],'invariants':contract.invariants,
        'baseline':[],'discovery':[],'generation_inputs':[]}
    for parent in [run,run.with_name(run.name+'.intake')]:
        for path in (parent/'generation-snapshots').glob('*.json'):
            metadata,prompt,files=load(path)
            public_metadata={k:metadata[k] for k in ['model','agent','framework_source'] if k in metadata}
            public_metadata['tool_model_settings_sha256']=hashlib.sha256(json.dumps(metadata['tool_model_settings'],sort_keys=True).encode()).hexdigest()
            report['generation_inputs'].append({'path':str(path),'metadata':public_metadata,
                'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'files':{name:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in files.items()}})
    for r in controls['ranks']:
        result=r['result'];assert result['valid'] and len(result['samples'])==5
        assert result['controls']['wrong_output_rejected'] and all(x['passed'] for x in result['controls']['reset'])
        traces=[]
        for name,record in result['diagnostic_metrics'].items():
            assert len(record['values'])==3 and len(record['evidence'])==3
            for path,value in zip(record['evidence'],record['values']):
                windowed,window=trace_window(read(path),'opop.communication_metric_workload');assert window
                recalc=normalize(windowed,{'kernels':[]},r['rank']);field=record['specification']['source']['field']
                assert abs(recalc[field]-value)<1e-8
                traces.append({'metric':name,'path':path,'sha256':sha(path),'value':value,'window':window,'nccl_kernel_count':recalc['communication_kernel_count']})
        report['baseline'].append({'rank':r['rank'],'core':r['framework_source'],'controls':result['controls'],'samples':result['samples'],
            'metric_samples':result['metric_samples'],'metrics':result['metrics'],'traces':traces})
    report['aggregates']={name:{'aligned_max_samples':[max(v) for v in zip(*(r['result']['metric_samples'][name] for r in controls['ranks']))]} for name in request['diagnostic_metrics']}
    for value in report['aggregates'].values():value['median']=statistics.median(value['aligned_max_samples'])
    report['clean_max_rank_samples']={name:[max(r['result']['samples'][i][name] for r in controls['ranks']) for i in range(5)] for name in ['step_ms','peak_allocated_bytes','allocated_bytes']}
    for receipt in run.glob('jobs/discover-*/run/receipt.json'):
        for rank in read(receipt)['ranks']:
            path=Path(rank['result']['discovery']);d=read(path);assert sha(d['trace'])==d['trace_sha256']
            comm=d['communication']
            report['discovery'].append({'path':str(path),'sha256':sha(path),'rank':rank['rank'],'calls':len(d['calls']),'groups':len(d['groups']),
                'trace':d['trace'],'trace_sha256':d['trace_sha256'],'profiled_wall_ms':d['profiled_wall_ms'],'capabilities':d['capabilities'],
                'communication':{k:v for k,v in comm.items() if k!='collectives'},
                'collective_operations':{op:sum(c['operation']==op for c in comm['collectives']) for op in sorted({c['operation'] for c in comm['collectives']})}})
    report['limits']=['No search/acceptance in this batch.',
        'All four numeric diagnostics were collected in separate sets of invocations, so sum/union/overlap across fields must not be algebraically combined as if from one trace.',
        'Source main retains a consensus CUDA scalar absent from this adapter. Frozen source allocated/peak metrics are 512 bytes larger; attribution needs isolated live-owner experiment.',
        'CPU stored-observation negatives are comparator tests; they are not live GPU fault injection.',
        'Discovery is strongly perturbed; collective duration includes waits, not wire time/critical path.']
    write_json(output,report)
    print(json.dumps({'audit':str(output),'bytes':output.stat().st_size,'source_fidelity':report['source_fidelity']['status'],
        'clean_medians':{k:statistics.median(v) for k,v in report['clean_max_rank_samples'].items()}}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--memory-probe',action='store_true');a=p.parse_args()
    if not a.memory_probe:audit(a.run,a.source,a.output)
    else:
        prepared=read(a.run/'prepared.json');request=read(a.run/'request.json');adapter=Path(prepared['controls_path']).parent/'reference-source/opop_adapter.py'
        job=ExecutionRequest(entry=Path(__file__).resolve(),function='worker_entry',cwd=adapter.parent,output=a.output/'run',devices=request['devices'],distributed_backend='nccl',
            lease_dir=Path(request['lease_dir']),worker_timeout_s=request['worker_timeout_s'],wait_for_devices=True,
            parameters={'request':request,'adapter':str(adapter),'source':str(a.source)})
        write_json(a.output/'protocol.json',job.model_dump(mode='json'));result=LocalExecutor().run(job);write_json(a.output/'result.json',result)
        print(json.dumps({'status':result['status'],'path':str(a.output/'result.json')}))
