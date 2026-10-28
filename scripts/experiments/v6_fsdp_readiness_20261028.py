"""Full source readiness, then a fresh original-entry evaluator/discovery run.

No candidate search or performance acceptance. Source observations remain on
the server for a later reviewed independent adapter mapping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


GOAL = ("Minimize the full training step_ms of the original FP32 GPT-2 Medium FSDP2 workload, "
        "preserving the source's full two-rank global batch, logits, gradient and two-update parameter "
        "quality, numerical policy, completion boundaries and peak_allocated_bytes <=1.05 baseline "
        "constraint. Collect applicable communication diagnostics to understand this distributed task. "
        "Implementation choices may change when full semantics and the memory budget are preserved.")


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def union_ms(events):
    end=None;total=0.
    for start,stop in sorted((float(e['ts']),float(e['ts'])+float(e['dur'])) for e in events):
        total+=max(0.,stop-max(start,end if end is not None else start));end=max(stop,end if end is not None else stop)
    return total/1000.


def source_readiness(args, project):
    out=args.output/'source-readiness'
    command=[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(project/'train.py'),'--output',str(out),'--profile']
    write_json(args.output/'source-protocol.json',{'at':now(),'argv':command,'source_hashes':{p.name:sha(p) for p in project.iterdir() if p.is_file()},
        'scope':'Full frozen source, independent serial global-batch quality, five clean samples and three separate profile invocations per rank. No optimization acceptance.'})
    with (args.output/'source-controller.log').open('wb') as log:
        result=subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
    record={'status':'error' if result.returncode else 'completed','returncode':result.returncode,'ranks':[]}
    if result.returncode:
        record['quality_files']=[str(p) for p in out.glob('quality-rank-*.json')];return record
    for rank in range(2):
        path=out/f'result-rank-{rank}.json';data=read(path)
        assert data['passed'] and len(data['samples'])==5 and len(data['communication_profiles'])==3
        traces=[]
        for filename in data['communication_profiles']:
            raw=read(filename);events=raw['traceEvents']
            windows=[e for e in events if e.get('name')=='source.complete_training_step' and e.get('ph')=='X']
            assert len(windows)==1;window=windows[0];start=window['ts'];stop=start+window['dur']
            kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X' and start<=e['ts']<stop]
            nccl=[e for e in kernels if 'nccl' in e.get('name','').lower()]
            traces.append({'path':filename,'sha256':sha(filename),'window_ms':window['dur']/1000.,
                'kernel_count':len(kernels),'nccl_count':len(nccl),'nccl_names':sorted({e['name'] for e in nccl}),
                'nccl_kernel_sum_ms':sum(e['dur'] for e in nccl)/1000.,'nccl_kernel_union_ms':union_ms(nccl),
                'all_kernel_union_ms':union_ms(kernels),
                'limits':'Same-clock profiler intervals, separate perturbed invocations; neither sum nor union proves wire time or exposed critical path.'})
        record['ranks'].append({'rank':rank,'result':str(path),'sha256':sha(path),'samples':data['samples'],
            'metrics':data['metrics'],'max_rank_samples':data['max_rank_samples'],'profiles':traces,
            'quality':[{'step':x['step'],'passed':x['passed'],'negative_rejected':x['negative_rejected'],
                        'elements':sum(c['elements'] for c in x['checks'].values()),
                        'max_abs_error':max(c['max_abs_error'] for c in x['checks'].values())} for x in data['quality']],
            'source_observations':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.glob(f'source-step-*-rank-{rank}.pt'))]})
    return record


def original_entry(args, project):
    out=args.output/'fsdp-original'
    command=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),
        'optimize-project','--project',str(project),'--command','torchrun --standalone --nproc-per-node=2 train.py --output /tmp/fsdp-source --profile',
        '--goal',GOAL,'--devices','0','1','--output',str(out),'--discover-only']
    write_json(args.output/'original-entry-protocol.json',{'at':now(),'argv':command,
        'scope':'Fresh original project/command/goal only, real intake/evaluator/controls/discovery. No supplied adapter/groups/candidate or source-readiness measurements. No candidate search in this batch.'})
    with (args.output/'original-entry.log').open('wb') as log:
        result=subprocess.run(command,cwd=project,stdout=log,stderr=subprocess.STDOUT)
    row={'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode,'output':str(out)}
    if (out/'prepared.json').exists():
        prepared=read(out/'prepared.json');row.update(prepared_status=prepared['status'],adapter_sha256=prepared.get('adapter_sha256'),
            controls_path=prepared.get('controls_path'),prepared_sha256=sha(out/'prepared.json'))
    if (out/'request.json').exists():
        request=read(out/'request.json');row.update(request_sha256=sha(out/'request.json'),diagnostic_metrics=request.get('diagnostic_metrics',{}))
    return row


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ['output','config']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    project=Path(__file__).resolve().parents[2]/'examples/v6_fsdp_training'
    write_json(args.output/'protocol.json',{'at':now(),'framework_source':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp')),
        'project':str(project),'goal':GOAL,'sequence':['full source readiness','conditional fresh original-entry evaluator/discovery'],
        'scope':'One serial GPU batch, frozen source/default FSDP2. No baseline weakening, precision changes, quality relaxation, score-dependent retries or formal optimization acceptance. '
                'Real Agent/API calls with observe-only usage; all raw evidence remains server-side. No Goal, model polling, delivery tests or bulk local transfer.'})
    results={}
    for name,operation in [('source-readiness',lambda:source_readiness(args,project)),('fsdp-original',lambda:original_entry(args,project))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        if name=='fsdp-original' and results['source-readiness']['status']!='completed':
            results[name]={'status':'not_run','reason':'Full source readiness did not pass; no optimization-ready claim'}
        else:
            try:results[name]=operation()
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results);print(json.dumps({'stage':name,'status':results[name]['status'],'at':now()}),flush=True)
    terminal={'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors','finished_at':now(),
        'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':main()
