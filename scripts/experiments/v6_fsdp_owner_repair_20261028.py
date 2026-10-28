"""One evidence-led evaluator repair through the existing real feedback path.

No production prompt change, candidate search, or automatic adapter adoption.
"""
import argparse
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
from scripts.experiments.v6_audit_fsdp_20261028 import independent_source, read, sha


def preparation(args):
    old=read(args.parent/'prepared.json');adapter=Path(old['controls_path']).parent/'reference-source/opop_adapter.py'
    evidence={
        'scope':'Developer source-fidelity feedback, not an optimization suggestion. Preserve the frozen request and existing observation ABI.',
        'prior_adapter':str(adapter),'prior_adapter_sha256':sha(adapter),
        'finding':'Source train.py main() retains the CUDA scalar passed after all_reduce(MIN) and before every measured() invocation. The baseline adapter omits that live source-local allocation. Its loss/logits/all shard gradients/post-update parameters match the independent serial-global source over two continuous updates exactly; this is a source memory lifetime mismatch only.',
        'evidence':{'source_peak_allocated_bytes':2239245312,'adapter_peak_allocated_bytes':2239244800,
            'source_allocated_bytes':1462069760,'adapter_allocated_bytes':1462069248,
            'isolated_probe':str(args.probe/'result.json'),'isolated_probe_sha256':sha(args.probe/'result.json'),
            'probe_result':'Real two-rank A/H/H/A: A is the unchanged adapter; H additionally holds the same source CUDA scalar outside timing. All three samples per arm and both ranks change allocated and peak by exactly 512 bytes; final full source quality each arm is zero error. No frozen score was changed.'},
        'request':'Use the prior adapter as the starting point and make the minimal source-lifetime correction through actual ownership. Preserve the source scalar semantics/lifetime and release boundary, existing full run/reset/observe and native memory metric reads, source files, request and quality. Do not add a constant to metrics or change precision, model, input, timed work or measurement boundaries. Keep understanding.json truthful about source-required semantics versus implementation choices. No candidate optimization is requested.'}
    feedback=args.output/'source-owner-feedback.json';write_json(feedback,evidence)
    command=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
        '--request',str(args.parent/'request.json'),'--feedback',str(feedback),'--output',str(args.output/'fsdp-repaired'),'--prepare-only']
    write_json(args.output/'repair-protocol.json',{'at':now(),'argv':command,'request_sha256':sha(args.parent/'request.json'),
        'scope':'Existing preparation feedback interface, one new repair workflow. Controls can trigger its normal autonomous repair loop; no repeat because a score failed. This is developer-assisted evaluator correction, not a fresh unaided original-entry cohort.'})
    with (args.output/'repair-controller.log').open('wb') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    return {'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode}


def source_check(args):
    run=args.output/'fsdp-repaired';p=read(run/'prepared.json');assert p['status']=='prepared'
    assert read(run/'request.json')==read(args.parent/'request.json')
    result=independent_source(run,args.source)
    rows=[]
    for rank in p['evaluation']['ranks']:
        source=read(args.source/f"result-rank-{rank['rank']}.json")
        rows.append({'rank':rank['rank'],'actual_samples':rank['result']['samples'],
            'source_samples':source['samples'],'same_allocated_and_peak':all(
                current[name]==expected[name] for current,expected in zip(rank['result']['samples'],source['samples'],strict=True)
                for name in ['allocated_bytes','peak_allocated_bytes'])})
    result.update(adapter_sha256=p['adapter_sha256'],memory=rows,
        limits='Source mapping/allocator sample checks only. Generated code ownership must be reviewed before adoption/search; numbers alone cannot rule out hardcoded corrections.')
    write_json(args.output/'source-check.json',result)
    return {'status':'completed' if all(r['same_allocated_and_peak'] for r in rows) else 'completed_with_memory_difference',
        'path':str(args.output/'source-check.json'),'adapter_sha256':p['adapter_sha256']}


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ['output','config','parent','source','probe']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'protocol.json',{'at':now(),'framework_source':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp')),
        'sequence':['real evaluator source-owner repair and controls','full independent source mapping and memory comparison'],
        'scope':'No production changes, new intake, groups/candidates, search or performance acceptance. Original baseline/failures/samples remain frozen. '
                'New raw references remain server-side; no local bulk transfer, Goal, model polling or delivery tests.'})
    results={}
    for name,op in [('repair',lambda:preparation(args)),('source-check',lambda:source_check(args))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        if name=='source-check' and results['repair']['status']!='completed':results[name]={'status':'not_run','reason':'Preparation failed'}
        else:
            try:results[name]=op()
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results);print(json.dumps({'stage':name,'status':results[name]['status'],'at':now()}),flush=True)
    terminal={'phase':'completed' if all(x['status']=='completed' for x in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':main()
