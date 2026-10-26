"""Independent checks after the complete autonomous cohort; no feedback injection."""
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source


def ddp_reference(context):
    import torch
    from examples.v6_validation.source_fidelity import ddp_serial_reference
    from kernel_optimizer.project.evaluation import compare
    result=ddp_serial_reference(context)
    parameters=context['parameters'];request=parameters['request']
    refs=[torch.load(Path(parameters['controls'])/f'reference-rank-{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
          for rank in range(request['workload']['world_size'])]
    checks=[{'step':step,**compare(refs[0][step]['gradients'],refs[1][step]['gradients'],{'atol':0,'rtol':0})}
            for step in range(request['continuous_steps'])]
    result['exact_replica_gradients']=checks
    result['valid'] &= all(x['passed'] for x in checks)
    return result


def main():
    root=Path('/root/autodl-tmp/dev/v6-work/next-stage-20261027')
    write_json(root/'followup-protocol.json',{'at':datetime.now(timezone.utc).isoformat(),
        'framework_source':framework_source(),'scope':'Independent checks only, after autonomous DDP and its planned validation finish; no feedback into that cohort',
        'sequence':['standard Qwen B/C AB/BA, exact existing version, no publication',
                    'DDP original-source serial global-batch comparison and exact rank replica comparison'],
        'repetition':'One protocol execution; no score-dependent repeat',
        'reason':'Fixed Qwen diagnostic showed separable gain; complete its standard independent comparison. Independently audit the newly generated DDP adapter against its actual source.'})
    while not (root/'controller-completed.json').exists():time.sleep(5)
    config=load_config(root/'private-config.yaml',[])
    results={}
    try:
        source=root/'qwen-fixed';protocol=json.loads((source/'protocol.json').read_text())
        target=root/'qwen-independent'
        fork_run(source,target)
        search=ProjectSearch(config,target)
        report=promote_version(search,source,revision_id=protocol['versions']['C'],publish=False)
        results['qwen']={'status':'completed','output':str(target),'decision':report['decision'],
            'measurement':report['measurement'],'quality':report['quality'],'coverage_errors':report['coverage_errors']}
    except Exception:
        results['qwen']={'status':'error','traceback':traceback.format_exc()}
    write_json(root/'followup-result.json',results)
    try:
        target=root/'ddp-raw';prepared=json.loads((target/'prepared.json').read_text())
        request=json.loads((target/'request.json').read_text());source=Path(prepared['controls_path']).parent/'reference-source'
        run=LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(),function='ddp_reference',cwd=source,
            output=root/'ddp-source-audit',devices=request['devices'][:1],lease_dir=Path(request['lease_dir']),
            worker_timeout_s=request['worker_timeout_s'],wait_for_devices=True,
            parameters={'source':str(source),'controls':prepared['controls_path'],'request':request}))
        results['ddp_source_fidelity']={'status':run['status'],'run':run}
    except Exception:
        results['ddp_source_fidelity']={'status':'error','traceback':traceback.format_exc()}
    write_json(root/'followup-result.json',results)
    write_json(root/'followup-completed.json',{'at':datetime.now(timezone.utc).isoformat(),'results':results})
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    os.umask(0o077)
    main()
