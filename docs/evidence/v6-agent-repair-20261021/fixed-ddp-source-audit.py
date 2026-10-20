from pathlib import Path
import json
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021');b=root/'fresh-fixed/ddp-communication';prepared=json.loads((b/'prepared.json').read_text());request=json.loads((b/'request.json').read_text());controls=Path(prepared['controls_path']);source=controls.parent/'reference-source'
result=LocalExecutor().run(ExecutionRequest(entry=root/'source-audit-microloss-worker.py',function='audit',cwd=source,output=root/'fresh-fixed-source-audit/ddp-communication',devices=request['devices'][:1],lease_dir=request['lease_dir'],wait_for_devices=True,worker_timeout_s=request['worker_timeout_s'],parameters={'source':str(source),'controls':str(controls),'request':request}))
write_json(root/'fresh-fixed-source-audit/ddp-communication.json',result)
row={'case':'ddp-communication','status':result['status'],'valid':bool(result.get('ranks')) and all(r.get('result',{}).get('valid') for r in result.get('ranks',[])),'method':'Independent original-source global-batch updates, full parameters/gradients/loss, all additional normalized per-rank micro-loss outputs and exact replica gradients'}
write_json(root/'fresh-fixed-source-audit-result.json',{'runs':[row]});print(json.dumps(row),flush=True)
