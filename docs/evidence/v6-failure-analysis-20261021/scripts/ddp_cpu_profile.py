"""Independent diagnostic CPU profiles of the already quality-checked arms."""
import json
from pathlib import Path
import sys
import time
root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
while not (root/'qwen-math/result.json').exists() or not json.loads((root/'qwen-math/result.json').read_text()).get('completed'):
    time.sleep(2)
freeze=Path('/root/autodl-tmp/dev/opop-v6-freezes/187617ce3164')
sys.path[:0]=[str(freeze/'src'),str(freeze)]
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
out=root/'ddp-cpu';out.mkdir(exist_ok=True)
rows=[]
for name in ['ddp-communication-0-parent','ddp-communication-0-passthrough','ddp-communication-0-trial-92d5f02cbf',
             'ddp-communication-0-trial-d17bd1eb8f','ddp-step-0-trial-b0a02885e2']:
    spec=json.loads((root/'ddp-ablation'/name/'request.json').read_text())
    spec['output']=str(out/name)
    spec['parameters']['action']='python_runtime_profile'
    result=LocalExecutor().run(ExecutionRequest.model_validate(spec))
    write_json(out/(name+'.json'),result)
    rows.append({'name':name,'status':result['status'],'failure':result.get('failure')})
    write_json(out/'summary.json',rows)
    if result['status']!='passed':raise RuntimeError(result.get('failure'))
write_json(out/'completed.json',{'completed':True})
