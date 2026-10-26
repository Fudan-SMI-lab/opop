import json
import sys
import traceback
from pathlib import Path
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
from kernel_optimizer.project.nsys import command_prefix,export

root=Path('/root/autodl-tmp/dev/box43-preflight-20261027')
entry=root/'gpu-probe.py'
entry.write_bytes(Path('.scratch/box43_gpu_probe.py').read_bytes())
nsys='/root/autodl-tmp/tools/nsys-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys'
result={}
for name,devices,backend in [('communication',['1','0'],'nccl'),('extension',['0'],None),('nsys',['1'],None)]:
    output=root/'tools'/name
    env={'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,GRAPH,P2P,SHM,NET',
         'NCCL_DEBUG_FILE':str(output/'nccl.%h.%p.log')} if backend else {}
    try:
        run=LocalExecutor().run(ExecutionRequest(entry=entry,cwd=root,devices=devices,
            distributed_backend=backend, output=output,lease_dir=Path('/root/autodl-tmp/dev/v6-work/device-leases'),
            env=env,parameters={'action':name},worker_timeout_s=600,wait_for_devices=True,
            launcher=command_prefix(nsys) if name=='nsys' else []))
        row={'run':run,'passed':run['status']=='passed' and all(r.get('result',{}).get('valid') for r in run['ranks'])}
        if name=='nsys':
            row['export']=export(nsys,output,[0])
            row['passed'] &= all(r['status']=='observed' and r['cuda_kernels']>0 for r in row['export'])
        result[name]=row
    except Exception:
        result[name]={'passed':False,'error':traceback.format_exc()}
    write_json(root/'tools-result.json',result)
    print(name,result[name]['passed'],flush=True)
