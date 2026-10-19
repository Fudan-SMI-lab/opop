"""Bounded real probes for the 2026-10-20 preparation, not optimization benchmarks."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/dev/v6-work/preflight-20261020')
CODE = Path('/root/autodl-tmp/dev/opop-v6')
NSYS = '/root/autodl-tmp/tools/nsys-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys'


def run(name, command, timeout, env=None):
    started = time.time()
    with (ROOT / f'{name}.log').open('w') as out:
        child = subprocess.Popen(command, cwd=ROOT, env=env or os.environ.copy(),
                                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
    row = {'name':name,'command':command,'exit_code':code,'started_unix':started,
           'elapsed_seconds':time.time()-started,'timeout_seconds':timeout}
    with (ROOT / 'status.jsonl').open('a') as out:
        out.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    return code


def idle():
    result = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if result.strip():
        raise RuntimeError('GPU processes are active; preflight did not acquire the device pair')


def main():
    os.umask(0o077)
    os.environ.update({'PYTHONPATH':f'{CODE}/src:{CODE}','PYTHONDONTWRITEBYTECODE':'1',
        'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'1',
        'TMPDIR':str(ROOT),'TRITON_CACHE_DIR':str(ROOT.parent/'cache/triton')})
    sys.path[:0] = [str(CODE/'src'),str(CODE)]
    import kernel_optimizer
    assert Path(kernel_optimizer.__file__).resolve().is_relative_to(CODE)
    (ROOT/'runtime.json').write_text(json.dumps({'source':kernel_optimizer.__file__,
        'python':sys.version,'executable':sys.executable,
        'packages':{p:md.version(p) for p in ['torch','triton','transformers','optuna','pydantic']}},indent=2))
    idle()
    run('gpu-before',['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],10)
    run('topology',['nvidia-smi','topo','-m'],10)
    run('p2p-read',['nvidia-smi','topo','-p2p','r'],10)
    run('p2p-write',['nvidia-smi','topo','-p2p','w'],10)
    run('nsys-version',[NSYS,'--version'],10)
    run('ncu-version',['/usr/local/cuda/bin/ncu','--version'],10)
    run('opencode-version',['/usr/local/bin/opencode','--version'],15)
    env = os.environ.copy()
    env.update({'NCCL_DEBUG':'INFO','NCCL_DEBUG_SUBSYS':'INIT,GRAPH,P2P,SHM,NET',
                'NCCL_DEBUG_FILE':str(ROOT/'nccl-%h-%p.log')})
    launch = [sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc-per-node=2']
    run('nccl-latency',launch+[str(ROOT/'probe.py'),'latency',str(ROOT)],60,env)
    idle()
    run('torch-profiler',launch+[str(ROOT/'probe.py'),'profile',str(ROOT)],60,env)
    idle()
    base = [NSYS,'profile','--sample=none','--cpuctxsw=none']
    if run('nsys-capture',base+['--trace=cuda,nvtx','--output='+str(ROOT/'communication')]+launch+[str(ROOT/'comm_probe.py'),'trace',str(ROOT)],90,env)==0:
        run('nsys-export',[NSYS,'export','--type=sqlite','--output='+str(ROOT/'communication.sqlite'),str(ROOT/'communication.nsys-rep')],45)
    idle()
    run('gpu-metrics',base+['--trace=cuda','--gpu-metrics-devices=all','--gpu-metrics-frequency=1000',
        '--output='+str(ROOT/'gpu-metrics'),sys.executable,'-B',str(ROOT/'metrics_probe.py')],20)
    idle()
    run('ncu-counter',['/usr/local/cuda/bin/ncu','--target-processes','all','--launch-count','1',
        '--metrics','sm__cycles_elapsed.avg',sys.executable,'-B',str(ROOT/'probe.py'),'ncu',str(ROOT)],30)
    idle()
    run('pcie-idle',['nvidia-smi','dmon','-s','t','-d','1','-c','2'],6)
    run('pcie-traffic',launch+[str(ROOT/'comm_probe.py'),'traffic',str(ROOT)],30,env)
    idle()
    single = os.environ.copy()
    single['CUDA_VISIBLE_DEVICES']='1'
    config = str(ROOT.parent/'configs/box42-v6.yaml')
    cli = [sys.executable,'-B','-m','kernel_optimizer.cli','--config',config]
    run('doctor',cli+['doctor'],120,single)
    run('agent-smoke',cli+['agent-smoke','--module','generator','--task','level1:19'],3700,single)
    idle()
    run('gpu-after',['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu','--format=csv'],10)
    (ROOT/'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z'))


if __name__=='__main__':
    main()
