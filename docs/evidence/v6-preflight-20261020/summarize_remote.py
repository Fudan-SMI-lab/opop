"""Export only allowlisted, non-secret evidence from the real preflight."""
import json
from pathlib import Path
import sqlite3
import subprocess

R = Path('/root/autodl-tmp/dev/v6-work/preflight-20261020')


def union(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    return sum(b-a for a,b in merged)


def main():
    status=[json.loads(x) for x in (R/'status.jsonl').read_text().splitlines()]
    summary={'scope':'Real environment and provider preflight, not general C3 acceptance or optimization gains',
             'preparation':json.loads((R/'preparation.json').read_text()),
             'runtime':json.loads((R/'runtime.json').read_text()),
             'checks':status,'completed':(R/'DONE').exists()}
    summary['historical_integrity']=json.loads((R/'historical-integrity.json').read_text())
    summary['latency']=json.loads((R/'latency.json').read_text())
    summary['torch_profiler']=[]
    for rank in range(2):
        rows=json.loads((R/f'windows-rank{rank}.json').read_text())
        raw=[]
        for row in rows:
            events=json.loads(Path(row['raw_trace']).read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
            raw.append({'window':row['window'],'kernel_count':len(kernels),
                        'nccl_count':sum('nccl' in e.get('name','').lower() for e in kernels)})
        summary['torch_profiler'].append({'rank':rank,'windows':len(rows),
            'cuda_windows':sum(x['kernel_count']>0 for x in raw),
            'nccl_windows':sum(x['nccl_count']>0 for x in raw),'raw_windows':raw})
    with sqlite3.connect(f'file:{R}/communication.sqlite?mode=ro',uri=True) as con:
        markers=con.execute("SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value) LIKE 'box4_rank%_capture_step%' ORDER BY n.start").fetchall()
        kernels=con.execute('SELECT k.start,k.end,k.deviceId,k.globalPid,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
        result=[]
        for rank in range(2):
            identity=json.loads((R/f'trace-rank{rank}.json').read_text())
            comm=[];compute=[];c_count=0;k_count=0
            for a,b,tid,label in markers:
                if not label.startswith(f'box4_rank{rank}_'):continue
                assert ((tid>>24)&0xffffff)==identity['pid']
                selected=[k for k in kernels if ((k[3]>>24)&0xffffff)==identity['pid'] and a<=k[0]<=k[1]<=b]
                assert all(k[2]==rank for k in selected)
                c=[(k[0],k[1]) for k in selected if 'nccl' in k[5].lower()]
                k=[(k[0],k[1]) for k in selected if 'nccl' not in k[5].lower()]
                comm.extend(c);compute.extend(k);c_count+=len(c);k_count+=len(k)
            result.append({'rank':rank,'pid':identity['pid'],'correctness':identity['correctness'],
                'communication_kernel_count':c_count,'compute_kernel_count':k_count,
                'communication_union_us':union(comm)/1000,'compute_union_us':union(compute)/1000,
                'overlap_union_us':(union(comm)+union(compute)-union(comm+compute))/1000})
        summary['nsys']={'marked_windows':len(markers),'ranks':result,
            'scope':'PID/device and time-enclosed diagnostic windows; not general operator attribution or causal critical path'}
    summary['actual_channel_routes']=sorted({line.split(' via ',1)[1] for p in R.glob('nccl-*.log')
        for line in p.read_text().splitlines() if 'Channel ' in line and ' via ' in line})
    summary['pcie']={'scope':'device-wide 1-second MB/s samples, not per-operator wire bytes',
        'idle':(R/'pcie-idle.log').read_text(),'load':(R/'dmon-rank0.log').read_text()}
    summary['counter_probe']={name:{'permission_denied':'ERR_NVGPUCTRPERM' in (R/f'{name}.log').read_text(),
        'raw_log':f'{name}.log'} for name in ['ncu-counter','gpu-metrics']}
    summary['agent_smoke']={'status_lines':[line for line in (R/'agent-smoke.log').read_text().splitlines()
        if line.startswith(('got ','run dir:'))]}
    summary['cgroup']={p:Path(p).read_text().strip() for p in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.high','/sys/fs/cgroup/cpu.max']}
    summary['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
    summary['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv'],text=True)
    (R/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ['completed','nsys','actual_channel_routes','counter_probe','agent_smoke','gpu_state']},indent=2))


if __name__=='__main__':
    main()
