"""Derive a compact, source-linked table from terminal experimental records."""
from pathlib import Path
import datetime,json
root=Path('/root/autodl-tmp/dev/v6-work/agent-repair-20261021')
def read(p):return json.loads(p.read_text())
def outcome(d):
    return {k:d.get(k) for k in ['trial','decision','measurement','quality','coverage_errors'] if k in d}
repairs=[]
for case in ['qwen-ttft','qwen-throughput','ddp-step','ddp-communication']:
    for row in read(root/case/'result.json')['runs']:
        p=Path(row['output'])/'trials'/row.get('trial','missing')/'result.json'
        repairs.append({'case':case,'arm':row['arm'],'source':str(p),
            **(outcome(read(p)) if p.exists() else {'error':row.get('error'),'decision':row.get('decision')})})
fresh=[]
for name in ['fresh-final-launch.json','fresh-fixed-launch.json']:
    data=read(root/name)
    if isinstance(data,dict):data=[data]
    for launch in data:
        b=Path(launch['output']);log=Path(launch['log']).read_text(errors='replace')
        fresh.append({'case':launch['case'],'source':str(b),'head':launch['head'],
            'completed_two_rounds':'"status": "completed_experiment"' in log,
            'trials':[{'source':str(p),**outcome(read(p))} for p in sorted((b/'trials').glob('*/result.json'),key=lambda p:p.stat().st_mtime)]})
independent={}
for name in ['independent-validation','fresh-final-independent','fresh-fixed-independent']:
    p=root/(name+'-result.json')
    if p.exists():independent[name]=[{'case':r.get('case'),'arm':r.get('arm'),'mode':r.get('mode'),
        'status':r.get('status'),'error':r.get('error'),'source':r.get('output'),**outcome(r)} for r in read(p)['runs']]
summary={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Derived result table; original immutable records and endpoints remain authoritative. Assisted repairs, raw-entry autonomous runs, and developer-started independent checks are separate.',
    'repairs':repairs,'fresh':fresh,'independent':independent,
    'continuation':{p.name:read(p) for p in [root/'ddp-automatic-continuation-result.json',
        root/'ddp-automatic-continuation-independent-result.json'] if p.exists()}}
(root/'final-summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'repairs':len(repairs),'fresh':len(fresh),'completed':sum(r['completed_two_rounds'] for r in fresh)}))
