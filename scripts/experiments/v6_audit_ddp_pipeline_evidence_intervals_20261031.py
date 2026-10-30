from pathlib import Path
from collections import Counter
import json,sqlite3,statistics,subprocess
from kernel_optimizer.project.communication import trace_window
from scripts.experiments.v6_audit_ddp_observation_route_fixed_20261031 import union_duration
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from kernel_optimizer.runtime.execution import write_json
r=Path('/root/autodl-tmp/dev/v6-work/ddp-pipeline-communication-evidence-20261031');a=read(r/'terminal-audit.json');rows=[]
for kind,paths,label in [('discovery',sorted((r/'ddp-evidence/jobs').glob('discover-*/run/discovery-rank-*.json')),'opop::0'),('timeline',sorted((r/'ddp-evidence/jobs').glob('timeline-*/run/timeline-analysis-rank-*.json')),'opop.timeline_workload')]:
 for p in paths:
  z=read(p); t=read(z['trace']); sliced,window=trace_window(t,label); assert window is not None
  ks=[e for e in sliced['traceEvents']if e.get('cat')=='kernel'and e.get('ph')=='X']
  cs=[e for e in ks if 'nccl'in e['name'].lower()];ns=[e for e in ks if e not in cs]
  ci=[(e['ts'],e['ts']+e['dur'])for e in cs];ni=[(e['ts'],e['ts']+e['dur'])for e in ns]
  cu=union_duration(cs);nu=union_duration(ns);overlap=(cu+nu-union_duration(cs+ns))/1000
  assert len(cs)==8 and abs(sum(e['dur']for e in cs)/1000-z['communication']['communication_kernel_sum_ms'])<1e-8
  assert abs(overlap-z['communication']['compute_overlap_ms'])<1e-7
  comm=z['communication'];payloads=Counter(c['logical_input_bytes']for c in comm['collectives'])
  assert payloads=={6144:1,56702976:6,157535232:1} and sum(k*v for k,v in payloads.items())==497759232
  rows.append({'kind':kind,'path':str(p),'trace_sha256':sha(z['trace']),'window':window,'kernels':len(ks),'nccl_names':dict(Counter(e['name']for e in cs)),'nccl_streams':dict(Counter(str(e['args'].get('stream'))for e in cs)), 'graph_kernels':sum(bool(e['args'].get('graph node id'))for e in ks),'graph_launches':sum('cudaGraphLaunch'in e.get('name','')for e in sliced['traceEvents']), 'nccl_sum_ms':sum(e['dur']for e in cs)/1000,'union_ms':cu/1000,'overlap_ms':overlap,'payloads':dict(payloads)})
cal=[]
for i in range(3):
 pair=[x['result']['rows'][i]for x in a['calibration']]; aligned=[max(v)for v in zip(*(x['samples_ms']for x in pair))]
 cal.append({'bytes':pair[0]['logical_bytes'],'aligned_rankmax_samples_ms':aligned,'rankmax_before_median_ms':statistics.median(aligned)})
native=[{'rank':n['rank'],'sum_ms':sum(c['sum_ms']for c in n['nccl']),'sha256':n['sha256']}for n in a['native']]
record={'base_audit_sha256':sha(r/'terminal-audit.json'),'traces':rows,'calibration_rankmax':cal,'native_nccl_sums':native,'scope':'Independent raw event union sweep on each existing workload window, no GPU/profiling rerun. Synthetic aligned samples are not model timing or wire lower bounds.'}
assert not (r/'numerical-extra.json').exists()
write_json(r/'numerical-extra.json',record)
print(json.dumps({'sha256':sha(r/'numerical-extra.json'),'traces':len(rows),'calibration_rankmax':cal,'native_nccl_sums':native}))
