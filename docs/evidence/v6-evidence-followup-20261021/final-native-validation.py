from pathlib import Path
import datetime,hashlib,json,sqlite3,time,traceback
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
root=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
rows=[];done=set()
while len(done)<3:
 source=root/'fresh-independent-result.json'
 reports=json.loads(source.read_text())['runs'] if source.exists() else []
 for report in reports:
  case=report['case']
  if case in done:continue
  done.add(case)
  row={'case':case,'scope':'Developer native-path check of the exact full final version staged by independent validation. Separate intrusive profiling; no generation, candidate edits or publication. Timing from this trace is not optimization performance.'}
  try:
   if report.get('quality')!='pass' or not report.get('bundles'):
    row.update(status='skipped',reason='No quality-passing complete candidate in the independent result',independent_status=report.get('status'),independent_error=report.get('error'))
   else:
    config=load_config(root/(case+'-fresh-private.yaml'),[])
    out=Path(report['output']);search=ProjectSearch(config,out)
    revision=search.revisions.revision(report['decision']['revision'])
    result=search.collect_extra(revision,['nsys'])
    raw=out/'final-native.json';write_json(raw,result)
    value=result['nsys'];ranks=[]
    for rank,evidence in enumerate(value.get('ranks',[])):
     if evidence.get('status')!='observed':
      ranks.append({'rank':rank,'status':evidence.get('status'),'reason':evidence.get('reason')});continue
     path=Path(evidence['evidence']);db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
     kernels=db.execute('SELECT s.value,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
     launches=db.execute("SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY s.value").fetchall()
     selected={}
     for name,graph in kernels:
      if any(term in name.lower() for term in ['nccl','flash','fmha','bias_relu','pool']):
       key=(name,bool(graph));selected[key]=selected.get(key,0)+1
     ranks.append({'rank':rank,'status':'observed','sqlite':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'graph_launch_apis':dict(launches),'kernels':len(kernels),'graph_node_kernels':sum(bool(g) for _,g in kernels),'nccl_kernels':sum('nccl' in n.lower() for n,_ in kernels),'selected_kernels':[{'name':k[0],'graph_node':k[1],'count':v} for k,v in sorted(selected.items())]})
     db.close()
    row.update(status=value['status'],revision=revision.id,bundles=list(revision.bundles),raw=str(raw),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),provenance=value.get('provenance'),ranks=ranks)
  except Exception as exc:row.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
  rows.append(row);write_json(root/'final-native-result.json',{'runs':rows})
  print(json.dumps({k:v for k,v in row.items() if k not in ['provenance','ranks']}),flush=True)
 time.sleep(5)
write_json(root/'final-native-completed.json',{'completed':True,'runs':len(rows),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
