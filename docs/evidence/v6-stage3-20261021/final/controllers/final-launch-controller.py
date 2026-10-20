from pathlib import Path
import subprocess,json,os,yaml
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
r=Path('/root/opop-v6-stage3')
head='187617ce316403ea627c0d33392cc020bca2cb63'
f=Path('/root/autodl-tmp/dev/opop-v6-freezes/187617ce3164')
environment=dict(os.environ,PYTHONPATH=str(f/'src')+':'+str(f))
output=r/'nccl-tuning-probe'
revision=fork_run('/root/opop-v6-stage2/stage2-autonomy-final-21897c1/ddp-step',output)
search=ProjectSearch(load_config(r/'deployment.yaml',[]),output)
actual=search.collect_extra(revision,['nccl_tuning'])
write_json(r/'nccl-tuning-probe-result.json',actual)
assert actual['nccl_tuning']['status']=='observed',actual['nccl_tuning']['status']
print('NCCL collector actual-worker probe passed',flush=True)
cases=json.loads((r/'intake-validation-launch.json').read_text())
records=[]
for index,row in enumerate(cases):
    name=row['case']
    args=[arg for arg in row['command'] if arg!='--intake-only']
    project=f/('examples/v6_graph_inference' if name=='graph-latency' else 'examples/v6_validation')
    args[args.index('--project')+1]=str(project)
    destination=r/'final-e2e-187617c'/name
    args[args.index('--output')+1]=str(destination)
    configfile=r/('final-'+name+'-deployment.yaml')
    settings=yaml.safe_load((r/'deployment.yaml').read_text())
    settings['opencode']['port']=15500+index
    if name.startswith('qwen'):
        settings['project_defaults']['devices']=['GPU-69c59246-e383-f1f1-1f46-5e2dfebb858d']
    elif not name.startswith('ddp'):
        settings['project_defaults']['devices']=['GPU-4bead555-f80c-df43-5a46-daa7a0ec80b2']
    configfile.write_text(yaml.safe_dump(settings));configfile.chmod(0o600)
    args[args.index('--config')+1]=str(configfile)
    args+=['--rounds','2']
    logpath=r/('final-'+name+'.log')
    with logpath.open('wb') as log:
        process=subprocess.Popen(args,cwd=f,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records.append({'case':name,'pid':process.pid,'output':str(destination),'command':args,'config':str(configfile),
        'head':head,'framework_source':framework_source(),'log':str(logpath),
        'scope':'Fresh complete pipeline from raw project/goal/command; no supplied request/adapter/groups/candidate'})
write_json(r/'final-e2e-launch.json',records)
print(json.dumps([{'case':x['case'],'pid':x['pid']} for x in records]),flush=True)
