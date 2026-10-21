from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, time
from kernel_optimizer.project.generation_snapshot import public_settings
from kernel_optimizer.runtime.execution import DeviceLease, write_json

# NEW_KEY is supplied through SSH stdin; never persist this script with a key.
root = Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
work = root.parent
freeze = Path('/root/autodl-tmp/dev/opop-v6-freezes/a6f2b6e88ec3')
private = work/'private-runtime'/'credential-rotation-20261021'
private.mkdir(mode=0o700, exist_ok=False)
launches = json.loads((root/'fresh-launch.json').read_text())
independent = json.loads((root/'fresh-independent-launch.json').read_text())
assert not (root/'fresh-independent-result.json').exists()
rows = []
for launch in launches:
    output = Path(launch['output'])
    events = [json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
    start = [e for e in events if e['type']=='PROJECT_AGENT_STARTED'][-1]['payload']
    assert start['role'] in {'selection', 'candidate'}
    assert events[-2]['type']=='PROJECT_PROVIDER_RETRY'
    assert len(list((output/'trials').glob('*/result.json')))==2
    rows.append({**launch, 'resume_session': start['session'], 'resume_role': start['role'],
        'resume_sandbox': start['sandbox'],
        'request_sha256': hashlib.sha256((output/'request.json').read_bytes()).hexdigest()})
for name in ['fresh-launch.json', 'fresh-independent-launch.json']:
    (root/name.replace('.json','-before-key-rotation.json')).write_bytes((root/name).read_bytes())

def alive(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'

# Stop the idle waiter first, so process replacement cannot be classified as a failed experiment.
targets = [(independent['pid'], 'fresh-independent.py')]
targets += [(row['pid'], row['output']) for row in rows]
for pid, expected in targets:
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().replace('\0',' ')
    assert expected in command, (pid, expected)
    os.kill(pid, signal.SIGTERM)
    deadline=time.monotonic()+35
    while alive(pid) and time.monotonic()<deadline:time.sleep(.2)
    assert not alive(pid), f'Process did not terminate cleanly: {pid}'
    print('stopped idle controller',pid,flush=True)

configs=[work/'.opencode/opencode.jsonc']+[Path(row['resume_sandbox'])/'opencode.json' for row in rows]
changes=[]
for index,path in enumerate(configs):
    before=path.read_bytes();value=json.loads(before)
    assert value['provider']['zhipuai']['options']['baseURL']=='https://open.bigmodel.cn/api/coding/paas/v4'
    prior_public=public_settings(value)
    backup=private/f'provider-before-{index}.json';backup.write_bytes(before);backup.chmod(0o600)
    value['provider']['zhipuai']['options']['apiKey']=NEW_KEY
    assert public_settings(value)==prior_public
    tmp=path.with_name(path.name+'.rotation-tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.chmod(0o600);tmp.replace(path)
    changes.append({'path':str(path),'only_credential_changed':True,'mode':'0600'})

queue=work/'.opop-provider-coordination'
queue_key=hashlib.sha256(b'zhipuai').hexdigest()[:24]
state_path=queue/(queue_key+'.json')
with DeviceLease(queue,queue_key+'-state',wait=True):
    before=json.loads(state_path.read_text())
    write_json(root/'provider-queue-before-key-rotation.json',before)
    after={**before,'not_before':0,'failures':0,'healthy':0,'last_status':200,
        'admission':'provider_availability','credential_rotation_verified':True}
    after.pop('external_wait',None)
    write_json(state_path,after)

env=dict(os.environ,PYTHONPATH=str(freeze/'src')+':'+str(freeze))
new_launches=[]
for row in rows:
    output=Path(row['output'])
    assert hashlib.sha256((output/'request.json').read_bytes()).hexdigest()==row['request_sha256']
    command=['/root/autodl-tmp/dev-venv/bin/python','-B','-u','-m','kernel_optimizer.cli',
        '--config',str(root/(row['case']+'-fresh-private.yaml')),'optimize-project',
        '--request',str(output/'request.json'),'--output',str(output),
        '--resume-generation',row['resume_sandbox'],'--rounds','1']
    with Path(row['log']).open('ab') as log:
        proc=subprocess.Popen(command,cwd=freeze,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    new_launches.append({**row,'original_pid':row['pid'],'original_command':row['command'],
        'pid':proc.pid,'command':command,'recovery':'One interrupted third-round role in the same provider session; prior two trials retained.'})
    print('resumed',row['case'],proc.pid,row['resume_session'],flush=True)
write_json(root/'fresh-launch.json',new_launches)
with (root/'fresh-independent.log').open('ab') as log:
    proc=subprocess.Popen(['/root/autodl-tmp/dev-venv/bin/python','-B','-u',str(root/'fresh-independent.py')],
        cwd=freeze,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(root/'fresh-independent-launch.json',{'pid':proc.pid,'framework':str(freeze),'previous_pid':independent['pid']})
write_json(root/'credential-rotation-resume.json',{
    'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'User-authorized replacement credential, already verified by a real glm-5.3 HTTP200 response. Private config replacement and existing framework resume-generation, no optimization diagnosis or candidate edits supplied.',
    'framework':str(freeze),'config_changes':changes,
    'cases':[{k:v for k,v in row.items() if k in ['case','pid','original_pid','resume_session','resume_role','resume_sandbox','request_sha256','recovery']} for row in new_launches],
    'independent_pid':proc.pid,'old_provider_wait_cleared_after_successful_new_key_check':True})
print('independent waiter',proc.pid,flush=True)
