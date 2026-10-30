"""Read-only startup identities/activity, not an outcome or full access audit."""
import argparse
from datetime import datetime,timezone
import hashlib,json,sqlite3,subprocess
from pathlib import Path
from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha

def digest(raw):return hashlib.sha256(raw).hexdigest()

def main(root):
    protocol=read(root/'protocol.json');config=load_config(root/'private-config.yaml',[])
    source=ProjectSearch(config,Path(protocol['source']));search=ProjectSearch(config,root/'ddp-search')
    assert all(sha(path)==value for path,value in protocol['frozen_identities'].items())
    assert framework_source()==protocol['core'] and source.revisions.current().id==protocol['parent']
    assert source.request==search.request and source.prepared==search.prepared
    assert not source.revisions.current().bundles and not search.revisions.current().bundles
    assert not (search.output/'latest-feedback.json').exists()
    assert (search.output/'request.json').read_bytes()==(source.output/'request.json').read_bytes()
    jobs=[]
    for job in sorted((search.output/'jobs').iterdir()):
        request_path=job/'run/request.json'
        if not request_path.exists():continue
        request=read(request_path)
        assert request['parameters']['request']==search.request.portable()
        assert request['parameters']['reference_dir']==search.prepared['controls_path']
        assert request['parameters']['bundles']==[]
        assert sha(request['parameters']['adapter'])==source.prepared['adapter_sha256']
        manifest=read(Path(request['cwd'])/'.opop-revision.json')
        assert all(sha(Path(request['cwd'])/name)==value for name,value in manifest['files'].items())
        starts=[]
        for path in sorted((job/'run').glob('rank-*-started.json')):
            started=read(path);assert started['parameters_ref']['request_sha256']==sha(request_path)
            assert all(started['revision']['files'][name]==value for name,value in manifest['files'].items())
            starts.append({'path':str(path),'sha256':sha(path),'rank':started['rank']})
        receipt_path=job/'run/receipt.json';receipt=read(receipt_path)if receipt_path.exists()else None
        if receipt:
            for rank in receipt['ranks']:
                assert rank['framework_source']==protocol['core']
                assert rank['binding']['torch_uuid']==search.request.devices[rank['rank']]
        jobs.append({'name':job.name,'request_sha256':sha(request_path),'action':request['parameters']['action'],
            'entry_sha256':sha(request['entry']),'manifest_sha256':sha(Path(request['cwd'])/'.opop-revision.json'),
            'started':starts,'status':receipt['status']if receipt else'running','receipt_sha256':sha(receipt_path)if receipt else None})
    roles=[];private=root/'startup-private-history';private.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    for path in sorted((search.output/'generation-snapshots').glob('*.json')):
        meta,prompt,files=load(path)
        if not meta['fresh_session']:continue
        sandbox=Path(meta['sandbox']);assert meta['framework_source']==protocol['core']
        assert all((sandbox/name).read_bytes()==raw for name,raw in files.items())
        assert not any('previous-candidate/'in name for name in files)
        history={};db.execute('BEGIN')
        for table in ['session','message','part']:
            key='id'if table=='session'else'session_id';order='id'if table=='session'else'time_created,id'
            cursor=db.execute('select * from '+table+' where '+key+'=? order by '+order,(meta['session'],))
            names=[x[0]for x in cursor.description];rows=[]
            for row in cursor:
                item=dict(zip(names,row))
                if isinstance(item.get('data'),str):item['data']=json.loads(item['data'])
                rows.append(item)
            history[table]=rows
        db.rollback();history_path=private/(meta['session']+'.json');write_json(history_path,history)
        tools=[]
        for row in history['part']:
            part=row['data']
            if part.get('type')=='tool':
                state=part.get('state',{});tools.append({'part':row['id'],'tool':part.get('tool'),'status':state.get('status'),
                    'input':state.get('input'),'output':state.get('output'),'metadata':state.get('metadata')})
        tools_path=private/(meta['session']+'-tools.json');write_json(tools_path,tools)
        roles.append({'role':meta['role'],'generation':meta['id'],'session':meta['session'],'snapshot_sha256':sha(path),
            'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'all_initial_bytes_equal_snapshot':True,
            'input_hashes':{name:digest(raw)for name,raw in files.items()},'prompt_sha256':digest(prompt.encode()),
            'settings_sha256':digest(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()),
            'no_historical_candidate_files':True,'fresh_session':True,
            'assistant_messages':sum(row['data'].get('role')=='assistant'for row in history['message']),
            'tools':len(tools),'private_history_sha256':sha(history_path),'private_tools_sha256':sha(tools_path)})
    db.close()
    discovery_path=search.output/'discovery.json';discovery=read(discovery_path)if discovery_path.exists()else None
    record={'at':datetime.now(timezone.utc).isoformat(),'state':read(root/'state.json'),'core':protocol['core'],
        'frozen_files_verified':len(protocol['frozen_identities']),'request_prepared_equal_current_device_source':True,
        'source_current':source.revisions.current().id,'branch_current':search.revisions.current().id,'bundles':[],
        'controls_path':search.prepared['controls_path'],'jobs':jobs,'roles':roles,
        'discovery':None if discovery is None else{'sha256':sha(discovery_path),'revision':discovery['revision'],
            'calls':len(discovery['calls']),'groups':len(discovery['groups']),
            'cross_rank':discovery['communication']['cross_rank']},
        'controller':subprocess.run(['ps','-p',str(read(root/'launch.json')['pid']),'-o','pid,ppid,sid,stat,etime'],capture_output=True,text=True).stdout,
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),
        'scope':'Startup identity/activity only; discovery and role execution do not establish candidate quality, communication gain or acceptance. Full later tool/parameter/result/trace audit pending. External filesystem nonhermetic; no claim of complete access review from this snapshot.'}
    write_json(root/'startup-verification.json',record)
    print(json.dumps({'sha256':sha(root/'startup-verification.json'),'bytes':(root/'startup-verification.json').stat().st_size,
        'at':record['at'],'frozen':record['frozen_files_verified'],'jobs':[{k:x[k]for k in ['name','action','status']}for x in jobs],
        'roles':[{k:r[k]for k in ['role','generation','session','initial_files','initial_bytes','assistant_messages','tools']}for r in roles],
        'branch_current':record['branch_current'],'df':record['df'],'gpu_workers':record['gpu_workers']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
