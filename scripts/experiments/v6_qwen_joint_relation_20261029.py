"""Fixed full-output relation controls for the failed joint candidate, no search."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import json
import traceback
from kernel_optimizer.runtime.execution import ExecutionRequest,LocalExecutor,write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():return datetime.now(timezone.utc).isoformat()


def inputs(parent):
    audit=read(parent/'terminal-audit.json');assert read(parent/'followup-completed.json')['phase']=='completed'
    assert audit['core']==framework_source() and audit['trial']['clean_samples']==0
    candidates=[j for j in audit['jobs']if j['action']=='evaluate'];assert len(candidates)==1
    candidate=Path(candidates[0]['request']);assert sha(candidate)==candidates[0]['request_sha256']
    previous=parent.parent/'qwen-attention-fixed-20261029'
    baseline=Path(read(previous/'followup-completed.json')['results']['0-B-observed']['request'])
    prior=read(previous/'protocol.json');observer=Path(prior['observer']);reference=prior['source_reference']
    assert sha(observer)=='a06b6c0e14495bffb68aa498c845ce66378223e9c95579b0b66a7bd463cc566c'
    assert sha(reference['path'])==reference['sha256']
    expected=dict(audit['frozen_identities_verified'])
    requests={'B':baseline,'C':candidate}
    for path in requests.values():
        req=read(path);started=read(path.parent/'rank-0-started.json')
        expected[str(path)]=sha(path)
        for name,h in started['revision']['files'].items():
            file=Path(req['cwd'])/name;assert sha(file)==h;expected[str(file)]=h
    assert all(sha(p)==h for p,h in expected.items())
    original=read(candidate);assert original['parameters']['action']=='evaluate'
    assert original['parameters']['request']==read(baseline)['parameters']['request']
    assert original['parameters']['request']['repeats']==5 and original['parameters']['request']['continuous_steps']==2
    assert original['parameters']['bundles'][0]['parameters']=={}
    assert sha(original['parameters']['bundles'][0]['file'])==audit['trial']['candidate_sha256']=='a86694e792df3baac1ea6d93e5de574fc43fa1818ebee4ce452d4f4557295178'
    return requests,observer,reference,expected,audit


def main(args):
    os.umask(0o077);requests,observer,reference,expected,audit=inputs(args.parent)
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'candidate_sha256':audit['trial']['candidate_sha256'],
            'frozen_hashes':len(expected),'observer_sha256':sha(observer),'parameters':{}}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    sequence=[('B',True),('C',False),('C',True)]
    write_json(args.output/'protocol.json',{'at':now(),'core':framework_source(),'parent':str(args.parent),
        'parent_audit_sha256':sha(args.parent/'terminal-audit.json'),'source_requests':{k:str(v)for k,v in requests.items()},
        'sequence':sequence,'observer':str(observer),'observer_sha256':sha(observer),'source_reference':reference,
        'candidate_sha256':audit['trial']['candidate_sha256'],'frozen_identities':expected,
        'disk':disk_observation(Path('/root/autodl-tmp')),
        'scope':'B-observed/C-unobserved/C-observed once each. Exact new joint candidate and original full work/quality/TTFT contract. Only output path and for O diagnostic observer/reference differ. No new Agent, tuning, discovery, reference copies, native trace, formal acceptance or pointer changes. CPU snapshots perturb host/synchronization/lifetimes. Preserve all failures/samples.'})
    results={}
    for index,(arm,observed)in enumerate(sequence):
        name=f'{index}-{arm}-'+('observed'if observed else'unobserved')
        write_json(args.output/'state.json',{'phase':'running','stage':name,'at':now()})
        request=ExecutionRequest.model_validate(read(requests[arm]));values={'output':args.output/name,'wait_for_devices':True}
        if observed:values.update(entry=observer,function='observed_worker',parameters={**request.parameters,'diagnostic_source_reference':reference['path']})
        try:
            run=LocalExecutor().run(request.model_copy(update=values));path=args.output/(name+'-result.json');write_json(path,run)
            side=args.output/name/'relation-observations.json'
            results[name]={'status':'attempt_completed'if run.get('worker_pids_reaped')and(not observed or side.exists())else'execution_error',
                'worker_status':run['status'],'result':str(path),'result_sha256':sha(path),'request':str(args.output/name/'request.json'),
                'observer_summary':str(side)if observed and side.exists()else None}
        except Exception:results[name]={'status':'execution_error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results);print(json.dumps({'endpoint':name,'at':now(),**results[name]}),flush=True)
    unchanged=all(sha(p)==h for p,h in expected.items())
    terminal={'phase':'completed'if unchanged and all(r['status']=='attempt_completed'for r in results.values())else'completed_with_errors',
        'finished_at':now(),'results':results,'all_frozen_hashes_unchanged':unchanged,
        'scope':'Completed attempts are not successful quality or formal acceptance. Timings diagnostic only.','disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
