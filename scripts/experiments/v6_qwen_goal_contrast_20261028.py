"""Same-core, interleaved fixed Qwen goal contrast; no new optimization acceptance."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation

ORDER = ['BT-1', 'T-1', 'D-1', 'BD-1', 'BD-2', 'D-2', 'T-2', 'BT-2']
ADAPTER = '24ebefd7e2b6e3f04d1cdb547bbdecfbfbbb7e2575812107cd2d951a318172ad'


def now():
    return datetime.now(timezone.utc).isoformat()


def equal_observations(a, b, path='root'):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), path
        return {'tensors': 1, 'elements': a.numel()}
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        children = [equal_observations(a[k], b[k], path+'.'+str(k)) for k in a]
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        children = [equal_observations(x, y, path+'.'+str(i)) for i,(x,y) in enumerate(zip(a,b,strict=True))]
    else:
        assert a == b, path
        children = []
    return {k: sum(c[k] for c in children) for k in ['tensors', 'elements']}


def preflight(args, config):
    import torch
    searches = {key: ProjectSearch(config, path) for key,path in [('T',args.ttft),('D',args.decode)]}
    requests = {k:read(s.output/'request.json') for k,s in searches.items()}
    changed = sorted(k for k in requests['T'].keys()|requests['D'].keys() if requests['T'].get(k)!=requests['D'].get(k))
    assert changed == ['devices','goal','notes','objective']
    assert not requests['T']['objective']['constraints'] and not requests['D']['objective']['constraints']
    baseline_files = []; references = {}; identities = {}; contracts = {}
    for key,s in searches.items():
        current=s.revisions.current(); baseline=s.revisions.revision(s.prepared['revision'])
        assert current.bundles and not baseline.bundles
        assert s.prepared['adapter_sha256'] == ADAPTER
        controls=Path(s.prepared['controls_path']); reference=controls/'reference-rank-0.pt'
        assert sha(controls.parent/'reference-source/opop_adapter.py') == ADAPTER
        baseline_files.append({f.path:f.sha256 for f in baseline.files})
        contracts[key]=s.revisions.load(current.contract_ref,ProjectContract).model_dump(mode='json')
        assert current.contract_ref == baseline.contract_ref
        independent=list(s.output.with_name(s.output.name+'-independent').glob('version-trials/*/result.json'))
        assert len(independent)==1 and read(independent[0])['decision']['accepted']
        identities[key]={'source':str(s.output),'current':current.id,'baseline':baseline.id,
            'bundles':list(current.bundles),'files':{f.path:f.sha256 for f in current.files},
            'contract_ref':current.contract_ref,'request_sha256':sha(s.output/'request.json'),
            'controls_path':str(controls),'reference_sha256':sha(reference),
            'prior_independent':str(independent[0]),'prior_independent_sha256':sha(independent[0])}
        references[key]=torch.load(reference,map_location='cpu',weights_only=False)
    assert baseline_files[0] == baseline_files[1]
    for field in ['workload','quality','completion','state_protocol','world_size','frozen_files','allowed_scopes']:
        assert contracts['T'][field] == contracts['D'][field],field
    equal = equal_observations(references['T'],references['D'])
    return {'at':now(),'core':framework_source(),'order':ORDER,'identities':identities,
        'baseline_files':baseline_files[0],'changed_request_keys':changed,'reference_tensor_equality':equal,
        'contract_differences':{k:{arm:contracts[arm][k] for arm in contracts} for k in contracts['T'] if contracts['T'][k]!=contracts['D'][k]},
        'deployment_override':{'devices':['0'],'scope':'Only device selection in memory; frozen requests and pointers unchanged.'},
        'scope':'Fixed previously accepted versions under their own original contracts. Same current core, source bytes and adapter. No cross-contract imports, new candidate search, formal acceptance or prompt adoption. No new API call is needed for fixed execution. All original five samples per endpoint retained; separate baseline under each contract, interleaved reverse order. No new trace collection or reference copies.'}


def main(args):
    os.umask(0o077); config=load_config(args.config,[])
    protocol=preflight(args,config)
    if args.preflight:
        write_json(args.preflight,protocol)
        print(json.dumps({'preflight':str(args.preflight),'reference_equality':protocol['reference_tensor_equality']}));return
    assert not (args.output/'protocol.json').exists()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol['disk']=disk_observation(Path('/root/autodl-tmp'));write_json(args.output/'protocol.json',protocol)
    results={}; endpoints={}; branches={}; identities=protocol['identities']
    try:
        for key,parent in [('BT',args.ttft),('T',args.ttft),('D',args.decode),('BD',args.decode)]:
            identity=identities['T' if key in ['T','BT'] else 'D']
            revision=identity['baseline' if key.startswith('B') else 'current']
            fork_run(parent,args.output/key,revision_id=revision)
            search=ProjectSearch(config,args.output/key)
            search.request=search.request.model_copy(update={'devices':['0']})
            branches[key]=search
        for label in ORDER:
            write_json(args.output/'state.json',{'phase':'running','stage':label,'at':now()})
            key=label.split('-')[0]; search=branches[key]
            try:
                run=search.execute(search.revisions.current(),label,'evaluate')
                path=args.output/(label+'.json');write_json(path,run);endpoints[label]=run
                valid=valid_run(run)
                results[label]={'status':'completed' if valid else 'worker_or_quality_failure','path':str(path),'sha256':sha(path)}
                if valid:
                    assert len(run['ranks'])==1
                    result=run['ranks'][0]['result']
                    assert len(result['samples'])==5 and result['math_policy_unchanged']
                    assert run['ranks'][0]['framework_source']==protocol['core']
                    results[label]['metrics']=result['metrics']
            except Exception:results[label]={'status':'error','traceback':traceback.format_exc()}
            write_json(args.output/'followup-results.json',results)
            print(json.dumps({'endpoint':label,'status':results[label]['status'],'at':now()}),flush=True)
        if all(results[n]['status']=='completed' for n in ORDER):
            # Scores are diagnostic descriptions of this fixed interleaving only.
            comparisons={'T_vs_D':[{'parent':endpoints['T-'+i],'candidate':endpoints['D-'+i]} for i in ['1','2']],
                'BT_vs_T':[{'parent':endpoints['BT-'+i],'candidate':endpoints['T-'+i]} for i in ['1','2']],
                'BD_vs_D':[{'parent':endpoints['BD-'+i],'candidate':endpoints['D-'+i]} for i in ['1','2']]}
            scores={name:{goal:paired_decision(pairs,branches[key].request.objective,branches[key].prepared['evaluation'])
                for goal,key in [('ttft','T'),('decode','D')]} for name,pairs in comparisons.items()}
            write_json(args.output/'diagnostic-scores.json',{'scores':scores,
                'scope':'Diagnostic comparison only. No promotion, original result rewrite or same-generation causal attribution. Generation histories and goals differ; measurement core and window are now shared.'})
            results['contrast']={'status':'completed','path':str(args.output/'diagnostic-scores.json')}
        else:results['contrast']={'status':'not_run','reason':'At least one fixed endpoint failed; no retry or partial acceptance.'}
        for key,parent in [('T',args.ttft),('D',args.decode)]:
            check=ProjectSearch(config,parent)
            assert check.revisions.current().id==identities[key]['current']
            assert sha(parent/'request.json')==identities[key]['request_sha256']
            assert sha(Path(identities[key]['controls_path'])/'reference-rank-0.pt')==identities[key]['reference_sha256']
    except Exception:
        results['controller']={'status':'error','traceback':traceback.format_exc()}
    write_json(args.output/'followup-results.json',results)
    terminal={'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors',
        'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','output','ttft','decode']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    main(parser.parse_args())
