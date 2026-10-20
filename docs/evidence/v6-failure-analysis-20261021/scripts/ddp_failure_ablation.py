"""Frozen DDP failures: parent, site passthrough, and unchanged candidate."""
import copy
import json
from pathlib import Path
import sys
import time

freeze = Path('/root/autodl-tmp/dev/opop-v6-freezes/187617ce3164')
sys.path[:0] = [str(freeze/'src'),str(freeze)]
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
base = Path('/root/opop-v6-stage3/final-e2e-187617c')
out = Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021/ddp-ablation')
out.mkdir(parents=True,exist_ok=True)
for trial in ['trial-9a65eeeca9','trial-af78dd3283','trial-3c2ed50501','trial-87dde5b43d']:
    while not (out.parent/trial/'exit.json').exists():
        time.sleep(2)

rows = []
for case,trials in [('ddp-communication',['trial-92d5f02cbf','trial-d17bd1eb8f']),
                    ('ddp-step',['trial-b0a02885e2'])]:
    sources = [json.loads(sorted((base/case/'jobs').glob(t+'*-screen/run/request.json'))[0].read_text()) for t in trials]
    arms = ['parent','passthrough']+trials
    for pair in range(2):
        for arm in (arms if pair == 0 else list(reversed(arms))):
            spec = copy.deepcopy(sources[trials.index(arm)] if arm in trials else sources[0])
            params = spec['parameters']
            params['action'] = 'binding_ablation'
            params['collector_options'] = {}
            if arm == 'parent':
                params['bundles'] = []
            elif arm == 'passthrough':
                params['collector_options']['passthrough_bundles'] = [b['id'] for b in params['bundles']]
            label = f'{case}-{pair}-{arm}'
            spec['output'] = str(out/label)
            spec['wait_for_devices'] = True
            request = ExecutionRequest.model_validate(spec)
            result = LocalExecutor().run(request)
            write_json(out/(label+'.json'),result)
            row={'case':case,'pair':pair,'arm':arm,'status':result['status'],'failure':result.get('failure'),
                 'ranks':[{'rank':r['rank'],'valid':r.get('result',{}).get('valid'),
                           'metrics':r.get('result',{}).get('metrics'),
                           'checks':r.get('result',{}).get('checks'),
                           'setup':r.get('result',{}).get('setup'),
                           'coverage':r.get('result',{}).get('coverage')} for r in result.get('ranks',[])]}
            rows.append(row)
            write_json(out/'summary.json',{'scope':'Developer diagnostic, no candidate publication; frozen original source and reference',
                       'framework_commit':'187617ce316403ea627c0d33392cc020bca2cb63','runs':rows})
            print(json.dumps(row),flush=True)
            if result['status']!='passed' or not all(r['valid'] for r in row['ranks']):
                raise RuntimeError('Diagnostic failed; preserve receipt')
write_json(out/'completed.json',{'completed':True,'runs':len(rows)})
