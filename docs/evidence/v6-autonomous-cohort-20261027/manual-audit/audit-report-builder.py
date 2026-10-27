from pathlib import Path
import ast
import difflib
import hashlib
import json
import sqlite3
from collections import Counter
from kernel_optimizer.project.generation_snapshot import load

root = Path('/root/autodl-tmp/dev/v6-work/autonomous-cohort-20261027')
out = Path('/root/autodl-tmp/dev/v6-work/cohort-schema-audit-20261027')
reports = {}
for name in ['qwen-ttft', 'resnet-latency', 'ddp-communication']:
    run = root/name
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    order = [e['payload']['session'] for e in events if e['type']=='PROJECT_AGENT_FINISHED' and e['payload']['role']=='candidate']
    report = {'request': json.loads((run/'request.json').read_text()), 'trials': [],
              'revision_events': [e for e in events if e['type'] in ['PROJECT_DISCOVERED','PROJECT_COMMITTED','PROJECT_ABORTED']]}
    for path in sorted((run/'trials').glob('*/result.json')):
        value = json.loads(path.read_text())
        row = {k:value[k] for k in ['trial','selection','proposal','measurement','decision','candidate_agent','selection_agent']}
        row['round'] = order.index(value['candidate_agent']['session'])+1
        row.update(source_result=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row['endpoints'] = []
        for pair_index, pair in enumerate(value.get('pairs', [])):
            for arm, endpoint in pair.items():
                row['endpoints'].append({'pair': pair_index, 'arm': arm, 'status': endpoint['status'],
                    'ranks': [{'rank':rank['rank'], 'checks':rank['result'].get('checks'),
                               'metrics':rank['result'].get('metrics'), 'samples':rank['result'].get('samples'),
                               'metric_samples':rank['result'].get('metric_samples'), 'coverage':rank['result'].get('coverage'),
                               'framework_source':rank.get('framework_source')} for rank in endpoint['ranks']]})
        for role in ['candidate_agent','selection_agent']:
            receipt = value[role]
            snapshot = next((Path(p) for p in receipt['generation_snapshots'] if json.loads(Path(p).read_text())['fresh_session']), None)
            if snapshot:
                manifest, prompt, inputs = load(snapshot)
                previous = inputs.get('inputs/previous-feedback.txt')
                row[role+'_input'] = {'snapshot': str(snapshot), 'framework_source':manifest['framework_source'],
                    'feedback_file_bytes':len(previous) if previous else 0,
                    'feedback_sha256':hashlib.sha256(previous).hexdigest() if previous else None,
                    'previous_trial_mentions': sorted(set(__import__('re').findall(r'trial-[a-z0-9]{10}', value['proposal']['rationale']))) if role=='candidate_agent' else []}
        after = value['resources']['extra_collectors'].get('after', {})
        row['collection_plan'] = value['resources'].get('collection_plan')
        row['collector_status'] = {key:item.get('status') for key,item in after.items()}
        if 'nsys' in after:
            candidate_file = Path(value['candidate_agent']['sandbox'])/'candidate.py'
            tree = ast.parse(candidate_file.read_text())
            jit_names = [node.name for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
                         and any('jit' in ast.unparse(dec) for dec in node.decorator_list)]
            native = []
            for rank in after['nsys'].get('ranks', []):
                kernels = rank.get('kernels', [])
                counts = Counter(k['name'] for k in kernels)
                record = {'rank':rank['rank'], 'sqlite':rank['evidence'], 'kernel_count':len(kernels),
                    'graph_kernel_count':sum(bool(k.get('graphNodeId')) for k in kernels),
                    'jit_function_matches':{func:sum(n for key,n in counts.items() if key.startswith(func)) for func in jit_names},
                    'selected_kernel_names':[{'name':key,'count':n} for key,n in counts.items()
                                             if any(key.startswith(func) for func in jit_names)]}
                with sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True) as con:
                    tables={x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if {'StringIds','CUPTI_ACTIVITY_KIND_RUNTIME'} <= tables:
                        record['graph_runtime_calls']=[{'name':n,'count':c} for n,c in con.execute(
                            "SELECT s.value,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY s.value")]
                record['sqlite_sha256']=hashlib.sha256(Path(rank['evidence']).read_bytes()).hexdigest()
                native.append(record)
            row['native']=native
        report['trials'].append(row)
    report['trials'].sort(key=lambda r:r['round'])
    for row in report['trials']:
        revision=json.loads((run/'project/revisions'/(row['decision']['revision']+'.json')).read_text())
        row['installed_bundles']=revision['bundles']
    receipts=[e['payload'] for e in events if e['type']=='PROJECT_AGENT_FINISHED']
    receipts += [json.loads((root/(name+'.intake')/'result.json').read_text())['agent']]
    report['usage']={'sessions':len(receipts),'reported_cost':sum(r['usage']['reported_cost'] for r in receipts),
                     'roles':dict(Counter(r['role'] for r in receipts)),'scope':'Provider estimates only, no API cap.'}
    independent=json.loads(next((root/(name+'-independent')/'version-trials').glob('*/result.json')).read_text())
    report['independent']={k:independent[k] for k in ['revision','parent','quality','coverage_errors','measurement','decision','publication']}
    report['independent']['endpoints']=[{'pair':i,'arm':arm,'ranks':[{'rank':r['rank'],'result':{k:r['result'].get(k) for k in ['checks','samples','metric_samples','coverage']}} for r in endpoint['ranks']]}
                                        for i,pair in enumerate(independent['pairs']) for arm,endpoint in pair.items()]
    reports[name]=report

ab=json.loads((root/'collection-ab/result.json').read_text())
snapshots=[load(Path(row['agent']['generation_snapshots'][0])) for row in ab['runs']]
(left,lp,lf),(right,rp,rf)=snapshots
comparison={'same_framework':left['framework_source']==right['framework_source'],'same_model':left['model']==right['model'],
            'same_tool_settings':left['tool_model_settings']==right['tool_model_settings'],
            'changed_input_files':[key for key in sorted(set(lf)|set(rf)) if lf.get(key)!=rf.get(key)],
            'prompt_diff':list(difflib.unified_diff(lp.splitlines(),rp.splitlines(),lineterm=''))}
(out/'collection-input-comparison.json').write_text(json.dumps(comparison,indent=2))
(out/'cohort-audit.json').write_text(json.dumps(reports,indent=2))
print(json.dumps({'projects':{name:{'rounds':[(r['round'],r['trial'],r['decision']['accepted'],len(r['installed_bundles'])) for r in report['trials']],
    'native':[{'round':r['round'],'summary':r.get('native')} for r in report['trials'] if r['decision']['accepted']],
    'usage':report['usage']} for name,report in reports.items()},'collector_inputs_identical':not comparison['changed_input_files'],
    'report_bytes':(out/'cohort-audit.json').stat().st_size},indent=2))
