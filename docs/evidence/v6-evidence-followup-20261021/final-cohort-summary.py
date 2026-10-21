"""Summarize completed records without changing any search decision."""
from pathlib import Path
import collections,datetime,hashlib,json

root=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
def read(path):return json.loads(path.read_text())
def provenance(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
rows=[]
for case in ['qwen-ttft','resnet-latency','ddp-communication']:
    run=root/'fresh'/case
    log=root/(case+'-fresh.log')
    assert '"status": "completed_experiment"' in log.read_text(),case
    trials=[]
    for path in sorted((run/'trials').glob('*/result.json'),key=lambda p:p.stat().st_mtime):
        value=read(path)
        screening=[]
        for observation in value.get('parameter_search',{}).get('observations',[]):
            ranks=[]
            for rank in observation.get('run',{}).get('ranks',[]):
                endpoint=rank.get('result',{})
                routes={}
                for bundle,coverage in endpoint.get('coverage',{}).items():
                    counts=collections.Counter()
                    for route in coverage.get('candidate_declared_routes',[]):
                        counts[route['route']]+=route['count']
                    routes[bundle]={'declared_routes':dict(counts),'required_sites':len(coverage.get('required',[])),
                                   'reference_invocations':sum(coverage.get('reference_invocations',{}).values())}
                ranks.append({'rank':rank['rank'],'valid':endpoint.get('valid'),'checks':endpoint.get('checks'),
                              'metrics':endpoint.get('metrics'),'coverage_summary':routes})
            screening.append({'revision':observation.get('revision'),'result':observation.get('result'),
                              'constraints':observation.get('constraints'),'ranks':ranks})
        trials.append({'source':provenance(path),'trial':value['trial'],'decision':value.get('decision'),
                       'measurement':value.get('measurement'),'screening':screening,
                       'post_collectors':{k:{field:v.get(field) for field in ['status','reason','provenance']}
                            for k,v in value.get('resources',{}).get('extra_collectors',{}).get('after',{}).items()}})
    assert len(trials)==3,(case,len(trials))
    rows.append({'case':case,'status':'completed_experiment','trials':trials,'log':provenance(log)})
independent=read(root/'fresh-independent-result.json')
native=read(root/'final-native-result.json')
assert read(root/'fresh-independent-completed.json')['completed']
assert read(root/'final-native-completed.json')['completed']
assert len(independent['runs'])==len(native['runs'])==3
assert all('error' not in r for r in independent['runs'])
assert all(r.get('status')!='error' for r in native['runs'])
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Three original raw-entry tasks, three autonomous optimization rounds each; credential-only recovery. Independent full-version and native validation are developer-started and reported separately. No decisions rewritten.',
        'framework_commit':'a6f2b6e88ec31f169acd888601fcbb32cc2c1f9c',
        'framework_source_sha256':'6346dc694f9d74c23c60f1c75d035a6c602946ac53ba5b2a3c84a68e8df477ba',
        'recovery_inputs':provenance(root/'resume-input-integrity.json'),
        'runs':rows,'independent':independent,'native':native,
        'interpretation':'Raw timing acceptance, complete correctness, native path execution and independent reproducibility are separate observations. Screening values are not paired causal optimization evidence.'}
(root/'final-cohort-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'completed_cases':len(rows),'trials':sum(len(r['trials']) for r in rows),
                  'independent':[(r['case'],r['status'],r.get('decision',{}).get('accepted')) for r in independent['runs']],
                  'native':[(r['case'],r['status']) for r in native['runs']]}))
