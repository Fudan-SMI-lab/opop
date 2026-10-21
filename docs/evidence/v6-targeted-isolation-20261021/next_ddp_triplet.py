from pathlib import Path
import json
from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.measurement import valid_run,metric,paired_decision
from kernel_optimizer.runtime.execution import write_json

old=Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021')
root=Path('/root/autodl-tmp/dev/v6-work/targeted-isolation-20261021')
source=old/'fresh/ddp-communication';out=root/'ddp-triplet'
candidate_id='03c5e5abcdf4f4cf519e4e909e2b48a3d8e54414dd924a31dabd04f5f3878fbb'
parent_id='30c61d369b619ea2ef7756a5f654344ad37e8e99cd3debc69aee3d68613ae531'
bundle_id='trial-recovered-0f41339f0a-param-afbf38f4'
parent=fork_run(source,out,revision_id=parent_id)
search=ProjectSearch(load_config(old/'ddp-communication-fresh-private.yaml',[]),out)
candidate=search.revisions.revision(candidate_id)
protocol={'scope':'Developer fixed-candidate diagnostic; no generation, candidate changes, publication, or contract relaxation.',
          'source':str(source),'parent':parent.id,'candidate':candidate.id,'bundle':bundle_id,
          'order':['parent','passthrough','candidate','candidate','passthrough','parent'],
          'arms':'Same original adapter and references; passthrough uses existing binding_ablation without candidate build/computation. Candidate is exact previously rejected barrier implementation.',
          'measurement':'Fresh workers per endpoint, original two continuous checks, two warmups, five clean samples and three separate communication samples. Single fixed six-endpoint block; not repeated until pass.'}
write_json(out/'protocol.json',protocol)
rows=[];pairs=[{},{}]
for index,arm in enumerate(protocol['order']):
    pair=0 if index<3 else 1
    revision=parent if arm=='parent' else candidate
    options={'passthrough_bundles':[bundle_id]} if arm=='passthrough' else {}
    name=f'triplet-{index}-{arm}'
    run=search.execute(revision,name,'binding_ablation',collector_options=options)
    row={'index':index,'pair':pair,'arm':arm,'run':run,'receipt':str(out/'jobs'/name/'run/receipt.json')}
    rows.append(row)
    write_json(out/'result.json',{'protocol':protocol,'runs':rows})
    assert valid_run(run),(index,arm,run.get('failure'))
    if arm!='passthrough':pairs[pair][arm]=run
    print(json.dumps({'index':index,'arm':arm,'step_ms':metric(run,'step_ms'),
                      'communication_kernel_sum_ms':metric(run,'communication_kernel_sum_ms')}),flush=True)
anchor=search.prepared['evaluation']
decision=paired_decision(pairs,search.request.objective,anchor)
summary=[]
for pair in range(2):
    endpoints={r['arm']:r['run'] for r in rows if r['pair']==pair}
    values={arm:{name:metric(run,name) for name in ['step_ms','communication_kernel_sum_ms']} for arm,run in endpoints.items()}
    summary.append({'pair':pair,'values':values,'passthrough_minus_parent_ms':values['passthrough']['step_ms']-values['parent']['step_ms'],
                    'candidate_minus_parent_ms':values['candidate']['step_ms']-values['parent']['step_ms'],
                    'candidate_minus_passthrough_ms':values['candidate']['step_ms']-values['passthrough']['step_ms']})
write_json(out/'result.json',{'protocol':protocol,'runs':rows,'summary':summary,'original_contract_diagnostic_decision':decision,
    'interpretation':'Six endpoint triplet embeds parent/candidate AB/BA but includes intervening passthrough endpoints. Differences include runtime state and noise; they are not stable subtractable costs. Original decision is unchanged.'})
write_json(out/'completed.json',{'completed':True,'endpoints':len(rows)})
print(json.dumps({'completed':True,'summary':summary,'decision':decision}),flush=True)
