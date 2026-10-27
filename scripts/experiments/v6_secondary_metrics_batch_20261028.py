"""Isolated secondary metrics, fixed DDP integration and one real intake schema A/B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.intake import IntakeProposal, resolve_request
from kernel_optimizer.project.measurement import metric, valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import ddp_request, endpoint_summary
from scripts.experiments.v6_secondary_metrics_checks_20261028 import run as checks


def intake(args, config):
    import copy
    import jsonschema
    snapshot=args.previous/'ddp-step-original.intake/generation-snapshots/generation-03a5c9d61b0a.json'
    saved,prompt,inputs=load(snapshot)
    old_schema=json.loads(inputs['inputs/intake.schema.json'])
    new_schema=IntakeProposal.model_json_schema()
    reduced=copy.deepcopy(new_schema)
    reduced['$defs']['Description']['properties'].pop('diagnostic_metrics')
    reduced['$defs'].pop('DiagnosticMetric')
    assert reduced==old_schema, 'schema treatment changed more than the optional field and its definition'
    environment=json.loads(inputs['inputs/deployment.json'])
    source=Path(json.loads(inputs['inputs/repository.json'])['root'])
    goal=json.loads(inputs['inputs/intent.json'])['goal']
    root=args.output/'intake-ab'
    write_json(root/'protocol.json',{'snapshot':str(snapshot),'at':now(),'original_framework':saved['framework_source'],
        'framework_source':framework_source(),'original_input_hashes':{k:hashlib.sha256(v).hexdigest() for k,v in inputs.items()},
        'scope':'Fresh control/treatment on the same isolated prototype core, explicit shared source change versus original snapshot. '
                'Only intake.schema.json changes initially: optional diagnostic_metrics and its schema/description. '
                'Prompt, goal, source, deployment, worker environment, model/tool settings unchanged. No candidate or diagnoses supplied.'})
    rows=[]
    for arm,schema in [('control',old_schema),('treatment',new_schema)]:
        store=RunStore.create(root,arm,{'kind':'secondary-metric-intake-replay'})
        def validate(directory):
            raw=json.loads((directory/'intake.json').read_text())
            jsonschema.validate(raw,schema)
            resolve_request(IntakeProposal.model_validate(raw),source,goal,environment)
        try:
            edit=None if arm=='control' else {'path':'inputs/intake.schema.json',
                'old':inputs['inputs/intake.schema.json'].decode(),'new':json.dumps(new_schema,indent=2)}
            with Runtime(project_agent_config(config),log_dir=store.run_dir) as runtime:
                directory,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit=edit,validator=validate,allow_source_change=True)
            proposal=IntakeProposal.model_validate_json((directory/'intake.json').read_bytes())
            request=resolve_request(proposal,source,goal,environment)
            row={'arm':arm,'status':proposal.status,'agent':receipt,'proposal':proposal.model_dump(mode='json'),
                'request':request.portable() if request else None}
        except Exception:
            row={'arm':arm,'status':'error','traceback':traceback.format_exc()}
        rows.append(row);write_json(root/'result.json',{'runs':rows})
        print(json.dumps({'stage':'intake','arm':arm,'status':row['status'],'at':now()}),flush=True)
    if all('agent' in r for r in rows):
        (a,ap,af),(b,bp,bf)=[load(Path(r['agent']['generation_snapshots'][0])) for r in rows]
        changed=[k for k in sorted(set(af)|set(bf)) if af.get(k)!=bf.get(k)]
        assert changed==['inputs/intake.schema.json'] and ap==bp
        assert all(a[k]==b[k] for k in ['framework_source','model','agent','tool_model_settings'])
        write_json(root/'input-comparison.json',{'changed_inputs':changed,'same_prompt_core_model_tools':True,
            'bytes':{arm:len(f['inputs/intake.schema.json']) for arm,f in [('control',af),('treatment',bf)]}})
    return {'status':'completed' if all(r['status']=='ready' for r in rows) else 'completed_with_errors','result':str(root/'result.json')}


def fixed(args, config, *, from_intake=False):
    search=ProjectSearch(config,args.previous/'ddp-step-original');candidate=search.revisions.current()
    assert candidate.id=='09b717664af4cfdce3978a3001b1e69d4f0d7a9fda43cba20a99e6c08653b772'
    spec={'communication_kernel_sum_ms':{'source':{'collector':'torch_communication','field':'communication_kernel_sum_ms'},
        'unit':'ms','rank_aggregation':'max','aggregation_order':'ranks_then_samples'}}
    order=['off','on','on','off'];root=args.output/'fixed-ddp'
    if from_intake:
        treatment=json.loads((args.output/'intake-ab/result.json').read_text())['runs'][1]
        if not treatment.get('request') or not treatment['request'].get('diagnostic_metrics'):
            return {'status':'not_selected','reason':'Real treatment did not supply secondary metric; no manual fill-in.'}
        spec=treatment['request']['diagnostic_metrics'];order=['on'];root=args.output/'intake-metric-integration'
    write_json(root/'protocol.json',{'at':now(),'order':order,'diagnostic_metrics':spec,'candidate':candidate.id,
        'from_real_intake':from_intake,'scope':'Fixed accepted adapter/candidate/quality/step objective, only explicit secondary metric setting changes. '
        'Five clean samples then three separate profiles per selected metric. No search or acceptance; no frozen endpoint backfill. '
        'Intake integration uses only its declared metric mapping with this audited adapter, not a new original-entry evaluator cohort.'})
    rows=[]
    for index,arm in enumerate(order):
        job=root/f'{index}-{arm}'
        request=ddp_request(search,candidate,job,arm='C',action='evaluate')
        task={**request.parameters['request'],'repeats':search.request.repeats,
              'diagnostic_metrics':spec if arm=='on' else {}}
        assert not task['objective'].get('source')
        request=request.model_copy(update={'parameters':{**request.parameters,'request':task}})
        write_json(job/'protocol.json',request.model_dump(mode='json'))
        execution=LocalExecutor().run(request)
        row={'index':index,'arm':arm,'receipt':str(job/'run/receipt.json'),**endpoint_summary(execution)}
        if valid_run(execution):
            row['secondary_aggregates']={name:metric(execution,name,s['rank_aggregation'],s['aggregation_order']) for name,s in spec.items()} if arm=='on' else {}
        rows.append(row);write_json(root/'result.json',{'endpoints':rows})
        print(json.dumps({'stage':root.name,'index':index,'arm':arm,'valid':row['valid']}),flush=True)
    return {'status':'completed' if all(r['valid'] for r in rows) else 'completed_with_errors','result':str(root/'result.json')}


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','previous']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    config=load_config(args.config,[])
    write_json(args.output/'protocol.json',{'at':now(),'framework_source':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp')),
        'sequence':['CPU integration checks','fixed DDP off/on/on/off','one real intake schema control/treatment','selected metric integration if present'],
        'scope':'Isolated prototype only. Production sources/prompts unchanged, no candidate acceptance or allocator change. '
        'No API caps, bulk local transfer, Goal, delivery probes or model polling. Explicit secondary metrics only; no name-based routing.'})
    results={}
    for name,operation in [('cpu-checks',lambda:checks(args.output/'cpu-checks')),
                           ('fixed-ddp',lambda:fixed(args,config)),('intake-ab',lambda:intake(args,config)),
                           ('intake-metric',lambda:fixed(args,config,from_intake=True))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        if name!='cpu-checks' and results['cpu-checks']['status']!='completed':
            results[name]={'status':'skipped','reason':'CPU integration failed'}
        else:
            try:results[name]=operation()
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'stage':name,'status':results[name]['status']}),flush=True)
    terminal={'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':main()
