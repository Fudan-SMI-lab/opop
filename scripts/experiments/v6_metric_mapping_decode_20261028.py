"""One schema-clarity replay and a same-adapter Qwen decode-goal experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import metric, paired_decision, valid_run
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_autonomous_cohort_20261027 import independent
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import ddp_request, endpoint_summary


NOTE = (
    "Discovery requirements, output aliases, and sampling configuration are different interfaces. "
    "required_observations uses exact field names from the discovery/capability manifest, not "
    "collector-qualified paths or evaluator metric aliases; a numeric collector field required in "
    "discovery uses its field name without inventing a collector prefix. source is forwarded unchanged "
    "to the provider and is not framework configuration. The evaluator obtains the number of profiled "
    "repetitions from deployment.profiling.metric_repeats (default 3), not source.profiling. "
    "Preserve the source repetition count; if it differs from available sampling configuration, report "
    "the missing setup rather than declaring an ignored nested override."
)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intake_overlay(repo, destination):
    shutil.copytree(repo/'src/kernel_optimizer',destination/'kernel_optimizer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=destination/'kernel_optimizer/project/intake.py';old=path.read_text();text=old
    edits=[('class Description(BaseModel):','from kernel_optimizer.project.metric_diagnostics import DiagnosticMetric, DESCRIPTION, validate_sources\n\n\nclass Description(BaseModel):'),
        ('    required_observations: list[str] = Field(default_factory=list)',
         '    required_observations: list[str] = Field(default_factory=list)\n    diagnostic_metrics: dict[str, DiagnosticMetric] = Field(default_factory=dict, description=DESCRIPTION)'),
        ('    return ProjectRequest(source_root=source,',
         '    validate_sources({k:v.model_dump() for k,v in d.diagnostic_metrics.items()}, environment["profiling"], d.objective.model_dump())\n    return ProjectRequest(source_root=source,'),
        ('        communication=d.communication, required_observations=d.required_observations,',
         '        communication=d.communication, required_observations=d.required_observations, diagnostic_metrics=d.diagnostic_metrics,')]
    for before,after in edits:
        assert text.count(before)==1; text=text.replace(before,after)
    path.write_text(text);compile(text,str(path),'exec')
    return {'path':str(path),'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':sha(path)}


def intake_replay(args, config):
    import jsonschema
    from kernel_optimizer.project.intake import IntakeProposal, resolve_request
    original=read(args.previous/'intake-ab/result.json')['runs'][1]
    snapshot=Path(original['agent']['generation_snapshots'][0]);saved,prompt,inputs=load(snapshot)
    old_schema=json.loads(inputs['inputs/intake.schema.json']);assert IntakeProposal.model_json_schema()==old_schema
    schema=copy.deepcopy(old_schema)
    schema['$defs']['Description']['properties']['diagnostic_metrics']['description'] += ' '+NOTE
    environment=json.loads(inputs['inputs/deployment.json']);source=Path(json.loads(inputs['inputs/repository.json'])['root'])
    goal=json.loads(inputs['inputs/intent.json'])['goal'];root=args.output/'intake-ab'
    write_json(root/'protocol.json',{'at':now(),'snapshot':str(snapshot),'source_core':saved['framework_source'],
        'replay_core':framework_source(),'change':NOTE,'scope':'One added generic paragraph in diagnostic_metrics schema description only. '
        'Same isolated intake core for both fresh arms; common explicit source change from old snapshot. No answers or previous diagnoses supplied.'})
    rows=[]
    for arm,schema_value in [('control',old_schema),('treatment',schema)]:
        store=RunStore.create(root,arm,{'kind':'metric-interface-schema-replay'})
        def validate(directory):
            raw=read(directory/'intake.json');jsonschema.validate(raw,schema_value)
            resolve_request(IntakeProposal.model_validate(raw),source,goal,environment)
        try:
            edit=None if arm=='control' else {'path':'inputs/intake.schema.json','old':inputs['inputs/intake.schema.json'].decode(),'new':json.dumps(schema,indent=2)}
            with Runtime(project_agent_config(config),log_dir=store.run_dir) as runtime:
                generated,receipt=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit=edit,validator=validate,allow_source_change=True)
            proposal=IntakeProposal.model_validate_json((generated/'intake.json').read_bytes())
            request=resolve_request(proposal,source,goal,environment)
            row={'arm':arm,'status':proposal.status,'agent':receipt,'proposal':proposal.model_dump(mode='json'),'request':request.portable() if request else None}
        except Exception:row={'arm':arm,'status':'error','traceback':traceback.format_exc()}
        rows.append(row);write_json(root/'result.json',{'runs':rows})
    if all('agent' in r for r in rows):
        (a,ap,af),(b,bp,bf)=[load(Path(r['agent']['generation_snapshots'][0])) for r in rows]
        changed=[k for k in sorted(set(af)|set(bf)) if af.get(k)!=bf.get(k)]
        assert changed==['inputs/intake.schema.json'] and ap==bp
        assert all(a[k]==b[k] for k in ['framework_source','model','agent','tool_model_settings'])
        write_json(root/'input-comparison.json',{'changed_inputs':changed,'same_prompt_core_model_tools':True,
            'bytes':{arm:len(f['inputs/intake.schema.json']) for arm,f in [('control',af),('treatment',bf)]}})
    return {'status':'completed' if all(r['status']=='ready' for r in rows) else 'completed_with_errors','result':str(root/'result.json')}


def intake_subprocess(args):
    repo=Path(__file__).resolve().parents[2];overlay=args.output/'intake-overlay'
    manifest=intake_overlay(repo,overlay);write_json(args.output/'intake-overlay-manifest.json',manifest)
    argv=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--mode','intake','--config',str(args.config),
        '--output',str(args.output),'--previous',str(args.previous),'--qwen-parent',str(args.qwen_parent)]
    env={**os.environ,'PYTHONPATH':str(overlay)+':'+str(repo/'src')+':'+str(repo)}
    with (args.output/'intake-controller.log').open('wb') as log:
        result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
    return {'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode}


def ddp_execution(args, config, label, settings, action):
    parent=args.previous.parent/'ddp-step-followup-20261028/ddp-step-original'
    search=ProjectSearch(config,parent);revision=search.revisions.current()
    assert revision.id=='09b717664af4cfdce3978a3001b1e69d4f0d7a9fda43cba20a99e6c08653b772'
    job=args.output/label
    request=ddp_request(search,revision,job,arm='C',action=action)
    task={**request.parameters['request'],'repeats':search.request.repeats,**settings}
    request=request.model_copy(update={'parameters':{**request.parameters,'request':task}})
    write_json(job/'protocol.json',request.model_dump(mode='json'));result=LocalExecutor().run(request)
    row={'status':'completed' if valid_run(result) else 'completed_with_errors','receipt':str(job/'run/receipt.json'),
        'scope':'Existing frozen adapter/candidate with selected collection settings only; no new evaluator, search or formal acceptance.'}
    if action=='evaluate':
        row.update(endpoint_summary(result))
        row['execution_status']=row.pop('status');row['status']='completed' if row['valid'] else 'completed_with_errors'
        if row['valid']:
            row['secondary']={name:metric(result,name,spec['rank_aggregation'],spec['aggregation_order'])
                              for name,spec in settings.get('diagnostic_metrics',{}).items()}
    write_json(job/'result.json',row);return row


def runtime_smoke(args,config):
    spec={'communication_kernel_sum_ms':{'source':{'collector':'torch_communication','field':'communication_kernel_sum_ms'},
        'unit':'ms','rank_aggregation':'max','aggregation_order':'ranks_then_samples'}}
    return ddp_execution(args,config,'production-runtime-smoke',{'diagnostic_metrics':spec},'evaluate')


def collection_integration(args, config):
    rows=[]
    for arm in read(args.output/'intake-ab/result.json')['runs']:
        if not arm.get('request'):
            rows.append({'arm':arm['arm'],'status':'not_run'});continue
        # Exercise the actual declared discovery names as well as numeric sampling.
        settings={k:arm['request'][k] for k in ['communication','required_observations','profiling']}
        settings['diagnostic_metrics']=arm['request'].get('diagnostic_metrics',{})
        for action in ['profile','evaluate']:
            try:result=ddp_execution(args,config,'mapping-'+arm['arm']+'-'+action,settings,action)
            except Exception:result={'status':'error','traceback':traceback.format_exc()}
            rows.append({'arm':arm['arm'],'action':action,**result})
    write_json(args.output/'collection-integration.json',{'runs':rows})
    return {'status':'completed' if all(r['status']=='completed' for r in rows) else 'completed_with_errors','result':str(args.output/'collection-integration.json')}


def decode_request(parent):
    old=read(parent/'request.json');request=copy.deepcopy(old)
    request['goal']='Maximize decode_tokens_per_second for the same interactive Qwen3-4B workload: batch 1, prompt 256 tokens, complete greedy generation of 32 new tokens. Retain the original model, inputs, BF16/SDPA/noTF32 policies, full logits tolerance, exact tokens, state lifetimes and both completion boundaries. Decode throughput counts the 31 post-first-token steps; prefill and all 32 tokens remain required. TTFT and end-to-end metrics remain diagnostics. No new quality or performance constraints.'
    request['objective']={'name':'decode_tokens_per_second','unit':'tokens/s','direction':'maximize','constraints':copy.deepcopy(old['objective']['constraints'])}
    request['devices']=['0']
    assert 'this request optimizes only ttft_ms' in request['notes'][1]
    request['notes'][1]=request['notes'][1].replace('this request optimizes only ttft_ms','this goal variant optimizes only decode_tokens_per_second')
    assert 'peak_allocated_bytes and decode metrics' in request['notes'][8]
    request['notes'][8]=request['notes'][8].replace('peak_allocated_bytes and decode metrics','peak_allocated_bytes and ttft_ms')
    assert {k for k in old if old[k]!=request[k]}=={'goal','objective','devices','notes'}
    return request


def qwen_variant(args, config):
    request=decode_request(args.qwen_parent);path=args.output/'qwen-decode-request.json';write_json(path,request)
    argv=[sys.executable,'-B','-m','kernel_optimizer.cli','--config',str(args.config),'optimize-project',
        '--request',str(path),'--reuse-evaluator',str(args.qwen_parent),'--output',str(args.output/'qwen-decode'),'--rounds','3']
    write_json(args.output/'qwen-variant-protocol.json',{'at':now(),'argv':argv,'parent':str(args.qwen_parent),
        'parent_request_sha256':sha(args.qwen_parent/'request.json'),'request':request,'framework_source':framework_source(),
        'scope':'Same-adapter same-workload goal variant; fresh real contract adaptation and controls, only baseline revision reused. '
                'No optimized TTFT bundles or mapping replay answers enter the three-round search. All formal results remain separate.'})
    with (args.output/'qwen-controller.log').open('wb') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    return {'status':'completed' if result.returncode==0 else 'error','returncode':result.returncode}


def qwen_independent(args,config):
    path=args.output/'qwen-decode/prepared.json'
    if path.exists() and read(path)['status']=='prepared':
        assert read(path)['adapter_sha256']==read(args.qwen_parent/'prepared.json')['adapter_sha256']=='24ebefd7e2b6e3f04d1cdb547bbdecfbfbbb7e2575812107cd2d951a318172ad'
    spec=read(args.qwen_parent.parent/'protocol.json')['specifications'][0]
    return independent(args,config,{**spec,'name':'qwen-decode'})


def cross_goals(args):
    rows=[]
    goals={'ttft':read(args.qwen_parent/'request.json')['objective'],'decode':read(args.output/'qwen-decode-request.json')['objective']}
    for label,run in [('ttft',args.qwen_parent),('decode',args.output/'qwen-decode')]:
        results=list(run.with_name(run.name+'-independent').glob('version-trials/*/result.json'))
        if len(results)!=1:
            rows.append({'goal':label,'status':'unavailable','reason':'No unique independent accepted-version evaluation'});continue
        source=read(results[0]);anchor=read(run/'prepared.json')['evaluation']
        rows.append({'goal':label,'source_result':str(results[0]),'sha256':sha(results[0]),'original_decision':source['decision'],
            'cross_scores':{name:paired_decision(source['pairs'],objective,anchor) for name,objective in goals.items()}})
    write_json(args.output/'cross-goals.json',{'versions':rows,'scope':'Diagnostic rescoring only, separate measurement periods/anchors. No imported cross-contract bundles, altered acceptance or causal ranking between versions.'})
    return {'status':'completed','result':str(args.output/'cross-goals.json')}


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ['config','output','previous','qwen-parent']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',default='batch',choices=['batch','intake']);args=p.parse_args();os.umask(0o077)
    args.output.mkdir(parents=True,exist_ok=True);config=load_config(args.config,[])
    if args.mode=='intake':
        result=intake_replay(args,config);write_json(args.output/'intake-stage-result.json',result)
        raise SystemExit(0 if result['status']=='completed' else 1)
    write_json(args.output/'protocol.json',{'at':now(),'framework_source':framework_source(),'disk':disk_observation(Path('/root/autodl-tmp')),
        'sequence':['production opt-in DDP smoke','isolated intake schema paragraph A/B','actual collection-settings integration','Qwen same-adapter decode goal three rounds','independent source/full-version AB/BA','cross-metric diagnostic'],
        'scope':'Production intake/prompt unchanged. One Qwen cohort, serial GPU jobs, all samples retained, no score-dependent retries or API caps. '
                'No Goal, bulk local transfers, delivery probes or model polling.'})
    results={}
    for name,operation in [('runtime-smoke',lambda:runtime_smoke(args,config)),('intake-ab',lambda:intake_subprocess(args)),
        ('collection-integration',lambda:collection_integration(args,config)),('qwen-decode',lambda:qwen_variant(args,config)),
        ('qwen-independent',lambda:qwen_independent(args,config)),('cross-goals',lambda:cross_goals(args))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        if name!='runtime-smoke' and results['runtime-smoke']['status']!='completed':
            results[name]={'status':'not_run','reason':'Production runtime smoke failed'}
        else:
            try:results[name]=operation()
            except Exception:results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results);print(json.dumps({'stage':name,'status':results[name]['status'],'at':now()}),flush=True)
    terminal={'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors',
              'finished_at':now(),'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':main()
