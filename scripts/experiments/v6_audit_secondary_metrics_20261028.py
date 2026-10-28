"""Read-only verification of the isolated secondary-metric batch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics

from kernel_optimizer.project.communication import available_fields, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import metric
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_step_goal_20261028 import read, sha, endpoints
from scripts.experiments.v6_audit_ddp_tp_20261028 import normalize_parameters


def audit(root, output):
    terminal=read(root/'followup-completed.json');manifest=read(root/'overlay-manifest.json')
    frozen=Path(read(root/'launch.json')['framework']);overlay=root/'overlay'
    changes={x['path']:x for x in manifest['changes']}
    for path in (overlay/'kernel_optimizer').rglob('*'):
        if not path.is_file() or path.suffix not in ['.py','.md']:continue
        name=path.relative_to(overlay).as_posix();before=frozen/'src'/name
        if name in changes:
            assert sha(path)==changes[name]['after_sha256']
            assert (sha(before) if before.exists() else None)==changes[name]['before_sha256']
        else:assert path.read_bytes()==before.read_bytes(),name
    report={'terminal':terminal['phase'],'finished_at':terminal['finished_at'],'overlay_manifest':manifest,
        'all_source_hashes_verified':True,'endpoints':[],'intake':[],'findings':[]}
    expected_core=read(root/'protocol.json')['framework_source']
    parameters=[];sources=[]
    for folder in ['fixed-ddp','intake-metric-integration']:
        for ep in read(root/folder/'result.json')['endpoints']:
            receipt=read(ep['receipt']);checked=endpoints([{'diagnostic':receipt}])[0]
            path=Path(ep['receipt']).parent.parent;protocol=read(path/'protocol.json')
            for rank in receipt['ranks']:assert rank['framework_source']==expected_core
            normalized=normalize_parameters(protocol['parameters']);specs=normalized['request'].pop('diagnostic_metrics')
            parameters.append(normalized)
            source={p.relative_to(path/'source').as_posix():sha(p) for p in (path/'source').rglob('*')
                if p.is_file() and '__pycache__' not in p.parts};sources.append(source)
            traces=[];series={}
            for rank in receipt['ranks']:
                result=rank['result'];assert result['diagnostics']=={}
                assert all(not set(specs)&set(s) for s in result['samples'])
                if not specs:assert result['metric_samples']=={} and 'diagnostic_metrics' not in result
                for name,spec in specs.items():
                    sampled=result['diagnostic_metrics'][name]
                    assert sampled['status']=='observed' and sampled['specification']==spec
                    assert len(sampled['values'])==len(sampled['evidence'])==3
                    assert sampled['values']==result['metric_samples'][name]
                    assert statistics.median(sampled['values'])==result['metrics'][name]
                    for file,value in zip(sampled['evidence'],sampled['values']):
                        trace,window=trace_window(read(file),'opop.communication_metric_workload');assert window
                        kernels=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel']
                        recomputed=sum(e['dur'] for e in kernels if 'Collective name' in e.get('args',{}) or 'nccl' in e.get('name','').lower())/1000
                        assert abs(recomputed-value)<1e-8
                        traces.append({'rank':rank['rank'],'path':file,'sha256':sha(file),'sum_ms':value,'window':window})
                    series.setdefault(name,[]).append(sampled['values'])
            aligned={name:[max(x) for x in zip(*ranks)] for name,ranks in series.items()}
            for name,values in aligned.items():
                assert statistics.median(values)==metric(receipt,name,'max','ranks_then_samples')==ep['secondary_aggregates'][name]
            report['endpoints'].append({'folder':folder,'index':ep['index'],'arm':ep['arm'],
                'receipt_sha256':sha(ep['receipt']),'endpoint':checked,'source_hashes':source,
                'all_clean_max_rank_samples_ms':ep['max_rank_step_samples_ms'],
                'clean_median_ms':ep['median_max_rank_step_ms'],'secondary_specification':specs,
                'all_secondary_max_rank_samples':aligned,'secondary_aggregates':ep['secondary_aggregates'],'traces':traces})
    assert all(x==parameters[0] for x in parameters)
    assert all(x==sources[0] for x in sources)
    report['same_parameters_except_secondary_mapping']=True
    report['cpu_checks']=read(root/'followup-results.json')['cpu-checks']
    original=root.parent/'ddp-step-followup-20261028/ddp-step-original.intake/generation-snapshots/generation-03a5c9d61b0a.json'
    old,op,of=load(original);rows=read(root/'intake-ab/result.json')['runs']
    snapshots=[load(Path(row['agent']['generation_snapshots'][0])) for row in rows]
    (a,ap,af),(b,bp,bf)=snapshots
    assert ap==bp==op and af==of
    assert [k for k in sorted(set(af)|set(bf)) if af.get(k)!=bf.get(k)]==['inputs/intake.schema.json']
    for k in ['framework_source','model','agent','tool_model_settings']:assert a[k]==b[k]
    for k in ['model','agent','tool_model_settings']:assert a[k]==old[k]
    schema=json.loads(bf['inputs/intake.schema.json']);reduced=copy.deepcopy(schema)
    reduced['$defs'].pop('DiagnosticMetric');reduced['$defs']['Description']['properties'].pop('diagnostic_metrics')
    assert reduced==json.loads(af['inputs/intake.schema.json'])
    report['input_comparison']={'same_prompt_model_tools':True,'same_core_between_arms':True,
        'core_changed_from_original':a['framework_source']!=old['framework_source'],
        'core':a['framework_source'],'changed_initial_files':['inputs/intake.schema.json'],
        'bytes':{arm:len(files['inputs/intake.schema.json']) for arm,files in [('control',af),('treatment',bf)]}}
    for row,(_,_,files) in zip(rows,snapshots):
        report['intake'].append({'arm':row['arm'],'status':row['status'],'agent':row['agent'],
            'request':row['request'],'proposal':row['proposal'],
            'initial_input_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}})
    discovery=next((root.parent/'ddp-step-followup-20261028/ddp-step-original/jobs').rglob('discovery-rank-0.json'))
    observed=read(discovery)
    names=available_fields(observed['communication'])|{x['name'] for x in observed['capabilities']['fields'] if x['status']=='observed'}
    report['findings'].append({'kind':'resolved_discovery_name_mismatch','arm':'treatment','discovery':str(discovery),
        'sha256':sha(discovery),'available_fields':sorted(names),
        'unsupported_required_names':sorted(set(rows[1]['request']['required_observations'])-names),
        'scope':'Code-reviewed name-set replay against actual discovery; no new original-entry discovery ran in this batch.'})
    report['findings'].append({'kind':'ignored_nested_sampling_setting',
        'source':rows[1]['request']['diagnostic_metrics'],
        'actual_repeats_source':'request.profiling.metric_repeats default 3; source.profiling is forwarded but unused by TorchCommunication',
        'scope':'Actual count matches requested 3 by default, not by honoring the nested override.'})
    report['findings'].append({'kind':'scope_limit','text':'Treatment lists thread counts as source-required numerics; baseline implementation freedom is not byte-identical between fresh responses. It retains DDP/no_sync/foreach alternatives and all workload/quality/score requirements. No new adapter or optimization cohort ran.'})
    report['decision']='Runtime explicit opt-in path supported by this batch; automatic intake mapping has a demonstrated discovery-name defect. No formal candidate acceptance.'
    write_json(output,report)
    print(json.dumps({'audit':str(output),'bytes':output.stat().st_size,'endpoints':len(report['endpoints']),
        'unsupported_required_names':report['findings'][0]['unsupported_required_names']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.output)
