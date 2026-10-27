"""Separate fixed-version diagnostics, then fresh original-entry step optimization."""
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_autonomous_cohort_20261027 import autonomous, independent
from scripts.experiments.v6_box43_next_stage_20261027 import now
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation
from scripts.experiments.v6_ddp_tp_diagnostics_20261028 import ddp_request, endpoint_summary


def fixed(args, config, *, profile):
    search = ProjectSearch(config, args.previous/'ddp-step')
    candidate = search.revisions.current(); baseline = search.revisions.revision(search.prepared['revision'])
    assert candidate.id == '3c0b8d6225206094e75872f98c88d05594c6e04e6287f4072b4fdc2b1725e3f4'
    root = args.output/('fixed-path' if profile else 'fixed-communication')
    order = ['B','C'] if profile else ['B','C','C','B']
    source = json.loads((args.previous.parent/'ddp-semantics-20261027/ddp-communication/request.json').read_text())['objective']
    write_json(root/'protocol.json', {'at': now(), 'order': order, 'candidate': candidate.id, 'baseline': baseline.id,
        'action': 'profile' if profile else 'evaluate', 'clean_repeats': search.request.repeats,
        'communication_metric_repeats': 3, 'diagnostic_objective': None if profile else source,
        'scope': 'Frozen versions, not optimization or new acceptance. Native discovery is intrusive and separate from plain metric profiles. '
                 'Communication endpoints explicitly select the retained collector source only for this diagnostic request. '
                 'No frozen contract/request/result overwritten; all clean samples and raw traces retained.'})
    rows = []
    for i, arm in enumerate(order):
        job = root/f'{i}-{arm}'
        request = ddp_request(search, baseline if arm=='B' else candidate, job, arm='C', action='profile' if profile else 'evaluate')
        task = {**request.parameters['request'], 'repeats': search.request.repeats}
        if not profile:
            task['objective'] = source
            task['profiling'] = {**task.get('profiling', {}), 'metric_repeats': 3}
        request = request.model_copy(update={'parameters': {**request.parameters, 'request': task}})
        write_json(job/'protocol.json', request.model_dump(mode='json'))
        result = LocalExecutor().run(request)
        row = {'index': i, 'arm': arm, 'status': result['status'], 'receipt': str(job/'run/receipt.json')}
        if profile:
            row['discoveries'] = [r.get('result') for r in result['ranks']]
        else:
            row.update(endpoint_summary(result))
            if row['valid']:
                values = [r['result']['metric_samples']['communication_kernel_sum_ms'] for r in result['ranks']]
                assert all(len(v)==3 for v in values)
                row['max_rank_communication_samples_ms'] = [max(v[i] for v in values) for i in range(3)]
                row['median_communication_ms'] = statistics.median(row['max_rank_communication_samples_ms'])
        rows.append(row); write_json(root/'result.json', {'endpoints': rows})
        print(json.dumps({'stage': root.name, 'arm': arm, 'index': i, 'status': row['status']}), flush=True)
    return {'status': 'completed' if all(r['status']=='passed' and r.get('valid',True) for r in rows) else 'completed_with_errors',
            'result': str(root/'result.json')}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config','output','previous']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); os.umask(0o077); args.output.mkdir(parents=True,exist_ok=True)
    config = load_config(args.config,[])
    spec = {'name': 'ddp-step-original', 'case': 'ddp', 'devices': ['0','1'],
        'command': 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp-step --profile',
        'goal': "Minimize maximum-rank clean step_ms for the source's complete accumulated training step, taking aligned rank maxima before the median over repeats. Retain the original step_ms <= 1.05x original-baseline constraint, all workload, precision, full quality and exact rank-replica requirements and the completion boundary. Communication kernel duration remains a separately profiled diagnostic, not wire time or the primary score."}
    project = Path(__file__).resolve().parents[2]/'examples/v6_validation'
    write_json(args.output/'protocol.json', {'at': now(),'framework_source':framework_source(), 'specification':spec,
        'sequence':['fixed native B/C paths','fixed B/C/C/B communication diagnostics','fresh original-entry DDP step goal three rounds','source/negative/full-version independent checks'],
        'scope':'No supplied adapter, bundle, groups or developer diagnostics in original-entry task. Production unchanged. '
                'Fixed diagnostics never overwrite prior independent acceptance. New adapter source checks may require manual observation-schema mapping; '
                'execution failure or naming mismatch is not automatically numerical infidelity or acceptance. '
                'No API caps, score-dependent repeats, Goal, delivery probes or bulk local transfer.',
        'disk':disk_observation(Path('/root/autodl-tmp'))})
    results={}
    for name, operation in [('fixed-path',lambda:fixed(args,config,profile=True)),
                            ('fixed-communication',lambda:fixed(args,config,profile=False)),
                            ('ddp-step-original',lambda:autonomous(args,config,spec,project)),
                            ('ddp-independent',lambda:independent(args,config,spec))]:
        write_json(args.output/'state.json',{'at':now(),'phase':'running','stage':name})
        try:
            results[name]=operation()
        except Exception:
            results[name]={'status':'error','traceback':traceback.format_exc()}
        write_json(args.output/'followup-results.json',results)
        print(json.dumps({'stage':name,'status':results[name]['status']}),flush=True)
    terminal={'finished_at':now(),'phase':'completed' if all(r['status']=='completed' for r in results.values()) else 'completed_with_errors',
              'results':results,'disk':disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    main()
