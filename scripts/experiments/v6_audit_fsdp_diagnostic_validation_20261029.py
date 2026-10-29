"""Read-only audit of exact generated FSDP diagnostic execution, not acceptance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import statistics
import torch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import endpoint


def flatten(value, path='output'):
    if isinstance(value,dict):
        return {p:v for k,x in value.items() for p,v in flatten(x,path+'.'+k).items()}
    if isinstance(value,(list,tuple)):
        return {p:v for i,x in enumerate(value) for p,v in flatten(x,f'{path}[{i}]').items()}
    return {path:value}


def audit(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json')
    assert terminal['phase']=='completed' and terminal['all_frozen_hashes_unchanged']
    assert framework_source()==protocol['core']
    identities=protocol['frozen_identities'];assert all(sha(p)==h for p,h in identities.items())
    source=read(protocol['original_request']);task=source['parameters']['request'];rows=[]
    controls=Path(source['parameters']['reference_dir'])
    baseline=endpoint(read(root/'derived-baseline-observed-result.json'),task)
    for arm in ['derived-baseline-observed','candidate-observed']:
        request=read(root/arm/'request.json')
        expected={**source,'entry':protocol['generated']['probe'],'function':'run','output':str(root/arm),'wait_for_devices':True,
            'parameters':{**source['parameters'],'diagnostic_context':{'original_entry':source['entry'],
                'original_function':source['function'],'independent_reference':None,'arm':arm}}}
        if arm=='derived-baseline-observed':expected['parameters']['bundles']=[]
        assert request==expected
        run=read(root/(arm+'-result.json'));assert run['status']=='passed' and run['worker_pids_reaped']
        assert run['entry_sha256']==sha(request['entry'])==protocol['generated']['probe_sha256']
        for rank in [0,1]:
            worker=read(root/arm/f'rank-{rank}.json');v=worker['result'];d=read(root/arm/f'diagnostic-rank-{rank}.json')
            assert worker['status']=='passed' and worker['framework_source']==protocol['core'] and v['math_policy_unchanged']
            assert d['invocation']['called_exactly_once'] and d['original_invocation']['status']=='returned'
            assert d['nonfinite_values_encoded']==0 and d['independent_reference']is None
            assert all(d['instrumentation'][k]['restored']for k in ['compare_observer','dist_counter'])
            mirror=d['original_invocation']['mirror']
            assert mirror['valid']==v['valid'] and mirror['samples_count']==len(v['samples'])
            assert mirror['metric_names']==sorted(v['metrics'])
            # The generated mirror intentionally selects check fields, omitting
            # atol/rtol; full originals are separately retained in compare_calls.
            for mc,vc in zip(mirror['checks'],v['checks'],strict=True):
                assert mc=={k:vc[k] for k in mc}
            if arm=='candidate-observed':
                old=read(Path(protocol['original_request']).parent/f'rank-{rank}.json')['result']
                assert v['checks']==old['checks'] and v['valid'] is False and not v['samples'] and not v['metrics']
                assert 'diagnostic_metrics'not in v
            else:
                assert v['valid'] and len(v['samples'])==5 and all(c['passed'] and c['max_abs_error']==0 for c in v['checks'])
            assert len(d['compare_calls'])==len(v['checks'])==2
            reference=torch.load(controls/f'reference-rank-{rank}.pt',map_location='cpu',weights_only=False)
            assert isinstance(reference,list) and len(reference)==2
            calls=[]
            for i,c in enumerate(d['compare_calls']):
                original={k:x for k,x in c['original'].items()if k!='raised'}
                assert original=={k:x for k,x in v['checks'][i].items()if k!='continuous_step'}
                f=c['full_traversal'];paths=f['paths'];ref=flatten(reference[i]);assert len(ref)==len(paths)==587
                tensors=[p for p in paths if p['kind']=='tensor'];assert len(tensors)==586
                assert sum(p['numel']for p in tensors)+1==(361257090 if rank==0 else 361255042)
                assert not f['structural_mismatches'] and all(p.get('all_finite_difference',True)for p in paths)
                fp=hashlib.sha256();top_rows=[]
                for p in paths:
                    b=ref[p['path']]
                    if torch.is_tensor(b):
                        assert p['shape']==list(b.shape) and p['dtype']==str(b.dtype) and p['numel']==b.numel()
                        for t in p.get('top_mismatches',[]):
                            assert float(b.flatten()[t['flat_index']])==t['expected']
                            error=float((torch.tensor(t['actual'],dtype=b.dtype)-torch.tensor(t['expected'],dtype=b.dtype)).abs())
                            assert error==t['abs_error']
                            top_rows.append({'path':p['path'],**t,'outside_tolerance':error>task['quality']['atol']+task['quality']['rtol']*abs(t['expected'])})
                    else:assert p['kind']=='scalar_exact' and p['equal'] and b is True
                    fp.update(json.dumps([p['path'],p.get('torch_equal',p.get('exact_equal',p.get('equal'))),p.get('sha256_actual_bytes'),
                        p.get('max_abs_error'),p.get('mismatch_allclose_count',p.get('mismatch_count')),p.get('structure')],sort_keys=True).encode())
                assert fp.hexdigest()==f['call_fingerprint_sha256']
                assert f['paths_equal']==sum(p['torch_equal']for p in tensors)
                assert f['paths_exceeding_tolerance']==sum(p.get('mismatch_allclose_count',0)>0 for p in tensors)
                assert f['max_abs_error_overall']==max(p['max_abs_error']for p in tensors)
                categories={}
                for name in ['loss','logits','gradients','parameters']:
                    ps=[p for p in tensors if p['path'].split('.')[1]==name]
                    categories[name]={'paths':len(ps),'nonexact_paths':sum(not p['torch_equal']for p in ps),
                        'max_abs_error':max(p['max_abs_error']for p in ps),'mismatch_elements':sum(p.get('mismatch_allclose_count',0)for p in ps)}
                calls.append({'step':i,'original':c['original'],'categories':categories,
                    'path_count':len(paths),'full_comparator_and_traversal_wall_s':f['observer_wall_s'],
                    'top_values_checked_against_reference':top_rows,'collective_api_delta':c['collectives_since_previous_compare'],
                    'summary_fingerprint_recomputed':True,'actual_tensor_hashes_not_recomputable_without_raw':True})
            del reference,ref,b;gc.collect()
            # Counts are API observations only. Baseline has 21 invocations: 2 quality,
            # 2 warmup, 5 clean and 4*3 separate metric invocations.
            if arm=='derived-baseline-observed':
                assert d['collective_counts_final']['calls']['all_gather_single']==49*21
                assert d['collective_counts_final']['calls']['reduce_scatter_single']==25*21
            log=root/arm/f'rank-{rank}.log';text=log.read_text()
            assert 'Traceback (most recent call last)'not in text
            rows.append({'arm':arm,'rank':rank,'worker_status':worker['status'],'valid':v['valid'],'checks':v['checks'],
                'sample_count':len(v['samples']),'coverage':v['coverage'],'bundle_hashes':v['bundle_hashes'],
                'diagnostic':{'path':str(root/arm/f'diagnostic-rank-{rank}.json'),'sha256':sha(root/arm/f'diagnostic-rank-{rank}.json')},
                'calls':calls,'collective_api_counts_final':d['collective_counts_final'],
                'runtime_facts_before':d['runtime_facts_before'],'runtime_facts_after':d['runtime_facts_after'],
                'absent':d['absent_observations'],'log_sha256':sha(log),'warnings_present':'Warning:'in text})
    assert all(sha(p)==h for p,h in identities.items()) and not torch.cuda.is_initialized()
    limits=protocol['limitations']+[
        'Observed first-step logits/loss equal controls; nonzero first-step gradients/updated parameters are within tolerance. Second-step only logits exceeds tolerance among full summary paths. This narrows the evidence gap but does not isolate a unique causal mechanism.',
        'Compared reference shapes/dtypes/counts and persisted selected expected values on CPU; actual full tensors were not saved, so independently recomputing all actual errors/digests remains impossible.',
        'All eight persisted largest-error logits entries per failing rank are within tolerance; original mismatch counts describe other entries. top_mismatches label is not a violation list.',
        'observer_wall_s is stored inside full_traversal but starts before original compare. There are not two separate measured wall clocks in this probe.',
        'Fingerprint recomputation only checks summary serialization; equal baseline fingerprints across steps/ranks do not imply equal underlying tensor contents or replicas.',
        'Four separate communication metrics are four sets of traces. No diagnostic timing, API count, kernel sum or cross-field difference is an acceptance score/wire/critical-path cost.']
    report={'at':datetime.now(timezone.utc).isoformat(),'core':protocol['core'],'terminal_sha256':sha(root/'followup-completed.json'),
        'generated':protocol['generated'],'rows':rows,'baseline_endpoint':baseline,'frozen_files_verified':len(identities),
        'identities':identities,'limits':limits,'scope':'Fixed exact-byte diagnostic validated on this second task after one real persistence repair; original failed quality preserved. No optimization/source-fidelity/production acceptance.',
        'audit_fixture_error':'Initial audit compared the selected original_invocation check mirror to the full original check; candidate mirror omits atol/rtol by code. Corrected to compare selected fields and independently check full compare_calls.original. No worker/probe/experiment rerun.',
        'next':'Test compact evidence interpretation and existing supplemental_evidence transport with real roles on both completed task observations; no manual causal answer or candidate solution.'}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k not in ['identities','baseline_endpoint']}
    public['baseline_summary']={'max_rank_then_median':baseline['max_rank_then_median'],'raw_trace_count':sum(len(r['traces'])for r in baseline['ranks']),
        'raw_trace_hashes':[{'path':t['path'],'sha256':t['sha256']}for r in baseline['ranks']for t in r['traces']]}
    public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'frozen_files':len(identities),'ranks':len(rows),
        'trace_count':public['baseline_summary']['raw_trace_count'],'baseline':baseline['max_rank_then_median']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
