"""Audit fixed setup observations and saved-value arithmetic; never rerun model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import torch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    p=read(root/'protocol.json');terminal=read(root/'followup-completed.json');identities=read(root/'request-identities.json')
    assert terminal['phase']=='completed' and p['core']==framework_source()
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    assert sha(p['probe'])==p['probe_sha256'] and sha(p['plan'])==p['plan_sha256']
    rows={};diagnostics={}
    for arm in ['baseline','candidate']:
        out=root/(arm+'-observed');request=read(out/'request.json');original=read(p['source_requests'][arm])
        expected={**original,'entry':p['probe'],'function':'run','output':str(out),'wait_for_devices':True,
            'parameters':{**original['parameters'],'diagnostic_context':{'original_entry':original['entry'],
                'original_function':original['function'],'independent_reference':None,'arm':arm+'-observed'}}}
        assert request==expected
        for item in identities[arm]['source'].values():assert sha(item['path'])==item['sha256']
        worker=read(out/'rank-0.json');started=read(out/'rank-0-started.json');d=read(out/'diagnostic-evidence-rank-0.json');v=worker['result']
        assert started['revision']['files']=={n:x['sha256']for n,x in identities[arm]['source'].items()}
        assert worker['status']=='passed' and worker['framework_source']==p['core'] and v['valid'] and v['math_policy_unchanged']
        assert len(v['checks'])==2 and all(x['passed']and x['compared_elements']==4861988 and x['max_abs_error']==0 for x in v['checks'])
        assert len(v['samples'])==5
        for k,value in v['metrics'].items():assert math.isclose(statistics.median(s[k]for s in v['samples']),value,rel_tol=1e-12,abs_tol=1e-8),k
        for sample in v['samples']:
            assert sample['generated_tokens']==32 and sample['decode_tokens']==31
            assert math.isclose(sample['ttft_ms']+sample['decode_ms'],sample['end_to_end_ms'],abs_tol=1e-6)
            assert math.isclose(31000/sample['decode_ms'],sample['decode_tokens_per_second'],rel_tol=1e-12)
        assert d['original_outcome']['original_returned'] and d['original_outcome']['error_type']is None
        assert d['instrumentation']['errors']==[] and d['instrumentation']['status']=='installed'
        for key in ['adapter','reference','collector']:
            item=d['instrumentation']['identity'][key];assert sha(item['path'])==item['sha256']
        diagnostics[arm]=d
        rows[arm]={'worker_sha256':sha(out/'rank-0.json'),'request_sha256':sha(out/'request.json'),
            'evidence_sha256':sha(out/'diagnostic-evidence-rank-0.json'),'evidence_bytes':(out/'diagnostic-evidence-rank-0.json').stat().st_size,
            'checks':v['checks'],'samples':v['samples'],'metrics':v['metrics'],'setup':v['setup'],'coverage':v['coverage'],
            'constraint_diagnostic_only':{'ttft_limit':24.45004275,'ttft_ms':v['metrics']['ttft_ms'],'within_original_limit':v['metrics']['ttft_ms']<=24.45004275}}
    assert diagnostics['baseline']['install_windows']==[]
    d=diagnostics['candidate'];assert len(d['install_windows'])==1;w=d['install_windows'][0]
    assert (w['assert_close_total'],w['assert_close_passed'],w['assert_close_failed'])==(198,196,2)
    assert w['assert_close_unfinished']==0 and len(w['assert_close_sites'])==128 and w['assert_close_retained_failures']==2
    assert w['assert_close_overflow']==69 and w['assert_close_failure_overflow']==0 and w['evicted_success_records']==1
    assert sum(w['assert_close_unretained_sites'].values())==69 and len(w['evicted_success_stubs'])==1
    assert [x['seq']for x in w['assert_close_sites']]==[1]+list(range(3,129))+[198]
    assert w['evicted_success_stubs'][0]['seq']==2 and w['evicted_success_stubs'][0]['evicted_for_failure_seq']==198
    validated_top=0;failures=[]
    for x in w['assert_close_sites']:
        assert not x['observer_error'] and x['comparison']['nonfinite_difference_count']==0
        assert all(item['all_finite']for item in x['operands'])
        c=x['comparison']
        for label,field,tol in [('top_call_tolerance_violations','violates_call_tolerance_count',c['call_tolerance']),
                                ('top_request_quality_violations','violates_request_quality_count',c['request_quality_tolerance'])]:
            top=c.get(label,[]);assert len(top)==min(64,c[field])
            errors=[]
            for item in top:
                actual=torch.tensor(item['actual'],dtype=torch.float32);expected=torch.tensor(item['expected'],dtype=torch.float32)
                error=float((actual-expected).abs());allowed=float(tol['atol']+tol['rtol']*expected.abs())
                assert error==item['abs_error'] and allowed==item['allowed'] and error>allowed
                assert 0<=item['flat_index']<c['numel'];errors.append(error);validated_top+=1
            assert errors==sorted(errors,reverse=True)
        if x['argmax']:
            for side in ['actual','expected']:
                a=x['argmax'][side];assert a['top_value']-a['second_value']==a['top1_minus_top2_margin']
        if x['failed']:failures.append(x)
        else:assert x['outcome']=={'passed':True} and c['max_abs_error']==0
    assert [x['seq']for x in failures]==[1,198]
    assert [x['comparison']['max_abs_error']for x in failures]==[.28125,22.03125]
    assert [x['comparison']['violates_request_quality_count']for x in failures]==[10716,151790]
    refs=w['reference_calls'];assert len(refs)==w['reference_call_total']==167 and w['reference_call_overflow']==0
    assert [x['seq']for x in refs]==list(range(1,168))
    for ref in refs:
        result=ref['result']['past_key_values'];assert result['layers']==36
        assert result['first_layer_keys_shape']==[1,8,result['seq_length'],128]
        before=ref['kwargs']['past_key_values']
        if before is not None:
            assert before['layers']==36 and before['first_layer_keys_shape']==[1,8,before['seq_length'],128]
            assert result['seq_length']==before['seq_length']+1 and ref['result']['returned_past_same_object_as_input']
            assert result['object_id']==before['object_id']==ref['input_past_object_id']
    left,right=refs[-2:]
    assert left['seq']==166 and right['seq']==167
    assert left['input_past_object_id']==right['input_past_object_id']
    assert left['kwargs']['input_ids']['tokens_head']==right['kwargs']['input_ids']['tokens_head']==[264]
    assert left['kwargs']['past_key_values']['seq_length']==258 and right['kwargs']['past_key_values']['seq_length']==259
    assert left['result']['past_key_values']['seq_length']==259 and right['result']['past_key_values']['seq_length']==260
    assert not torch.cuda.is_initialized()
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':p['core'],
        'frozen_files':len(p['frozen_identities']),'clean_samples':10,'zero_error_full_checks':4,'workers':rows,
        'retention':{k:v for k,v in w.items()if k not in ['assert_close_sites','reference_calls']},
        'failed_comparisons':failures,'all_retained_success_statistics_zero':True,'saved_top_value_arithmetic_checked':validated_top,
        'reference_phase_counts':dict(Counter(x['spec']['phase']for x in refs)),'last_two_reference_calls':refs[-2:],
        'limitations':['Full operands not saved. Aggregate counts/max/mean are probe reports; saved top values and arithmetic independently checked, not full raw recomputation.',
            'Reference and comparison counters are separate, without unified cross-hook timestamp. Linking last two refs to chainE is code/order inference from source625-629, not direct event cross-ID.',
            'Same live cache object, same token264 but input lengths258/259 are observed; the two operands do not represent equal decode positions. Does not isolate all numerical causes or certify the unused fast path.',
            'Fused first-step setup operand mismatch also exceeds request .02/.02 in reported10716elements, with saved violating examples. Not a new full32token accelerated-quality evaluation; no unique Triton/arithmetic cause.',
            'Probe summary timing/lifetime/CPU copy effects persist; observed timers are not acceptance scores or removable overhead.',
            '126 retained success details are exact, one success detail evicted and69 success details absent; cannot infer their raw errors from pass outcomes.',
            'Baseline empty windows and original returns passed; GPU runtime hook restoration is by examined code plus CPU tests, not independently logged identity checks.',
            'No source comparison rerun, independent_reference null, manual source mapping boundary remains. No native trace/optimized Graph proof or new acceptance.'],
        'audit_fixture_error':'Initial saved-summary auditor assumed top-violation keys exist for exact comparisons. Producer omits those arrays when count is zero; KeyError retained. Corrected audit accepts absent arrays only when count zero; no worker/CPU fixture/model rerun or probe change.',
        'decision':'Observation succeeded in this bounded case, original fallback/quality and failed TTFT remain. Freeze exact evidence; do not alter candidate or adopt collector/prompt. Any later continuation should consume original framework feedback plus unchanged observation artifacts, without developer causal repair.'}
    write_json(root/'terminal-audit.json',report)
    public={**report,'terminal_audit_sha256':sha(root/'terminal-audit.json')};write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'samples':10,'checks':4,'top_values':validated_top,'failure_max':[.28125,22.03125],'reference_calls':167}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
