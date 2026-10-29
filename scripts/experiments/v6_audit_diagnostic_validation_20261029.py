"""Recompute all generated diagnostic outputs on CPU; never rerun a model."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha


def comparison(actual,expected):
    a=actual['logits'].float();b=expected['logits'].float();error=(a-b).abs()
    return {'tokens_exact':torch.equal(actual['tokens'],expected['tokens']),
        'token_mismatch_positions':torch.nonzero(actual['tokens'][0]!=expected['tokens'][0]).flatten().tolist(),
        'logits_bitwise_equal':torch.equal(actual['logits'],expected['logits']),
        'finite':bool(torch.isfinite(a).all()),'logits_shape':list(a.shape),'logits_elements':a.numel(),
        'max_abs_error':float(error.max()),'per_position_max_abs_error':error.amax(dim=-1)[0].tolist(),
        'outside_float32_source_tolerance':int((error>(.02+.02*b.abs())).sum()),
        'semantics':'CPU float32 actual minus expected, .02+.02*abs(expected). Not a reconstruction of GPU BF16 assert_close mismatch count.'}


def audit(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    assert terminal['all_frozen_hashes_unchanged'] and all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    assert sha(protocol['probe'])==protocol['probe_sha256']=='16af16d674dc05872e5ff835a5b165c5473819ed9b4e743be12bb2ef64e74110'
    source=torch.load(protocol['source_reference']['path'],map_location='cpu',weights_only=True)
    endpoints=[]
    for arm in ['baseline','candidate']:
        out=root/(arm+'-observed');req=read(out/'request.json');original=read(protocol['source_requests'][arm])
        changed=[k for k in sorted(set(req)|set(original))if req.get(k)!=original.get(k)]
        assert set(changed)<={'entry','function','output','wait_for_devices','parameters'}
        assert {k:v for k,v in req['parameters'].items()if k!='diagnostic_context'}==original['parameters']
        context=req['parameters']['diagnostic_context'];assert context['original_entry']==original['entry'] and context['original_function']==original['function']
        worker=read(out/'rank-0.json');started=read(out/'rank-0-started.json');result=read(root/(arm+'-observed-result.json'))
        assert worker['framework_source']==protocol['core'] and result['worker_pids_reaped']
        assert started['parameters_ref']['request_sha256']==sha(out/'request.json')
        assert all(sha(Path(req['cwd'])/n)==h for n,h in started['revision']['files'].items())
        d=read(out/'diagnostic-index-rank-0.json');assert len(d['artifacts'])==4
        for p in d['provenance']['files'].values():assert p['status']=='present' and sha(p['path'])==p['sha256']
        assert d['provenance']['revision_manifest']==started['revision']
        assert len(d['instrument_restore_report'])==8 and all(e['restored']for e in d['instrument_restore_report'])
        events=d['events'];counts=Counter(e['kind']for e in events)
        assert counts['original_entry_call']==1 and counts['observe_begin']==2
        assert [x['seq']for x in events]==list(range(len(events)))
        assert all(a['t_ns']<=b['t_ns']for a,b in zip(events,events[1:]))
        quality=[x for x in d['capture_summaries']if x['capture']]
        assert [x['kind']for x in quality]==['greedy_run','teacher_forced_replay']*2
        assert [x['ordinal']for x in quality]==[0,1,2,3] and all(x['mode']=='quality'for x in quality)
        raw=[];relations=[]
        for i,artifact in enumerate(d['artifacts']):
            assert sha(artifact['path'])==artifact['sha256'] and Path(artifact['path']).stat().st_size==artifact['bytes']
            t=torch.load(artifact['path'],map_location='cpu',weights_only=True);raw.append(t)
            assert t['tokens'].dtype==torch.int64 and t['tokens'].shape==(1,32)
            assert t['logits'].dtype==torch.bfloat16 and t['logits'].shape==(1,32,151936)
            assert quality[i]['tokens']==t['tokens'][0].tolist() and quality[i]['raw_artifact']==artifact['sha256']
            relations.append({'ordinal':i,'kind':quality[i]['kind'],'artifact':artifact,'against_source':comparison(t,source)})
        pairs=[]
        for i in [0,2]:
            c=comparison(raw[i+1],raw[i]);old=d['greedy_vs_replay_pairwise'][i//2]
            assert old['tokens_equal']==c['tokens_exact'] and old['differing_positions']==c['token_mismatch_positions']
            assert old['replay_vs_greedy_logits_max_abs_error']==c['max_abs_error']
            assert old['replay_vs_greedy_logits_per_step_max_abs_error']==c['per_position_max_abs_error']
            for detail in old.get('first_divergence_detail',[]):
                pos=detail['position'];assert detail['greedy_token']==int(raw[i]['tokens'][0,pos]) and detail['replay_token']==int(raw[i+1]['tokens'][0,pos])
                assert detail['teacher_token_fed_at_previous_position']==int(raw[i]['tokens'][0,pos-1])
                for key,t in [('greedy_logits_top2',raw[i]),('replay_logits_top2',raw[i+1])]:
                    values,ids=t['logits'][0,pos].float().topk(2);record=detail[key]
                    assert record['top1_id']==int(ids[0]) and record['top2_id']==int(ids[1])
                    assert record['top1_value']==float(values[0]) and record['top2_value']==float(values[1])
                    assert record['margin_top1_minus_top2']==float(values[0]-values[1])
            pairs.append(c)
        assert all(x['passed']for x in d['negative_probes'])
        assert d['negative_probes'][1]['detail']['observation_path']=='output.generated_tokens'
        routes=Counter((e['phase'],e['route'],e['tracked'])for e in events if e['kind']=='candidate_route')
        if arm=='baseline':
            assert worker['status']=='passed' and worker['result']['valid'] and worker['result']['math_policy_unchanged']
            assert len(worker['result']['samples'])==5 and len(worker['result']['checks'])==2
            assert all(c['passed']and c['max_abs_error']==0 for c in worker['result']['checks'])
            assert counts['adapter_run_begin']==9 and counts['observe_end']==2 and not routes
            assert all(r['against_source']['logits_bitwise_equal']and r['against_source']['tokens_exact']for r in relations)
        else:
            assert worker['status']=='failed' and worker['error']=='teacher-forced replay token sequence differs from greedy run'
            assert 'result'not in worker and d['original_error']['error']==worker['error']
            assert counts['adapter_run_begin']==2 and counts['observe_end']==1 and counts['observe_error']==1
            assert counts['original_exception']==1 and not counts['original_entry_returned']
            assert all(r['against_source']['logits_bitwise_equal']and r['against_source']['tokens_exact']for r in relations[:3])
            assert relations[3]['against_source']['token_mismatch_positions']==[1,3,6,10,14,19,22,23,24,27,29,31]
            assert relations[3]['against_source']['max_abs_error']==31.125
        endpoints.append({'arm':arm,'request_sha256':sha(out/'request.json'),'changed_top_level_fields':changed,
            'entry_sha256':sha(req['entry']),'worker_status':worker['status'],'original_error':worker.get('error'),
            'revision':started['revision'],'samples':worker.get('result',{}).get('samples',[]),
            'returned_outer_checks':worker.get('result',{}).get('checks'),'diagnostic_index_sha256':sha(out/'diagnostic-index-rank-0.json'),
            'index_bytes':(out/'diagnostic-index-rank-0.json').stat().st_size,'event_counts':dict(counts),
            'ordered_quality_captures':[{k:q[k]for k in ['ordinal','kind','mode','phase_at_call','capture']}for q in quality],
            'routes':[{'phase':p,'route':r,'tracked':t,'count':n}for(p,r,t),n in routes.items()],
            'raw_relations':relations,'pairwise_recomputed':pairs,'restored_hooks':d['instrument_restore_report'],
            'negative_probes':d['negative_probes'],'binding_snapshots':d['binding_snapshots']})
    report={'at':datetime.now(timezone.utc).isoformat(),'core':protocol['core'],'frozen_files_verified':len(protocol['frozen_identities']),
        'probe_sha256':protocol['probe_sha256'],'terminal_phase':terminal['phase'],'endpoints':endpoints,
        'conclusion':'Original real-agent probe executed without manual changes and captured the missing continuous full-output relation: only second teacher wrong. Baseline passed; candidate remains original failure. Single-case diagnostic feasibility, not performance acceptance or generic production adoption.',
        'limitations':protocol['known_probe_limitations'],'frozen_identities':protocol['frozen_identities']}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k!='frozen_identities'}
    for e in public['endpoints']:
        e.pop('binding_snapshots',None)
    public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'full_tensors':8,'baseline_clean_samples':5,
        'candidate_failure':endpoints[1]['original_error'],'second_teacher_source':endpoints[1]['raw_relations'][3]['against_source']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
