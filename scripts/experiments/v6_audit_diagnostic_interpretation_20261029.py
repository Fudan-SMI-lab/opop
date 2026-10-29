"""Audit proposed evidence packets; never edits or routes corrected answers."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import torch
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_diagnostic_interpretation_20261029 import build_cases, INSTRUCTIONS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(a.keys() | b.keys()) for p in differences(a.get(k), b.get(k), path+'.'+k)]
    return [path] if a != b else []


def audit(root):
    terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    cases,identities=build_cases(root.parent)
    assert identities==read(root/'protocol.json')['frozen_identities']
    assert framework_source()==read(root/'protocol.json')['core']
    out=root/'audit';out.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    roles=[];prompts=[];settings=[]
    for case,data in cases.items():
        result=terminal['results'][case];packet=read(result['packet'])
        report=packet['diagnostic_evidence_report'];sandbox=Path(result['agent']['sandbox'])
        snapshots=[]
        for path in (root/case/'generation-snapshots').glob('*.json'):
            m,p,files=load(path)
            if not m['fresh_session']:continue
            assert m['framework_source']==framework_source() and INSTRUCTIONS in p
            assert set(files)=={'inputs/'+k for k in data['inputs']}|{'inputs/worker-environment.json'}
            for k,v in data['inputs'].items():assert files['inputs/'+k]==(v.encode() if isinstance(v,str) else v)
            snapshots.append((path,m,p,files))
        assert len(snapshots)==1
        path,m,p,files=snapshots[0];prompts.append(p);settings.append(m['tool_model_settings'])
        parts=[{'id':ident,'data':json.loads(raw)} for ident,raw in db.execute('select id,data from part where session_id=? order by time_created,id',(m['session'],))]
        messages=[json.loads(raw) for (raw,) in db.execute('select data from message where session_id=?',(m['session'],))]
        write_json(out/f'{case}-session-parts.json',parts);write_json(out/f'{case}-messages.json',messages)
        tools=[];possible_errors=[];later_hits=[]
        for part in parts:
            d=part['data']
            if d.get('type')!='tool':continue
            s=d['state'];inp=s.get('input',{});output=str(s.get('output',''));whole=json.dumps(inp)+output
            tools.append({'part':part['id'],'tool':d['tool'],'status':s.get('status'),'input':inp,'output':output,'metadata':s.get('metadata')})
            if 'Traceback (most recent call last)' in output or s.get('metadata',{}).get('exit') not in [None,0]:
                possible_errors.append({'part':part['id'],'tool':d['tool'],'exit':s.get('metadata',{}).get('exit'),'output_sha256':hashlib.sha256(output.encode()).hexdigest()})
            hits=[v for v in ['terminal-audit.json','public-audit.json','qwen-routed-independent-20261029','qwen-joint-feedback-routed-20261029','trial-4f5fce6e6b','fsdp-continuation-20261028'] if v in whole]
            if hits:later_hits.append({'part':part['id'],'terms':hits})
        write_json(out/f'{case}-complete-tools.json',tools)
        cited={};citations=[]
        for section in ['facts','hypotheses']:
            for i,row in enumerate(report[section]):
                for e in row['evidence']:
                    if e['path']not in cited:cited[e['path']]=sha(e['path'])
                    h=cited[e['path']]
                    citations.append({'claim':f'{section}[{i}]',**e,'hash_matches':h==e['sha256']})
        assert all(e['hash_matches']for e in citations)
        write_json(out/f'{case}-citation-audit.json',citations)
        original=read(data['feedback'])
        before=json.loads(route(json.dumps(original),current_revision=data['current_revision']))
        after=json.loads(route(json.dumps({**original,'supplemental_evidence':packet}),current_revision=data['current_revision']))
        assert after['supplemental_evidence']==packet
        assert {k:v for k,v in before.items()if k!='supplemental_evidence'}=={k:v for k,v in after.items()if k!='supplemental_evidence'}
        index=read(sandbox/'inputs/evidence-index.json')['original_evidence_index']
        old=index.get('original_execution_request',index.get('original_execution_requests',{}).get('candidate'))
        request_diff=differences(read(old['path']),read(Path(data['validation'])/'candidate-observed/request.json'))
        roles.append({'case':case,'snapshot':str(path),'snapshot_sha256':sha(path),'session':m['session'],
            'core':m['framework_source'],'input_count':len(files),'explicit_input_bytes':sum(len(v.encode()if isinstance(v,str)else v)for v in data['inputs'].values()),
            'inputs':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for k,v in files.items()},
            'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),'settings_sha256':digest(m['tool_model_settings']),
            'tools':len(tools),'tool_counts':dict(Counter(t['tool']for t in tools)),'possible_tool_errors':possible_errors,
            'later_artifact_string_hits':later_hits,'access_scope':'Complete commands/outputs and retained scripts reviewed separately; no later-answer access observed, filesystem not hermetic.',
            'assistant_messages':sum(v.get('role')=='assistant'for v in messages),
            'provider_estimate':sum(v.get('cost',0)for v in messages if v.get('role')=='assistant'),
            'packet_bytes':Path(result['packet']).stat().st_size,'packet_sha256':sha(result['packet']),
            'facts':len(report['facts']),'hypotheses':len(report['hypotheses']),'citation_occurrences':len(citations),'distinct_cited_files':len(cited),
            'all_citation_file_hashes_match':True,'semantic_locators_not_certified_by_hash':True,
            'route_packet_preserved':True,'other_routed_fields_equal':True,'actual_downstream_delivery':False,
            'original_vs_observed_request_differences':request_diff,
            'retained_history_sha256':sha(out/f'{case}-session-parts.json'),'retained_messages_sha256':sha(out/f'{case}-messages.json')})
    db.close();assert prompts[0]==prompts[1] and settings[0]==settings[1]
    # Recheck saved Qwen outputs on CPU, including full independent logits.
    qr=root.parent/'diagnostic-author-validation-20261029'
    source=torch.load(root.parent/'validation-runs/readiness-20261020-a/qwen_inference/interactive-reference.pt',map_location='cpu',weights_only=False)
    q=[]
    for arm in ['baseline-observed','candidate-observed']:
        for ordinal in range(4):
            kind='teacher_forced_replay' if ordinal%2 else 'greedy_run'
            p=qr/arm/f'infer-{ordinal:02d}-{kind}-rank-0.pt';v=torch.load(p,map_location='cpu',weights_only=False)
            delta=(v['logits'].float()-source['logits'].float()).abs()
            q.append({'arm':arm,'ordinal':ordinal,'path':str(p),'sha256':sha(p),'tokens_equal':torch.equal(v['tokens'],source['tokens']),
                'token_positions':(v['tokens']!=source['tokens'])[0].nonzero().flatten().tolist(),
                'logits_equal':torch.equal(v['logits'],source['logits']),'finite':bool(torch.isfinite(v['logits']).all()),
                'max_abs':float(delta.max()),'first_decode_max_abs':float(delta[:,1].max()),
                'source_tolerance_violations_cpu_float32':int((delta>(.02+.02*source['logits'].float().abs())).sum())})
    assert all(v['logits_equal']and v['tokens_equal']for v in q[:-1]) and q[-1]['max_abs']==31.125
    qindex=read(qr/'candidate-observed/diagnostic-index-rank-0.json')
    assert qindex['negative_probes'][1]['detail']['observation_path']=='output.generated_tokens'
    # Summaries can be checked internally, not independently reconstructed as tensors.
    fr=root.parent/'fsdp-diagnostic-validation-20261029';f=[]
    for rank in [0,1]:
        d=read(fr/f'candidate-observed/diagnostic-rank-{rank}.json');p=d['compare_calls'][0]['full_traversal']['paths']
        grads={x['path'].removeprefix('output.gradients.')for x in p if x['path'].startswith('output.gradients.')and not x['torch_equal']}
        params={x['path'].removeprefix('output.parameters.')for x in p if x['path'].startswith('output.parameters.')and not x['torch_equal']}
        logits=next(x for x in d['compare_calls'][1]['full_traversal']['paths']if x['path']=='output.logits')
        top=logits['top_mismatches']
        assert len(top)==8 and all(t['abs_error']<=1e-6+1e-4*abs(t['expected'])for t in top)
        assert params<=grads and len(grads-params)==48
        worker=read(fr/f'candidate-observed/rank-{rank}.json')['result']
        assert 'math_policy_unchanged'not in d['original_invocation']['mirror'] and worker['math_policy_unchanged']
        f.append({'rank':rank,'first_nonexact_gradients':len(grads),'first_nonexact_parameters':len(params),
            'nonexact_gradient_with_exact_parameter':len(grads-params),'top8_logits_all_within_tolerance':True,
            'second_logit_violations':logits['mismatch_allclose_count'],'mirror_math_policy_field_present':False,
            'worker_math_policy_unchanged':True,'actual_raw_tensors_saved':False})
    assert not torch.cuda.is_initialized()
    report={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['phase'],'core':framework_source(),
        'frozen_files_verified':len(identities),'roles':roles,'same_generic_prompt_and_settings':True,
        'qwen_saved_outputs':q,'qwen_negative_scope':'Both recorded negatives are tokens, not a logit negative.',
        'fsdp_summary_checks':f,'cuda_initialized':False,'production_adoption':False,
        'decision':'Do not forward either whole packet: useful observations coexist with factual/scope/causal errors. Preserve exact generated packets.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k not in ['roles','qwen_saved_outputs']}
    public['roles']=[{k:v for k,v in row.items()if k not in ['inputs','possible_tool_errors']}for row in roles]
    public['qwen_output_checks']=[{k:v for k,v in x.items()if k not in ['path','sha256']}for x in q]
    public['terminal_audit_sha256']=sha(root/'terminal-audit.json')
    write_json(root/'public-audit.json',public)
    print(json.dumps({'terminal_audit_sha256':public['terminal_audit_sha256'],'roles':[{k:r[k]for k in ['case','tools','provider_estimate','distinct_cited_files','original_vs_observed_request_differences']}for r in roles],'df':report['df']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
