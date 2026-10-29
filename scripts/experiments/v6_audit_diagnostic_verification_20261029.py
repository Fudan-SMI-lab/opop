"""Read-only audit of verifier outputs; 'usable' is not a factual quality gate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from kernel_optimizer.project.feedback import route
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_diagnostic_interpretation_20261029 import build_cases
from scripts.experiments.v6_diagnostic_verification_20261029 import INSTRUCTIONS, REVIEW_SCHEMA, claim_ids


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def citations(value):
    if isinstance(value,dict):
        if all(k in value for k in ['path','sha256','locator']):yield value
        for v in value.values():yield from citations(v)
    elif isinstance(value,list):
        for v in value:yield from citations(v)


def audit(root):
    protocol=read(root/'protocol.json');done=read(root/'followup-completed.json')
    assert done['phase']=='completed' and done['frozen_files_unchanged']
    assert framework_source()==protocol['core']
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    cases,_=build_cases(root.parent);old=read(root.parent/'diagnostic-interpretation-20261029/followup-results.json')
    rows=[];prompts=[];settings=[];a=root/'audit';a.mkdir(exist_ok=True)
    for case,data in cases.items():
        data['inputs']['proposed-packet.json']=Path(old[case]['packet']).read_bytes()
        data['inputs']['verification-schema.json']=json.dumps(REVIEW_SCHEMA,indent=2)
        result=done['results'][case];receipt=result['agent'];snapshots=[]
        for path in (root/case/'generation-snapshots').glob('*.json'):
            m,p,files=load(path)
            if not m['fresh_session']:continue
            assert m['role']=='diagnostic_verification' and m['framework_source']==framework_source() and INSTRUCTIONS in p
            assert set(files)=={'inputs/'+k for k in data['inputs']}|{'inputs/worker-environment.json'}
            for k,v in data['inputs'].items():assert files['inputs/'+k]==(v.encode()if isinstance(v,str)else v)
            snapshots.append((path,m,p,files))
        assert len(snapshots)==1
        path,m,p,files=snapshots[0];prompts.append(p);settings.append(m['tool_model_settings'])
        parts=read(a/f'{case}-session-parts.json');tools=[];errors=[];later=[]
        for part in parts:
            d=part['data']
            if d.get('type')!='tool':continue
            st=d['state'];inp=st.get('input',{});out=str(st.get('output',''));whole=json.dumps(inp)+out
            tools.append({'part':part['id'],'tool':d['tool'],'state':st})
            if st.get('status')=='error' or st.get('metadata',{}).get('exit')not in [None,0]:
                errors.append({'part':part['id'],'tool':d['tool'],'exit':st.get('metadata',{}).get('exit')})
            hit=[x for x in ['terminal-audit.json','public-audit.json','qwen-routed-independent-20261029','qwen-joint-feedback-routed-20261029','trial-4f5fce6e6b','fsdp-continuation-20261028']if x in whole]
            if hit:later.append({'part':part['id'],'terms':hit})
        write_json(a/f'{case}-complete-tools.json',tools)
        outputs={n:read(v['path'])for n,v in result['files'].items()}
        for v in result['files'].values():assert sha(v['path'])==v['sha256']
        review=outputs['verification.json'];packet=outputs['supplemental-evidence.json']
        assert sorted(c['id']for c in review['claims'])==sorted(claim_ids(read(old[case]['packet'])))
        hcache={};cites=[]
        for name,value in outputs.items():
            for e in citations(value):
                if e['path']not in hcache:hcache[e['path']]=sha(e['path'])if Path(e['path']).is_file()else None
                cites.append({'output':name,**e,'actual_sha256':hcache[e['path']],'hash_matches':hcache[e['path']]==e['sha256']})
        write_json(a/f'{case}-citations.json',cites)
        original=read(data['feedback']);before=json.loads(route(json.dumps(original),current_revision=data['current_revision']))
        after=json.loads(route(json.dumps({**original,'supplemental_evidence':packet}),current_revision=data['current_revision']))
        assert after['supplemental_evidence']==packet
        assert {k:v for k,v in after.items()if k!='supplemental_evidence'}=={k:v for k,v in before.items()if k!='supplemental_evidence'}
        sandbox=Path(receipt['sandbox']);retained=[]
        for sub in ['checks','cpu-checks']:
            for cp in sorted((sandbox/sub).rglob('*')):
                if cp.is_file():retained.append({'path':str(cp),'sha256':sha(cp),'bytes':cp.stat().st_size})
        rows.append({'case':case,'snapshot':str(path),'snapshot_sha256':sha(path),'session':m['session'],
            'input_count':len(files),'explicit_input_bytes':sum(len(v.encode()if isinstance(v,str)else v)for v in data['inputs'].values()),
            'input_hashes':{k:hashlib.sha256(v).hexdigest()for k,v in files.items()},'prompt_sha256':hashlib.sha256(p.encode()).hexdigest(),
            'settings_sha256':digest(m['tool_model_settings']),'contract_repairs':receipt['contract_repairs'],
            'usage':receipt['usage'],'tools':len(tools),'tool_counts':dict(Counter(t['tool']for t in tools)),
            'tool_exit_or_state_errors':errors,'later_artifact_term_hits':later,
            'access_audit':'Full recorded commands/outputs and retained CPU scripts reviewed; no later-answer access observed. Nonhermetic external filesystem; keyword checks alone are not proof.',
            'outputs':result['files'],'verifier_overall':review['overall'],'verdict_counts':dict(Counter(c['verdict']for c in review['claims'])),
            'packet_facts':len(packet['diagnostic_evidence_report']['facts']),'citation_occurrences':len(cites),'distinct_cited_files':len(hcache),
            'citation_hash_errors':[e for e in cites if not e['hash_matches']],
            'exact_packet_transport_passed':True,'actual_selector_candidate_delivery':False,
            'retained_checks':retained,'history_sha256':sha(a/f'{case}-session-parts.json')})
    assert prompts[0]==prompts[1] and settings[0]==settings[1]
    # Counter-evidence is kept separate from generated packets and future role inputs.
    original_script=Path(old['qwen']['agent']['sandbox'])/'checks/verify_artifacts.py'
    assert sha(original_script)=='3452d1a9c5f349bc1487063b9046fc46c8f5d7b05d71f5ed9e5c6ea08e4fd1bd'
    qr=root.parent/'diagnostic-author-validation-20261029';qi=read(qr/'candidate-observed/diagnostic-index-rank-0.json')
    evalpath=qr/'framework/ad3c962c5491/src/kernel_optimizer/project/evaluation.py'
    lines=evalpath.read_text().splitlines();assert 'compare(adapter.observe(state, result)'in lines[385]
    events=[e for e in qi['events']if e['kind']in ['observe_end','observe_error','adapter_run_begin']]
    assert any(e['seq']==49 and e['kind']=='observe_end'for e in events)
    assert any(e['seq']==50 and e['kind']=='adapter_run_begin'for e in events)
    assert qi['negative_probes'][1]['detail']['observation_path']=='output.generated_tokens'
    fs=root.parent/'fsdp-diagnostic-validation-20261029';counter=[]
    for rank in [0,1]:
        d=read(fs/f'candidate-observed/diagnostic-rank-{rank}.json');paths=d['compare_calls'][0]['full_traversal']['paths']
        eq=[x['path']for x in paths if x['path'].startswith('output.parameters.')and x['torch_equal']]
        logits=next(x for x in d['compare_calls'][1]['full_traversal']['paths']if x['path']=='output.logits')
        top=[{**t,'outside_tolerance':t['abs_error']>1e-6+1e-4*abs(t['expected'])}for t in logits['top_mismatches']]
        assert all(not t['outside_tolerance']for t in top)
        worker=read(fs/f'candidate-observed/rank-{rank}.json')['result']
        assert worker['setup'] and not worker['samples'] and not worker['metrics']
        counter.append({'rank':rank,'equal_parameter_paths':eq,'top_absolute_logit_entries':top,
            'setup_observations_exist':worker['setup'],'formal_measurement_samples':0,
            'runtime_math_policy_unchanged':worker['math_policy_unchanged'],'mirror_has_math_policy_field':'math_policy_unchanged'in d['original_invocation']['mirror']})
    probe=Path(read(root.parent/'diagnostic-interpretation-20261029/fsdp/sandboxes/diagnostic_interpretation-61385441501d/inputs/evidence-index.json')['diagnostic_implementation']['probe'])
    pl=probe.read_text().splitlines()
    details={'qwen_original_script_exists':{'path':str(original_script),'sha256':sha(original_script)},
        'qwen_compare_scope':{'code_path':str(evalpath),'sha256':sha(evalpath),'lines_383_391':lines[382:391],'events':events,
            'interpretation':'First observe returned then second run began, so first compare executed; second observe prevented its own compare. No returned checks array does not mean zero compare calls.'},
        'qwen_negatives':qi['negative_probes'],'fsdp_summary_and_setup_counter_evidence':counter,
        'fsdp_timer':{'probe':str(probe),'sha256':sha(probe),'lines_339_374':pl[338:374],
            'scope':'Clock starts before original comparator, stops after traversal; not isolated added observer cost.'}}
    write_json(a/'counter-evidence.json',details)
    final={'at':datetime.now(timezone.utc).isoformat(),'terminal_phase':done['phase'],'core':framework_source(),
        'frozen_files_verified':len(protocol['frozen_identities']),'roles':rows,'counter_evidence_sha256':sha(a/'counter-evidence.json'),
        'decision':'Verifier study closed. Do not adopt overall usable or forward whole packets. No repeated verifier or production gate. Next experiment may provide raw observation indexes directly to existing candidate roles.',
        'scope':'No new model/GPU/quality/performance acceptance. Original source mapping, developer-assisted adapter and observer limitations remain.',
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',final)
    public={**final,'roles':[{k:v for k,v in row.items()if k not in ['input_hashes','retained_checks','tool_exit_or_state_errors']}for row in rows],
        'terminal_audit_sha256':sha(root/'terminal-audit.json')}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'terminal_audit_sha256':public['terminal_audit_sha256'],'roles':[{k:r[k]for k in ['case','tools','distinct_cited_files','citation_hash_errors','verdict_counts']}for r in rows],'df':final['df']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
