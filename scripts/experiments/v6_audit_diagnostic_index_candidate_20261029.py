"""Audit immutable candidate replays, including non-consumption of new evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_diagnostic_index_candidate_20261029 import prepare


def digest(value):return hashlib.sha256(value).hexdigest()


def main(root):
    cases,frozen=prepare(root.parent);done=read(root/'followup-completed.json');protocol=read(root/'protocol.json')
    assert done['phase']=='completed'and done['frozen_files_unchanged']and protocol['core']==framework_source()
    assert frozen==protocol['frozen_identities']and all(sha(p)==h for p,h in frozen.items())
    rows={};a=root/'audit'
    for case,data in cases.items():
        result=done['results'][case];receipt=result['agent'];snapshot=Path(result['initial_snapshot'])
        meta,prompt,files=load(snapshot)
        assert files==data['changed_files']and prompt==data['prompt']
        assert meta['tool_model_settings']==data['saved']['tool_model_settings']and meta['framework_source']==framework_source()
        sandbox=Path(receipt['sandbox'])
        assert all((sandbox/n).read_bytes()==raw for n,raw in files.items())
        for name in ['candidate','proposal']:assert sha(sandbox/(name+('.py'if name=='candidate'else'.json')))==result[name+'_sha256']
        parts=read(a/f'{case}-session-parts.json');tools=[p for p in parts if p['data'].get('type')=='tool']
        evidence=[];errors=[];mentions=[];commands=[];later=[]
        for i,part in enumerate(tools):
            d=part['data'];s=d['state'];inp=s.get('input',{});out=str(s.get('output',''));cmd=json.dumps(inp)
            if any(x in cmd for x in ['evidence-index.json','diagnostic-author-validation-20261029','fsdp-diagnostic-validation-20261029']):
                evidence.append({'index':i,'part':part['id'],'input':inp})
            if s.get('status')=='error'or s.get('metadata',{}).get('exit')not in [None,0]:
                errors.append({'index':i,'part':part['id'],'exit':s.get('metadata',{}).get('exit'),'traceback':'Traceback (most recent call last)'in out})
            if 'Traceback (most recent call last)'in out or 'AssertionError'in out:
                mentions.append({'index':i,'part':part['id'],'exit':s.get('metadata',{}).get('exit'),
                    'scope':'Text marker only; may quote the original recorded failure. Inspect command/output before assigning a new probe failure.'})
            if d['tool']=='bash':commands.append({'index':i,'part':part['id'],'command':inp.get('command'),'output_sha256':digest(out.encode()),'output_bytes':len(out.encode())})
            hits=[t for t in ['terminal-audit.json','public-audit.json','supplemental-evidence.json','verification.json','qwen-routed-independent-20261029','qwen-joint-feedback-routed-20261029','trial-4f5fce6e6b']if t in cmd]
            if hits:later.append({'index':i,'part':part['id'],'terms':hits})
        assert not evidence and not later
        write_json(a/f'{case}-command-index.json',commands)
        # Preserve exact CPU probe source from the session, including variants.
        probe_sources=[]
        for j,part in enumerate(tools):
            cmd=part['data']['state'].get('input',{}).get('command','')
            if any(x in cmd for x in ['test_stateless.py','test_candidate.py','test_candidate_fsdp_keep.py']):
                path=a/f'{case}-probe-command-{j:02d}.txt';path.write_text(cmd)
                probe_sources.append({'path':str(path),'sha256':sha(path)})
        rows[case]={'session':meta['session'],'snapshot':str(snapshot),'snapshot_sha256':sha(snapshot),
            'input_count':len(files),'only_changed_input':'inputs/previous-feedback.txt',
            'feedback_bytes_before':len(data['files']['inputs/previous-feedback.txt']),'feedback_bytes_after':len(files['inputs/previous-feedback.txt']),
            'prompt_sha256':digest(prompt.encode()),'settings_sha256':digest(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()),
            'candidate_sha256':result['candidate_sha256'],'candidate_bytes':(sandbox/'candidate.py').stat().st_size,
            'proposal_sha256':result['proposal_sha256'],'parameter_space':result['parameter_space'],
            'contract_repairs':receipt['contract_repairs'],'usage':receipt['usage'],'tools':len(tools),
            'tool_counts':dict(Counter(p['data']['tool']for p in tools)),'nonzero_tool_exit_or_state':errors,'exception_text_mentions':mentions,
            'index_or_new_observation_read_commands':evidence,'later_answer_command_hits':later,
            'evidence_consumption':'Complete recorded commands/outputs reviewed: no read of added index, Agent-authored probe/plan, raw or summary observations. Qwen grep exposed full pointer; FSDP traversal truncated before path. Presence is not consumption.',
            'history_sha256':sha(a/f'{case}-session-parts.json'),'probe_source_records':probe_sources}
    qtools=read(a/'qwen-tools.json');ftools=read(a/'fsdp-tools.json')
    assert 'evidence-index.json'in qtools[11]['data']['state']['output']
    assert 'supplemental_evidence.diagnostic_artifacts.index. ...'in ftools[9]['data']['state']['output']
    assert 'evidence-index.json'not in ftools[9]['data']['state']['output']
    assert 'cd /tmp/opencode &&'in ftools[67]['data']['state']['input']['command']
    temp=a/'fsdp-temporary-candidate.py';assert sha(temp)=='ebb1046df7f50dabb76c2bd46edf1d6b2f3305872c891df5c1b2d0d0097a73f2'
    assert 'Whole-step CUDA Graph execution plan for the accumulated DDP'in temp.read_text()
    access={'scope':'Complete recorded commands/outputs and CPU probe histories reviewed. Nonhermetic filesystem; do not claim isolation.',
        'qwen':'No later-answer read observed. Removed its own /tmp/opencode/ctest and __pycache__; exact source/patch commands retained, ephemeral raw CPU output directory absent.',
        'fsdp':'Wrong-cwd CPU test at tool67 imported /tmp/opencode/candidate.py, an older DDP candidate, then failed expected FSDP call-ID assertion. Its source was not printed in the session; observed output does not prove semantic uptake. No GPU/model invocation in that failed test. Current old file SHA/mtime archived; this indirect other-case import precludes a blanket no-other-case-access claim. Later explicit sandbox-cwd mock passed. Agent removed copied test_candidate_fsdp_keep.py; retained source commands and remaining test_candidate.py copied to audit.',
        'gpu_scope':'Controller scheduled no GPU workers. Recorded probes import libraries or use CPU tensors/mocks; no new real model, distributed, CUDA graph or collective execution found. GPU empty at audit.'}
    review={'fixed_execution_justified':True,'qwen_parameters':{},'fsdp_parameters':{'keep_unsharded_blocks':2},
        'qwen':'Fresh cache/layer objects on every return, stateless length/data_ptr guards and foreign-prefix copy, separate prefill pool, alternating greedy/teacher direct-call setup checks plus refresh/recovery. Still shared mutable master tensors; arbitrary ownership/concurrency not proven. Final-KV check follows second greedy only, not all chains. Runtime fallback handles exceptions, not arbitrary numerical errors. Unique historical mirror cause and exact lifetime/quality guarantees remain unsupported. Empty parameter space; validate exact bytes once.',
        'fsdp':'Parent-forward delegation, default K2 policy on trailing blocks, generated choices1/2/3. Fixed K2 follows actual default and declared primary, not a search or best parameter selection. No vocab-sharded head arithmetic. Revert-to-True assumes original policy and is not arbitrary prior-policy restoration. Memory bound, exact speedup, wire-bound/critical-path and unique GEMM-cause claims unproven; measure full quality/constraint. K3 historical acceptance not evidence for this new code.',
        'decision':'Index-only consumption not demonstrated in either case. Do not claim diagnostic workflow closure or production adoption. Fixed original-byte B/C validation may assess generated implementations independently; no prompt change or repeating replay until pass.'}
    result={'at':datetime.now(timezone.utc).isoformat(),'core':framework_source(),'finished_at':done['finished_at'],
        'frozen_files_verified':len(frozen),'frozen_identities':frozen,'roles':rows,'access_review':access,'code_review':review,
        'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json',result)
    public={k:v for k,v in result.items()if k!='frozen_identities'}
    public['roles']={case:{k:v for k,v in row.items()if k not in ['parameter_space','probe_source_records']}for case,row in rows.items()}
    public['terminal_audit_sha256']=sha(root/'terminal-audit.json');write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'roles':{k:{f:r[f]for f in ['tools','tool_counts','nonzero_tool_exit_or_state']}for k,r in rows.items()},'df':result['df']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
