"""Recompute retained attention tensors/traces and audit a real candidate role."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sqlite3
import torch
from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def digest(value): return hashlib.sha256(value.encode()).hexdigest()


def main(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json')
    evidence=read(root/'run/attention-evidence.json');previous=read(Path(protocol['parent'])/'run/first-decode.json')
    assert terminal['phase']=='completed' and protocol['core']==framework_source()==evidence['core']
    assert sha(Path(protocol['parent'])/'run/first-decode.json')==evidence['preceding_report_sha256']
    assert all(sha(p)==h for p,h in protocol['frozen_identities'].items())
    def tensor(row):
        assert sha(row['path'])==row['sha256']
        return torch.load(row['path'],map_location='cpu',weights_only=True)
    raw={};rows={}
    for name,row in evidence['rows'].items():
        raw[name]=tensor(row['first_attention_raw'])
        full=tensor(row['full_decode_output']);old=tensor(previous['arms'][name+'-eager']['output_raw'])
        assert all(torch.equal(full[k],old[k]) for k in full)
        assert compare(full,old,{'atol':.02,'rtol':.02})==row['comparison_to_previous_eager']
        assert row['matches_previous_unobserved_eager_values']
        for key in ['q','k','v']:
            assert list(raw[name][key].shape)==row['first_interface'][key]['shape']
            assert str(raw[name][key].dtype)==row['first_interface'][key]['dtype']
            assert torch.isfinite(raw[name][key]).all()
        assert torch.isfinite(raw[name]['output']).all()
        assert torch.equal(tensor(row['profile']['raw_output']),raw[name]['output'])
        assert row['profile']['output_exact_to_full_forward_hook']
        profile=row['profile'];assert sha(profile['path'])==profile['sha256']
        events=read(profile['path'])['traceEvents']
        ops=sorted({e.get('name','')for e in events if e.get('cat')=='cpu_op'and ('attention'in e.get('name','')or 'softmax'in e.get('name',''))})
        kernels=sorted({e.get('name','')for e in events if e.get('cat')=='kernel'})
        assert ops==profile['cpu_attention_ops'] and kernels==profile['kernels']
        rows[name]=row
    d,s=raw['dynamic'],raw['static'];n=d['k'].shape[2];heads=s['k'].shape[1]//d['k'].shape[1]
    logical={'q_equal':torch.equal(d['q'],s['q']),'dynamic_k_shape':list(d['k'].shape),'static_k_shape':list(s['k'].shape),
        'k_valid_equal':torch.equal(d['k'].repeat_interleave(heads,dim=1),s['k'][:,:,:n]),
        'v_valid_equal':torch.equal(d['v'].repeat_interleave(heads,dim=1),s['v'][:,:,:n]),'head_replication':heads,
        'first_attention_output_max_abs_error':float((d['output'].float()-s['output'].float()).abs().max())}
    assert logical==evidence['logical_inputs']
    mask=s['tensor_options']['attn_mask']
    assert mask.dtype==torch.bool and mask[:,:,:,:n].all() and not mask[:,:,:,n:].any()
    assert not s['k'][:,:,n:].count_nonzero() and not s['v'][:,:,n:].count_nonzero()
    assert not d['tensor_options'] and d['options']['attn_mask'] is None
    candidate_protocol=read(root/'candidate-protocol.json')
    saved,prompt,original=load(Path(candidate_protocol['snapshot']))
    result=terminal['results']['candidate-continuation'];agent=result['agent'];sandbox=Path(agent['sandbox'])
    actual,ap,files=load(Path(agent['generation_snapshots'][0]))
    assert set(original)==set(files) and len(files)==33
    changed=[k for k in files if files[k]!=original[k]];assert changed==['inputs/previous-feedback.txt']
    old=original[changed[0]].decode();new=files[changed[0]].decode();field=candidate_protocol['added_field']
    obj=json.loads(new);assert obj.pop(field)==candidate_protocol['facts'] and obj==json.loads(old)
    assert new[:old.rfind('}')]==old[:old.rfind('}')]
    assert ap==prompt and actual['tool_model_settings']==saved['tool_model_settings'] and actual['framework_source']==saved['framework_source']==framework_source()
    assert all((sandbox/k).read_bytes()==v for k,v in files.items())
    assert sha(sandbox/'candidate.py')==result['candidate_sha256']
    proposal=read(sandbox/'proposal.json');assert proposal==result['proposal'] and proposal['parameter_space']=={}
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    parts=[{'part_id':i,'data':json.loads(v)}for i,v in db.execute('select id,data from part where session_id=? order by time_created,id',(agent['session'],))];db.close()
    audit=root/'audit';audit.mkdir(exist_ok=True);private=audit/'candidate-session-parts.json'
    if private.exists():assert read(private)==parts
    else:write_json(private,parts)
    tools=[];probes=[]
    for row in parts:
        p=row['data']
        if p.get('type')!='tool':continue
        st=p.get('state',{});inp=st.get('input',{});command=inp.get('command','');out=str(st.get('output',''))
        tools.append({'part':row['part_id'],'tool':p['tool'],'status':st.get('status'),'exit':st.get('metadata',{}).get('exit'),
            'path':inp.get('filePath'),'command_preview':command[:350],
            'input_sha256':digest(json.dumps(inp,sort_keys=True)),'output_sha256':digest(out),'output_bytes':len(out.encode()),
            'traceback':'Traceback (most recent call last)'in out,'fact_field_in_output':field in out})
        if any(x in command for x in ['# CPU-level sanity','# CPU-mocked runner','patch context restores','from transformers.utils import (']):
            path=audit/(row['part_id']+'-probe.json')
            write_json(path,{'input':inp,'output':out,'status':st.get('status'),'exit':st.get('metadata',{}).get('exit')})
            probes.append({'part':row['part_id'],'path':str(path),'sha256':sha(path),'exit':st.get('metadata',{}).get('exit'),
                'scope':'Inline CPU import/view/fake-graph/patch probe; no full CUDA model or timing validation.'})
    import transformers.masking_utils as mu
    import transformers.utils.import_utils as iu
    import transformers.integrations.sdpa_attention as sdpa
    library={}
    for fn in [iu.is_tracing,mu._ignore_causal_mask_sdpa,sdpa.use_gqa_in_sdpa]:
        path=Path(inspect.getfile(fn));library[fn.__name__]={'path':str(path),'file_sha256':sha(path),'source':inspect.getsource(fn)}
    write_json(audit/'installed-library-functions.json',library)
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':framework_source(),
        'terminal_sha256':sha(root/'followup-completed.json'),'attention_report_sha256':sha(root/'run/attention-evidence.json'),
        'frozen_identities_verified':protocol['frozen_identities'],'all_raw_tensors_and_trace_summaries_recomputed':True,
        'attention':{'logical_inputs':logical,'mask_true_valid_prefix_false_padding':True,'padded_kv_zero':True,'rows':rows},
        'initial_input_audit':{'changed':changed,'files':{k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}for k,v in files.items()},
            'original_bytes':len(old.encode()),'new_bytes':len(new.encode()),'old_fields_and_prefix_preserved':True,
            'prompt_sha256':digest(prompt),'settings_sha256':digest(json.dumps(actual['tool_model_settings'],sort_keys=True)),
            'other_inputs_prompt_settings_core_equal':True,'inputs_unchanged_after_role':True},
        'role':{'agent':agent,'candidate_sha256':result['candidate_sha256'],'proposal_sha256':sha(sandbox/'proposal.json'),
            'tools':tools,'counts':dict(Counter(t['tool']+':'+str(t['status'])for t in tools)),
            'tracebacks':sum(t['traceback']for t in tools),'nonzero_exits':[t['part']for t in tools if t['exit']not in [None,0]],
            'probes':probes,'private_session':{'path':str(private),'sha256':sha(private)}},
        'installed_library_functions':{k:{x:y for x,y in v.items()if x!='source'}for k,v in library.items()},
        'review':{
            'actual_reads':'prt_184718063001M7pocV241p4d6c printed the new facts; prt_18474af21001tCek7cMlJOEqsU read linked attention report. Agent also read first-decode report, diagnostic script, request and installed Transformers mask/GQA/tracing source. No raw tensor load or GPU model probe in recorded commands.',
            'mechanism':'New per-length DynamicCache graphs. Temporary masking_utils.is_tracing replacement during capture omits CUDA-capture term; original function restored on context exit. Master KV buffers refreshed per invocation, tails updated per step, static logits cloned, caller mirror cache views. Setup eager vs graph logits and all layer tails tested bitwise by candidate code, not yet executed on full GPU here.',
            'source_support':'Installed is_tracing includes stream capture; non-XPU _ignore_causal_mask_sdpa disallows mask skipping while tracing; CUDA use_gqa_in_sdpa requires mask None. This supports a specific capture-dependent path hypothesis. Joint interface differences and source reading do not isolate unique historical numerical causality.',
            'limitations':'Claims ruling out all lifetime issues, kernel-for-kernel equivalence, guaranteed full32-token quality/TTFT and causal host cost exceed this evidence. Framework signature still excludes teacher stride even if candidate guard accepts it. Shared-pool multi-graph state/mirror views need real complete invocation checks. Runtime global patch restored by code but multithread composability not proven.',
            'probe_failure':'One import probe failed because transformers.utils does not export is_jit_tracing; final code imports from utils.import_utils. A grep ended exit2. CPU fake graph/views/patch probes are not real CUDA quality. Preserve all commands/edits/errors.',
            'next':'Validate exact generated code with unchanged full request in isolated fixed endpoints. Empty parameters, no manual code fix, no new formal acceptance or production change.'},
        'scope':'Diagnostic controls and one real generation, not autonomous search/full candidate acceptance. Raw/private state stays server-side.'}
    write_json(root/'terminal-audit.json',report)
    print(json.dumps({'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json'),'bytes':(root/'terminal-audit.json').stat().st_size,
        'logical_inputs':logical,'candidate_sha256':result['candidate_sha256'],'tools':report['role']['counts'],'tracebacks':report['role']['tracebacks'],
        'nonzero_exits':report['role']['nonzero_exits'],'probes':len(probes)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
