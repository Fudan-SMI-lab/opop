"""Audit the single original-failure diagnostic author; no GPU execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_diagnostic_author_20261029 import build_inputs, validate


def audit(root):
    protocol=read(root/'protocol.json'); terminal=read(root/'followup-completed.json')
    assert terminal['phase']=='completed' and protocol['core']==framework_source()
    result=terminal['results']['diagnostic-author']; agent=result['agent']
    inputs, expected, index=build_inputs(root.parent)
    assert expected==protocol['frozen_identities'] and all(sha(p)==h for p,h in expected.items())
    meta,prompt,files=load(Path(agent['generation_snapshots'][0]))
    assert meta['framework_source']==protocol['core'] and agent['contract_repairs']==0
    assert set(files)=={'inputs/'+k for k in inputs}|{'inputs/worker-environment.json'}
    for name,data in inputs.items():
        assert files['inputs/'+name]==(data.encode() if isinstance(data,str) else data)
    assert all((Path(agent['sandbox'])/name).read_bytes()==data for name,data in files.items())
    validate(Path(agent['sandbox']))
    assert sha(result['probe'])==result['probe_sha256']=='16af16d674dc05872e5ff835a5b165c5473819ed9b4e743be12bb2ef64e74110'
    assert sha(result['plan'])==result['plan_sha256']=='12e6a7e991e2f9780cf5cf1dfe855711ad2335a4ebcf827ab18a76a3fc2138dc'
    private=root/'audit'; private.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    parts=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from part where session_id=? order by time_created,id',(agent['session'],))]
    messages=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from message where session_id=? order by time_created,id',(agent['session'],))]
    db.close()
    write_json(private/'session-parts.json',parts);write_json(private/'session-messages.json',messages)
    tools=[]; probes=[]; hits=[]
    forbidden=['qwen-joint-relation-20261029','qwen-joint-feedback-20261029','qwen-joint-feedback-routed-20261029',
        'qwen-routed-independent-20261029','qwen-attention-fixed-20261029','relation-observations.json',
        'fixed_continuous_full_output_diagnostic','d5bd32d47d6b4d8165f617810a4dbeabed5c5c850932e6f74aaafd9adaa4bf2b',
        'ce85acc72901aa6bfe77ca88b28977a8a28d83ff2dbd94dd566076e95d4ceba3']
    for part in parts:
        p=part['data']
        if p.get('type')!='tool':continue
        st=p.get('state',{}); inp=st.get('input',{}); out=str(st.get('output','')); cmd=inp.get('command','')
        row={'part':part['id'],'tool':p['tool'],'status':st.get('status'),'path':inp.get('filePath'),
            'command':cmd,'exit':st.get('metadata',{}).get('exit'),'output_bytes':len(out.encode()),
            'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'traceback':'Traceback (most recent call last)'in out}
        tools.append(row)
        found=[term for term in forbidden if term in json.dumps(inp)+out]
        if found:hits.append({'part':part['id'],'terms':found})
        if '/tmp/opencode/diagtest' in json.dumps(inp):
            dst=private/(part['id']+'-cpu-probe.json');write_json(dst,{'input':inp,'output':out,'status':st.get('status'),'exit':row['exit']})
            probes.append({'path':str(dst),'sha256':sha(dst)})
    assert not hits,hits
    # Preserve only what still exists. Agent deleted previous mock run outputs;
    # command/output history survives, not every early temporary artifact.
    source=Path('/tmp/opencode/diagtest'); target=private/'final-cpu-fixture'
    if source.exists() and not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    fixture={str(p.relative_to(target)):sha(p)for p in target.rglob('*')if p.is_file()}
    assert fixture
    import torch
    from kernel_optimizer.project.evaluation import corrupt,compare
    reference=torch.load(index['controls_reference']['path'],map_location='cpu',weights_only=True)[0]
    damaged=corrupt(reference)
    differences=[k for k in reference if not (torch.equal(reference[k],damaged[k])if isinstance(reference[k],torch.Tensor)else reference[k]==damaged[k])]
    negative=compare(damaged,reference,json.loads(inputs['task.json'])['quality'])
    assert differences==['generated_tokens'] and not negative['passed']
    report={'at':datetime.now(timezone.utc).isoformat(),'core':protocol['core'],'terminal_phase':terminal['phase'],
        'scope':'Single real diagnostic author, syntax/schema and CPU mocks only. No GPU diagnostic execution or production adoption.',
        'agent':{'session':agent['session'],'model':agent['model'],'repairs':agent['contract_repairs'],'usage':agent['usage'],
            'snapshot':agent['generation_snapshots'][0],'snapshot_sha256':sha(agent['generation_snapshots'][0]),
            'input_files':{k:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for k,v in files.items()},
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'settings_sha256':hashlib.sha256(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()).hexdigest(),
            'tool_inventory':dict(Counter(t['tool']for t in tools)),'tools':tools,'later_evidence_hits':hits,
            'access_review':'Actual tool paths, command source and outputs inspected: supplied inputs, indexed original requests/source/references, framework source, generated files and CPU fixture only. Filesystem not hermetic; no absolute unobservability claim.',
            'private_parts':{'path':str(private/'session-parts.json'),'sha256':sha(private/'session-parts.json')},
            'private_messages':{'path':str(private/'session-messages.json'),'sha256':sha(private/'session-messages.json')}},
        'generated':{k:result[k]for k in ['probe','probe_sha256','plan','plan_sha256']},
        'cpu_probes':probes,'final_cpu_fixture':{'path':str(target),'files':fixture},
        'cpu_errors_preserved':['Initial event(kind) parameter collision raised TypeError; agent corrected signature inside author session.',
            'Binding snapshot test raised KeyError(status); agent fixed that test. Reference summary status was then overwritten by facts.status; agent corrected dictionary merge order in diagnostic.py for both frozen and independent references. Final mock checks passed.',
            'Caught traceback and pipelines can exit zero. Mock framework does not reproduce actual signature dispatch/full CUDA workload.',
            'Early mock run directories were deleted by agent commands; final fixture plus complete session tool histories retained.'],
        'framework_negative_cpu':{'changed_fields':differences,'result':negative,'scope':'Token corruption, NOT independent logit sensitivity validation.'},
        'review':{'execute_exact_bytes_once':True,'execution_boundary':'Original evaluator called once, original exception re-raised after persistence; no model rewrite, reset or precision/quality changes found. Actual BindingSession.install returns list as expected.',
            'limitations':['Event kind overwritten by greedy_run/teacher_forced_replay; begin/end must be distinguished by fields/order, not invented labels.',
                'CPU copies, digests, retained binding/state/exception frames and allocator queries perturb synchronization, allocations and lifetimes; timings never accepted.',
                'No after-observe output snapshot. Independent source summary compares only first greedy; raw captures permit separate full audit.',
                'Reference and pairwise summaries do not reproduce full GPU BF16 assert_close tolerance semantics; source logits retain original quality threshold, not advisory acceptance.',
                'Framework corrupt negative changes generated_tokens, not logits. No finite claim without raw audit.',
                'Route labels/missing labels are not native or fallback proof; physical binding does not override teacher-stride signature mismatch.',
                'Restored hooks do not certify arbitrary global ownership/concurrency or absence of observer effects; import module and collector references persist.',
                'Passed_with_evidence only describes absence of exception; actual worker/evaluator quality must be checked separately.',
                'Authored narrative overstates unchanged lifetimes, path guarantees and root-cause implications. These claims are not adopted.']},
        'frozen_identities_verified':expected}
    write_json(root/'terminal-audit.json',report)
    public={k:report[k]for k in ['at','core','terminal_phase','scope','generated','cpu_errors_preserved','framework_negative_cpu','review']}
    public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    public['agent']={k:v for k,v in report['agent'].items()if k not in ['tools','private_messages']}
    public['frozen_files_verified']=len(expected);public['cpu_fixture_files']=len(fixture)
    public['tool_errors']=[{k:t[k]for k in ['part','tool','status','exit','traceback','output_sha256']}for t in tools if t['traceback']or t['exit']not in [None,0]]
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'tools':public['agent']['tool_inventory'],'negative_fields':differences,'fixture_files':len(fixture)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
