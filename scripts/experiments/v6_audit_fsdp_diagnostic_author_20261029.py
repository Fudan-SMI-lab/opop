"""Audit original FSDP diagnostic author and the concrete CPU persistence defect."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_fsdp_diagnostic_author_20261029 import build_inputs,INSTRUCTIONS,ADDITIONAL,validate


def audit(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    result=terminal['results']['diagnostic-author'];a=result['agent'];inputs,expected,index=build_inputs(root.parent)
    assert expected==protocol['frozen_identities'] and all(sha(p)==h for p,h in expected.items())
    meta,prompt,files=load(Path(a['generation_snapshots'][0]));assert meta['framework_source']==protocol['core']==framework_source()
    assert INSTRUCTIONS+ADDITIONAL in prompt and a['contract_repairs']==0
    assert set(files)=={'inputs/'+k for k in inputs}|{'inputs/worker-environment.json'}
    for n,v in inputs.items():assert files['inputs/'+n]==(v.encode()if isinstance(v,str)else v)
    assert all((Path(a['sandbox'])/n).read_bytes()==v for n,v in files.items())
    validate(Path(a['sandbox']));assert sha(result['probe'])==result['probe_sha256']=='dd550e29654e55d7eade7483c785f0501872956ab43e7bf4f8afcf7abd076249'
    assert sha(result['plan'])==result['plan_sha256']=='9d33cc021e5cea2488d9e18baeb26585c39a6c12611e1f5de3a3bdcad7ce2fd1'
    private=root/'audit';private.mkdir(exist_ok=True)
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    parts=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],))]
    messages=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from message where session_id=? order by time_created,id',(a['session'],))];db.close()
    write_json(private/'session-parts.json',parts);write_json(private/'session-messages.json',messages)
    tools=[];cpu=[];hits=[]
    forbidden=['/v6-work/diagnostic-author-20261029','/v6-work/diagnostic-author-validation-20261029',
        'trial-4f5fce6e6b','/v6-work/fsdp-continuation-20261028','qwen-joint-relation-20261029','qwen-routed-independent-20261029']
    for p in parts:
        d=p['data']
        if d.get('type')!='tool':continue
        s=d.get('state',{});i=s.get('input',{});o=str(s.get('output',''));cmd=i.get('command','')
        row={'part':p['id'],'tool':d['tool'],'status':s.get('status'),'path':i.get('filePath'),'command':cmd,
            'exit':s.get('metadata',{}).get('exit'),'output_sha256':hashlib.sha256(o.encode()).hexdigest(),'output_bytes':len(o.encode()),
            'error_markers':any(x in o for x in ['command not found','MOCK CHECK FAILURES','Traceback (most recent call last)'])}
        tools.append(row);found=[n for n in forbidden if n in json.dumps(i)+o]
        if found:hits.append({'part':p['id'],'terms':found})
        if '/tmp/opencode/diagcheck' in json.dumps(i):
            file=private/(p['id']+'-cpu.json');write_json(file,{'input':i,'output':o,'exit':row['exit']});cpu.append({'path':str(file),'sha256':sha(file)})
    assert not hits,hits
    target=private/'author-final-cpu-fixture';source=Path('/tmp/opencode/diagcheck')
    if not target.exists():shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    fixture={str(p.relative_to(target)):sha(p)for p in target.rglob('*')if p.is_file()}
    evidence=read(private/'persistence-cpu/result.json');assert evidence['probe_sha256']==result['probe_sha256'] and not evidence['cuda_initialized']
    assert evidence['cases'][0]['result_preserved']
    for c in evidence['cases'][1:]:assert c['original_result']['valid']is False and c['observed_status']=='raised' and c['error_type']=='ValueError' and not c['evidence_files']
    review={'execute_original_on_gpu':False,'reason':'Real-framework CPU fixture proves nonfinite observations convert original valid=false into JSON ValueError with no evidence. Preserve original bytes; one declared real author repair before any GPU validation.',
        'finite_plumbing':'Finite CPU fixture preserved original returned fields; author CPU mock also exercised raised exception and missing entry configuration. No model/GPU/distributed execution was validated.',
        'limitations':['Original adapter.observe already materializes CPU gradients/parameters; these were not compared after first failure, not absent from execution.',
            'Full traversal only persists summaries, selected values and hashes, not full actual tensors. Cannot independently reconstruct every statistic or actual digest from saved output.',
            'top_mismatches selects largest absolute errors over all elements, including within-tolerance elements. CPU fixture demonstrates label ambiguity; it is not a list solely of tolerance violations.',
            'No dtype-equivalence enforcement or complete structural/nonfinite verdict in observer. Original comparator result is the quality decision; scalar formula differs from math.isclose.',
            'observer_wall_s includes the original compare call, not isolated observer overhead. CPU full traversal extends ownership and inter-update intervals; mismatch-dependent work differs across ranks.',
            'First collective delta includes preparation/reset/binding and step work; final counts include later phases when reached. API counts/first_arg_elements are not physical collectives, bytes, bandwidth or critical path. Alias/functional/symmetric-memory bypasses need review.',
            'Before/after math flags straddle prepare policy initialization; equality is not guaranteed or required. Use original evaluator math_policy_unchanged at its actual boundary.',
            'Code catches original BaseException then restores hooks sequentially; it does not implement a finally guarantee against all instrumentation failures, and partially failed installs may not be restored.',
            'Root-cause/determinism guarantees from equal rerun fingerprints, and unchanged-lifetime/microsecond-overhead claims, are unsupported. Proposed rerun/K=0 controls not authorized as automatic next steps.',
            'Independent_reference is metadata only and null. This does not validate source fidelity or generic production adoption.']}
    report={'at':datetime.now(timezone.utc).isoformat(),'core':protocol['core'],'terminal_phase':terminal['phase'],
        'agent':{'session':a['session'],'model':a['model'],'repairs':a['contract_repairs'],'usage':a['usage'],
            'snapshot':a['generation_snapshots'][0],'snapshot_sha256':sha(a['generation_snapshots'][0]),
            'inputs':{n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for n,v in files.items()},
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'settings_sha256':hashlib.sha256(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()).hexdigest(),
            'tool_inventory':dict(Counter(x['tool']for x in tools)),'tools':tools,'later_artifact_hits':hits,
            'access_review':'Complete actual paths/commands/outputs reviewed: original supplied task/source, frozen framework and installed Torch, generated files and own mock only. External filesystem not hermetic; not proof of impossible unlogged access.',
            'session_parts':{'path':str(private/'session-parts.json'),'sha256':sha(private/'session-parts.json')}},
        'generated':{k:result[k]for k in ['probe','probe_sha256','plan','plan_sha256']},'cpu_history':cpu,
        'author_cpu_fixture':{'path':str(target),'files':fixture},
        'author_errors':['rg missing: pipeline still exit0; fallback grep performed.',
            'First mock exited1 because exact Python 5e-6 was compared to FP32 4.999999873689376e-6. Agent changed only test to 1e-12 tolerance; no model quality tolerance change.',
            'Second command mistyped /root/audodl-tmp, suppressed that error, then fallback correct Python mock passed.',
            'Author mock used TemporaryDirectory, so ephemeral outputs no longer exist. Complete command/output/edit history and final mock source retained; no invented raw retention.'],
        'persistence_cpu':{'path':str(private/'persistence-cpu/result.json'),'sha256':sha(private/'persistence-cpu/result.json'),'result':evidence},
        'review':review,'frozen_identities':expected,'scope':'No GPU experiment or optimization acceptance; exact original author output retained, concrete CPU defect warrants narrow separate real request.'}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k not in ['frozen_identities','cpu_history','author_cpu_fixture']}
    public['agent']={k:v for k,v in report['agent'].items()if k!='tools'};public['frozen_files_verified']=len(expected)
    public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'tools':report['agent']['tool_inventory'],'execute_original_on_gpu':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
