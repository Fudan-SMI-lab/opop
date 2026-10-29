"""Audit the real persistence repair and independent CPU boundary checks."""
import argparse
import ast
from collections import Counter
from contextlib import redirect_stderr
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import torch
import torch.distributed as dist
from kernel_optimizer.project import evaluation
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_fsdp_diagnostic_repair_20261029 import build_inputs, INSTRUCTIONS, ADDITIONAL, REPAIR, validate


def strict(path):
    def reject(value):
        raise ValueError('Bare nonfinite JSON token: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


ENTRY = '''import torch
from kernel_optimizer.project import evaluation
def run(context):
    context['calls'].append('original')
    if context.get('before_compare'):
        raise context['sentinel']
    actual={'logits':torch.tensor([1e-4]),'later':torch.tensor([float('-inf')])}
    expected={'logits':torch.zeros(1),'later':torch.zeros(1)}
    check=evaluation.compare(actual,expected,{'atol':1e-6,'rtol':1e-4})
    context['original_result']={'valid':check['passed'],'checks':[check],'samples':[],'metrics':{}}
    if context.get('sentinel') is not None:
        raise context['sentinel']
    return context['original_result']
'''


def extra_cpu(probe, output):
    assert not output.exists()
    output.mkdir(parents=True)
    entry=output/'original_entry.py';entry.write_text(ENTRY)
    rows=[]
    for label, before, raised, writes in [
        ('raise-after-negative-inf',False,True,'normal'),
        ('raise-before-compare',True,True,'normal'),
        ('full-write-fallback',False,False,'fallback'),
        ('all-write-fail-return',False,False,'none'),
        ('all-write-fail-raise',False,True,'none')]:
        spec=importlib.util.spec_from_file_location('independent_repair_probe',probe)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        saved={n:getattr(dist,n,None)for n in m._DIST_FUNCTIONS};comparison=evaluation.compare
        path=output/label;path.mkdir()
        sentinel=RuntimeError('same original exception object') if raised else None
        context={'rank':0,'world_size':1,'device':'cpu','output':str(path),'calls':[],
            'sentinel':sentinel,'before_compare':before,
            'parameters':{'action':'evaluate','bundles':[],
                'diagnostic_context':{'original_entry':str(entry),'original_function':'run','arm':label}}}
        original_write=m._atomic_write_json;write_calls=[]
        def injected(target,payload):
            write_calls.append(str(target))
            if writes=='none' or (writes=='fallback' and not str(target).endswith('.minimal.json')):
                raise OSError('Declared CPU persistence failure injection')
            return original_write(target,payload)
        m._atomic_write_json=injected
        stderr=io.StringIO();error=None;result=None
        with redirect_stderr(stderr):
            try:result=m.run(context)
            except BaseException as exc:error=exc
        assert context['calls']==['original']
        assert error is sentinel
        if not raised:
            assert {k:v for k,v in result.items()if k!='diagnostic_index'}==context['original_result']
            assert result['valid'] is False
        assert evaluation.compare is comparison and all(getattr(dist,n,None)is f for n,f in saved.items())
        files=list(path.glob('*.json'));data=[strict(p)for p in files]
        if writes=='none':
            assert not files and 'evidence persistence failed' in stderr.getvalue()
            if not raised:assert result is context['original_result'] and 'diagnostic_index'not in result
        else:
            assert len(data)==1
            if before:assert not data[0]['compare_calls'] and any('none observed'in x for x in data[0]['absent_observations'])
            if label=='raise-after-negative-inf':
                assert '-Infinity' in json.dumps(data[0]) and data[0]['original_invocation']['status']=='raised'
            if writes=='fallback':assert result['diagnostic_index']==str(files[0]) and data[0]['fallback']
        (path/'stderr.txt').write_text(stderr.getvalue())
        rows.append({'case':label,'original_calls':len(context['calls']),'exception_object_preserved':error is sentinel if raised else None,
            'return_preserved':result is context['original_result'] if writes=='none' and not raised else not raised,
            'hooks_restored':True,'write_attempts':write_calls,'evidence':[{'path':str(p),'sha256':sha(p)}for p in files],
            'stderr_sha256':sha(path/'stderr.txt'),'status':'passed'})
    assert not torch.cuda.is_initialized()
    write_json(output/'result.json',{'cases':rows,'cuda_initialized':False,'entry_sha256':sha(entry),'probe_sha256':sha(probe)})
    return read(output/'result.json')


def audit(root):
    protocol=read(root/'protocol.json');terminal=read(root/'followup-completed.json');assert terminal['phase']=='completed'
    result=terminal['results']['diagnostic-author'];a=result['agent'];inputs,expected,index=build_inputs(Path(protocol['parent']))
    assert expected==protocol['frozen_identities'] and all(sha(p)==h for p,h in expected.items())
    meta,prompt,files=load(Path(a['generation_snapshots'][0]))
    assert meta['framework_source']==protocol['core']==framework_source() and INSTRUCTIONS+ADDITIONAL+REPAIR in prompt
    assert set(files)=={'inputs/'+k for k in inputs}|{'inputs/worker-environment.json'} and a['contract_repairs']==0
    for n,v in inputs.items():assert files['inputs/'+n]==(v.encode()if isinstance(v,str)else v)
    assert all((Path(a['sandbox'])/n).read_bytes()==v for n,v in files.items())
    validate(Path(a['sandbox']));probe=Path(result['probe'])
    assert sha(probe)==result['probe_sha256']=='9495eeaddcb50a0db6375923e37243bff0ded8d5bf97d75e66f83d821330fe8b'
    assert sha(result['plan'])==result['plan_sha256']=='8cc8aa131441bbb1da4d98f71e404aa818c8e7cf3e48fbe4adfcd7a1b2440a9c'
    old=ast.parse(files['inputs/previous-diagnostic.py']);new=ast.parse(probe.read_text())
    definitions=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    od,nd=definitions(old),definitions(new)
    changed=[n for n in od if od[n]!=nd.get(n)];added=[n for n in nd if n not in od]
    assert set(changed)=={'run','_atomic_write_json'} and set(added)=={'_json_safe','_count_nonfinite'}
    cpu=root/'audit/persistence-cpu';fixture=read(cpu/'result.json')
    assert fixture['probe_sha256']==sha(probe) and not fixture['cuda_initialized']
    fields={}
    for c in fixture['cases']:
        assert c['result_preserved'] and c['observed_status']=='returned' and c['compare_restored'] and c['collective_restored']
        d=strict(cpu/c['case']/'diagnostic-rank-0.json');g=next(p for p in d['compare_calls'][0]['full_traversal']['paths']if p['path']=='output.gradients.x')
        assert d['original_invocation']['mirror']['valid']is False
        fields[c['case']]={k:g.get(k)for k in ['max_abs_error','mean_abs_error','all_finite_difference','mismatch_allclose_count']}
    assert fields['nan']['max_abs_error']=='NaN' and fields['nan']['all_finite_difference']is False
    assert fields['inf']['max_abs_error']=='Infinity' and fields['inf']['all_finite_difference']is False
    oldfinite=read(Path(protocol['parent'])/'audit/persistence-cpu/finite/diagnostic-rank-0.json')
    traversal=strict(cpu/'finite/diagnostic-rank-0.json')['compare_calls'][0]['full_traversal']
    original_traversal=oldfinite['compare_calls'][0]['full_traversal']
    assert {k:v for k,v in traversal.items()if k!='observer_wall_s'}=={k:v for k,v in original_traversal.items()if k!='observer_wall_s'}
    extra=extra_cpu(probe,root/'audit/independent-boundaries')
    db=sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro',uri=True)
    parts=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from part where session_id=? order by time_created,id',(a['session'],))]
    messages=[{'id':i,'data':json.loads(d)}for i,d in db.execute('select id,data from message where session_id=? order by time_created,id',(a['session'],))];db.close()
    write_json(root/'audit/session-parts.json',parts);write_json(root/'audit/session-messages.json',messages)
    tools=[];hits=[]
    forbidden=['/v6-work/diagnostic-author-validation-20261029','qwen-joint-relation-20261029','qwen-routed-independent-20261029','trial-4f5fce6e6b','/v6-work/fsdp-continuation-20261028']
    for p in parts:
        d=p['data']
        if d.get('type')!='tool':continue
        s=d['state'];i=s.get('input',{});o=str(s.get('output',''))
        tools.append({'id':p['id'],'tool':d['tool'],'status':s.get('status'),'input':i,'output_sha256':hashlib.sha256(o.encode()).hexdigest(),
            'output_bytes':len(o.encode()),'exit':s.get('metadata',{}).get('exit')})
        found=[n for n in forbidden if n in json.dumps(i)+o]
        if found:hits.append({'part':p['id'],'terms':found})
    assert not hits
    prior=read(Path(protocol['parent'])/'terminal-audit.json')
    limitations=prior['review']['limitations']+[
        'Nonfinite strings are explicit sentinels, never finite errors. NaN comparison count can remain zero; use the false finite flag. No actual FSDP nonfinite observation has been demonstrated.',
        'Fallback write handling also changed. If both writes fail, original outcome survives with stderr but no evidence; missing evidence is a diagnostic failure, not project acceptance.',
        'Agent exception tests checked class/message, not object identity; independent new CPU fixture verifies identical exception objects and all installed module hooks.',
        'Finite traversal numerical/structural/digest fields match old saved CPU case exactly; observer_wall_s differs. Entire evidence is not byte-identical because schema, provenance, timing and encoding metadata changed.',
        'Plan claims equality from torch_equal/finite flags and deterministic fingerprints do not certify nonfinite element identity or full tensors. Original quality remains authoritative.']
    report={'at':datetime.now(timezone.utc).isoformat(),'core':protocol['core'],'terminal_phase':terminal['phase'],
        'generated':{k:result[k]for k in ['probe','probe_sha256','plan','plan_sha256']},
        'agent':{'session':a['session'],'model':a['model'],'repairs':a['contract_repairs'],'usage':a['usage'],
            'assistant_messages':sum(m['data'].get('role')=='assistant'for m in messages),
            'snapshot':a['generation_snapshots'][0],'snapshot_sha256':sha(a['generation_snapshots'][0]),
            'inputs':{n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()}for n,v in files.items()},
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'settings_sha256':hashlib.sha256(json.dumps(meta['tool_model_settings'],sort_keys=True).encode()).hexdigest(),
            'tool_inventory':dict(Counter(t['tool']for t in tools)),'tools':tools,'later_artifact_hits':hits,
            'access_review':'Complete actual tool inputs/outputs reviewed: supplied inputs, own files/tests, named CPU fixture, frozen framework, environment import paths. No later Qwen/FSDP answers observed; external filesystem is not hermetic.',
            'session_parts_sha256':sha(root/'audit/session-parts.json')},
        'author_failures':['Sanitizer mock expected 3 nonfinite values although input contained 4; test corrected.',
            'First fixture lacked project PYTHONPATH; subsequent attempt precreated prohibited output directory. Both failed before fixture cases.',
            'Strict evidence test incorrectly expected finite gradient equality; actual fixture has finite 2e-7 difference. Test corrected, original model quality unchanged.',
            'Extra-case test syntax bracket error corrected; final four cases passed. Agent deleted/repeated its own persistence fixture after successful run; final files and full tool history survive, not all intermediate raw files.'],
        'code_diff':{'changed_definitions':changed,'added_definitions':added,'all_other_definitions_equal':True},
        'persistence_cpu':{'path':str(cpu/'result.json'),'sha256':sha(cpu/'result.json'),'fields':fields,'finite_traversal_equal_old_except_observer_wall_s':True,
            'observer_wall_s_old_new':[original_traversal['observer_wall_s'],traversal['observer_wall_s']]},
        'audit_fixture_error':'Initial audit incorrectly compared observer_wall_s across separate CPU calls. Only this timing differed; assertion narrowed to numerical/structural/digest fields. No generated code change or persistence fixture rerun.',
        'independent_boundaries':{'path':str(root/'audit/independent-boundaries/result.json'),'sha256':sha(root/'audit/independent-boundaries/result.json'),'result':extra},
        'review':{'execute_exact_bytes_once':True,'limitations':limitations,
            'decision':'Exact repaired bytes suitable for one declared derived-baseline/original-candidate two-rank observation. CPU persistence passed; no GPU/source-fidelity/performance acceptance yet. No production adoption.'},
        'frozen_identities':expected,'scope':'One real narrow repair plus independent CPU audit; original probe/failed cases preserved. No model/GPU/distributed run by this audit.'}
    assert all(sha(p)==h for p,h in expected.items()) and not torch.cuda.is_initialized()
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k!='frozen_identities'};public['agent']={k:v for k,v in report['agent'].items()if k!='tools'}
    public['frozen_files_verified']=len(expected);public['audit']={'path':str(root/'terminal-audit.json'),'sha256':sha(root/'terminal-audit.json')}
    write_json(root/'public-audit.json',public)
    print(json.dumps({'audit_sha256':sha(root/'terminal-audit.json'),'cpu_cases':3+len(extra['cases']),'tools':report['agent']['tool_inventory'],'execute_exact_bytes_once':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);audit(p.parse_args().root)
