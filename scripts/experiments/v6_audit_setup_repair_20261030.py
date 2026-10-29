"""Audit one generated retention repair, actual tools and independent CPU evidence."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_qwen_setup_diagnostic_repair_20261030 import build_inputs, INSTRUCTIONS, EXTRA, REPAIR


def main(root):
    terminal=read(root/'followup-completed.json');protocol=read(root/'protocol.json')
    a=terminal['results']['diagnostic-author'];agent=a['agent'];sandbox=Path(agent['sandbox'])
    assert terminal['phase']=='completed' and agent['contract_repairs']==0 and protocol['core']==framework_source()
    inputs,frozen,_=build_inputs(root.parent)
    assert frozen==protocol['frozen_identities'] and all(sha(p)==h for p,h in frozen.items())
    meta,prompt,files=load(Path(agent['generation_snapshots'][0]))
    assert meta['fresh_session'] and meta['framework_source']==protocol['core'] and INSTRUCTIONS+EXTRA+REPAIR in prompt
    assert set(files)=={'inputs/'+n for n in inputs}|{'inputs/worker-environment.json'}
    for n,v in inputs.items():assert files['inputs/'+n]==(v.encode() if isinstance(v,str)else v)
    assert all((sandbox/n).read_bytes()==v for n,v in files.items())
    parent=read(root.parent/'qwen-setup-author-20261030/followup-completed.json')['results']['diagnostic-author']
    oldmeta,_,_=load(Path(parent['agent']['generation_snapshots'][0]))
    assert meta['tool_model_settings']==oldmeta['tool_model_settings']
    for k in ['probe','plan']:assert sha(a[k])==a[k+'_sha256']
    history=root/'terminal-private-history';tools=read(history/'tools.json');private=read(history/'history.json')
    initial={};inventory=[]
    for i,t in enumerate(tools,1):
        inp=t['input'];out=t.get('output','')
        inventory.append({'number':i,'part':t['part'],'tool':t['tool'],'input':inp,
            'status':t.get('status'),'exit':t.get('metadata',{}).get('exit'),
            'output_sha256':hashlib.sha256(out.encode()).hexdigest(),'output_bytes':len(out.encode())})
        if t['tool']=='write' and inp.get('filePath') in [a['probe'],a['plan']]:
            p=Path(inp['filePath']);assert p.read_text()==inp['content']
            dest=history/('initial-'+p.name);dest.write_text(inp['content']);initial[p.name]=sha(dest)
        if t['tool'] in ['write','edit'] or t.get('metadata',{}).get('exit') not in [0,None]:
            write_json(history/f'tool-{i}.json',t)
    write_json(history/'tool-review-index.json',inventory)
    fixture=history/'final-author-cpu-fixture';assert not fixture.exists()
    shutil.copytree('/tmp/opencode/diag-repair-checks',fixture)
    old=ast.parse((sandbox/'inputs/previous-diagnostic.py').read_text());new=ast.parse(Path(a['probe']).read_text())
    oldtop={n.name:ast.dump(n,include_attributes=False)for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newtop={n.name:ast.dump(n,include_attributes=False)for n in new.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    changed=[k for k in oldtop if oldtop[k]!=newtop.get(k)]
    assert changed==['_Recorder']
    cpu=read(root/'audit/independent-cpu/result.json')
    assert cpu['status']=='passed' and cpu['probe_sha256']==a['probe_sha256'] and not cpu['cuda_initialized']
    report={'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],'core':protocol['core'],
        'frozen_files':len(frozen),'input_files':len(files),'explicit_bytes':sum(len(v.encode()if isinstance(v,str)else v)for v in inputs.values()),
        'input_sizes':protocol['input_sizes'],'initial_inputs_byte_equal':True,'tool_model_settings_same_as_prior_author':True,
        'snapshot_sha256':sha(agent['generation_snapshots'][0]),'agent':{k:agent[k]for k in ['session','model','usage','contract_repairs']},
        'assistant_messages':sum(m['data'].get('role')=='assistant'for m in private['message']),
        'tool_counts':dict(Counter(t['tool']for t in tools)),'initial_generated_sha256':initial,
        'generated':{k:a[k]for k in ['probe','probe_sha256','plan','plan_sha256']},
        'generated_bytes':{k:Path(a[k]).stat().st_size for k in ['probe','plan']},
        'changed_top_level_functions_or_classes':changed,'history_sha256':sha(history/'history.json'),'tools_sha256':sha(history/'tools.json'),
        'final_author_fixture_files':{str(p.relative_to(fixture)):sha(p)for p in fixture.rglob('*')if p.is_file()},
        'independent_cpu':cpu,
        'review':{
            'execute_exact_bytes_once':True,
            'reason':'Original late-failure sequence now saves failure with one declared eviction; independent mixed overflow, caught installation, return/exception identity and BaseException boundaries pass. Suitable for one declared B/C observational validation, not production adoption or quality/performance acceptance.',
            'actual_access':'Complete recorded paths, commands, outputs and CPU sources reviewed. Supplied inputs, named CPU fixture/results, original indexed baseline entry and current framework source only; no observed later accepted implementation, developer terminal/public audit or other-case diagnostic answer reads. Candidate lines440-700 actually read; approximate setup-call count independently derived by author, not supplied in new failure input. Nonhermetic filesystem, no absolute absence proof.',
            'author_errors':'Wrong installed path; rg missing despite pipeline exit0; first CPU suite had invented line0 expectation, two missing-rtol ValueErrors and bound-method identity assertion. Later suite still expected line0. Agent changed only its CPU fixture, final10 checks pass; tool38 reran after pass with suppressed output and printed CPU checks all pass. Generated diagnostic/plan each written once with no later edits. Final fixture/evidence/full history retained, prior overwritten temporary outputs not all available.',
            'test_limits':'Author real BindingSession integration uses diagnostic_passthrough, not candidate installation; after-restore test calls original assert, not dormant retained reference closure. Independent corrected integration actually installs a tiny synthetic CPU builder with a caught late failure. All CPU mechanics, not original model/GPU quality.',
            'retention':'128 detailed records/window, at most32 failures; later failures evict oldest retained success, save stub and exact sequence. More than32 failures counted but not retained. Unretained64site keys plus other (65possible), reference first2048 calls unchanged. No message or specific candidate-error routing. Per-window scope only, not universal resource bound on all possible error/window counts.',
            'timing_change':'Detail now collected AFTER original comparator returns/raises, and skipped for unretained successes. Thus copy/synchronization schedule changes relative to old observer; claim retention-only/no timing change is false. Comparator called once, original outcome preserved in examined fixtures; no model replay.',
            'limitations':['Summary/top64finite violations only, no complete operands/digests; cannot independently reconstruct all aggregate stats from saved raw.',
                'CPU FP32 setup comparison against request tolerance is not original full-quality/source proof. Nonfinite excluded from finite violation counts; zero not pass. Same-numel shapes may flatten.',
                'Four CPU copies across describe/statistics for two operands, not two; reductions/intermediates and exact memory/overhead unmeasured. Dormant reference closures keep recorder and add Python frames.',
                'No universal zero CUDA allocation/allocator/lifetime/concurrency guarantee. Metadata exceptions can affect paths; no blanket observer-error immunity.',
                'Reference first-layer metadata/object IDs and length are partial provenance, not all-layer values/aliases or unique cause. Setup interiors execute; only outer parent path was previously quality-validated.',
                'Plan calls full operands but saves summaries; promises all comparisons yet caps apply. Baseline pass does not bound observer effect; matching routes does not prove all internals identical.',
                'Hashing occurs before evaluator, not at evidence write. Plan ~1e-5 dormant overhead and <1.5MB target not measured guarantees. No independent_reference supplied.',
                'Proposed additional profiling/library probes not adopted. Cache advancement/rounding/unique cause remain conditional; no performance/quality acceptance from CPU checks.'],
            'fixture_error':cpu['fixture_error'],
            'production':'Core/prompt/sampler/allocator/quality/acceptance unchanged, closed studies remain closed.'}}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k!='final_author_fixture_files'}
    public['final_author_fixture_count']=len(report['final_author_fixture_files']);public['terminal_audit_sha256']=sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'generated_bytes':report['generated_bytes'],'assistant':report['assistant_messages'],'tools':report['tool_counts'],'frozen':len(frozen),'execute_exact_bytes_once':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
