"""Audit exact author outputs and retained CPU evidence; no GPU execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_qwen_setup_author_20261030 import build_inputs, validate


def apply_recorded_diff(text, patch):
    """Replay actual tool diff, including edits matched fuzzily by the tool."""
    source = text.splitlines(keepends=True); output = []; cursor = 0
    lines = patch.splitlines(keepends=True); i = 0
    while i < len(lines):
        match = re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@', lines[i])
        if not match:
            i += 1; continue
        start = int(match.group(1))-1
        assert start >= cursor
        output.extend(source[cursor:start]); cursor = start; i += 1
        while i < len(lines) and not lines[i].startswith('@@'):
            line = lines[i]; i += 1
            if line.startswith((' ', '-')):
                assert source[cursor] == line[1:]
                cursor += 1
            if line.startswith((' ', '+')): output.append(line[1:])
    return ''.join(output+source[cursor:])


def main(root):
    terminal = read(root/'followup-completed.json'); protocol = read(root/'protocol.json')
    result = terminal['results']['diagnostic-author']; agent = result['agent']
    assert terminal['phase'] == 'completed' and agent['contract_repairs'] == 0
    assert framework_source() == protocol['core']
    inputs, expected, index = build_inputs(root.parent)
    assert expected == protocol['frozen_identities'] and all(sha(p) == h for p,h in expected.items())
    meta, prompt, files = load(Path(agent['generation_snapshots'][0]))
    assert meta['framework_source'] == protocol['core'] and meta['fresh_session']
    assert set(files) == {'inputs/'+n for n in inputs}|{'inputs/worker-environment.json'}
    for n,v in inputs.items(): assert files['inputs/'+n] == (v.encode() if isinstance(v,str) else v)
    sandbox = Path(agent['sandbox']); validate(sandbox)
    assert all((sandbox/n).read_bytes() == v for n,v in files.items())
    assert sha(result['probe']) == result['probe_sha256'] == '8ab9078e71390b240d8a471cc765084a58b9681d3170f6a8a4050a7221fb2ad4'
    assert sha(result['plan']) == result['plan_sha256'] == 'c0dc33c969b45437517d2f2282138a650a4f62f2014c4b06b0dc992decf097c8'
    private = root/'terminal-private-history'; tools = read(private/'tools.json'); history = read(private/'history.json')
    initial = {}; reconstructed = {}; inventory = []; full_outputs = []
    for i,t in enumerate(tools, 1):
        inp = t['input']; out = t.get('output',''); path = inp.get('filePath')
        inventory.append({'number':i,'part':t['part'],'tool':t['tool'],'input':inp,
            'exit':t.get('metadata',{}).get('exit'),'output_sha256':hashlib.sha256(out.encode()).hexdigest(),
            'output_bytes':len(out.encode()),'traceback':'Traceback (most recent call last)' in out})
        if t['tool'] in ['write','edit'] or 'python - <<' in inp.get('command',''):
            write_json(private/(f'tool-{i}.json'), t)
        for raw in re.findall(r'/root/[^\s]+/tool-output/[^\s]+', out):
            source = Path(raw.strip()); assert source.is_file()
            target = private/source.name
            if not target.exists(): shutil.copy2(source,target)
            full_outputs.append({'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size})
        if path in [str(sandbox/'diagnostic.py'), str(sandbox/'diagnostic-plan.json')]:
            if t['tool'] == 'write':
                initial[path] = inp['content']; reconstructed[path] = inp['content']
            if t['tool'] == 'edit':
                if reconstructed[path].count(inp['oldString']) == 1:
                    reconstructed[path] = reconstructed[path].replace(inp['oldString'],inp['newString'])
                else:
                    reconstructed[path] = apply_recorded_diff(reconstructed[path],t['metadata']['diff'])
    for p,text in reconstructed.items():
        assert Path(p).read_text() == text
        (private/('initial-'+Path(p).name)).write_text(initial[p])
    fixture = private/'final-author-cpu-fixture'; fixture.mkdir(exist_ok=True)
    for name in ['test_install_window.py','test_run_contract.py','fake_candidate.py','evidence.json','orig_ok.py','orig_boom.py','out_ok','out_err','out_nan']:
        source = Path('/tmp/opencode/diagtest')/name; target = fixture/name
        if not target.exists():
            if source.is_dir(): shutil.copytree(source,target)
            else: shutil.copy2(source,target)
    write_json(private/'tool-review-index.json',inventory)
    cpu = read(root/'audit/independent-cpu-cap/result.json')
    assert cpu['late_failure']['failure_operands_or_outcome_saved'] is False
    assert not cpu['cuda_initialized'] and cpu['probe_sha256'] == result['probe_sha256']
    report = {'at':datetime.now(timezone.utc).isoformat(),'finished_at':terminal['finished_at'],
        'core':protocol['core'],'frozen_files':len(expected),'input_files':len(files),'explicit_bytes':sum(len(v.encode() if isinstance(v,str) else v)for v in inputs.values()),
        'snapshot_sha256':sha(agent['generation_snapshots'][0]),'initial_inputs_byte_equal':True,
        'agent':{k:agent[k] for k in ['session','model','usage','contract_repairs']},
        'assistant_messages':sum(m['data'].get('role')=='assistant' for m in history['message']),
        'tool_counts':dict(Counter(t['tool'] for t in tools)),
        'tool_errors':[{k:t[k] for k in ['number','part','tool','exit','traceback','output_sha256']} for t in inventory if t['traceback'] or t['exit'] not in [None,0]],
        'generated':{k:result[k] for k in ['probe','probe_sha256','plan','plan_sha256']},
        'initial_generated_sha256':{Path(p).name:hashlib.sha256(x.encode()).hexdigest() for p,x in initial.items()},
        'full_history_sha256':sha(private/'history.json'),'tools_sha256':sha(private/'tools.json'),
        'truncated_outputs_preserved':full_outputs,
        'final_cpu_fixture':{str(p.relative_to(fixture)):sha(p) for p in fixture.rglob('*') if p.is_file()},
        'independent_cpu':cpu,
        'audit_fixture_error':'First audit assumed every edit.oldString literally matched. Tool45 accepted a fuzzy match with a trailing-parenthesis discrepancy; its actual unified diff records the edit. Second audit used all metadata diffs, but tool61 display diff normalized indentation. Corrected auditor uses exact input replacement when available and actual diff for fuzzy tool45, verifying final bytes. No author/model/CPU fixture rerun or generated-code change.',
        'review':{
            'access':'Complete recorded commands, paths, outputs and probe sources reviewed: supplied inputs, indexed original artifacts, installed framework/library source, generated files and own CPU fixtures. No observed later audit/accepted implementation/other diagnostic answer reads. Shared /tmp/opencode/diagtest contained older files; current tests used own rewritten named fixtures, real framework traceback/import paths, no observed older fixture import. Nonhermetic filesystem, not absolute absence proof.',
            'errors':'Wrong installed path; rg absent despite pipeline exit0; nonexistent DynamicCacheLayer; DynamicCache(layers=...) rejected; mock None cache; own git log empty branch; malformed initial persistence syntax edited before compile; mock shape/reference-ID failures and missing result; PosixPath.sep bug fixed by agent; mock max1 versus1.0000001192 equality changed to1e-5 in CPU test only. Full history and final fixtures retained, repeated tests overwrote earlier temporary outputs.',
            'test_limits':'Final tiny CPU BindingSession/real DynamicCache and stub-entry tests pass, not full candidate/GPU quality. Dormant test merely compares two reports without invoking dormant wrapper. Edge restore assert uses or True; separate install fixture actually checks restoration. Agent exception test checked type/message, independent fixture checks same object and once-only entry. No CUDA initialization in independent checks.',
            'cap_defect':'Exact probe records first128 assertions, including successes, then drops all later operands/outcomes. CPU fixture128passes+1failure gives129total/128saved successes/overflow1 and no failure record; original failure preserved. Known candidate code if reaching recorded exact chainE has1fused failure +31exact per-step +72KV +31A +31B +31D +1E =198 comparisons; E exceeds cap. Conditional code count, not a new model trace.',
            'decision':'Do not launch GPU with this probe: concrete retention gap affects required late comparison. Preserve exact output; one narrow real repair may address bounded late-failure retention using CPU fixture, no handwritten fix, model rerun, tolerance change or production adoption.',
            'statistics':'Saved summaries/top<=64 true finite violations, no complete raw operands or digests; cannot independently recompute all aggregate statistics. CPU FP32 elementwise request tolerance on setup operands is not original full-observation quality. NaN/Inf excluded from finite violation counts and separately counted; zero violations does not mean pass. Same-numel/different-shape flattened stats and argmax across <=2D not generic structural/row quality.',
            'ownership':'Per-site _describe_operand copies each operand then _comparison_stats copies both again; claims only two copies/~2.5MB peak not established. Reference closures persist after install as dormant extra Python calls and keep recorder state. CPU synchronization/allocations/hash reads/frame references affect state/lifetimes; no proof of zero CUDA allocator impact or1e-5 overhead. Hashing occurs before evaluator, not at evidence write as plan claims.',
            'scope':'Plan says interiors never execute, but they execute during setup; outer measured path is parent fallback. Baseline matching does not bound observer effects; matching U/O outcomes do not establish identical internals. Reference metadata only first-layer keys/length/object IDs, not all-layer values/alias maps. No independent source reference supplied. Generic hooks restored on examined normal/error paths, not universal observer-failure/concurrency guarantees.',
            'no_adoption':'Conditional cache-advance and rounding hypotheses remain unproved by CPU toy. No production collector/prompt/sampler/quality/acceptance change, new optimization acceptance, or reopened reviewer/prompt study.'}}
    write_json(root/'terminal-audit.json',report)
    public={k:v for k,v in report.items() if k not in ['final_cpu_fixture']}
    public['final_cpu_fixture_files']=len(report['final_cpu_fixture'])
    public['terminal_audit_sha256']=sha(root/'terminal-audit.json')
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':public['terminal_audit_sha256'],'tools':report['tool_counts'],'frozen':len(expected),'decision':report['review']['decision']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
