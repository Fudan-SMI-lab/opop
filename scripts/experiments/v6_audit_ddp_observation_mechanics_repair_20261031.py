"""Terminal audit of one generated diagnostic repair; no role/model/GPU replay."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_observation_mechanics_repair_20261031 import prepare, expected_prompt, INSTRUCTIONS, REPAIR
from scripts.experiments.v6_diagnostic_author_20261029 import validate

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-observation-mechanics-repair-20261031')


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    assert not (ROOT/'terminal-audit.json').exists()
    protocol = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal == read(ROOT/'state.json') and terminal['results'] == read(ROOT/'followup-results.json')
    assert terminal['phase'] == 'completed'
    result = terminal['results']['diagnostic-author']; agent = result['agent']
    events = [json.loads(line) for line in (ROOT/'diagnostic-author/events.jsonl').read_text().splitlines()]
    finishes = [x['payload'] for x in events if x['type'] == 'PROJECT_AGENT_FINISHED']
    assert finishes == [agent] and agent['contract_repairs'] == 0
    meta, prompt, files = load(Path(result['initial_snapshot']))
    sandbox = Path(meta['sandbox'])
    inputs, frozen, index, oldmeta, oldfiles, oldaudit = prepare(ROOT.parent)
    expected = {'inputs/'+k:v.encode() if isinstance(v,str) else v for k,v in inputs.items()}
    assert set(files) == set(expected)|{'inputs/worker-environment.json'}
    assert all(files[k] == v for k,v in expected.items())
    assert all(files[k] == v for k,v in oldfiles.items())
    assert all((sandbox/k).read_bytes() == v for k,v in files.items())
    assert prompt == expected_prompt(INSTRUCTIONS+REPAIR, json.loads(files['inputs/worker-environment.json']))
    assert meta['framework_source'] == protocol['core'] == framework_source()
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert frozen == protocol['frozen_identities']
    assert all(sha(p) == h for p,h in frozen.items())
    validate(sandbox)
    assert sha(sandbox/'diagnostic.py') == '92279e28dc1e548efe86ef44467a3b0d09f7d94586da3dc16c0a3f55b773db96'
    assert sha(sandbox/'diagnostic-plan.json') == '0a82e46248377b94a7ec21954eb34d2bdf556d65796557556923ebf9eeefcf13'
    private = ROOT/'terminal-private-history'
    history = read(private/'messages.json'); tools = read(private/'tools.json')
    assert len(tools) == 41
    assert sum(x['data'].get('role') == 'assistant' for x in history['message']) == agent['usage']['messages'] == 29
    content = {}; writes = {}; edits = {}; access = []; lines = truncated = 0
    for n,t in enumerate(tools,1):
        inp = t.get('input') or {}; out = t.get('output') or ''; md = t.get('metadata') or {}
        access.append({'n':n,'part':t['part'],'tool':t['tool'],'status':t['status'],
            'input':inp,'output_sha256':digest(out),'output_bytes':len(out.encode()),
            'exit':md.get('exit'),'output_truncated':md.get('truncated',False)})
        if t['tool'] == 'write':
            name = inp['filePath']; content[name] = inp['content']; writes[name] = writes.get(name,0)+1
            if Path(name).parent == sandbox: (private/('initial-'+Path(name).name)).write_text(inp['content'])
        if t['tool'] == 'edit':
            name = inp['filePath']; assert content[name].count(inp['oldString']) == 1
            content[name] = content[name].replace(inp['oldString'],inp['newString'],1)
            edits[name] = edits.get(name,0)+1
        if t['tool'] == 'read' and t['status'] == 'completed':
            name = inp['filePath']; text = content.get(name)
            if text is None: text = Path(name).read_text()
            actual = text.splitlines()
            for line in out.splitlines():
                match = re.match(r'^(\d+): (.*)$',line)
                if not match: continue
                shown = match[2]; original = actual[int(match[1])-1]
                if '... (line truncated to ' in shown:
                    assert original.startswith(shown.split('... (line truncated to ')[0]); truncated += 1
                else: assert shown == original,(n,match[1])
                lines += 1
    for name in ('diagnostic.py','diagnostic-plan.json'):
        path = str(sandbox/name)
        assert content[path].encode() == (sandbox/name).read_bytes()
        assert writes[path] == 1 and edits.get(path,0) == 0
    retained = private/'retained-cpu-fixtures'; retained.mkdir(exist_ok=True)
    for path in Path('/tmp/opencode/diag-repair-check').rglob('*'):
        if path.is_file():
            dest = retained/path.relative_to('/tmp/opencode/diag-repair-check')
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
    assert content['/tmp/opencode/diag-repair-check/harness.py'].encode() == (retained/'harness.py').read_bytes()
    write_json(private/'complete-access-index.json',access)
    oldtree = ast.parse(Path(index['probe']['path']).read_text()); newtree = ast.parse((sandbox/'diagnostic.py').read_text())
    def definitions(tree):
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    olddefs = definitions(oldtree); newdefs = definitions(newtree)
    changed = sorted(k for k in olddefs.keys() & newdefs.keys() if olddefs[k] != newdefs[k])
    cpu = read(ROOT/'audit/independent-cpu-mechanics/result.json'); reports = cpu['reports']
    assert read(ROOT/'audit/independent-cpu-mechanics/execution.json')['returncode'] == 0
    assert cpu['probe_sha256'] == result['probe_sha256']
    assert cpu['cuda_initialized'] is False and cpu['process_group_initialized'] is False
    assert reports['enclosing_range']['ambiguous']['status'] == 'unknown'
    assert [x['invocation_ordinal'] for x in reports['missing_trace_index']['traces']] == [9,10,11]
    assert reports['missing_trace_index']['traces'][1]['mapping_status'] == 'unknown'
    assert reports['unknown_trace']['traces'][0]['mapping_status'] == 'unknown'
    assert reports['no_python_zero_grad']['hook_ordinals'] == [None,None,None]
    assert reports['broadcast_bound']['retained_list_length'] == 4096 and reports['broadcast_bound']['none_stubs'] == 0
    assert reports['return']['same_return_object'] and reports['return']['valid_preserved'] and reports['return']['original_calls'] == 1
    assert reports['exception']['same_exception_object'] and reports['exception']['original_calls'] == 1
    assert all(reports['return']['patches_restored_before_fixture_cleanup'].values())
    assert all(reports['exception']['patches_restored_before_fixture_cleanup'].values())
    assert reports['partial_setup']['original_calls'] == 0
    assert all(v for k,v in reports['partial_setup']['patches_restored_before_fixture_cleanup'].items() if k != 'profile')
    assert not list((ROOT/'diagnostic-author/jobs').glob('*'))
    machine = {'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}
    audit = {'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],
        'controller_phase':terminal['phase'],'real_author_finished':True,'snapshot':result['initial_snapshot'],
        'agent':agent,'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
        'explicit_bytes':sum(map(len,expected.values())),'original_initial_files_unchanged':len(oldfiles),
        'exact_inputs_prompt_model_tools':True,'prompt_sha256':digest(prompt),'core':protocol['core'],
        'frozen_files':len(frozen),'all_frozen_unchanged':True,'source_current':protocol['source_current'],
        'probe_sha256':result['probe_sha256'],'probe_bytes':(sandbox/'diagnostic.py').stat().st_size,
        'plan_sha256':result['plan_sha256'],'plan_bytes':(sandbox/'diagnostic-plan.json').stat().st_size,
        'probe_writes':1,'probe_edits':0,'plan_writes':1,'plan_edits':0,'outputs_reconstructed_exact':True,
        'changed_definitions':changed,'added_definitions':sorted(newdefs.keys()-olddefs.keys()),
        'tools':len(tools),'read_lines_verified':lines,'explicit_truncated_prefixes':truncated,
        'private_history_sha256':sha(private/'messages.json'),'private_access_sha256':sha(private/'complete-access-index.json'),
        'tool_review':[
            'All41 tool inputs/outputs, generated code/plan and authored inert harness reviewed; read lines matched original files; single initial writes reconstruct exact final code/plan. Original inputs/indexed B/C artifacts, framework, installed PyTorch source/API, accepted ccd90 code and own CPU fixtures accessed. No observed private-config/later-audit/reviewer/other-implementation answer reads; filesystem nonhermetic.',
            'Tool27 creates a two-element CPU Parameter for API inspection, no model/FW/BW/process group/real collective/profiler/CUDA initialization. Tool37 runs one synthetic harness; no role/model/GPU rerun by auditor. Tool11 wrong installed path exits2; tool23 wrong flat result schema prints None, corrected reader24. Tail pipeline exit0 is not independent pass proof.',
            'Author harness identity assertion all(identities()) iterates dictionary keys and is vacuous, API/schema orTrue assertions are vacuous. Recorded return/exception/partial-setup fields nevertheless inspected; independent unchanged-sequence fixture checks restoration identities. Author real-trace extraction is a positive read-only control, not a negative or new GPU execution.'
        ],
        'code_limits':[
            'Normal entry is called once with SAME successful result object and SAME uncaught exception in the independent fixture. Partial observer installation failure restores earlier patches, records setup_failed and stops before original entry; no universal always-invoke-on-observer-failure promise. The injected non-class profiler is not the installed current API.',
            'Zero_grad ledger remains a partial Python anchor, not actual evaluator invocation identity. Graph callbacks can remain unassigned; order-derived warmup/measure labels and repeat-index joins are hypotheses. CPU hooks are not GPU readiness. Equal-payload buckets lack individual collective identity; no wire time, GPU consumer, Work/Future completion or critical path proof.',
            'Post-entry zeros-like bucket inventory allocates about497759232B; no-kernels and1-2GB plan claims are unsupported/inexact. Retained DDP/parameter/hook owners affect lifetime. Hook/trace/broadcast caps do not bound invocations/errors universally; below2MB hard bound and generic every-exit restoration are not proved.',
            'Duplicate trace indices, nested range ordering and arbitrary metadata/teardown errors are not universally certified. Fixed current request has one metric directory/three unique indices and installed profiler class; any actual collection/quality error must be preserved, not repaired or retried.'
        ],
        'cpu_fixture_sha256':sha(ROOT/'audit/independent-cpu-mechanics/fixture.py'),
        'cpu_result_sha256':sha(ROOT/'audit/independent-cpu-mechanics/result.json'),'cpu_reports':reports,
        'cpu_adaptation':'Only EXPECTED probe SHA changed; exact original sequence executed once in new directory; old outputs untouched.',
        'new_gpu_jobs':0,'new_quality_checks':0,'new_clean_samples':0,
        'decision':'Eligible for ONE declared exact-byte fixed B-observed then C-observed on pinned original requests after disk budget. This tests scoped observability and original quality, not universal observer safety/performance acceptance. Setup failure aborts and is retained; baseline gate fails -> C not_run. No further author repair, observer-off rerun, optimization candidate or production adoption scheduled.',
        'machine':machine}
    write_json(ROOT/'terminal-audit.json',audit)
    public = {k:v for k,v in audit.items() if k not in ('agent','snapshot','cpu_reports','tool_review','machine')}
    public.update(assistant_messages=29,contract_repairs=0,provider_estimate=agent['usage']['reported_cost'],
        cost_scope='Observed provider estimate, not invoice',terminal_audit_sha256=sha(ROOT/'terminal-audit.json'),
        cpu_scope=cpu['scope'],actual_author_activity='API/metadata inspection, two-element CPU Parameter, inert synthetic fixtures; no observed model/process group/real collective/CUDA execution.')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),
        'read_lines':lines,'truncated':truncated,'frozen':len(frozen),'changed':changed,'added':audit['added_definitions'],'machine':machine}))


if __name__ == '__main__': main()
