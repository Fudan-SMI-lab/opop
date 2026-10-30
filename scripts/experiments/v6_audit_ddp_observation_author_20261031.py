"""Read-only terminal audit; raw role history remains server-private."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_communication_observation_author_20261031 import build_inputs, INSTRUCTIONS
from scripts.experiments.v6_diagnostic_author_20261029 import validate

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-observation-author-20261031')


def digest(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def main():
    assert not (ROOT/'terminal-audit.json').exists()
    p = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal == read(ROOT/'state.json')
    assert terminal['results'] == read(ROOT/'followup-results.json')
    assert terminal['phase'] == 'completed_with_errors'
    error = terminal['results']['diagnostic-author']['traceback']
    assert 'assert prompt==INSTRUCTIONS' in error and error.endswith('AssertionError\n')
    events = [json.loads(line) for line in (ROOT/'diagnostic-author/events.jsonl').read_text().splitlines()]
    finishes = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    assert len(finishes) == 1
    agent = finishes[0]
    assert agent['contract_repairs'] == 0 and len(agent['generation_snapshots']) == 1
    meta, prompt, files = load(Path(agent['generation_snapshots'][0]))
    sandbox = Path(meta['sandbox'])
    inputs, frozen, index, parent = build_inputs(ROOT.parent)
    expected = {'inputs/'+k: v.encode() if isinstance(v,str) else v for k,v in inputs.items()}
    assert set(files) == set(expected)|{'inputs/worker-environment.json'}
    assert all(files[k] == v for k,v in expected.items())
    assert all((sandbox/k).read_bytes() == v for k,v in files.items())
    prefix = ('Work on the supplied project artifacts. Inputs and framework code are fixed. '
              'Write only the requested output files in this directory; do not modify external source, '
              'provider configuration, quality thresholds or workload. The framework will execute and '
              'validate the output independently.\n')
    env = json.loads(files['inputs/worker-environment.json'])
    suffix = (' Read inputs/worker-environment.json before library/API inspection. '
              f"The measured worker uses Python {env['python']}; use that executable for CPU checks. "
              'Other system/Conda environments may have different library versions. GPU work remains in leased framework workers.')
    assert prompt == prefix+INSTRUCTIONS+suffix
    assert meta['framework_source'] == p['core'] == framework_source()
    old = read(ROOT.parent/'ddp-communication-redecision-20261031/followup-completed.json')['results']['selection']
    oldmeta, _, _ = load(Path(old['initial_snapshot']))
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert all(sha(path) == h for path,h in p['frozen_identities'].items())
    assert all(sha(path) == h for path,h in frozen.items())
    validate(sandbox)
    assert sha(sandbox/'diagnostic.py') == '024441f7d80bddaa51590b3630fbcc3969ac31b05548546ad2089106a10b5e98'
    assert sha(sandbox/'diagnostic-plan.json') == '862a728f25e8056ed7d61659440f22e323e9e299f57932e35b16f8dbd534bf68'
    private = ROOT/'terminal-private-history'
    history = read(private/'messages.json'); tools = read(private/'tools.json')
    assert len(tools) == 87
    assert sum(x['data'].get('role') == 'assistant' for x in history['message']) == agent['usage']['messages'] == 71
    content = {}; writes = {}; edits = {}; access = []; lines = truncated = 0
    for n,t in enumerate(tools,1):
        inp=t.get('input') or {}; out=t.get('output') or ''; md=t.get('metadata') or {}
        access.append({'n':n,'part':t['part'],'tool':t['tool'],'status':t['status'],
            'input':inp,'output_sha256':digest(out),'output_bytes':len(out.encode()),
            'exit':md.get('exit'),'output_truncated':md.get('truncated',False)})
        if t['tool'] == 'write':
            name=inp['filePath']; content[name]=inp['content']; writes[name]=writes.get(name,0)+1
            if Path(name).parent == sandbox:
                (private/('initial-'+Path(name).name)).write_text(inp['content'])
        if t['tool'] == 'edit':
            name=inp['filePath']; assert content[name].count(inp['oldString']) == 1,(n,name)
            content[name]=content[name].replace(inp['oldString'],inp['newString'],1)
            edits[name]=edits.get(name,0)+1
        if t['tool'] == 'read' and t['status'] == 'completed':
            name=inp['filePath']; text=content.get(name)
            if text is None: text=Path(name).read_text()
            actual=text.splitlines()
            for line in out.splitlines():
                match=re.match(r'^(\d+): (.*)$',line)
                if not match: continue
                shown=match[2]; original=actual[int(match[1])-1]
                if '... (line truncated to ' in shown:
                    assert original.startswith(shown.split('... (line truncated to ')[0]); truncated+=1
                else: assert shown == original,(n,match[1])
                lines+=1
    for name in ('diagnostic.py','diagnostic-plan.json'):
        path=str(sandbox/name)
        assert content[path].encode() == (sandbox/name).read_bytes()
    assert edits[str(sandbox/'diagnostic.py')] == 16
    # Reconstruct only reviewed, pure JSON-print commands. No recorded probes,
    # shell commands, provider calls or model/collective work are rerun.
    reconstructed=[]
    for n in (15,16,17,18,19):
        command=tools[n-1]['input']['command']
        code=command.split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
        assert 'import json' in code and not any(s in code for s in ('import torch','subprocess','write_text','os.system'))
        buf=io.StringIO()
        with redirect_stdout(buf): exec(compile(code,f'pure-json-tool-{n}','exec'),{})
        assert buf.getvalue() == tools[n-1]['output'],n
        reconstructed.append(n)
    retained=private/'retained-cpu-fixtures'; retained.mkdir(exist_ok=True)
    for source in [Path('/tmp/opencode/diag/harness.py'),*Path('/tmp/opencode/diag/work').glob('*.py')]:
        if source.is_file(): shutil.copyfile(source,retained/source.name)
    write_json(private/'complete-access-index.json',access)
    cpu=read(ROOT/'audit/independent-cpu-mechanics/result.json')
    assert read(ROOT/'audit/independent-cpu-mechanics/execution.json')['returncode'] == 0
    assert cpu['probe_sha256'] == sha(sandbox/'diagnostic.py')
    assert cpu['cuda_initialized'] is False and cpu['process_group_initialized'] is False
    assert cpu['reports']['partial_setup']['original_calls'] == 0
    assert cpu['reports']['enclosing_range']['ambiguous']['status'] == 'observed'
    assert not list((ROOT/'diagnostic-author/jobs').glob('*'))
    machine={'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}
    a={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],
       'controller_phase':terminal['phase'],'controller_error':'Bare-body prompt postcheck omitted the existing 575-byte framework wrapper; original controller/error retained.',
       'real_author_finished':True,'agent':agent,'snapshot':str(Path(agent['generation_snapshots'][0])),
       'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
       'explicit_bytes':sum(map(len,expected.values())),'exact_inputs_prompt_model_tools':True,
       'prompt_sha256':digest(prompt),'core':p['core'],'frozen_files':len(p['frozen_identities']),
       'all_frozen_unchanged':True,'source_current':p['source_current'],
       'probe_sha256':sha(sandbox/'diagnostic.py'),'probe_bytes':(sandbox/'diagnostic.py').stat().st_size,
       'plan_sha256':sha(sandbox/'diagnostic-plan.json'),'plan_bytes':(sandbox/'diagnostic-plan.json').stat().st_size,
       'probe_writes':1,'probe_edits':16,'plan_writes':1,'plan_edits':0,'outputs_reconstructed_exact':True,
       'tools':len(tools),'read_lines_verified':lines,'explicit_truncated_prefixes':truncated,
       'pure_json_outputs_reconstructed':reconstructed,'private_history_sha256':sha(private/'messages.json'),
       'private_access_sha256':sha(private/'complete-access-index.json'),
       'tool_review':[
        'Complete87 recorded tool inputs/outputs and embedded CPU sources reviewed; source read lines independently matched. Exact initial code plus16 edits and single plan write reconstructed. Supplied inputs/indexed original B/C rank0 excerpts, original parent metric trace, framework and installed PyTorch API/source, own rewritten CPU fixtures accessed. No observed private-config/later-audit/reviewer/other-candidate answer reads. Filesystem nonhermetic; not absolute absence proof.',
        'Author ran actual small CPU Linear FW/BW/SGD and world_size1 Gloo DDP/broadcast tests, beyond explicit no-model/no-real-collective author instruction. They are not pure mocks or original two-rank/full-model quality. CPU profiler issued gpuGetDeviceCount code100; no successful CUDA/model GPU execution observed.',
        'SyntaxError tool32, missing ijson hidden by compound exit0 tool41, repeated float/dict comparison42/43, fixture missing Path63, artifact_written mismatch63, param-id requires_grad error68/70, missing profile72, UnboundLocalError73, synthetic payload mismatch74 and cap-key KeyError86 retained. Agent modified fixtures and generated probe; final harness passed and repeated. Pipeline tail can hide failure. Agent removed own __pycache__ and overwrote ephemeral fixture outputs; final fixtures and complete history retained, not all overwritten raw outputs.',
        'Plan claims ambiguity negative executed, but recorded fixture checked unique synthetic ranges plus positive payload matching, not overlapping-range/missing-index negatives. Exception author check type/message, profiler restoration class-only; independent fixture checks actual identity.'
       ],
       'code_limits':[
        'Python zero_grad ledger is not evaluator invocation identity; captured Graph replays bypass it and same-ordinal hook records can span multiple calls. Hook callback time is not GPU readiness. Three no_sync microbatches collapse by ordinal/depth, not individual burst identity.',
        'Trace mapping uses list position and can label missing/unknown trace associations observed. Enclosing-range predicate retains outermost range; overlapping inner ranges can be marked observed. Same-trace launch correlation is useful but does not establish tensor consumer/wire/critical-path edges.',
        'Post-entry _get_zeros_like_grad_buckets materializes approximately497759232B zeros-like buffers; no-kernels-enqueued claim is unsupported, no GPU measurement performed here. Retained DDP/parameters/hooks alter lifetimes. Metadata is not universally hard-bounded: overflow broadcasts append None and invocations/errors remain uncapped; trace-file truncation lacks omitted counter. Raw hook times are summarized, not all persisted.',
        'Normal original exception identity and patch restoration pass independent Python-entry fixture. Dict result shallow-copied (nested object preserved). Injected partial observer API setup failure loses previous patch list and prevents original entry. These conditional CPU counterexamples are not historical GPU/NCCL faults.'
       ],
       'cpu_fixture_sha256':sha(ROOT/'audit/independent-cpu-mechanics/fixture.py'),
       'cpu_result_sha256':sha(ROOT/'audit/independent-cpu-mechanics/result.json'),
       'cpu_reports':cpu['reports'],'new_gpu_jobs':0,'new_quality_checks':0,'new_clean_samples':0,
       'decision':'Exact0244 probe is not eligible for fixed GPU observation. One new bounded diagnostic-author repair may consume exact CPU facts plus original inputs/probe/plan; no handwritten repair, optimization candidate, production adoption or repeated-until-pass loop.',
       'machine':machine}
    write_json(ROOT/'terminal-audit.json',a)
    public={k:v for k,v in a.items() if k not in ('agent','cpu_reports','tool_review','code_limits','machine','snapshot')}
    public.update(assistant_messages=71,contract_repairs=0,provider_estimate=agent['usage']['reported_cost'],
                  cost_scope='Observed provider estimate, not invoice',
                  terminal_audit_sha256=sha(ROOT/'terminal-audit.json'),
                  cpu_scope=cpu['scope'],
                  actual_author_scope_deviation='Small CPU model/Gloo work occurred despite no-model/no-real-collective instruction; no successful CUDA GPU execution observed.',
                  code_findings=a['code_limits'])
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),
        'read_lines':lines,'truncated':truncated,'frozen':len(p['frozen_identities']),'machine':machine}))


if __name__ == '__main__': main()
