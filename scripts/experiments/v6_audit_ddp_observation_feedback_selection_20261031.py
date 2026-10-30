"""Read-only selector audit and bounded existing-log scope check; no workload rerun."""
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
from types import SimpleNamespace

from kernel_optimizer.config import load_config
from kernel_optimizer.project.experiment_policy import check_selection, eligible_groups
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_observation_feedback_selection_20261031 import prepare

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-observation-feedback-selection-20261031')


def digest(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def main():
    p = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal['phase'] == 'completed' and read(ROOT/'state.json') == terminal
    assert read(ROOT/'followup-results.json') == terminal['results']
    assert framework_source() == p['core']
    assert not (ROOT/'terminal-audit.json').exists()
    inputs, feedback, oldmeta, oldprompt, oldfiles, frozen, _, index = prepare(
        SimpleNamespace(parent=ROOT.parent/'ddp-accepted-communication-trial-20261030'))
    result = terminal['results']['selection']; agent = result['agent']
    meta, prompt, files = load(Path(result['initial_snapshot'])); sandbox = Path(meta['sandbox'])
    expected = {'inputs/'+k: v.encode() if isinstance(v, str) else v for k, v in inputs.items()}
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    expected['inputs/worker-environment.json'] = oldfiles['inputs/worker-environment.json']
    assert files == expected and prompt == oldprompt and meta['fresh_session']
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert meta['framework_source'] == p['core']
    assert all((sandbox/n).read_bytes() == v for n, v in files.items())
    assert all(files[n] == v for n, v in oldfiles.items() if n != 'inputs/previous-feedback.txt')
    assert all(sha(path) == h for path, h in p['frozen_identities'].items())
    assert all(sha(path) == h for path, h in frozen.items())
    assert sha(ROOT/'original-feedback.json') == p['raw_feedback_sha256']
    assert digest(feedback) == p['feedback_sha256'] == sha(ROOT/'routed-feedback.json')
    source = ProjectSearch(load_config(ROOT/'private-config.yaml', []), Path(p['source']))
    current = source.revisions.current()
    assert current.id == p['parent'] and len(current.bundles) == 1
    assert all(files['inputs/source/'+f.path] == source.store.get_artifact(f.content_ref) for f in current.files)
    for k, name in [('request_sha256','request.json'), ('prepared_sha256','prepared.json'), ('discovery_sha256','discovery.json')]:
        assert read(ROOT/'preparation-reuse.json')[k] == sha(Path(p['source'])/name)
    selection = read(sandbox/'selection.json'); discovery = source.current_discovery()
    check_selection(sandbox, discovery, json.loads(inputs['search.json'])['allowed_scopes'], eligible_groups(discovery, leaf_only=False))
    assert sha(sandbox/'selection.json') == result['selection_sha256']
    events = [json.loads(line) for line in (ROOT/'selection/events.jsonl').read_text().splitlines()]
    assert sum(e['type'] == 'PROJECT_AGENT_FINISHED' for e in events) == 1
    assert not (ROOT/'controller.log').read_text().strip()
    private = ROOT/'terminal-private-history'; session = agent['session']
    tools = read(private/(session+'-tools.json')); history = read(private/(session+'.json'))
    assert len(tools) == 36 and agent['contract_repairs'] == 0
    assert sum(x['data'].get('role') == 'assistant' for x in history['message']) == agent['usage']['messages'] == 26
    access = []; count = trunc = 0; recon = []
    # These exact scripts were reviewed as JSON/text reads and arithmetic only.
    pure = [10,11,12,15,16,17,18,19,20,22,23,24,25,26,28,29,34]
    for i, t in enumerate(tools, 1):
        inp, out = t['input'], t['output']; extra = (t['metadata'] or {}).get('outputPath')
        access.append({'n':i, 'part':t['part'], 'tool':t['tool'], 'input':inp,
            'output_sha256':digest(out), 'output_bytes':len(out.encode()), 'status':t['status'],
            'exit':(t['metadata'] or {}).get('exit'), 'full_output_path':extra})
        assert extra is None
        if t['tool'] == 'read':
            path = Path(inp['filePath']); lines = path.read_text().splitlines()
            assert path.is_relative_to(sandbox/'inputs')
            for row in out.splitlines():
                m = re.match(r'^(\d+): (.*)$', row)
                if not m: continue
                value, original = m[2], lines[int(m[1])-1]
                if '... (line truncated to ' in value:
                    assert original.startswith(value.split('... (line truncated to ')[0]); trunc += 1
                else: assert value == original, (i, m[1])
                count += 1
        elif i in pure:
            command = inp['command']
            if "<<'EOF'\n" in command:
                script = command.split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
            else:
                argv = shlex.split(command); script = argv[argv.index('-c')+1]
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): assert all(x.name == 'json' for x in node.names)
                assert not isinstance(node, (ast.ImportFrom, ast.AsyncFunctionDef))
                if isinstance(node, ast.Attribute): assert not node.attr.startswith('__') or node.attr == '__name__'
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in ['exec','eval','compile','__import__']
            def safe_open(name, mode='r', *args, **kwargs):
                path = (sandbox/str(name)).resolve(); assert mode == 'r'
                assert path.is_relative_to(sandbox) and str(path.relative_to(sandbox)) in files
                return io.StringIO(files[str(path.relative_to(sandbox))].decode())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree,'reviewed-json-output-reconstruction','exec'), {'open':safe_open})
            assert buf.getvalue() == out, (i, 'output mismatch')
            recon.append(i)
    assert tools[7]['output'].encode() == files['inputs/previous-feedback.txt'][:4000]
    # Reconstruct the sole generated selection write, without executing its script.
    script = tools[34]['input']['command'].split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
    tree = ast.parse(script); values = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ['rationale','hypothesis']:
            values[n.targets[0].id] = ast.literal_eval(n.value)
    values['ids'] = [c['id'] for c in json.loads(files['inputs/calls.json'])]
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sel' for t in n.targets))
    rebuilt = {ast.literal_eval(k): values[v.id] if isinstance(v, ast.Name) else ast.literal_eval(v) for k, v in zip(node.keys,node.values)}
    assert json.dumps(rebuilt, indent=2).encode() == (sandbox/'selection.json').read_bytes()
    write_json(private/'complete-access-index.json', access)
    (private/'selection-write-script.py').write_text(script)

    # A bounded additional evidence question: what does the logged full-size AR establish?
    # Existing raw logs/code only; no collector, model, profile, or GPU rerun.
    tuning = json.loads(files['inputs/observations/current/nccl_tuning-f6198e01.json'])
    matched = []
    for entry in tuning['evidence']:
        path = Path(entry['file']); assert sha(path) == entry['sha256']
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if 'AllReduce: 497759232 Bytes' in line:
                matched.append({'file':str(path), 'sha256':sha(path), 'line':line_no, 'text':line})
    assert len(matched) == 1
    timeline = json.loads(files['inputs/observations/current/execution_timeline-1221e205.json'])
    trace_rows = []
    for rank in timeline['ranks']:
        cs = rank['communication']['collectives']; kernels = [k for c in cs for k in c['kernels']]
        assert len(cs) == len(kernels) == 13
        assert sum(c['logical_input_bytes'] for c in cs) == 497759232
        trace_rows.append({'rank':rank['rank'], 'kernels':len(kernels),
            'logical_bytes':sum(c['logical_input_bytes'] for c in cs),
            'sum_ms':sum(k['duration_us'] for k in kernels)/1000,
            'scope':'Existing timeline capture only; no clean/wire/critical-path estimate'})
    calibration = json.loads(files['inputs/observations/current/collective_calibration-1af5837f.json'])
    cal_rows = []
    for rank in calibration['ranks']:
        for row in rank['rows']:
            assert statistics.median(row['samples_ms']) == row['median_completion_ms']
            cal_rows.append({'rank':rank['rank'], 'bytes':row['logical_bytes'], 'median_ms':row['median_completion_ms'], 'interpretation':rank['interpretation']})
    assert {x['bytes'] for x in cal_rows} == {28351488,157541376}
    evaluation = Path(__import__('kernel_optimizer.project.evaluation',fromlist=['__file__']).__file__)
    code = evaluation.read_text(); tree = ast.parse(code)
    block = next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and any(isinstance(v,ast.Constant) and v.value=='nccl_tuning' for v in n.test.comparators))
    scope_check = {'question':'Does the full-size AllReduce tuning record establish a measured steady-step fused reduction or wire floor?',
        'matched_original_log':matched, 'collector_original_scope':tuning['scope'],
        'collector_original_interpretation':tuning['interpretation'],
        'evaluation_sha256':sha(evaluation), 'evaluation_action':ast.get_source_segment(code,block),
        'accepted_code_sha256':digest(files['inputs/source/opop_candidates/trial-d06aa22e2d.py']),
        'existing_timeline_periods':trace_rows, 'existing_calibration':cal_rows,
        'answer':'No. One real full-payload tuning decision occurs in a separate worker whose action resets and invokes warmup+1 times, including installation/setup. Log rows have no per-invocation duration/sequence join. Actual steady timeline has13 reductions; calibration has only28MB/157MB. This establishes an observed full-size tuning decision/capability, not clean fused-step quality, speed, wire saturation or a33ms lower bound. Exact placement in first/setup invocation remains code/order inference, not a timestamp join.',
        'activity':'Read-only CPU parsing of retained logs/code/JSON; no original fixture/model/collector/native rerun.'}
    write_json(ROOT/'communication-log-scope-check.json', scope_check)
    assert not list((ROOT/'selection/jobs').glob('*'))
    audit = {'at':datetime.now(timezone.utc).isoformat(), 'terminal':terminal['finished_at'], 'phase':terminal['phase'],
        'core':p['core'], 'frozen_files':len(p['frozen_identities']), 'all_frozen_unchanged':True,
        'generation':meta['id'], 'session':session, 'initial_files':len(files), 'initial_bytes':sum(map(len,files.values())),
        'actual_inputs_route_prompt_model_tools_verified':True, 'original_nonfeedback_inputs_equal':True,
        'feedback_sha256':digest(feedback), 'raw_feedback_sha256':p['raw_feedback_sha256'], 'source_current':current.id,
        'selection_sha256':sha(sandbox/'selection.json'), 'scope':selection['scope'], 'groups':selection['groups'],
        'affected_calls':len(selection['affected_calls']), 'output_writes':1, 'output_edits':0,
        'output_write_reconstructed_exact':True, 'contract_repairs':agent['contract_repairs'],
        'assistant_messages':agent['usage']['messages'], 'tools':len(tools), 'provider_estimate':agent['usage']['reported_cost'],
        'cost_scope':'Provider observation estimate, not invoice', 'private_history_sha256':sha(private/(session+'.json')),
        'private_access_sha256':sha(private/'complete-access-index.json'), 'read_lines_verified':count,
        'explicit_truncated_prefixes':trunc, 'pure_json_outputs_reconstructed_exact':recon,
        'new_gpu_jobs':0, 'communication_log_scope_check_sha256':sha(ROOT/'communication-log-scope-check.json'),
        'access_review':[
            'All36 recorded tool inputs/outputs/embedded scripts reviewed. Reads supplied request/project/contract/environment, full failed5e3a and acceptedccd90 code, original feedback and current timeline/calibration/tuning, partial prior77ac CPU result. No model/CUDA/collective probe observed; Python JSON/text arithmetic and schema only.',
            'No tool opens any of15 new observation-evidence files. Tool10 exposes nested supplemental pointer; tool21 lists paths only. No new probe/plan/worker/diagnostic/rawtrace contents consumed. Delivery is not consumption and no claim of response to new observation facts is made.',
            'Tool34 actually opens priorCPU observations and prints3500characters, ending before all outstanding counterexamples. No new77ac code/proposal/index read. No observed later accepted/auditor/reviewer answers or private-provider-config content read; external filesystem is nonhermetic.',
            'Tool4 wrong source directory stderr hidden by subsequent ls exit0; tool9 grep/head empty compound result. Retained, not treated as execution success. No schema repairs; selection written once and basic/jsonschema checks only.'
        ],
        'claims_audit':[
            'Selection autonomously proposes batch-four compute plus one full-gradient AllReduce and SGD graph, replacing13 steady reductions. This differs from closed77ac multicommunicator/Future repair and from the previous three-batch/unchanged13-reduction selection; it is a joint compute/communication-scheduling hypothesis, not observed benefit.',
            '33ms wire floor,14-15GB/s wire rate,4-5ms recoverable host gap and55-62ms clean prediction are unsupported. Calibration measures separate Work wait plus synchronization; captures cannot be subtracted from clean scores and tuning has no per-step timing join.',
            'Old5e3a regression cannot rule out concurrent graphs or establish saturation: setup failed params.index before those paths and full quality ran original eager fallback. Full failedcode was read but not reconciled with the output premise.',
            'Full-size tuning log is real but includes setup/warmup. It is not a validated steady fused step, exact replica proof, or fixed wire floor. Same-size matching does not bind a protocol decision to every timed kernel.',
            'Batched loss is mathematically comparable only under unchanged token/label weighting; floating reduction order, full gradients/updatedparameters/continuous state/exact replicas, ordering, reset and fallback neutrality still need actual generated-code and original full-quality checks.',
            'No new observation consumption, candidate execution, quality, search/independent acceptance or production adoption established.'
        ],
        'next_decision':'The concrete log-scope question has been executed and retained. Proceed with ONE isolated ordinary candidate generation for the selector-authored joint batching/full-gradient-reduction hypothesis, using its unedited choice and ordinary original evidence. Narrow support is actual13 steady reductions/unchanged full payload, source accumulation boundary and recorded full-size tuning capability, NOT the rejected causal/55-62ms/floor claims. No manual algorithm/code/collector or new selector/prompt; no automatic GPU/search. Review exact implementation before any execution. Closed77ac/Qwen/FSDP/observer chains remain closed.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
                   'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',audit)
    public = {k:v for k,v in audit.items() if k not in ['session','private_history_sha256','private_access_sha256','machine']}
    public['terminal_audit_sha256'] = sha(ROOT/'terminal-audit.json')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'), 'public':sha(ROOT/'public-terminal-audit.json'),
        'scope_check':sha(ROOT/'communication-log-scope-check.json'), 'read_lines':count, 'truncated':trunc,
        'reconstructed':recon, 'machine':audit['machine']}))


if __name__ == '__main__': main()
