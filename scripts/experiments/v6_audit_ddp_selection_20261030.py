"""Read-only terminal audit of the isolated ordinary DDP selector; no model/GPU run."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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
from scripts.experiments.v6_ddp_communication_selection_20261030 import prepare

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-selection-20261030')


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    p = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal['phase'] == 'completed'
    assert read(ROOT/'state.json') == terminal
    assert read(ROOT/'followup-results.json') == terminal['results']
    assert framework_source() == p['core']
    assert not (ROOT/'terminal-audit.json').exists()
    inputs, feedback, oldmeta, oldprompt, oldfiles, frozen, oldpath = prepare(
        SimpleNamespace(parent=ROOT.parent/'ddp-accepted-communication-trial-20261030'))
    result = terminal['results']['selection']; agent = result['agent']
    meta, prompt, files = load(Path(result['initial_snapshot']))
    expected = {'inputs/'+k: v.encode() if isinstance(v, str) else v for k, v in inputs.items()}
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    expected['inputs/worker-environment.json'] = oldfiles['inputs/worker-environment.json']
    assert files == expected and prompt == oldprompt
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert meta['fresh_session'] and meta['framework_source'] == p['core']
    sandbox = Path(meta['sandbox'])
    assert all((sandbox/n).read_bytes() == v for n,v in files.items())
    assert sha(ROOT/'original-feedback.json') == p['raw_feedback_sha256']
    assert digest(feedback) == p['feedback_sha256'] and json.loads(feedback)['supplemental_evidence'] is None
    assert all(sha(path) == h for path,h in p['frozen_identities'].items())
    assert all(sha(path) == h for path,h in frozen.items())
    source = ProjectSearch(load_config(ROOT/'private-config.yaml', []), Path(p['source']))
    current = source.revisions.current()
    assert current.id == p['parent'] and len(current.bundles) == 1
    assert all(files['inputs/source/'+f.path] == source.store.get_artifact(f.content_ref) for f in current.files)
    assert read(ROOT/'preparation-reuse.json')['discovery_sha256'] == sha(Path(p['source'])/'discovery.json')
    selection_path = sandbox/'selection.json'; selection = read(selection_path)
    discovery = source.current_discovery()
    check_selection(sandbox, discovery, json.loads(inputs['search.json'])['allowed_scopes'], eligible_groups(discovery, leaf_only=False))
    assert sha(selection_path) == result['selection_sha256']

    private = ROOT/'terminal-private-history'; session = agent['session']
    history = read(private/(session+'.json')); tools = read(private/(session+'-tools.json'))
    assert len(tools) == 43 and agent['contract_repairs'] == 0
    read_lines = 0; truncated_lines = 0; access = []
    for i, tool in enumerate(tools):
        inp, out = tool['input'], tool['output']; extra = (tool['metadata'] or {}).get('outputPath')
        access.append({'ordinal':i+1, 'part':tool['part'], 'tool':tool['tool'], 'status':tool['status'],
                       'input':inp, 'output_bytes':len(out.encode()), 'output_sha256':digest(out),
                       'exit':(tool['metadata'] or {}).get('exit'), 'full_output_path':extra,
                       'full_output_sha256':sha(extra) if extra and Path(extra).is_file() else None})
        if tool['tool'] == 'read':
            path = Path(inp['filePath']); lines = path.read_text().splitlines()
            for line in out.splitlines():
                m = re.match(r'^(\d+): (.*)$', line)
                if not m: continue
                value, original = m[2], lines[int(m[1])-1]
                if '... (line truncated to ' in value:
                    assert original.startswith(value.split('... (line truncated to ')[0]); truncated_lines += 1
                else: assert original == value, (i+1, m[1])
                read_lines += 1
    write_json(private/(session+'-complete-access-index.json'), access)
    # Reconstruct the only output write from literal AST, without executing the Agent script.
    command = tools[41]['input']['command']; script = command.split('\n',1)[1].rsplit('\nEOF',1)[0]
    tree = ast.parse(script)
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='selection' for t in n.targets))
    calls = [c['id'] for c in json.loads(files['inputs/calls.json'])]
    reconstructed = {ast.literal_eval(k): calls if isinstance(v,ast.Name) and v.id=='calls' else ast.literal_eval(v) for k,v in zip(node.keys,node.values)}
    output_bytes = json.dumps(reconstructed, indent=2).encode()
    assert output_bytes == selection_path.read_bytes()
    (private/'selection-write-script.py').write_text(script)
    assert len(calls) == len(set(calls)) == len(selection['affected_calls']) == 1055

    # Recheck values actually available to the selector from retained input artifacts.
    current_inputs = {Path(n).name:json.loads(v) for n,v in files.items() if n.startswith('inputs/observations/current/')}
    calibration = next(v for n,v in current_inputs.items() if n.startswith('collective_calibration-'))
    calibration_rows = []
    for rank in calibration['ranks']:
        for row in rank['rows']:
            median = statistics.median(row['samples_ms'])
            assert median == row['median_completion_ms']
            assert abs(row['logical_bytes']/(median/1000)-row['logical_payload_bytes_per_second']) < 1e-4
            calibration_rows.append({'rank':rank['rank'],'logical_bytes':row['logical_bytes'],'samples_ms':row['samples_ms'],
                                     'median_ms':median, 'interpretation':rank['interpretation']})
    diagnostic_name = next(n for n in files if n.startswith('inputs/observations/previous-after/candidate_diagnostics-'))
    diagnostics = json.loads(files[diagnostic_name])
    assert 'RuntimeError' not in tools[25]['output'] and 'ambiguous' not in tools[25]['output']
    assert 'Boolean value of Tensor' in files[diagnostic_name].decode()
    assert 'params.index(s)' in files['inputs/previous-candidate/candidate.py'].decode()
    failures = [{'rank_index':i, 'events':r['events']} for i,r in enumerate(diagnostics['ranks'])]
    write_json(private/'original-diagnostic-events-rechecked.json', failures)
    measurement = json.loads(feedback)['project_measurement']
    assert measurement['candidate_values'] == [108.537192,111.151905]
    assert measurement['parent_values'] == [71.284842,72.901868]
    jobs = sorted(str(x) for x in (ROOT/'selection/jobs').glob('*'))
    assert not jobs
    machine = {'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
               'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True),
               'controller':subprocess.run(['ps','-p',str(read(ROOT/'launch.json')['pid']),'-o','pid,ppid,stat,etime'],text=True,capture_output=True).stdout}
    a = {'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],
         'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,
         'generation':meta['id'],'session':session,'snapshot_sha256':sha(result['initial_snapshot']),
         'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'actual_initial_bytes_verified':True,
         'actual_registry_route_equal':True,'feedback_sha256':digest(feedback),'raw_feedback_sha256':p['raw_feedback_sha256'],
         'prompt_model_tools_core_unchanged':True,'source_current':current.id,'selection_sha256':sha(selection_path),
         'selected_scope':selection['scope'],'selected_groups':selection['groups'],'affected_calls':len(calls),
         'output_write_reconstructed_exact':True,'output_write_count':1,'code_edits':0,'contract_repairs':agent['contract_repairs'],
         'assistant_messages':agent['usage']['messages'],'tools':len(tools),'usage':agent['usage'],
         'private_history_sha256':sha(private/(session+'.json')),'complete_access_index_sha256':sha(private/(session+'-complete-access-index.json')),
         'read_output_lines_verified':read_lines,'explicit_truncated_lines':truncated_lines,
         'tool_review':[
             'All43 recorded tool inputs/outputs and embedded scripts reviewed. Reads supplied task/schema/index/contract/source, partial original feedback/proposal/bundle, current timeline/calibration/tuning and installed torch hook source. No observed later accepted/auditor/reviewer answers; filesystem nonhermetic.',
             'No model/GPU/collective execution observed. Library import/inspect only; controller has zero jobs. Tool42 writes selection once; tool43 performs schema/ID/coverage checks, not execution.',
             'Tool26 opens packaged original diagnostics but its400-character rank prefixes end before exception content. No read of failed candidate.py or installed optimized implementation observed. Tool24 reads bundle metadata only. Opening a file is not full consumption.',
             'Tool3 wrong source directory exits2; tool15 missing source directory hidden by final pipeline exit0; tool12 grep may fail despite compound status. Errors retained.'
         ],
         'calibration_recomputed':calibration_rows,'original_failed_pairs_rechecked':measurement,
         'claims_audit':[
             'Autonomously selects whole step plus two existing AllReduce groups; proposes multiple communicators, bucket concurrency/split and early updates. This is a new testable communication-scheduling hypothesis, not measured improvement.',
             'Inferring GPU saturation from prior72-to108/111ms regression is unsupported: original setup failed at params.index(s) before private buffers/capture/benchmark, and quality ran original eager fallback. No concurrent microbatch execution was established.',
             'Synthetic calibration explicitly is not a lower bound/wire throughput. Calling it a hard floor or only remaining lever is unsupported.',
             'Same-timeline uncovered NCCL intervals are descriptive, not recoverable critical-path savings; combining with clean72ms to predict58-64ms is unvalidated. Tuning logs do not join all timed kernels to two-channel SIMPLE settings or prove extra communicator throughput.',
             'Default allreduce division-then-SUM source inspection does not establish exact numerical equivalence for new grouping/chunking/order. Early updates need real dependency/state-lifetime checks. Fallback neutrality, original-Graph preservation and sub-ms overhead are unproved.',
             'No selection/schema success constitutes candidate quality, native communication benefit, search/independent acceptance or production adoption.'
         ],
         'next_decision':'One bounded existing candidate generation from the unedited generated selection and ordinary original failure feedback/current source. No new selector, manual correction table, GPU/search/parameter/collector stage. Review exact output implementation before any later execution; preserve errors and do not retry until pass.',
         'machine':machine,'new_gpu_jobs':jobs}
    write_json(ROOT/'terminal-audit.json',a)
    public = {k:a[k] for k in ['at','terminal','phase','core','frozen_files','all_frozen_unchanged','initial_files','initial_bytes',
              'feedback_sha256','raw_feedback_sha256','selection_sha256','selected_scope','selected_groups','affected_calls',
              'output_write_reconstructed_exact','contract_repairs','assistant_messages','tools','read_output_lines_verified','explicit_truncated_lines',
              'tool_review','claims_audit','next_decision']}
    public.update({'terminal_audit_sha256':sha(ROOT/'terminal-audit.json'),'reported_provider_estimate':agent['usage']['reported_cost'],
                   'provider_cost_scope':'Observed estimate, not invoice','new_gpu_jobs':0})
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),
                      'frozen_files':a['frozen_files'],'read_lines':read_lines,'truncated_lines':truncated_lines,'selection_sha256':a['selection_sha256'],'machine':machine}))


if __name__ == '__main__':
    main()
