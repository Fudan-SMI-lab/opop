"""Read-only terminal audit of one generated repair; private outputs stay remote.

The separate CPU fixture is executed once before this reader. This reader does not
run the model, generated build, collective, provider, or CPU fixture again.
"""
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

from kernel_optimizer.config import load_config
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-contract-repair-20261030')
FREEZE = ROOT/'framework/f3d245c26fdd'


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    # The input index names its frozen fixture, not a newly deployed reader path.
    import importlib.util
    spec = importlib.util.spec_from_file_location('frozen_repair_controller', FREEZE/'scripts/experiments/v6_ddp_communication_contract_repair_20261030.py')
    controller = importlib.util.module_from_spec(spec); spec.loader.exec_module(controller)
    p = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal['phase'] == 'completed' and read(ROOT/'state.json') == terminal
    assert read(ROOT/'followup-results.json') == terminal['results']
    assert not (ROOT/'terminal-audit.json').exists()
    inputs, feedback, oldmeta, oldprompt, expected, frozen, details = controller.prepare(
        SimpleNamespace(parent=ROOT.parent/'ddp-communication-candidate-20261030'))
    result = terminal['results']['candidate']; agent = result['agent']
    meta, prompt, files = load(Path(result['initial_snapshot'])); sandbox = Path(meta['sandbox'])
    assert files == expected and all((sandbox/n).read_bytes() == v for n,v in files.items())
    _, _, oldfiles = load(Path(p['previous_snapshot']))
    assert len(oldfiles) == 40 and all(files[n] == v for n,v in oldfiles.items())
    assert prompt.replace(controller.REPAIR, '', 1) == oldprompt
    assert digest(controller.REPAIR) == p['repair_paragraph_sha256']
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert meta['fresh_session'] and meta['framework_source'] == p['core'] == framework_source()
    assert all(sha(path) == h for path,h in p['frozen_identities'].items())
    assert all(sha(path) == h for path,h in frozen.items())
    assert digest(feedback) == p['feedback_sha256'] and json.loads(feedback)['supplemental_evidence'] is None
    source = ProjectSearch(load_config(ROOT/'private-config.yaml', []), Path(p['source']))
    current = source.revisions.current()
    assert current.id == p['parent'] and len(current.bundles) == 1
    assert all(files['inputs/source/'+f.path] == source.store.get_artifact(f.content_ref) for f in current.files)
    for name in ['request','prepared','discovery']:
        assert sha(Path(p['source'])/(name+'.json')) == p[name+'_sha256']
    validate_proposal(sandbox)
    assert sha(sandbox/'candidate.py') == result['candidate_sha256']
    assert sha(sandbox/'proposal.json') == result['proposal_sha256']
    proposal = read(sandbox/'proposal.json')
    assert proposal['parameter_space'] == {} and proposal['requires'] == []
    assert proposal['replaces'] == list(current.bundles)
    private = ROOT/'terminal-private-history'; session = agent['session']
    history = read(private/(session+'.json')); tools = read(private/(session+'-tools.json'))
    assert len(tools) == 67 and agent['contract_repairs'] == 0
    assert sum(x['data'].get('role') == 'assistant' for x in history['message']) == 59
    assert len(agent['generation_snapshots']) == 1
    events = [json.loads(l) for l in (ROOT/'candidate/events.jsonl').read_text().splitlines()]
    assert not any(e['type'] == 'PROJECT_OUTPUT_REJECTED' for e in events)

    # All raw commands/output remain in the private index. Never print tool3:
    # it read the role's own provider configuration and contains a credential.
    content = {}; access = []; read_lines = truncated_lines = 0; writes = {}; edits = {}
    for ordinal,t in enumerate(tools, 1):
        inp = t['input']; out = t['output'] or ''; metadata = t.get('metadata') or {}
        access.append({'ordinal':ordinal,'part':t['part'],'tool':t['tool'],'status':t['status'],
            'input':inp,'output_bytes':len(out.encode()),'output_sha256':digest(out),
            'exit':metadata.get('exit'),'full_output_path':metadata.get('outputPath'),
            'private_configuration_output':ordinal == 3})
        if t['tool'] == 'write':
            path = inp['filePath']; content[path] = inp['content']
            writes[path] = writes.get(path,0)+1
            if Path(path).parent == sandbox:
                (private/('initial-'+Path(path).name)).write_text(inp['content'])
        elif t['tool'] == 'edit':
            path = inp['filePath']; assert content[path].count(inp['oldString']) == 1
            content[path] = content[path].replace(inp['oldString'],inp['newString'],1)
            edits[path] = edits.get(path,0)+1
        if t['tool'] == 'read':
            path = Path(inp['filePath']); text = content.get(str(path))
            if text is None: text = path.read_text()
            lines = text.splitlines()
            for line in out.splitlines():
                m = re.match(r'^(\d+): (.*)$', line)
                if not m: continue
                shown, original = m[2], lines[int(m[1])-1]
                if '... (line truncated to ' in shown:
                    assert original.startswith(shown.split('... (line truncated to ')[0]); truncated_lines += 1
                else: assert shown == original, (ordinal,m[1])
                read_lines += 1
    for name in ['candidate.py','proposal.json']:
        path = str(sandbox/name)
        assert content[path].encode() == (sandbox/name).read_bytes()
        assert writes[path] == 1 and edits.get(path,0) == 0
    fixture = Path('/tmp/opencode/check_repaired_candidate.py')
    assert content[str(fixture)].encode() == fixture.read_bytes()
    retained = private/'retained-cpu-fixtures'; retained.mkdir(exist_ok=True)
    shutil.copyfile(fixture, retained/fixture.name)
    write_json(private/(session+'-complete-access-index.json'), access)

    # Reconstruct the complete pure feedback-print outputs without rerunning any
    # recorded shell command, model/provider call or probe.
    d = json.loads(feedback); reconstructed = {}
    reconstructed[6] = str(feedback.count('\n'))+' inputs/previous-feedback.txt\n'+feedback.encode()[-8000:].decode()
    buf = io.StringIO()
    def walk(o, path='', depth=0):
        if isinstance(o,dict):
            for k,v in o.items():
                if k in ('affected_calls','read_set','calls','edits') and isinstance(v,list) and len(v)>20:
                    print(path+'/'+k, f'<{len(v)} items>'); continue
                walk(v,path+'/'+k,depth+1)
        elif isinstance(o,list):
            if len(o)>20: print(path,f'<list {len(o)}>'); return
            for i,v in enumerate(o): walk(v,path+f'[{i}]',depth+1)
        else: print(path,'=',str(o)[:300])
    with redirect_stdout(buf): walk(d)
    reconstructed[8] = buf.getvalue(); buf = io.StringIO()
    with redirect_stdout(buf):
        print('TOP KEYS:',list(d.keys()))
        for k,v in d.items():
            if k == 'resources': print('resources keys:',list(v.keys())[:20]); continue
            print('---',k,'---'); print(json.dumps(v)[:4000])
    reconstructed[9] = buf.getvalue()
    for n,text in reconstructed.items():
        metadata = tools[n-1].get('metadata') or {}
        actual = (private/Path(metadata['outputPath']).name).read_text() if metadata.get('truncated') else tools[n-1]['output']
        assert actual == text, ('feedback_output',n)
    source_slices = {33:('evaluation.py',120,260),35:('discovery.py',55,200),43:('evaluation.py',260,395)}
    for n,(name,start,end) in source_slices.items():
        text = (FREEZE/'src/kernel_optimizer/project'/name).read_text()
        assert tools[n-1]['output'] == '\n'.join(text.splitlines()[start-1:end])+'\n'
    code = (sandbox/'candidate.py').read_text()
    old_code = files['inputs/source/opop_candidates/trial-d06aa22e2d.py'].decode()
    def functions(text):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(text)) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    oldfn, newfn = functions(old_code), functions(code)
    common = sorted(set(oldfn)&set(newfn)); unchanged = [n for n in common if oldfn[n] == newfn[n]]
    cpu_path = ROOT/'audit/cpu-contract/result.json'; cpu = read(cpu_path)
    assert cpu['candidate_sha256'] == result['candidate_sha256'] and cpu['candidate_bytes_unchanged']
    assert cpu['cuda_initialized'] is False and cpu['real_collectives'] == cpu['model_runs'] == 0
    assert read(ROOT/'audit/cpu-contract-execution.json')['exit_code'] == 0
    assert cpu['cases']['small_enqueued_future_error']['buffer_first'] == 8
    assert cpu['cases']['first_half_future_and_wait_error']['first_half_completion_error_suppressed']
    assert cpu['cases']['sentinel_mismatch']['actual_sentinel_results'] == [False]
    assert not list((ROOT/'candidate/jobs').glob('*'))
    machine = {'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True),
        'controller':subprocess.run(['ps','-p',str(read(ROOT/'launch.json')['pid']),'-o','pid,ppid,stat,etime'],capture_output=True,text=True).stdout}
    a = {'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],
        'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,
        'generation':meta['id'],'session':session,'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
        'original40inputs_byteequal':True,'actual_initial_bytes_verified':True,'actual_registry_route_equal':True,
        'feedback_sha256':digest(feedback),'supplemental_evidence':None,
        'production_prompt_file_model_tools_core_unchanged':True,'experiment_repair_paragraph_bytes':860,
        'assistance':'Explicit developer CPU-contract facts plus earlier source/metric mapping; not unaided ordinary cohort or same-prompt AB.',
        'source_current':current.id,'candidate_sha256':result['candidate_sha256'],'candidate_bytes':len(code.encode()),
        'proposal_sha256':result['proposal_sha256'],'proposal_bytes':(sandbox/'proposal.json').stat().st_size,
        'output_writes':{'candidate.py':1,'proposal.json':1},'output_edits':{'candidate.py':0,'proposal.json':0},
        'exact_output_and_cpu_fixture_writes_edits_reconstructed':True,'contract_repairs':0,
        'parameter_space':proposal['parameter_space'],'requires':proposal['requires'],'replaces':proposal['replaces'],
        'assistant_messages':59,'tools':67,'usage':agent['usage'],'private_history_sha256':sha(private/(session+'.json')),
        'complete_access_index_sha256':sha(private/(session+'-complete-access-index.json')),
        'read_output_lines_verified':read_lines,'explicit_truncated_lines':truncated_lines,
        'full_feedback_outputs_verified':{str(n):digest(v) for n,v in reconstructed.items()},
        'retained_author_cpu_fixture_sha256':sha(retained/fixture.name),'cpu_contract_result_sha256':sha(cpu_path),
        'accepted_graph_common_functions':common,'accepted_graph_ast_equal_functions':unchanged,
        'tool_review':[
            'All67 recorded tool inputs/outputs and embedded probe sources reviewed, provider-config tool3 limited to private metadata/redacted review. Full outputs6/8/9 reconstructed exactly from original routed feedback; reads checked against actual files. Source/adapter/binding/default-hook/Future library and supplied prior code/CPU fixture/installed accepted code were read. Other-case and audit filenames were listed, with no observed later audit/reviewer-answer content read. Filesystem nonhermetic.',
            'Agent actual CPU work used real torch Futures, exact nested hook with synthetic equal-peer SUM, fake Graph/model state, and real BindingSession unsupported CPU fallback. No real model training/process group/collective/GPU worker observed. Tool55 did attempt CUDA lazy initialization through unmocked synchronize and failed NoCUDA GPUs available under hidden CUDA; cannot claim no CUDA attempt.',
            'Tool54 failed because fake Graph step reached unmocked CUDA synchronize;55 isolated traceback.56 patched only test synchronize.57 tail pipeline exit0 concealed identity AssertionError,58 changed test expectation to returned loss dict,59 passed. Its sentinel test had fired0 and short-circuited before the real sentinel comparison; an assert comparing fresh equal tensors orTrue is vacuous. Final fixtures/initial write/two edits/full commands retained, not every overwritten temporary output.',
            'Tool25 wrong path exit2;40 hidden nonexistent-path grep/empty output;63 AST index flag was bucket.index, not Tensor list.index;66 empty git-log compound status retained. Tool67 deleted only own __pycache__. Candidate/proposal each written once, no generated code edits or framework schema repairs.',
            'Role read its own private provider configuration at tool3. Prior startup audit output incident remains separately recorded; terminal reader does not reprint it, rotate credentials or publish private content.'
        ],
        'claims_audit':[
            'New candidate actually includes GraphA first3 raw microbatches, eager fourth DDP backward and GraphB SGD. It no longer relies on replaced bundle remaining in context.references; references still bind original eager adapter. Absorption is code evidence, not GPU equivalence/native/quality proof.',
            'Original split enqueue, both-enqueued get_future, and failed asynchronous Future counterexamples now respectively return full correct CPU bucket, avoid reissue, and propagate error. One independent fixture reran those fixed sequences once, with unchanged generated code/constants and no GPU/model/real collective.',
            'Remaining conditional CPU counterexamples: small unsplit bucket can enqueue SUM then get_future raises, falling through works-empty reissue and doubling4 to8; one-half get_future plus wait failure is suppressed while complement succeeds; genuine sentinel comparisonFalse at fired13 only logs/rebaselines and leaves graph_hook/concurrent. These are synthetic interface counterexamples, not historical NCCL faults or full task numerical results.',
            'Author sentinel fixture did not reach the sentinel condition because fired0 short-circuited it. Independent fixture explicitly stubbed only captured Graph execution, counted13, wrapped real sentinel comparator and observedFalse. This scope does not imply GPU drift or full reference comparison.',
            'Graph capture override restoresFalse rather than generic prior value; limited object/ptr/grad guards not arbitrary values/aliases/concurrency proof. Setup collectives precede peer consensus; rank-local fallback/default-group changes remain asymmetric failure risks. Graph partial-update exception followed by eager parent is not universally state-neutral.',
            'Scratch timing is6 sequential13-call chains plus6 concurrent14-call chains plus4warmups=166AllReduce, then1three-int consensus; proposal about85calls/two-int flags incorrect. No timings persisted. Scratch214244352B excludes communicator/allocator ownership; universal peak/no-regression claims unsupported. Sentinels small previous-call slices, missing names skipped; no earlySGD implemented.',
            'Calibration floor,8-14ms saving/predicted step and cross-profile gap interpretations are hypotheses. NCCL sums not wire/criticalpath. Generation/schema/CPU pass does not establish original2continuous quality,5clean timing, communication acceptance or production suitability.'
        ],
        'decision':'Do not stage/run77ac onGPU: repaired original cases but concrete error-propagation/duplicate-SUM/sentinel contracts still fail. Close this bounded repair chain without a second repair, hand edit, parameter trial or reviewer. Next bounded evidence step is a read-only inventory of existing real communication requests, prerequisites and measured gaps, to choose a fresh task or a separately justified decision scope; do not silently claim research complete or auto-restart Qwen/FSDP/DDP candidates.',
        'new_gpu_jobs':0,'new_quality_checks':0,'new_clean_samples':0,'machine':machine}
    write_json(ROOT/'terminal-audit.json',a)
    public = {k:v for k,v in a.items() if k not in ['session','usage','private_history_sha256','complete_access_index_sha256','machine']}
    public.update(terminal_audit_sha256=sha(ROOT/'terminal-audit.json'),reported_provider_estimate=agent['usage']['reported_cost'],provider_cost_scope='Observed provider estimate, not invoice')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),
        'cpu_sha256':sha(cpu_path),'frozen_files':len(p['frozen_identities']),'read_lines':read_lines,
        'truncated_lines':truncated_lines,'equal_functions':unchanged,'machine':machine}))


if __name__ == '__main__': main()
