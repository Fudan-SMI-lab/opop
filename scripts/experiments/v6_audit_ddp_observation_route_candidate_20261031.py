"""Terminal read-only generation audit. No model, candidate or GPU rerun."""
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from types import SimpleNamespace

from kernel_optimizer.config import load_config
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_observation_route_candidate_20261031 import prepare

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-observation-route-candidate-20261031')
CODE = '508ad3823d3a7b980df7076023b3567242f12a2544cb9bbeae0ada66c6f52f0a'
PROPOSAL = '853ae1eb048e1d590e5b2ba03d75dfec378a4fc20bf4a7b0faa478ba03e1de89'


def digest(x):
    return hashlib.sha256(x.encode() if isinstance(x, str) else x).hexdigest()


def main():
    p = read(ROOT/'protocol.json'); terminal = read(ROOT/'followup-completed.json')
    assert terminal['phase'] == 'completed' and read(ROOT/'state.json') == terminal
    assert read(ROOT/'followup-results.json') == terminal['results']
    assert not (ROOT/'terminal-audit.json').exists()
    assert framework_source() == p['core']
    inputs, feedback, oldmeta, oldprompt, oldfiles, selector_files, frozen, details = prepare(
        SimpleNamespace(parent=ROOT.parent/'ddp-observation-feedback-selection-20261031'))
    result = terminal['results']['candidate']; agent = result['agent']
    meta, prompt, files = load(Path(result['initial_snapshot'])); sandbox = Path(meta['sandbox'])
    expected = {'inputs/'+k:v.encode() if isinstance(v,str) else v for k,v in inputs.items()}
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    expected['inputs/worker-environment.json'] = selector_files['inputs/worker-environment.json']
    assert files == expected and prompt == oldprompt and meta['fresh_session']
    assert meta['model'] == oldmeta['model'] and meta['tool_model_settings'] == oldmeta['tool_model_settings']
    assert meta['framework_source'] == p['core']
    assert all((sandbox/n).read_bytes() == v for n,v in files.items())
    assert all(sha(path) == h for path,h in p['frozen_identities'].items())
    assert all(sha(path) == h for path,h in frozen.items())
    assert digest(feedback) == p['feedback_sha256']
    assert json.loads(feedback)['supplemental_evidence'] == json.loads(selector_files['inputs/previous-feedback.txt'])['supplemental_evidence']
    assert sha(ROOT/'original-feedback.json') == p['original_feedback_sha256']
    source = ProjectSearch(load_config(ROOT/'private-config.yaml',[]),Path(p['source']))
    current = source.revisions.current()
    assert current.id == p['parent'] and len(current.bundles) == 1
    assert all(files['inputs/source/'+f.path] == source.store.get_artifact(f.content_ref) for f in current.files)
    for key, name in [('request_sha256','request.json'),('prepared_sha256','prepared.json'),('discovery_sha256','discovery.json')]:
        assert p[key] == sha(Path(p['source'])/name)
    validate_proposal(sandbox)
    assert sha(sandbox/'candidate.py') == result['candidate_sha256'] == CODE
    assert sha(sandbox/'proposal.json') == result['proposal_sha256'] == PROPOSAL
    proposal = read(sandbox/'proposal.json')
    assert proposal['parameter_space'] == {} and proposal['requires'] == []
    assert proposal['replaces'] == list(current.bundles)
    events = [json.loads(line) for line in (ROOT/'candidate/events.jsonl').read_text().splitlines()]
    assert sum(e['type'] == 'PROJECT_AGENT_FINISHED' for e in events) == 1
    assert not any(e['type'] == 'PROJECT_OUTPUT_REJECTED' for e in events)
    assert not (ROOT/'controller.log').read_text().strip()
    private = ROOT/'terminal-private-history'; session = agent['session']
    tools = read(private/(session+'-tools.json')); history = read(private/(session+'.json'))
    assert len(tools) == 59 and agent['contract_repairs'] == 0
    assert sum(x['data'].get('role') == 'assistant' for x in history['message']) == agent['usage']['messages'] == 55
    access = []; contents = {}; read_lines = truncated = 0; reconstructed = []
    # Exact scripts reviewed as JSON/text reads only. No original model/CPU probe is rerun.
    pure = [8,9,10,11,17,18,19,20,21,40,41]
    for n,t in enumerate(tools,1):
        inp,out = t['input'],t['output']; metadata = t['metadata'] or {}
        assert metadata.get('outputPath') is None
        access.append({'n':n,'part':t['part'],'tool':t['tool'],'input':inp,'status':t['status'],
            'exit':metadata.get('exit'),'output_bytes':len(out.encode()),'output_sha256':digest(out)})
        if t['tool'] == 'write' and Path(inp['filePath']).parent == sandbox:
            name = Path(inp['filePath']).name; contents[name] = inp['content']
            (private/('initial-'+name)).write_text(contents[name])
        elif t['tool'] == 'edit' and Path(inp['filePath']).parent == sandbox:
            name = Path(inp['filePath']).name
            assert contents[name].count(inp['oldString']) == 1,(n,name)
            contents[name] = contents[name].replace(inp['oldString'],inp['newString'],1)
            (private/('after-tool-'+str(n)+'-'+name)).write_text(contents[name])
        if t['tool'] == 'read':
            path = Path(inp['filePath'])
            text = contents[path.name] if path.parent == sandbox and path.name in contents else path.read_text()
            lines = text.splitlines()
            for row in out.splitlines():
                m = re.match(r'^(\d+): (.*)$',row)
                if not m: continue
                shown, original = m[2],lines[int(m[1])-1]
                if '... (line truncated to ' in shown:
                    assert original.startswith(shown.split('... (line truncated to ')[0]); truncated += 1
                else: assert shown == original,(n,m[1])
                read_lines += 1
        elif n in pure:
            command = inp['command']; argv = shlex.split(command)
            script = argv[argv.index('-c')+1]
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import): assert all(x.name in ['json','re'] for x in node.names)
                assert not isinstance(node,(ast.ImportFrom,ast.AsyncFunctionDef))
                if isinstance(node,ast.Attribute): assert not node.attr.startswith('__') or node.attr == '__name__'
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
                    assert node.func.id not in ['exec','eval','compile','__import__']
            def safe_open(name,mode='r',*args,**kwargs):
                path = (sandbox/str(name)).resolve(); assert mode == 'r'
                relative = str(path.relative_to(sandbox)); assert relative in files
                return io.StringIO(files[relative].decode())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree,'reviewed-json-output-reconstruction','exec'),{'open':safe_open})
            shown = out[out.index('=== EVALUATION ==='): ] if n == 11 else out
            assert buf.getvalue() == shown,(n,'pure output differs')
            reconstructed.append(n)
    for name in ['candidate.py','proposal.json']:
        assert contents[name].encode() == (sandbox/name).read_bytes()
    # The actual inspected framework output is unchanged exact code, not a fresh tool probe.
    from kernel_optimizer.project import evaluation, binding
    for n,module,start,end in [(31,evaluation,0,8000),(32,evaluation,8000,20000),(33,evaluation,20000,30000),(34,binding,0,9000)]:
        assert tools[n-1]['output'] == Path(module.__file__).read_text()[start:end]+'\n'
        reconstructed.append(n)
    fixtures = {}; retained = private/'retained-cpu-fixtures'; retained.mkdir(exist_ok=True)
    for name in ['run_sim.py','cand_sim.py','gate_fail_test.py']:
        path = Path('/tmp/opencode')/name; assert path.is_file()
        shutil.copyfile(path,retained/name); fixtures[name] = sha(path)
    # Final toy copy differs by precisely the CPU guard removal. Its FakeGraph.replay is a no-op.
    assert (retained/'cand_sim.py').read_text() == contents['candidate.py'].replace(
        'p.dtype is not torch.float32 or p.device.type != "cuda"','p.dtype is not torch.float32')
    write_json(private/'complete-access-index.json',access)
    assert not list((ROOT/'candidate/jobs').glob('*'))
    audit = {'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],
        'core':p['core'],'frozen_files':len(p['frozen_identities']),'frozen_identities':p['frozen_identities'],
        'all_frozen_unchanged':True,'generation':meta['id'],'session':session,
        'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
        'actual_inputs_route_prompt_model_tools_verified':True,'source_current':current.id,
        'feedback_sha256':digest(feedback),'original_feedback_sha256':p['original_feedback_sha256'],
        'selection_sha256':p['selection_sha256'],'canonical_selection_sha256':p['canonical_selection_sha256'],
        'candidate_sha256':CODE,'candidate_bytes':len(contents['candidate.py'].encode()),
        'proposal_sha256':PROPOSAL,'proposal_bytes':len(contents['proposal.json'].encode()),
        'output_writes':{'candidate.py':1,'proposal.json':1},'output_edits':{'candidate.py':3,'proposal.json':0},
        'exact_initial_writes_edits_reconstructed':True,'contract_repairs':0,
        'parameters':{},'requires':proposal['requires'],'replaces':proposal['replaces'],
        'assistant_messages':55,'tools':59,'provider_estimate':agent['usage']['reported_cost'],
        'cost_scope':'Observed provider estimate, not invoice','private_history_sha256':sha(private/(session+'.json')),
        'private_access_sha256':sha(private/'complete-access-index.json'),'retained_cpu_fixtures':fixtures,
        'read_lines_verified':read_lines,'explicit_truncated_prefixes':truncated,
        'pure_outputs_reconstructed_exact':reconstructed,'new_gpu_jobs':0,'full_quality_checks':0,'clean_samples':0,
        'access_review':[
            'All59 recorded commands/inputs/outputs/embedded CPU scripts reviewed. Reads supplied original request/selection/adapter/source/old5e3a, original feedback diagnostics, old worker results/logs, installed acceptedccd90, actual framework evaluation/binding and installed torch APIs. Directory listing exposes other-case names only. No observed later auditor/reviewer/private-config content reads; filesystem remains nonhermetic.',
            'No tool opens any15 new observation-evidence files or current collector JSON files. Tool18 prints full generic new index pointer/scope/limits, not the indexed probe/plan/observed ranks/diagnostic contents. Existing CPU evidence files also not opened. Delivery is not actual observation consumption.',
            'Tool18 actually prints original params.index tensor-bool exception and both rank propagation records;17/20/21/26/27 read internal fallback and full-check facts. Final proposal correctly recognizes original eager fallback rather than executed concurrent graphs. This autonomous correction was not supplied by developer audit.',
            'CPU Linear grad-view/tied-weight toy and two-layer GPT2 sequential/batched FW/BW executed. Tool39 reports grad max3.725290298461914e-8, max-normalized relative3.5598466752162494e-7, equal scalar losses, not full original GPU/replica/continuous SGD evidence. Exact candidate unsupported CPU path delegates parent three times.',
            'Later toy modifies candidate CUDA-device guard, replaces streams/Graph/all_reduce with Python stubs, and uses small GPT2. FakeGraph.replay does nothing; capture context executes update eagerly once. Printed changed-parameters and graph labels therefore do not prove real graph replay or distributed quality. No successful CUDA or real processgroup/collective observed.',
            'Errors preserved: Parameter called as function36; bias=False yields None.numel37; monkeypatch-before-transformers import TypeError47/54; initial setup compares zero flat before batched computation48/49, then Agent code edit50 moves gate; wrong expected fallback label51 corrected test52; lambda-yield SyntaxError53. Several traceback pipelines exit0. Forced-negative MATH_RTOL toy55 tests setup restoration only, not all failures after update. Agent removed own __pycache__; retained final scripts and full history do not recover all overwritten ephemeral outputs.'
        ],
        'code_review':{'fixed_execution_justified':True,'scope':'One exact fixed current-accepted-parent B then generated C, original full evaluator/constraints/metrics, empty parameters; no acceptance/publishing/retry.',
            'facts_and_limits':[
                'Whole-step REPLACEMENT removes old bundle; references delegate original eager adapter, not the accepted graph. New code implements its own batch-four raw FW/BW Graph, one eager full-gradient SUM then division, and SGD Graph. This is a real generated joint compute/communication scheduling hypothesis, not measured communication benefit.',
                'First evaluator invocation delegates parent; second performs local full sequential-vs-batched gradient gate, bounded warmups/captures/sentinels/peer MIN before fast execution. At most one of the two outer continuous checks exercises new fast path. Original evaluator full gradients/updated params/loss/finite/exact replicas remain authority, not self-checks.',
                'Agent initial code gate compared uncomputed flat zeros; final edit50 moves it after actual batched backward. Full gate2e-7/2e-5 is stricter locally than outer1e-6/1e-4 but not a proof of global reduced error, future state, finite safety or cross-rank equality. NaN margins are not explicitly rejected by this setup max comparison.',
                'Batched tokens are copied once; guards check input-list/model identities and parameter pointers, not all token contents/shapes/dtypes/aliases/lr. Fixed original task reuses unchanged token values and plain SGD; no generic mutable-input/owner or concurrency guarantee.',
                'Fallback restores gradient references but not updated weights. Exception after update/synchronize or post-step grad-identity check can fall through to another parent update. Rank-asymmetric errors after agreement can diverge paths. These are code-level conditional limitations, not observed GPU faults; ordinary full quality/failure records must be preserved, no injected faults or claim universal safe fallback.',
                'Stale-stream override finally setsFalse rather than restoring arbitrary prior flag. Shared graph-pool ordering and allreduce stream ordering require real scoped execution. Scratch/graph/setup peaks and expected1.9-2.1GB are unmeasured; no new memory acceptance constraint.',
                '33ms wire floor,4-5ms recoverable gap,55-62ms gain and quality preserved by construction are unsupported. Full payload remains497759232B if reached; lower call count alone is not less communication volume or independent whole-step acceptance. No new native evidence yet.'
            ]},
        'audit_reader_errors':['First read-only reconstruction expected tool11 whole output to be JSON print; command also prepended ls -la metadata. Original attempt1 retained, reader corrected to compare exact JSON suffix plus separately reviewed listing. No Agent/CPU fixture/model/GPU rerun.'],
        'next_decision':'Declare ONE exact-byte fixed B/C validation against current acceptedccd90 using existing fork/stage/execute. Original full two continuous quality/two warmup/five clean and three separate diagnostic profiles per rank remain. Baseline validity gates C, no timing gate/reanchor. No Agent/search/tuning/collector choice/new Nsight/source-control rerun/reference copy/pointer publication/formal acceptance. Preserve any setup fallback, quality failure, tail or regression without retry. Exact generated code is not hand-edited.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
            'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',audit)
    public = {k:v for k,v in audit.items() if k not in ['frozen_identities','session','private_history_sha256','private_access_sha256','retained_cpu_fixtures','machine']}
    public['terminal_audit_sha256'] = sha(ROOT/'terminal-audit.json')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),
        'reads':read_lines,'truncated':truncated,'reconstructed':reconstructed,'machine':audit['machine']}))


if __name__ == '__main__': main()
