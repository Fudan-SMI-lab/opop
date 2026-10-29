"""Read-only role/output audit; no model, GPU, or Agent calls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from kernel_optimizer.config import load_config
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_qwen_cache_interface_candidate_20261030 import prepare

CODE = '7e78b82dd21391e3db3d6292e3798feff1908f53259aa732a3c011660c28fb92'
PROPOSAL = '048d30b638809aad76ef99e757be41ea20345d1def2f72da5c332cc56aaf94b6'


def digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def main(root):
    protocol = read(root/'protocol.json'); terminal = read(root/'followup-completed.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    frozen = protocol['frozen_identities']
    assert len(frozen) == 492 and all(sha(p) == h for p, h in frozen.items())
    parent = root.parent/'qwen-setup-cpu-feedback-20261030'
    inputs, feedback, saved, prompt, oldfiles, _, _ = prepare(SimpleNamespace(parent=parent))
    expected = {'inputs/'+k: v.encode() if isinstance(v, str) else v for k,v in inputs.items()}
    expected['inputs/previous-feedback.txt'] = feedback.encode()
    expected['inputs/worker-environment.json'] = oldfiles['inputs/worker-environment.json']
    row = terminal['results']['candidate']; agent = row['agent']
    meta, actual_prompt, files = load(Path(row['initial_snapshot'])); sandbox = Path(meta['sandbox'])
    assert files == expected and actual_prompt == prompt
    assert meta['fresh_session'] and meta['framework_source'] == saved['framework_source'] == protocol['core']
    assert meta['tool_model_settings'] == saved['tool_model_settings']
    assert all((sandbox/k).read_bytes() == v for k,v in files.items())
    assert len(files) == 33 and sum(map(len, files.values())) == 130630326
    assert json.loads(feedback)['supplemental_evidence'] is None
    assert json.loads(feedback)['previous_trial'] == 'trial-2a7d20b4d9'
    validate_proposal(sandbox)
    assert sha(sandbox/'candidate.py') == CODE and sha(sandbox/'proposal.json') == PROPOSAL
    proposal = read(sandbox/'proposal.json')
    assert proposal['parameter_space'] == {} and proposal['eliminated_calls'] == []
    assert proposal['requires'] == [] and proposal['replaces'] == []
    assert proposal['replacement_calls'] == ['323bcbe006b343648b346af9']
    events = [json.loads(x) for x in (root/'candidate/events.jsonl').read_text().splitlines()]
    finished = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    assert len(finished) == 1 and finished[0]['session'] == agent['session']
    assert agent['contract_repairs'] == 0
    private = root/'terminal-private-history'
    tools = read(private/(agent['session']+'-tools.json'))
    history = read(private/(agent['session']+'.json'))
    assert len(tools) == 57
    inventory = []
    for n,t in enumerate(tools, 1):
        inventory.append({'number':n,'part':t['part'],'tool':t['tool'],'input':t['input'],
            'status':t['status'],'exit':t.get('metadata',{}).get('exit'),
            'output_sha256':digest(t.get('output','')),'output_bytes':len(t.get('output','').encode())})
    write_json(private/'complete-tool-inventory.json', inventory)
    initial = tools[32]['input']['content']; rebuilt = initial
    (private/'candidate-generated-before.py').write_text(initial)
    # Tool34 used fuzzy matching: its returned diff records the actual typo removed.
    removed = "3e293c00b1fb98fbe29f, call 323bcbe006b343648b343648b346af9's exact id\n323bcbe006b343648b346af9).  The prefill boundary stays unbound so prefill\nmath, cache production and ttft_ms are the raw parent's."
    assert rebuilt.count(removed) == 1 and ('-'+removed.splitlines()[0]) in tools[33]['metadata']['diff']
    rebuilt = rebuilt.replace(removed, tools[33]['input']['newString'].split('\n',1)[1],1)
    for n in [35,36,37]:
        edit = tools[n-1]['input']; assert rebuilt.count(edit['oldString']) == 1
        rebuilt = rebuilt.replace(edit['oldString'],edit['newString'],1)
    assert rebuilt.encode() == (sandbox/'candidate.py').read_bytes()
    assert tools[53]['input']['content'].encode() == (sandbox/'proposal.json').read_bytes()
    fixture = tools[38]['input']['content']
    for n in [41,44,46,47,49]:
        edit = tools[n-1]['input']; assert fixture.count(edit['oldString']) == 1
        fixture = fixture.replace(edit['oldString'], edit['newString'],1)
    fixture_path = Path('/tmp/opencode/test_candidate_cpu.py')
    assert fixture.encode() == fixture_path.read_bytes()
    (private/'final-cpu-fixture.py').write_bytes(fixture_path.read_bytes())
    extra = []
    for n,t in enumerate(tools,1):
        path = t.get('metadata',{}).get('outputPath')
        if path:
            p = Path(path); target = private/p.name
            if not target.exists(): target.write_bytes(p.read_bytes())
            assert sha(target) == sha(p)
            extra.append({'tool':n,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    search = ProjectSearch(load_config(parent/'private-config.yaml',[]),parent/'qwen-continuation')
    current = search.revisions.current()
    assert current.id == json.loads(files['inputs/project.json'])['current_revision']
    assert current.id == '2a42b9b182b77d54a4028e779d8ba8f8bee5dabec4a83917015ed52e486e0dd1' and not current.bundles
    assert not list((root/'candidate').glob('jobs/*'))
    assert all(sha(p) == h for p,h in frozen.items())
    findings = {
        'delivery':'33 actual initial files exactly reproduce current registry/plan routing and historical6ec4 artifacts; supplemental=null. Same selected scope/prompt/model/tools/core; changed prior feedback/history is not same-snapshot A/B. Prior CPU assistance and manual source mapping remain.',
        'reads':'All57 recorded tool inputs/outputs and retained scripts reviewed. Tools12/13/15 read original parent AttributeError and actual target signatures;21-27 read installed Cache/DynamicLayer/Qwen/SDPA sources. No observed access to later audit/accepted/reviewer answers; external filesystem is not hermetic. Large tool6 output preserved; file size is not proof of complete context use.',
        'cpu':'No recorded GPU/model/collective execution. Actual tensors are CPU;3-layer BF16 toy and fake Graph. Final fixture manually emulates fast tail, while real replacement calls fail CUDA-device guard; printed stale-lineage/_absorb labels do not establish those paths. No actual BindingSession dispatch or full CUDA quality. Successful fixture rerun at55 retained.',
        'errors':'Tool14 wrong dev-v6-work path hides stderr/pipelineexit0.30 DynamicCache(layers=)TypeError;40 missing step_caches;42 wrong untouched-tail assertion;45 CPU guard versus expected fastpath;48 wrong route-label assertion.50 misspelled dev-venchain hidden before later test exit0. Agent changes test expectations/fixture, not model tolerance.57 deletes own __pycache__; overwritten ephemeral test outputs are not all retained, complete tool history and final fixture are.',
        'code':'Genuine DynamicCache/DynamicLayer objects replace opaque handle;31 prebuilt reused caches view shared masters. Ownership and refresh rely on object and layer0 key pointer; not complete all-layer/values/alias/lifetime certification. Teacher stride32,1 still mismatches framework1,1. Direct A/O/P selfchecks bypass framework dispatch; finalKV check is initial oracle chain only. Parent-mutation step in P is a code control, not actual framework transition proof.',
        'remaining_defects':'fast_enabled=True precedes verify; caught verify failure does not clear it, so a mapping may be returned despite disable_reason. Failure before enable returns{} and can cause installed-sites mismatch. Runtime Graph exception uses eager interior but later emits fast_graph_replay label. These are code-review risks, not observed current-GPU outcomes or a unique historical cause.',
        'unadopted':'Previously GPU-proven interior, selfchecks guarantee quality/state isolation, prefill unchanged guarantees TTFT,74% profile-gap reclaimable critical path,sub-ms copy and potential memory gain are unsupported. No native evidence for newcode. No parameter/search/acceptance/production change.',
        'next':'One declared exact-byte B then C fixed diagnostic is justified to expose actual dispatch, setup, full two-check quality and original constraints. Outer evaluator remains authority despite selfcheck weakness. No hand repair, observer, new role, native collector or formal acceptance; preserve failure and all samples; failed baseline skips candidate.'}
    report = {'at':datetime.now(timezone.utc).isoformat(),'terminal_at':terminal['finished_at'],
        'reason':'batch_finished','controller_status':terminal['phase'],'core':framework_source(),
        'frozen_files':492,'frozen_identities':frozen,'all_frozen_unchanged':True,'source_current':current.id,
        'inputs':len(files),'input_bytes':sum(map(len,files.values())),'feedback_bytes':len(feedback.encode()),
        'feedback_sha256':digest(feedback),'snapshot_sha256':sha(row['initial_snapshot']),
        'prompt_model_tools_core_unchanged':True,'candidate_sha256':CODE,'candidate_bytes':(sandbox/'candidate.py').stat().st_size,
        'proposal_sha256':PROPOSAL,'proposal_bytes':(sandbox/'proposal.json').stat().st_size,'parameters':{},
        'initial_code_sha256':digest(initial),'four_code_edits_reconstructed_exactly':True,
        'final_cpu_fixture_sha256':sha(private/'final-cpu-fixture.py'),'cpu_fixture_reconstructed_exactly':True,
        'session':agent['session'],'assistant_messages':sum(x['data'].get('role')=='assistant' for x in history['message']),
        'tool_counts':dict(Counter(t['tool'] for t in tools)),'tools':57,'schema_repairs':0,'usage':agent['usage'],
        'private_history_sha256':sha(private/(agent['session']+'.json')),'extra_private_outputs':extra,
        'tool_exit_failures':[x['number'] for x in inventory if x['exit'] not in [0,None]],
        'gpu_stage':'not_run','quality':'not_run','clean_samples':0,'search_acceptance':'not_run',
        'independent_acceptance':'not_run','code_review':{'fixed_execution_justified':True,'parameters':{}},
        'findings':findings,'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'live_gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)}
    write_json(root/'terminal-audit.json', report)
    public = {k:v for k,v in report.items() if k not in ['frozen_identities','session','usage','extra_private_outputs']}
    public['provider_cost_estimate'] = 2.39222807294
    public['cost_scope'] = 'Provider estimate, observe-only, not invoice.'
    write_json(root/'public-terminal-audit.json',public)
    print(json.dumps({'status':'passed','audit_sha256':sha(root/'terminal-audit.json'),
        'public_sha256':sha(root/'public-terminal-audit.json'),'code_bytes':report['candidate_bytes'],
        'proposal_bytes':report['proposal_bytes'],'tool_counts':report['tool_counts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
