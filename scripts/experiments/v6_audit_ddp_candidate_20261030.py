"""Terminal identity/tool/code audit; no model, GPU, collective or candidate rerun."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
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
from scripts.experiments.v6_ddp_communication_candidate_20261030 import prepare

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-candidate-20261030')


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def main():
    p=read(ROOT/'protocol.json'); terminal=read(ROOT/'followup-completed.json')
    assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal
    assert read(ROOT/'followup-results.json')==terminal['results']
    assert not (ROOT/'terminal-audit.json').exists()
    assert framework_source()==p['core']
    inputs,feedback,oldmeta,oldprompt,oldfiles,selector_files,frozen,details=prepare(
        SimpleNamespace(parent=ROOT.parent/'ddp-communication-selection-20261030'))
    result=terminal['results']['candidate']; agent=result['agent']
    meta,prompt,files=load(Path(result['initial_snapshot'])); sandbox=Path(meta['sandbox'])
    expected={'inputs/'+n:v.encode() if isinstance(v,str) else v for n,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode()
    expected['inputs/worker-environment.json']=selector_files['inputs/worker-environment.json']
    assert files==expected and prompt==oldprompt
    assert meta['model']==oldmeta['model'] and meta['tool_model_settings']==oldmeta['tool_model_settings']
    assert meta['fresh_session'] and meta['framework_source']==p['core']
    assert all((sandbox/n).read_bytes()==v for n,v in files.items())
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    assert all(sha(path)==h for path,h in frozen.items())
    assert digest(feedback)==p['feedback_sha256'] and json.loads(feedback)['supplemental_evidence'] is None
    assert sha(ROOT/'original-feedback.json')==p['original_feedback_sha256']
    source=ProjectSearch(load_config(ROOT/'private-config.yaml',[]),Path(p['source']))
    current=source.revisions.current()
    assert current.id==p['parent'] and len(current.bundles)==1
    assert all(files['inputs/source/'+f.path]==source.store.get_artifact(f.content_ref) for f in current.files)
    assert sha(Path(p['source'])/'discovery.json')==p['discovery_sha256']
    validate_proposal(sandbox)
    assert sha(sandbox/'candidate.py')==result['candidate_sha256']
    assert sha(sandbox/'proposal.json')==result['proposal_sha256']
    proposal=read(sandbox/'proposal.json')
    assert proposal['parameter_space']=={} and proposal['requires']==[]
    assert proposal['replaces']==list(current.bundles)

    private=ROOT/'terminal-private-history';session=agent['session']
    history=read(private/(session+'.json'));tools=read(private/(session+'-tools.json'))
    assert len(tools)==101 and agent['contract_repairs']==1
    assert sum(x['data'].get('role')=='assistant' for x in history['message'])==90
    # Exact ordered writes/edits, including the normal contract-repair proposal.
    content={}; states={}; access=[]; read_lines=0; truncated_lines=0
    for ordinal,t in enumerate(tools,1):
        inp=t['input'];out=t['output'] or '';metadata=t.get('metadata') or {}
        access.append({'ordinal':ordinal,'part':t['part'],'tool':t['tool'],'status':t['status'],
            'input':inp,'output_bytes':len(out.encode()),'output_sha256':digest(out),
            'exit':metadata.get('exit'),'full_output_path':metadata.get('outputPath')})
        if t['tool']=='write' and Path(inp['filePath']).parent==sandbox:
            name=Path(inp['filePath']).name;content[name]=inp['content']
            (private/('initial-'+name)).write_text(inp['content'])
        elif t['tool']=='edit' and Path(inp['filePath']).parent==sandbox:
            name=Path(inp['filePath']).name
            assert content[name].count(inp['oldString'])==1,(ordinal,name)
            content[name]=content[name].replace(inp['oldString'],inp['newString'],1)
        if ordinal==98: states['pre_repair']=content.copy()
        if t['tool']=='read':
            path=Path(inp['filePath'])
            text=content[path.name] if path.parent==sandbox and path.name in content else path.read_text()
            lines=text.splitlines()
            for line in out.splitlines():
                m=re.match(r'^(\d+): (.*)$',line)
                if not m: continue
                shown,original=m[2],lines[int(m[1])-1]
                if '... (line truncated to ' in shown:
                    assert original.startswith(shown.split('... (line truncated to ')[0]);truncated_lines+=1
                else: assert shown==original,(ordinal,m[1])
                read_lines+=1
    assert content['candidate.py'].encode()==(sandbox/'candidate.py').read_bytes()
    assert content['proposal.json'].encode()==(sandbox/'proposal.json').read_bytes()
    write_json(private/(session+'-complete-access-index.json'),access)
    # Tool21 was exclusively a slice of the original feedback's parameter search,
    # dominated by affected-call IDs; verify its complete output without rerunning tools.
    assert tools[20]['output']==json.dumps(json.loads(feedback)['parameter_search'],indent=1)[3500:20000]+'\n'
    repair_paths=[Path(x) for x in agent['generation_snapshots'] if x!=result['initial_snapshot']]
    assert len(repair_paths)==1
    repair,rprompt,rfiles=load(repair_paths[0])
    assert not repair['fresh_session'] and repair['session']==meta['session']
    assert {k:v for k,v in rfiles.items() if k.startswith('inputs/')}==files
    assert rfiles['candidate.py']==content['candidate.py'].encode()
    assert rfiles['proposal.json']==states['pre_repair']['proposal.json'].encode()
    assert repair['history_ref'] and repair['model']==meta['model'] and repair['tool_model_settings']==meta['tool_model_settings']
    (private/'contract-repair-prompt.txt').write_text(rprompt)
    events=[json.loads(l) for l in (ROOT/'candidate/events.jsonl').read_text().splitlines()]
    rejects=[e for e in events if e['type']=='PROJECT_OUTPUT_REJECTED']
    assert len(rejects)==1
    assert rejects[0]['payload']['artifacts']['candidate.py']=='sha256:'+result['candidate_sha256']
    assert rejects[0]['payload']['artifacts']['proposal.json']=='sha256:'+digest(states['pre_repair']['proposal.json'])
    # Retain final surviving CPU fixtures, not a claim all overwritten outputs survive.
    retained=private/'retained-cpu-fixtures';retained.mkdir(exist_ok=True)
    fixtures={}
    for name in ['mocktest.py','mocktest2.py','gloo_e2e.py']:
        path=Path('/tmp/opencode/hooktest')/name
        assert path.is_file();shutil.copyfile(path,retained/name);fixtures[name]=sha(path)
    cpu=read(ROOT/'audit/cpu-contract/result.json')
    assert cpu['candidate_sha256']==result['candidate_sha256'] and cpu['candidate_bytes_unchanged']
    assert cpu['cuda_initialized'] is False and cpu['model_runs']==cpu['real_collectives']==0
    assert read(ROOT/'audit/cpu-contract-execution.json')['exit_code']==0
    jobs=list((ROOT/'candidate/jobs').glob('*'));assert not jobs
    machine={'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),
        'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True),
        'controller':subprocess.run(['ps','-p',str(read(ROOT/'launch.json')['pid']),'-o','pid,ppid,stat,etime'],capture_output=True,text=True).stdout}
    a={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],
       'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,
       'generation':meta['id'],'repair_generation':repair['id'],'session':session,
       'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'actual_initial_bytes_verified':True,
       'feedback_sha256':digest(feedback),'original_feedback_sha256':p['original_feedback_sha256'],
       'selection_sha256':p['selection_sha256'],'canonical_selection_sha256':p['canonical_selection_sha256'],
       'actual_registry_route_equal':True,'prompt_model_tools_core_unchanged':True,'source_current':current.id,
       'candidate_sha256':result['candidate_sha256'],'candidate_bytes':len(content['candidate.py'].encode()),
       'proposal_sha256':result['proposal_sha256'],'proposal_bytes':len(content['proposal.json'].encode()),
       'output_writes':{'candidate.py':1,'proposal.json':1},'output_edits':{'candidate.py':3,'proposal.json':2},
       'exact_initial_write_edits_reconstructed':True,'repair_changed_code':False,
       'contract_repairs':1,'repair_error':rejects[0]['payload']['error'],
       'pre_repair_proposal_sha256':digest(states['pre_repair']['proposal.json']),
       'parameter_space':proposal['parameter_space'],'requires':proposal['requires'],'replaces':proposal['replaces'],
       'assistant_messages':90,'tools':101,'usage':agent['usage'],
       'private_history_sha256':sha(private/(session+'.json')),
       'complete_access_index_sha256':sha(private/(session+'-complete-access-index.json')),
       'read_output_lines_verified':read_lines,'explicit_truncated_lines':truncated_lines,
       'retained_cpu_fixtures':fixtures,'cpu_contract_result_sha256':sha(ROOT/'audit/cpu-contract/result.json'),
       'tool_review':[
         'All101 recorded tool inputs/outputs/embedded scripts reviewed. Read supplied selection/feedback/failed5e3a, installedccd90, original worker results/logs/sidecars, current and prior timelines/tuning/calibration/ownership, real framework and installed torch sources. Tool34 lists audit filenames but no observed audit-answer read. Framework find exposed other-case filenames, not contents. No observed later accepted/reviewer/auditor answers; filesystem remains nonhermetic.',
         'Unlike the selector, candidate actually read params.index(s) exception, both rank internal fallback routes, installed optimized source and binding.py references semantics. Exposure did not ensure correct final composition reasoning.',
         'Actual CPU imports/mocks and two-process Gloo small-MLP probes ran. Gloo defines copied hook logic rather than executing exact candidate build/binding; default/split/parity/manual sequential state have bitwise matching grads in that toy. No optimizer updates/full original continuous quality/NCCL/CUDA Graph validation. No actual GPU/model worker observed; controller jobs empty.',
         'Tool84 equal fake timings failed no-gain assertion;85 changed test margin/clock then Tensor-bool assertion failed;86 test-only correction passed.87 claimed degrade case never injects failure or flips schedule.91 dist.DDP AttributeError hidden by tail pipeline exit0;92 corrected DDP import,93/94 varied toy split threshold/bucket cap;97 repeats CPU tests with output suppression.83 SyntaxError hidden by echo exit0.49 NoneType.get,55 wrong path FileNotFoundError,68 builtin source TypeError/caught, grep misses/compound exit0 preserved.',
         'Agent deleted its own __pycache__; final CPU fixtures and complete commands/edits/session parts retained, not every overwritten ephemeral output. Exactly one normal framework relation repair: requires-old/replaces-empty became replaces-old/requires-empty; only proposal changed.'
       ],
       'claims_audit':[
         'Final replaces removes sole old Graph bundle. transactions.stage filters replaced IDs; BindingSession project references bind original adapter.run. New code contains no Graph implementation. Claim that delegation preserves installed GraphA/GraphB or an exact accepted-parent no-gain fallback is contradicted. CPU real-binding counterexample confirms original parent reference; it is not a GPU timing result.',
         'Exact hook CPU fault injection: second split enqueue failure returns half bucket; after both enqueues get_future failure reduces whole already-reduced bucket again; asynchronous failed future is swallowed by collect_all then callback ignoring child results; sentinel mismatch only logs anomaly and keeps concurrent hook mode. These are conditional contract counterexamples, not observed NCCL faults or full numerical rejection.',
         'Two new NCCL groups, parity routing and last-bucket split are generated communication-scheduling mechanisms, not current native evidence. Uninitialized scratch timing uses6 sequential13-call chains and6 concurrent14-call chains plus4 warmups=166 collectives, then1 consensus call; no timings persisted. Proposal about85 calls and universal performance/bitwise/fallback guarantees unproved.',
         'Scratch214244352B excludes groups/library allocator ownership. Hook/group closures persist even no-gain. Setup collectives precede consensus; local registration/fallback can diverge ranks. No universal asymmetric-failure/concurrent-owner safety demonstrated.',
         'Sentinels compare small slices to previous call rather than frozen full reference, skip missing names and do not prove all bucket halves. Identity tests partial object/ptr checks. No early SGD implemented: original parent performs its usual update. Defaults/CPU Gloo do not certify exact GPU reduction order.',
         'Prior setup failure correctly identified, but cross-profile gap attribution, calibration hard floor/tuning-to-kernel join,58-64ms prediction and no-regression claims remain unsupported. Native sums are not wire/critical path. Generation/schema success is not full quality/search/independent acceptance.'
       ],
       'decision':'Do not stage or execute this candidate on GPU. Preserve exact generated outputs and original failures. One CPU interface/failure counterexample acquisition completed once; no model/collective/source run. Next bounded step may address these concrete interface failures through the existing candidate role with explicitly declared CPU-fact assistance, not a hand-written implementation or optimization retry until pass; review before any later execution.',
       'new_gpu_jobs':0,'new_quality_checks':0,'new_clean_samples':0,'machine':machine}
    write_json(ROOT/'terminal-audit.json',a)
    public={k:a[k] for k in ['at','terminal','phase','core','frozen_files','all_frozen_unchanged','initial_files','initial_bytes',
        'feedback_sha256','original_feedback_sha256','selection_sha256','canonical_selection_sha256',
        'candidate_sha256','candidate_bytes','proposal_sha256','proposal_bytes','output_writes','output_edits',
        'exact_initial_write_edits_reconstructed','repair_changed_code','contract_repairs','repair_error','pre_repair_proposal_sha256',
        'parameter_space','requires','replaces','assistant_messages','tools','read_output_lines_verified','explicit_truncated_lines',
        'cpu_contract_result_sha256','tool_review','claims_audit','decision','new_gpu_jobs','new_quality_checks','new_clean_samples']}
    public.update(terminal_audit_sha256=sha(ROOT/'terminal-audit.json'),reported_provider_estimate=agent['usage']['reported_cost'],
                  provider_cost_scope='Observed provider estimate, not invoice')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit_sha256':sha(ROOT/'terminal-audit.json'),'public_sha256':sha(ROOT/'public-terminal-audit.json'),
        'frozen_files':a['frozen_files'],'read_lines':read_lines,'truncated_lines':truncated_lines,'machine':machine}))


if __name__=='__main__': main()
