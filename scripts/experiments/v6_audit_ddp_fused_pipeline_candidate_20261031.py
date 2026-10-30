"""Complete recorded generation audit; does not rerun Agent/model/CPU probes."""
import ast,contextlib,hashlib,io,json,re,shlex,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from kernel_optimizer.config import load_config
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read,sha
from scripts.experiments.v6_ddp_fused_pipeline_candidate_20261031 import prepare
ROOT=Path('/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-candidate-20261031')
CODE='eef828816dd138cb69b0d12c960012b1970a0724b426a258ec6f8fcb6900abd7'
PROPOSAL='0f0a977afdfcb94e9a46f9c950bc4efc587ac72e16cec9fbfed750255234da94'
def digest(x):return hashlib.sha256(x.encode()if isinstance(x,str)else x).hexdigest()

def main():
    p=read(ROOT/'protocol.json');terminal=read(ROOT/'followup-completed.json');result=terminal['results']['candidate'];a=result['agent']
    assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal and read(ROOT/'followup-results.json')==terminal['results']
    assert not(ROOT/'terminal-audit.json').exists() and framework_source()==p['core']
    inputs,feedback,oldmeta,oldprompt,oldfiles,selectorfiles,frozen,details=prepare(SimpleNamespace(parent=ROOT.parent/'ddp-fused-communication-selection-20261031'))
    meta,prompt,files=load(Path(result['initial_snapshot']));box=Path(meta['sandbox'])
    expected={'inputs/'+n:v.encode()if isinstance(v,str)else v for n,v in inputs.items()}
    expected.update({'inputs/previous-feedback.txt':feedback.encode(),'inputs/worker-environment.json':selectorfiles['inputs/worker-environment.json']})
    assert files==expected and prompt==oldprompt and meta['fresh_session']
    assert meta['model']==oldmeta['model'] and meta['tool_model_settings']==oldmeta['tool_model_settings'] and meta['framework_source']==p['core']
    assert all((box/n).read_bytes()==v for n,v in files.items())
    assert all(sha(path)==h for path,h in p['frozen_identities'].items()) and all(sha(path)==h for path,h in frozen.items())
    assert digest(feedback)==p['feedback_sha256']==sha(ROOT/'routed-feedback.json')
    assert json.loads(feedback)['supplemental_evidence'] is None and sha(ROOT/'original-feedback.json')==p['original_feedback_sha256']
    source=ProjectSearch(load_config(ROOT/'private-config.yaml',[]),Path(p['source']));current=source.revisions.current()
    assert current.id==p['parent'] and len(current.bundles)==1
    assert all(files['inputs/source/'+f.path]==source.store.get_artifact(f.content_ref)for f in current.files)
    for key,name in [('request_sha256','request.json'),('prepared_sha256','prepared.json'),('discovery_sha256','discovery.json')]:assert p[key]==sha(source.output/name)
    validate_proposal(box);assert sha(box/'candidate.py')==CODE==result['candidate_sha256'];assert sha(box/'proposal.json')==PROPOSAL==result['proposal_sha256']
    proposal=read(box/'proposal.json');assert proposal['parameter_space']=={} and proposal['requires']==[] and proposal['replaces']==list(current.bundles)
    events=[json.loads(x)for x in(ROOT/'candidate/events.jsonl').read_text().splitlines()]
    assert [e['type']for e in events]==['RUN_CREATED','PROJECT_AGENT_STARTED','PROJECT_GENERATION_SNAPSHOT','PROJECT_AGENT_FINISHED'] and events[-1]['payload']==a
    assert not(ROOT/'controller.log').read_text() and read(ROOT/'session-health.json')['status']=='passed'
    private=ROOT/'terminal-private-history';tools=read(private/(a['session']+'-tools.json'));history=read(private/(a['session']+'.json'))
    assert len(tools)==43 and a['contract_repairs']==0 and sum(x['data'].get('role')=='assistant'for x in history['message'])==a['usage']['messages']==38
    contents={};access=[];lines=trunc=0;reconstructed=[];failed_json=[]
    pure=[4,10,15,17,18,19,20,21,22,23,24,27,29,30]
    for n,t in enumerate(tools,1):
        inp,out=t['input'],t['output'];md=t['metadata']or{};assert md.get('outputPath')is None and t['status']=='completed'
        access.append({'n':n,'part':t['part'],'tool':t['tool'],'input':inp,'output_bytes':len(out.encode()),'output_sha256':digest(out),'status':t['status'],'exit':md.get('exit'),'truncated':md.get('truncated')})
        if t['tool']=='write' and Path(inp['filePath']).parent==box:
            name=Path(inp['filePath']).name;contents[name]=inp['content'];(private/('initial-'+name)).write_text(contents[name])
        elif t['tool']=='edit' and Path(inp['filePath']).parent==box:
            name=Path(inp['filePath']).name;assert contents[name].count(inp['oldString'])==1
            contents[name]=contents[name].replace(inp['oldString'],inp['newString'],1);(private/('after-tool-'+str(n)+'-'+name)).write_text(contents[name])
        if t['tool']=='read':
            path=Path(inp['filePath']);assert path.is_relative_to(box)
            raw=contents[path.name] if path.parent==box else files[path.relative_to(box).as_posix()].decode()
            actual=raw.splitlines()
            for number,shown in re.findall(r'(?m)^(\d+): (.*)$',out):
                original=actual[int(number)-1]
                if '... (line truncated to 'in shown:assert original.startswith(shown.split('... (line truncated to ')[0]);trunc+=1
                else:assert original==shown,(n,number)
                lines+=1
        elif n in pure:
            cmd=inp['command'];script=shlex.split(cmd)[2]if n==4 else cmd.split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
            tree=ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):assert all(x.name in ['json','collections']for x in node.names)
                assert not isinstance(node,ast.ImportFrom)
                if isinstance(node,ast.Attribute):assert not node.attr.startswith('__') or node.attr=='__name__'
                if isinstance(node,ast.Call)and isinstance(node.func,ast.Name):assert node.func.id not in ['exec','eval','compile','__import__']
            def safe_open(name,*args,**kwargs):
                assert name in files and not args and not kwargs;return io.StringIO(files[name].decode())
            buf=io.StringIO();failure=None
            try:
                with contextlib.redirect_stdout(buf):exec(compile(tree,'<stdin>','exec'),{'open':safe_open})
            except Exception as exc:failure=exc
            if n in [20,21,23]:
                assert type(failure)is TypeError and out.startswith(buf.getvalue()) and out.rstrip().endswith('TypeError: '+str(failure));failed_json.append(n)
            else:assert failure is None and buf.getvalue()==out,(n,'output differs');reconstructed.append(n)
    for name in ['candidate.py','proposal.json']:assert contents[name].encode()==(box/name).read_bytes()
    assert tools[15]['output']==files['inputs/worker-environment.json'].decode()
    assert tools[12]['output'].endswith(files['inputs/source/opop_evaluate.py'].decode())
    assert tools[11]['output'].endswith('\n'.join(files['inputs/source/README.md'].decode().splitlines()[:50])+'\n')
    retained=private/'retained-cpu-fixtures';retained.mkdir(exist_ok=True);fixtures={}
    for n,name in [(31,'cpu_check.py'),(32,'chunk_check.py'),(40,'full_sim.py')]:
        command=tools[n-1]['input']['command'];initial=command.split("<<'EOF'\n",1)[1].split('\nEOF',1)[0]+'\n'
        (retained/('initial-'+name)).write_text(initial)
        if n==40:
            edit=tools[41]['input'];assert initial.count(edit['oldString'])==1;final=initial.replace(edit['oldString'],edit['newString'],1)
        else:final=initial
        path=Path('/tmp/opencode/candtest')/name;assert path.read_text()==final;shutil.copyfile(path,retained/name);fixtures[name]=sha(path)
    assert Path('/tmp/opencode/candtest/candidate.py').read_bytes()==(box/'candidate.py').read_bytes()
    write_json(private/'complete-access-index.json',access)
    cpu_root=ROOT/'audit/cpu-stream-consumer';cpu=read(cpu_root/'result/result.json');execution=read(cpu_root/'execution.json')
    assert execution['returncode']==0 and cpu['candidate_sha256']==CODE and cpu['fixture_sha256']==sha(cpu_root/'fixture.py')
    assert sha(cpu_root/'result/result.json')=='0c40b6ef9f05bdf6f19525c134142cfb124875bf0f120309e25aa6fb3fa0d477'
    assert not cpu['torch_cuda_initialized'] and cpu['ordered_completion_positive_control']['max_weight_error']==0
    assert cpu['deferred_collective_completion']['max_weight_error']>0 and cpu['saved_parameter_consumer']['early_update_then_next_backward_exception']
    assert not list((ROOT/'candidate/jobs').glob('*')) and all(sha(path)==h for path,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],'reason':'batch_finished','core':p['core'],
        'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'source_current':current.id,'generation':meta['id'],'session':a['session'],
        'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'actual_inputs_route_prompt_model_tools_verified':True,
        'candidate_sha256':CODE,'candidate_bytes':len(contents['candidate.py'].encode()),'proposal_sha256':PROPOSAL,'proposal_bytes':len(contents['proposal.json'].encode()),
        'output_writes':{'candidate.py':1,'proposal.json':1},'output_edits':{'candidate.py':1,'proposal.json':0},'exact_writes_and_edits_reconstructed':True,
        'parameters':{},'requires':[],'replaces':proposal['replaces'],'contract_repairs':0,'assistant_messages':38,'tools':43,'provider_estimate':a['usage']['reported_cost'],'cost_scope':'Provider estimate, not invoice',
        'read_lines_verified':lines,'explicit_truncated_prefixes':trunc,'pure_outputs_reconstructed_exact':reconstructed,'failed_json_stdout_and_exception_reconstructed':failed_json,
        'private_history_sha256':sha(private/(a['session']+'.json')),'private_access_sha256':sha(private/'complete-access-index.json'),'retained_cpu_fixtures':fixtures,
        'new_gpu_jobs':0,'full_original_quality_checks':0,'clean_samples':0,
        'access_review':[
            'All43 recorded commands/inputs/outputs/embedded scripts reviewed. Current508a/fullsmallindependentfeedback/selection/source/adapter/contracts read; calibration and tuning JSON prefixes opened, Nsys full arrays parsed with schema mistakes and integer-name grouping. Current execution_timeline JSON never opened: proposal overlap/33.71 and17.87ms partition reuse selection statements, not new direct read. Ownership prefix only. No original all-rank fullquality raw checks in compact route.',
            'Directory find / exposes external same-name candidate filenames, not observed file content reads. Remaining accessed content is supplied inputs and own /tmp CPU tests/candidate copies. No recorded privateconfig/laterauditor/reviewer content reads; FSnonhermetic, not absoluteabsence. Metadata output/display never printed.',
            'Tools31/32/34 run smallCPU GPT2 construction,31 FW/BW,34 exactcandidate import/build with missingparent returns{}. Tools40/41/43 two-input-set singleCPU simulation, no real ranks/processgroup/collective/CUDA or original fullmodel. Simulation computes all gradients before any update and uses synchronous tensor SUM, not exact _fast_step/Graphs/streams. It cannot validate intended overlap or async Work dependency.',
            'Preserved TypeError int-not-subscriptable20/21/23;24 converts numeric demangledName IDs to strings rather than resolving kernel names, so no matching NCCL-name line.31/34 token-id warnings retained.40None-grad update failure with stderr suppressed,41 repeats error through tail pipeline exit0,42 changes only toy grad reattachment,43 then passes. No candidate repair from that toy. Initialcode33/edit35 replaces synchronous allreduce with async_op=True and late Work.wait; finalcode496->500lines exact. No deleted fixtures observed; pre-edit candidate copy overwritten but exact write/edit retained.'
        ],
        'code_review':[
            'Generated3 contiguous gradient chunks via originalname classifier [38,48,62]parameters. Actual full-size chunk elements require ln_f placement: chunk0=21265152/chunk1=28351488/chunk2=74823168, total124439808; proposal comments misplace1536 ln_f elements into chunk2. Code uses real offset/numel, not these prose sizes.',
            'Whole-step replacement removes fixed-observation-route; fallback context.reference is originaladapter, not installed508a chain. Firstcall parent and secondsetup occur within originalquality sequence, not lastwarmup as proposal says. One of two continuous outerchecks would be optimized if reached.',
            'Exact _fast_step submits all backward graphs/events then async collective on internal backend stream; caller commstream div/event and update stream replay occur before final synchronize/Work.wait. No explicit collective-completion dependency before consumers. Device-wide synchronization after consumers cannot retroactively establish correct ordering.',
            'Later autograd.backward(inputs=nextchunk) still reads upper-layer saved parameters. Earlychunk SGD ready after itsgradient is not proof all other backward consumers finished. No originalCPU toy covered this ordering. Graph capture replay does not replace required dependency reasoning.',
            'Sharedgraphpool/replayorder/concurrentupdate ownership not universally proved; copiedtokens/limitedmodel-inputlist-paramptrguards, postupdatefallback restoresgradrefsNOTweights, peerMIN/rankasym/globaloverrideFalse limitations persist. Localgates2e-7/2e-5 and sentinel checks are not globalreduced/finite/fullcontinuousquality proof.',
            '42-44ms forecast/fixedfabric-onlyremaininglever/2rankSUMexactness and verbatimfallback guarantees unsupported. Same497759232B total if reached; threecollectives not volume reduction or independentacceptance. No GPU numerical rejection or native newmechanism measured.'
        ],
        'cpu_evidence':{'fixture_sha256':cpu['fixture_sha256'],'result_sha256':sha(cpu_root/'result/result.json'),'execution_sha256':sha(cpu_root/'execution.json'),
            'ordered_positive_error':0,'deferred_equal_peer_sum_weight_error':cpu['deferred_collective_completion']['max_weight_error'],'updates_before_sum':[0,1,2],
            'small_CPU_positive_max_abs':0,'small_CPU_early_update_next_segment_exception':cpu['saved_parameter_consumer']['early_update_then_next_backward_exception'],
            'scope':cpu['limits'],'source':'Exact candidate nestedfast/controlflow under inert scheduler + exactclassifier/helper on tinyCPU12layerGPT2; once only/no sourcecontrols/GPU/realcollectives. Conditional examples, not historical NCCLfault or fullquality.'},
        'gpu_eligible':False,'audit_reader_errors':['Local PowerShell/python nestedquote SyntaxError beforeSSH, then missingreader FileNotFoundError. Kept .scratch/fused-pipeline-history-reader-attempt-1.json; extracted history with real UTF8 reader afterward. No model/role/CPUfixture rerun.'],
        'next_decision':'ONE explicitly CPU-fact-assisted ordinary candidate correction on unchanged original33 inputs except append-only supplemental CPU facts and four exact previous-code/proposal/result/index files. Original production prompt/model/tools and existingfeedbacksuffix unchanged; no repair algorithm/handwritten implementation/terminalaudit table. No selector/GPU/automatictrial or repair-until-pass. If same branch remains concretely defective, preserve failure rather than append another automatic repair.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',report)
    public={k:v for k,v in report.items()if k not in ['session','private_history_sha256','private_access_sha256','retained_cpu_fixtures','machine']};public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json');write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),'read_lines':lines,'truncated':trunc,'reconstructed':reconstructed,'failed_json':failed_json,'candidate_bytes':report['candidate_bytes'],'proposal_bytes':report['proposal_bytes'],'cpu_result':report['cpu_evidence']['result_sha256'],'machine':report['machine']}))

if __name__=='__main__':main()
