"""Read-only complete recorded generation audit; no Agent, model or probe rerun."""
import ast,hashlib,json,re,shutil,subprocess
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
from scripts.experiments.v6_ddp_fused_pipeline_cpu_feedback_20261031 import prepare

ROOT=Path('/root/autodl-tmp/dev/v6-work/ddp-fused-pipeline-cpu-feedback-20261031')
CODE='02b93360c4d53add1ef14fc91fb32f757b09db43597888f880c1cd25793d614a'
PROPOSAL='dabb839bd7b29b8b073ff0a134b3114f602b4797c7c37331b14b0ab4424b4a11'
def digest(v):return hashlib.sha256(v.encode()if isinstance(v,str)else v).hexdigest()

def main():
 p=read(ROOT/'protocol.json');terminal=read(ROOT/'followup-completed.json');result=terminal['results']['candidate'];a=result['agent']
 assert not(ROOT/'terminal-audit.json').exists()
 assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal and read(ROOT/'followup-results.json')==terminal['results']
 assert framework_source()==p['core'] and not(ROOT/'controller.log').read_text()
 inputs,feedback,om,op,old,_,frozen,details=prepare(SimpleNamespace(parent=ROOT.parent/'ddp-fused-pipeline-candidate-20261031'))
 meta,prompt,files=load(Path(result['initial_snapshot']));box=Path(meta['sandbox'])
 expected={'inputs/'+n:v.encode()if isinstance(v,str)else v for n,v in inputs.items()};expected.update({'inputs/previous-feedback.txt':feedback.encode(),'inputs/worker-environment.json':old['inputs/worker-environment.json']})
 assert files==expected and all((box/n).read_bytes()==v for n,v in files.items()) and len(files)==37 and sum(map(len,files.values()))==8682320
 assert prompt==op and meta['model']==om['model'] and meta['tool_model_settings']==om['tool_model_settings'] and meta['framework_source']==p['core']
 assert all(files[n]==v for n,v in old.items()if n!='inputs/previous-feedback.txt')
 assert {k:v for k,v in json.loads(feedback).items()if k!='supplemental_evidence'}=={k:v for k,v in json.loads(old['inputs/previous-feedback.txt']).items()if k!='supplemental_evidence'}
 assert json.loads(feedback)['supplemental_evidence']['cpu_contract_observations']==json.loads(files['inputs/cpu-contract-evidence/cpu-observations.json'])
 assert digest(feedback)==p['feedback_sha256']==sha(ROOT/'routed-feedback.json') and sha(ROOT/'original-feedback.json')==p['original_feedback_sha256']
 assert sha(ROOT/'diagnostic-feedback.json')==p['augmented_feedback_sha256']
 assert all(sha(n)==h for n,h in p['frozen_identities'].items()) and p['frozen_identities']==frozen
 source=ProjectSearch(load_config(ROOT/'private-config.yaml',[]),Path(p['source']));current=source.revisions.current()
 assert current.id==p['parent'] and current.bundles==('fixed-observation-route',)
 assert all(files['inputs/source/'+f.path]==source.store.get_artifact(f.content_ref)for f in current.files)
 for key,name in [('request_sha256','request.json'),('prepared_sha256','prepared.json'),('discovery_sha256','discovery.json')]:assert sha(source.output/name)==p[key]
 validate_proposal(box);proposal=read(box/'proposal.json');assert sha(box/'candidate.py')==CODE==result['candidate_sha256'] and sha(box/'proposal.json')==PROPOSAL==result['proposal_sha256']
 assert proposal['requires']==[] and proposal['replaces']==list(current.bundles)
 assert proposal['parameter_space']['params'][0]['name']=='chunk_plan' and proposal['parameter_space']['params'][0]['choices']==['fine8','fine9','coarse5'] and proposal['parameter_space']['constraints']==[]
 events=[json.loads(x)for x in(ROOT/'candidate/events.jsonl').read_text().splitlines()]
 assert [e['type']for e in events]==['RUN_CREATED','PROJECT_AGENT_STARTED','PROJECT_GENERATION_SNAPSHOT','PROJECT_AGENT_FINISHED'] and events[-1]['payload']==a
 assert read(ROOT/'session-health.json')['status']=='passed'
 private=ROOT/'terminal-private-history';history=read(private/(a['session']+'.json'));tools=read(private/(a['session']+'-tools.json'))
 assert len(tools)==68 and a['contract_repairs']==0 and a['usage']['messages']==sum(x['data'].get('role')=='assistant'for x in history['message'])==58
 contents={};access=[];lines=trunc=0;writes={};edits={};fixtures={};retained=private/'retained-cpu-fixtures';retained.mkdir(exist_ok=True)
 for n,t in enumerate(tools,1):
  inp,out=t['input'],t['output'];md=t['metadata']or{}
  assert md.get('outputPath')is None and (t['status']=='completed'or n==7)
  access.append({'n':n,'part':t['part'],'tool':t['tool'],'input':inp,'status':t['status'],'exit':md.get('exit'),'truncated':md.get('truncated'),'output_bytes':len(out.encode()),'output_sha256':digest(out)})
  if t['tool']in['write','edit']:
   path=Path(inp['filePath']);assert path.parent==box;name=path.name
   if t['tool']=='write':contents[name]=inp['content'];writes[name]=writes.get(name,0)+1;(private/('initial-'+name)).write_text(contents[name])
   else:
    assert contents[name].count(inp['oldString'])==1;contents[name]=contents[name].replace(inp['oldString'],inp['newString'],1);edits[name]=edits.get(name,0)+1
   (private/(f'after-tool-{n}-'+name)).write_text(contents[name])
  elif t['tool']=='read':
   path=Path(inp['filePath'])
   if n==7:assert path==box/'adapter.py' and not out;continue
   assert path.is_relative_to(box)
   raw=contents[path.name]if path.parent==box else files[path.relative_to(box).as_posix()].decode();actual=raw.splitlines()
   for number,shown in re.findall(r'(?m)^(\d+): (.*)$',out):
    original=actual[int(number)-1]
    if '... (line truncated to 'in shown:assert original.startswith(shown.split('... (line truncated to ')[0]);trunc+=1
    else:assert original==shown,(n,number)
    lines+=1
  elif t['tool']=='bash':
   command=inp['command']
   for dest,body in re.findall(r"cat > (\S+) <<'EOF'\n(.*?)\nEOF",command,re.S):
    assert dest.startswith('/tmp/opencode/chunkcheck/');fixtures[dest]=body+'\n';(retained/(f'tool-{n}-'+Path(dest).name)).write_text(body+'\n')
   if n==30:
    tree=ast.parse(command.split("<<'EOF'\n",1)[1].split('\nEOF',1)[0]);call=next(x for x in ast.walk(tree)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='replace');before,after=map(ast.literal_eval,call.args)
    target='/tmp/opencode/chunkcheck/check.py';assert before in fixtures[target];fixtures[target]=fixtures[target].replace(before,after)
  else:raise AssertionError((n,t['tool']))
 assert writes=={'candidate.py':1,'proposal.json':1}and edits=={'candidate.py':5}
 for name in contents:assert contents[name].encode()==(box/name).read_bytes()
 assert 'DEFAULT_PLAN = "fine8"'in contents['candidate.py']
 for path,text in fixtures.items():assert Path(path).read_text()==text;shutil.copyfile(path,retained/Path(path).name)
 assert tools[15]['output'].startswith(files['inputs/search.json'][:3000].decode()+'\n')
 assert tools[16]['output']==files['inputs/cpu-contract-evidence/index.json'].decode()
 prev_api=ROOT.parent/'ddp-fused-pipeline-candidate-20261031/audit/cpu-stream-consumer/result/installed-all-reduce-source.txt'
 assert tools[17]['output']==prev_api.read_text()+'\n'
 lib=Path('/root/autodl-tmp/dev-venv/lib/python3.12/site-packages')
 assert tools[26]['output']==(lib/'torch/distributed/c10d_logger.py').read_text()
 text=(lib/'transformers/utils/output_capturing.py').read_text()
 for n,term,length in [(24,'def install_all_output_capturing_hooks',3500),(25,'def recursively_install_hooks',4000),(26,'def install_output_capuring_hook',3500)]:
  i=text.find(term);assert i>=0 and tools[n-1]['output']==text[i:i+length]+'\n'
 i=text.find('def capture_outputs');assert tools[22]['output'].split('\n\n',1)[1]==text[i-200:i+6000]+'\n'
 modeltext=(lib/'transformers/models/gpt2/modeling_gpt2.py').read_text();modeltree=ast.parse(modeltext);model_lines=modeltext.splitlines(keepends=True)
 for n,cls in [(20,'GPT2Model'),(32,'GPT2LMHeadModel'),(59,'GPT2Block')]:
  c=next(x for x in modeltree.body if isinstance(x,ast.ClassDef)and x.name==cls);f=next(x for x in c.body if isinstance(x,ast.FunctionDef)and x.name=='forward');start=min([f.lineno]+[d.lineno for d in f.decorator_list]);src=''.join(model_lines[start-1:f.end_lineno]);src=src[-2500:]if n==59 else src
  assert tools[n-1]['output']==src+'\n',(n,'source output')
 write_json(private/'complete-access-index.json',access)
 assert not list((ROOT/'candidate/jobs').glob('*'))and all(sha(n)==h for n,h in frozen.items())
 report={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],'reason':'batch_finished','core':p['core'],'frozen_files':len(frozen),'frozen_identities':frozen,'all_frozen_unchanged':True,'source_current':current.id,'generation':meta['id'],'session':a['session'],
 'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'actual_inputs_route_prompt_model_tools_verified':True,'original_nonfeedback_inputs_equal':32,'assistance':'Explicit developer CPU facts; ordinary prompt unchanged/no extra repair paragraph, not unaided or same-snapshot AB. Frozen wording33nonfeedback corrected in startup as32of33 without mutation.',
 'candidate_sha256':CODE,'candidate_bytes':len(contents['candidate.py'].encode()),'proposal_sha256':PROPOSAL,'proposal_bytes':len(contents['proposal.json'].encode()),'output_writes':writes,'output_edits':edits,'exact_writes_and_edits_reconstructed':True,'parameter_space':proposal['parameter_space'],'fixed_parameters':{},'fixed_default_from_generated_code':'fine8','requires':[],'replaces':proposal['replaces'],'contract_repairs':0,'assistant_messages':58,'tools':68,'provider_estimate':a['usage']['reported_cost'],'cost_scope':'Provider estimate, not invoice','read_lines_verified':lines,'explicit_truncated_prefixes':trunc,'exact_other_outputs':[16,17,18,20,23,24,25,26,27,32,59],
 'private_history_sha256':sha(private/(a['session']+'.json')),'private_access_sha256':sha(private/'complete-access-index.json'),'retained_cpu_fixtures':{Path(n).name:digest(v)for n,v in fixtures.items()},'new_gpu_jobs':0,'full_original_quality_checks':0,'clean_samples':0,
 'access_review':[
 'All68 recorded tool inputs/outputs/scripts reviewed; full inline CPU observations in tool5 and full oldeef828 code14, index17 read. Named independent fixture and standalone CPU-result file were not opened, but facts were consumed inline. Oldproposal/selection long-line prefixes truncated. No currentcollector JSON or current508a/original training source file opened in this generation: tool16 lists collectors and search first3000bytes. Their prediction claims are inherited selection/proposal content, not new capture analysis.',
 'Actual library all_reduce/GPT2/Transformers output-hook source reads. CPU model probes small4layerhidden16 and actual12layerhidden768/vocab50257/npos1024; full-size input is one seed31000 sequence repeated4 times, oneCPU/rank, not original four independent microbatches/two-rank workload. No recorded CUDA allocation/model/processgroup/real collective; build67 invokes torch.cuda.is_available but uses unsupported path, not CUDA quality. Nonhermetic FS/no absoluteabsence of external accesses; no recorded private config/reviewer/later audit content reads. Config filename only in listings. Full raw history stays server.',
 'Copied helper tests do real eager CPU FWBW/updates, not exact candidate setup/faststep binding or CUDA capture/replay/streams. Tool44 tiny chain fullgrad bitwise;45/52 earlyupdates pass tinyCPU;66 full-size singleCPU updatedparams bitwise0 after fixing test scale. No two continuous updates from the live generated CUDA implementation, no exact replicas or NCCL-completion proof.',
 'Preserved read7 missingrootadapter,ls12 missingpath exit2,library21AttributeError/22ImportError,28SyntaxError,29Tensor-vs-NoneTypeError.30/33/35/36/38 tied-grad mismatches despite exit0,37 tinyrounding residual,40 misleading/trivial assertions,43 rootgrad doubling.53 handwritten metadata total180979968 instead124439808 and wronggroupadjacency assertion;54 sameassert;56 candidateedit corrects adjacency.63 sim divides2 withoutpeerSUM gives0.00013570633018389344diff;64 identicalCPUbackwards0;65 fails toclearpreviousparamgrads beforechain so reportsdifferences;66 test-only corrected single-rank scaling yields0. Bothfailedandpassedoutputs preserved.67unsupportedparent only;68 deletes own__pycache__. Overwritten fixture versions reconstructed from commands; final fixtures retained, not all ephemeral tensors.'
 ],
 'code_review':{'fixed_execution_justified':True,'scope':'One original full B/C validation of exact bytes with empty parameters, thereby generated DEFAULT_PLAN fine8. No parameter search/otherchoices/retry/handfix. Eligibility is scoped empirical validation, not correctness certification.',
 'findings':[
 'Async Work.wait is now inside commstream before division/done-event, and update waits producer/done plus declared consumer events. Chain cuts limit traversal; tiedhead gradient explicit dlogits.T@z plus finalembedding. Previous concrete two CPU counterexamples motivate changes; no new independent execution of those fixtures this turn.',
 'Candidate selfselects coarse5/fine8/fine9 parameter choices; fixed emptyparams uses own fine8 default, not an auditor-selected chunking. Fine8 eightchunks:1536 + six14175744 +39383808 elements=124439808; byte sizes6144 + six56702976 +157535232=497759232. Logical total unchanged, actual steady collective/count/overlap remain unmeasured.',
 'Setup gates compare sequential-local vs chained batched fullgrads and one uncommunicated graphreplay, not the true overlapped allreduce/update path. CPU manual two-contribution exactness does not guarantee GPU kernel/numerics or two-rank replicas. Original full outerchecks remain authority. Firstparent/secondsetup is quality, not lastwarmup. REPLACEMENT fallback uses originaladapter, not installed508a chain.',
 'Shared-pool capture/replay/cutgrad address/lifetime dependency still needs actual CUDA validation; fast replay has no Python clear of cut.grad but captured first allocation/write behavior cannot be certified from eager CPU tests. Do not claim accumulation failure from that absence alone. Success path removes13hooks; setup failure before removal leaves hooks/cut_log holding model outputs, not unconditional cleanup.',
 'Copiedtokens/limitedptridentityguards lack allcontent/dtype/shape/lr/alias coverage; fallback restoresgradrefsNOTupdatedweights, so postupdateerror canexecuteparent again. Rankasymmetricfailure aftercollectives and globaloverridefinallyFalse limitations persist. PeerMIN/MAX checks plan agreement, not universal fault safety.',
 '42-47ms bound/onlyscheduling lever/15.3GBps wire/unchanged exact GPU floating guarantee and no-regression fallback claims unsupported. CUDA quality, clean scores, memory and native communication have not run. No handfix or another candidate correction scheduled.'
 ]},
 'audit_reader_errors':['Read-only local rg used nonexistent project/parameters.py; no source/model/GPU execution. Any subsequent audit-reader failure is retained separately.'],
 'next_decision':'ONE declared fixed current508a B then exact02b933 C under original evaluator, emptyparameters/generateddefault only. Baseline fullquality gate; original2continuous/2warmup/5clean/rank plus original3 postclean metricprofiles. No automatic formal acceptance/parameter retries/native/independent/publication. Preserve any failure and close automatic correction chain.',
 'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
 write_json(ROOT/'terminal-audit.json',report)
 public={k:v for k,v in report.items()if k not in ['frozen_identities','session','private_history_sha256','private_access_sha256','retained_cpu_fixtures','machine']};public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json');write_json(ROOT/'public-terminal-audit.json',public)
 print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),'read_lines':lines,'truncated':trunc,'fixture_files':len(fixtures),'candidate_bytes':report['candidate_bytes'],'proposal_bytes':report['proposal_bytes'],'machine':report['machine']}))

if __name__=='__main__':main()
