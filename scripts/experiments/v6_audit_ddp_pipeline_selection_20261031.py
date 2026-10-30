"""Read-only complete selector provenance/output audit; no model or GPU rerun."""
import ast
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
from types import SimpleNamespace

from kernel_optimizer.config import load_config
from kernel_optimizer.project.experiment_policy import check_selection, eligible_groups
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_pipeline_communication_selection_20261031 import prepare

ROOT=Path('/root/autodl-tmp/dev/v6-work/ddp-pipeline-communication-selection-20261031')
def digest(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()

def main():
    p=read(ROOT/'protocol.json');terminal=read(ROOT/'followup-completed.json')
    assert not(ROOT/'terminal-audit.json').exists()
    assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal
    assert read(ROOT/'followup-results.json')==terminal['results']
    assert not(ROOT/'controller.log').read_text().strip() and framework_source()==p['core']
    cfg=project_agent_config(load_config(ROOT/'private-config.yaml',[]))
    source,data,feedback,expected,frozen,details=prepare(SimpleNamespace(parent=ROOT.parent/'ddp-pipeline-communication-evidence-20261031'),cfg)
    r=terminal['results']['selection'];a=r['agent'];m,prompt,files=load(Path(r['initial_snapshot']));box=Path(m['sandbox'])
    om,op,of=load(ROOT.parent/'ddp-observation-feedback-selection-20261031/selection/generation-snapshots/generation-999c08dfa37c.json')
    assert files==expected and prompt==op and m['fresh_session']
    assert m['model']==om['model'] and m['tool_model_settings']==om['tool_model_settings'] and m['framework_source']==p['core']
    assert all((box/n).read_bytes()==v for n,v in files.items())
    assert p['input_hashes']=={n:digest(v) for n,v in files.items()}
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    assert all(sha(path)==h for path,h in frozen.items())
    assert sha(ROOT/'original-feedback.json')==p['raw_feedback_sha256']
    assert digest(feedback)==p['routed_feedback_sha256']==sha(ROOT/'routed-feedback.json')
    current=source.revisions.current();assert current.id==p['current'] and len(current.bundles)==1
    assert all(files['inputs/source/'+f.path]==source.store.get_artifact(f.content_ref) for f in current.files)
    for key,name in [('request_sha256','request.json'),('prepared_sha256','prepared.json'),('discovery_sha256','discovery.json')]:
        assert p[key]==sha(source.output/name)==read(ROOT/'preparation-reuse.json')[key]
    selection=read(box/'selection.json');d=source.current_discovery()
    check_selection(box,d,data['policy']['allowed_scopes'],eligible_groups(d,leaf_only=False))
    assert sha(box/'selection.json')==r['selection_sha256']
    events=[json.loads(x) for x in(ROOT/'selection/events.jsonl').read_text().splitlines()]
    assert [e['type'] for e in events]==['RUN_CREATED','PROJECT_AGENT_STARTED','PROJECT_GENERATION_SNAPSHOT','PROJECT_AGENT_FINISHED']
    assert events[-1]['payload']==a
    private=ROOT/'terminal-private-history';history=read(private/(a['session']+'.json'));tools=read(private/(a['session']+'-tools.json'))
    assert len(tools)==27 and a['contract_repairs']==0
    assert sum(x['data'].get('role')=='assistant' for x in history['message'])==a['usage']['messages']==16
    access=[];count=trunc=0;reconstructed=[]
    pure=[8,15,16,17,18,19,20,22,23,25,27]
    for i,t in enumerate(tools,1):
        inp,out=t['input'],t['output'];meta=t['metadata'] or {}
        assert meta.get('outputPath') is None and t['status']=='completed'
        access.append({'n':i,'part':t['part'],'tool':t['tool'],'input':inp,'output_sha256':digest(out),
            'output_bytes':len(out.encode()),'status':t['status'],'exit':meta.get('exit'),'truncated':meta.get('truncated')})
        if t['tool']=='read':
            path=Path(inp['filePath']);assert path.is_relative_to(box/'inputs');lines=files[path.relative_to(box).as_posix()].decode().splitlines()
            for number,content in re.findall(r'(?m)^(\d+): (.*)$',out):
                original=lines[int(number)-1]
                if '... (line truncated to ' in content:
                    assert original.startswith(content.split('... (line truncated to ')[0]);trunc+=1
                else:assert content==original,(i,number)
                count+=1
        elif i in pure:
            command=inp['command'];argv=shlex.split(command)
            script=argv[argv.index('-c')+1] if '-c' in argv else command.split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
            tree=ast.parse(script)
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):assert all(v.name=='json' for v in n.names)
                if isinstance(n,ast.ImportFrom):assert n.module=='collections' and all(v.name=='defaultdict' for v in n.names)
                assert not isinstance(n,ast.AsyncFunctionDef)
                if isinstance(n,ast.Attribute):assert not n.attr.startswith('__')
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in ['exec','eval','compile','__import__']
            def safe_open(name,mode='r',*args,**kwargs):
                path=(box/str(name)).resolve();assert mode=='r' and path.is_relative_to(box)
                relative=path.relative_to(box).as_posix()
                raw=files[relative] if relative in files else (box/'selection.json').read_bytes() if relative=='selection.json' else None
                assert raw is not None;return io.StringIO(raw.decode())
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):exec(compile(tree,'reviewed-json-reconstruction','exec'),{'open':safe_open})
            generated=buf.getvalue()
            assert generated==out,(i,'output mismatch');reconstructed.append(i)
        elif t['tool']=='write':
            assert i==26 and Path(inp['filePath'])==box/'selection.json'
            assert inp['content'].encode()==(box/'selection.json').read_bytes()
    assert sum(t['tool']=='write' for t in tools)==1 and not any(t['tool']=='edit' for t in tools)
    write_json(private/'complete-access-index.json',access)

    # Same retained capture, raw SQLite ns and stream intervals only. No new capture.
    native=json.loads(files['inputs/observations/current/nsys-9ed9145c.json']);native_checks=[]
    for rank in native['ranks']:
        con=sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True)
        ks=list(con.execute('select k.start,k.end,s.value,k.streamId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start'));con.close()
        assert sorted((k[0],k[1],k[2],k[3])for k in ks)==sorted((k['start'],k['end'],k['name'],k['streamId'])for k in rank['kernels'])
        nk=[k for k in ks if 'nccl'in k[2].lower()];assert len(nk)==8
        def union(items):
            end=None;total=0
            for start,finish,*_ in sorted(items):
                total+=max(0,finish-max(start,start if end is None else end));end=max(finish,finish if end is None else end)
            return total
        pre=[k for k in ks if k[1]<=nk[0][0]]
        during=[k for k in ks if k[0]>=nk[0][0]and k[1]<=nk[-1][1]and 'nccl'not in k[2].lower()]
        post=[k for k in ks if k[0]>=nk[-1][1]]
        noncomm=[k for k in ks if 'nccl'not in k[2].lower()and k[3]in {n[3]for n in nk}]
        native_checks.append({'rank':rank['rank'],'sqlite_sha256':sha(rank['evidence']),'kernels':len(ks),
            'span_ms':(max(k[1]for k in ks)-min(k[0]for k in ks))/1e6,'nccl_sum_ms':sum(e-s for s,e,_,_ in nk)/1e6,
            'nccl_streams':dict(Counter(k[3]for k in nk)),'non_nccl_on_comm_stream_count':len(noncomm),'non_nccl_on_comm_stream_sum_ms':sum(e-s for s,e,_,_ in noncomm)/1e6,
            'pre_count':len(pre),'pre_union_ms':union(pre)/1e6,'during_count':len(during),'during_union_ms':union(during)/1e6,
            'post_count':len(post),'post_union_ms':union(post)/1e6,'gaps_between_nccl_ms':[(b[0]-a[1])/1e6 for a,b in zip(nk,nk[1:])]})
    write_json(ROOT/'native-arithmetic-check.json',{'scope':'Same retained SQLite/normalized records and proper ns/1e6 units only; no new GPU/capture/clean inference.',
        'rows':native_checks,'unit_error':'Tools19/20 label nanosecond span as us and ns/1000 as ms. Final rationale rescales several numbers plausibly but wrongly calls the same-capture interval a causal critical path and imports separate calibration/tuning as LL-overhead attribution.'})
    assert not list((ROOT/'selection/jobs').glob('*'))
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],'reason':'batch_finished',
        'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'source_current':current.id,
        'generation':m['id'],'session':a['session'],'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
        'actual_inputs_route_prompt_model_tools_verified':True,'raw_feedback_sha256':p['raw_feedback_sha256'],'feedback_sha256':digest(feedback),
        'selection_sha256':sha(box/'selection.json'),'selection_bytes':(box/'selection.json').stat().st_size,
        'scope':selection['scope'],'groups':selection['groups'],'affected_calls':len(selection['affected_calls']),
        'output_writes':1,'output_edits':0,'output_write_reconstructed_exact':True,'contract_repairs':0,
        'assistant_messages':16,'tools':27,'provider_estimate':a['usage']['reported_cost'],'cost_scope':'Provider estimate, not invoice',
        'read_lines_verified':count,'explicit_truncated_prefixes':trunc,'pure_outputs_reconstructed_exact':reconstructed,
        'private_history_sha256':sha(private/(a['session']+'.json')),'private_access_sha256':sha(private/'complete-access-index.json'),
        'native_arithmetic_check_sha256':sha(ROOT/'native-arithmetic-check.json'),'native_arithmetic':native_checks,'new_gpu_jobs':0,
        'access_review':[
            'All27 recorded tool inputs/outputs and embedded Python scripts reviewed. Every read line/prefix verified;11 JSON-only outputs independently reconstructed exactly; one selection write, no edits. Full current02b933/originalsource/bindingcontract/routedindependentfeedback read. FourcurrentcollectorJSON opened: full calibration samples and tuning decisions, all native kernels parsed for arithmetic, selected timeline collective data and partial other fields printed. Resource read byte-truncated, ownership prefix only; exposure is not full reasoning correctness.',
            'Recorded commands only supplied files/directories and pure JSON/schema/arithmetic. No observed model/tensor/CUDA/processgroup/collective probe or privateconfig/lateraudit/reviewer answer content access. opencode.json name listed only; filesystem nonhermetic, no absolute absence claim.',
            'All27 statescompleted/observed bash exits0/0schema. Tool9 missing previous-candidate/source directory stderrhidden by compound shell followed by successful ls; do not call every component successful. Unit labels in19/20 retained, no experiment rerun.'
        ],
        'claims_audit':[
            'Selection proposes dedicated NCCL group with NCCL_PROTO Simple, averaging folded into updates, optional fine9 segmentation. Original seven groups/seven calls preserved; no candidate generated or parameter selected by auditor.',
            'It explicitly saw tuning interpretation that kernel entry names do not identify selected protocol and logs include setup/warmup. Nevertheless rationale attributes six-chunk excess over separate calibration to LL instead of SIMPLE and predicts2.9ms recovery; this is not supported by per-event protocol identity or controlled comparison.',
            'Same-stream native intervals establish serialization/overlap only, not a37ms causal critical path or the only remaining optimization route. Distinct periods cannot be subtracted to infer protocol cost or wire/critical-path benefit. Synthetic logical rate is not a fixed SHM ceiling.',
            'CPU submit bounds <=0.395ms do not establish GPU arrival skew or rule out producer waiting. Timeline is a separate diagnostic, not clean timing. Actual497759232B is not a universal communication-volume lower bound.',
            'FP32 scale0.5 exactness alone does not certify fused-add rounding, underflow/overflow, gradient/update ordering, exact replicas and original continuous full-state quality. Reference after whole-bundle replacement is originaladapter, not automatically the installed optimized parent.',
            'Protocol environment/new-group scope and whether the installed NCCL dispatches multiple functions behind a named kernel remain concrete unanswered evidence questions. Optional fine9 is not authorized as an automatic retry of closed parameter/correction chains.',
            'Original baseline anchor133.379621 is a constraint anchor; actual independent reference114.757729 is separate. Currentaccepted joint57.8545851% remains; no new quality/acceptance or isolated communication effect.'
        ],
        'next_decision':'Preserve this selection without candidate/GPU/parameterexecution. Declare ONE bounded read-only protocol-attribution evidence author using exact current implementation, selection and original current artifacts. Ask what existing evidence can establish about actual steady-collective protocol identity and setting/group scope, and what remains missing. No desired protocol, handwritten implementation, audit correction table or live collector/observer repair supplied; generation only, inspect outputs before any separate analyzer execution. Closed chains stay closed.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',report)
    public={k:v for k,v in report.items() if k not in ['session','private_history_sha256','private_access_sha256','machine']};public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),'read_lines':count,'truncated':trunc,'reconstructed':reconstructed,'native':native_checks,'machine':report['machine']}))

if __name__=='__main__':main()
