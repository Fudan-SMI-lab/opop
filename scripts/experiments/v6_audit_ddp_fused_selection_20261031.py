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
from scripts.experiments.v6_ddp_fused_communication_selection_20261031 import prepare

ROOT=Path('/root/autodl-tmp/dev/v6-work/ddp-fused-communication-selection-20261031')
def digest(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()

def main():
    p=read(ROOT/'protocol.json');terminal=read(ROOT/'followup-completed.json')
    assert not(ROOT/'terminal-audit.json').exists()
    assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal
    assert read(ROOT/'followup-results.json')==terminal['results']
    assert not(ROOT/'controller.log').read_text().strip() and framework_source()==p['core']
    cfg=project_agent_config(load_config(ROOT/'private-config.yaml',[]))
    source,data,feedback,expected,frozen,details=prepare(SimpleNamespace(parent=ROOT.parent/'ddp-fused-communication-evidence-20261031'),cfg)
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
    assert len(tools)==31 and a['contract_repairs']==0
    assert sum(x['data'].get('role')=='assistant' for x in history['message'])==a['usage']['messages']==24
    access=[];count=trunc=0;reconstructed=[]
    pure=[11,19,20,21,22,23,24,25,26,27,28,29,31]
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
            argv=shlex.split(inp['command']);script=argv[argv.index('-c')+1];tree=ast.parse(script)
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):assert all(v.name in ['json','itertools'] for v in n.names)
                assert not isinstance(n,(ast.ImportFrom,ast.AsyncFunctionDef))
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
            if i==23:
                counts=Counter(generated.splitlines());generated=''.join(f'{n:7d} {line}\n' for line,n in sorted(counts.items(),key=lambda v:(v[1],v[0]),reverse=True)[:40])
            assert generated==out,(i,'output mismatch');reconstructed.append(i)
        elif t['tool']=='write':
            assert i==30 and Path(inp['filePath'])==box/'selection.json'
            assert inp['content'].encode()==(box/'selection.json').read_bytes()
    assert sum(t['tool']=='write' for t in tools)==1 and not any(t['tool']=='edit' for t in tools)
    write_json(private/'complete-access-index.json',access)

    # Independently check the selector's normalized-timestamp arithmetic against
    # the retained SQLite from that SAME native capture; no new capture/probe.
    native=json.loads(files['inputs/observations/current/nsys-9ed9145c.json']);native_checks=[]
    for rank in native['ranks']:
        con=sqlite3.connect('file:'+rank['evidence']+'?mode=ro',uri=True)
        ks=list(con.execute('select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start'));con.close()
        nk=next(k for k in ks if 'nccl' in k[2].lower())
        intervals=[]
        for start,end,_ in ks:
            if intervals and start<=intervals[-1][1]:intervals[-1][1]=max(end,intervals[-1][1])
            else:intervals.append([start,end])
        before=[k for k in ks if k[1]<=nk[0]];after=[k for k in ks if k[0]>=nk[1]]
        assert len(ks)==len(rank['kernels']) and sorted((k[0],k[1],k[2]) for k in ks)==sorted((k['start'],k['end'],k['name']) for k in rank['kernels'])
        native_checks.append({'rank':rank['rank'],'evidence_sha256':sha(rank['evidence']),'kernels':len(ks),
            'union_ms':sum(e-s for s,e in intervals)/1e6,'span_ms':(max(k[1] for k in ks)-min(k[0] for k in ks))/1e6,
            'nccl_ms':(nk[1]-nk[0])/1e6,'pre_nccl_kernel_sum_ms':sum(e-s for s,e,_ in before)/1e6,
            'post_nccl_kernel_sum_ms':sum(e-s for s,e,_ in after)/1e6,'pre_count':len(before),'post_count':len(after)})
    write_json(ROOT/'native-arithmetic-check.json',{'scope':'Read-only same-capture SQLite/normalized JSON identity and units; no GPU rerun, wire or clean critical-path inference.',
        'rows':native_checks,'unit_error':'Agent tools28/29 divide nanoseconds by1000 yet print ms: outputs are microseconds. Final rationale52.2/17.87/31.53/2.80ms agrees with proper ns/1e6 recomputation, but command labels remain wrong. Pre/post counts955/149, not both compute-before-AR; SGD followsAR.'})
    assert not list((ROOT/'selection/jobs').glob('*'))
    assert all(sha(path)==h for path,h in p['frozen_identities'].items())
    report={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],'reason':'batch_finished',
        'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'source_current':current.id,
        'generation':m['id'],'session':a['session'],'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),
        'actual_inputs_route_prompt_model_tools_verified':True,'raw_feedback_sha256':p['raw_feedback_sha256'],'feedback_sha256':digest(feedback),
        'selection_sha256':sha(box/'selection.json'),'selection_bytes':(box/'selection.json').stat().st_size,
        'scope':selection['scope'],'groups':selection['groups'],'affected_calls':len(selection['affected_calls']),
        'output_writes':1,'output_edits':0,'output_write_reconstructed_exact':True,'contract_repairs':0,
        'assistant_messages':24,'tools':31,'provider_estimate':a['usage']['reported_cost'],'cost_scope':'Provider estimate, not invoice',
        'read_lines_verified':count,'explicit_truncated_prefixes':trunc,'pure_outputs_reconstructed_exact':reconstructed,
        'private_history_sha256':sha(private/(a['session']+'.json')),'private_access_sha256':sha(private/'complete-access-index.json'),
        'native_arithmetic_check_sha256':sha(ROOT/'native-arithmetic-check.json'),'native_arithmetic':native_checks,'new_gpu_jobs':0,
        'access_review':[
            'All31 recorded inputs/outputs/embedded scripts reviewed. Full current508a, originalsource/adapter/contracts, routed independent decision/measurement read. Allfour currentcollectorJSON actually opened; calibrationrankrecords printed fully, tuningdecisions printed, timeline/rank0 and collectionplan/ownership only prefixes. Nsys arrays all parsed for arithmetic but not all records printed. Delivery is not complete textual consumption.',
            'Only JSON parsing/arithmetic/schema and supplied directory/file reads; no recorded model/tensor/CUDA/processgroup/collective probe, external audit/reviewer/privateconfig content read. opencode.json directory name exposure only. FSnonhermetic, no absolute absence claim.',
            'All31 tool states completed, shell exits0, no schema repair. Embedded tool28/29 unit-label error retained. No ad hoc retries/probe/test. Startup whole-line-versus-explicit-prefix reader error remains separate; no Agent rerun.'
        ],
        'claims_audit':[
            'New selector-authored execution-plan hypothesis segments batched backward, then chunk SUM/divide and SGD with readiness events; distinct from closed77ac multi-communicator/Future repair. Current actualsource and samecapture no compute/communication overlap support testing serialization changes, not an observed gain.',
            'Standalone32.51ms reference is not fixed wire time, a lower bound, proof of no collective headroom or exclusive remaining optimization route. Topology/SHM channels do not prove policy cannot change any transport-related behavior. Separate tuning decisions are not individually joined to timed kernels.',
            'Payload497759232B is actual current exactgradient communication, not a proved universal irreducible bytes lower bound. Bucketinventory membership does not uniquely prove current timed flat-buffer ownership/readiness.',
            'Prediction29-37ms from21+31.5/C lacks measured chunk scheduling/transport/launch/SM interference and omits aggregate traffic/dependency constraints. 20.7ms nonNCCL includes2.8ms AFTERAR, not all pre-collective compute. No clean savings or fixedwire inference from native intervals.',
            'Identical chunk boundaries and FP32 SUM do not by themselves certify accumulation/reduction order, exactreplicas, source tolerance or continuous updates. EarlySGD must preserve all backward parameter consumers including tiedembedding; gradientready is not parameter-safe-to-update proof.',
            'Replacing installed whole-step bundle does not automatically preserve its fallback; context.references points originaladapter. Crossrank agreement does not establish arbitrary rankasymmetric failure neutrality. Existing508a conditional fallback/owner limits persist.',
            'No candidate code/newquality/formal/independentacceptance generated here. Source/mapping/CPUassistance and predecessorplanner historicalplan access remain provenance boundaries.'
        ],
        'next_decision':'Proceed with ONE isolated ordinary candidate generation from exact unedited selection/current508a/ordinaryinputs. Narrow support is actual explicit serialization and real whole-step producer/collective/update structure, NOT claimed fixedwire/predicted29-37ms/numerical or fallback guarantees. Generated implementation must establish concrete dependencies and pass review before any declaredGPU; no manualgroup/K/code/collector or terminalaudit answers supplied, no automatictrial or repeatedrepair.',
        'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',report)
    public={k:v for k,v in report.items() if k not in ['session','private_history_sha256','private_access_sha256','machine']};public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json')
    write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),'read_lines':count,'truncated':trunc,'reconstructed':reconstructed,'native':native_checks,'machine':report['machine']}))

if __name__=='__main__':main()
