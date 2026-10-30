"""Read-only terminal audit; no role, model, GPU, or collective execution."""
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
from scripts.experiments.v6_ddp_communication_redecision_20261031 import prepare

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-redecision-20261031')
def digest(x): return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()


def main():
    p=read(ROOT/'protocol.json'); terminal=read(ROOT/'followup-completed.json')
    assert terminal['phase']=='completed' and read(ROOT/'state.json')==terminal
    assert read(ROOT/'followup-results.json')==terminal['results'] and framework_source()==p['core']
    assert not (ROOT/'terminal-audit.json').exists()
    inputs,feedback,oldmeta,oldprompt,oldfiles,frozen,oldpath=prepare(SimpleNamespace(parent=ROOT.parent/'ddp-accepted-communication-trial-20261030'))
    result=terminal['results']['selection']; agent=result['agent']
    meta,prompt,files=load(Path(result['initial_snapshot'])); sandbox=Path(meta['sandbox'])
    expected={'inputs/'+k:v.encode() if isinstance(v,str) else v for k,v in inputs.items()}
    expected['inputs/previous-feedback.txt']=feedback.encode(); expected['inputs/worker-environment.json']=oldfiles['inputs/worker-environment.json']
    assert files==expected and prompt==oldprompt and meta['fresh_session']
    assert meta['model']==oldmeta['model'] and meta['tool_model_settings']==oldmeta['tool_model_settings'] and meta['framework_source']==p['core']
    assert all((sandbox/n).read_bytes()==v for n,v in files.items())
    assert all(sha(path)==h for path,h in p['frozen_identities'].items()) and all(sha(path)==h for path,h in frozen.items())
    assert sha(ROOT/'original-feedback.json')==p['raw_feedback_sha256'] and digest(feedback)==p['feedback_sha256']
    source=ProjectSearch(load_config(ROOT/'private-config.yaml',[]),Path(p['source'])); current=source.revisions.current()
    assert current.id==p['parent'] and len(current.bundles)==1
    assert all(files['inputs/source/'+f.path]==source.store.get_artifact(f.content_ref) for f in current.files)
    for k,n in [('request_sha256','request.json'),('prepared_sha256','prepared.json'),('discovery_sha256','discovery.json')]:
        assert read(ROOT/'preparation-reuse.json')[k]==sha(Path(p['source'])/n)
    selection=read(sandbox/'selection.json'); discovery=source.current_discovery()
    check_selection(sandbox,discovery,json.loads(inputs['search.json'])['allowed_scopes'],eligible_groups(discovery,leaf_only=False))
    assert sha(sandbox/'selection.json')==result['selection_sha256']
    private=ROOT/'terminal-private-history'; session=agent['session']; tools=read(private/(session+'-tools.json'))
    history=read(private/(session+'.json'))
    assert len(tools)==34 and agent['contract_repairs']==0
    access=[]; count=0; trunc=0; recon=[]
    for i,t in enumerate(tools,1):
        inp,out=t['input'],t['output']; extra=(t['metadata']or{}).get('outputPath')
        access.append({'n':i,'part':t['part'],'tool':t['tool'],'input':inp,'output_sha256':digest(out),'output_bytes':len(out.encode()),'status':t['status'],'exit':(t['metadata']or{}).get('exit'),'full_output_path':extra})
        assert extra is None
        if t['tool']=='read':
            lines=Path(inp['filePath']).read_text().splitlines()
            for row in out.splitlines():
                m=re.match(r'^(\d+): (.*)$',row)
                if not m:continue
                value,original=m[2],lines[int(m[1])-1]
                if '... (line truncated to ' in value:
                    assert original.startswith(value.split('... (line truncated to ')[0]);trunc+=1
                else:assert value==original,(i,m[1])
                count+=1
        elif i not in [4,7,8,11,32,33]:
            # All these exact recorded scripts were reviewed: JSON/interval arithmetic only.
            # Reconstruct displayed bytes against immutable inputs; never run shell/model/probe.
            argv=shlex.split(inp['command']); scripts=[argv[j+1] for j,v in enumerate(argv) if v=='-c']
            assert scripts
            base=sandbox if i==34 else sandbox/'inputs'
            def safe_open(name,mode='r',*args,**kwargs):
                assert mode=='r'
                path=(base/str(name)).resolve()
                assert path.is_relative_to(sandbox) and (str(path.relative_to(sandbox)) in files or path==sandbox/'selection.json')
                return io.StringIO(path.read_text())
            b=io.StringIO()
            for script in scripts:
                tree=ast.parse(script)
                for node in ast.walk(tree):
                    if isinstance(node,ast.Import): assert all(x.name in ['json','sys'] for x in node.names)
                    assert not isinstance(node,(ast.ImportFrom,ast.AsyncFunctionDef))
                    if isinstance(node,ast.Attribute):assert not node.attr.startswith('__') or node.attr=='__name__'
                    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):assert node.func.id not in ['exec','eval','compile','__import__']
                with contextlib.redirect_stdout(b):exec(compile(tree,'reviewed-json-output-reconstruction','exec'),{'open':safe_open})
            assert b.getvalue()==out,(i,'output reconstruction mismatch')
            recon.append(i)
    assert tools[7]['output']==str(len(feedback.encode()))+' '+str(sandbox/'inputs/previous-feedback.txt')+'\n'+feedback.encode()[:3000].decode()
    assert tools[31]['output']==files['inputs/observations/previous-after/binding_cost-2087a4f7.json'].decode()
    # Literal-only reconstruction of the sole selection write.
    script=tools[32]['input']['command'].split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]; tree=ast.parse(script)
    vals={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['rationale','hypothesis']:
            vals[n.targets[0].id]=ast.literal_eval(n.value)
    node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sel' for t in n.targets))
    vals['all_ids']=[c['id'] for c in json.loads(files['inputs/calls.json'])]
    rebuilt={ast.literal_eval(k):vals[v.id] if isinstance(v,ast.Name) else ast.literal_eval(v) for k,v in zip(node.keys,node.values)}
    assert (json.dumps(rebuilt,indent=2)+'\n').encode()==(sandbox/'selection.json').read_bytes()
    write_json(private/'complete-access-index.json',access);(private/'selection-write-script.py').write_text(script)
    timeline=json.loads(files['inputs/observations/current/execution_timeline-1221e205.json']); cal=json.loads(files['inputs/observations/current/collective_calibration-1af5837f.json'])
    trace_rows=[]
    for rank in timeline['ranks']:
        cs=rank['communication']['collectives']; kernels=[k for c in cs for k in c['kernels']]
        assert len(cs)==len(kernels)==13
        trace_rows.append({'rank':rank['rank'],'kernels':len(kernels),'logical_bytes':sum(c['logical_input_bytes']for c in cs),'sum_ms':sum(k['duration_us']for k in kernels)/1000,'chain_span_ms':(max(k['start_us']+k['duration_us']for k in kernels)-min(k['start_us']for k in kernels))/1000,'native_names':sorted(set(k['name']for k in kernels)),'scope':'Exact existing timeline period; not clean score/wire/critical-path bound'})
    calibration=[]
    for rank in cal['ranks']:
        for row in rank['rows']:
            assert statistics.median(row['samples_ms'])==row['median_completion_ms']
            calibration.append({'rank':rank['rank'],'bytes':row['logical_bytes'],'samples':row['samples_ms'],'median_ms':row['median_completion_ms'],'scope':rank['interpretation']})
    assert not list((ROOT/'selection/jobs').glob('*'))
    a={'at':datetime.now(timezone.utc).isoformat(),'terminal':terminal['finished_at'],'phase':terminal['phase'],'core':p['core'],'frozen_files':len(p['frozen_identities']),'all_frozen_unchanged':True,'generation':meta['id'],'session':session,'initial_files':len(files),'initial_bytes':sum(map(len,files.values())),'actual_inputs_route_prompt_model_tools_verified':True,'feedback_sha256':digest(feedback),'raw_feedback_sha256':p['raw_feedback_sha256'],'source_current':current.id,'selection_sha256':sha(sandbox/'selection.json'),'scope':selection['scope'],'groups':selection['groups'],'affected_calls':len(selection['affected_calls']),'output_writes':1,'output_edits':0,'output_write_reconstructed_exact':True,'contract_repairs':agent['contract_repairs'],'assistant_messages':agent['usage']['messages'],'tools':len(tools),'provider_estimate':agent['usage']['reported_cost'],'cost_scope':'Observed provider estimate, not invoice','private_history_sha256':sha(private/(session+'.json')),'private_access_sha256':sha(private/'complete-access-index.json'),'read_lines_verified':count,'explicit_truncated_prefixes':trunc,'read_only_output_reconstruction':recon+[8,32],'new_gpu_jobs':0,
       'access_review':['All34 tool commands/outputs/embedded scripts reviewed; reads supplied inputs/current installed ccd90/source, original failed diagnostics, current timeline/calibration/tuning/ownership. Only JSON/text arithmetic and schema checks; no model, CUDA initialization or collective probe observed. External FS nonhermetic.','No recorded read opens decision-evidence code/proposal/result/index or named CPU fixture. Tool9 prints nested CPU field structure only; new factual delivery is not full consumption. No observed later accepted/auditor/reviewer answer or private-provider-config read.','Tool29 has stderr suppression/head in a compound pipeline; complete available output retained and read-only reconstruction matches. No traceback/tool error observed in this session. No hidden exit0 treated as GPU quality.','First read-only audit reader rejected a harmless with-open context in pure JSON reconstruction. Attempt retained; reader permits with-open through a read-only, input-path-limited wrapper. No Agent/model/CPU fixture/GPU rerun.'],
       'claims_audit':['New selection proposes batching three no_sync microbatches, keeping13 AllReduce/497759232B; it is a computation hypothesis, not a new communication implementation. No candidate, quality, acceptance or production change.','Tool30 reads exact prior params.index Tensor-bool exception, but output still labels old timed eager fallback as sequential3-replay+merge and assigns66ms machinery cost. Neither fallback mechanism nor its causal cost follows from original code/results. Replacement does not automatically retain accepted parent Graph via context.references.','Tool21 explicitly exposes calibration not-a-lower-bound scope; selection nevertheless asserts transport/communication/volume floors, saturation and impossibility of other communication routes. Logical throughput and peer capability do not establish physical wire saturation or recoverable critical path.','Same-capture kernel order/gaps do not establish bucket GPU input-ready time, rank arrival, or that last bucket cannot be earlier. CPU hooks and inventory do not provide full GPU dependency proof; clocks/capture identities must remain separate.','GEMM CTA/occupancy/scheduling-overhead estimates and62-67ms prediction are unmeasured; no exact batched implementation/full gradient comparison. Quality, fallback neutrality, loss reduction equivalence and constraint are not guaranteed.'],
       'existing_trace_recomputed':trace_rows,'existing_calibration_recomputed':calibration,
       'reader_errors':'Attempt1 rejected harmless with-open; attempt2 rejected type(...).__name__. Exact errors retained; read-only reconstruction wrapper corrected, no role/model/CPU fixture/GPU rerun.',
       'next_decision':'Preserve selector output without generating its compute candidate. One isolated diagnostic_author for an unresolved communication evidence question on exact accepted DDP: what same-invocation observations can distinguish input/readiness/queue/completion waiting from transfer, and what remains unidentifiable? Author chooses bounded removable instrumentation; no manual collector/algorithm, optimization code, GPU author probes, reference copies or automatic execution. Review exact probe before any separately declared observation.77ac repair chain stays closed; no selector retry.',
       'machine':{'df':subprocess.check_output(['df','-B1','--output=size,used,avail','/root/autodl-tmp'],text=True),'gpu_workers':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)}}
    write_json(ROOT/'terminal-audit.json',a)
    public={k:v for k,v in a.items() if k not in ['session','private_history_sha256','private_access_sha256','existing_calibration_recomputed','machine']}
    public['terminal_audit_sha256']=sha(ROOT/'terminal-audit.json');write_json(ROOT/'public-terminal-audit.json',public)
    print(json.dumps({'audit':sha(ROOT/'terminal-audit.json'),'public':sha(ROOT/'public-terminal-audit.json'),'reads':count,'truncated':trunc,'reconstructed':recon,'machine':a['machine']}))

if __name__=='__main__':main()
