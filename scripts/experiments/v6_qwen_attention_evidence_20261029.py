"""First-attention interface controls, followed by one factual candidate continuation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.agents import ProjectAgents
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.runtime_config import project_agent_config
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import Runtime
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def now(): return datetime.now(timezone.utc).isoformat()


def attention_worker(context):
    import torch
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache, StaticCache
    from kernel_optimizer.project.evaluation import load_adapter, compare
    params=context['parameters']; output=Path(context['output']); quality=params['request']['quality']
    preceding=read(params['preceding_report']); prefix=torch.load(preceding['prefix']['raw']['path'],map_location='cpu',weights_only=True)
    assert sha(preceding['prefix']['raw']['path'])==preceding['prefix']['raw']['sha256']
    adapter=load_adapter(Path(params['adapter']));state=adapter.prepare(context,params['request']);model=state['model']
    policies=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.get_default_dtype())
    original=F.scaled_dot_product_attention; captured={}; rows={}; raw={}
    def save(name,values):
        path=output/(name+'.pt');torch.save(values,path);return {'path':str(path),'sha256':sha(path)}
    def structure(t):return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'device':str(t.device)}
    with torch.inference_mode():
        pairs=[(r['keys'].to(state['device']),r['values'].to(state['device']))for r in prefix]
        token=torch.tensor(preceding['prefix']['token'],device=state['device'],dtype=torch.int64)
        prompt=preceding['prefix']['prompt'];max_len=preceding['prefix']['max_static_len']
        for name in ['dynamic','static']:
            cache=DynamicCache(config=model.config) if name=='dynamic' else StaticCache(config=model.config,max_cache_len=max_len)
            if name=='dynamic':
                for i,(k,v) in enumerate(pairs):cache.update(k.clone(),v.clone(),i)
            else:
                cache.early_initialization(1,pairs[0][0].shape[1],pairs[0][0].shape[-1],pairs[0][0].dtype,state['device'])
                for layer,(k,v) in zip(cache.layers,pairs):
                    layer.keys[:,:,:prompt].copy_(k);layer.values[:,:,:prompt].copy_(v);layer.cumulative_length.fill_(prompt)
            assert all(torch.equal(l.keys[:,:,:prompt].cpu(),p['keys'])and torch.equal(l.values[:,:,:prompt].cpu(),p['values'])for l,p in zip(cache.layers,prefix))
            calls=[]
            def wrapped(q,k,v,*args,**kwargs):
                result=original(q,k,v,*args,**kwargs)
                if not calls:captured[name]=(q,k,v,args,kwargs,result)
                calls.append({'q':structure(q),'k':structure(k),'v':structure(v),
                              'kwargs':{key:structure(value) if isinstance(value,torch.Tensor)else value for key,value in kwargs.items()}})
                return result
            F.scaled_dot_product_attention=wrapped
            try:result=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            finally:F.scaled_dot_product_attention=original
            values={'tokens':result.logits[:,-1].argmax(-1,keepdim=True).cpu(),'logits':result.logits.cpu()}
            target=torch.load(preceding['arms'][name+'-eager']['output_raw']['path'],map_location='cpu',weights_only=True)
            q,k,v,args,kwargs,attn=captured[name]
            assert not args, 'Unexpected positional attention options; preserve the error, review before mapping.'
            raw[name]={'q':q.detach().cpu().clone(),'k':k.detach().cpu().clone(),'v':v.detach().cpu().clone(),
                       'output':attn.detach().cpu().clone(),
                       'tensor_options':{key:value.detach().cpu().clone()for key,value in kwargs.items()if isinstance(value,torch.Tensor)},
                       'options':{key:value for key,value in kwargs.items()if not isinstance(value,torch.Tensor)}}
            rows[name]={'attention_calls':len(calls),'first_interface':calls[0],
                'first_attention_raw':save(name+'-first-attention',raw[name]),'full_decode_output':save(name+'-decode',values),
                'matches_previous_unobserved_eager_values':all(torch.equal(values[key],target[key])for key in values),
                'comparison_to_previous_eager':compare(values,target,quality)}
        dq,dk,dv=raw['dynamic']['q'],raw['dynamic']['k'],raw['dynamic']['v']
        sq,sk,sv=raw['static']['q'],raw['static']['k'],raw['static']['v']
        logical={'q_equal':torch.equal(dq,sq),'dynamic_k_shape':list(dk.shape),'static_k_shape':list(sk.shape)}
        if sk.shape[1]%dk.shape[1]==0 and sk.shape[2]>=dk.shape[2]:
            groups=sk.shape[1]//dk.shape[1]
            logical.update(k_valid_equal=torch.equal(dk.repeat_interleave(groups,dim=1),sk[:,:,:dk.shape[2]]),
                v_valid_equal=torch.equal(dv.repeat_interleave(groups,dim=1),sv[:,:,:dv.shape[2]]),head_replication=groups)
        logical['first_attention_output_max_abs_error']=float((raw['dynamic']['output'].float()-raw['static']['output'].float()).abs().max())
        # Two separate attention-only profiles, not model timing or normal-path
        # native proof. Recreate actual tensor strides before calling the same ABI.
        for name in ['dynamic','static']:
            def restore(t,s):
                x=torch.empty_strided(s['shape'],s['stride'],dtype=t.dtype,device=state['device']);x.copy_(t);return x
            interface=rows[name]['first_interface']
            q,k,v=[restore(raw[name][key],interface[key])for key in ['q','k','v']]
            options={**raw[name]['options'],**{key:restore(t,interface['kwargs'][key])for key,t in raw[name]['tensor_options'].items()}}
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                answer=original(q,k,v,**options);torch.cuda.synchronize()
            path=output/(name+'-attention-trace.json');prof.export_chrome_trace(str(path))
            trace=read(path)['traceEvents']
            rows[name]['profile']={'path':str(path),'sha256':sha(path),
                'cpu_attention_ops':sorted({e.get('name','')for e in trace if e.get('cat')=='cpu_op' and ('attention' in e.get('name','')or 'softmax' in e.get('name',''))}),
                'kernels':sorted({e.get('name','')for e in trace if e.get('cat')=='kernel'}),
                'output_exact_to_full_forward_hook':torch.equal(answer.cpu(),raw[name]['output']),
                'raw_output':save(name+'-profile-answer',answer.cpu()),
                'scope':'Separately profiled one-attention invocation with restored value/shape/stride/options. Not full model native paths, clean timing or critical-path attribution.'}
        assert policies==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.get_default_dtype())
        report={'core':framework_source(),'adapter_sha256':sha(params['adapter']),'preceding_report_sha256':sha(params['preceding_report']),
            'rows':rows,'logical_inputs':logical,'math_policy_unchanged':True,
            'limits':'Python wrapper retains first-attention tensors; compare full decode to preceding no-wrapper controls. Interface pairs can differ in padding/mask/head layout/GQA option; a pair difference does not isolate a single factor. No new full32-token candidate quality or optimization acceptance.'}
        write_json(output/'attention-evidence.json',report)
        return {'valid':all(r['matches_previous_unobserved_eager_values'] for r in rows.values()),'report':str(output/'attention-evidence.json')}


def candidate(args,config,report):
    snapshot=args.parent.parent/'qwen-relation-feedback-replay-20261029/treatment/generation-snapshots/generation-713b51281927.json'
    saved,prompt,files=load(snapshot);original=files['inputs/previous-feedback.txt'].decode();feedback=json.loads(original)
    evidence=read(report);first=read(args.parent/'run/first-decode.json')
    facts={'scope':'New fixed numerical diagnostic facts, not a root-cause prescription or candidate acceptance.',
        'first_decode':{name:{'max_abs_logit_error':row['source_first_decode']['max_abs_error'],
                             'matches_prior_failed_first_decode':row['equals_prior_failed_first_decode']}for name,row in first['arms'].items()},
        'input_control':'All four first-decode modes use exact same real model, token and complete prefix KV; all prefixes unchanged. Dynamic eager matches source; the other three outputs are bitwise equal. Static eager failure means Graph replay and framework binding are not necessary for this observed difference.',
        'binding_signature_predicate':{k:first['binding_signature_predicate'][k]for k in ['greedy_matches','teacher_matches']},
        'binding_limit':'Actual greedy token stride [1,1], teacher slice [32,1]. Frozen signature admits first only; do not assume teacher executes the same replacement from missing route labels.',
        'attention_logical_inputs':evidence['logical_inputs'],
        'attention_interfaces':{name:{'interface':r['first_interface'],'full_decode_preserved':r['matches_previous_unobserved_eager_values'],
                                     'profile_cpu_ops':r['profile']['cpu_attention_ops'],'profile_output_exact':r['profile']['output_exact_to_full_forward_hook']}for name,r in evidence['rows'].items()},
        'limits':evidence['limits'], 'evidence':{'path':str(report),'sha256':sha(report)}}
    field='same_input_attention_diagnostic';assert field not in feedback
    end=original.rfind('}');new=original[:end]+',\n'+json.dumps(field)+':'+json.dumps(facts)+'\n'+original[end:]
    changed=json.loads(new);assert changed.pop(field)==facts and changed==feedback
    write_json(args.output/'candidate-protocol.json',{'snapshot':str(snapshot),'change_file':'inputs/previous-feedback.txt','added_field':field,
        'original_bytes':len(original.encode()),'new_bytes':len(new.encode()),'facts':facts,
        'scope':'One fresh candidate role, same original prompt/model/tools/core/other inputs. Only new observed facts appended; no developer candidate, manual optimization code or unique cause answer. Generation/schema only; no search/acceptance. Not an A/B superiority test or retry of a frozen arm.'})
    store=RunStore.create(args.output,'candidate-continuation',{'kind':'same-input-attention-facts'})
    with Runtime(project_agent_config(config),log_dir=store.run_dir) as runtime:
        directory,agent=ProjectAgents(runtime,store).replay_snapshot(snapshot,edit={'path':'inputs/previous-feedback.txt','old':original,'new':new},validator=validate_proposal)
    actual,ap,af=load(Path(agent['generation_snapshots'][0]));assert ap==prompt and actual['tool_model_settings']==saved['tool_model_settings']
    expected={**files,'inputs/previous-feedback.txt':new.encode()};assert af==expected
    return {'status':'completed','agent':agent,'candidate_sha256':sha(directory/'candidate.py'),'proposal':read(directory/'proposal.json'),
            'only_declared_feedback_field_changed':True,'scope':'Generated code not independently GPU validated in this batch.'}


def main(args):
    os.umask(0o077);config=load_config(args.config,[]);config.usage.mode='observe_only'
    prior=read(args.parent/'protocol.json');assert read(args.parent/'followup-completed.json')['phase']=='completed'
    assert read(args.parent/'terminal-audit.json')['all_raw_tensors_and_comparisons_recomputed']
    assert prior['core']==framework_source() and all(sha(p)==h for p,h in prior['frozen_identities'].items())
    template=ExecutionRequest.model_validate(read(args.parent/'run/request.json'))
    if args.preflight:
        print(json.dumps({'status':'passed','core':framework_source(),'preceding_report_sha256':sha(args.parent/'run/first-decode.json')}));return
    args.output.mkdir(parents=True,exist_ok=True);assert not(args.output/'protocol.json').exists()
    write_json(args.output/'protocol.json',{'at':now(),'parent':str(args.parent),'core':framework_source(),
        'frozen_identities':prior['frozen_identities'],'scope':'Two first-attention interface controls + two separately profiled attention calls, then one fresh factual candidate continuation only if controls reproduce prior eager values. No formal acceptance or production adoption.'})
    results={}
    try:
        write_json(args.output/'state.json',{'phase':'running','stage':'attention-probe','at':now()})
        run=LocalExecutor().run(template.model_copy(update={'entry':Path(__file__).resolve(),'function':'attention_worker',
            'parameters':{**template.parameters,'preceding_report':str(args.parent/'run/first-decode.json')},'output':args.output/'run','wait_for_devices':True}))
        write_json(args.output/'worker-result.json',run)
        valid=run['status']=='passed' and all((r.get('result')or{}).get('valid')for r in run['ranks'])
        results['attention-probe']={'status':'completed'if valid else'error','worker_status':run['status'],'result':str(args.output/'worker-result.json')}
        if valid:
            write_json(args.output/'state.json',{'phase':'running','stage':'candidate-continuation','at':now()})
            results['candidate-continuation']=candidate(args,config,args.output/'run/attention-evidence.json')
        else:results['candidate-continuation']={'status':'not_run','reason':'First-attention probe did not reproduce preceding eager controls.'}
        assert all(sha(p)==h for p,h in prior['frozen_identities'].items())
    except Exception:results['controller']={'status':'error','traceback':traceback.format_exc()}
    terminal={'phase':'completed'if all(v['status']=='completed'for v in results.values())else'completed_with_errors','finished_at':now(),'results':results}
    write_json(args.output/'followup-results.json',results);write_json(args.output/'state.json',terminal);write_json(args.output/'followup-completed.json',terminal)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['parent','output','config']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',action='store_true');main(p.parse_args())
