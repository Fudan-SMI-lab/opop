"""Locate first prefill/first decode numerical divergence, using frozen source."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

root=Path('/root/autodl-tmp/dev/v6-work/failure-analysis-20261021')
while not (root/'ddp-ablation/completed.json').exists():
    time.sleep(2)
freeze=Path('/root/autodl-tmp/dev/opop-v6-freezes/187617ce3164')
sys.path[:0]=[str(freeze/'src'),str(freeze)]
base=Path('/root/opop-v6-stage3/final-e2e-187617c/qwen-ttft')
job=next((base/'jobs').glob('trial-af78*-screen/run/request.json'))
params=json.loads(job.read_text())['parameters'];request=params['request']
os.environ['CUDA_VISIBLE_DEVICES']=request['devices'][0]
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
import torch.nn.functional as F
from kernel_optimizer.runtime.execution import DeviceLease
from kernel_optimizer.project.evaluation import load_adapter
from transformers.cache_utils import DynamicCache,StaticCache

out=root/'qwen-math';out.mkdir(exist_ok=True)
report={'scope':'Developer diagnosis; instrumented tensors, no timing acceptance', 'framework_commit':'187617ce316403ea627c0d33392cc020bca2cb63'}

def save(): (out/'result.json').write_text(json.dumps(report,indent=2))
def cmp(a,b):
    if isinstance(a,(tuple,list)): return [cmp(x,y) for x,y in zip(a,b)]
    a,b=a.detach().float().cpu(),b.detach().float().cpu()
    if a.shape != b.shape:
        return {'shape':list(a.shape),'reference_shape':list(b.shape),'shape_changed':True}
    err=(a-b).abs();bad=err>(.02+.02*b.abs())
    return {'shape':list(a.shape),'max_abs':float(err.max()),'different':int((a!=b).sum()),'bad':int(bad.sum()),'elements':a.numel()}
def clone(x):
    if isinstance(x,(tuple,list)): return tuple(clone(y) for y in x)
    if torch.is_tensor(x): return x.detach().clone()
    return x
def cpu(x):
    if isinstance(x,(tuple,list)):return tuple(cpu(y) for y in x)
    return x.detach().cpu().clone() if torch.is_tensor(x) else x
def load(path):
    spec=importlib.util.spec_from_file_location('diagnostic_candidate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

with DeviceLease(Path(request['lease_dir']),request['devices'][0],wait=True), torch.inference_mode():
    adapter=load_adapter(Path(params['adapter']))
    state=adapter.prepare({'device':torch.device('cuda:0'),'rank':0,'world_size':1},request)
    model=state['model'];ids=state['inputs']
    records={};handles=[]
    names=['model.embed_tokens','model.rotary_emb']
    names+=['model.layers.0.'+x for x in ['input_layernorm','self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.q_norm','self_attn.k_norm','self_attn.o_proj','self_attn','post_attention_layernorm','mlp.gate_proj','mlp.up_proj','mlp.down_proj','mlp']]
    names += ['model.layers.'+str(i) for i in range(36)]+['model.norm','lm_head']
    for name in names:
        def hook(mod, inputs, result, name=name):
            if name.endswith('self_attn'): result=result[0]
            records[name]=clone(result)
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    original_sdpa=F.scaled_dot_product_attention
    sdpa_calls=[]
    def sdpa(q,k,v,*args,**kwargs):
        n=len(sdpa_calls)
        if n%36==0:
            records['sdpa.q']=clone(q);records['sdpa.k']=clone(k);records['sdpa.v']=clone(v)
            mask=kwargs.get('attn_mask')
            if torch.is_tensor(mask):records['sdpa.mask']=clone(mask)
        result=original_sdpa(q,k,v,*args,**kwargs)
        sdpa_calls.append({'qshape':list(q.shape),'kshape':list(k.shape),'qstride':list(q.stride()),'kstride':list(k.stride()),
                           'mask':str(getattr(kwargs.get('attn_mask'),'dtype',None)),
                           'enable_gqa':kwargs.get('enable_gqa',False),'is_causal':kwargs.get('is_causal',False)})
        if n%36==0:records['sdpa.out']=clone(result)
        return result
    F.scaled_dot_product_attention=sdpa
    records={};sdpa_calls=[]
    baseline=model(input_ids=ids,use_cache=True,logits_to_keep=1)
    baseline_logits=cpu(baseline.logits);baseline_records={k:cpu(v) for k,v in records.items()}
    report['prefill_baseline_sdpa']=sdpa_calls[0]
    candidate=load(Path(params['bundles'][0]['file']))
    engine=candidate._GraphPrefill(model,model.forward,lambda *a:None,candidate.PREFILL_CALL)
    records={};sdpa_calls=[]
    result=engine(input_ids=ids,use_cache=True,logits_to_keep=1)
    report['prefill_graph_logits']=cmp(result.logits,baseline_logits)
    report['prefill_graph_modules']={k:cmp(v,baseline_records[k]) for k,v in records.items() if k in baseline_records}
    report['prefill_graph_sdpa_calls']=sdpa_calls
    # Clone output/KV comparisons before another replay overwrites static buffers.
    report['prefill_graph_cache']=[{'layer':i,'k':cmp(a.keys,b.keys),'v':cmp(a.values,b.values)} for i,(a,b) in enumerate(zip(result.past_key_values.layers,baseline.past_key_values.layers))]
    save();print('prefill graph',report['prefill_graph_logits'],flush=True)
    # Decode from identical prefill tensors and the identical teacher token.
    token=baseline.logits[:,-1].argmax(-1,keepdim=True)
    def dynamic():
        c=DynamicCache(config=model.config)
        for a,b in zip(c.layers,baseline.past_key_values.layers):
            a.keys=b.keys.clone();a.values=b.values.clone();a.is_initialized=True;a.dtype=a.keys.dtype;a.device=a.keys.device
        return c
    records={};sdpa_calls=[]
    dynamic_out=model(input_ids=token,past_key_values=dynamic(),use_cache=True,logits_to_keep=1)
    dynamic_logits=cpu(dynamic_out.logits);dynamic_records={k:cpu(v) for k,v in records.items()}
    report['decode_dynamic_sdpa']=sdpa_calls[0]
    cache=StaticCache(config=model.config,max_cache_len=296)
    cache.early_initialization(batch_size=1,num_heads=8,head_dim=128,dtype=torch.bfloat16,device=ids.device)
    for a,b in zip(cache.layers,baseline.past_key_values.layers):
        a.keys[:,:,:256].copy_(b.keys);a.values[:,:,:256].copy_(b.values);a.cumulative_length.fill_(256)
    records={};sdpa_calls=[]
    static_out=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
    report['decode_static_logits']=cmp(static_out.logits,dynamic_logits)
    report['decode_static_modules']={k:cmp(v,dynamic_records[k]) for k,v in records.items() if k in dynamic_records and k not in ('sdpa.k','sdpa.v','sdpa.mask')}
    report['decode_static_sdpa']=sdpa_calls[0]
    save();print('static decode',report['decode_static_logits'],flush=True)
    # Primitive SDPA: exact same query/key/value; pad only K/V + mask.
    q=dynamic_records['sdpa.q'].to(ids.device);k=dynamic_records['sdpa.k'].to(ids.device);v=dynamic_records['sdpa.v'].to(ids.device)
    kp=F.pad(k,(0,0,0,39));vp=F.pad(v,(0,0,0,39))
    mask=torch.zeros((1,1,1,296),dtype=q.dtype,device=ids.device);mask[:,:,:,257:]=torch.finfo(q.dtype).min
    from torch.nn.attention import sdpa_kernel,SDPBackend
    report['isolated_sdpa']=[]
    for backend in [None,SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH,SDPBackend.CUDNN_ATTENTION]:
        import contextlib
        row={'backend':str(backend)}
        try:
            with sdpa_kernel(backend) if backend is not None else contextlib.nullcontext():
                r=original_sdpa(q,k,v,enable_gqa=True)
                p=original_sdpa(q,kp,vp,attn_mask=mask,enable_gqa=True)
            row['padded_vs_dynamic']=cmp(p,r)
        except Exception as exc:row['error']=str(exc)[:500]
        report['isolated_sdpa'].append(row)
    for h in handles:h.remove()
    F.scaled_dot_product_attention=original_sdpa
    report['completed']=True;save()
