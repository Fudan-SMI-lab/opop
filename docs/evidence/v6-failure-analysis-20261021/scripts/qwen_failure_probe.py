"""Developer diagnosis of frozen candidates; never publishes a revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("trial")
p.add_argument("output")
p.add_argument("--variant", default="original")
args = p.parse_args()
freeze = Path('/root/autodl-tmp/dev/opop-v6-freezes/187617ce3164')
sys.path[:0] = [str(freeze/'src'), str(freeze)]
root = Path('/root/opop-v6-stage3/final-e2e-187617c')/args.case
job = next((root/'jobs').glob(args.trial+'*-screen/run/request.json'))
params = json.loads(job.read_text())['parameters']
request = params['request']
os.environ['CUDA_VISIBLE_DEVICES'] = request['devices'][0]
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from kernel_optimizer.runtime.execution import DeviceLease
from kernel_optimizer.project.evaluation import load_adapter
from kernel_optimizer.project.binding import BindingSession

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
report = {'case': args.case, 'trial': args.trial, 'variant': args.variant,
          'framework_commit': '187617ce316403ea627c0d33392cc020bca2cb63',
          'diagnostic_only': True, 'torch': torch.__version__}

def save():
    (out/'result.json').write_text(json.dumps(report, indent=2))

def compare(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    err = (a-b).abs()
    bad = err > (0.02 + 0.02*b.abs())
    return {'max_abs': float(err.max()), 'bad': int(bad.sum()), 'elements': b.numel()}

def compare_sequence(actual, baseline):
    return {'tokens_equal': bool(torch.equal(actual['tokens'], baseline['tokens'])),
            'tokens': actual['tokens'].tolist(),
            'steps': [compare(actual['logits'][:,i], baseline['logits'][:,i])
                      for i in range(baseline['logits'].shape[1])]}

def cpu_result(r):
    return {k:r[k].detach().cpu().clone() for k in ('tokens','logits')}

with DeviceLease(Path(request['lease_dir']), request['devices'][0], wait=True):
    adapter = load_adapter(Path(params['adapter']))
    state = adapter.prepare({'device': torch.device('cuda:0'), 'rank':0, 'world_size':1}, request)
    adapter.set_mode(state, 'quality')
    adapter.reset(state)
    baseline = cpu_result(adapter.run(state))
    control = cpu_result(adapter.run(state))
    report['baseline_repeat'] = compare_sequence(control, baseline)
    report['baseline_tokens'] = baseline['tokens'].tolist()
    save()
    bundle = params['bundles'][0]
    file = Path(bundle['file'])
    report['candidate_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    if args.variant != 'original':
        code = file.read_text()
        if args.variant == 'ttft_eager_torch':
            code = code.replace('self.mode = "init"', 'self.mode = "eager"')
        elif args.variant == 'ttft_eager_triton':
            code = code.replace('self.mode = "init"', 'self.mode = "eager"').replace('self.use_tr = False', 'self.use_tr = True')
        elif args.variant in ('ttft_unpacked_torch','ttft_unpacked_triton'):
            code = code.replace('self.mode = "init"', 'self.mode = "eager"')
            if args.variant.endswith('triton'):
                code=code.replace('self.use_tr = False','self.use_tr = True')
            code = code.replace('qkv = torch.mm(x, self.w_qkv_t[i])',
                'qkv = torch.cat([torch.mm(x, w) for w in self.w_qkv_t[i].split([QH*D, KH*D, KH*D], dim=1)], dim=1)')
            code = code.replace('gu = torch.mm(x2, self.w_gu_t[i])',
                'gu = torch.cat([torch.mm(x2, w) for w in self.w_gu_t[i].split(I, dim=1)], dim=1)')
        elif args.variant == 'decode_no_graph':
            code = code.replace('EAGER_STEPS_BEFORE_CAPTURE = 3', 'EAGER_STEPS_BEFORE_CAPTURE = 1000000')
        elif args.variant == 'graph_clone_kv':
            code = code.replace('layer.__dict__.update(layer_dict)', 'layer.__dict__.update({k: v.clone() if torch.is_tensor(v) else v for k, v in layer_dict.items()})')
        elif args.variant == 'decode_live_window_no_graph':
            code = code.replace('EAGER_STEPS_BEFORE_CAPTURE = 3', 'EAGER_STEPS_BEFORE_CAPTURE = 1000000')
            code = code.replace('q, st["k_cache"][i], st["v_cache"][i], attn_mask=st["mask"], enable_gqa=True',
                                'q, st["k_cache"][i][:,:,:st["pos"]+1], st["v_cache"][i][:,:,:st["pos"]+1], attn_mask=None, enable_gqa=True')
        elif args.variant == 'graph_preserve_prefill_mask':
            code += '''
# Diagnostic only: freeze the eager no-mask decision for this validated full,
# unpadded prefill. This is NOT a general masking rule or framework change.
import inspect as _inspect
import transformers.masking_utils as _masking
_capture_original = _GraphPrefill._capture
def _diagnostic_capture(self, ids):
    original = _masking._ignore_causal_mask_sdpa
    signature = _inspect.signature(original)
    def fixed(*args, **kwargs):
        values = signature.bind(*args, **kwargs).arguments
        if (values.get('padding_mask') is None and values.get('q_length') == values.get('kv_length')
                and values.get('q_offset') == 0 and values.get('local_attention_size') is None):
            return True
        return original(*args, **kwargs)
    _masking._ignore_causal_mask_sdpa = fixed
    try: return _capture_original(self, ids)
    finally: _masking._ignore_causal_mask_sdpa = original
_GraphPrefill._capture = _diagnostic_capture
'''
        elif args.variant == 'static_live_attention_window':
            import torch.nn.functional as F
            original_sdpa = F.scaled_dot_product_attention
            ratio = state['model'].config.num_attention_heads // state['model'].config.num_key_value_heads
            def diagnostic_sdpa(q,k,v,*pargs,**kwargs):
                mask=kwargs.get('attn_mask')
                if q.shape[-2] == 1 and torch.is_tensor(mask):
                    allowed=mask if mask.dtype == torch.bool else mask == 0
                    length=int(allowed[0,0,0].sum())
                    prefix=torch.arange(allowed.shape[-1],device=allowed.device)<length
                    if not torch.equal(allowed[0,0,0],prefix):
                        raise ValueError('Diagnostic requires a single contiguous valid prefix')
                    k,v=k[:,:,:length],v[:,:,:length]
                    if k.shape[1] == q.shape[1] and ratio>1:
                        smallk,smallv=k[:,::ratio],v[:,::ratio]
                        if not torch.equal(smallk.repeat_interleave(ratio,dim=1),k):raise ValueError('non-repeated K')
                        if not torch.equal(smallv.repeat_interleave(ratio,dim=1),v):raise ValueError('non-repeated V')
                        k,v=smallk,smallv
                    kwargs={**kwargs,'attn_mask':None,'enable_gqa':True}
                return original_sdpa(q,k,v,*pargs,**kwargs)
            F.scaled_dot_product_attention=diagnostic_sdpa
        else:
            raise ValueError(args.variant)
        file = out/'diagnostic-candidate.py'
        file.write_text(code)
        report['diagnostic_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
    binding = BindingSession(adapter, state, bundle['specs'], bundle_id=bundle['id'])
    binding.install(file, bundle['parameters'])
    adapter.reset(state)
    report['candidate_invocations'] = []
    for repeat in range(2):
        actual = cpu_result(binding.invoke())
        row = compare_sequence(actual, baseline)
        row.update(invocation=repeat, counts=dict(binding.counts), fallbacks=dict(binding.fallbacks), telemetry=binding.path_telemetry())
        report['candidate_invocations'].append(row)
        save()
        print(json.dumps({'invocation':repeat,'max_abs':max(s['max_abs'] for s in row['steps']),
                          'first_bad_step':next((i for i,s in enumerate(row['steps']) if s['bad']),None),
                          'tokens_equal':row['tokens_equal'], 'counts':row['counts'], 'fallbacks':row['fallbacks']}),flush=True)
    binding.close()
    report['parent_after_install'] = compare_sequence(cpu_result(adapter.run(state)), baseline)
    report['completed'] = True
    save()
