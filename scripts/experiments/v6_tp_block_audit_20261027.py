"""First-block TP observations and same-input projection arithmetic diagnostics.

No hook changes the model output. FP32 operations below are separate developer
comparisons, never a change to source workload precision or its quality gate.
"""
from pathlib import Path
import hashlib
import json
import sys


def difference(actual, expected):
    import torch
    if actual.shape != expected.shape:
        return {'status': 'shape_mismatch', 'actual_shape': list(actual.shape),
                'reference_shape': list(expected.shape)}
    delta = actual.float() - expected.float()
    return {'status': 'compared', 'exact_equal': torch.equal(actual, expected),
            'elements': actual.numel(), 'max_abs_error': delta.abs().max().item(),
            'rms_error': delta.square().mean().sqrt().item()}


def projection_diagnostics(saved, assets, device):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open

    index = assets/'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    rows = []
    for short in ['self_attn.o_proj', 'mlp.down_proj']:
        name = 'layer.0.'+short
        key = 'model.layers.0.'+short+'.weight'
        with safe_open(assets/(mapping[key] if mapping else 'model.safetensors'), framework='pt', device='cpu') as f:
            weight = f.get_tensor(key)
        x = saved['tensors'][name+'.input'].to(device)
        w = weight.to(device)
        source_output = saved['tensors'][name+'.output'].to(device)
        # Reference operands are fixed single-GPU inputs and checkpoint bytes.
        # These two local GEMMs plus a local sum model arithmetic partitioning;
        # they are not a timing or execution emulation of NCCL/DTensor.
        assert x.shape[-1] == w.shape[-1] and w.shape[-1] % 2 == 0
        xs, ws = x.chunk(2, dim=-1), w.chunk(2, dim=-1)
        full_bf16 = F.linear(x, w)
        partial_bf16 = [F.linear(a.contiguous(), b.contiguous()) for a, b in zip(xs, ws)]
        partial_fp32 = [F.linear(a.float().contiguous(), b.float().contiguous()) for a, b in zip(xs, ws)]
        full_fp32 = F.linear(x.float(), w.float())
        values = {
            'full_bf16': full_bf16,
            'split_bf16_then_bf16_sum': partial_bf16[0]+partial_bf16[1],
            'split_bf16_then_fp32_sum_cast': (partial_bf16[0].float()+partial_bf16[1].float()).to(x.dtype),
            'split_fp32_sum_cast': (partial_fp32[0]+partial_fp32[1]).to(x.dtype),
            'full_fp32_cast': full_fp32.to(x.dtype),
        }
        rows.append({'name': name, 'input_shape': list(x.shape), 'weight_shape': list(w.shape),
            'checkpoint_key': key,
            'weight_sha256': hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'against_recorded_single_gpu_output': {k: difference(v, source_output) for k, v in values.items()},
            'against_full_bf16': {k: difference(v, full_bf16) for k, v in values.items()},
            'scope': 'Identical saved single-GPU operands; local contiguous halves and local sum, no NCCL. Descriptive arithmetic isolation, not an optimized model or a quality threshold.'})
    return rows


def run(context):
    import torch
    from transformers import Qwen3ForCausalLM
    from kernel_optimizer.project.evaluation import compare
    from kernel_optimizer.runtime.execution import write_json

    params = context['parameters']
    sys.path.insert(0, params['source'])
    import common
    common.configure(int(str(context['device']).split(':')[-1]))
    assets = Path(params['assets'])
    reference = torch.load(params['reference'], map_location='cpu', weights_only=True, mmap=True)
    options = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    if context['world_size'] == 2:
        options.update(tp_plan='auto', tp_size=2)
        model = Qwen3ForCausalLM.from_pretrained(assets, **options).eval()
    else:
        model = Qwen3ForCausalLM.from_pretrained(assets, **options).to(context['device']).eval()
    tensors, metadata, hooks = {}, {}, []

    def save(name, value):
        value = value[0] if isinstance(value, tuple) else value
        if not isinstance(value, torch.Tensor):
            metadata[name] = {'status': 'not_a_tensor', 'type': type(value).__name__}
            return
        row = {'global_shape': list(value.shape), 'dtype': str(value.dtype), 'stride': list(value.stride())}
        if hasattr(value, 'to_local'):
            row['placements'] = [str(p) for p in value.placements]
            value = value.to_local()
        row['local_shape'] = list(value.shape)
        metadata[name] = row
        tensors[name] = value.detach().cpu().clone()

    def pre_hook(name):
        def hook(module, arguments, kwargs):
            value = arguments[0] if arguments else kwargs['hidden_states']
            save(name+'.input', value)
        return hook

    def post_hook(name):
        def hook(module, arguments, output):
            save(name+'.output', output)
        return hook

    names = ['input_layernorm', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
             'self_attn.q_norm', 'self_attn.k_norm', 'self_attn.o_proj', 'self_attn',
             'post_attention_layernorm', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'mlp']
    modules = [('embed_tokens', model.model.embed_tokens), ('layer.0', model.model.layers[0])]
    modules += [('layer.0.'+name, model.model.layers[0].get_submodule(name)) for name in names]
    for name, module in modules:
        hooks.append(module.register_forward_pre_hook(pre_hook(name), with_kwargs=True))
        hooks.append(module.register_forward_hook(post_hook(name)))
    with torch.inference_mode():
        output = model(input_ids=reference['input_ids'].to(context['device']), use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    first_logits = output.logits.detach().cpu()
    result = {'scope': 'First-block prefill diagnostic; original BF16 model output unchanged. Neither intermediate error nor same-input arithmetic establishes full 32-token quality.',
        'rank': context['rank'], 'world_size': context['world_size'], 'actual_tp_plan': getattr(model, '_tp_plan', None),
        'source_first_logits': compare(first_logits, reference['logits'][:, :1], {'atol': .02, 'rtol': .02}),
        'source_first_token': compare(first_logits.argmax(-1), reference['tokens'][:, :1], {'atol': 0, 'rtol': 0}),
        'metadata': metadata}
    artifact = Path(context['output'])/f'block-observations-rank-{context["rank"]}.pt'
    torch.save({'tensors': tensors, 'metadata': metadata}, artifact)
    result['observations'] = {'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    if context['world_size'] == 1:
        torch.save({'tensors': tensors, 'metadata': metadata}, params['single_observations'])
    else:
        saved = torch.load(params['single_observations'], map_location='cpu', weights_only=True, mmap=True)
        boundaries = []
        for name, actual in tensors.items():
            expected = saved['tensors'][name]
            row = {'name': name, 'actual_shape': list(actual.shape), 'reference_shape': list(expected.shape)}
            if actual.shape == expected.shape:
                row.update(layout='same_shape', **difference(actual, expected))
            else:
                hypotheses = []
                for axis, size in enumerate(expected.shape):
                    shape = list(expected.shape); shape[axis] //= context['world_size']
                    if size % context['world_size'] == 0 and shape == list(actual.shape):
                        part = expected.chunk(context['world_size'], dim=axis)[context['rank']]
                        hypotheses.append({'layout': 'contiguous_rank_slice', 'axis': axis, **difference(actual, part)})
                row.update(status='layout_hypotheses' if hypotheses else 'not_directly_comparable', hypotheses=hypotheses)
            boundaries.append(row)
        result['boundaries'] = boundaries
        result['layout_scope'] = 'Same-shape comparisons and reported rank-contiguous layout hypotheses; mismatch alone is not evidence of wrong sharding or weights.'
        with torch.inference_mode():
            result['same_input_projection_diagnostics'] = projection_diagnostics(saved, assets, context['device'])
    write_json(Path(context['output'])/f'block-audit-rank-{context["rank"]}.json', result)
    return result
