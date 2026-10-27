"""Observe official TP weights and prefill boundaries; no optimizer or quality edits."""
from pathlib import Path
import json
import sys


def run(context):
    import torch
    from safetensors import safe_open
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
    observations, hooks = {}, []

    def record(name):
        def hook(module, arguments, output):
            value = output[0] if isinstance(output, tuple) else output
            if hasattr(value, 'to_local'):
                value = value.to_local()
            observations[name] = value.detach().cpu().clone()
        return hook

    modules = [('embed_tokens', model.model.embed_tokens)]
    modules += [(f'layer.{i}', layer) for i, layer in enumerate(model.model.layers)]
    modules += [('norm', model.model.norm), ('lm_head', model.lm_head)]
    for name, module in modules:
        hooks.append(module.register_forward_hook(record(name)))
    with torch.inference_mode():
        output = model(input_ids=reference['input_ids'].to(context['device']), use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    first_logits = output.logits.detach().cpu()
    result = {'scope': 'Developer prefill-only intermediate/weight diagnostic. Hidden-state error is descriptive, not a source quality threshold or a complete 32-token test.',
              'rank': context['rank'], 'world_size': context['world_size'], 'actual_tp_plan': getattr(model, '_tp_plan', None),
              'source_first_logits': compare(first_logits, reference['logits'][:, :1], {'atol': .02, 'rtol': .02}),
              'source_first_token': compare(first_logits.argmax(-1), reference['tokens'][:, :1], {'atol': 0, 'rtol': 0})}
    original = Path(params['single_observations'])
    if context['world_size'] == 1:
        torch.save(observations, original)
        result['boundary_shapes'] = {name: list(value.shape) for name, value in observations.items()}
    else:
        saved = torch.load(original, map_location='cpu', weights_only=True, mmap=True)
        differences = []
        for name, actual in observations.items():
            expected = saved[name]
            row = {'name': name, 'actual_shape': list(actual.shape), 'reference_shape': list(expected.shape)}
            if actual.shape != expected.shape:
                row['status'] = 'local_shape_not_directly_comparable'
            else:
                delta = actual.float()-expected.float()
                row.update(status='compared', exact_equal=torch.equal(actual, expected),
                           max_abs_error=delta.abs().max().item(), rms_error=delta.square().mean().sqrt().item())
            differences.append(row)
        result['boundaries'] = differences
        index = assets/'model.safetensors.index.json'
        if index.exists():
            mapping = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(assets/'model.safetensors', framework='pt', device='cpu') as f:
                mapping = {name: 'model.safetensors' for name in f.keys()}
        weights = []
        for name, parameter in model.named_parameters():
            if name not in mapping:
                weights.append({'name': name, 'status': 'source_key_unavailable'})
                continue
            actual = parameter.to_local() if hasattr(parameter, 'to_local') else parameter
            actual = actual.detach().cpu()
            with safe_open(assets/mapping[name], framework='pt', device='cpu') as f:
                source = f.get_slice(name)
                shape = source.get_shape()
                row = {'name': name, 'local_shape': list(actual.shape), 'source_shape': shape, 'tested_layouts': []}
                if list(actual.shape) == shape:
                    row['tested_layouts'].append({'layout': 'replicated', 'exact': torch.equal(actual, source[:])})
                for axis, size in enumerate(shape):
                    expected_shape = shape[:]
                    expected_shape[axis] = size//context['world_size']
                    if size % context['world_size'] or list(actual.shape) != expected_shape:
                        continue
                    sl = [slice(None)]*len(shape)
                    block = expected_shape[axis]
                    sl[axis] = slice(context['rank']*block, (context['rank']+1)*block)
                    row['tested_layouts'].append({'layout': 'contiguous_rank_slice', 'axis': axis,
                                                 'exact': torch.equal(actual, source[tuple(sl)])})
                row['status'] = 'exact_layout_found' if any(x['exact'] for x in row['tested_layouts']) else 'unresolved'
                weights.append(row)
        result['weights'] = weights
        result['weight_scope'] = 'Tests full replication and shape-compatible rank-contiguous slices against checkpoint bytes; unresolved layout is not automatically a wrong weight.'
    write_json(Path(context['output'])/f'prefill-audit-rank-{context["rank"]}.json', result)
    return result
