"""Replay official TP projection modules on saved identical first-block operands."""
import hashlib
import json
from pathlib import Path
import sys

from scripts.experiments.v6_tp_block_audit_20261027 import difference


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metadata(tensor):
    local = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
    return {'global_shape': list(tensor.shape), 'global_stride': list(tensor.stride()),
            'placements': [str(p) for p in getattr(tensor, 'placements', ())],
            'local_shape': list(local.shape), 'local_stride': list(local.stride()),
            'dtype': str(local.dtype), 'sha256': digest(local)}


def run(context):
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import Qwen3ForCausalLM
    from kernel_optimizer.runtime.execution import write_json

    params = context['parameters']
    sys.path.insert(0, params['source'])
    import common
    common.configure(int(str(context['device']).split(':')[-1]))
    assert context['world_size'] == 2
    assets, device, rank = Path(params['assets']), context['device'], context['rank']
    single_path = Path(params['single_observations'])
    tp_path = Path(params['prior_tp'])/f'block-observations-rank-{rank}.pt'
    single = torch.load(single_path, map_location='cpu', weights_only=True, mmap=True)
    previous = torch.load(tp_path, map_location='cpu', weights_only=True, mmap=True)
    model = Qwen3ForCausalLM.from_pretrained(assets, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, tp_plan='auto', tp_size=2).eval()
    index = assets/'model.safetensors.index.json'
    mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
    rows, raw = [], {}
    with torch.inference_mode():
        for short in ['q_proj', 'k_proj', 'v_proj']:
            name = 'layer.0.self_attn.'+short
            module = getattr(model.model.layers[0].self_attn, short)
            key = 'model.layers.0.self_attn.'+short+'.weight'
            with safe_open(assets/(mapping[key] if mapping else 'model.safetensors'), framework='pt', device='cpu') as f:
                checkpoint = f.get_tensor(key)
            weight = module.weight.to_local() if hasattr(module.weight, 'to_local') else module.weight
            slice_cpu = checkpoint.chunk(2, dim=0)[rank]
            weight_match = difference(weight.cpu(), slice_cpu)
            x = single['tensors'][name+'.input'].to(device)
            prior_input = previous['tensors'][name+'.input'].to(device)
            old_full = single['tensors'][name+'.output'].to(device)
            old_tp = previous['tensors'][name+'.output'].to(device)
            row = {'name': name, 'checkpoint_key': key, 'input': metadata(x),
                'prior_tp_input': metadata(prior_input), 'weight': metadata(module.weight),
                'checkpoint_sha256': digest(checkpoint), 'checkpoint_row_slice': weight_match,
                'same_input': difference(prior_input, x), 'bias_present': module.bias is not None}
            # Never silently use an unverified contiguous-row layout as truth.
            if not weight_match.get('exact_equal') or not row['same_input'].get('exact_equal') or module.bias is not None:
                row['status'] = 'operand_mapping_unresolved'
                rows.append(row)
                continue
            full_weight = checkpoint.to(device)
            official = module(x)
            row['official_output_layout'] = metadata(official)
            official = official.to_local() if hasattr(official, 'to_local') else official
            values = {'official_module': official,
                'local_weight_bf16': F.linear(x, weight),
                'local_contiguous_bf16': F.linear(x.contiguous(), weight.contiguous()),
                'checkpoint_slice_bf16': F.linear(x, slice_cpu.to(device)),
                'full_bf16_slice': F.linear(x, full_weight).chunk(2, dim=-1)[rank],
                'full_fp32_cast_slice': F.linear(x.float(), full_weight.float()).to(x.dtype).chunk(2, dim=-1)[rank],
                'local_fp32_cast': F.linear(x.float(), weight.float()).to(x.dtype),
                'local_bf16_repeat': F.linear(x, weight)}
            reference = old_full.chunk(2, dim=-1)[rank]
            row.update(status='compared', against_recorded_single_slice={k: difference(v, reference) for k, v in values.items()},
                against_recorded_tp={k: difference(v, old_tp) for k, v in values.items()},
                against_official={k: difference(v, official) for k, v in values.items()},
                output_metadata={k: metadata(v) for k, v in values.items()})
            raw.update({name+'.'+k: v.detach().cpu() for k, v in values.items()})
            rows.append(row)
    torch.cuda.synchronize()
    artifact = Path(context['output'])/f'colwise-outputs-rank-{rank}.pt'
    torch.save(raw, artifact)
    result = {'rank': rank, 'world_size': 2, 'actual_tp_plan': getattr(model, '_tp_plan', None),
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'math_policy': {'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                        'float32_matmul_precision': torch.get_float32_matmul_precision()},
        'source_observations': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in [single_path, tp_path]],
        'projections': rows, 'artifact': {'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()},
        'scope': 'Same-input colwise projection diagnostics. Actual official module and checkpoint row slices '
                 'are compared only after exact local weight and input mapping. BF16 full-vs-shard arithmetic '
                 'and separate FP32 controls do not change the source contract, emulate all NCCL execution, '
                 'or establish full 32-token model quality. Prior full TP rejection remains.'}
    write_json(Path(context['output'])/f'colwise-result-rank-{rank}.json', result)
    return result
