"""Source-faithful single GPU / official TP=2 readiness, not an optimization result."""
from pathlib import Path
import gc
import hashlib
import json
import sys


def run(context):
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import Qwen3ForCausalLM
    from kernel_optimizer.project.evaluation import compare

    params = context['parameters']
    sys.path.insert(0, params['source'])
    import common
    import qwen_inference as original
    common.configure(int(str(context['device']).split(':')[-1]))
    source = Path(params['source'])
    config = json.loads((source/'workloads.json').read_text())['qwen_inference']
    workload = next(w for w in config['workloads'] if w['id'] == 'interactive')
    quality = {'atol': config['correctness']['logits_atol'], 'rtol': config['correctness']['logits_rtol']}
    reference = torch.load(params['reference'], map_location='cpu', weights_only=True, mmap=True)
    inputs = reference['input_ids'].to(context['device'])
    assert tuple(inputs.shape) == (workload['batch'], workload['prompt_tokens'])
    options = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    if context['world_size'] == 2:
        # Transformers initializes its mesh over the already-bound default group.
        options.update(tp_plan='auto', tp_size=2)
        model = Qwen3ForCausalLM.from_pretrained(params['assets'], **options).eval()
    else:
        model = Qwen3ForCausalLM.from_pretrained(params['assets'], **options).to(context['device']).eval()

    def barrier():
        if dist.is_initialized():
            dist.barrier()

    checks = []
    for step in range(2):
        barrier()
        tokens, logits, _ = original.infer(model, inputs, workload['new_tokens'], capture=True)
        checks.append({'step': step, 'full_logits': compare(logits.cpu(), reference['logits'], quality),
                       'exact_tokens': compare(tokens.cpu(), reference['tokens'], {'atol': 0, 'rtol': 0})})
        del tokens, logits
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(2):
        barrier()
        original.infer(model, inputs, workload['new_tokens'])
    samples = []
    for _ in range(5):
        barrier()
        torch.cuda.reset_peak_memory_stats()
        _, _, metrics = original.infer(model, inputs, workload['new_tokens'])
        metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        samples.append(metrics)
    barrier()
    diagnostic = common.profile_once(lambda: original.infer(model, inputs, workload['new_tokens']),
                                     Path(context['output'])/f'tp{context["world_size"]}-rank{context["rank"]}-trace.json')
    return {'valid': all(c[k]['passed'] for c in checks for k in ['full_logits', 'exact_tokens']),
            'scope': 'Developer readiness of the original full inference loop using official library TP; no autonomous optimization or formal speedup acceptance.',
            'workload': workload, 'quality': quality, 'checks': checks, 'samples': samples,
            'summary': common.summarize(samples), 'diagnostic': diagnostic,
            'tp_plan': getattr(model, '_tp_plan', None), 'world_size': context['world_size'],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'reference_sha256': hashlib.sha256(Path(params['reference']).read_bytes()).hexdigest(),
            'source_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                              for name in ['qwen_inference.py', 'common.py', 'workloads.json']},
            'interpretation': 'NCCL kernel duration includes waiting; it is neither link traffic nor exposed critical-path cost. Quality failure disqualifies this case at the unchanged tolerance.'}
