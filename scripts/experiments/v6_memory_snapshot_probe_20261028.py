"""Isolated evaluator snapshot patch and low-allocation, optional GC observation."""
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import threading
import time
import types


def patched_source():
    from kernel_optimizer.project import evaluation
    source = Path(evaluation.__file__).read_text()
    tree = ast.parse(source)
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_worker_entry')
    invoke = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == 'invoke')
    lines = source.splitlines(keepends=True)
    original = ''.join(lines[invoke.lineno-1:invoke.end_lineno])
    new = original
    peak_pair = ('            lifecycle_allocated_peak = max(lifecycle_allocated_peak, torch.cuda.max_memory_allocated())\n'
                 '            lifecycle_reserved_peak = max(lifecycle_reserved_peak, torch.cuda.max_memory_reserved())')
    assert new.count(peak_pair) == 2
    new = new.replace(peak_pair,
        '            allocator = torch.cuda.memory_stats()\n'
        '            lifecycle_allocated_peak = max(lifecycle_allocated_peak, allocator.get("allocated_bytes.all.peak", 0))\n'
        '            lifecycle_reserved_peak = max(lifecycle_reserved_peak, allocator.get("reserved_bytes.all.peak", 0))')
    for call, key in [('max_memory_allocated', 'allocated_bytes.all.peak'),
                      ('max_memory_reserved', 'reserved_bytes.all.peak'),
                      ('memory_allocated', 'allocated_bytes.all.current'),
                      ('memory_reserved', 'reserved_bytes.all.current')]:
        old = 'torch.cuda.'+call+'()'; assert new.count(old) == 1
        new = new.replace(old, 'allocator.get("'+key+'", 0)')
    assert 'before_run' in new and new.count('torch.cuda.memory_stats()') == 2
    changed = ''.join(lines[:invoke.lineno-1])+new+''.join(lines[invoke.end_lineno:])
    compile(changed, 'isolated-evaluation-snapshot.py', 'exec')
    return source, changed, original, new


def evaluator(context):
    from kernel_optimizer.project import evaluation
    if context['parameters'].get('memory_read_policy', 'original') == 'original':
        return evaluation
    assert context['parameters']['memory_read_policy'] == 'snapshot'
    original, changed, before, after = patched_source()
    path = Path(context['output'])/f'evaluation-snapshot-rank-{context["rank"]}.py'
    path.write_text(changed)
    spec = importlib.util.spec_from_file_location('isolated_memory_evaluator_'+str(context['rank']), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from kernel_optimizer.runtime.execution import write_json
    write_json(path.with_suffix('.json'), {'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'patched_sha256': hashlib.sha256(changed.encode()).hexdigest(), 'before': before, 'after': after,
        'scope': 'Isolated experiment copy. Production evaluator is unchanged. Only adjacent allocator reads differ.'})
    return module


def source_record(module, context):
    path = Path(module.__file__)
    return {'policy': context['parameters'].get('memory_read_policy', 'original'),
            'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope': 'Actual evaluator module used by this worker; production core SHA alone does not identify the isolated copy.'}


def run(context):
    from kernel_optimizer.runtime.execution import write_json
    module = evaluator(context)
    if not context['parameters'].get('passive_gc', False):
        return {**module.worker_entry(context), 'evaluation_source': source_record(module, context)}
    events, timing = [], []
    phase, counts = ['prepare'], {}

    def callback(kind, info):
        row = {'phase': kind, 'at_ns': time.monotonic_ns(), 'thread': threading.get_ident(),
               'mode': phase[0], **info}
        if kind == 'start' and info['generation'] == 2:
            frame, stack = sys._getframe(1), []
            for _ in range(7):
                if frame is None:
                    break
                stack.append({'file': frame.f_code.co_filename, 'line': frame.f_lineno, 'function': frame.f_code.co_name})
                frame = frame.f_back
            row['trigger_stack'] = stack
            del frame
        events.append(row)

    old_loader = module.load_adapter
    def loader(path):
        adapter = old_loader(path)
        old_mode, old_metrics = adapter.set_mode, adapter.metrics
        def mode(state, name):
            phase[0] = name
            return old_mode(state, name)
        def metrics(state, result, elapsed):
            # Entry is after the evaluator timer. No before_run replacement,
            # file reads, CPU sampling or added synchronization inside its boundary.
            ended = time.monotonic_ns()
            name = phase[0]; index = counts.get(name, 0); counts[name] = index+1
            timing.append((name, index, ended, elapsed))
            return old_metrics(state, result, elapsed)
        adapter.set_mode, adapter.metrics = mode, metrics
        return adapter
    module.load_adapter = loader
    enabled = gc.isenabled(); gc.callbacks.append(callback)
    try:
        return {**module.worker_entry(context), 'evaluation_source': source_record(module, context)}
    finally:
        module.load_adapter = old_loader; gc.callbacks.remove(callback)
        assert gc.isenabled() == enabled
        write_json(Path(context['output'])/f'gc-passive-rank-{context["rank"]}.json', {
            'rank': context['rank'], 'events': events, 'timing': timing, 'gc_enabled': enabled,
            'clock': {'host': platform.node(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                      'monotonic': str(time.get_clock_info('monotonic')), 'perf_counter': str(time.get_clock_info('perf_counter'))},
            'scope': 'Passive GC callbacks, no CPU counter observer or before_run patch. Timing records at '
                     'metrics entry follow the timer by unmeasured Python call latency; inferred start/end '
                     'are approximate. Callback/record allocations can shift GC; compare with unobserved arms. '
                     'GC is never disabled or forced.'})


def control(context):
    """Real CUDA allocation lifetimes through both complete evaluator implementations."""
    import torch
    import torch.cuda.memory as cuda_memory
    module = evaluator(context)
    torch.cuda.empty_cache()
    old_loader, old_stats = module.load_adapter, torch.cuda.memory_stats
    old_memory_module_stats = cuda_memory.memory_stats
    calls = [0]
    def stats(*args, **kwargs):
        calls[0] += 1
        return old_stats(*args, **kwargs)
    def prepare(ctx, request):
        persistent = torch.zeros(131072, device=ctx['device'])
        temporary = torch.empty(2097152, device=ctx['device'])
        del temporary
        return {'n': 0, 'persistent': persistent, 'metric_owner': None, 'device': ctx['device']}
    def reset(state):
        state['n'] = 0; state['metric_owner'] = None
    def invoke(state):
        state['n'] += 1
        transient = torch.empty(524288, device=state['device'])
        output = torch.full((16384,), float(state['n']), device=state['device'])
        del transient
        return output
    def metrics(state, result, elapsed):
        # A retained post-timer allocation exercises the read after adapter.metrics.
        state['metric_owner'] = torch.empty(262144, device=state['device'])
        return {'latency_ms': elapsed}
    adapter = types.SimpleNamespace(prepare=prepare, reset=reset, run=invoke, metrics=metrics,
        observe=lambda state, value: {'output': value}, models=lambda state: {}, set_mode=lambda state, name: None)
    module.load_adapter = lambda path: adapter
    torch.cuda.memory_stats = stats
    cuda_memory.memory_stats = stats
    try:
        result = module.worker_entry(context)
        return {**result, 'evaluation_source': source_record(module, context), 'memory_stats_calls': calls[0], 'fixture': 'persistent, prepare transient, run transient, '
                'returned result and post-timer owner; legacy/invocation lifetime variants run separately'}
    finally:
        module.load_adapter = old_loader; torch.cuda.memory_stats = old_stats
        cuda_memory.memory_stats = old_memory_module_stats
