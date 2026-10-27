"""Experiment-only GC/CPU observations around the unchanged evaluator invocation."""
import gc
import json
from pathlib import Path
import resource
import sys
import threading
import time


def cpu_observation():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    def counters(path):
        p = Path(path)
        return p.read_text() if p.exists() else None
    return {'process_user_s': usage.ru_utime, 'process_system_s': usage.ru_stime,
            'minor_faults': usage.ru_minflt, 'major_faults': usage.ru_majflt,
            'voluntary_switches': usage.ru_nvcsw, 'involuntary_switches': usage.ru_nivcsw,
            'max_rss_kib': usage.ru_maxrss, 'cgroup_cpu_stat': counters('/sys/fs/cgroup/cpu.stat'),
            'cgroup_cpu_pressure': counters('/sys/fs/cgroup/cpu.pressure'),
            'gc_count': gc.get_count(), 'gc_stats': gc.get_stats()}


def run(context):
    from kernel_optimizer.project import evaluation
    from kernel_optimizer.runtime.execution import write_json

    intervention = context['parameters'].get('gc_probe', 'unchanged')
    if intervention == 'unchanged':
        return evaluation.worker_entry(context)
    assert intervention in {'observe', 'disable_during_measure_phase'}
    original_enabled = gc.isenabled()
    events, invocations = [], []
    position = {'mode': 'prepare', 'index': -1, 'active': None}
    counts = {}

    def callback(phase, info):
        # No tensor operation, forced collection, GPU synchronization or file I/O.
        row = {'phase': phase, 'at_ns': time.monotonic_ns(), 'thread': threading.get_ident(),
               'mode': position['mode'], 'invocation': position['active'], **info}
        if phase == 'start' and info['generation'] == 2:
            frame, frames = sys._getframe(1), []
            for _ in range(5):
                if frame is None:
                    break
                frames.append({'file': frame.f_code.co_filename, 'line': frame.f_lineno,
                               'function': frame.f_code.co_name})
                frame = frame.f_back
            row['trigger_stack'] = frames
            del frame
        events.append(row)

    original_loader = evaluation.load_adapter

    def load(path):
        adapter = original_loader(path)
        before, metrics, mode = adapter.before_run, adapter.metrics, adapter.set_mode

        def set_mode(state, name):
            position['mode'] = name
            if intervention == 'disable_during_measure_phase':
                if name == 'measure':
                    gc.disable()
                elif original_enabled:
                    gc.enable()
            return mode(state, name)

        def before_run(state):
            result = before(state)
            name = position['mode']; index = counts.get(name, 0); counts[name] = index+1
            # Files/CPU counters are read outside the evaluator timer. Their
            # observer effect is tested against unchanged fresh endpoints.
            row = {'mode': name, 'index': index, 'before': cpu_observation(),
                   'gc_enabled': gc.isenabled(), 'start_ns': time.monotonic_ns()}
            invocations.append(row)
            position['active'] = len(invocations)-1
            return result

        def measured(state, result, elapsed_ms):
            ended = time.monotonic_ns()
            active = position['active']; position['active'] = None
            row = invocations[active]
            row.update(end_ns=ended, evaluator_elapsed_ms=elapsed_ms, after=cpu_observation())
            return metrics(state, result, elapsed_ms)

        adapter.before_run, adapter.metrics, adapter.set_mode = before_run, measured, set_mode
        return adapter

    gc.callbacks.append(callback)
    evaluation.load_adapter = load
    try:
        result = evaluation.worker_entry(context)
        return {**result, 'gc_probe': {'intervention': intervention,
                                     'artifact': str(Path(context['output'])/f'gc-rank-{context["rank"]}.json')}}
    finally:
        evaluation.load_adapter = original_loader
        gc.callbacks.remove(callback)
        gc.enable() if original_enabled else gc.disable()
        write_json(Path(context['output'])/f'gc-rank-{context["rank"]}.json', {
            'rank': context['rank'], 'intervention': intervention, 'original_gc_enabled': original_enabled,
            'gc_threshold': gc.get_threshold(), 'invocations': invocations, 'events': events,
            'scope': 'Developer diagnostics only. Timer brackets include pre-timer synchronization and small '
                     'boundary uncertainty. GC callbacks perturb Python allocation; CPU file reads are outside '
                     'the timer. Container throttling is shared, not process-specific. Disabling GC changes '
                     'runtime/lifetimes for the whole measure phase (warmup, reset and samples); it is not '
                     'an accepted optimization and costs must never be subtracted from existing scores.'})
