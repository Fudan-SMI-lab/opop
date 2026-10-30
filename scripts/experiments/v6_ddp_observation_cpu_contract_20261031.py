"""One CPU-only mechanics check of exact generated DDP observation bytes.

No model, tensor computation, process group, collective, CUDA initialization or
profiler execution. Synthetic event fixtures test reporting and restoration only.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

EXPECTED = "024441f7d80bddaa51590b3630fbcc3969ac31b05548546ad2089106a10b5e98"


def main(args):
    os.umask(0o077)
    assert hashlib.sha256(args.probe.read_bytes()).hexdigest() == EXPECTED
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.profiler
    from torch.nn.parallel import DistributedDataParallel as DDP
    assert not torch.cuda.is_initialized()
    assert not torch.distributed.is_initialized()
    targets = [(DDP, '__init__'), (DDP, 'no_sync'),
               (torch.optim.Optimizer, 'zero_grad'), (torch.optim.SGD, 'step'),
               (torch.distributed, 'broadcast'), (torch.profiler, 'profile')]
    originals = [getattr(owner, name) for owner, name in targets]

    def restore():
        for (owner, name), value in zip(targets, originals):
            setattr(owner, name, value)

    def identities():
        return [getattr(owner, name) is value for (owner, name), value in zip(targets, originals)]

    def load(label):
        spec = importlib.util.spec_from_file_location('observation_cpu_' + label, args.probe)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    m = load('analysis')
    reports = {}
    # One unique enclosing range is a positive control. Two overlapping inner
    # ranges under a common outer range cannot identify a unique inner submitter.
    def trace(path, ambiguous):
        events = [dict(ph='X', cat='user_annotation', name=m.WINDOW_LABEL, ts=0, dur=100),
                  dict(ph='X', cat='kernel', name='ncclSynthetic', ts=40, dur=4,
                       args={'correlation': 7, 'Seq': 1, 'Process Group Name': 'synthetic',
                             'Collective name': 'allreduce', 'dtype': 'Float', 'In msg nelems': 4}),
                  dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', ts=20, dur=1,
                       pid=1, tid=1, args={'correlation': 7}),
                  dict(ph='X', cat='cpu_op', name='c10d::outer', ts=10, dur=20, pid=1, tid=1)]
        if ambiguous:
            events += [dict(ph='X', cat='cpu_op', name='c10d::inner_a', ts=15, dur=10, pid=1, tid=1),
                       dict(ph='X', cat='cpu_op', name='c10d::inner_b', ts=16, dur=11, pid=1, tid=1)]
        path.write_text(json.dumps({'traceEvents': events}))
        return m._extract_trace(path)

    positive = trace(args.output/'unique.json', False)
    ambiguous = trace(args.output/'ambiguous.json', True)
    reports['enclosing_range'] = {
        'positive': positive['collectives'][0]['cpu_submission'],
        'ambiguous': ambiguous['collectives'][0]['cpu_submission'],
        'fixture_unique_innermost': False,
        'scope': 'Synthetic same-clock trace, not an observed NCCL execution.'}
    phases = {'invocations': [{'ordinal': n, 'phase': 'profile'} for n in (9, 10, 11)]}
    traces = [dict(positive, file=f'metric-rank-0-{i}.json') for i in (0, 2)]
    reports['missing_trace_index'] = m._join_traces(m._Observer(), traces, phases, {})
    reports['unknown_trace'] = m._join_traces(m._Observer(),
        [{'file': 'metric-rank-0-0.json', 'status': 'unknown', 'reason': 'no workload window'}], phases, {})

    # Mirror event absence only: a graph replay need not call Python zero_grad.
    # No actual graph/model is run. Three subsequent profile hook events retain
    # the previous setup ordinal if no new boundary marker is observed.
    obs = m._Observer()
    obs.mark_zero_grad()
    obs.record_broadcast()
    for _ in range(3):
        obs.profiler_depth += 1
        obs.record_hook(0)
        obs.profiler_depth -= 1
    reports['no_python_zero_grad'] = {
        'synthetic_profile_regions': 3, 'hook_ordinals': [r[0] for r in obs.hook_events],
        'phases': m._phases(obs, {'request': {'continuous_steps': 2, 'warmup': 2, 'repeats': 5}}),
        'scope': 'Conditional event sequence; not a GPU replay trace or historical per-call proof.'}
    obs2 = m._Observer()
    for _ in range(m.MAX_BROADCAST_EVENTS + 17):
        obs2.record_broadcast()
    reports['broadcast_bound'] = {'declared': m.MAX_BROADCAST_EVENTS,
        'retained_list_length': len(obs2.broadcasts), 'none_stubs': obs2.broadcasts.count(None)}

    entry = args.output/'entry.py'
    entry.write_text("def run(context):\n    context['calls'].append('original')\n"
                     "    if 'sentinel' in context: raise context['sentinel']\n"
                     "    return context['return_object']\n")
    for label in ('return', 'exception', 'partial_setup'):
        module = load(label)
        out = args.output/label
        out.mkdir()
        result_object = {'valid': False, 'nested': object()}
        context = {'rank': 0, 'world_size': 1, 'device': 'cpu', 'output': str(out),
                   'calls': [], 'return_object': result_object,
                   'parameters': {'request': {}, 'diagnostic_context': {
                       'original_entry': str(entry), 'original_function': 'run'}}}
        sentinel = RuntimeError('fixture original exception')
        if label == 'exception':
            context['sentinel'] = sentinel
        if label == 'partial_setup':
            # Non-class profiler API injects a setup failure after earlier
            # patches were installed. It is not a current installed API claim.
            torch.profiler.profile = object()
        returned = None
        error = None
        try:
            returned = module.run(context)
        except BaseException as exc:
            error = exc
        finally:
            restored = identities()
            restore()  # fixture cleanup, not a generated-code repair
        reports[label] = {'original_calls': len(context['calls']),
            'same_return_object': returned is result_object,
            'same_nested_object': isinstance(returned, dict) and returned.get('nested') is result_object['nested'],
            'valid_preserved': isinstance(returned, dict) and returned.get('valid') is False,
            'same_exception_object': error is sentinel,
            'error_type': type(error).__name__ if error else None,
            'patches_restored_before_fixture_cleanup': dict(zip([name for _, name in targets], restored))}
    assert all(identities())
    assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
    payload = {'probe_sha256': EXPECTED,
               'scope': 'One exact-byte CPU mechanics fixture; synthetic events and Python entry only. No CUDA/model/tensor compute/real collective/profiler. Findings are not historical GPU failures.',
               'reports': reports, 'fixture_restored_all': True,
               'cuda_initialized': False, 'process_group_initialized': False}
    (args.output/'result.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
