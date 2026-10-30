"""One CPU-only counterexample acquisition for unedited generated candidate bytes.

No model, CUDA initialization, process group, real collective, staging or publication.
Actual nested hook code is obtained through its live closures; collective completion
is an explicitly synthetic equal-peer SUM using real CPU tensors and torch Futures.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from kernel_optimizer.project.binding import BindingSession
from kernel_optimizer.project.discovery import _phase

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-candidate-20261030')
CANDIDATE = ROOT/'candidate/sandboxes/candidate-376a7946115e/candidate.py'
EXPECTED = '890659f73c7c88e704e21682afb584b548dd181447307ef18c4bcfefa54f150f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cells(function):
    return dict(zip(function.__code__.co_freevars, (c.cell_contents for c in function.__closure__ or ())))


def main():
    output = ROOT/'audit/cpu-contract'
    output.mkdir(parents=True, exist_ok=False)
    assert sha(CANDIDATE) == EXPECTED
    assert not torch.cuda.is_initialized()
    loader = importlib.util.spec_from_file_location('exact_generated_candidate', CANDIDATE)
    mod = importlib.util.module_from_spec(loader); loader.loader.exec_module(mod)
    ident = mod.PROJECT_CALL_ID
    record = {'candidate_sha256': EXPECTED, 'fixture_sha256': sha(__file__),
              'scope': 'CPU interface counterexamples, not GPU/NCCL/model quality or performance. Exact candidate code/constants unchanged; synthetic state and controlled collective outcomes explicitly injected.',
              'cases': {}}

    # Real BindingSession, exact candidate installation. The proposal removes the
    # sole previous bundle; no old optimized callable is installed in this session.
    proposal = json.loads(CANDIDATE.with_name('proposal.json').read_text())
    installed = json.loads((CANDIDATE.parent/'inputs/search.json').read_text())['installed_bundles']
    surviving = [b['id'] for b in installed if b['id'] not in proposal['replaces']]
    assert not surviving and not proposal['requires']
    calls = []
    returned = object()
    def original(state):
        calls.append(state); return returned
    adapter = SimpleNamespace(run=original, models=lambda state: {})
    binding = BindingSession(adapter, {}, [{'id':ident,'kind':'project','phase':'run','ranks':[0,1]}], world_size=2)
    token = _phase.set('run')
    try:
        with patch.object(torch.cuda, 'is_available', return_value=False):
            sites = binding.install(CANDIDATE, {})
        result = adapter.run({'sentinel':'CPU binding only'})
        assert sites == [ident] and result is returned and len(calls) == 1
        assert binding.references[ident] is original
        record['cases']['binding'] = {'surviving_old_bundles':surviving,
            'reference_is_original_adapter':True,'same_return_object':True,'original_calls':len(calls),
            'routes':binding.path_telemetry(), 'limit':'Unsupported-environment path only; actual revision filtering independently checked against transactions.stage source; no staged version or CUDA step.'}
    finally:
        binding.close(); _phase.reset(token)
    assert adapter.run is original

    with patch.object(torch.cuda, 'is_available', return_value=False):
        replacement = mod.build({'references':{ident:lambda state: None},'world_size':2})[ident]
    make_hook = cells(cells(replacement)['_setup'])['_make_hook']
    assert make_hook.__code__.co_filename == str(CANDIDATE)
    n = mod.SPLIT_ELEMENTS  # Do not lower the actual generated split threshold.

    class Bucket:
        def __init__(self): self.tensor = torch.full((n,), 4.0)
        def buffer(self): return self.tensor
        def index(self): return 0

    def hook_case(case):
        issued = []
        class Work:
            def __init__(self, tensor, ordinal): self.tensor, self.ordinal = tensor, ordinal
            def wait(self): return True
            def get_future(self):
                if case == 'both_enqueued_future_error' and self.ordinal == 1:
                    raise RuntimeError('injected get_future error after both enqueues')
                future = torch.futures.Future()
                if case == 'async_failure' and self.ordinal == 1:
                    future.set_exception(RuntimeError('injected asynchronous collective error'))
                else:
                    future.set_result([self.tensor])
                return future
        def all_reduce(tensor, group=None, async_op=False, **kwargs):
            ordinal = len(issued)+1
            issued.append({'ordinal':ordinal,'group':str(group),'elements':tensor.numel()})
            if case == 'second_enqueue_error' and ordinal == 2:
                raise RuntimeError('injected second enqueue error')
            # Explicit equal-peer SUM stub: each successful operation doubles
            # this local buffer. No collective/backend is initialized.
            tensor.mul_(2)
            return Work(tensor, ordinal)
        hook, state = make_hook('A','B')
        bucket = Bucket()
        with patch.object(mod.dist, 'all_reduce', side_effect=all_reduce):
            future = hook(None,bucket)
            exception = None
            try:
                value = future.wait()
            except Exception as exc:
                value = None; exception = type(exc).__name__+': '+str(exc)
        row = {'input_elements':n,'returned_elements':None if value is None else value.numel(),
               'returned_exception':exception, 'buffer_first':bucket.tensor[0].item(),
               'buffer_last':bucket.tensor[-1].item(),'schedule':state['schedule'],'issued':issued}
        if case == 'success':
            assert value.numel() == n and torch.equal(bucket.tensor, torch.full_like(bucket.tensor,4))
        elif case == 'second_enqueue_error':
            assert value.numel() == n//2 and exception is None
            assert torch.equal(bucket.tensor, torch.full_like(bucket.tensor,4))
        elif case == 'both_enqueued_future_error':
            assert value.numel() == n and len(issued) == 3
            assert torch.equal(bucket.tensor, torch.full_like(bucket.tensor,8))
        elif case == 'async_failure':
            assert exception is None and value.numel() == n and state['schedule'] == 'concurrent'
        return row

    for case in ['success','second_enqueue_error','both_enqueued_future_error','async_failure']:
        record['cases'][case] = hook_case(case)

    # Actual steady-state replacement branch under synthetic pre-established
    # hook state. This does not execute setup, a model, or a distributed backend.
    routes = []
    parameter = torch.nn.Parameter(torch.ones(1)); parameter.grad = torch.ones(1)
    raw = SimpleNamespace(named_parameters=lambda: [(mod.SENTINEL_NAMES[0],parameter)])
    model = object(); returned = object(); parent_calls = []
    def parent(state):
        parent_calls.append(1); plan['hook_state']['fired'] += 1
        return returned
    with patch.object(torch.cuda, 'is_available', return_value=False):
        replacement = mod.build({'references':{ident:parent},'world_size':2,
                                  'record_route':lambda call, route: routes.append(route)})[ident]
    plan = cells(replacement)['plan']
    plan.update(mode='hook',raw=raw,model=model,params=[parameter],param_ptrs=[parameter.data_ptr()],
                hook_state={'fired':0,'schedule':'concurrent'},
                sentinels={mod.SENTINEL_NAMES[0]:torch.zeros(1)})
    assert replacement({'raw_model':raw,'model':model}) is returned
    assert routes == ['hook_step_anomaly'] and plan['mode']=='hook' and plan['hook_state']['schedule']=='concurrent'
    record['cases']['sentinel_mismatch'] = {'routes':routes,'mode_after':plan['mode'],
        'schedule_after':plan['hook_state']['schedule'],'same_return_object':True,'parent_calls':len(parent_calls),
        'limit':'Synthetic initial plan/sentinel state; demonstrates mismatch handling, not historical model drift.'}
    assert sha(CANDIDATE) == EXPECTED and not torch.cuda.is_initialized()
    record.update(cuda_initialized=False, real_collectives=0, model_runs=0, candidate_bytes_unchanged=True)
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__': main()
