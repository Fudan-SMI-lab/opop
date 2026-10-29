"""Bounded CPU check of frozen cache types against the real binding dispatcher.

No model, CUDA, candidate build, graph, numerical validation, or candidate edit.
The only adapted signature field is input_ids.device (cuda -> cpu).
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from kernel_optimizer.project.binding import BindingSession
from kernel_optimizer.project.discovery import phase, tensor_description
from kernel_optimizer.runtime.provenance import framework_source

CODE = '6ec4e97a4bd39a7d16c82109b49b841f0258c039d56de15eeb6f80696e4906fa'
CALL = '323bcbe006b343648b346af9'


def main(candidate, calls, output):
    assert not output.exists(), 'retain prior results; use a declared new directory'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == CODE
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('frozen_handle_candidate', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = next(x for x in json.loads(calls.read_text()) if x['id'] == CALL)
    original_signature = original['rank_specs']['0']['signatures'][0]
    adapted = copy.deepcopy(original)
    for signatures in [adapted['signatures'], adapted['rank_specs']['0']['signatures']]:
        for signature in signatures:
            assert signature['kwargs']['input_ids']['device'] == 'cuda'
            signature['kwargs']['input_ids']['device'] = 'cpu'

    class Parent(torch.nn.Module):
        def forward(self, input_ids, past_key_values, use_cache, logits_to_keep):
            return past_key_values.get_seq_length()

    parent = Parent()
    adapter = SimpleNamespace(models=lambda state: {'model': parent})
    session = BindingSession(adapter, {}, [adapted], bundle_id='cpu-interface-only')
    forward = parent.forward
    session.install(candidate, {}, diagnostic_passthrough=True)
    token = torch.zeros((1, 1), dtype=torch.int64)
    cache = DynamicCache()
    handle = module._CacheHandle(object(), 2)
    kwargs = dict(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    records = []
    try:
        with phase('decode'):
            for label, current_cache, current_token in [
                ('dynamic_greedy', cache, token),
                ('exact_generated_handle_greedy', handle, token),
                ('dynamic_teacher_stride', cache, torch.zeros((1, 32), dtype=torch.int64)[:, :1]),
            ]:
                arguments = {**kwargs, 'past_key_values': current_cache, 'input_ids': current_token}
                signature = {'args': [], 'kwargs': tensor_description(arguments)}
                row = {'case': label, 'signature': signature,
                       'matches_adapted_original': signature in adapted['rank_specs']['0']['signatures']}
                try:
                    row['returned'] = parent(**arguments)
                except AttributeError as error:
                    row['exception'] = {'type': type(error).__name__, 'message': str(error)}
                records.append(row)
    finally:
        session.close()
    assert records[0]['matches_adapted_original'] and records[0]['returned'] == 0
    assert not records[1]['matches_adapted_original']
    assert records[1]['exception']['message'] == "'_CacheHandle' object has no attribute 'get_seq_length'"
    assert not records[2]['matches_adapted_original'] and records[2]['returned'] == 0
    assert session.counts[CALL] == 1 and session.fallbacks[CALL] == 2
    assert parent.forward == forward and not torch.cuda.is_initialized()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps({
        'candidate_sha256': CODE, 'calls_sha256': hashlib.sha256(calls.read_bytes()).hexdigest(),
        'core': framework_source(), 'original_signature': original_signature,
        'scope': 'CPU signature/dispatch only: original input device changed to cpu; exact generated handle class, real BindingSession diagnostic_passthrough, synthetic parent calling get_seq_length. Candidate build/replacement, model, graphs and GPU not run. Not a historical per-call trace or numerical proof.',
        'records': records, 'counts': dict(session.counts), 'fallbacks': dict(session.fallbacks),
        'restored': True, 'cuda_initialized': torch.cuda.is_initialized(),
    }, indent=2) + '\n')
    print(output.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--calls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.candidate, args.calls, args.output)
