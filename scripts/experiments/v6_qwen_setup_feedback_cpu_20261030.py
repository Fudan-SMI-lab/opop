"""Bounded CPU controls for exact generated code; no model/CUDA execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

CODE = '4693d1472b012cfa786209e8ce2ede85a84ae443d7cc1c2212d2680ff42dcd38'


def main(candidate, output):
    assert not output.exists(), 'New output required; preserve prior attempts'
    output.mkdir(parents=True)
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == CODE
    assert not torch.cuda.is_initialized()
    spec = importlib.util.spec_from_file_location('frozen_candidate', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Same singleton slice operation as the generated B/E setup paths.
    teacher = torch.arange(32, dtype=torch.int64).reshape(1, 32)
    token = teacher[:, 2:3]
    contiguous = token.contiguous()
    cache = DynamicCache()
    cache.update(torch.zeros(1, 1, 258, 2), torch.zeros(1, 1, 258, 2), 0)
    calls, routes = [], []

    def parent(**kwargs):
        passed = kwargs['past_key_values']
        before = passed.get_seq_length()
        passed.update(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), 0)
        calls.append({'before': before, 'after': passed.get_seq_length(),
                      'same_cache': passed is cache, 'stride': list(kwargs['input_ids'].stride())})
        return SimpleNamespace(past_key_values=passed)

    runner = SimpleNamespace(fast_enabled=True, batch=1, home=torch.device('cpu'))
    replacement = module._DecodeReplacement([runner], module.DECODE_MODEL_CALL,
                                             parent, lambda _, route: routes.append(route))
    replacement(input_ids=contiguous, past_key_values=cache, use_cache=True, logits_to_keep=1)
    parent(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
    assert contiguous is token and tuple(contiguous.stride()) == (32, 1)
    assert routes == ['parent_fallback_signature']
    assert [(c['before'], c['after']) for c in calls] == [(258, 259), (259, 260)]

    # Test build's success branch without setup, graph, or a CUDA tensor.
    # Only a CPU subclass advertises a device; setup/verify are explicit stubs.
    class CPUAdvertisedCuda(torch.Tensor):
        @property
        def device(self):
            return torch.device('cuda', 0)

    inputs = torch.zeros((1, 256), dtype=torch.int64).as_subclass(CPUAdvertisedCuda)
    seen = []

    def setup(self):
        seen.append('setup_stub')
        self.decode_graphs = ['stub_graph']

    def verify(self, repl):
        seen.append('verify_stub')
        assert self.fast_enabled and repl._runner_box[0] is self

    context = {'targets': {module.DECODE_MODEL_CALL: {'kind': 'module'}},
               'references': {module.DECODE_MODEL_CALL: parent},
               'state': {'device': torch.device('cuda', 0), 'model': object(),
                         'new_tokens': 32, 'prompt_tokens': 256, 'batch': 1,
                         'inputs': inputs, 'logits_atol': .02, 'logits_rtol': .02}}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
         patch.object(torch.cuda, 'synchronize', return_value=None), \
         patch.object(module._FastDecodeRunner, 'setup', setup), \
         patch.object(module._FastDecodeRunner, 'verify_fast_path', verify):
        result = module.build(context)
    built = result[module.DECODE_MODEL_CALL]._runner_box[0]
    assert seen == ['setup_stub', 'verify_stub']
    assert not built.fast_enabled and built.disable_reason == 'not_attempted'
    assert built.decode_graphs == []
    assert not torch.cuda.is_initialized()
    report = {'candidate_sha256': CODE, 'scope': 'CPU interface/control-flow only; no model, graph, GPU or full quality',
              'singleton_slice': {'shape': list(token.shape), 'stride': list(token.stride()),
                                  'is_contiguous': token.is_contiguous(), 'contiguous_returns_same_object': contiguous is token},
              'exact_replacement_guard': {'routes': routes, 'parent_calls': calls,
                    'limit': 'CPU cache example plus exact guard; not a trace of historical GPU cache state'},
              'stubbed_successful_build': {'calls': seen, 'fast_enabled': built.fast_enabled,
                                           'disable_reason': built.disable_reason, 'graph_count': len(built.decode_graphs),
                    'limit': 'Independent latent success-branch defect; actual recorded GPU execution failed E before this branch'},
              'cuda_initialized': torch.cuda.is_initialized()}
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.candidate, args.output)
