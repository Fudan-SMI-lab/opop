"""Same-input first-decode arithmetic controls; diagnostic only, no candidate edits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def now():
    return datetime.now(timezone.utc).isoformat()


def first_decode(context):
    import torch
    from transformers.cache_utils import DynamicCache, StaticCache
    from kernel_optimizer.project.evaluation import load_adapter, compare, load_bundle_specs
    from kernel_optimizer.project.discovery import tensor_description
    params = context['parameters']; root = Path(context['output']); request = params['request']
    adapter = load_adapter(Path(params['adapter'])); state = adapter.prepare(context, request)
    model, inputs = state['model'], state['inputs']; quality = request['quality']
    policies = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
    reference = torch.load(params['independent_reference'], map_location='cpu', weights_only=True)
    source = {k: reference[k] for k in ['tokens', 'logits']}
    report = {'scope': 'One full original baseline source control, then four same-input first-decode controls. No binding/candidate implementation, full-candidate quality, optimization timing or acceptance. KV differences are descriptive, without invented thresholds.',
              'core': framework_source(), 'adapter_sha256': sha(params['adapter']), 'source_sha256': sha(params['independent_reference']),
              'sequence': ['dynamic-eager', 'dynamic-graph', 'static-eager', 'static-graph'], 'arms': {}}

    def save(name, value):
        path = root / (name + '.pt'); torch.save(value, path)
        return {'path': str(path), 'sha256': sha(path)}

    def values(cache, length):
        return [{'keys': l.keys[:, :, :length, :].detach().cpu().clone(),
                 'values': l.values[:, :, :length, :].detach().cpu().clone()} for l in cache.layers]

    def digest_values(rows):
        h = hashlib.sha256()
        for row in rows:
            for key in ['keys', 'values']:
                t = row[key].contiguous(); h.update(str((key, tuple(t.shape), t.dtype)).encode())
                h.update(t.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    with torch.inference_mode():
        adapter.reset(state)
        tokens, logits, _ = adapter._infer(model, inputs, state['new_tokens'], capture=True)
        baseline = {'tokens': tokens.cpu().clone(), 'logits': logits.cpu().clone()}
        report['full_baseline'] = {'comparison': compare(baseline, source, quality), 'raw': save('full-baseline', baseline)}
        assert report['full_baseline']['comparison']['passed'] and torch.equal(baseline['tokens'], source['tokens'])
        del tokens, logits
        adapter.reset(state)
        prefill = model(input_ids=inputs, use_cache=True, logits_to_keep=1)
        token = prefill.logits[:, -1].argmax(-1, keepdim=True)
        prompt = inputs.shape[1]; max_len = prompt + state['new_tokens'] - 1
        pairs = [(layer.keys.detach().clone(), layer.values.detach().clone()) for layer in prefill.past_key_values.layers]
        prefix = [{'keys': k.cpu().clone(), 'values': v.cpu().clone()} for k, v in pairs]
        expected = {'tokens': source['tokens'][:, 1:2], 'logits': source['logits'][:, 1:2]}
        report['prefix'] = {'raw': save('prefix-kv', prefix), 'value_sha256': digest_values(prefix),
                            'layers': len(prefix), 'prompt': prompt, 'max_static_len': max_len,
                            'token': token.cpu().tolist(), 'token_shape': list(token.shape), 'token_stride': list(token.stride()),
                            'prefill_logits': compare(prefill.logits.cpu(), source['logits'][:, :1], quality)}
        assert report['prefix']['prefill_logits']['passed']
        # Record the actual predicate inputs; do not infer historical dispatch
        # merely from absent route labels.
        saved_bundle = read(params['candidate_request'])['parameters']['bundles'][0]
        spec = next(s for s in load_bundle_specs(saved_bundle) if s['id'] in saved_bundle['expected_calls'])
        teacher = source['tokens'].to(inputs.device)[:, :1]
        def signature(ids):
            return {'args': [], 'kwargs': tensor_description({'input_ids': ids, 'past_key_values': prefill.past_key_values,
                    'use_cache': True, 'logits_to_keep': 1})}
        report['binding_signature_predicate'] = {'greedy': signature(token), 'teacher_slice': signature(teacher),
            'greedy_matches': signature(token) in spec['signatures'], 'teacher_matches': signature(teacher) in spec['signatures'],
            'spec_file_sha256': sha(saved_bundle['specs_path']),
            'scope': 'Actual source-generated first token and teacher tensor slice checked against frozen recorded signature. This is a code/predicate control, not a historical per-call event trace.'}
        old = torch.load(params['old_greedy'], map_location='cpu', weights_only=True)
        old_first = {'tokens': old['tokens'][:, 1:2], 'logits': old['logits'][:, 1:2]}
        eager_kv = None; outputs = {}
        for name in report['sequence']:
            write_json(root / 'probe-progress.json', {'at': now(), 'arm': name})
            try:
                is_static, is_graph = name.startswith('static'), name.endswith('graph')
                static_input = token.clone(); graph = None
                def new_dynamic():
                    c = DynamicCache(config=model.config)
                    for index, (key, value) in enumerate(pairs): c.update(key.clone(), value.clone(), index)
                    return c
                if is_static:
                    cache = StaticCache(config=model.config, max_cache_len=max_len)
                    cache.early_initialization(1, pairs[0][0].shape[1], pairs[0][0].shape[-1], pairs[0][0].dtype, inputs.device)
                    def restore():
                        for layer, (key, value) in zip(cache.layers, pairs):
                            layer.keys.zero_(); layer.values.zero_()
                            layer.keys[:, :, :prompt].copy_(key); layer.values[:, :, :prompt].copy_(value)
                            layer.cumulative_length.fill_(prompt)
                    restore()
                else:
                    cache = new_dynamic()
                before = values(cache, prompt)
                assert digest_values(before) == report['prefix']['value_sha256']
                if is_graph:
                    stream = torch.cuda.Stream(device=inputs.device)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            warm_cache = cache if is_static else new_dynamic()
                            model(input_ids=static_input, past_key_values=warm_cache, use_cache=True, logits_to_keep=1)
                    torch.cuda.current_stream().wait_stream(stream)
                    if is_static: restore()
                    # Dynamic warmups use separate objects and do not change cache.
                    assert digest_values(values(cache, prompt)) == report['prefix']['value_sha256']
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        result = model(input_ids=static_input, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    graph.replay()
                else:
                    result = model(input_ids=static_input, past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                result_logits = result.logits.detach().cpu().clone()
                result_values = {'tokens': result_logits[:, -1].argmax(-1, keepdim=True), 'logits': result_logits}
                after = values(cache, prompt + 1)
                prefix_unchanged = all(torch.equal(row[k][:, :, :prompt], base[k]) for row, base in zip(after, prefix) for k in row)
                descriptive = []
                if eager_kv is None: eager_kv = after
                for index, (row, base) in enumerate(zip(after, eager_kv)):
                    descriptive.append({'layer': index, **{k + '_max_abs_error': float((row[k].float()-base[k].float()).abs().max()) for k in row}})
                report['arms'][name] = {'status': 'completed', 'input_prefix_value_sha256': digest_values(before),
                    'input_token_exact': bool(torch.equal(static_input, token)), 'output_raw': save(name + '-output', result_values),
                    'updated_kv_raw': save(name + '-updated-kv', after), 'prefix_unchanged': prefix_unchanged,
                    'source_first_decode': compare(result_values, expected, quality),
                    'equals_prior_failed_first_decode': all(torch.equal(result_values[k], old_first[k]) for k in result_values),
                    'finite_logits': bool(torch.isfinite(result_logits).all()), 'exact_source_token': bool(torch.equal(result_values['tokens'], expected['tokens'])),
                    'kv_vs_dynamic_eager_descriptive': descriptive, 'seq_length_after': int(cache.get_seq_length()),
                    'execution': 'CUDAGraph capture then one explicit replay' if is_graph else 'direct parent eager',
                    'native_path_limit': 'No profiler/native kernel attribution or performance claim in this control.'}
                outputs[name] = result_values
                del graph, result, cache
            except Exception:
                report['arms'][name] = {'status': 'error', 'traceback': traceback.format_exc()}
            write_json(root / 'first-decode.json', report)
        report['cross_arm'] = {a + '__' + b: {'tokens_exact': bool(torch.equal(outputs[a]['tokens'], outputs[b]['tokens'])),
            'logits_exact': bool(torch.equal(outputs[a]['logits'], outputs[b]['logits'])),
            'max_abs_logit_error': float((outputs[a]['logits'].float()-outputs[b]['logits'].float()).abs().max())}
            for i, a in enumerate(outputs) for b in list(outputs)[i+1:]}
        report['math_policy_unchanged'] = policies == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_default_dtype())
        assert report['math_policy_unchanged']
        write_json(root / 'first-decode.json', report)
        return {'valid': all(a['status'] == 'completed' for a in report['arms'].values()), 'report': str(root / 'first-decode.json'),
                'scope': 'Diagnostic execution only, not full candidate quality or acceptance.'}


def main(args):
    prior = read(args.parent / 'protocol.json'); audit = read(args.parent / 'terminal-audit.json')
    assert audit['core'] == framework_source() and read(args.parent / 'followup-completed.json')['phase'] == 'completed'
    assert all(sha(p) == h for p, h in prior['frozen_identities'].items())
    source = read(args.parent / '0-B-observed-result.json')
    assert source['status'] == 'passed'
    request = ExecutionRequest.model_validate(prior['template'])
    parameters = {**request.parameters, 'independent_reference': prior['source_reference']['path'],
        'old_greedy': str(args.parent / 'fixed-validation/jobs/2-control-observed/run/observation-00-greedy.pt'),
        'candidate_request': str(args.parent / 'fixed-validation/jobs/2-control-observed/run/request.json')}
    if args.preflight:
        print(json.dumps({'status': 'passed', 'core': framework_source(), 'adapter_sha256': sha(parameters['adapter']),
                          'reference_sha256': sha(parameters['independent_reference']), 'sequence': ['dynamic-eager','dynamic-graph','static-eager','static-graph']}))
        return
    args.output.mkdir(parents=True, exist_ok=True); assert not (args.output / 'protocol.json').exists()
    write_json(args.output / 'protocol.json', {'at': now(), 'parent': str(args.parent), 'core': framework_source(),
        'parent_audit_sha256': sha(args.parent / 'terminal-audit.json'), 'frozen_identities': prior['frozen_identities'],
        'source_reference': prior['source_reference'], 'sequence': ['dynamic-eager','dynamic-graph','static-eager','static-graph'],
        'scope': 'Same real-model first-decode inputs, no binding/candidate code. One baseline full-output control; no formal acceptance, objective timing, profiler or new Agent generation.',
        'disk': disk_observation(Path('/root/autodl-tmp'))})
    write_json(args.output / 'state.json', {'phase': 'running', 'stage': 'first-decode', 'at': now()})
    results = {}
    try:
        run = LocalExecutor().run(request.model_copy(update={'entry': Path(__file__).resolve(), 'function': 'first_decode',
            'parameters': parameters, 'output': args.output / 'run', 'wait_for_devices': True}))
        write_json(args.output / 'worker-result.json', run)
        completed = run['status'] == 'passed' and all((r.get('result') or {}).get('valid') for r in run['ranks'])
        results['first-decode'] = {'status': 'completed' if completed else 'error',
            'worker_status': run['status'], 'result': str(args.output / 'worker-result.json'), 'report': str(args.output / 'run/first-decode.json')}
        assert all(sha(p) == h for p, h in prior['frozen_identities'].items())
        results['frozen-identities'] = {'status': 'passed'}
    except Exception:
        results['controller'] = {'status': 'error', 'traceback': traceback.format_exc()}
    terminal = {'phase': 'completed' if all(v['status'] in ['completed','passed'] for v in results.values()) else 'completed_with_errors',
        'finished_at': now(), 'results': results, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    write_json(args.output / 'followup-results.json', results); write_json(args.output / 'state.json', terminal)
    write_json(args.output / 'followup-completed.json', terminal)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ['parent', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--preflight', action='store_true'); main(p.parse_args())
