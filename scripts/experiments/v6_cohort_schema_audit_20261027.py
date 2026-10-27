"""Manual, source-reviewed observation mapping for the frozen cohort artifacts.

This repairs the independent audit recipe, not adapters or optimizer acceptance.
No source tensor, generated artifact, quality tolerance or frozen result is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json


def check(context):
    import torch
    params = context['parameters']
    request = params['request']
    controls = Path(params['controls'])
    refs = [torch.load(controls/f'reference-rank-{rank}.pt', map_location='cpu', weights_only=True, mmap=True)
            for rank in range(len(request['devices']))]
    if params['case'] == 'resnet':
        reference = torch.load(params['source_reference'], map_location='cpu', weights_only=True, mmap=True)
        # Code review: observe returns result.argmax(-1) as predicted_top1.
        expected = {'logits': reference['logits'], 'predicted_top1': reference['logits'].argmax(-1)}
        checks = [{'step': step, **compare({name: actual[name] for name in expected}, expected, request['quality'])}
                  for step, actual in enumerate(refs[0])]
        return {'valid': len(checks) == request['continuous_steps'] and all(x['passed'] for x in checks),
                'checks': checks, 'mapping': {'source.logits': 'observe.logits', 'source.logits.argmax(-1)': 'observe.predicted_top1'},
                'scope': 'Full 32x1000 logits and exact 32 predicted classes, no additional observations certified.'}

    # Code review: observe appends raw_model.parameters() to ordered lists.
    # The original source builds the same GPT2 model and uses this same iterator.
    source = Path(params['source'])
    sys.path.insert(0, str(source))
    import gpt2_ddp_training as original
    original.configure(0)
    config = json.loads((source/'workloads.json').read_text())['gpt2_ddp_training']
    model = original.make_model(config, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.0, weight_decay=0.0, foreach=False)
    inputs = [tokens for rank in range(request['workload']['world_size'])
              for tokens in original.make_inputs(config, request['workload'], rank, 'cuda:0')]
    names = [name for name, _ in model.named_parameters()]
    checks = []
    for step in range(request['continuous_steps']):
        loss = original.accumulate(model, optimizer, inputs)
        gradients = [parameter.grad.detach().cpu() for parameter in model.parameters()]
        optimizer.step()
        parameters = [parameter.detach().cpu() for parameter in model.parameters()]
        for rank, reference in enumerate(refs):
            actual = reference[step]
            assert isinstance(actual['gradients'], list) and isinstance(actual['parameters'], list)
            assert len(actual['gradients']) == len(actual['parameters']) == len(names)
            checks.append({'rank': rank, 'step': step,
                           'gradients': compare(actual['gradients'], gradients, request['quality']),
                           'parameters': compare(actual['parameters'], parameters, request['quality'])})
        mean_loss = sum(float(reference[step]['loss']) for reference in refs)/len(refs)
        checks.append({'step': step, 'global_loss': compare(mean_loss, float(loss.detach()), request['quality']),
                       'exact_replica_gradients': compare(refs[0][step]['gradients'], refs[1][step]['gradients'], {'atol': 0, 'rtol': 0})})
        del gradients, parameters
    return {'valid': all(value['passed'] for row in checks for key, value in row.items()
                         if key in {'gradients', 'parameters', 'global_loss', 'exact_replica_gradients'}),
            'checks': checks, 'parameter_order': names, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'mapping': 'Observed list index matches original raw_model.parameters() order; no sorting, name guessing or dropped tensors.',
            'scope': 'Original source serial exact global batch, two continuous optimizer updates, full gradients/parameters/loss and exact replica relation.'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resnet-reference', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for case, name in [('resnet', 'resnet-latency'), ('ddp', 'ddp-communication')]:
        run = args.cohort/name
        prepared = json.loads((run/'prepared.json').read_text())
        request = json.loads((run/'request.json').read_text())
        source = Path(prepared['controls_path']).parent/'reference-source'
        evidence = {'adapter': str(source/'opop_adapter.py'),
                    'adapter_sha256': hashlib.sha256((source/'opop_adapter.py').read_bytes()).hexdigest(),
                    'source_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
                    'previous_audit_preserved': str(args.cohort/(name+'-audit'))}
        execution = LocalExecutor().run(ExecutionRequest(
            entry=Path(__file__).resolve(), function='check', cwd=source, output=args.output/name,
            devices=request['devices'][:1] if case == 'ddp' else [], lease_dir=Path(request['lease_dir']),
            worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
            parameters={'case': case, 'request': request, 'controls': prepared['controls_path'],
                        'source': str(source), 'source_reference': str(args.resnet_reference)}))
        results[case] = {'evidence': evidence, 'run': execution,
                         'valid': execution['status'] == 'passed' and all(r['result']['valid'] for r in execution['ranks'])}
        write_json(args.output/'result.json', results)
        print(json.dumps({'case': case, 'valid': results[case]['valid']}), flush=True)


if __name__ == '__main__':
    main()
