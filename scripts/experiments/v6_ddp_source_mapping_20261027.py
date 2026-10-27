"""Reviewed accumulated_loss mapping; independent original-source trajectory.

The generated adapter uses dictionary parameter names and accumulated_loss.
Do not change its bytes or substitute its anchored reference for this audit.
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
    params = context['parameters']; request = params['request']
    source = Path(params['source']); sys.path.insert(0, str(source))
    import gpt2_ddp_training as original
    original.configure(0)
    config = json.loads((source/'workloads.json').read_text())['gpt2_ddp_training']
    model = original.make_model(config, 0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=0., weight_decay=0., foreach=False)
    inputs = [tokens for rank in range(request['workload']['world_size'])
              for tokens in original.make_inputs(config, request['workload'], rank, 'cuda:0')]
    refs = [torch.load(Path(params['controls'])/f'reference-rank-{rank}.pt', map_location='cpu', weights_only=True, mmap=True)
            for rank in range(request['workload']['world_size'])]
    checks = []
    for step in range(request['continuous_steps']):
        loss = original.accumulate(model, optimizer, inputs)
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
        optimizer.step()
        weights = {name: p.detach().cpu() for name, p in model.named_parameters()}
        for rank, rows in enumerate(refs):
            actual = rows[step]
            assert set(actual['gradients']) == set(gradients) == set(actual['parameters'])
            checks.append({'rank': rank, 'step': step,
                           'gradients': compare(actual['gradients'], gradients, request['quality']),
                           'parameters': compare(actual['parameters'], weights, request['quality'])})
        mean_loss = sum(float(rows[step]['accumulated_loss']) for rows in refs)/len(refs)
        checks.append({'step': step, 'global_loss': compare(mean_loss, float(loss.detach()), request['quality']),
                       'exact_replica_gradients': compare(refs[0][step]['gradients'], refs[1][step]['gradients'], {'atol': 0, 'rtol': 0}),
                       'exact_replica_parameters': compare(refs[0][step]['parameters'], refs[1][step]['parameters'], {'atol': 0, 'rtol': 0})})
        del gradients, weights
    return {'valid': all(value['passed'] for row in checks for key,value in row.items() if key in
                        {'gradients','parameters','global_loss','exact_replica_gradients','exact_replica_parameters'}),
            'checks': checks, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'mapping': {'source_loss': 'mean_rank_accumulated_loss', 'gradients': 'same named_parameters dictionary', 'parameters': 'same named_parameters dictionary'},
            'scope': 'Original source exact global batch, full independent two-step serial trajectory; no anchoring to distributed states, no tolerance or frozen artifact edits. Extra generated checks are not substituted for independent comparisons.'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    request = json.loads((args.project/'request.json').read_text())
    prepared = json.loads((args.project/'prepared.json').read_text())
    source = Path(prepared['controls_path']).parent/'reference-source'
    adapter = source/'opop_adapter.py'
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == 'a4056384d718eebc668c897357ba3468306756aefa07fa77236ffb8feea3ec2d'
    result = LocalExecutor().run(ExecutionRequest(entry=Path(__file__).resolve(), function='check', cwd=source,
        output=args.output/'worker', devices=request['devices'][:1], lease_dir=Path(request['lease_dir']),
        worker_timeout_s=request['worker_timeout_s'], wait_for_devices=True,
        parameters={'request': request, 'source': str(source), 'controls': prepared['controls_path']}))
    write_json(args.output/'source-mapping-audit.json', {'adapter': str(adapter), 'adapter_sha256': prepared['adapter_sha256'],
        'previous_audit_preserved': str(args.project.parent/'ddp-communication-audit'), 'execution': result})
    print(json.dumps({'status': result['status'], 'rank_results': [r.get('result') for r in result['ranks']]}))


if __name__ == '__main__':
    main()
