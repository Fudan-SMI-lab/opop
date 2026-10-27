"""Source-reviewed audits and dependency-only TP retry; never edits frozen results."""
from pathlib import Path
import json
import sys


def resnet_source(context):
    import torch
    from kernel_optimizer.project.evaluation import compare
    params = context['parameters']
    run = Path(params['run'])
    prepared = json.loads((run/'prepared.json').read_text())
    request = json.loads((run/'request.json').read_text())
    controls = Path(prepared['controls_path'])
    reference = torch.load(params['reference'], map_location='cpu', weights_only=True, mmap=True)
    actual = torch.load(controls/'reference-rank-0.pt', map_location='cpu', weights_only=True, mmap=True)
    # Reviewed actual observe(): pred_classes = result.argmax(-1).cpu().
    expected = {'logits': reference['logits'], 'pred_classes': reference['logits'].argmax(-1)}
    checks = [{'step': i, **compare({k: row[k] for k in expected}, expected, request['quality'])}
              for i, row in enumerate(actual)]
    result = {'numerical_source_fidelity': len(checks) == request['continuous_steps'] and all(c['passed'] for c in checks),
              'checks': checks, 'mapping': 'Source logits and logits.argmax(-1) map to observed logits/pred_classes; additional observations are not certified by this fixture.'}
    source = controls.parent/'reference-source'
    sys.path.insert(0, str(source))
    import resnet_inference as original
    assets = Path(request['assets']['model_config']).parent
    output = Path(context['output'])/'original-source'
    # Execute the actual unmodified entry (both source workloads), preserving
    # Python-local tensor lifetimes in main(), not an adapter reconstruction.
    sys.argv = [str(source/'resnet_inference.py'), '--assets', str(assets), '--output', str(output),
                '--warmup', '2', '--repeats', '5']
    original.main()
    baseline = json.loads((output/'result.json').read_text())
    offline = next(row for row in baseline['workloads'] if row['workload']['id'] == 'offline')
    result.update(original_entry_result=baseline, original_offline_peak=offline['summary']['peak_allocated_bytes'],
                  adapter_baseline_metrics=prepared['evaluation']['ranks'][0]['result']['metrics'],
                  scope='Independent source execution and full numerical comparison; source memory fidelity is assessed separately from frozen optimizer acceptance.')
    return result
