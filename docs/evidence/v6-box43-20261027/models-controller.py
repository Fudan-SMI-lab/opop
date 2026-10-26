"""One-off cloned-host checks. Reuses frozen adapters through the public CLI."""
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source

root = Path('/root/autodl-tmp/dev/box43-preflight-20261027')
old = Path('/root/autodl-tmp/dev/v6-work/evidence-followup-20261021/fresh')
config = '/root/autodl-tmp/dev/v6-work/configs/box43-v6.yaml'
result = {'scope': 'Frozen evaluator migration, fresh controls and full comparison to retained old reference tensors; no optimization or cross-host speedup claim',
          'framework_source': framework_source(), 'cases': {}}
for case, selectors in [('qwen-ttft', ['auto']), ('ddp-communication', ['auto']), ('resnet-latency', ['1'])]:
    target = root/'models'/case
    parent = old/case
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', config,
            'optimize-project', '--request', str(parent/'request.json'), '--reuse-evaluator', str(parent),
            '--devices', *selectors, '--output', str(target), '--prepare-only']
    row = result['cases'][case] = {'command': argv, 'selectors': selectors}
    try:
        with (root/(case+'-prepare.log')).open('w') as log:
            completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        row['returncode'] = completed.returncode
        prepared = json.loads((target/'prepared.json').read_text())
        row['preparation_status'] = prepared['status']
        row['controls'] = prepared['evaluation']
        row['adapter_sha256'] = prepared['adapter_sha256']
        row['placement'] = prepared['reuse']['placement']
        if completed.returncode != 0 or prepared['status'] != 'prepared':
            raise RuntimeError('preparation failed; see retained logs')
        request = ProjectRequest.model_validate_json((target/'request.json').read_bytes())
        old_prepared = json.loads((parent/'prepared.json').read_text())
        source = Path(prepared['controls_path']).parent/'reference-source'
        evaluation = LocalExecutor().run(ExecutionRequest(entry=source/'opop_evaluate.py', cwd=source,
            devices=selectors if selectors != ['auto'] else request.devices,
            distributed_backend=request.distributed_backend, output=target/'old-reference-comparison',
            lease_dir=request.lease_dir, worker_timeout_s=request.worker_timeout_s, wait_for_devices=True,
            parameters={'adapter': str(source/'opop_adapter.py'), 'request': request.portable(),
                        'action': 'evaluate', 'reference_dir': old_prepared['controls_path'], 'bundles': []}))
        row['old_reference_comparison'] = evaluation
        row['passed'] = valid_run(evaluation)
    except Exception:
        row['passed'] = False
        row['error'] = traceback.format_exc()
    write_json(root/'models-result.json', result)
    print(case, row['passed'], flush=True)
result['passed'] = all(row['passed'] for row in result['cases'].values())
write_json(root/'models-result.json', result)
