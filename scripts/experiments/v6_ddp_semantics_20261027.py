"""Fresh DDP original-entry validation after tested intake adoption; TP diagnosis."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.runtime.execution import ExecutionRequest, LocalExecutor, write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_autonomous_cohort_20261027 import autonomous, independent
from scripts.experiments.v6_box43_next_stage_20261027 import now


def disk_observation(path):
    probe = subprocess.run(['df', '-B1', '--output=size,used,avail,pcent,target', str(path)], capture_output=True, text=True)
    return {'at': now(), 'path': str(path), 'df_returncode': probe.returncode, 'df': probe.stdout, 'df_error': probe.stderr,
            'statvfs_free_bytes': shutil.disk_usage(path).free,
            'interpretation': 'df reflects the 150 GiB container quota; statvfs has reported host space for some nested paths and quota for the mount root. Compare both; plan against df.'}


def tp_block(args):
    previous = json.loads((args.tp_audit/'tp-retry-protocol.json').read_text())['new_request']
    framework = Path(__file__).resolve().parents[2]
    environment = {**previous['env'], 'PYTHONPATH': os.pathsep.join(map(str, [args.tp_audit/'tp-dependencies', framework/'src', framework]))}
    rows = []
    for count in [1, 2]:
        request = ExecutionRequest.model_validate(previous).model_copy(update={
            'entry': Path(__file__).with_name('v6_tp_block_audit_20261027.py'), 'function': 'run',
            'devices': [str(i) for i in range(count)], 'distributed_backend': 'nccl' if count == 2 else None,
            'cwd': framework, 'env': environment, 'output': args.output/f'tp-block-{count}',
            'parameters': {**previous['parameters'], 'single_observations': str(args.output/'single-block-observations.pt')}})
        write_json(args.output/f'tp-block-{count}-protocol.json', request.model_dump(mode='json'))
        result = LocalExecutor().run(request)
        rows.append({'world_size': count, 'execution': result})
        write_json(args.output/'tp-block-result.json', {'runs': rows})
        if result['status'] != 'passed':
            break
    return {'status': 'completed' if len(rows)==2 and all(r['execution']['status']=='passed' for r in rows) else 'completed_with_errors', 'runs': rows}


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ['config', 'output', 'tp-audit']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    project = Path(__file__).resolve().parents[2]/'examples/v6_validation'
    spec = {'name': 'ddp-communication', 'case': 'ddp', 'devices': ['auto'],
            'command': 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp --profile',
            'goal': "Reduce the source's diagnostic maximum-rank summed NCCL kernel duration for the accumulated training step, retaining its step-time constraint and full correctness checks."}
    write_json(args.output/'protocol.json', {'at': now(), 'framework_source': framework_source(),
        'sequence': ['fresh original-entry DDP three rounds', 'source audit, relational negative probe, independent full-version AB/BA if accepted', 'TP first-block observations and same-input arithmetic'],
        'specification': spec, 'disk': disk_observation(args.output),
        'scope': 'Adopted intake paragraph only; no old adapter, candidate, feedback or developer diagnosis supplied to autonomous DDP. No API/token/cost caps, no until-pass repeat. TP is separate developer diagnosis; no source precision or quality changes. GPU jobs serial; old results preserved.'})
    results = {}

    def stage(name, operation):
        try:
            results[name] = operation()
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
        write_json(args.output/'controller-result.json', results)
        write_json(args.output/'disk-latest.json', disk_observation(args.output))
        print(json.dumps({'at': now(), 'stage': name, 'status': results[name]['status']}), flush=True)

    try:
        stage('ddp-communication', lambda: autonomous(args, config, spec, project))
        stage('ddp-independent', lambda: independent(args, config, spec))
        stage('tp-block', lambda: tp_block(args))
    finally:
        terminal = {'at': now(), 'results': results, 'disk': disk_observation(args.output)}
        write_json(args.output/'controller-completed.json', terminal)
        write_json(args.output/'followup-completed.json', terminal)


if __name__ == '__main__':
    main()
