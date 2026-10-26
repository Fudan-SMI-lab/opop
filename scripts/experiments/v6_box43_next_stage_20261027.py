"""Recorded experiment recipe, not framework routing or model-specific optimization.

Run Qwen's fixed-version diagnostics before the independent raw-entry DDP cohort.
Private config remains outside the repository. All measurements use existing workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback

from kernel_optimizer.config import load_config
from kernel_optimizer.project.branches import fork_run
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.versioning import promote_version, stage_version
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source


def now():
    return datetime.now(timezone.utc).isoformat()


def environment(devices):
    fields = 'index,uuid,pstate,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used'
    result = subprocess.run(['nvidia-smi', '--id='+','.join(devices), '--query-gpu='+fields,
                             '--format=csv'], capture_output=True, text=True)
    return {'at': now(), 'gpu_returncode': result.returncode, 'gpu': result.stdout,
            'gpu_error': result.stderr, 'container_cpu_stat': Path('/sys/fs/cgroup/cpu.stat').read_text(),
            'container_memory_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text())}


def qwen(args, config):
    root = args.output/'qwen-fixed'
    old = args.work/'targeted-isolation-20261021'
    prior = json.loads((old/'qwen-repair-ab-result.json').read_text())
    chosen = next(r for r in prior['runs'] if r['arm'] == 'treatment')
    source = ProjectSearch(config, Path(chosen['output']))
    incoming = source.revisions.revision(chosen['decision']['revision'])
    previous = source.revisions.revision(incoming.parent_id)
    added = sorted(set(incoming.bundles)-set(previous.bundles))
    if not added or set(previous.bundles)-set(incoming.bundles):
        raise ValueError('recorded passthrough experiment needs the additive version pair')
    base = args.migration/'models/qwen-ttft'
    baseline = fork_run(base, root)
    search = ProjectSearch(config, root)
    parent = stage_version(search.revisions, source.output, previous.id)
    candidate = stage_version(search.revisions, source.output, incoming.id)
    versions = {'B': baseline, 'P': parent, 'T': candidate, 'C': candidate}
    order = ['B', 'P', 'T', 'C', 'C', 'T', 'P', 'B']
    protocol = {'scope': 'Developer fixed-candidate diagnostics; no generation or publication',
        'at': now(), 'framework_source': framework_source(), 'source_output': str(source.output),
        'source_candidate': incoming.id, 'source_parent': previous.id,
        'versions': {name: r.id for name, r in versions.items()}, 'passthrough_bundles': added,
        'blocks': 2, 'order_per_block': order, 'request': search.request.portable(),
        'controls': search.prepared['controls_path'], 'TMPDIR': os.environ.get('TMPDIR'),
        'diagnostics': ['python_runtime P/T/C', 'nsys C'],
        'interpretation': 'All fixed endpoints retained. B/C and P/C use existing paired rules for diagnosis only; P/T differences cannot be deducted from candidate scores. Intrusive traces are separate.'}
    write_json(root/'protocol.json', protocol)
    rows = []
    for block in range(2):
        for index, arm in enumerate(order):
            job = f'clean-{block}-{index}-{arm}'
            before = environment(search.request.devices)
            run = search.execute(versions[arm], job, 'binding_ablation' if arm == 'T' else 'evaluate',
                collector_options={'passthrough_bundles': added} if arm == 'T' else {})
            row = {'block': block, 'position': index, 'arm': arm, 'valid': valid_run(run),
                   'receipt': str(root/'jobs'/job/'run/receipt.json'), 'before': before,
                   'after': environment(search.request.devices)}
            rows.append(row)
            write_json(root/'progress.json', {'endpoints': rows})
            print(json.dumps({'stage': 'qwen', **{k: row[k] for k in ('block', 'position', 'arm', 'valid')}}), flush=True)
            if not row['valid']:
                write_json(root/'completed.json', {'status': 'endpoint_failed', 'at': now(), 'row': row})
                return {'status': 'endpoint_failed', 'output': str(root)}
    comparisons = {}
    for label, indices in {'original_vs_complete': [(0, 3), (7, 4)],
                           'parent_vs_complete': [(1, 3), (6, 4)]}.items():
        pairs = []
        for block in range(2):
            for left, right in indices:
                pairs.append({role: json.loads(Path(rows[block*8+i]['receipt']).read_text())
                              for role, i in [('parent', left), ('candidate', right)]})
        comparisons[label] = paired_decision(pairs, search.request.objective, search.prepared['evaluation'])
    summary = {}
    for arm in versions:
        endpoints = []
        for row in rows:
            if row['arm'] != arm:
                continue
            run = json.loads(Path(row['receipt']).read_text())
            result = run['ranks'][0]['result']
            samples = [s['ttft_ms'] for s in result['samples']]
            endpoints.append({'block': row['block'], 'position': row['position'],
                'ttft_ms': result['metrics']['ttft_ms'], 'samples': samples,
                'mad_ms': statistics.median(abs(v-statistics.median(samples)) for v in samples),
                'complete_generation_ms': result['metrics']['framework.elapsed_ms'],
                'checks': result['checks'], 'coverage': result['coverage'], 'receipt': row['receipt']})
        summary[arm] = endpoints
    result = {'status': 'clean_completed', 'at': now(), 'arms': summary, 'comparisons': comparisons,
              'scope': protocol['scope'], 'diagnostics': {}}
    write_json(root/'result.json', result)
    for arm in ('P', 'T', 'C'):
        run = search.execute(versions[arm], 'python-'+arm, 'python_runtime_profile',
            collector_options={'passthrough_bundles': added} if arm == 'T' else {})
        result['diagnostics']['python_'+arm] = run
        write_json(root/'result.json', result)
    native = search.collect_extra(candidate, ['nsys'])
    write_json(root/'native.json', native)
    result['diagnostics']['nsys'] = {'status': native['nsys']['status'], 'path': str(root/'native.json')}
    result['status'] = 'completed'
    write_json(root/'result.json', result)
    write_json(root/'completed.json', {'status': 'completed', 'at': now(), 'comparisons': comparisons})
    return {'status': 'completed', 'output': str(root), 'comparisons': comparisons}


def ddp(args, config):
    output = args.output/'ddp-raw'
    project = Path(__file__).resolve().parents[2]/'examples/v6_validation'
    goal = ("Reduce the source's diagnostic maximum-rank summed NCCL kernel duration for the "
            'accumulated training step, retaining its step-time constraint and full correctness checks.')
    command = 'torchrun --standalone --nproc_per_node=2 gpt2_ddp_training.py --output /tmp/original-ddp --profile'
    argv = [sys.executable, '-B', '-m', 'kernel_optimizer.cli', '--config', str(args.config),
        'optimize-project', '--project', str(project), '--command', command, '--goal', goal,
        '--devices', 'auto', '--output', str(output), '--rounds', '3']
    write_json(args.output/'ddp-protocol.json', {'scope': 'Three raw-entry autonomous rounds; no adapter, groups, candidate or developer diagnosis supplied',
        'at': now(), 'framework_source': framework_source(), 'argv': argv,
        'project_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(project.iterdir()) if p.is_file()},
        'independent_followup': 'If an optimized current version exists, compare the exact full version against its original reference in independent AB/BA; no publication.'})
    with (args.output/'ddp-controller.log').open('wb') as log:
        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    result = {'at': now(), 'returncode': process.returncode, 'output': str(output)}
    if process.returncode == 0 and (output/'prepared.json').exists():
        search = ProjectSearch(config, output)
        result.update(current=search.revisions.current().id, bundles=list(search.revisions.current().bundles),
                      trials=[str(p) for p in sorted((output/'trials').glob('*/result.json'))])
        if result['bundles']:
            validation = args.output/'ddp-independent'
            fork_run(output, validation, revision_id=search.prepared['revision'])
            independent = ProjectSearch(config, validation)
            report = promote_version(independent, output, publish=False)
            result['independent'] = {'decision': report['decision'], 'measurement': report['measurement'],
                                     'quality': report['quality'], 'output': str(validation)}
            if report['quality'] == 'pass':
                revision = independent.revisions.revision(report['revision'])
                native = independent.collect_extra(revision, ['nsys'])
                write_json(validation/'native.json', native)
    write_json(args.output/'ddp-result.json', result)
    print(json.dumps({'stage': 'ddp', **result}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work', type=Path, default=Path('/root/autodl-tmp/dev/v6-work'))
    parser.add_argument('--migration', type=Path, default=Path('/root/autodl-tmp/dev/box43-preflight-20261027'))
    parser.add_argument('--stage', choices=['all', 'qwen', 'ddp'], default='all')
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config, [])
    results = {}
    for name, operation in [('qwen', qwen), ('ddp', ddp)]:
        if args.stage not in ('all', name):
            continue
        try:
            results[name] = operation(args, config)
        except Exception:
            results[name] = {'status': 'error', 'traceback': traceback.format_exc()}
            print(json.dumps({'stage': name, **results[name]}), flush=True)
        write_json(args.output/'controller-result.json', results)
    write_json(args.output/'controller-completed.json', {'at': now(), 'results': results,
        'data_free_bytes': shutil.disk_usage(args.output).free})


if __name__ == '__main__':
    main()
