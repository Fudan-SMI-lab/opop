"""Read-only terminal audit; never runs a model, candidate, or collective."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def compact_coverage(c):
    # Preserve the full object on the server; public reports use only counts.
    return {k: v for k, v in c.items() if k not in ('covered_calls', 'expected_calls', 'eligible_calls')}


def audit(root):
    protocol = read(root / 'protocol.json')
    terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed'
    assert framework_source() == protocol['core']
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    search = ProjectSearch(load_config(root / 'private-config.yaml', []), root / 'ddp-search')
    source = ProjectSearch(search.config, Path(protocol['source']))
    request = search.request.portable()
    assert request == source.request.portable()
    assert source.revisions.current().id == protocol['parent']
    assert len(search.revisions.current().bundles) == 1
    paths = list((search.output / 'trials').glob('*/result.json'))
    assert len(paths) == 1
    trial = read(paths[0]); assert trial['decision']['accepted']
    assert len(trial['parameter_search']['observations']) == 1
    assert trial['bundle']['parameters'] == trial['proposal']['parameter_space'] == {}
    assert paired_decision(trial['pairs'], request['objective'], search.prepared['evaluation']) == trial['measurement']
    candidate = Path(trial['candidate_agent']['sandbox']) / 'candidate.py'
    expected_code = 'ccd90dc158788a164d6fda242572530e9cdc9ca50563af1a1d9d597f0ac12e0b'
    assert sha(candidate) == expected_code
    jobs = []; endpoints = []; metric_traces = []; native = []; transport = []; extra = []
    for folder in sorted((search.output / 'jobs').glob('*/run')):
        req = read(folder / 'request.json'); receipt = read(folder / 'receipt.json')
        assert req['parameters']['request'] == request
        assert req['devices'] == protocol['devices']
        assert req['parameters']['reference_dir'] == search.prepared['controls_path']
        assert sha(req['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert sha(req['entry']) == receipt['entry_sha256']
        manifest = read(Path(req['cwd']) / '.opop-revision.json')
        assert all(sha(Path(req['cwd']) / n) == h for n, h in manifest['files'].items())
        assert receipt['status'] == 'passed' and len(receipt['ranks']) == 2
        for b in req['parameters']['bundles']:
            assert sha(b['file']) == expected_code and b['parameters'] == {}
        job = {'job': folder.parent.name, 'request_sha256': sha(folder / 'request.json'),
               'receipt_sha256': sha(folder / 'receipt.json'), 'action': req['parameters']['action'],
               'entry_sha256': sha(req['entry']), 'source_manifest_sha256': sha(Path(req['cwd']) / '.opop-revision.json'),
               'bundles': [b['id'] for b in req['parameters']['bundles']], 'ranks': []}
        for rank in receipt['ranks']:
            k = rank['rank']; result = rank['result']; started = read(folder / f'rank-{k}-started.json')
            assert rank['status'] == 'passed' and result['valid'] and rank['framework_source'] == protocol['core']
            assert rank['binding']['torch_uuid'] == protocol['devices'][k]
            assert started['parameters_ref']['request_sha256'] == sha(folder / 'request.json')
            assert all(started['revision']['files'][n] == h for n, h in manifest['files'].items())
            log = (folder / f'rank-{k}.log').read_text()
            job['ranks'].append({'rank': k, 'result_sha256': sha(folder / f'rank-{k}.json'),
                                 'started_sha256': sha(folder / f'rank-{k}-started.json'),
                                 'log_sha256': sha(folder / f'rank-{k}.log'),
                                 'traceback_in_log': 'Traceback (most recent call last)' in log})
        jobs.append(job)
        if req['parameters']['action'] in ('evaluate', 'binding_ablation'):
            assert valid_run(receipt)
            ep = {'job': folder.parent.name, 'ranks': [], 'aggregates': {}}
            for rank in receipt['ranks']:
                k = rank['rank']; z = rank['result']
                assert z['math_policy_unchanged'] and len(z['samples']) == 5
                assert len(z['checks']) == 2 and all(c['passed'] and c['max_abs_error'] == 0 for c in z['checks'])
                ep['ranks'].append({'rank': k, **{f: z[f] for f in ['checks', 'coverage', 'samples', 'metrics', 'metric_samples', 'setup', 'diagnostics', 'bundle_hashes']}})
                for name, d in z['diagnostic_metrics'].items():
                    assert d['status'] == 'observed' and len(d['values']) == len(d['evidence']) == 3
                    assert statistics.median(d['values']) == z['metrics'][name]
                    for path, value in zip(d['evidence'], d['values'], strict=True):
                        trace, window = trace_window(read(path), 'opop.communication_metric_workload'); assert window
                        calc = normalize(trace, {'kernels': []}, k)
                        assert abs(calc[d['specification']['source']['field']] - value) < 1e-8
                        kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and 'nccl' in e.get('name', '').lower()]
                        metric_traces.append({'job': folder.parent.name, 'rank': k, 'metric': name,
                            'path': path, 'sha256': sha(path), 'window': window, 'value': value,
                            'kernel_names': dict(Counter(e['name'] for e in kernels)),
                            'logical_input_bytes': calc['logical_input_bytes'],
                            'collectives': dict(Counter(c['operation'] for c in calc['collectives'])),
                            'scope': d['scope']})
            for name in receipt['ranks'][0]['result']['metrics']:
                series = [r['result']['metric_samples'].get(name) or [x[name] for x in r['result']['samples']] for r in receipt['ranks']]
                values = [max(v) for v in zip(*series, strict=True)]; median = statistics.median(values)
                ep['aggregates'][name] = {'aligned_rank_max': values, 'median': median, 'MAD': statistics.median(abs(v - median) for v in values)}
            endpoints.append(ep)
            print(json.dumps({'endpoint': folder.parent.name, 'step': ep['aggregates']['step_ms']}), flush=True)
        normalized = folder / 'nsys-normalized.json'
        if normalized.exists():
            for row in read(normalized):
                con = sqlite3.connect('file:' + row['evidence'] + '?mode=ro', uri=True)
                counts = list(con.execute('select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value'))
                runtime = list(con.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId group by s.value'))
                con.close(); nccl = [x for x in counts if 'nccl' in x[0].lower()]
                assert sum(x[1] for x in counts) == row['cuda_kernels'] == len(row['kernels'])
                assert sum(x[1] for x in nccl) == row['nccl_kernels']
                assert abs(sum(x[2] for x in nccl) / 1e6 - row['nccl_kernel_sum_ms']) < 1e-8
                native.append({'job': folder.parent.name, 'rank': row['rank'], 'path': row['evidence'],
                    'sha256': sha(row['evidence']), 'normalized_sha256': sha(normalized), 'kernels': row['cuda_kernels'],
                    'nccl': [{'name': n, 'count': c, 'sum_ms': d / 1e6} for n, c, d in nccl], 'runtime_counts': dict(runtime),
                    'hardware_traffic_status': row['hardware_traffic_status']})
        for path in folder.glob('nccl*.log'):
            lines = path.read_text().splitlines()
            channel = [x for x in lines if ' via ' in x and 'Channel ' in x]
            transport.append({'path': str(path), 'sha256': sha(path), 'channel_lines': channel})
        for path in folder.glob('candidate-diagnostics-rank-*.json'):
            extra.append({'path': str(path), 'sha256': sha(path), 'data': read(path)})
    assert len(jobs) == 12 and len(endpoints) == 5 and len(metric_traces) == 30 and len(native) == 2
    private = root / 'terminal-private-history'; agents = []
    events = [json.loads(line) for line in (search.output / 'events.jsonl').read_text().splitlines()]
    for event in events:
        if event['type'] != 'PROJECT_AGENT_FINISHED': continue
        a = event['payload']; meta, prompt, files = load(Path(a['generation_snapshots'][0])); sandbox = Path(a['sandbox'])
        assert meta['framework_source'] == protocol['core']
        assert all((sandbox / name).read_bytes() == data for name, data in files.items())
        tools = read(private / (a['session'] + '-tools.json')); history = read(private / (a['session'] + '.json'))
        rows = [{'ordinal': i + 1, 'tool': t['tool'], 'input': t['input'], 'input_sha256': digest(json.dumps(t['input'], sort_keys=True)),
                 'output_sha256': digest(t['output']), 'output_bytes': len(t['output'].encode()), 'status': t['status'],
                 'exit': t['metadata'].get('exit'), 'truncated': t['metadata'].get('truncated'),
                 'output_path': t['metadata'].get('outputPath')} for i, t in enumerate(tools)]
        write_json(private / (a['session'] + '-audit-index.json'), rows)
        agents.append({'role': a['role'], 'session': a['session'], 'snapshot': a['generation_snapshots'][0],
                       'inputs': {n: {'bytes': len(d), 'sha256': digest(d)} for n, d in files.items()},
                       'input_bytes': sum(map(len, files.values())), 'prompt_sha256': digest(prompt),
                       'settings_sha256': digest(json.dumps(meta['tool_model_settings'], sort_keys=True)),
                       'history_sha256': sha(private / (a['session'] + '.json')), 'tools_sha256': sha(private / (a['session'] + '-tools.json')),
                       'tools': len(tools), 'assistant_messages': sum(m['data'].get('role') == 'assistant' for m in history['message']),
                       'usage': a.get('usage'), 'contract_repairs': a.get('contract_repairs')})
        if a['role'] == 'candidate':
            code = None; edits = []
            for i, t in enumerate(tools):
                inp = t['input']
                target = Path(inp.get('filePath', ''))
                if not target.is_absolute(): target = sandbox / target
                if target != candidate: continue
                if t['tool'] == 'write':
                    assert code is None; code = inp['content']; (private / 'candidate-generated-before.py').write_text(code)
                if t['tool'] == 'edit':
                    assert code.count(inp['oldString']) == 1
                    code = code.replace(inp['oldString'], inp['newString']); edits.append(i + 1)
            assert code.encode() == candidate.read_bytes()
            agents[-1]['code_edit_tools'] = edits
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    report = {'at': datetime.now(timezone.utc).isoformat(), 'terminal_sha256': sha(root / 'followup-completed.json'),
              'core': protocol['core'], 'frozen_files': len(protocol['frozen_identities']), 'all_frozen_unchanged': True,
              'source_current': source.revisions.current().id, 'current': search.revisions.current().id, 'active_bundles': list(search.revisions.current().bundles),
              'candidate_sha256': expected_code, 'proposal_sha256': sha(candidate.with_name('proposal.json')),
              'trial_sha256': sha(paths[0]), 'trial': trial['trial'], 'decision': trial['decision'], 'measurement': trial['measurement'],
              'selection': trial['selection'], 'jobs': jobs, 'endpoints': endpoints, 'metric_traces': metric_traces,
              'native': native, 'transport': transport, 'candidate_diagnostics': extra, 'agents': agents,
              'counts': {'jobs': len(jobs), 'clean_rank_samples': 50, 'full_quality_checks': 20, 'metric_traces': len(metric_traces), 'native_sqlites': 2},
              'independent': 'not_run_in_this_batch; new search acceptance requires separate declared complete-version validation',
              'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True),
              'gpu_workers': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], text=True)}
    write_json(root / 'terminal-numerical-audit.json', report)
    print(json.dumps({'audit_sha256': sha(root / 'terminal-numerical-audit.json'), 'counts': report['counts'],
                      'native': [{k: v for k, v in n.items() if k not in ['runtime_counts', 'path']} for n in native],
                      'agents': [{k: a[k] for k in ['role', 'tools', 'assistant_messages', 'input_bytes']} for a in agents]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
