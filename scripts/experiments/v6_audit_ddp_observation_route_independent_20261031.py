"""Read-only audit of the declared exact-version DDP independent closure."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.communication import normalize, trace_window
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_ddp_observation_route_independent_20261031 import CODE, ACCEPTED, BASELINE, SOURCE_CURRENT
from scripts.experiments.v6_audit_ddp_observation_route_fixed_20261031 import union_duration


def audit(root):
    assert not (root/'terminal-audit.json').exists()
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    assert terminal == read(root/'state.json') and terminal['results']==read(root/'followup-results.json')
    assert terminal['phase'] == 'completed' and framework_source() == protocol['core']
    frozen = protocol['frozen_identities']; assert all(sha(p) == h for p, h in frozen.items())
    config = load_config(root / 'private-config.yaml', [])
    search = ProjectSearch(config, root / 'ddp-independent')
    source = ProjectSearch(config, Path(protocol['source']))
    assert source.revisions.current().id == SOURCE_CURRENT and search.revisions.current().id == BASELINE
    assert source.prepared == search.prepared and source.request == search.request
    paths = list((search.output / 'version-trials').glob('*/result.json')); assert len(paths) == 1
    t = read(paths[0]); assert t['source_revision'] == ACCEPTED and t['parent'] == BASELINE
    assert t['publication'] == 'exploration' and t['bundles'] == list(source.revisions.revision(ACCEPTED).bundles)
    imported = search.revisions.revision(t['revision']); original = source.revisions.revision(ACCEPTED)
    assert imported.files == original.files and imported.bundles == original.bundles and imported.contract_ref == original.contract_ref
    request = search.request.portable()
    assert paired_decision(t['pairs'], search.request.objective, search.prepared['evaluation']) == t['measurement']
    assert t['measurement']['constraints'][0]['original_reference'] == 133.379621
    assert t['measurement']['constraints'][0]['limit'] == 140.04860205
    jobs = []; traces = []; receipts = {}
    for path in sorted((search.output / 'jobs').glob('*/run/receipt.json')):
        folder = path.parent; req = read(folder / 'request.json'); receipt = read(path)
        assert req['parameters']['request'] == request and req['devices'] == protocol['devices']
        assert req['function'] == 'run' and req['wait_for_devices'] and req['parameters']['action'] == 'evaluate'
        assert req['parameters']['reference_dir'] == search.prepared['controls_path']
        assert sha(req['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert sha(req['entry']) == receipt['entry_sha256']
        manifest = read(Path(req['cwd']) / '.opop-revision.json')
        assert all(sha(Path(req['cwd']) / n) == h for n, h in manifest['files'].items())
        for b in req['parameters']['bundles']:
            assert sha(b['file']) == CODE and b['parameters'] == {}
        assert valid_run(receipt) and len(receipt['ranks']) == 2
        job = {'job': folder.parent.name, 'request_sha256': sha(folder / 'request.json'),
               'receipt_sha256': sha(path), 'entry_sha256': sha(req['entry']),
               'manifest_sha256': sha(Path(req['cwd']) / '.opop-revision.json'),
               'bundles': [b['id'] for b in req['parameters']['bundles']], 'ranks': [], 'aggregates': {}}
        for rank in receipt['ranks']:
            k = rank['rank']; z = rank['result']; started = read(folder / f'rank-{k}-started.json')
            assert rank['status'] == 'passed' and z['valid'] and z['math_policy_unchanged']
            assert rank['framework_source'] == protocol['core'] and rank['binding']['torch_uuid'] == protocol['devices'][k]
            assert started['parameters_ref']['request_sha256'] == sha(folder / 'request.json')
            assert all(started['revision']['files'][n] == h for n, h in manifest['files'].items())
            assert len(z['samples']) == 5 and len(z['checks']) == 2
            assert all(c['passed'] and c['compared_elements'] == 248879620 for c in z['checks'])
            if req['parameters']['bundles']:
                cov=z['coverage'][req['parameters']['bundles'][0]['id']]
                assert {v['route']:v['count'] for v in cov['candidate_declared_routes']} == {'parent_eager_baseline':1,'graph_setup_capture':1}
                assert cov['reference_invocations'] == {'5d7d6bb13691b02f533f62b0':1}
            log = (folder / f'rank-{k}.log').read_text()
            job['ranks'].append({'rank': k, 'result_sha256': sha(folder / f'rank-{k}.json'),
                'log_sha256': sha(folder / f'rank-{k}.log'), 'traceback_in_log': 'Traceback (most recent call last)' in log,
                **{n: z[n] for n in ['samples', 'checks', 'metrics', 'metric_samples', 'coverage', 'setup', 'diagnostics', 'bundle_hashes']}})
            for name, d in z['diagnostic_metrics'].items():
                assert d['status'] == 'observed' and len(d['values']) == len(d['evidence']) == 3
                assert statistics.median(d['values']) == z['metrics'][name]
                for evidence, value in zip(d['evidence'], d['values'], strict=True):
                    trace, window = trace_window(read(evidence), 'opop.communication_metric_workload'); assert window
                    calc = normalize(trace, {'kernels': []}, k)
                    assert abs(calc[d['specification']['source']['field']] - value) < 1e-8
                    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and 'nccl' in e.get('name', '').lower()]
                    assert len(kernels) == (1 if req['parameters']['bundles'] else 13)
                    payloads = [e['args']['In msg nelems']*4 for e in kernels]
                    assert sorted(payloads) == ([497759232] if req['parameters']['bundles'] else [28351488]*12+[157541376])
                    assert abs(sum(e['dur'] for e in kernels)/1000-value)<1e-8
                    assert abs(union_duration(kernels)/1000-calc['communication_union_ms'])<1e-8
                    traces.append({'job': job['job'], 'rank': k, 'path': evidence, 'sha256': sha(evidence),
                        'payloads':payloads,'union_ms':calc['communication_union_ms'],
                        'nccl_graph_nodes':[e['args'].get('graph node id') for e in kernels],
                        'window': window, 'value_ms': value, 'kernel_names': dict(Counter(e['name'] for e in kernels)),
                        'logical_input_bytes': calc['logical_input_bytes'], 'collectives': dict(Counter(c['operation'] for c in calc['collectives']))})
        for name in receipt['ranks'][0]['result']['metrics']:
            series = [r['result']['metric_samples'].get(name) or [x[name] for x in r['result']['samples']] for r in receipt['ranks']]
            values = [max(v) for v in zip(*series, strict=True)]; median = statistics.median(values)
            job['aggregates'][name] = {'aligned_rank_max': values, 'median': median, 'MAD': statistics.median(abs(v - median) for v in values)}
        jobs.append(job); receipts[job['job']] = receipt
        print(json.dumps({'job': job['job'], 'step': job['aggregates']['step_ms']}), flush=True)
    assert len(jobs) == 4 and len(traces) == 24
    pairs = []
    for i, pair in enumerate(t['pairs']):
        endpoints = {}
        for arm, rec in pair.items():
            job = next(x for x in jobs if x['job'].endswith(f'pair-{i}-{arm}'))
            assert rec == receipts[job['job']]
            endpoints[arm] = job['aggregates']['step_ms']
        gain = endpoints['parent']['median'] - endpoints['candidate']['median']
        noise = 2 * max(v['MAD'] for v in endpoints.values())
        assert abs(gain - t['measurement']['paired_gains'][i]) < 1e-8
        assert abs(noise - t['measurement']['noise_bounds'][i]) < 1e-8
        pairs.append({'endpoints': endpoints, 'gain': gain, 'noise': noise})
    source_reuse = read(root / 'source-evidence-reuse.json')
    assert all(sha(p) == h for p, h in source_reuse['files'].items())
    nr = read(root / 'native-candidate-reuse.json'); natives = []
    for n in nr['ranks']:
        assert sha(n['path']) == n['sha256'] and sha(nr['normalized_path']) == n['normalized_sha256']
        con = sqlite3.connect('file:' + n['path'] + '?mode=ro', uri=True)
        rows = list(con.execute('select s.value,k.graphNodeId,k.end-k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName'))
        apis = list(con.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId group by s.value'))
        con.close(); comm = [x for x in rows if 'nccl' in x[0].lower()]
        launches = {name: count for name, count in apis if name.startswith('cudaGraphLaunch')}
        assert len(rows) == 1105 and sum(bool(x[1]) for x in rows) == 1103 and sum(launches.values()) == 2
        assert len(comm) == 1 and all(not x[1] for x in comm)
        natives.append({'rank': n['rank'], 'sha256': n['sha256'], 'kernels': len(rows), 'graph_kernels': sum(bool(x[1]) for x in rows),
                        'launches': launches, 'nccl_names': dict(Counter(x[0] for x in comm)), 'nccl_sum_ms': sum(x[2] for x in comm) / 1e6})
    assert all(sha(p) == h for p, h in frozen.items())
    m = t['measurement']
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'],
        'terminal_reason': 'batch_finished', 'terminal_sha256': sha(root / 'followup-completed.json'), 'core': protocol['core'],
        'frozen_files': len(frozen), 'all_frozen_unchanged': True, 'trial_sha256': sha(paths[0]),
        'trial': {k: t[k] for k in ['trial', 'revision', 'parent', 'source_revision', 'bundles', 'quality', 'coverage_errors', 'measurement', 'decision', 'publication']},
        'candidate_sha256': CODE, 'gain_percent': 100 * (1 - m['objective'] / m['reference']),
        'manual_pairs': pairs, 'jobs': jobs, 'metric_traces': traces, 'native_reused': natives,
        'source_reuse_scope': source_reuse['scope'], 'source_reuse_hashes_verified': len(source_reuse['files']),
        'counts': {'jobs': 4, 'ranks': 8, 'clean_rank_samples': 40, 'full_quality_checks': 16, 'metric_traces': 24},
        'pointers': {'source_accepted_revision': ACCEPTED, 'source_current': source.revisions.current().id, 'independent_current': search.revisions.current().id},
        'scope': 'Independent complete-version closure, publication exploration only. Exact508a native reused from preceding formal validation. Joint batching and communication fusion with unchanged logical bytes, not isolated communication benefit or wire/critical-path proof. First quality parent then second optimized setup, not two optimized checks. Small nonzero full comparison errors retained; exact replicas observed. Finite/memory/guard/fallback limitations remain. Reviewed source/metric mapping assistance retained; no new source/model/control run, native trace, Agent or parameter choice.',
        'next_decision':'Declare ONE normal ProjectSearch.discover on exact newly independently accepted508a revision, forked explicitly from accepted exploration. Existing real collection planner chooses normal collectors. Needed current-code registry before any next candidate; no selector/candidate/tuning/acceptance/manual collector. Preserve errors and audit before any further step.',
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = dict(report); public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    public['jobs'] = [{**{k: v for k, v in job.items() if k != 'ranks'}, 'ranks': [
        {k: v for k, v in rank.items() if k not in ['coverage', 'diagnostics', 'bundle_hashes']} for rank in job['ranks']]} for job in jobs]
    write_json(root / 'public-terminal-audit.json', public)
    print(json.dumps({'audit_sha256': public['terminal_audit_sha256'], 'public_sha256': sha(root / 'public-terminal-audit.json'),
        'public_bytes': (root / 'public-terminal-audit.json').stat().st_size, 'counts': report['counts'], 'measurement': m, 'gain_percent': report['gain_percent']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
