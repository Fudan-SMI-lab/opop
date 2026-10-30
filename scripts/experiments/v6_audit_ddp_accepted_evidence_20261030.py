"""Read-only accepted-version discovery/collector audit; no workload rerun."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.aggregation import merge_ranks
from kernel_optimizer.project.communication import normalize, match_ranks, trace_window
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.nccl_tuning import read_logs
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze, cross_rank
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def digest(s): return hashlib.sha256(s.encode()).hexdigest()


def audit(root):
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed' and protocol['core'] == framework_source()
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    cfg = load_config(root / 'private-config.yaml', [])
    search = ProjectSearch(cfg, root / 'ddp-evidence'); source = ProjectSearch(cfg, Path(protocol['source']))
    old = source.revisions.current(); current = search.revisions.current(); d = read(search.output / 'discovery.json')
    assert old.id == protocol['source_revision'] and current.parent_id == old.id
    assert current.files == old.files and current.bundles == old.bundles and current.contract_ref == old.contract_ref
    assert search.request == source.request and search.prepared == source.prepared
    assert read(source.output / 'discovery.json')['revision'] != old.id
    assert not list((search.output / 'trials').glob('*/result.json'))
    assert not (search.output / 'latest-feedback.json').exists()
    plan = search.revisions.load(current.plan_ref, ExecutionPlan)
    registry = search.revisions.load(current.registry_ref, GroupRegistry)
    old_plan = source.revisions.load(old.plan_ref, ExecutionPlan)
    old_registry = source.revisions.load(old.registry_ref, GroupRegistry)
    assert plan.id == d['plan'] and registry.id == d['registry'] and d['revision'] == current.id
    assert {g.id for g in registry.groups} == {g['id'] for g in d['groups']}
    assert {c.id for c in plan.calls} == {c['id'] for c in d['calls']}
    assert set(d['plan_change']['retired_calls']) == {c.id for c in old_plan.calls} - {c.id for c in plan.calls}
    assert set(d['plan_change']['new_calls']) == {c.id for c in plan.calls} - {c.id for c in old_plan.calls}
    assert set(d['plan_change']['retired_groups']) == {g.id for g in old_registry.groups} - {g.id for g in registry.groups}
    assert registry.parent_id == old_registry.id
    jobs = []; profiles = []; raw_discovery = []; timelines = []; timeline_raw = []; calibration = []
    fields = ['communication_kernel_count', 'communication_kernel_sum_ms', 'communication_union_ms',
              'compute_overlap_ms', 'communication_without_compute_ms', 'logical_input_bytes']
    for folder in sorted((search.output / 'jobs').glob('*/run')):
        req = read(folder / 'request.json'); rec = read(folder / 'receipt.json')
        assert req['parameters']['request'] == search.request.portable() and req['devices'] == protocol['devices']
        assert req['parameters']['reference_dir'] == search.prepared['controls_path']
        assert sha(req['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert sha(req['entry']) == rec['entry_sha256']
        manifest = read(Path(req['cwd']) / '.opop-revision.json')
        assert all(sha(Path(req['cwd']) / name) == h for name, h in manifest['files'].items())
        assert len(req['parameters']['bundles']) == 1
        b = req['parameters']['bundles'][0]; assert sha(b['file']) == protocol['candidate_sha256'] and b['parameters'] == {}
        assert rec['status'] == 'passed' and len(rec['ranks']) == 2
        job = {'job': folder.parent.name, 'action': req['parameters']['action'], 'request_sha256': sha(folder / 'request.json'),
            'receipt_sha256': sha(folder / 'receipt.json'), 'entry_sha256': sha(req['entry']),
            'manifest_sha256': sha(Path(req['cwd']) / '.opop-revision.json'), 'ranks': []}
        for row in rec['ranks']:
            rank = row['rank']; z = row['result']; start = read(folder / f'rank-{rank}-started.json')
            assert row['status'] == 'passed' and z['valid'] and row['framework_source'] == protocol['core']
            assert row['binding']['torch_uuid'] == protocol['devices'][rank]
            assert start['parameters_ref']['request_sha256'] == sha(folder / 'request.json')
            assert all(start['revision']['files'][name] == h for name, h in manifest['files'].items())
            log = (folder / f'rank-{rank}.log').read_text()
            job['ranks'].append({'rank': rank, 'result_sha256': sha(folder / f'rank-{rank}.json'),
                'started_sha256': sha(folder / f'rank-{rank}-started.json'), 'log_sha256': sha(folder / f'rank-{rank}.log'),
                'traceback_in_log': 'Traceback (most recent call last)' in log, 'result_keys': list(z)})
            assert 'checks' not in z and 'samples' not in z
            if job['action'] == 'profile':
                raw = read(z['discovery']); assert sha(raw['trace']) == raw['trace_sha256']
                trace = read(raw['trace']); calc = normalize(trace, raw, rank)
                assert all(abs(calc[k] - raw['communication'][k]) < 1e-8 for k in fields)
                raw_discovery.append(raw)
                api = Counter(e.get('name') for e in trace['traceEvents'] if e.get('cat') in ['cuda_runtime', 'cuda_driver'] and 'cudaGraphLaunch' in e.get('name', ''))
                profiles.append({'rank': rank, 'discovery_sha256': sha(z['discovery']), 'trace_sha256': raw['trace_sha256'],
                    'kernels': len(raw['kernels']), 'calls': len(raw['calls']), 'groups': len(raw['groups']),
                    'profiled_wall_ms': raw['profiled_wall_ms'], 'communication': {k: calc[k] for k in fields},
                    'operations': dict(Counter(c['operation'] for c in calc['collectives'])),
                    'payloads': dict(Counter(c['logical_input_bytes'] for c in calc['collectives'])),
                    'graph_launch_api_names': dict(api), 'coverage': raw['coverage']})
            if job['action'] == 'collective_calibration':
                c = z['calibration']; assert (c['warmup'], c['repeats']) == (2, 5)
                for value in c['rows']:
                    assert value['quality'] and len(value['samples_ms']) == 5
                    assert statistics.median(value['samples_ms']) == value['median_completion_ms']
                    assert abs(value['logical_bytes'] / (value['median_completion_ms'] / 1000) - value['logical_payload_bytes_per_second']) < 1e-4
                calibration.append({'rank': rank, 'result': c})
        jobs.append(job)
    assert len(jobs) == 4 and len(raw_discovery) == len(calibration) == 2
    merged = merge_ranks(raw_discovery)
    assert merged['calls'] == d['calls'] and merged['call_resources_by_rank'] == d['call_resources']
    assert match_ranks([r['communication'] for r in raw_discovery]) == d['communication']['cross_rank']
    for path in sorted((search.output / 'jobs').glob('timeline-*/run/timeline-analysis-rank-*.json')):
        z = read(path); trace = read(z['trace']); assert sha(z['trace']) == z['trace_sha256']
        gradients = {e['name']: e['parameters'] for e in z['gradient_readiness']['events']}
        inventory = {'buckets': [{k: v for k, v in b.items() if k not in ['observed_parameter_count', 'cpu_post_accumulate_times_us', 'last_cpu_post_accumulate_us', 'timed_collective_identity', 'interpretation']} for b in z['gradient_readiness']['buckets']]}
        calc = analyze(trace, rank=z['rank'], anchors=z['clock_anchors'], domain=z['clock']['domain'], gradients=gradients, inventory=inventory)
        for k in ['devices', 'clock', 'cuda_synchronization', 'copies_and_memsets', 'workload_window', 'gradient_readiness']:
            assert calc[k] == z[k], k
        assert calc['communication'] == z['communication']
        timeline_raw.append(z)
        sliced, _ = trace_window(trace, 'opop.timeline_workload')
        timelines.append({'rank': z['rank'], 'path': str(path), 'sha256': sha(path), 'trace_sha256': z['trace_sha256'],
            'profiled_wall_ms': z['profiled_wall_ms'], 'communication': {k: z['communication'][k] for k in fields},
            'devices': {dev: {k: v for k, v in a.items() if not k.endswith('intervals_us')} for dev, a in z['devices'].items()},
            'clock': z['clock'], 'sync_apis': len(z['cuda_synchronization']),
            'graph_launches': dict(Counter(e['name'] for e in sliced['traceEvents'] if 'cudaGraphLaunch' in e.get('name', ''))),
            'gradient_events': len(z['gradient_readiness']['events']), 'gradient_buckets': len(z['gradient_readiness']['buckets']),
            'gradient_hooks_enabled': z['gradient_hooks_enabled'], 'gradient_hook_missing': z['gradient_hook_missing'],
            'gradient_stream_markers': z['gradient_stream_markers']})
    assert len(timelines) == 2
    cross = cross_rank(timeline_raw); assert cross == d['extra_collectors']['execution_timeline']['cross_rank']
    tuning_paths = sorted((search.output / 'jobs').glob('nccl-tuning-*/run/nccl*.log'))
    tuning = read_logs(tuning_paths); supplied = d['extra_collectors']['nccl_tuning']
    assert all(tuning[k] == supplied[k] for k in tuning)
    transports = []
    for path in sorted((search.output / 'jobs').glob('*/run/nccl*.log')):
        transports.append({'path': str(path), 'sha256': sha(path), 'channels': [s for s in path.read_text().splitlines() if 'Channel ' in s and ' via ' in s]})
    a = d['collection_plan']['agent']; private = root / 'terminal-private-history'
    history = read(private / (a['session'] + '.json')); tools = read(private / (a['session'] + '-tools.json'))
    meta, prompt, files = load(Path(a['generation_snapshots'][0])); sandbox = Path(meta['sandbox'])
    assert meta['framework_source'] == protocol['core'] and meta['fresh_session']
    assert all((sandbox / name).read_bytes() == raw for name, raw in files.items())
    old_meta, old_prompt, _ = load(root.parent / 'ddp-communication-trial-20261030/ddp-search/generation-snapshots/generation-fb697d303320.json')
    assert prompt == old_prompt and meta['tool_model_settings'] == old_meta['tool_model_settings']
    access = []; writes = 0
    for i, tool in enumerate(tools, 1):
        inp = tool['input']; output = tool['output']; num_lines = 0
        if tool['tool'] == 'read' and Path(inp['filePath']).is_file():
            relative = Path(inp['filePath']).relative_to(sandbox).as_posix(); lines = files[relative].decode().splitlines()
            for number, content in re.findall(r'(?m)^(\d+): (.*)$', output):
                assert lines[int(number) - 1] == content, (i, number)
                num_lines += 1
        if tool['tool'] == 'write':
            writes += 1; assert Path(inp['filePath']) == sandbox / 'collection.json'
            assert inp['content'].encode() == (sandbox / 'collection.json').read_bytes()
        access.append({'ordinal': i, 'tool': tool['tool'], 'input': inp, 'output_sha256': digest(output),
            'output_bytes': len(output.encode()), 'status': tool['status'], 'metadata': {k: (tool['metadata'] or {}).get(k) for k in ['exit', 'truncated']},
            'displayed_source_lines_exact': num_lines})
    assert len(tools) == 14 and writes == 1 and not a['contract_repairs']
    write_json(private / 'complete-access-review.json', {'tools': access,
        'review': 'All 14 actual commands/read paths/output records reviewed. Reads confined to initial sandbox inputs and directory listings; grep returned both observations and request lines. Bash9 counts/greps existing input; bash14 JSON schema/name validation only. No model/CUDA/collective probe or external answer read observed; filesystem is nonhermetic. Observation reads are partial, not full 7127-line consumption. Shape/schema success is not factual correctness.'})
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    lineage = Counter(json.dumps(g.lineage, sort_keys=True, default=str) for g in registry.groups)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'], 'terminal_reason': 'batch_finished',
        'terminal_sha256': sha(root / 'followup-completed.json'), 'core': protocol['core'], 'frozen_files': len(protocol['frozen_identities']),
        'all_frozen_unchanged': True, 'source_current': old.id, 'current': current.id, 'code_sha256': protocol['candidate_sha256'],
        'same_code_bundle_contract': True, 'active_bundles': list(current.bundles), 'discovery_sha256': sha(search.output / 'discovery.json'),
        'registry': {'id': registry.id, 'parent': registry.parent_id, 'calls': len(plan.calls), 'groups': len(registry.groups),
                     'plan_change': d['plan_change'], 'lineage_counts': dict(lineage)},
        'jobs': jobs, 'profiles': profiles, 'timelines': timelines, 'cross_rank_timeline': cross, 'calibration': calibration,
        'tuning': tuning, 'transport_logs': transports, 'collection_plan': d['collection_plan']['plan'],
        'agent': {'session': a['session'], 'generation': meta['id'], 'initial_files': len(files), 'initial_bytes': sum(map(len, files.values())),
            'assistant_messages': sum(x['data'].get('role') == 'assistant' for x in history['message']), 'tools': len(tools),
            'contract_repairs': a['contract_repairs'], 'usage': a['usage'], 'same_prompt_model_tools': True,
            'history_sha256': sha(private / (a['session'] + '.json')), 'tools_sha256': sha(private / (a['session'] + '-tools.json')),
            'complete_access_review_sha256': sha(private / 'complete-access-review.json')},
        'limitations': [
            'Discovery plus 3 planner-selected diagnostics only; four jobs/eight ranks valid. No new clean model samples, full model quality checks, candidate/selector/trial or acceptance. Synthetic calibration has20rank samples/correctnesschecks only.',
            'Planner claims no CUDA Graph/no installed candidate are contradicted by exact accepted bundle and current trace Graph launch APIs. Eager attention does not exclude Graph execution. No hand edits to generated plan.',
            'Planner initial inputs omit explicit accepted revision/bundle/source code metadata. It read original task and partial current observations, not current candidate code. This explains an information boundary, not a proved unique cause or permission to change production prompts.',
            'Planner claim that the separate timeline can attribute discovery rank asymmetry is too strong. New capture has its own CPU clock bounds/submit spreads, not proof of prior period cause, wire time or GPU arrival.',
            'Timeline sequences differ from discovery; no cross-capture sequence identity join. CPU gradient callbacks under Graph do not observe all three captured microbatches;148observed callbacks are not full592-event execution proof.',
            'NCCL tuning decisions include setup/warmup/peer checks and differ in capture period from kernel names. Counts are log records, not physical per-step counts; no wire/critical-path inference.',
            'Fresh registry lineage is surviving call/installed bundle containment, not multi-bundle coexistence or causal group attribution. Prior full independent quality remains separate.',
            'No new Nsys selected. Existing prior native evidence is historical exact-version evidence only; current raw Torch discovery/timeline audited directly.',
        ],
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k not in ['registry', 'transport_logs']}
    public['registry'] = {**report['registry'], 'plan_change': {k: len(v) if isinstance(v, list) else v for k, v in d['plan_change'].items()}}
    public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    write_json(root / 'public-terminal-audit.json', public)
    print(json.dumps({'audit_sha256': public['terminal_audit_sha256'], 'public_sha256': sha(root / 'public-terminal-audit.json'),
        'public_bytes': (root / 'public-terminal-audit.json').stat().st_size, 'profiles': profiles,
        'timelines': [{k: v for k, v in t.items() if k not in ['clock', 'gradient_stream_markers']} for t in timelines],
        'registry': public['registry'], 'agent': report['agent']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
