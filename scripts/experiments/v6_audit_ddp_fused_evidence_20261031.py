"""Read-only accepted-version discovery/collector audit; no workload rerun."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sqlite3
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
    source_current = source.revisions.current(); old = source.revisions.revision(protocol["source_revision"]); current = search.revisions.current(); d = read(search.output / 'discovery.json')
    assert old.id == protocol['source_revision'] and current.parent_id == old.id
    assert current.files == old.files and current.bundles == old.bundles and current.contract_ref == old.contract_ref
    assert search.request == source.request and search.prepared == source.prepared
    assert source_current.id == protocol['source_current']
    assert old.id != source_current.id
    assert not list((search.output / 'trials').glob('*/result.json'))
    assert (search.output / 'latest-feedback.json').read_bytes() == (source.output / 'version-trials/version-aefa021e172f/result.json').read_bytes()
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
    jobs = []; profiles = []; raw_discovery = []; timelines = []; timeline_raw = []; calibration = []; natives = []
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
        normalized = folder / 'nsys-normalized.json'
        if normalized.exists():
            from kernel_optimizer.project.nsys import normalize as native_normalize
            for row in read(normalized):
                calc = native_normalize(Path(row['evidence']))
                assert all(calc[k] == row[k] for k in calc)
                con=sqlite3.connect('file:'+row['evidence']+'?mode=ro',uri=True)
                kernels=list(con.execute('select s.value,k.graphNodeId,k.end-k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName'))
                apis=list(con.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME k join StringIds s on s.id=k.nameId group by s.value'))
                joins=list(con.execute("select r.correlationId,count(k.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId left join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=r.correlationId where s.value like 'cudaGraphLaunch%' group by r.correlationId order by r.start"));con.close()
                comm=[k for k in kernels if 'nccl' in k[0].lower()]
                assert len(kernels)==row['cuda_kernels']==len(row['kernels']) and len(comm)==row['nccl_kernels']
                assert abs(sum(v[2] for v in comm)/1e6-row['nccl_kernel_sum_ms'])<1e-8
                natives.append({'rank':row['rank'],'path':row['evidence'],'sha256':sha(row['evidence']),
                    'normalized_sha256':sha(normalized),'kernels':len(kernels),'graph_kernels':sum(bool(v[1]) for v in kernels),
                    'launches':{n:c for n,c in apis if n.startswith('cudaGraphLaunch')},'launch_correlations':joins,
                    'nccl':[{'name':n,'graph_node':g,'sum_ms':dt/1e6} for n,g,dt in comm],
                    'hardware_traffic_status':row['hardware_traffic_status']})
        jobs.append(job)
    assert len(jobs) == 5 and len(raw_discovery) == len(calibration) == 2
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
    assert len(timelines) == len(natives) == 2
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
            path = Path(inp['filePath'])
            if path.is_relative_to(sandbox):
                relative = path.relative_to(sandbox).as_posix(); raw = files[relative]
            else: raw = path.read_bytes()
            lines = raw.decode().splitlines()
            for number, content in re.findall(r'(?m)^(\d+): (.*)$', output):
                assert lines[int(number) - 1] == content, (i, number)
                num_lines += 1
        if tool['tool'] == 'write':
            writes += 1; assert Path(inp['filePath']) == sandbox / 'collection.json'
            assert inp['content'].encode() == (sandbox / 'collection.json').read_bytes()
        access.append({'ordinal': i, 'tool': tool['tool'], 'input': inp, 'output_sha256': digest(output),
            'output_bytes': len(output.encode()), 'status': tool['status'], 'metadata': {k: (tool['metadata'] or {}).get(k) for k in ['exit', 'truncated']},
            'displayed_source_lines_exact': num_lines})
    assert len(tools) == 39 and writes == 1 and not a['contract_repairs']
    # Tool11's provider configuration is deliberately hash-only. Never expose its
    # input values or output/preview/display metadata in public summaries.
    extra_reads = []
    for path in [root/'launch.json',root/'protocol.json',
        root.parent/'agent-repair-20261021/fresh-e2e/ddp-communication/sandboxes/collection-planner-be66b3712021/collection.json',
        root.parent/'agent-repair-20261021/fresh-fixed/ddp-communication/sandboxes/collection-planner-332d16b5b0f3/collection.json']:
        extra_reads.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    pure = tools[33]
    raw = read(search.output/'jobs/discover-0ee3a52759/run/discovery-rank-0.json')
    rendered=''
    for c in raw['calls']:
        if c['id'] in ('231ae1be6ab1e455d286ec7a','5d7d6bb13691b02f533f62b0'):
            rendered+=json.dumps({k:v for k,v in c.items() if k!='signatures'},indent=1)[:800]+'\n---\n'
    rendered+='groups: '+str([g['id'] for g in raw.get('groups',[])][:10])+'\n'
    assert pure['output']==rendered
    assert tools[12]['output']==(search.output/'jobs/discover-0ee3a52759/run/receipt.json').read_text()+(search.output/'jobs/discover-0ee3a52759/run/rank-0.json').read_text()
    assert tools[31]['output']==Path(extra_reads[2]['path']).read_text()+'===\n'+Path(extra_reads[3]['path']).read_text()
    cat=(root/'launch.json').read_text()+(root/'protocol.json').read_text()
    assert tools[18]['output']==''.join(cat.splitlines(keepends=True)[:200])
    write_json(private/'complete-access-review.json',{'tools':access,'extra_reads':extra_reads,
        'review':'All39 recorded inputs/outputs reviewed; read-line contents verified, selected JSON and concatenated-file outputs reconstructed exactly. Tool10 git128; tool11 private provider config read (hash-only audit); tool19 launch/protocol,22 frozen controller,32 two historical planner outputs actually read. These are additional external inputs, not six-input-only reasoning. No recorded model/tensor/CUDA/processgroup/collective execution; tools34 and39 only JSON parsing/schema. Current installed candidate code and original latest-feedback result not opened in recorded tools. Nonhermetic FS; no absolute access-absence or unaided-cohort claim.'})
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    lineage = Counter(json.dumps(g.lineage, sort_keys=True, default=str) for g in registry.groups)
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'], 'terminal_reason': 'batch_finished',
        'terminal_sha256': sha(root / 'followup-completed.json'), 'core': protocol['core'], 'frozen_files': len(protocol['frozen_identities']),
        'all_frozen_unchanged': True, 'source_current': source_current.id, 'incoming_revision': old.id, 'current': current.id, 'code_sha256': protocol['candidate_sha256'],
        'same_code_bundle_contract': True, 'active_bundles': list(current.bundles), 'discovery_sha256': sha(search.output / 'discovery.json'),
        'registry': {'id': registry.id, 'parent': registry.parent_id, 'calls': len(plan.calls), 'groups': len(registry.groups),
                     'plan_change': d['plan_change'], 'lineage_counts': dict(lineage)},
        'jobs': jobs, 'profiles': profiles, 'timelines': timelines, 'cross_rank_timeline': cross, 'calibration': calibration,
        'native': natives, 'tuning': tuning, 'transport_logs': transports, 'collection_plan': d['collection_plan']['plan'],
        'agent': {'session': a['session'], 'generation': meta['id'], 'initial_files': len(files), 'initial_bytes': sum(map(len, files.values())),
            'assistant_messages': sum(x['data'].get('role') == 'assistant' for x in history['message']), 'tools': len(tools),
            'contract_repairs': a['contract_repairs'], 'usage': a['usage'], 'same_prompt_model_tools': True,
            'history_sha256': sha(private / (a['session'] + '.json')), 'tools_sha256': sha(private / (a['session'] + '-tools.json')),
            'complete_access_review_sha256': sha(private / 'complete-access-review.json')},
        'limitations': [
            'Five jobs/ten ranks: discovery and four planner-selected existing collectors. No new clean model samples/full quality/trial/acceptance. Calibration is synthetic full-payload reference only, not model quality or wire lower bound.',
            'Planner actually read own config, launch/protocol/frozen controller and two historical collection plans. External historical-plan exposure is confirmed; not limited to six initial files. No record of current candidate source or copied latest-feedback content read. FS remains nonhermetic.',
            'Plan four-eager-microbatch description contradicts current batched CUDA Graph implementation; one inventory bucket does not prove exact native timed flat-buffer ownership. Prior accepted full-quality evidence is separate.',
            'CPU submission clocks/callback absence do not establish GPU producer readiness/consumer completion. Separate calibration cannot bound waiting versus wire or explain another capture. Profile/native NCCL sums include wait; no clean/critical-path subtraction.',
            'Current new Nsys is independently recomputed from both actual SQLite exports, not inherited old native counts. Transport channels are actual logs; kernel entry/tuning protocol records belong to different scopes.',
            'Registry lineage is surviving-call or installed-bundle containment, not causal group attribution/multiple active bundles. Source current remains empty; accepted exploration and isolated discovery are distinct.',
            'Tool10 git error and empty/missing-path shell pipelines preserved; schema success is not factual correctness. Raw provider config/history kept server-only, output inspection thereafter hash/redaction only.',
        ],
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k not in ['registry', 'transport_logs']}
    public['registry'] = {**report['registry'], 'plan_change': {k: len(v) if isinstance(v, list) else v for k, v in d['plan_change'].items()}}
    public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    write_json(root / 'public-terminal-audit.json', public)
    print(json.dumps({'audit_sha256': public['terminal_audit_sha256'], 'public_sha256': sha(root / 'public-terminal-audit.json'),
        'public_bytes': (root / 'public-terminal-audit.json').stat().st_size, 'profiles': profiles, 'native': natives,
        'timelines': [{k: v for k, v in t.items() if k not in ['clock', 'gradient_stream_markers']} for t in timelines],
        'registry': public['registry'], 'agent': report['agent']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
