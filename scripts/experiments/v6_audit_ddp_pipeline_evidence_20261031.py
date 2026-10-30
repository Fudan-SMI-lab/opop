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
    assert not (root/'terminal-audit.json').exists()
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    assert terminal == read(root/'state.json') and terminal['results'] == read(root/'followup-results.json')
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
    assert (search.output / 'latest-feedback.json').read_bytes() == (source.output / 'version-trials/version-19574e13ce62/result.json').read_bytes()
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
    launch=read(root/'launch.json'); pre=read(root/'frozen-preflight.json'); startup=read(root/'startup-verification.json')
    reuse=read(root/'preparation-reuse.json')
    assert launch['core']==pre['core']==startup['core']==protocol['core']
    assert pre['frozen_files']==len(protocol['frozen_identities'])==1769
    assert sha(search.output/'request.json')==reuse['request_sha256']
    assert sha(search.output/'prepared.json')==reuse['prepared_sha256']
    assert reuse['controls_path']==search.prepared['controls_path']
    assert current.bundles==('fixed-fused-pipeline',)
    assert source.revisions.bundle(current.bundles[0])==search.revisions.bundle(current.bundles[0])
    events=[json.loads(x) for x in (search.output/'events.jsonl').read_text().splitlines()]
    assert Counter(e['type'] for e in events)==Counter({k:1 for k in ['RUN_CREATED','PROJECT_BRANCH_CREATED','PROJECT_DISCOVERED','PROJECT_AGENT_STARTED','PROJECT_GENERATION_SNAPSHOT','PROJECT_AGENT_FINISHED']})
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
            raw_rank=read(folder/f'rank-{rank}.json')
            assert set(row)-set(raw_rank)=={'rank','pid','exit_code'}
            assert all(row[k]==v for k,v in raw_rank.items()) and row['exit_code']==0
            assert row['status'] == 'passed' and z['valid'] and row['framework_source'] == protocol['core']
            assert row['binding']['torch_uuid'] == protocol['devices'][rank]
            assert start['parameters_ref']['request_sha256'] == sha(folder / 'request.json')
            assert all(start['revision']['files'][name] == h for name, h in manifest['files'].items())
            log = (folder / f'rank-{rank}.log').read_text()
            job['ranks'].append({'rank': rank, 'result_sha256': sha(folder / f'rank-{rank}.json'),
                'started_sha256': sha(folder / f'rank-{rank}-started.json'), 'log_sha256': sha(folder / f'rank-{rank}.log'),
                'traceback_in_log': 'Traceback (most recent call last)' in log, 'warning_lines':[s for s in log.splitlines() if 'Warning:' in s or '[transformers]' in s or 'profiler_' in s], 'result_keys': list(z)})
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
    access = []; writes = 0; read_sets = {}
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
                read_sets.setdefault(str(path),set()).add(int(number))
        if tool['tool'] == 'write':
            writes += 1; assert Path(inp['filePath']) == sandbox / 'collection.json'
            assert inp['content'].encode() == (sandbox / 'collection.json').read_bytes()
        access.append({'ordinal': i, 'tool': tool['tool'], 'input': inp, 'output_sha256': digest(output),
            'output_bytes': len(output.encode()), 'status': tool['status'], 'metadata': {k: (tool['metadata'] or {}).get(k) for k in ['exit', 'truncated']},
            'displayed_source_lines_exact': num_lines})
    assert len(tools) == 21 and writes == 1 and not a['contract_repairs']
    assert len(files)==6 and sum(map(len,files.values()))==93892
    # Every line of all six initial files was displayed, including the two
    # byte-truncated/continued observations reads. This proves exposure only.
    for name,raw in files.items():
        assert read_sets[str(sandbox/name)]==set(range(1,len(raw.decode().splitlines())+1)), name
    config=sandbox/'opencode.json'; config_text=config.read_text()
    assert tools[13]['output']=='?? inputs/\n?? opencode.json\n'+config_text
    config_access={'path':str(config),'sha256':sha(config),'bytes':config.stat().st_size,
                   'full_output_sha256':digest(tools[13]['output']), 'values_redacted':True}
    original_source=Path(tools[11]['input']['filePath']); original_readme=Path(tools[12]['input']['filePath'])
    extra_reads=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}for p in [original_source,original_readme]]
    log_path=Path(re.findall(r'/[^ ;]+\.log',tools[14]['input']['command'])[0])
    log_lines=log_path.read_text().splitlines()
    selected=sorted(set(x for x in log_lines if re.search(r'NCCL INFO (Algorithm|Protocol|Channel|Using|NVLS|Connected)',x)))[:30]
    assert tools[14]['output']=='\n'.join(selected)+'\n---\n'+str(sum('via SHM/direct/direct'in x for x in log_lines))+'\n'
    ranks=[read(search.output/f'jobs/discover-710b2b40ab/run/rank-{i}.json')for i in (0,1)]
    assert 'samples'not in ranks[0] and tools[15]['output'].endswith("KeyError: 'samples'\n")
    assert tools[15]['metadata']['exit']==1
    assert tools[16]['output']==str(list(ranks[0]))+'\n'
    expected=''
    for rank,row in enumerate(ranks):
        z=row['result']; expected+=f"rank {rank} keys: {list(z)[:12]}\n  summary: {z.get('summary')}\n  diagnostic: {z.get('diagnostic')}\n"
    assert expected==tools[17]['output']
    script=tools[19]['input']['command'].split("<<'EOF'\n",1)[1].rsplit('\nEOF',1)[0]
    try: compile(script,'<stdin>','exec')
    except SyntaxError as err: assert err.lineno==9 and tools[19]['metadata']['exit']==1
    else: raise AssertionError('recorded syntax error not reproduced')
    from jsonschema import Draft202012Validator
    authored=read(sandbox/'collection.json'); schema=json.loads(files['inputs/collection.schema.json'])
    Draft202012Validator(schema).validate(authored)
    assert set(authored['collectors'])<=set(json.loads(files['inputs/available-collectors.json']))
    assert all(isinstance(v,str)and v for v in authored['unavailable_fields'].values())
    assert tools[20]['output']==f"schema OK; {len(authored['collectors'])} collectors, {len(authored['requested_fields'])} requested fields, {len(authored['unavailable_fields'])} unavailable fields\n"
    assert authored==d['collection_plan']['plan']
    write_json(private/'complete-access-review.json',{'tools':access,'extra_reads':extra_reads,
        'provider_configuration':config_access,'initial_read_line_coverage':{str(k):len(v)for k,v in read_sets.items()},
        'review':'All 21 complete recorded inputs and outputs reviewed. Every displayed source line and full six-input line union verified; plan single write exact; log grep and JSON outputs reconstructed; two Python errors preserved. Tool14 actually read own provider configuration: exact server-only suffix/hash verification, never display its content. Extra original training source/README and actual discovery logs/results read. No recorded current optimized source/latest-feedback/audit/reviewer answer read or model/tensor/CUDA/processgroup/collective probe. External FS nonhermetic, not an absolute absence or unaided-cohort claim.'})
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
            'assistant_messages': sum(x['data'].get('role') == 'assistant' for x in history['message']), 'tools': len(tools), 'displayed_source_lines_verified':sum(x['displayed_source_lines_exact'] for x in access), 'all_six_initial_files_fully_exposed':True, 'plan_sha256':sha(sandbox/'collection.json'), 'provider_config_access_hash_only':config_access,
            'contract_repairs': a['contract_repairs'], 'usage': a['usage'], 'same_prompt_model_tools': True,
            'history_sha256': sha(private / (a['session'] + '.json')), 'tools_sha256': sha(private / (a['session'] + '-tools.json')),
            'complete_access_review_sha256': sha(private / 'complete-access-review.json')},
        'limitations': [
            'Five jobs/ten ranks: discovery plus four autonomously selected existing collectors. No new model full-quality/clean/trial/acceptance; calibration quality is synthetic only.',
            'All six initial files fully displayed, plus original training source/README and actual current logs/results. Own provider config read at tool14, verified only on server by hash/exact suffix with values redacted. No recorded current optimized code/copied latest-feedback/later audit or reviewer answer reads; filesystem nonhermetic.',
            'Plan no-candidate-files claim is false: exact accepted 02b933 bundle is installed. Initial planner input omission does not mean no candidate exists; original four-microbatch source is not the current segmented implementation.',
            'Wait-dominated/arrival-skewed language is not established by asymmetric duration alone; separate ranks/profiles/calibration do not causally decompose waiting, wire or critical path. One profile per rank is not repeated clean timing.',
            'CPU submission/callbacks are not GPU readiness or downstream consumer completion. Empty callback evidence means unknown, not absent gradients. One passive bucket inventory cannot identify eight timed chunk owners uniquely.',
            'New actual SQLite/native/trace values are recomputed, not inherited. SHM channels are actual log evidence; RING_LL kernel names and separately scoped tuning decisions cannot be merged into a per-step protocol proof.',
            'Registry lineage/installed-bundle containment is not causal attribution or multiple active bundles. Source CURRENT remains empty; accepted exploration and isolated discovery are distinct.',
            'Tool16 wrong samples schema KeyError and tool20 syntax error retained; compound git/log pipelines can hide failures. JSON/schema success is not correctness of plan interpretation. No experiment or Agent rerun.',
        ],
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k not in ['registry', 'transport_logs']}
    public['registry'] = {**report['registry'], 'plan_change': {k: len(v) if isinstance(v, list) else v for k, v in d['plan_change'].items()}}
    public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    write_json(root / 'public-terminal-audit.json', public)
    print(json.dumps({'audit_sha256': public['terminal_audit_sha256'], 'public_sha256': sha(root / 'public-terminal-audit.json'),
        'public_bytes': (root / 'public-terminal-audit.json').stat().st_size, 'profiles': profiles, 'native': natives, 'calibration': calibration,
        'timelines': [{k: v for k, v in t.items() if k not in ['clock', 'gradient_stream_markers']} for t in timelines],
        'registry': public['registry'], 'agent': report['agent']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    audit(p.parse_args().root)
