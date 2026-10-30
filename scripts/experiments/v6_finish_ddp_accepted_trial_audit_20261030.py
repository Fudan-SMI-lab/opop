"""Reconcile actual saved DDP feedback trial evidence; no workload execution."""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import statistics
import re

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ExecutionPlan, GroupRegistry
from kernel_optimizer.project.communication import normalize
from kernel_optimizer.project.feedback import route, candidate_artifacts
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.observation_artifacts import feedback_artifacts
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.project.nccl_tuning import read_logs
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_ddp_accepted_trial_20261030 import digest

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-accepted-communication-trial-20261030')


def main():
    a = read(ROOT / 'terminal-numerical-audit.json')
    p = read(ROOT / 'protocol.json'); search = ROOT / 'ddp-search'
    trial = read(next((search / 'trials').glob('*/result.json')))
    graph = []; profiles = []; timelines = []; calibration = []; diagnostics = []
    for n in a['native']:
        con = sqlite3.connect('file:' + n['path'] + '?mode=ro', uri=True)
        grouped = list(con.execute('select graphNodeId is not null and graphNodeId != 0, count(*) from CUPTI_ACTIVITY_KIND_KERNEL group by 1'))
        nccl_graph = con.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where lower(s.value) like '%nccl%' and k.graphNodeId is not null and k.graphNodeId != 0").fetchone()[0]
        launches = list(con.execute("select r.correlationId,count(k.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId left join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId=r.correlationId where s.value like 'cudaGraphLaunch%' group by r.correlationId order by r.start"))
        con.close()
        graph.append({'rank': n['rank'], 'sqlite_sha256': n['sha256'], 'kernels_by_graph_flag': grouped,
                      'cudaGraphLaunch': sum(v for k, v in n['runtime_counts'].items() if k.startswith('cudaGraphLaunch')),
                      'launch_correlations_kernel_counts': launches, 'nccl_graph_kernels': nccl_graph})
    for path in sorted((search / 'jobs').glob('*profile/run/discovery-rank-*.json')):
        d = read(path); rank = int(path.stem.split('-')[-1]); assert sha(d['trace']) == d['trace_sha256']
        calc = normalize(read(d['trace']), d, rank)
        fields = ['communication_kernel_count', 'communication_kernel_sum_ms', 'communication_union_ms',
                  'compute_overlap_ms', 'communication_without_compute_ms', 'logical_input_bytes']
        assert all(abs(calc[k] - d['communication'][k]) < 1e-8 for k in fields)
        profiles.append({'rank': rank, 'path': str(path), 'sha256': sha(path), 'trace_sha256': d['trace_sha256'],
                         'kernels': len(d['kernels']), 'calls': len(d['calls']), 'groups': len(d['groups']),
                         'communication': {k: calc[k] for k in fields},
                         'operations': dict(Counter(c['operation'] for c in calc['collectives'])),
                         'graph_launch_rows': sum(e.get('name','').startswith(('cudaGraphLaunch','cuGraphLaunch')) for e in read(d['trace'])['traceEvents']),
                         'payloads': dict(Counter(c['logical_input_bytes'] for c in calc['collectives']))})
    for path in sorted((search / 'jobs').glob('timeline-*/run/timeline-analysis-rank-*.json')):
        d = read(path); assert sha(d['trace']) == d['trace_sha256']
        new = analyze(read(d['trace']), rank=d['rank'], anchors=d['clock_anchors'], domain=d['clock']['domain'])
        for k in ['devices', 'clock', 'cuda_synchronization', 'workload_window']: assert new[k] == d[k]
        fields = ['communication_kernel_count', 'logical_input_bytes', 'communication_kernel_sum_ms',
                  'communication_union_ms', 'compute_overlap_ms', 'communication_without_compute_ms']
        assert all(abs(new['communication'][k] - d['communication'][k]) < 1e-8 for k in fields)
        timelines.append({'path': str(path), 'sha256': sha(path), 'trace_sha256': d['trace_sha256'], 'rank': d['rank'],
                          'devices': d['devices'], 'communication': {k: d['communication'][k] for k in fields},
                          'graph_launch_rows': sum(e.get('name','').startswith(('cudaGraphLaunch','cuGraphLaunch')) for e in read(d['trace'])['traceEvents']),
                          'kernel_count': sum(e.get('cat')=='kernel' for e in read(d['trace'])['traceEvents']),
                          'profiled_wall_ms': d['profiled_wall_ms'], 'sync_api_count': len(d['cuda_synchronization'])})
    for path in sorted((search / 'jobs').glob('calibration-*/run/rank-*.json')):
        if '-started' in path.name: continue
        c = read(path)['result']['calibration']; assert (c['warmup'], c['repeats']) == (2, 5)
        for row in c['rows']:
            assert row['quality'] and len(row['samples_ms']) == 5
            assert statistics.median(row['samples_ms']) == row['median_completion_ms']
            assert abs(row['logical_bytes'] / (row['median_completion_ms'] / 1000) - row['logical_payload_bytes_per_second']) < 1e-4
        calibration.append({'rank': int(path.stem.split('-')[-1]), 'path': str(path), 'sha256': sha(path), 'result': c})
    tuning_paths = sorted((search / 'jobs').glob('nccl-tuning-*/run/nccl*.log'))
    tuning = read_logs(tuning_paths)
    supplied = trial['resources']['extra_collectors']['after'].get('nccl_tuning')
    if supplied: assert all(tuning[k] == supplied[k] for k in tuning)
    for path in sorted((search / 'jobs').glob('candidate-diagnostics-*/run/candidate-diagnostics-rank-*.json')):
        d = read(path)
        diagnostics.append({'path': str(path), 'sha256': sha(path), 'events': d['events'],
                            'scope': 'Run-only, does not execute observe/compare; normal False returns are not failures.'})
    telemetry = []
    for path in sorted((search / 'jobs').glob('telemetry-*/run/telemetry-rank-*.json')):
        telemetry.append({'path': str(path), 'sha256': sha(path), 'result': read(path)})

    source = ProjectSearch(load_config(ROOT / 'private-config.yaml', []), Path(p['source']))
    current = source.revisions.current()
    reg = source.revisions.load(current.registry_ref, GroupRegistry)
    plan = source.revisions.load(current.plan_ref, ExecutionPlan)
    raw = (ROOT / 'original-feedback.json').read_text()
    packaged, compact = feedback_artifacts(raw); historical = candidate_artifacts(raw, source.store)
    deliveries = []; access = []; private = ROOT / 'terminal-private-history'
    for ag in a['agents']:
        meta, prompt, files = load(Path(ag['snapshot']))
        if ag['role'] in ['selection', 'candidate']:
            kw = {} if ag['role'] == 'selection' else {'selected_groups': trial['selection']['groups'], 'plan': plan}
            actual = route(compact, current_revision=current.id, registry=reg, **kw)
            assert files['inputs/previous-feedback.txt'] == actual.encode()
            assert json.loads(actual)['supplemental_evidence'] is None
            for name, value in {**packaged, **historical}.items():
                assert files['inputs/' + name] == (value.encode() if isinstance(value, str) else value)
            deliveries.append({'role': ag['role'], 'feedback_bytes': len(actual.encode()), 'feedback_sha256': digest(actual),
                               'original_history_and_packaged_artifacts_equal': True, 'supplemental_evidence': None})
        tools = read(private / (ag['session'] + '-tools.json'))
        inventory = []
        for i, t in enumerate(tools):
            inp = t['input']; out = t['output']; extra = t['metadata'].get('outputPath')
            inventory.append({'ordinal': i + 1, 'tool': t['tool'], 'input': inp,
                              'output_sha256': digest(out), 'output_bytes': len(out.encode()), 'status': t['status'],
                              'exit': t['metadata'].get('exit'), 'full_output_path': extra,
                              'full_output_sha256': sha(extra) if extra and Path(extra).exists() else None})
            # Preserve exact generated CPU scripts despite Agent removing its tmp files.
            if t['tool'] == 'write' and inp.get('filePath', '').startswith('/tmp/opencode/'):
                (private / ('retained-' + Path(inp['filePath']).name)).write_text(inp['content'])
        write_json(private / (ag['session'] + '-complete-access-index.json'), inventory)
        read_lines = 0; truncated_lines = 0
        for tool in tools:
            if tool['tool'] != 'read': continue
            path = Path(tool['input']['filePath'])
            if not path.is_file(): continue
            lines = path.read_text().splitlines()
            for line in tool['output'].splitlines():
                m = re.match(r'^(\d+): (.*)$', line)
                if not m: continue
                value = m[2]; original = lines[int(m[1])-1]
                if '... (line truncated to ' in value:
                    assert original.startswith(value.split('... (line truncated to ')[0]); truncated_lines += 1
                else: assert value == original, (ag['role'], path.name, m[1])
                read_lines += 1
        access.append({'read_output_lines_verified': read_lines, 'truncated_lines': truncated_lines, 'role': ag['role'], 'tools': len(tools), 'complete_access_index_sha256': sha(private / (ag['session'] + '-complete-access-index.json'))})
    assert all(sha(path) == h for path, h in p['frozen_identities'].items())
    a.update({'profiles': profiles, 'timelines': timelines, 'graph_native': graph, 'calibration': calibration,
              'tuning': tuning, 'telemetry': telemetry, 'diagnostic_events': diagnostics, 'binding_cost': trial['resources']['extra_collectors']['after']['binding_cost'], 'actual_delivery': deliveries,
              'complete_access_indexes': access,
              'classification': 'Rejected accepted-version continuation. Both candidate pairs slower than accepted parent. Full quality passed on original eager fallback; proposed concurrent Graph mechanism not reached. No communication acceptance.',
              'candidate_review': [
                  'Single execution-plan bundle replaces the accepted bundle; empty parameters and exactly one screen. Current/source remain accepted parent; no coexistence or publication.',
                  'Both run-only diagnostic jobs, both ranks: one RuntimeError from params.index(s) ambiguous Tensor boolean, propagated through _setup and replacement. Failure occurs before gradient buffers/capture/benchmark; two frames are one exception, not two failures.',
                  'Every candidate quality path is first parent_eager_baseline then fallback_setup_failure:RuntimeError; framework fallback dict empty does not erase internal fallback. Full zero-error quality validates reached fallback, not concurrent Graphs.',
                  'Fallback context.references invokes original eager source, not replaced accepted Graph bundle. Proposal zero-regression fallback-to-accepted claim contradicted by code and actual timing.',
                  'Proposed three streams/private gradient buffers/ordered merge plus eager DDP fourth backward retains 13 collectives by design. Unreached schedule is not native mechanism evidence; no new Nsys.',
                  'Selector read current timeline/calibration/tuning and prior telemetry. Synthetic calibration is not a hard floor; CPU gradient callbacks are not GPU-ready timestamps or unique bucket joins. Cross-profile clean-wall decomposition and fabricated profiler subtraction unsupported.',
                  'Candidate was written once, no edits; role only library API inspection (CUDA unavailable), syntax/AST/schema checks, no runtime CPU model/collective or GPU probe. Missing historical eliminated-call IDs were printed and retained.',
                  'Global stream override reset-to-False, limited identity/sentinel guards, rank-asymmetric setup/partial-step fallback and concurrent ownership remain unproved. No unique numerical cause or general bitwise equivalence.',
                  'All97 tools and recorded outputs reviewed. Supplied sandbox sources/history/current artifacts and own outputs; planner also read own private provider configuration/git metadata, retained server-only. No observed external later answers; filesystem nonhermetic.',
                  'Planner falsely classifies replacement as additive and requests binding_cost; actual collector returns not_applicable, no binding workers or cost estimate. Timeline attribution and balanced-other-capture exclusion claims are too strong.',
                  'Read-only audit initially assumed every numbered read line was complete; producer explicitly truncates long lines at2000 characters. Reader corrected to verify prefix and retain truncation, without any role/model rerun.',
                  'All tool processes reported success; candidate telemetry head pipeline does not establish full input consumption. No framework schema repairs. No model/worker retry or generated code edits during audit.'
              ],
              'source_scope': 'Prior fresh current-device full source/replica/negative checks remain assistance; no new source tensor comparison.',
              'independent': 'not_run: rejected; no new independent full-version validation justified.'})
    write_json(ROOT / 'terminal-audit.json', a)
    public = {k: a[k] for k in ['at', 'core', 'frozen_files', 'all_frozen_unchanged', 'source_current', 'current', 'active_bundles', 'candidate_sha256', 'proposal_sha256', 'trial_sha256', 'trial', 'decision', 'measurement', 'counts', 'classification', 'candidate_review', 'source_scope', 'independent']}
    public.update({'terminal_audit_sha256': sha(ROOT / 'terminal-audit.json'), 'graph_native': graph,
                   'agents': [{k: ag[k] for k in ['role', 'tools', 'assistant_messages', 'input_bytes', 'usage', 'contract_repairs']} for ag in a['agents']],
                   'endpoint_aggregates': [{'job': ep['job'], 'aggregates': ep['aggregates']} for ep in a['endpoints']],
                   'communication_profiles': profiles, 'actual_delivery': deliveries})
    write_json(ROOT / 'public-terminal-audit.json', public)
    print(json.dumps({'audit': sha(ROOT / 'terminal-audit.json'), 'public': sha(ROOT / 'public-terminal-audit.json'),
                      'bytes': (ROOT / 'public-terminal-audit.json').stat().st_size, 'graph': graph, 'profiles': profiles,
                      'diagnostics': [{'path': d['path'], 'events': [{k: e.get(k) for k in ['function', 'event', 'type', 'message', 'count']} for e in d['events']]} for d in diagnostics]}))


if __name__ == '__main__': main()
