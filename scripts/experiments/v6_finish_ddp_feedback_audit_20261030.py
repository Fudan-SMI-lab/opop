"""Reconcile actual saved DDP feedback trial evidence; no workload execution."""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import statistics

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
from scripts.experiments.v6_audit_ddp_communication_feedback_20261030 import digest

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-feedback-20261030')


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
    supplied = trial['resources']['extra_collectors']['after']['nccl_tuning']
    assert all(tuning[k] == supplied[k] for k in tuning)
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
        access.append({'role': ag['role'], 'tools': len(tools), 'complete_access_index_sha256': sha(private / (ag['session'] + '-complete-access-index.json'))})
    assert all(sha(path) == h for path, h in p['frozen_identities'].items())
    a.update({'profiles': profiles, 'timelines': timelines, 'graph_native': graph, 'calibration': calibration,
              'tuning': tuning, 'telemetry': telemetry, 'diagnostic_events': diagnostics, 'actual_delivery': deliveries,
              'complete_access_indexes': access,
              'classification': 'New ordinary search acceptance of compute-segment CUDA Graph scheduling. Full work/quality/constraints passed. Native steady invocation keeps 13 AllReduce/rank and logical payload; not demonstrated communication-volume reduction or independent acceptance.',
              'candidate_review': [
                  'One execution-plan bundle, empty parameters, one screen despite budget3; no coexistence/post-accept rediscovery proof. The post-accept profile does not replace original discovery.',
                  'Graph A zeroes stable gradient bucket views then runs three raw-model microbatches with static causal mask; fourth DDP backward remains eager; graph B applies frozen SGD add_ updates. First quality invocation parent, second setup+graph. No blanket claim that both quality invocations were optimized.',
                  'set_override_stale_capture_stream is process-global, set true around capture then false; does not restore arbitrary prior true state or certify concurrent compositions.',
                  'Setup warmups call real reducer collectives before agreement. Rank-local failures can mismatch collective sequences. Later runtime catches lack cross-rank agreement and may fallback after partial update; blanket state-neutral/no silent corruption guarantees unsupported.',
                  'Identity guards do not validate arbitrary tensor value changes, input replacement inside same list, optimizer lr changes, all aliases or numerical drift. Sentinel loss/four gradients only once; original full evaluator quality is authority.',
                  'No profile-union subtraction from clean wall as host/critical-path cost; no warmed calibration sum as physical wire lower bound or proof communication cannot help. Selector tool29 divides ns by1e3 but labels ms; inter-NCCL gap claim mixes units.',
                  'CPU Gloo2rank toy has2layers/hidden32/S16/vocab101, inline planned logic not exact candidate CUDA/Graph/framework dispatch. First test gradients/loss equal but bucket addresses false; later pointer and settled complete-step test passed. Original full GPU evaluation separately validates tested workload.',
                  'Agent removed own CPU scripts/__pycache__; source recovered from complete tool inputs, not all ephemeral raw outputs. Official PyTorch engine.cpp/input_buffer.cpp fetched; no observed old accepted/reviewer/developer terminal answer read in recorded paths/commands/output review. External FS nonhermetic.',
                  'Selector KeyError rank and missing source ls with pipeline0 retained. Candidate missing _recursive_to AttributeError, multiline sed failure and no-match grep pipeline retained. Planner initial walker defined but not called produced no output. No framework schema repairs.',
                  'Read-only audit first assumed absolute candidate write path and failed reconstruction; exact relative-path resolution corrected reader only. Initial failed log retained; no workload/probe rerun.'
              ],
              'source_scope': 'Prior fresh current-device full source/replica/negative checks remain assistance. No fresh source tensor comparison in this batch.',
              'independent': 'Necessary declared exact complete-version B/C/C/B is next; publish=False, source/current frozen, no new search/parameter selection/native trace required because exact candidate native exists.'})
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
