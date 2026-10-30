"""Recompute saved DDP evidence; never executes candidate/model/collective work."""
from collections import Counter
from pathlib import Path
import json
import statistics
import sqlite3

from kernel_optimizer.project.communication import normalize
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.project.nccl_tuning import read_logs
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

ROOT = Path('/root/autodl-tmp/dev/v6-work/ddp-communication-trial-20261030')


def main():
    a = read(ROOT / 'terminal-numerical-audit.json')
    search = ROOT / 'ddp-search'
    profiles = []; timelines = []; calibration = []; diagnostics = []
    for rank in range(2):
        p = search / 'jobs/discover-81e52aab57/run' / f'rank-{rank}.json'
        d = read(read(p)['result']['discovery'])
        assert sha(d['trace']) == d['trace_sha256']
        calc = normalize(read(d['trace']), d, rank)
        fields = ['communication_kernel_count', 'communication_kernel_sum_ms', 'communication_union_ms',
                  'compute_overlap_ms', 'communication_without_compute_ms', 'logical_input_bytes']
        for k in fields:
            assert abs(calc[k] - d['communication'][k]) < 1e-8, (rank, k)
        profiles.append({'rank': rank, 'trace_sha256': d['trace_sha256'],
                         'calls': len(d['calls']), 'groups': len(d['groups']), 'kernels': len(d['kernels']),
                         'communication': {k: calc[k] for k in fields},
                         'operations': dict(Counter(c['operation'] for c in calc['collectives'])),
                         'payload_sizes': dict(Counter(c['logical_input_bytes'] for c in calc['collectives']))})
        p = search / 'jobs/calibration-e80297d009/run' / f'rank-{rank}.json'
        c = read(p)['result']['calibration']; rows = []
        assert c['warmup'] == 2 and c['repeats'] == 5
        for row in c['rows']:
            assert row['quality'] and len(row['samples_ms']) == 5
            median = statistics.median(row['samples_ms'])
            assert median == row['median_completion_ms']
            assert abs(row['logical_bytes'] / (median / 1000) - row['logical_payload_bytes_per_second']) < 1e-4
            rows.append({k: row[k] for k in ['operation', 'reduction', 'dtype', 'elements', 'logical_bytes',
                                           'samples_ms', 'median_completion_ms', 'logical_payload_bytes_per_second']})
        calibration.append({'rank': rank, 'result_sha256': sha(p), 'rows': rows, 'scope': c['interpretation']})
    for p in sorted((search / 'jobs').glob('timeline-*/run/timeline-analysis-rank-*.json')):
        old = read(p); assert sha(old['trace']) == old['trace_sha256']
        new = analyze(read(old['trace']), rank=old['rank'], anchors=old['clock_anchors'], domain=old['clock']['domain'])
        for k in ['devices', 'clock', 'cuda_synchronization', 'workload_window']:
            assert new[k] == old[k], (str(p), k)
        for k in ['communication_kernel_count', 'logical_input_bytes', 'communication_union_ms',
                  'communication_kernel_sum_ms', 'compute_overlap_ms', 'communication_without_compute_ms']:
            assert abs(new['communication'][k] - old['communication'][k]) < 1e-8
        comm = old['communication']
        timelines.append({'job': p.parent.parent.name, 'rank': old['rank'], 'analysis_sha256': sha(p),
                          'trace_sha256': old['trace_sha256'],
                          'devices': {k: {f: v[f] for f in ['kernel_count', 'kernel_span_us', 'kernel_active_union_us',
                                      'gaps_between_kernels_us']} for k, v in old['devices'].items()},
                          'communication': {k: comm[k] for k in ['communication_kernel_count', 'logical_input_bytes',
                              'communication_kernel_sum_ms', 'communication_union_ms', 'compute_overlap_ms', 'communication_without_compute_ms']},
                          'operations': dict(Counter(c['operation'] for c in comm['collectives'])),
                          'kernel_names': dict(Counter(k['name'] for c in comm['collectives'] for k in c['kernels'])),
                          'sync_api_count': len(old['cuda_synchronization']), 'profiled_wall_ms': old['profiled_wall_ms']})
    for p in sorted((search / 'jobs').glob('candidate-diagnostics-*/run/candidate-diagnostics-rank-*.json')):
        d = read(p)
        diagnostics.append({'job': p.parent.parent.name, 'rank_file': p.name, 'sha256': sha(p),
                            'events': [{k: e[k] for k in ['phase', 'function', 'line', 'event', 'type', 'message', 'count']} for e in d['events']]})
    trial = read(search / 'trials/trial-df2eb7acd2/result.json')
    collectors = []
    for phase, cs in trial['resources']['extra_collectors'].items():
        for name, c in cs.items():
            row = {'phase': phase, 'name': name, 'status': c['status']}
            if name == 'nccl_tuning':
                for e in c['evidence']: assert sha(e['file']) == e['sha256']
                row['decisions'] = [{k: d[k] for k in ['operation', 'logical_bytes', 'algorithm', 'protocol', 'occurrences']} for d in c['decisions']]
                row['scope'] = c['scope']
            collectors.append(row)
    before = search / 'sandboxes/candidate-af38755ba266/inputs/observations/current'
    tuning = read_logs((search / 'jobs/nccl-tuning-fb97b18609/run').glob('nccl*.log'))
    supplied_tuning = read(before / 'nccl_tuning-f6198e01.json')
    assert all(tuning[k] == supplied_tuning[k] for k in tuning)
    tuning_summary = [{k: d[k] for k in ['operation', 'logical_bytes', 'algorithm', 'protocol', 'occurrences']} for d in tuning['decisions']]
    calibration_max = []
    for idx in range(2):
        series = [x['rows'][idx]['samples_ms'] for x in calibration]
        values = list(map(max, zip(*series, strict=True)))
        median = statistics.median(values)
        calibration_max.append({'logical_bytes': calibration[0]['rows'][idx]['logical_bytes'],
                                'aligned_rank_max_ms': values, 'median': median,
                                'MAD': statistics.median(abs(v - median) for v in values)})
    graph_rows = []
    for n in a['native']:
        con = sqlite3.connect('file:' + n['path'] + '?mode=ro', uri=True)
        graph_kernels = con.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null and graphNodeId != 0').fetchone()[0]
        con.close()
        assert graph_kernels == 0 and n['runtime_counts'].get('cudaGraphLaunch', 0) == 0
        graph_rows.append({'rank': n['rank'], 'scope': 'baseline Nsys only', 'graph_kernels': graph_kernels, 'cudaGraphLaunch': 0})
    loss = read(ROOT / 'existing-control-loss-read.json')
    for endpoint in a['endpoints']:
        for rank in endpoint['ranks']:
            failure = rank['checks'][1]
            first = next(x['loss'] for x in loss['losses'] if x['rank'] == rank['rank'] and x['locator'] == '/0/loss')
            second = next(x['loss'] for x in loss['losses'] if x['rank'] == rank['rank'] and x['locator'] == '/1/loss')
            # The original compare stores actual/expected in its mismatch object; preserve it verbatim below.
            assert abs(first - second) == failure['max_abs_error']
            assert f'Expected {second} but got {first}' in failure['error']
    review = {
        'decision': 'One empty-parameter candidate rejected before clean timing; no independent rerun or acceptance.',
        'mechanism': 'Whole-project replacement proposes raw-model accumulation into one 497759232-byte gradient buffer, one SUM AllReduce/divide-by-two and captured full SGD step. Static 4D causal mask avoids a Python scalar branch. Actual optimized graph was not validated.',
        'phase_state': 'Original evaluator resets once then performs two continuous quality run/observe/compare calls before warmups. Candidate calls its first invocation warmup_parent and lazily captures on the second. Its fallback copies state.initial parameters then calls parent. Both second-call losses exactly equal the saved first-control losses, consistent with replay of initial-state work. This code/order/scalar evidence is not a full per-parameter transition trace or unique cause of every internal failure.',
        'failure': 'Each separate run-only sidecar/rank records one capture RuntimeError about AccumulateGrad stale default-stream references propagated across accumulate_only/full_step/_setup/replacement, plus one ordinary False peer agreement. Four frames are not four independent failures. Run-only valid is not observe/compare quality. Screen itself records the fallback label/RuntimeError type, not the full exception text.',
        'quality_scope': 'First continuous full check passed 248879620 elements exactly per rank; second compare stops at output.loss. Adapter.observe had already copied gradients/parameters and completed exact cross-rank gradient checks and its negative before compare. Those executed observations are not saved full actual tensors and were not compared after the first loss failure.',
        'access': 'Reviewed all four recorded command/input/output inventories and inline scripts. Reads concern packaged current-task source, observations, contract and installed torch/transformers libraries, plus own outputs. Shared /tmp/opencode was used for inline CPU toys and eliminated.json written by this role; no recorded import of old case code, accepted implementations or auditor/reviewer answers observed. Filesystem is nonhermetic; no absolute no-access claim. Truncated outputs and full recorded parts remain server-side; delivery is not full reading.',
        'cpu': 'CPU GPT2 one/two-layer hidden32 toys inspect scalar ops and static masks. Final one-step inline body simulation uses world=1 identity averaging and no real candidate import/binding/capture/NCCL/continuous two-step relation. Exact candidate build probes only unsupported_parent_path with CUDA hidden. These are not original GPU quality.',
        'errors': ['Candidate invalid transformers module paths and two invalid conditional imports retained; compound exit0 hid earlier SyntaxErrors.',
                   'Missing CUDA header and several empty grep results retained. CPU profiler queried GPU count and returned code100 while CPU work continued; no actual GPU model tensor execution recorded in role probes.',
                   'First flat-gradient toy allocated7 elements for15 and failed (stderr suppressed first time); next toy incorrectly accumulated reference gradients and printed False. Agent fixed the toy reference zeroing, then got True.',
                   'After-planner rg unavailable/127 and pipeline0 retained. No framework schema repairs.',
                   'Terminal reader attempts1/2 assumed optional pairs/diagnostics fields on an early invalid result; logs preserved and reader corrected. Exploratory reader used evaluate.py instead of evaluation.py. No model/candidate/collective/source rerun.'],
        'limits': ['Selector initially intended Graph with13 bucket reductions; candidate independently chose one flat reduction. No manual group/bucket/algorithm supplied.',
                   'Sentinel tolerance1e-2/1e-3 on three small gradients and loss is weaker than original quality and compares against first invocation; it is not an acceptance gate. Missing gradients may be skipped; fallback restores only initial parameters and suppresses copy errors, not arbitrary incoming state/optimizer/reducer/owner rollback.',
                   'Whole success path has three peer-agreement collectives. Later graph exception fallback lacks peer consensus. Retained Work handles and graph/flat buffers persist; no universal lifetime/concurrency or zero-memory-overhead claim.',
                   'NCCL tuning records cover setup and warmed calls; selected RING/SIMPLE entries are not an event join to the profiled RING_LL kernel names. Actual channel lines corroborate SHM/direct/direct for that worker only.',
                   'Calibration only measures observed28MB/157MB layouts, not the proposed497MB buffer. Logical payload rate is not wire capacity or bus bandwidth. NCCL sums include wait, and separate-profile busy/gap cannot be subtracted from clean wall as CPU overhead/critical path.',
                   'Candidate timeline follows reset warmups and profiles the fallback parent route; it cannot certify accelerated communication or the failed continuous relation. No candidate Nsys exists.',
                   'No formal clean samples, post-quality metric captures, constraint/noise pairs, independent acceptance, source publication or production adoption. Installation timings/allocator fields exist despite absence of clean performance. Historical outcomes stay frozen.']}
    report = {'numerical_audit_sha256': sha(ROOT / 'terminal-numerical-audit.json'), 'core': a['core'],
              'candidate_sha256': a['candidate_sha256'], 'proposal_sha256': a['proposal_sha256'],
              'trial_sha256': a['trial_sha256'], 'decision': a['decision'], 'counts': a['counts'],
              'frozen_files': a['frozen_files'], 'all_frozen_unchanged': True,
              'profiles': profiles, 'timelines': timelines, 'calibration': calibration, 'diagnostics': diagnostics,
              'collectors': collectors, 'tuning_decisions': tuning_summary, 'baseline_graph_rows': graph_rows,
              'calibration_aligned_rank_max': calibration_max,
              'existing_control_loss_read_sha256': sha(ROOT / 'existing-control-loss-read.json'),
              'losses': loss['losses'], 'review': review}
    write_json(ROOT / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k not in ['diagnostics']}
    public.update(terminal_audit_sha256=sha(ROOT / 'terminal-audit.json'),
                  agents=[{k: x[k] for k in ['role', 'input_bytes', 'tools', 'assistant_messages', 'contract_repairs', 'history_sha256', 'code_edit_tools', 'usage'] if k in x} for x in a['agents']],
                  endpoints=[{'job': e['job'], 'ranks': [{'rank': z['rank'], 'checks': z['checks'], 'setup': z['setup']} for z in e['ranks']]} for e in a['endpoints']],
                  native=[{k: n[k] for k in ['job', 'rank', 'sha256', 'kernels', 'nccl']} for n in a['native']])
    write_json(ROOT / 'public-terminal-audit.json', public)
    print(json.dumps({'audit': public['terminal_audit_sha256'], 'public': sha(ROOT / 'public-terminal-audit.json'),
                      'bytes': (ROOT / 'public-terminal-audit.json').stat().st_size,
                      'profiles': profiles, 'timelines': timelines, 'calibration': calibration}))


if __name__ == '__main__': main()
