"""Additional raw profile and complete recorded-access audit; no execution rerun."""
from collections import Counter
import json
from pathlib import Path

from kernel_optimizer.project.communication import normalize
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha

ROOT = Path('/root/autodl-tmp/dev/v6-work/fsdp-communication-trial-20261030')


def main():
    a = read(ROOT / 'terminal-numerical-audit.json'); search = ROOT / 'fsdp-search'
    profiles = []; timeline = []
    for job in ['discover-b47113febf', 'trial-4c587f8ea9-evaluation-c4187d98-profile']:
        for rank in range(2):
            result = read(search / 'jobs' / job / 'run' / f'rank-{rank}.json')['result']
            data = read(result['discovery']); assert sha(data['trace']) == data['trace_sha256']
            calc = normalize(read(data['trace']), data, rank)
            for k in ['communication_kernel_count', 'communication_kernel_sum_ms', 'communication_union_ms',
                      'compute_overlap_ms', 'communication_without_compute_ms', 'logical_input_bytes']:
                assert abs(calc[k] - data['communication'][k]) < 1e-8, (job, rank, k)
            profiles.append({'job': job, 'rank': rank, 'path': result['discovery'], 'sha256': sha(result['discovery']),
                             'trace_sha256': data['trace_sha256'], 'calls': len(data['calls']), 'groups': len(data['groups']),
                             'kernels': len(data['kernels']), 'communication': {k: calc[k] for k in ['communication_kernel_count', 'logical_input_bytes', 'payload_coverage']},
                             'operations': dict(Counter(c['operation'] for c in calc['collectives']))})
    for p in (search / 'jobs/timeline-145e2a1aa7/run').glob('timeline-analysis-rank-*.json'):
        old = read(p); raw = read(old['trace']); assert sha(old['trace']) == old['trace_sha256']
        new = analyze(raw, rank=old['rank'], anchors=old['clock_anchors'], domain=old['clock']['domain'])
        for k in ['devices', 'clock', 'cuda_synchronization', 'workload_window']:
            assert new[k] == old[k], k
        for k in ['communication_kernel_count', 'logical_input_bytes', 'communication_union_ms']:
            assert new['communication'][k] == old['communication'][k]
        timeline.append({'rank': old['rank'], 'analysis_sha256': sha(p), 'trace_sha256': old['trace_sha256'],
                         'devices': {k: {f: v[f] for f in ['kernel_count', 'kernel_span_us', 'kernel_active_union_us', 'gaps_between_kernels_us']} for k, v in old['devices'].items()},
                         'clock': old['clock'], 'sync_api_count': len(old['cuda_synchronization']), 'profiled_wall_ms': old['profiled_wall_ms']})
    trial = read(search / 'trials/trial-4c587f8ea9/result.json'); collectors = []
    for phase, cs in trial['resources']['extra_collectors'].items():
        for name, c in cs.items():
            row = {'phase': phase, 'name': name, 'status': c['status']}
            if name == 'collective_calibration':
                assert c['status'] == 'unavailable' and not c['layouts'] and 'run' not in c
                row.update(omitted=len(c['omitted']), reasons=dict(Counter(x['reason'] for x in c['omitted'])), scope='No calibration worker ran; only full-world SUM allreduce layouts supported, while original discovery has AllGather/ReduceScatter.')
            collectors.append(row)
    raw_selection = read(Path(trial['selection_agent']['sandbox']) / 'selection.json')
    assert len(raw_selection['affected_calls']) == 43 and len(trial['selection']['affected_calls']) == 2259
    narrative = {
        'scope': 'One ordinary trial, no developer group/K/implementation or historical candidate supplied. Source-owner assistance persists. No independent acceptance rerun because the search rejected.',
        'mechanism': 'Agent keeps all FSDP units unsharded through backward and frees local shard storage after forward, reconstructing it from the all-gather slice before to_sharded. Persistent per-param methods/hooks remain installed; per-invocation mesh flags restored in finally.',
        'selection': 'Agent wrote 16 groups/43 direct affected IDs; existing framework expanded the execution-plan dependency closure to 2259. One project.invoke replacement, empty parameters, no requires/replaces, no published bundle or post-accept rediscovery.',
        'access_review': 'All four recorded role command/input/output inventories and retained inline/CPU scripts reviewed. Accesses were packaged current-task inputs, installed torch FSDP code, own generated outputs and explicitly written/copied /tmp/opencode/cpu_check files. No observed old optimized implementation, other-case/reviewer/auditor answer access. Filesystem was not hermetic; this is a recorded-access finding, not an absolute no-access proof. Large tool outputs and their truncation are retained server-side; file delivery does not mean full model consumption.',
        'cpu': 'Real two-process Gloo GPT2 toy, 2 layers/hidden32/vocab64/sequence8/SGD .01, CUDA hidden by project_agent_config; explicit own candidate copied/imported. Final two continuous candidate updates, two following parent steps, and stock recovery after a simulated mid-forward exception passed the recorded toy assertions. It bypasses real framework binding and is not original CUDA/full-work quality. post_backward hook field stayed None; explicit after-step storage assertions, not that hook, established the toy result.',
        'errors': ['Selector own schema check failed because affected_calls was omitted, then added 43 IDs inside generation; zero framework schema repairs.',
                   'Candidate wrong sandbox path/absent previous-candidate, target dict missing shape, two invalid fsdp_modules imports retained.',
                   'CPU fixture initially reset between purported continuous updates; loss update1 failed. Agent removed that reset.',
                   'First storage assertion found 0/3 freed: missing _sharded_post_forward_param_data attribute was swallowed by the safety guard. Agent changed candidate to getattr before rerunning its toy.',
                   'Immediate post-exception shard-healed assertion failed. Agent moved the assertion after two stock steps; final success establishes eventual toy recovery, not immediate restoration.',
                   'Several piped CPU commands returned shell0 despite traceback. Their outputs and final success markers are retained, not inferred from exit alone.',
                   'Read-only terminal audit first omitted binding_ablation endpoints from evaluate classification. Attempt1 retained; reader fixed, only raw-file CPU recomputation repeated. An earlier exploratory reader treated nsys-normalized list as dict, failed before mutation; no worker rerun.'],
        'limits': ['Full evaluator quality compares against fresh current-device source-verified controls; there was no fresh serial source comparison in this search.',
                   'Per-param free/copy/record_stream and partial-install exceptions can be swallowed; no universal rollback, alias-owner or concurrent lifetime guarantee follows from these endpoints.',
                   'Setup has Python installation/retained hooks; no-allocation/zero-overhead claims are unsupported. Runtime allocated peak decreases, but reserved memory increases; not universal memory improvement.',
                   'Selector/proposal 96% communication-active/74% exposed and ~12.5GB/s wire-bound claims mix periods or interpret waiting-inclusive kernel time. They are not adopted. Runtime kernel names are RING_LL, independent of NCCL tuning tables.',
                   'All 216 metric traces are distinct periods, not components to subtract across median fields. Separate native captures are not clean scores or critical-path measurements.',
                   'SHM/direct/direct comes from actual scoped NCCL channel log lines; NODE/P2P capability alone is not transport proof. No physical byte/critical-path conclusion.',
                   'Run-only candidate diagnostic has empty events on both ranks, no observe/compare. Silence is not universal no-fallback proof.',
                   'Binding diagnostic deltas are -0.058937 and +2.273778ms across separate workers, not physical negative overhead or causal group costs.',
                   'Search rejection and every slow sample retained; no independent, parameter retry, source pointer publication or production adoption. DDP is next.']}
    report = {'numerical_audit_sha256': sha(ROOT / 'terminal-numerical-audit.json'), 'profiles': profiles, 'timeline': timeline,
              'collectors': collectors, 'review': narrative, 'core': a['core'], 'counts': a['counts'],
              'decision': a['decision'], 'measurement': a['measurement'], 'candidate_sha256': a['candidate_sha256'],
              'proposal_sha256': a['proposal_sha256'], 'frozen_files': a['frozen_files'], 'all_frozen_unchanged': True}
    write_json(ROOT / 'terminal-audit.json', report)
    public = {**report, 'terminal_audit_sha256': sha(ROOT / 'terminal-audit.json'),
              'agents': [{k: x[k] for k in ['role', 'input_bytes', 'tools', 'assistant_messages', 'contract_repairs', 'history_sha256', 'code_edit_tools'] if k in x} for x in a['agents']],
              'endpoints': [{'job': e['job'], 'aggregates': e['aggregates']} for e in a['endpoints']],
              'native': [{k: n[k] for k in ['job', 'rank', 'sha256', 'kernels', 'nccl']} for n in a['native']]}
    write_json(ROOT / 'public-terminal-audit.json', public)
    print(json.dumps({'audit': public['terminal_audit_sha256'], 'public': sha(ROOT / 'public-terminal-audit.json'), 'bytes': (ROOT / 'public-terminal-audit.json').stat().st_size, 'profiles': profiles, 'timeline': timeline}))


if __name__ == '__main__': main()
