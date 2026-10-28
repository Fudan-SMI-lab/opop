"""Audit the frozen original-entry Qwen search, including failures and telemetry scope."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from kernel_optimizer.config import load_config
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import agent_record
from scripts.experiments.v6_audit_qwen_end_to_end_20261029 import endpoint
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def audit(root):
    terminal = read(root / 'followup-completed.json'); assert terminal['phase'] == 'completed'
    config = load_config(root / 'private-config.yaml', [])
    s = ProjectSearch(config, root / 'qwen-original'); core = read(root / 'protocol.json')['core']
    reuse = read(root / 'preparation-reuse.json')
    baseline = s.revisions.revision(s.prepared['revision'])
    assert not s.revisions.current().bundles and not baseline.bundles
    assert baseline.id == reuse['revision'] and sha(s.output / 'request.json') == reuse['request_sha256']
    files = {f.path: f.sha256 for f in baseline.files}; assert files == reuse['files']
    for f in baseline.files:
        assert hashlib.sha256(s.store.get_artifact(f.content_ref)).hexdigest() == f.sha256
    source_check = read(root / 'reviewed-source-check.json'); assert valid_run(source_check)
    check = source_check['ranks'][0]['result']
    prior = root.parent / 'qwen-end-to-end-original-20261029'
    reference = read(prior / 'protocol.json')['independent_source_reference']
    assert sha(reference['path']) == check['source_reference_sha256']
    assert sha(Path(s.prepared['controls_path']) / 'reference-rank-0.pt') == check['controls_reference_sha256']
    parent = ProjectSearch(config, prior / 'qwen-original')
    assert parent.revisions.current().id == baseline.id and parent.prepared == s.prepared
    events = [json.loads(line) for line in (s.output / 'events.jsonl').read_text().splitlines()]
    agents = [e['payload'] for e in events if e['type'] == 'PROJECT_AGENT_FINISHED']
    order = [a['session'] for a in agents if a['role'] == 'candidate']
    trials = sorted([read(p) for p in (s.output / 'trials').glob('*/result.json')],
                    key=lambda t: order.index(t['candidate_agent']['session']))
    result = {'at': datetime.now(timezone.utc).isoformat(), 'terminal': terminal, 'core': core,
        'source_fidelity': source_check, 'reuse': reuse, 'source_files': files,
        'parent_pointer_unchanged': parent.revisions.current().id,
        'current': {'revision': s.revisions.current().id, 'bundles': []},
        'events': dict(Counter(e['type'] for e in events)), 'trials': [], 'agents': [], 'diagnostics': [], 'jobs': []}
    successful = []
    for t in trials:
        a = t['candidate_agent']; code = Path(a['sandbox']) / 'candidate.py'
        row = {'trial': t['trial'], 'decision': t['decision'],
            'selection': {k: t['selection'][k] for k in ['scope', 'groups', 'rationale', 'hypothesis']},
            'affected_calls_count': len(t['selection']['affected_calls']),
            'code': str(code), 'code_sha256': sha(code), 'code_bytes': code.stat().st_size,
            'proposal': {k: t['proposal'].get(k) for k in ['rationale', 'setup_work', 'quality_risks', 'replacement_calls']},
            'parameters': [], 'pair_count': len(t.get('pairs', []))}
        for o in t['parameter_search']['observations']:
            run = o['run']; pr = {'parameters': o['bundle']['parameters'], 'bundle': o['bundle']['id'],
                                  'status': run['status'], 'result': o['result']}
            if valid_run(run):
                pr['endpoint'] = endpoint(run, core); successful.append(pr['endpoint'])
            else:
                pr['errors'] = [{k: r.get(k) for k in ['pid', 'status', 'error_type', 'error', 'traceback', 'framework_source']}
                                for r in run['ranks']]
                assert all(r['framework_source'] == core for r in run['ranks'])
            row['parameters'].append(pr)
        if t.get('pairs'):
            assert paired_decision(t['pairs'], s.request.objective, s.prepared['evaluation']) == t['measurement']
            row['measurement'] = t['measurement']
            row['pairs'] = [{key: endpoint(run, core) for key, run in pair.items()} for pair in t['pairs']]
            successful.extend(ep for pair in row['pairs'] for ep in pair.values())
        result['trials'].append(row)
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    for a in agents:
        row = agent_record(a, db); m, prompt, inputs = load(Path(a['generation_snapshots'][0]))
        assert m['fresh_session'] and m['framework_source'] == core
        row['input_bytes_total'] = sum(len(v) for v in inputs.values())
        row['tool_output_bytes_observed'] = 0; row['old_answer_tool_mentions'] = []
        for ident, raw in db.execute('select id,data from part where session_id=? order by time_created,id', (a['session'],)):
            p = json.loads(raw)
            if p.get('type') != 'tool': continue
            state = p.get('state', {}); inp = json.dumps(state.get('input', {}))
            row['tool_output_bytes_observed'] += len(str(state.get('output', '')).encode())
            hits = [x for x in ['qwen-end-to-end-20261028', 'qwen-goal-contrast-20261028', 'metric-mapping-decode-20261028',
                              'trial-f71b5c1677', 'reviewed-source-check.json'] if x in inp]
            if hits: row['old_answer_tool_mentions'].append({'part': ident, 'matched': hits})
        result['agents'].append(row)
    db.close()
    for path in sorted((s.output / 'jobs').glob('*/run/rank-0-started.json')):
        started = read(path); execution = read(path.parent / 'request.json'); source = path.parents[1] / 'source'
        rank_record = read(path.parent / 'rank-0.json')
        assert rank_record['framework_source'] == core
        actual = started['revision']['files']
        assert all(actual.get(k) == v for k, v in files.items())
        assert all(sha(source / k) == v for k, v in actual.items())
        result['jobs'].append({'path': str(path), 'sha256': sha(path), 'revision': started['revision']['revision'],
                               'action': execution['parameters']['action'], 'source_files': actual})
        if execution['parameters']['action'] == 'candidate_diagnostics':
            rank_file = path.parent / 'rank-0.json'
            result['diagnostics'].append({'request': str(path.parent / 'request.json'), 'result_path': str(rank_file),
                'worker_status': rank_record['status'], 'result': rank_record.get('result'),
                'scope_from_core': 'candidate_diagnostics invokes tracked run() but never observe() or compare(); valid=true is execution/coverage, not full quality.'})
    result['counts'] = {'search_rounds': len(trials), 'real_roles': len(agents),
                        'clean_endpoints': len(successful), 'clean_samples': sum(len(ep['samples']) for ep in successful),
                        'full_evaluate_checks': sum(len(ep['checks']) for ep in successful)}
    feedback = s.output / 'latest-feedback.json'
    result['feedback_size'] = {'path': str(feedback), 'bytes': feedback.stat().st_size, 'sha256': sha(feedback),
        'scope': 'File size is not model context size. Per-role snapshot/input/tool-output sizes are observations, not complete token delivery accounting.'}
    result['review'] = {
        'first_two': 'Worker AssertionErrors inside observe teacher-forced logits relation before baseline compare returns; no clean samples or scored quality endpoints. Same error fingerprint is not proof of shared-pool cause, aliasing cause, or which side first diverged.',
        'third': 'Internal graph-vs-parent gate disables graph, then parent route predominates. Quality passes; screen feasible, paired constraint/noise fails. Empty fallbacks dict does not mean no reference route. No accepted active bundle, independent/native not_run.',
        'agent_claims_not_adopted': ['Subtracting profiled CUDA totals from clean wall does not isolate host dispatch cost.',
            'Leaving prefill code unchanged does not guarantee TTFT constraint.',
            'candidate_diagnostics success does not prove full quality or a nondeterministic passing rerun.',
            'A teacher replay assertion before baseline compare does not prove greedy capture matched the independent source.'],
        'next': 'Fixed quality-only endpoints with unobserved controls and observed greedy/teacher full outputs; preserve candidate bytes, request and references. This fills a demonstrated missing observation before any prompt or candidate change.'}
    result['disk'] = disk_observation(Path('/root/autodl-tmp'))
    write_json(root / 'terminal-audit.json', result)
    print(json.dumps({'audit': str(root / 'terminal-audit.json'), 'bytes': (root / 'terminal-audit.json').stat().st_size,
        'counts': result['counts'], 'repair_counts': [a['contract_repairs'] for a in result['agents']],
        'old_mentions': sum(len(a['old_answer_tool_mentions']) for a in result['agents'])}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True); audit(p.parse_args().root)
