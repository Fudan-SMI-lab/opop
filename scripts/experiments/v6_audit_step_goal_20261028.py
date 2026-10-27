"""Audit the frozen DDP goal variant and the real one-file evidence-index replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from kernel_optimizer.config import load_config
from kernel_optimizer.models.project import ProjectContract
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.measurement import paired_decision, valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def endpoints(pairs):
    rows = []
    for i, pair in enumerate(pairs):
        for arm, endpoint in pair.items():
            assert valid_run(endpoint)
            ranks = []
            for rank in endpoint['ranks']:
                value = rank['result']; assert value['math_policy_unchanged']
                assert len(value['checks']) == 2 and all(c['passed'] for c in value['checks'])
                assert len(value['samples']) == 5
                coverage = []
                for ident, c in value['coverage'].items():
                    assert not c['fallbacks'] and all(c['counts'].get(k, 0) > 0 for k in c['required'])
                    coverage.append({'bundle': ident, 'required_count': len(c['required']), 'fallbacks': c['fallbacks'],
                        'routes': c.get('candidate_declared_routes'), 'references': c.get('reference_invocations')})
                ranks.append({'rank': rank['rank'], 'checks': value['checks'], 'coverage': coverage,
                    'step_samples_ms': [s['step_ms'] for s in value['samples']], 'metrics': value['metrics'],
                    'metric_samples': value['metric_samples'], 'bundle_hashes': value['bundle_hashes'],
                    'framework_source': rank['framework_source']})
            rows.append({'pair': i, 'arm': arm, 'ranks': ranks})
    return rows


def audit(root, out):
    report = {'terminal': read(root/'followup-completed.json')['phase'], 'reviews': [], 'trials': []}
    initial = root.parent/'memory-snapshot-validation-20261028/measurement-review/result.json'
    original, prompt, inputs = load(Path(read(initial)['agent']['generation_snapshots'][0]))
    db = Path('/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db')
    conn = sqlite3.connect('file:'+str(db)+'?mode=ro', uri=True)
    for row in read(root/'review-ab/result.json')['runs']:
        m, p, files = load(Path(row['agent']['generation_snapshots'][0]))
        changed = [k for k in set(inputs)|set(files) if inputs.get(k) != files.get(k)]
        assert changed == ([] if row['arm'] == 'control' else ['inputs/ddp-index.json'])
        assert p == prompt and all(m[k] == original[k] for k in ['model','agent','tool_model_settings','framework_source'])
        if row['arm'] == 'treatment':
            new = json.loads(files['inputs/ddp-index.json']); old = json.loads(inputs['inputs/ddp-index.json'])
            for e in new['endpoints']:
                for artifact in e.pop('passive_gc_artifacts'): assert sha(artifact['path']) == artifact['sha256']
            assert new['endpoints'] == old
        tools = []
        for ident, data in conn.execute('select id,data from part where session_id=? order by time_created,id', (row['agent']['session'],)):
            part = json.loads(data)
            if part.get('type') != 'tool': continue
            state = part.get('state', {}); value = state.get('input', {})
            serial = json.dumps(value, sort_keys=True)
            tools.append({'part': ident, 'tool': part['tool'], 'status': state.get('status'),
                'input_sha256': hashlib.sha256(serial.encode()).hexdigest(),
                'read_path': value.get('filePath') if part['tool']=='read' else None,
                'gc_artifact_retrieval_mentioned': part['tool'] in ['read','bash'] and 'gc-passive-rank-' in serial})
        drafts = []
        for snapshot in row['agent']['generation_snapshots'][1:]:
            rm, rp, rf = load(Path(snapshot))
            drafts.append({'snapshot': snapshot, 'prompt': rp, 'history_status': rm['history_status'],
                           'draft_keys': list(json.loads(rf['review.json'])) if 'review.json' in rf else []})
        report['reviews'].append({'arm': row['arm'], 'session': row['agent']['session'], 'initial_changed_files': changed,
            'initial_input_bytes': sum(map(len, files.values())), 'same_prompt_model_tools_core': True,
            'contract_repairs': row['agent']['contract_repairs'], 'repair_snapshots': drafts,
            'review': row['review'], 'review_sha256': sha(row['review']), 'tools': tools,
            'gc_artifact_retrieval_mentioned': any(x['gc_artifact_retrieval_mentioned'] for x in tools),
            'usage': row['agent']['usage']})
    conn.close()
    run = root/'ddp-step'; request = read(run/'request.json'); parent = root.parent/'ddp-semantics-20261027/ddp-communication'
    old_request = read(parent/'request.json')
    assert {k for k in request if request[k] != old_request[k]} == {'goal','objective','notes','devices'}
    assert request['objective']['constraints'] == old_request['objective']['constraints']
    config = load_config(root/'private-config.yaml', []); search = ProjectSearch(config, run)
    baseline = search.revisions.revision(search.prepared['revision']); current = search.revisions.current()
    assert not baseline.bundles and len(current.bundles) == 1
    assert search.prepared['adapter_sha256'] == read(parent/'prepared.json')['adapter_sha256']
    assert sha(Path(search.prepared['controls_path']).parent/'reference-source/opop_adapter.py') == search.prepared['adapter_sha256']
    contract = search.revisions.load(current.contract_ref, ProjectContract)
    report['contract'] = {'reference': current.contract_ref, 'invariants': contract.invariants,
        'request_sha256': sha(run/'request.json'), 'objective': request['objective'],
        'same_workload_quality_source_measurement': True, 'adapter_sha256': search.prepared['adapter_sha256']}
    report['controls'] = [{'rank': r['rank'], 'valid': r['result']['valid'], 'controls': r['result']['controls']}
                          for r in search.prepared['evaluation']['ranks']]
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    report['event_counts'] = dict(Counter(e['type'] for e in events))
    role_order = [e['payload']['session'] for e in events if e['type']=='PROJECT_AGENT_FINISHED' and e['payload']['role']=='candidate']
    trials = [read(p) for p in (run/'trials').glob('*/result.json')]
    trials.sort(key=lambda t: role_order.index(t['candidate_agent']['session']))
    assert len(trials) == 3
    for t in trials:
        selection = t['selection']; snap = Path(t['selection_agent']['generation_snapshots'][0]); sm, sp, sf = load(snap)
        feedback_files = {k: {'bytes': len(v), 'sha256': hashlib.sha256(v).hexdigest()} for k,v in sf.items() if 'feedback' in k}
        candidate_path = Path(t['candidate_agent']['sandbox'])/'candidate.py'
        report['trials'].append({'trial': t['trial'], 'decision': t['decision'], 'measurement': t.get('measurement'),
            'scope': selection['scope'], 'groups': selection['groups'], 'affected_count': len(selection['affected_calls']),
            'rationale': selection['rationale'], 'hypothesis': selection['hypothesis'],
            'selection_snapshot': str(snap), 'feedback_inputs': feedback_files,
            'candidate': str(candidate_path), 'candidate_sha256': sha(candidate_path),
            'candidate_session': t['candidate_agent']['session'], 'candidate_repairs': t['candidate_agent']['contract_repairs'],
            'candidate_usage': t['candidate_agent']['usage'], 'paired_endpoints': endpoints(t['pairs']) if t.get('pairs') else []})
    report['current'] = {'revision': current.id, 'bundles': list(current.bundles),
        'post_acceptance_rediscovery': any(e['type']=='PROJECT_DISCOVERED' and e['payload']['parent']==current.id for e in events),
        'composition': 'one new bundle; no old optimized bundles imported'}
    p = next((root/'ddp-step-independent/version-trials').glob('*/result.json')); independent = read(p)
    assert paired_decision(independent['pairs'], request['objective'], search.prepared['evaluation']) == independent['measurement']
    report['independent'] = {'path': str(p), 'sha256': sha(p), 'measurement': independent['measurement'],
        'decision': independent['decision'], 'endpoints': endpoints(independent['pairs'])}
    extra = read(root/'independent-result.json')
    report['source_fidelity'] = {'valid': extra['source_fidelity_valid'],
        'rank_results': [r['result'] for r in extra['source_fidelity']['ranks']],
        'scope': 'Independent original serial trajectory vs baseline control references; optimized outputs separately compared to those references by every frozen evaluator endpoint.'}
    report['negative_probe'] = {k:v for k,v in extra['negative_probe'].items() if k!='run'}
    report['negative_probe']['rank_results'] = [r['result'] for r in extra['negative_probe']['run']['ranks']]
    report['cross_goals'] = read(root/'cross-goals.json')
    report['scope'] = 'Frozen evidence audit. All old results preserved. Missing step-goal communication diagnostics remain missing; no inferred values.'
    write_json(out, report)
    print(json.dumps({'audit': str(out), 'bytes': out.stat().st_size, 'accepted': independent['decision']['accepted'],
        'step_reduction_percent': 100*(1-independent['measurement']['objective']/independent['measurement']['reference'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); audit(args.root, args.output)
