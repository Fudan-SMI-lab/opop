"""Audit fixed generated policies and the separate real failed-feedback replay."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from kernel_optimizer.config import load_config
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.generated import validate_proposal
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_accepted_feedback_replay_20261028 import OLD, NEW
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha
from scripts.experiments.v6_audit_fsdp_search_20261028 import endpoint, agent_record
from scripts.experiments.v6_ddp_semantics_20261027 import disk_observation


def audit(root):
    terminal = read(root/'followup-completed.json')
    assert terminal['phase'] == 'completed'
    protocol = read(root/'protocol.json'); fixed = read(root/'fixed-protocol.json')
    results = read(root/'followup-results.json')
    search = ProjectSearch(load_config(root/'private-config.yaml', []), root/'fixed-validation')
    request = read(search.output/'request.json'); parent = search.revisions.current()
    assert parent.id == fixed['parent']
    assert sha(search.output/'request.json') == fixed['source_request_sha256']
    adapter = Path(search.prepared['controls_path']).parent/'reference-source/opop_adapter.py'
    assert sha(adapter) == fixed['adapter_sha256'] == search.prepared['adapter_sha256']
    for item in protocol['verified_source_references']:
        assert sha(item['path']) == item['sha256']
    report = {'at': datetime.now(timezone.utc).isoformat(), 'completed_at': terminal['finished_at'],
        'core': protocol['core'], 'terminal_sha256': sha(root/'followup-completed.json'),
        'unchanged_source_references': protocol['verified_source_references'],
        'unchanged_adapter_sha256': sha(adapter), 'request_sha256': sha(search.output/'request.json'),
        'current_pointer_unchanged': parent.id, 'fixed': {}, 'native': {}, 'disk': disk_observation(Path('/root/autodl-tmp'))}
    replay = read(Path(protocol['source_replay'])/'followup-results.json')
    parent_files = {f.path: f.sha256 for f in parent.files}
    for name in fixed['order']:
        run = read(root/(name+'.json')); row = results['fixed-gpu']['endpoints'][name]
        assert sha(row['path']) == row['sha256']
        revision = search.revisions.revision(row['identity']['revision'])
        files = {f.path: f.sha256 for f in revision.files}
        assert all(files.get(k) == v for k, v in parent_files.items())
        assert revision.contract_ref == parent.contract_ref
        parameters = None
        if name != 'inherited-k3':
            arm, depth = name.split('-d'); bundle = search.revisions.bundle(revision.bundles[0])
            assert len(revision.bundles) == 1
            parameters = {'keep_unsharded_blocks': 3, 'forward_prefetch_depth': int(depth)}
            assert bundle.parameters == parameters
            assert any(f.sha256 == replay[arm]['candidate_sha256'] for f in revision.files)
        checked = endpoint(run, request)
        constraints = constraints_for_runs([run], search.request.objective, search.prepared['evaluation'])
        assert constraints == row['constraints'] and all(c['passed'] for c in constraints)
        report['fixed'][name] = {'revision': revision.id, 'parameters': parameters,
            'constraints': constraints, 'run_sha256': sha(root/(name+'.json')), 'endpoint': checked}
        print(json.dumps({'audited': name, 'step_ms': checked['max_rank_then_median']['step_ms']['median']}), flush=True)
    for name in ['inherited-k3', 'control-d1', 'treatment-d1']:
        path = root/('native-'+name+'.json'); native = read(path); rows = []
        for rank in native['ranks']:
            db = sqlite3.connect('file:'+rank['evidence']+'?mode=ro', uri=True)
            counts = list(db.execute('select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id group by s.value'))
            db.close()
            assert sum(v[1] for v in counts) == len(rank['kernels'])
            nccl = [v for v in counts if 'nccl' in v[0].lower()]
            assert sum(v[1] for v in nccl) == rank['nccl_kernels']
            assert abs(sum(v[2] for v in nccl)/1e6-rank['nccl_kernel_sum_ms']) < 1e-8
            rows.append({'rank': rank['rank'], 'path': rank['evidence'], 'sha256': sha(rank['evidence']),
                'total_kernels': sum(v[1] for v in counts),
                'nccl': [{'name': n, 'count': c, 'sum_ms': t/1e6} for n,c,t in nccl]})
        report['native'][name] = {'path': str(path), 'sha256': sha(path), 'ranks': rows}
    role = results['failed-feedback-role']; agent = role['agent']
    original, prompt, files = load(Path(read(root/'failed-role-protocol.json')['snapshot']))
    saved, actual_prompt, actual_files = load(Path(agent['generation_snapshots'][0]))
    assert files == actual_files and actual_prompt == prompt.replace(OLD, NEW) and prompt.count(OLD) == 1
    assert all(saved[k] == original[k] for k in ['model', 'agent', 'tool_model_settings', 'framework_source'])
    assert saved['fresh_session']
    validate_proposal(Path(agent['sandbox']))
    candidate = Path(agent['sandbox'])/'candidate.py'
    assert sha(candidate) == role['candidate_sha256']
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    receipt = agent_record(agent, db)
    tool_details = []
    for ident, raw in db.execute('select id,data from part where session_id=? order by time_created,id', (agent['session'],)):
        part = json.loads(raw)
        if part.get('type') != 'tool': continue
        state = part.get('state', {}); inp = state.get('input', {}); output = str(state.get('output', ''))
        tool_details.append({'part': ident, 'tool': part['tool'], 'status': state.get('status'),
            'exit': state.get('metadata', {}).get('exit'), 'output_bytes': len(output.encode()),
            'output_sha256': hashlib.sha256(output.encode()).hexdigest(),
            'traceback_in_output': 'Traceback (most recent call last)' in output})
    db.close()
    report['failed_feedback_role'] = {'receipt': receipt, 'tools': tool_details,
        'input_count': len(files), 'all_initial_inputs_identical': True, 'only_prompt_change': {'old': OLD, 'new': NEW},
        'candidate': str(candidate), 'candidate_sha256': sha(candidate), 'proposal': role['proposal'],
        'gpu_validated': False,
        'review': 'Correctly identifies quality failure and replaces custom sharded-head arithmetic with parent delegation plus reshard policy. Claims of a uniquely identified numerical cause, wire-bound execution and a one-for-one exposed critical-path reduction are unsupported. This separate candidate was not executed by the five earlier fixed endpoints.'}
    report['counts'] = {'clean_rank_samples': sum(len(r['samples']) for v in report['fixed'].values() for r in v['endpoint']['ranks']),
        'quality_checks': sum(len(r['checks']) for v in report['fixed'].values() for r in v['endpoint']['ranks']),
        'raw_metric_traces': sum(len(r['traces']) for v in report['fixed'].values() for r in v['endpoint']['ranks'])}
    report['decision'] = {'production_prompt': 'unchanged; no demonstrated causal superiority',
        'sampler_and_context': 'unchanged; the earlier treatment read the existing parameter_trials=3',
        'formal_performance_acceptance': 'not performed',
        'limits': ['Fixed developer-selected shared parameters, not autonomous selection.',
            'Native collective counts alone do not prove prefetch scheduling or overlap. Use code and route evidence with raw traces.',
            'Each diagnostic metric has distinct invocations. NCCL duration includes waits; no wire/critical-path inference.',
            'All original samples and historical decisions remain unchanged.']}
    write_json(root/'terminal-audit.json', report)
    print(json.dumps({'audit': str(root/'terminal-audit.json'), 'counts': report['counts'],
        'role': agent['session'], 'repairs': agent['contract_repairs'],
        'tool_statuses': dict(Counter(t['status'] for t in tool_details))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
