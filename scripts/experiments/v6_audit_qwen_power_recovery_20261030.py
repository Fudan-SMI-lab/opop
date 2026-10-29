"""Read-only terminal audit; full session records remain private on the server."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats
import sqlite3
import subprocess

from kernel_optimizer.config import load_config
from kernel_optimizer.project.generation_snapshot import load
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.project.timeline import analyze
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def main(root):
    audit = read(root / 'recovery-audit.json')
    old = Path(audit['old_root']); branch = Path(audit['branch'])
    done = read(root / 'followup-completed.json')
    assert done['phase'] == 'completed_with_errors'
    assert done['results']['independent']['status'] == 'not_run'
    assert not (old / 'followup-completed.json').exists()
    assert all(sha(p) == h for p, h in audit['preserved_files'].items())
    assert all(sha(p) == h for p, h in audit['input_hashes'].items())
    config = load_config(root / 'private-config.yaml', [])
    search = ProjectSearch(config, branch)
    assert framework_source() == audit['core']
    assert search.revisions.current().id == audit['current']
    assert list(search.revisions.current().bundles) == audit['bundles']
    assert sha(audit['completed_trial']['path']) == audit['completed_trial']['sha256']
    private = root / 'terminal-private-history'; private.mkdir(exist_ok=False)
    db = sqlite3.connect('file:/root/autodl-tmp/dev/v6-work/private-runtime/data/opencode/opencode.db?mode=ro', uri=True)
    roles = []; snapshots = {}; sessions = {}
    for path in sorted((branch / 'generation-snapshots').glob('*.json')):
        meta, prompt, files = load(path); snapshots[meta['id']] = (meta, prompt, files)
        assert meta['framework_source'] == framework_source()
        actual = Path(meta['sandbox'])
        assert all((actual / n).read_bytes() == b for n, b in files.items())
        roles.append({'id': meta['id'], 'role': meta['role'], 'session': meta['session'],
            'fresh': meta['fresh_session'], 'inputs': len(files), 'snapshot_sha256': sha(path),
            'prompt_sha256': digest(prompt), 'input_hashes': {n: digest(b) for n, b in files.items()},
            'settings_sha256': digest(json.dumps(meta['tool_model_settings'], sort_keys=True))})
        if meta['session'] in sessions: continue
        sid = meta['session']; full = {}
        for table in ['message', 'part']:
            full[table] = [{'id': i, 'data': json.loads(d)} for i, d in db.execute(
                f'select id,data from {table} where session_id=? order by time_created,id', (sid,))]
        write_json(private / (sid + '.json'), full)
        tools = [p for p in full['part'] if p['data'].get('type') == 'tool']
        assistants = [m for m in full['message'] if m['data'].get('role') == 'assistant']
        manifest = []
        for p in tools:
            d = p['data']; state = d.get('state', {}); inp = state.get('input', {})
            # Commands and outputs, including inline probes and initial write/edit bytes, are retained privately.
            manifest.append({'part': p['id'], 'tool': d['tool'], 'status': state.get('status'),
                'input': inp, 'output': state.get('output'), 'metadata': state.get('metadata')})
        write_json(private / (sid + '-tools.json'), manifest)
        sessions[sid] = {'role': meta['role'], 'assistant_messages': len(assistants),
            'tools': len(tools), 'tool_kinds': dict(Counter(p['data']['tool'] for p in tools)),
            'provider_estimate': sum(m['data'].get('cost', 0) or 0 for m in assistants),
            'history_sha256': sha(private / (sid + '.json')),
            'tool_manifest_sha256': sha(private / (sid + '-tools.json'))}
    before, resumed, candidate = [snapshots[n] for n in ['generation-8d3438a88a77', 'generation-b58ccb84d0a8', 'generation-cb08f1ca8ed2']]
    assert before[0]['session'] == resumed[0]['session'] and not resumed[0]['fresh_session']
    assert before[2] == resumed[2] and before[0]['tool_model_settings'] == resumed[0]['tool_model_settings']
    assert candidate[2]['inputs/previous-feedback.txt'] == resumed[2]['inputs/previous-feedback.txt']
    assert digest(candidate[2]['inputs/previous-candidate/candidate.py']) == '4857a625d65effac222c9cf200a1b7ceff7750512bb14eadc1a6f12e70093fcd'
    jobs = []; traces = {}; old_device = search.request.devices
    for receipt in sorted(branch.glob('jobs/*/run/receipt.json')):
        value = read(receipt); request = read(receipt.parent / 'request.json')
        assert request['devices'] == old_device
        assert sha(request['entry']) == value['entry_sha256']
        assert sha(request['parameters']['adapter']) == search.prepared['adapter_sha256']
        rank = value['ranks'][0]
        assert rank['framework_source'] == framework_source() and value['worker_pids_reaped']
        assert rank['binding']['torch_uuid'] == old_device[0]
        result = rank.get('result', {})
        assert not result.get('samples') and not result.get('checks')
        bundles = request['parameters'].get('bundles', [])
        jobs.append({'job': receipt.parent.parent.name, 'action': request['parameters']['action'],
            'status': value['status'], 'valid': result.get('valid'), 'error': rank.get('error'),
            'entry_sha256': value['entry_sha256'], 'request_sha256': sha(receipt.parent / 'request.json'),
            'receipt_sha256': sha(receipt), 'bundles': [{'id': b['id'], 'code_sha256': sha(b['file'])} for b in bundles],
            'quality_checks': 0, 'clean_samples': 0, 'coverage': result.get('coverage')})
        if result.get('reports'):
            p = Path(result['reports'][0]); report = read(p)
            assert sha(report['trace']) == report['trace_sha256']
            recalculated = analyze(read(report['trace']), rank=0, anchors=report['clock_anchors'], domain=report['clock']['domain'])
            for key in ['devices', 'cuda_synchronization', 'clock']:
                assert recalculated[key] == report[key]
            traces['timeline'] = {'sha256': sha(p), 'trace_sha256': report['trace_sha256'],
                'devices': {k: {n: v[n] for n in ['kernel_count', 'kernel_span_us', 'kernel_active_union_us', 'gaps_between_kernels_us']} for k, v in report['devices'].items()},
                'scope': 'Recomputed separate perturbed timeline. No clean-wall subtraction or exclusive critical-path attribution.'}
        if request['parameters']['action'] == 'profile':
            raw = receipt.parent / 'trace-rank-0.json'
            events = read(raw)['traceEvents']; counts = Counter(e.get('cat') for e in events)
            assert counts['kernel'] == result['coverage']['cuda_kernels']
            traces['discovery'] = {'raw_sha256': sha(raw), 'categories': dict(counts),
                'discovery_sha256': sha(result['discovery']), 'call_sites': result['call_sites'], 'groups': result['groups']}
        if request['parameters']['action'] == 'python_runtime_profile':
            p = receipt.parent / 'python-rank-0.pstats'; stats = pstats.Stats(str(p))
            traces['python'] = {'pstats_sha256': sha(p), 'report_sha256': sha(receipt.parent / 'python-rank-0.json'),
                'total_calls': stats.total_calls, 'primitive_calls': stats.prim_calls, 'total_tt': stats.total_tt,
                'scope': 'Raw cProfile totals inspected; perturbed diagnostic, not clean host cost.'}
    assert len(jobs) == 6 and sum(j['status'] == 'failed' for j in jobs) == 3
    second = search.revisions.revision('f211410ebc62bba2546b3e6cc2dee2cff9a6085a3cfc802dce4231c4c8000511')
    assert len(second.bundles) == 1 and second.bundles[0] != 'fixed-index-read-qwen'
    bundle = search.revisions.bundle(second.bundles[0])
    assert bundle.parameters == {} and list(bundle.replaces) == ['fixed-index-read-qwen']
    code = next(f for f in second.files if f.path == bundle.installer)
    assert code.sha256 == 'f5b76ebde100c039d8ad9264e316dbade5acdb74c02f40308e958673e802be7e'
    blocked = branch / 'jobs/trial-recovered-4f3d7f11aa-param-2eb959a5-screen/run'
    assert not list(blocked.glob('*'))
    assert not (branch / 'trials/trial-recovered-4f3d7f11aa/result.json').exists()
    current_gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    assert old_device[0] not in current_gpus
    report = {'at': datetime.now(timezone.utc).isoformat(), 'core': framework_source(),
        'terminal': done['phase'], 'finished_at': done['finished_at'], 'frozen_files': len(audit['preserved_files']),
        'sessions': sessions, 'snapshots': roles, 'jobs': jobs, 'raw_diagnostics': traces,
        'routing': {'resumed_selector_initial_files_equal': True, 'candidate_feedback_byte_equal_selector': True,
            'scope': 'Existing resume_generation passes selector feedback through unchanged to candidate; unlike ordinary trial it does not call the selected-group route again. Same persisted selector, only previously unfinished second trial.'},
        'second_candidate': {'revision': second.id, 'bundle': second.bundles[0], 'code_sha256': code.sha256,
            'parameters': {}, 'gpu_worker_started': False, 'quality': 'not_run', 'search_acceptance': 'none', 'independent': 'not_run'},
        'hardware': {'old_devices': old_device, 'current_inventory': current_gpus,
            'gap': 'Recovery preflight checked availability but omitted exact UUID validation. Existing executor blocked before worker. No numerical rejection.'},
        'review': {'first_failure': 'Actual candidate diagnostic captured missing Qwen3Attention.rotary_emb in _extract/setup/build. Catch returned empty mapping, then declared-site guard failed. Three failed workers, no quality/clean samples.',
            'new_code': 'New model-level rotary extraction and installed fallback wrappers, plus numerical/helper changes. Six fused Triton kernels and internal autotuning require real execution; no manual edits.',
            'parent_boundary': 'Single active replacement bundle retires f065. BindingSession references original model.forward at install; no inherited f065 prefill Graph remains active. Prefill timing/quality guarantee and exclusive GPU-bottleneck claims unsupported.',
            'cpu_probes': 'Two-layer hidden128 BF16 CPU prompt6/three-decode Torch substitutions: TypeError, KeyError, exit0 NaN/FAILED preserved. Final fixture fixed missing out.copy_ and printed tokens equal with nonzero errors. This does not execute Triton/CUDA Graphs or original full workload.',
            'access': 'Recorded tools cover own inputs/historical candidate/current accepted source, original jobs and installed library/model config. No observed later-answer reads in reviewed history; external filesystem not hermetic. Complete commands/outputs/write-edit bytes retained privately.',
            'limits': 'Selfchecks bypass actual framework dispatch. Full source fidelity, quality, execution, search and independent acceptance remain separate. No new acceptance or bundle coexistence.'},
        'pointers': {'source': audit['baseline'], 'continuation': search.revisions.current().id},
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k != 'snapshots'}
    public['snapshot_summary'] = [{k: v for k, v in r.items() if k != 'input_hashes'} for r in roles]
    public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    write_json(root / 'public-terminal-audit.json', public)
    print(json.dumps({'status': 'audited', 'sha256': public['terminal_audit_sha256'], 'sessions': sessions, 'jobs': len(jobs), 'frozen': report['frozen_files']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
