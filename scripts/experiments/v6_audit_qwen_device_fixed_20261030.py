"""Audit fresh-device controls and fixed endpoints without retiming."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
from kernel_optimizer.config import load_config
from kernel_optimizer.project.constraints import constraints_for_runs
from kernel_optimizer.project.measurement import valid_run
from kernel_optimizer.project.search import ProjectSearch
from kernel_optimizer.runtime.execution import write_json
from kernel_optimizer.runtime.provenance import framework_source
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed'
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    search = ProjectSearch(load_config(root / 'private-config.yaml', []), root / 'qwen-fixed')
    identity = read(root / 'staged-identities.json')
    assert search.revisions.current().id == search.prepared['revision'] == identity['baseline']['id']
    assert search.revisions.revision(identity['candidate']['id']).files == tuple(search.revisions.revision(protocol['incoming_revision']).files)
    old = read(Path(protocol['source']) / 'request.json'); new = search.request.portable()
    assert {k for k in old if old[k] != new[k]} == {'devices'}
    assert new['devices'] == protocol['new_devices']
    endpoints = {}
    for name, path in [('controls', Path(search.prepared['controls_path']) / 'receipt.json'),
                       ('B', root / 'B-result.json'), ('C', root / 'C-result.json')]:
        run = read(path); assert valid_run(run)
        rank = run['ranks'][0]; result = rank['result']
        assert rank['framework_source'] == framework_source() == protocol['core']
        assert rank['binding']['torch_uuid'] == new['devices'][0]
        assert len(result['samples']) == 5
        for metric, value in result['metrics'].items():
            assert statistics.median(s[metric] for s in result['samples']) == value
        if name != 'controls':
            assert result['math_policy_unchanged']
            assert len(result['checks']) == 2 and all(c['passed'] and c['max_abs_error'] == 0 for c in result['checks'])
        folder = Path(search.prepared['controls_path']) if name == 'controls' else search.output / 'jobs' / name / 'run'
        request = read(folder / 'request.json')
        assert request['devices'] == new['devices'] and request['parameters']['request'] == new
        assert sha(request['parameters']['adapter']) == search.prepared['adapter_sha256']
        assert sha(request['entry']) == run['entry_sha256']
        bundle_files = []
        for b in request['parameters'].get('bundles', []):
            bundle_files.append({'id': b['id'], 'sha256': sha(b['file'])})
            assert sha(b['file']) == protocol['code_sha256'] and b['parameters'] == {}
        endpoints[name] = {'path': str(path), 'sha256': sha(path), 'request_sha256': sha(folder / 'request.json'),
            'entry_sha256': run['entry_sha256'], 'binding': rank['binding'], 'bundle_files': bundle_files,
            'checks': result.get('checks'), 'control_checks': result.get('controls'), 'samples': result['samples'],
            'metrics': result['metrics'], 'coverage': result.get('coverage'), 'setup': result.get('setup'),
            'constraints': constraints_for_runs([run], new['objective'], search.prepared['evaluation']),
            'log_sha256': sha(folder / 'rank-0.log')}
    source_run = read(root / 'source-check.json'); assert valid_run(source_run)
    source = source_run['ranks'][0]['result']
    assert len(source['checks']) == 2 and all(c['passed'] and c['exact_tokens'] and c['finite_logits'] and c['max_abs_error'] == 0 for c in source['checks'])
    assert all(not c['passed'] for c in source['negative_checks'].values())
    assert sha(protocol['source_reference']['path']) == source['source_reference_sha256']
    ref = Path(search.prepared['controls_path']) / 'reference-rank-0.pt'
    assert sha(ref) == source['controls_reference_sha256']
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'],
        'core': framework_source(), 'frozen_files': len(protocol['frozen_identities']), 'source': source,
        'endpoints': endpoints, 'new_controls_reference': {'path': str(ref), 'sha256': sha(ref),
            'note': 'New CUDA controls artifact happens to equal old reference SHA; execution is fresh, not a reference copy.'},
        'pointers': {'new': search.revisions.current().id, 'old': read(Path(protocol['source']) / 'project/current.json')},
        'review': {'quality': 'Controls/source gate and B/C full checks passed; candidate quality is on recorded parent fallback path, not successful optimized path.',
            'routes': 'C quality phase records 62 parent_fallback_setup_failure and 2 prefill_parent_passthrough, references 62/2. Empty framework fallbacks does not erase candidate internal fallback.',
            'constraint': 'Fresh TTFT original reference23.285755, limit24.45004275; fixed B24.5339 fails and C23.550697 passes. B gate was quality/coverage only. All samples retained; no threshold change.',
            'timing': 'Fixed B701.762096/C674.079858 ms are diagnostic, no paired-noise or accepted gain. Candidate setup3899.385769ms outside warm score; no clean-wall/profile subtraction.',
            'ownership': 'C lifecycle peak +119403008 bytes and runtime peak +99271168 bytes; not memory gain. Extra setup owners survive fallback. Arbitrary aliases/concurrency not certified.',
            'next': 'One separate existing run-only candidate_diagnostics may identify caught setup error. No manual repair, retiming, new source or candidate acceptance.'},
        'df': subprocess.check_output(['df', '-B1', '--output=size,used,avail', '/root/autodl-tmp'], text=True)}
    write_json(root / 'terminal-audit.json', report)
    public = {k: v for k, v in report.items() if k != 'endpoints'}
    public['endpoints'] = {k: {n: x for n, x in v.items() if n not in ['binding', 'control_checks']} for k, v in endpoints.items()}
    public['terminal_audit_sha256'] = sha(root / 'terminal-audit.json')
    write_json(root / 'public-terminal-audit.json', public)
    print(public['terminal_audit_sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
