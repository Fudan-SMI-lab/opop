"""Read-only full-tensor audit and saved-tensor CPU/CUDA comparator diagnostic."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    protocol = read(root / 'protocol.json')
    terminal = read(root / 'followup-completed.json')
    assert terminal['phase'] == 'completed' and terminal['frozen_inputs_unchanged']
    original = {**protocol['source_requests'], **protocol['source_files'],
                protocol['source_reference']['path']: protocol['source_reference']['sha256']}
    assert all(sha(p) == h for p, h in original.items())
    source = torch.load(protocol['source_reference']['path'], map_location='cpu', weights_only=True)
    source = {k: source[k] for k in ['tokens', 'logits']}
    quality = {'atol': .02, 'rtol': .02}
    report = {'at': datetime.now(timezone.utc).isoformat(), 'terminal_sha256': sha(root / 'followup-completed.json'),
              'protocol_sha256': sha(root / 'protocol.json'), 'core': protocol['core'],
              'all_frozen_input_hashes_verified': original, 'endpoints': {}, 'observations': {},
              'scope': 'Fixed diagnostic audit, not acceptance. Saved tensors only for CPU/CUDA arithmetic; no model rerun.'}
    request_paths = list(protocol['source_requests'])
    expected_keys = ['0-B-observed', '1-C1-unobserved', '2-C1-observed', '3-C2-unobserved', '4-C2-observed']
    assert list(terminal['results']) == expected_keys
    tensors = {}
    controls = {}
    for index, name in enumerate(expected_keys):
        stage = terminal['results'][name]
        result = read(Path(stage['result']))
        request = read(root / name / 'request.json')
        original_request = read(Path(request_paths[0 if index == 0 else 1 if index <= 2 else 2]))
        observed = index in [0, 2, 4]
        expected_request = dict(original_request)
        expected_request.update(output=str(root / name), wait_for_devices=True)
        if observed:
            expected_request.update(entry=request['entry'], function='observed_worker',
                parameters={**original_request['parameters'], 'diagnostic_source_reference': protocol['source_reference']['path']})
            assert sha(request['entry']) == 'a06b6c0e14495bffb68aa498c845ce66378223e9c95579b0b66a7bd463cc566c'
        assert request == expected_request, name
        for key, value in request['parameters'].items():
            if 'reference' in key and isinstance(value, str) and Path(value).is_file():
                controls[value] = sha(value)
            if key == 'reference_dir':
                for path in Path(value).glob('reference-rank-*.pt'):
                    controls[str(path)] = sha(path)
        rank = result['ranks'][0]
        worker_record = read(root / name / 'rank-0.json')
        started = read(root / name / 'rank-0-started.json')
        original_started = read(Path(request_paths[0 if index == 0 else 1 if index <= 2 else 2]).parent / 'rank-0-started.json')
        assert started['revision'] == original_started['revision']
        assert worker_record['framework_source'] == protocol['core']
        value = rank.get('result') or {}
        report['endpoints'][name] = {'status': result['status'], 'error': rank.get('error'),
            'entry_sha256': result['entry_sha256'], 'result_sha256': sha(stage['result']),
            'request_sha256': sha(root / name / 'request.json'), 'request_diff_exact': True,
            'checks': value.get('checks'), 'samples': value.get('samples'), 'coverage': value.get('coverage'),
            'framework_source': worker_record['framework_source'], 'revision': started['revision'],
            'revision_identity_matches_original': True}
        assert result['status'] == ('passed' if index == 0 else 'failed'), (name, result['status'])
        if not observed:
            continue
        summary = read(Path(stage['observer_summary']))
        assert summary['source_reference_sha256'] == protocol['source_reference']['sha256']
        rows = []
        for row in summary['observations']:
            path = Path(row['path']); assert sha(path) == row['sha256']
            raw = torch.load(path, map_location='cpu', weights_only=True)
            assert list(raw['tokens'].shape) == [1, 32] and list(raw['logits'].shape) == [1, 32, 151936]
            actual = compare(raw, source, quality)
            assert actual == row['source_comparison']
            delta = (raw['logits'].float() - source['logits'].float()).abs()
            assert delta.flatten(2).amax(-1).tolist() == row['max_abs_error_by_position']
            assert (delta > .02 + .02 * source['logits'].float().abs()).flatten(2).sum(-1).tolist() == row['outside_tolerance_by_position']
            assert torch.equal(raw['tokens'], source['tokens']) and torch.isfinite(raw['logits']).all()
            tensors[(name, row['phase'], len(rows) // 3)] = raw
            rows.append(row)
        for step, relation in enumerate(summary['relations']):
            before = tensors[(name, 'greedy', step)]; after = tensors[(name, 'greedy-after-observe', step)]
            teacher = tensors[(name, 'teacher', step)]
            assert all(torch.equal(before[k], after[k]) for k in before)
            assert all(torch.equal(teacher[k], source[k]) for k in teacher)
            assert compare(after, before, {'atol': 0, 'rtol': 0}) == relation['greedy_before_after']
            assert compare(teacher, before, quality) == relation['teacher_vs_greedy_before']
        report['observations'][name] = {'summary_sha256': sha(stage['observer_summary']),
            'observations': rows, 'relations': summary['relations'], 'routes': summary['candidate_routes'],
            'full_value_relations_recomputed': True}
    a = tensors[('2-C1-observed', 'greedy', 0)]
    b = tensors[('4-C2-observed', 'greedy', 0)]
    assert all(torch.equal(a[k], b[k]) for k in a)
    report['C1_C2_full_greedy_exactly_equal'] = True
    # Same immutable saved values, original dtype/tolerance/direction. This
    # isolates comparator arithmetic; it does not rerun or repair the candidate.
    arithmetic = {}
    for device in ['cpu', 'cuda:0']:
        x, y = source['logits'].to(device), a['logits'].to(device)
        try:
            torch.testing.assert_close(x, y, atol=.02, rtol=.02)
            arithmetic[device] = {'passed': True}
        except AssertionError as exc:
            arithmetic[device] = {'passed': False, 'message': str(exc)}
        arithmetic[device]['isclose_mismatch_count'] = int((~torch.isclose(x, y, atol=.02, rtol=.02)).sum())
        del x, y
    report['saved_tensor_comparator_diagnostic'] = arithmetic
    report['controls_hashes'] = controls
    assert '948870860ae9e3a0c0dbb16e529def4c6b163b86f5378dbccb3ca1bef6df1343' in controls.values()
    report['df'] = subprocess.check_output(['df', '-B1', '/root/autodl-tmp'], text=True)
    write_json(root / 'terminal-audit.json', report)
    print(json.dumps({'audit': str(root / 'terminal-audit.json'), 'bytes': (root / 'terminal-audit.json').stat().st_size,
                      'sha256': sha(root / 'terminal-audit.json'), 'comparator': arithmetic, 'df': report['df']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
