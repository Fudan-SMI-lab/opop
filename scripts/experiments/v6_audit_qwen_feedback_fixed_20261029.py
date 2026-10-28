"""Audit frozen generated-code endpoints and compare all retained full tensors."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import torch

from kernel_optimizer.project.evaluation import compare
from kernel_optimizer.runtime.execution import write_json
from scripts.experiments.v6_audit_fsdp_20261028 import read, sha


def main(root):
    protocol = read(root / 'protocol.json'); terminal = read(root / 'followup-completed.json')
    staged = read(root / 'staged-identities.json')
    assert terminal['phase'] == 'completed'
    assert all(sha(p) == h for p, h in protocol['frozen_identities'].items())
    reference = protocol['source_reference']; assert sha(reference['path']) == reference['sha256']
    source = torch.load(reference['path'], map_location='cpu', weights_only=True)
    source = {k: source[k] for k in ['tokens', 'logits']}; quality = {'atol': .02, 'rtol': .02}
    report = {'at': datetime.now(timezone.utc).isoformat(), 'finished_at': terminal['finished_at'],
        'core': protocol['core'], 'protocol_sha256': sha(root / 'protocol.json'),
        'terminal_sha256': sha(root / 'followup-completed.json'), 'staged': staged,
        'frozen_identities_verified': protocol['frozen_identities'], 'endpoints': {}, 'full_tensor_relations': {}}
    raw_outputs = {}; errors = []
    for index, (arm, observed) in enumerate(protocol['sequence']):
        name = f'{index}-{arm}-' + ('observed' if observed else 'unobserved')
        row = terminal['results'][name]; result = read(row['result']); req = read(row['request'])
        run = Path(row['request']).parent; worker = read(run / 'rank-0.json'); started = read(run / 'rank-0-started.json')
        assert sha(row['result']) == row['result_sha256']
        assert worker['framework_source'] == protocol['core']
        assert started['revision']['revision'] == staged[arm]['revision']
        assert started['revision']['files'] == staged[arm]['files']
        assert all(sha(Path(req['cwd']) / n) == h for n, h in staged[arm]['files'].items())
        expected = dict(protocol['template'])
        expected.update(cwd=req['cwd'], entry=protocol['observer'] if observed else str(Path(req['cwd']) / 'opop_evaluate.py'),
                        function='observed_worker' if observed else expected['function'], output=str(run), wait_for_devices=True)
        parameters = dict(expected['parameters']); parameters.update(adapter=str(Path(req['cwd']) / 'opop_adapter.py'), bundles=req['parameters']['bundles'])
        if observed: parameters['diagnostic_source_reference'] = reference['path']
        expected['parameters'] = parameters; assert req == expected
        assert [b['id'] for b in parameters['bundles']] == staged[arm]['bundles']
        for bundle in parameters['bundles']:
            assert bundle['parameters'] == {}
            assert sha(bundle['file']) == staged[arm]['files'][str(Path(bundle['file']).relative_to(req['cwd']))]
        rank = result['ranks'][0]; value = rank.get('result') or {}
        assert result['status'] == ('passed' if arm == 'B' else 'failed')
        if arm != 'B': errors.append(rank['error']); assert not value.get('samples')
        else: assert len(value['samples']) == 5 and len(value['checks']) == 2 and all(c['passed'] for c in value['checks'])
        record = {'status': result['status'], 'error': rank.get('error'), 'request_sha256': sha(row['request']),
            'request_diff_exact': True, 'entry_sha256': result['entry_sha256'], 'core': worker['framework_source'],
            'revision': started['revision'], 'samples': value.get('samples'), 'checks': value.get('checks'),
            'coverage': value.get('coverage'), 'constraints': row['constraints']}
        if observed:
            side = read(row['observer_summary']); assert side['observer_entry_sha256'] == protocol['observer_sha256']
            assert sha(protocol['observer']) == protocol['observer_sha256']
            tensors = []
            for observation in side['observations']:
                assert sha(observation['path']) == observation['sha256']
                output = torch.load(observation['path'], map_location='cpu', weights_only=True)
                assert output['logits'].shape == source['logits'].shape and output['tokens'].shape == source['tokens'].shape
                assert compare(output, source, quality) == observation['source_comparison']
                assert torch.equal(output['tokens'], source['tokens']) and torch.isfinite(output['logits']).all()
                delta = (output['logits'].float() - source['logits'].float()).abs()
                assert delta.flatten(2).amax(-1).tolist() == observation['max_abs_error_by_position']
                assert (delta > .02 + .02 * source['logits'].float().abs()).flatten(2).sum(-1).tolist() == observation['outside_tolerance_by_position']
                tensors.append(output)
            for step, relation in enumerate(side['relations']):
                greedy, teacher, after = tensors[3*step:3*step+3]
                assert all(torch.equal(greedy[k], after[k]) for k in greedy)
                assert all(torch.equal(teacher[k], source[k]) for k in teacher)
                assert compare(teacher, greedy, quality) == relation['teacher_vs_greedy_before']
            raw_outputs[arm] = tensors[0]
            record['observed'] = {'summary_sha256': sha(row['observer_summary']), 'observations': side['observations'],
                                  'relations': side['relations'], 'routes': side['candidate_routes'], 'all_raw_values_recomputed': True}
        report['endpoints'][name] = record
    assert len(set(errors)) == 1
    old = root.parent / 'qwen-relation-diagnostic-20261029'
    for name in ['2-C1-observed', '4-C2-observed']:
        path = old / name / 'observation-00-greedy.pt'
        values = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(values[k], raw_outputs['control'][k]) for k in values)
        report['full_tensor_relations'][name] = {'path': str(path), 'sha256': sha(path), 'equal_to_new_control': True}
    assert all(torch.equal(raw_outputs['control'][k], raw_outputs['treatment'][k]) for k in source)
    report['full_tensor_relations']['control_treatment'] = {'exact_equal': True, 'all_four_candidate_greedy_equal': True,
        'interpretation': 'Different generated cache/graph designs reproduced identical full greedy values. This does not uniquely isolate a shared mechanism; no new acceptance.'}
    from kernel_optimizer.config import load_config
    from kernel_optimizer.project.search import ProjectSearch
    config = load_config(root / 'private-config.yaml', [])
    report['pointers'] = {p: ProjectSearch(config, Path(p)).revisions.current().id for p in [protocol['source'], str(root / 'fixed-validation')]}
    assert set(report['pointers'].values()) == {protocol['source_revision']}
    report['scope'] = 'Five fixed attempts, four quality failures before clean timing; no performance acceptance or production adoption. Raw tensors remain server-side. Old decisions unchanged.'
    report['df'] = subprocess.check_output(['df', '-B1', '/root/autodl-tmp'], text=True)
    write_json(root / 'terminal-audit.json', report)
    print({'path': str(root / 'terminal-audit.json'), 'sha256': sha(root / 'terminal-audit.json'),
           'bytes': (root / 'terminal-audit.json').stat().st_size, 'all_four_greedy_values_equal': True, 'df': report['df']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
