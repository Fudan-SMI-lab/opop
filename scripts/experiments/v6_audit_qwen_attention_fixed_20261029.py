"""Audit frozen generated-code endpoints and compare all retained full tensors."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import statistics
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
        assert result['status'] == 'passed'
        assert len(value['samples']) == 5 and len(value['checks']) == 2
        assert all(c['passed'] and c['max_abs_error']==0 for c in value['checks'])
        assert value['math_policy_unchanged']
        for metric,number in value['metrics'].items():
            assert number==statistics.median(s[metric]for s in value['samples'])
        for sample in value['samples']:
            assert sample['generated_tokens']==32 and sample['decode_tokens']==31
            assert abs(sample['end_to_end_ms']-sample['ttft_ms']-sample['decode_ms'])<1e-9
            assert abs(sample['decode_tokens_per_second']-31000/sample['decode_ms'])<1e-9
        constraint=row['constraints'][0]
        assert constraint['original_reference']==24.051052 and constraint['limit']==25.2536046
        assert constraint['values']==[value['metrics']['ttft_ms']]
        assert constraint['passed']==(value['metrics']['ttft_ms']<=constraint['limit'])
        if arm!='B':
            coverage=value['coverage']['fixed-attention-candidate-continuation']
            assert coverage['candidate_declared_routes']==[{'call':'323bcbe006b343648b346af9','route':'graph_replay','count':62}]
            assert coverage['fallbacks']=={} and coverage['reference_invocations']=={}
        record = {'status': result['status'], 'error': rank.get('error'), 'request_sha256': sha(row['request']),
            'request_diff_exact': True, 'entry_sha256': result['entry_sha256'], 'core': worker['framework_source'],
            'revision': started['revision'], 'samples': value.get('samples'), 'checks': value.get('checks'),
            'coverage': value.get('coverage'), 'constraints': row['constraints'], 'metrics':value['metrics'], 'setup':value['setup']}
        if observed:
            side = read(row['observer_summary']); assert side['observer_entry_sha256'] == protocol['observer_sha256']
            assert sha(protocol['observer']) == protocol['observer_sha256']
            tensors = []
            for observation in side['observations']:
                assert sha(observation['path']) == observation['sha256']
                output = torch.load(observation['path'], map_location='cpu', weights_only=True)
                assert output['logits'].shape == source['logits'].shape and output['tokens'].shape == source['tokens'].shape
                assert compare(output, source, quality) == observation['source_comparison']
                assert all(torch.equal(output[k],source[k])for k in output) and torch.isfinite(output['logits']).all()
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
    assert all(torch.equal(raw_outputs['B'][k],raw_outputs['candidate-continuation'][k]) for k in source)
    report['full_tensor_relations']={'all_12_full_observed_outputs_exact_source':True,
        'old_failed_outputs_unchanged':True,'scope':'Observed source fidelity includes full32tokens/full logits before/after observe; not full arbitrary ownership or unobserved per-call native path proof.'}
    from kernel_optimizer.config import load_config
    from kernel_optimizer.project.search import ProjectSearch
    config = load_config(root / 'private-config.yaml', [])
    report['pointers'] = {p: ProjectSearch(config, Path(p)).revisions.current().id for p in [protocol['source'], str(root / 'fixed-validation')]}
    assert set(report['pointers'].values()) == {protocol['source_revision']}
    report['scope'] = 'Three quality-passing fixed endpoints, both candidate TTFT constraints failed; 15 clean samples and 6 fullchecks retained. No performance acceptance or production adoption. Raw tensors remain server-side.'
    report['review']={'route_limits':'Observed callback279 graph_replay covers2quality+2warmup+5clean times31, onlyrun phase. Code calls graph.replay before this label. Setup disable/guard/failure branches emit different labels; observed callbacks support successful setup/replay code path, but native confirmation requires a separate trace. Teacher stride does not match framework signature; missing teacher labels alone not native proof.',
        'quality':'New raw observed greedy/teacher/after values are exact source across2invocations; source quality gap is resolved for these endpoints. Existing old failures remain; multiple implementation changes preclude unique causal isolation.',
        'constraint':'Unobserved candidate26.510787 and observed25.917823 exceed frozen25.2536046; no retry until pass or ABBA to overturn this diagnostic constraint failure. Full generation395.008819/394.66852 are diagnostic, observer baseline689.836474 is not a paired clean acceptance anchor.',
        'ownership':'Setup retains masterKV/logits/Graph pools and mirror views. Fresh prefix copy occurs per invocation; full observed outputs immutable acrossobserve. Retained views/temporary global mask patch do not establish arbitrary concurrent composition/lifetime safety. Patch context exit restores by code; no retrospective runtime event recorded.',
        'next':'Separate fixed native path check and one ordinary selection/candidate trial from unchanged baseline using explicit factual diagnostic feedback, with failed candidate bytes available only as history. No automatic production adoption or prior candidate publication.'}
    report['df'] = subprocess.check_output(['df', '-B1', '/root/autodl-tmp'], text=True)
    write_json(root / 'terminal-audit.json', report)
    print({'path': str(root / 'terminal-audit.json'), 'sha256': sha(root / 'terminal-audit.json'),
           'bytes': (root / 'terminal-audit.json').stat().st_size, 'all_observed_outputs_exact_source': True, 'df': report['df']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
