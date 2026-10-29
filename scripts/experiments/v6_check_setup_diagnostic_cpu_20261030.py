"""One CPU boundary fixture for exact generated diagnostic bytes, never a model run."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import kernel_optimizer.project.binding as binding
from scripts.experiments.v6_audit_fsdp_20261028 import sha


def main(probe, output):
    output.mkdir(parents=True, exist_ok=False)
    before = sha(probe)
    assert before == '8ab9078e71390b240d8a471cc765084a58b9681d3170f6a8a4050a7221fb2ad4'
    spec = importlib.util.spec_from_file_location('exact_setup_probe', probe)
    diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
    original_assert = torch.testing.assert_close; original_binding = binding.BindingSession
    # Use the real torch comparator and original generated recording policy.
    rec = diag._Recorder({'atol': .02, 'rtol': .02}, {}, {})
    rec.open_window('cpu-boundary', __file__); restore = rec.patch_assert_close()
    caught = None
    try:
        for _ in range(diag._MAX_ASSERT_SITES):
            torch.testing.assert_close(torch.zeros(2), torch.zeros(2), atol=.012, rtol=.012)
        try:
            torch.testing.assert_close(torch.ones(2), torch.zeros(2), atol=.012, rtol=.012,
                                       msg='synthetic late finite mismatch')
        except AssertionError as error:
            caught = error
    finally:
        restore(); rec.close_window()
    assert caught is not None and torch.testing.assert_close is original_assert
    window = rec.windows[0]
    assert window['assert_close_total'] == 129 and window['assert_close_overflow'] == 1
    assert len(window['assert_close_sites']) == 128
    assert all(x['outcome'] == {'passed': True} for x in window['assert_close_sites'])
    evidence = rec.report({'original_exception_caught_by_fixture': str(caught)}, {})
    diag._write_evidence_json(output/'late-failure-evidence.json', evidence)

    # An uncapped nonfinite comparison remains an actual failure with truthful markers.
    nonfinite = diag._Recorder({'atol': .02, 'rtol': .02}, {}, {})
    nonfinite.open_window('cpu-nonfinite', __file__); restore = nonfinite.patch_assert_close()
    try:
        try:
            torch.testing.assert_close(torch.tensor([float('nan'), float('inf')]), torch.zeros(2), atol=.012, rtol=.012)
        except AssertionError:
            pass
        else:
            raise AssertionError('real comparator did not reject nonfinite')
    finally:
        restore(); nonfinite.close_window()
    diag._write_evidence_json(output/'nonfinite-evidence.json', nonfinite.report({}, {}))
    nf = nonfinite.windows[0]['assert_close_sites'][0]
    assert nf['comparison']['nonfinite_difference_count'] == 2 and 'raised' in nf['outcome']

    # The original entry is a small CPU fixture, not the original model evaluator.
    entry = output/'original_entry.py'
    entry.write_text("def run(context):\n    context['calls'].append('original')\n    if context.get('raise_original'):\n        raise context['sentinel']\n    return context['original_result']\n")
    outcomes = []
    for arm, raised in [('return-invalid', False), ('raise-original', True)]:
        sentinel = RuntimeError('original CPU fixture error')
        original_result = {'valid': False, 'checks': [{'passed': False}]}
        context = {'rank': 0, 'world_size': 1, 'device': 'cpu', 'output': str(output/arm),
            'parameters': {'request': {'quality': {'atol': .02, 'rtol': .02}}, 'bundles': [],
                'diagnostic_context': {'original_entry': str(entry), 'original_function': 'run', 'arm': arm}},
            'calls': [], 'raise_original': raised, 'sentinel': sentinel, 'original_result': original_result}
        if raised:
            try:
                diag.run(context)
            except RuntimeError as error:
                assert error is sentinel
            else:
                raise AssertionError('original exception not propagated')
        else:
            result = diag.run(context)
            assert result is original_result and result['valid'] is False
        assert context['calls'] == ['original']
        assert binding.BindingSession is original_binding and torch.testing.assert_close is original_assert
        path = output/arm/'diagnostic-evidence-rank-0.json'
        assert path.is_file()
        outcomes.append({'arm': arm, 'original_called_once': True, 'original_outcome_preserved': True,
            'hooks_restored': True, 'evidence_sha256': sha(path)})
    assert not torch.cuda.is_initialized() and sha(probe) == before
    report = {'probe': str(probe), 'probe_sha256': before, 'cuda_initialized': False,
        'scope': 'CPU-only original generated probe plus real torch comparator; no candidate/model/collective/GPU rerun.',
        'late_failure': {'total_calls': 129, 'saved_successes': 128, 'overflow': 1,
            'original_failure_preserved': True, 'failure_operands_or_outcome_saved': False,
            'evidence_sha256': sha(output/'late-failure-evidence.json')},
        'nonfinite': {'original_failure_preserved': True, 'nonfinite_difference_count': 2,
            'finite_tolerance_violation_count': nf['comparison']['violates_call_tolerance_count'],
            'note': 'Finite violation count excludes NaN/Inf; zero does not mean finite or pass.',
            'evidence_sha256': sha(output/'nonfinite-evidence.json')},
        'entry_cases': outcomes,
        'conclusion': 'Concrete evidence-retention defect: successful assertions exhaust the cap and a later failing comparison has no saved operands/outcome. Outer behavior remains unchanged in these CPU fixtures. No inference of actual model numerics.'}
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--probe', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.probe, a.output)
