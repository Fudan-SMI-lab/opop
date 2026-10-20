import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('enabled', [False, True])
def test_screening_rejection_can_collect_an_agent_selected_revision(tmp_path, monkeypatch, enabled):
    from kernel_optimizer.project.search import ProjectSearch
    from kernel_optimizer.project import inner_loop

    search = object.__new__(ProjectSearch)
    search.output = tmp_path
    search.request = SimpleNamespace(objective={}, profiling={'feedback_agent_plan': enabled})
    search.store = SimpleNamespace(get_artifact=lambda _: b'def build(context): return {}\n')
    search.revisions = SimpleNamespace(revision=lambda ident: SimpleNamespace(id=ident))
    bundle = SimpleNamespace(id='trial', installer='candidate.py',
        edits=[SimpleNamespace(path='candidate.py', content_ref='code')], model_dump=lambda **_: {'id': 'trial'})
    observations = [{'revision': f'r{i}', 'bundle': {'id': f'b{i}', 'parameters': {'n': i}},
        'constraints': [{'passed': False}], 'run': {'status': 'passed', 'ranks': [
            {'rank': 0, 'result': {'valid': True, 'metrics': {'step_ms': 100+i}}}]}}
        for i in (1, 2)]
    monkeypatch.setattr(inner_loop, 'tune', lambda *args: (None, {'observations': observations}))
    calls = []

    def plan(discovery, *, evaluation, revisions):
        assert enabled
        assert [r.id for r in revisions] == ['r1', 'r2']
        assert [r['revision'] for r in evaluation['observations']] == ['r1', 'r2']
        calls.append('plan')
        return {'plan': {'revision': 'r2', 'collectors': ['binding_cost']}, 'agent': {}}

    def collect(revision, names):
        calls.append((revision.id, names))
        return {'binding_cost': {'status': 'observed', 'provenance': {'revision': revision.id}}}

    search.plan_collectors = plan
    search.collect_extra = collect
    (tmp_path/'trials/trial').mkdir(parents=True)
    result = search._evaluate_generated(bundle, {'trial': 'trial', 'proposal': {'parameter_space': {}}}, {})
    assert result['decision'] == {'accepted': False, 'reasons': ['no_valid_parameter_candidate']}
    assert calls == (['plan', ('r2', ['binding_cost'])] if enabled else [])
    assert ('resources' in result) == enabled
    assert json.loads((tmp_path/'latest-feedback.json').read_text()) == result


def test_summary_does_not_infer_fallback_from_parent_calls():
    from kernel_optimizer.project.diagnostic_planning import endpoint_summary
    routes = [{'call': 'site', 'route': 'parent_fallback_setup_failure', 'count': 13}]
    summary = endpoint_summary({'status': 'passed', 'ranks': [{'rank': 0, 'result': {'valid': True,
        'coverage': {'bundle': {'counts': {'site': 13}, 'reference_invocations': {'site': 13},
                                'fallbacks': {}, 'candidate_declared_routes': routes}}}}]})
    assert summary['quality_and_coverage_valid']
    counts = summary['ranks'][0]['coverage']['bundle']
    assert counts['reference_invocations'] == 13
    assert counts['framework_fallbacks'] == 0
    assert counts['candidate_declared_routes'] == routes
    assert 'not independent native proof' in summary['interpretation']
