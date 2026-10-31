import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('previous,current,changes', [
    ((), ('new',), {'added': ['new'], 'removed': [], 'retained': []}),
    (('shared', 'old'), ('shared', 'old', 'new'),
     {'added': ['new'], 'removed': [], 'retained': ['shared', 'old']}),
    (('shared', 'old'), ('shared', 'new'),
     {'added': ['new'], 'removed': ['old'], 'retained': ['shared']}),
    (('old',), ('old',), {'added': [], 'removed': [], 'retained': ['old']}),
    (None, ('root',), None),
])
def test_planner_receives_actual_parent_composition(tmp_path, monkeypatch, previous, current, changes):
    from contextlib import nullcontext
    from kernel_optimizer.project import search as module

    search = object.__new__(module.ProjectSearch)
    search.output, search.config, search.store = tmp_path, None, None
    search.request = SimpleNamespace(profiling={}, portable=lambda: {},
        objective={'name': 'duration', 'unit': 'ms', 'direction': 'minimize'})
    search.prepared = {'revision': 'reference', 'evaluation': {'ranks': [
        {'rank': 0, 'result': {'metrics': {'duration': 10}}}]}}
    candidate = SimpleNamespace(id='candidate', parent_id='parent' if previous is not None else None,
                                bundles=current)
    # Neither the original reference nor discovery is the candidate's parent.
    revisions = {'reference': SimpleNamespace(bundles=('reference-only',)),
                 'discovery': SimpleNamespace(bundles=('discovery-only',)),
                 'parent': SimpleNamespace(bundles=previous)}
    search.revisions = SimpleNamespace(revision=revisions.__getitem__)
    captured = {}

    class Agent:
        def generate(self, role, inputs, prompt, outputs, *, validator):
            captured.update(json.loads(inputs['revisions.json']))
            result = tmp_path/'generated'
            result.mkdir()
            (result/'collection.json').write_text(json.dumps({
                'collectors': [], 'requested_fields': [], 'unavailable_fields': {},
                'rationale': 'No additional collection.', 'revision': candidate.id}))
            validator(result)
            return result, {}

    monkeypatch.setattr(module, 'Runtime', lambda *a, **kw: nullcontext(None))
    monkeypatch.setattr(module, 'ProjectAgents', lambda *a, **kw: Agent())
    search.plan_collectors({'revision': 'discovery', 'communication': {}, 'coverage': []},
                           evaluation={}, revisions=[candidate])
    assert captured['discovery_revision'] == 'discovery'
    row, = captured['candidates']
    assert row['revision'] == candidate.id and row['parent'] == candidate.parent_id
    assert row['bundles'] == list(current)
    assert row['parent_bundles'] == (list(previous) if previous is not None else None)
    assert row['bundle_changes'] == changes


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


@pytest.mark.parametrize('direction,aggregation,expected', [
    ('minimize', None, 74.676), ('maximize', None, 64.149),
    ('minimize', 'mean', (74.676 + 64.149) / 2),
])
def test_reference_objective_uses_the_requested_rank_reduction(direction, aggregation, expected):
    from kernel_optimizer.project.diagnostic_planning import objective_reference
    run = {'ranks': [{'result': {'metrics': {'duration': value}}} for value in [74.676, 64.149]]}
    objective = {'name': 'duration', 'unit': 'ms', 'direction': direction}
    if aggregation is not None:
        objective['rank_aggregation'] = aggregation
    result = objective_reference(run, objective)
    assert result['value'] == expected
    assert result['metric'] == 'duration'
    assert 'not a paired improvement' in result['scope']


def test_reference_objective_remains_unknown_if_a_rank_has_no_measurement():
    from kernel_optimizer.project.diagnostic_planning import objective_reference
    run = {'ranks': [{'result': {'metrics': {'duration': 7}}}, {'result': {'metrics': {}}}]}
    result = objective_reference(run, {'name': 'duration', 'unit': 'ms', 'direction': 'minimize'})
    assert result['value'] is None
    assert result['reason']
