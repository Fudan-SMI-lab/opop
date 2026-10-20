from types import SimpleNamespace

from kernel_optimizer.project.binding_cost import BindingCost


def test_cost_keeps_parent_and_quality_failures_do_not_become_overhead(tmp_path):
    parent = SimpleNamespace(id='parent', bundles=('inherited',))
    candidate = SimpleNamespace(id='child', parent_id='parent', bundles=('inherited', 'new'))
    calls = []

    def execute(revision, name, action, *, collector_options):
        calls.append((revision.id, action, collector_options))
        # One rank fails full quality although the worker itself exits normally.
        ranks = [{'rank': rank, 'result': {'valid': not (len(calls) == 2 and rank == 1),
                  'metrics': {'framework.elapsed_ms': 10 + rank}}} for rank in range(2)]
        return {'status': 'passed', 'ranks': ranks}

    search = SimpleNamespace(output=tmp_path, execute=execute,
                             revisions=SimpleNamespace(revision=lambda _: parent))
    result = BindingCost().collect(search, candidate)
    assert result['status'] == 'unavailable'
    assert 'elapsed_pairs' not in result
    assert calls == [('parent', 'binding_ablation', {}),
                     ('child', 'binding_ablation', {'passthrough_bundles': ['new']})]
    assert len(result['runs']) == 2
    # Replacing the parent's bundle cannot be measured by this additive control.
    candidate.bundles = ('new',)
    assert BindingCost().collect(search, candidate)['status'] == 'not_applicable'
    assert len(calls) == 2


def test_cost_preserves_reverse_order_and_rank_maxima(tmp_path):
    parent = SimpleNamespace(id='parent', bundles=())
    candidate = SimpleNamespace(id='child', parent_id='parent', bundles=('new',))
    calls = []

    def execute(revision, name, action, *, collector_options):
        calls.append(revision.id)
        values = [(10, 13), (29, 26), (27, 31), (14, 12)][len(calls)-1]
        return {'status': 'passed', 'ranks': [
            {'rank': rank, 'result': {'valid': True, 'metrics': {'framework.elapsed_ms': value}}}
            for rank, value in enumerate(values)]}

    search = SimpleNamespace(output=tmp_path, execute=execute,
                             revisions=SimpleNamespace(revision=lambda _: parent))
    result = BindingCost().collect(search, candidate)
    assert calls == ['parent', 'child', 'child', 'parent']
    assert result['status'] == 'observed'
    assert result['elapsed_pairs'] == [{'pair': 0, 'parent': 13, 'passthrough': 29},
                                       {'pair': 1, 'parent': 14, 'passthrough': 31}]


def test_passthrough_does_not_install_every_affected_descendant(tmp_path):
    import torch
    from kernel_optimizer.project.binding import BindingSession
    from kernel_optimizer.project.discovery import tensor_description

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    x = torch.tensor([[1., -2.]])
    expected = model(x).detach().clone()
    adapter = SimpleNamespace(models=lambda state: {'model': model}, run=lambda state: model(state))
    specs = [{'id': 'root', 'kind': 'module', 'target': 'model', 'phase': 'run',
              'signatures': [{'args': tensor_description((x,)), 'kwargs': {}}]},
             {'id': 'child', 'kind': 'functional', 'target': 'aten.relu.default', 'phase': 'run',
              'signatures': []}]
    binding = BindingSession(adapter, x, specs)
    # The affected closure contains a functional descendant; the candidate only
    # replaces its enclosing module. The ablation must not create global dispatch.
    installed = binding.install(tmp_path/'not-executed.py', {}, diagnostic_passthrough=True,
                                passthrough_calls=['root'])
    try:
        assert installed == ['root']
        assert binding._mode is None
        torch.testing.assert_close(binding.invoke(), expected)
        assert dict(binding.counts) == {'root': 1}
    finally:
        binding.close()
    torch.testing.assert_close(adapter.run(x), expected)
