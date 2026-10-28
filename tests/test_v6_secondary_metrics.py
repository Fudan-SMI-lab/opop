"""Exercise optional diagnostic collection without changing the clean objective."""
import pytest

from kernel_optimizer.project.metric_diagnostics import collect, validate_sources
from kernel_optimizer.project.request import ProjectRequest
from kernel_optimizer.project.ports import Registry


def test_real_evaluator_controls_and_distinct_collector_evidence(tmp_path):
    from scripts.experiments.v6_secondary_metrics_checks_20261028 import run
    result = run(tmp_path/'integration')
    assert result['status'] == 'completed'
    assert result['inner_loop_score'] == 1.
    assert result['inner_loop_secondary'] == 30.


def test_omitted_metrics_keep_legacy_request_and_do_not_resolve_plugins(tmp_path, monkeypatch):
    def unexpected(*args):
        raise AssertionError('unused metric interface initialized a collector')
    monkeypatch.setattr('kernel_optimizer.project.collectors.configured', unexpected)
    validate_sources({}, {'collector_plugins': {'unused':'not.installed:Plugin'}}, {'name':'step'})
    request = ProjectRequest(source_root=tmp_path, entry='run.py', files=['run.py'], goal='step',
        objective={'name':'step'}, workload={}, quality={}, devices=[], lease_dir=tmp_path/'leases')
    assert 'diagnostic_metrics' not in request.portable()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True])
def test_invalid_collector_values_are_not_reported_as_observed(tmp_path, monkeypatch, bad):
    class InvalidCollector:
        metric_fields = {'field':{}}
        def metric(self, *args):
            return {'values':[1.,bad,2.], 'evidence':[]}
    registry=Registry();registry.register('fixture',InvalidCollector())
    monkeypatch.setattr('kernel_optimizer.project.collectors.configured',lambda _:registry)
    task={'objective':{'name':'step'},'diagnostic_metrics':{
        'aux':{'source':{'collector':'fixture','field':'field'},'unit':'ms'}}}
    metrics={'step':1.};samples={}
    with pytest.raises(ValueError,match='nonfinite'):
        collect(task,None,None,None,None,{'output':str(tmp_path)},metrics,samples)
    assert metrics=={'step':1.} and samples=={}
