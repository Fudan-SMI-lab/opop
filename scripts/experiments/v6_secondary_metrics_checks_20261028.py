"""Executable CPU integration checks for the isolated secondary-metric prototype."""
import copy
import json
from pathlib import Path


class FixtureCollector:
    metric_fields = {'value': {'unit':'count'}, 'bad': {'unit':'count'}}

    def metric(self, adapter, state, operation, synchronize, context, source, repeats):
        # The new sampling path must follow the full clean phase, without taking
        # ownership of its samples or replacing the objective.
        assert state['mode']=='profile' and state['measured']==5
        path=Path(context['output'])/'fixture.json'
        assert not path.exists(), 'two metric streams overwrote the same evidence'
        values=[]
        for _ in range(repeats):
            adapter.reset(state); result=operation();synchronize()
            assert result.tolist()==[2.,3.]
            values.append(7.)
        path.write_text(json.dumps(values))
        return {'values':values if source['field']=='value' else [],'evidence':[str(path)]}


def run(output):
    from kernel_optimizer.project.evaluation import worker_entry
    from kernel_optimizer.project.metric_diagnostics import collect, validate_sources
    from kernel_optimizer.project.request import ProjectRequest
    from kernel_optimizer.project.measurement import metric, paired_decision
    from kernel_optimizer.project.inner_loop import ProjectTaskEvaluator
    from types import SimpleNamespace
    output.mkdir(parents=True,exist_ok=False)
    adapter=output/'adapter.py'
    adapter.write_text('''import torch
def prepare(c,r): return {'x':torch.tensor([1.,2.]),'mode':'quality','measured':0}
def reset(s): pass
def set_mode(s,m): s['mode']=m
def run(s):
 if s['mode']=='measure': s['measured']+=1
 return s['x']+1
def observe(s,r): return r
def metrics(s,r,t): return {'step_ms':t,'fixture':float(r.sum())}
def models(s): return {}
''')
    profiling={'collector_plugins':{'fixture':'scripts.experiments.v6_secondary_metrics_checks_20261028:FixtureCollector'},'metric_repeats':3}
    task={'objective':{'name':'step_ms','direction':'minimize'},'quality':{},'warmup':2,'repeats':3,'continuous_steps':2,
          'profiling':profiling,'measurement_lifetime':'invocation'}
    spec={'source':{'collector':'fixture','field':'value'},'unit':'count','rank_aggregation':'max','aggregation_order':'ranks_then_samples'}
    def evaluate(name, request):
        dest=output/name;dest.mkdir()
        return worker_entry({'output':str(dest),'device':'cpu','rank':0,'world_size':1,
            'parameters':{'request':request,'adapter':str(adapter),'action':'controls'}})
    old=evaluate('off',task)
    assert old['valid'] and old['metric_samples']=={} and old['diagnostics']=={} and 'diagnostic_metrics' not in old
    enabled=evaluate('on',{**task,'diagnostic_metrics':{'aux1':spec,'aux2':spec}})
    assert enabled['valid'] and len(enabled['samples'])==3
    assert all(set(s)==set(old['samples'][0]) and not set(s)&{'aux1','aux2'} for s in enabled['samples'])
    assert enabled['metric_samples']=={'aux1':[7.]*3,'aux2':[7.]*3}
    assert enabled['diagnostic_metrics']['aux1']['evidence']!=enabled['diagnostic_metrics']['aux2']['evidence']
    both=evaluate('primary-and-secondary',{**task,'objective':{'name':'primary','direction':'minimize','source':spec['source']},'diagnostic_metrics':{'aux1':spec}})
    assert both['diagnostics']['evidence']!=both['diagnostic_metrics']['aux1']['evidence']
    assert both['metric_samples']=={'primary':[7.]*3,'aux1':[7.]*3}
    rejected=[]
    for name,request in [('collision',{**task,'diagnostic_metrics':{'step_ms':spec}}),
                         ('adapter-collision',{**task,'diagnostic_metrics':{'fixture':spec}}),
                         ('missing-samples',{**task,'diagnostic_metrics':{'aux':{**spec,'source':{'collector':'fixture','field':'bad'}}}}),
                         ('unavailable-field',{**task,'diagnostic_metrics':{'aux':{**spec,'source':{'collector':'fixture','field':'absent'}}}})]:
        try:evaluate(name,request)
        except ValueError as exc:rejected.append({'case':name,'reason':str(exc)})
        else:raise AssertionError('expected explicit error: '+name)
    # Alternating slow ranks exercise the extra metric's own aggregation order
    # through the actual inner-loop adapter, independently of the main score.
    endpoint={'status':'passed','ranks':[{'result':{'valid':True,'metrics':{'step_ms':1.,'aux':10.},
        'samples':[{'step_ms':1.}]*3,'metric_samples':{'aux':x}}} for x in [[30,10,10],[10,30,10]]]}
    request=ProjectRequest(source_root=output,entry='adapter.py',files=['adapter.py'],goal='test',objective=task['objective'],
                           workload={},quality={},devices=[],lease_dir=output/'leases',diagnostic_metrics={'aux':spec})
    coordinator=SimpleNamespace(request=request,revisions=SimpleNamespace(stage=lambda b:'revision'),
        execute=lambda *args:endpoint,prepared={'evaluation':endpoint})
    template=SimpleNamespace(id='fixture',model_copy=lambda update:SimpleNamespace(**update))
    measured=ProjectTaskEvaluator(coordinator,template).evaluate(None,{},{} )
    assert measured.valid and measured.score==1. and measured.metrics['aux']==30.
    return {'status':'completed','off':old,'on':enabled,'primary_and_secondary':both,'rejected':rejected,
        'inner_loop_score':measured.score,'inner_loop_secondary':measured.metrics['aux'],
        'scope':'Real CPU evaluator/collector integration fixture; not a GPU performance result or real agent claim.'}
