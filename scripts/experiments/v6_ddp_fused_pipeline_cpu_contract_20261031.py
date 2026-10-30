"""Once-only CPU boundary acquisition for exact generated eef828. No repair code."""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.distributed as dist

CODE='eef828816dd138cb69b0d12c960012b1970a0724b426a258ec6f8fcb6900abd7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Schedule:
    def __init__(self,ordered):
        self.ordered=ordered;self.current='compute';self.nodes=[];self.last={};self.events={};self.grad=[0.,0.,0.];self.weights=[10.,10.,10.];self.log=[];self.works=[]
    def enqueue(self,stream,label,action=lambda:None,deps=()):
        dependencies=set(deps)
        if stream in self.last:dependencies.add(self.last[stream])
        n=len(self.nodes);self.nodes.append({'stream':stream,'label':label,'action':action,'deps':dependencies,'done':False});self.last[stream]=n;return n
    @contextmanager
    def stream(self,s):
        previous=self.current;self.current=s.name
        try:yield
        finally:self.current=previous
    def stream_object(self,name):
        owner=self
        class Stream:
            def __init__(self):self.name=name
            def wait_event(self,event):owner.enqueue(self.name,'wait:'+event.name,deps=[owner.events[event.name]])
        return Stream()
    def event(self,name):
        owner=self
        class Event:
            def __init__(self):self.name=name
            def record(self):owner.events[name]=owner.enqueue(owner.current,'event:'+name)
        return Event()
    def graph(self,label,action=lambda:None):
        return SimpleNamespace(replay=lambda:self.enqueue(self.current,label,action))
    def tensor(self,i):
        owner=self
        class Tensor:
            def div_(self,value):owner.enqueue(owner.current,'divide:'+str(i),lambda:owner.grad.__setitem__(i,owner.grad[i]/value));return self
        tensor=Tensor();tensor.chunk=i;return tensor
    def all_reduce(self,tensor,op,async_op):
        assert async_op is True and op==dist.ReduceOp.SUM
        i=tensor.chunk;deps=[self.last[self.current]] if self.current in self.last else []
        task=self.enqueue('nccl','sum:'+str(i),lambda:self.grad.__setitem__(i,2*self.grad[i]),deps)
        if self.ordered:self.enqueue(self.current,'return-order:'+str(i),deps=[task])
        owner=self
        class Work:
            def wait(self):
                owner.log.append('host-work-wait:'+str(i));assert owner.nodes[task]['done'];return True
        work=Work();self.works.append(work);return work
    def synchronize(self):
        # One admissible order of the submitted dependency graph. No timing model.
        priority={'compute':0,'comm':1,'update':2,'nccl':3}
        while any(not n['done'] for n in self.nodes):
            ready=[(i,n) for i,n in enumerate(self.nodes) if not n['done'] and all(self.nodes[d]['done']for d in n['deps'])]
            assert ready,'Cycle in synthetic dependency graph'
            i,n=min(ready,key=lambda x:(priority[x[1]['stream']],x[0]))
            n['action']();n['done']=True;self.log.append(n['label'])
        self.log.append('device-synchronize-return')

def exact_fast_case(module,ordered):
    schedule=Schedule(ordered)
    with patch.object(torch.cuda,'is_available',lambda:False):
        mapping=module.build({'references':{module.PROJECT_CALL_ID:lambda st:st},'world_size':2})
    replacement=mapping[module.PROJECT_CALL_ID]
    cells=dict(zip(replacement.__code__.co_freevars,[x.cell_contents for x in replacement.__closure__]))
    plan=cells['plan'];fast=cells['_fast_step'];views=[schedule.tensor(i) for i in range(3)]
    loss=SimpleNamespace();loss.detach=lambda:loss;loss.clone=lambda:loss
    plan.update(params=[SimpleNamespace(grad=None)for _ in views],views=views,world_size=2,static_loss=loss,
        graph_fwd=schedule.graph('forward'),seg_graphs=[schedule.graph('backward:'+str(i),lambda i=i:schedule.grad.__setitem__(i,2.))for i in range(3)],
        upd_graphs=[schedule.graph('update:'+str(i),lambda i=i:schedule.weights.__setitem__(i,schedule.weights[i]-.001*schedule.grad[i]))for i in range(3)],
        chunk_tensors=views,comm_stream=schedule.stream_object('comm'),update_stream=schedule.stream_object('update'),
        ready_events=[schedule.event('ready:'+str(i))for i in range(3)],done_events=[schedule.event('done:'+str(i))for i in range(3)])
    originals=[torch.cuda.stream,torch.cuda.current_stream,torch.cuda.synchronize,dist.all_reduce]
    with patch.object(torch.cuda,'stream',schedule.stream),patch.object(torch.cuda,'current_stream',lambda:schedule.stream_object(schedule.current)),patch.object(torch.cuda,'synchronize',schedule.synchronize),patch.object(dist,'all_reduce',schedule.all_reduce):
        result=fast()
    assert originals==[torch.cuda.stream,torch.cuda.current_stream,torch.cuda.synchronize,dist.all_reduce]
    expected=[9.998]*3
    return {'ordered_completion_control':ordered,'actual_exact_nested_fast_step':True,'result_loss_same_object':result['loss']is loss,
        'gradients':schedule.grad,'weights':schedule.weights,'expected_weights':expected,'max_weight_error':max(abs(a-b)for a,b in zip(schedule.weights,expected)),
        'execution_order':schedule.log,'dependency_edges':[{'id':i,'stream':n['stream'],'label':n['label'],'dependencies':sorted(n['deps'])}for i,n in enumerate(schedule.nodes)],
        'updates_before_corresponding_sum':[i for i in range(3) if schedule.log.index('update:'+str(i))<schedule.log.index('sum:'+str(i))],
        'scope':'Exact candidate nested _fast_step with inert graph/stream/event/Work and equal-peer SUM scheduler; no CUDA/NCCL/real collective/model. Negative is a permitted partial-order example, not historical GPU timing or failure.'}

def cpu_saved_parameter_case(module,source):
    from transformers import GPT2Config,GPT2LMHeadModel
    torch.set_num_threads(2);torch.manual_seed(20261020)
    config=GPT2Config(n_layer=12,n_embd=16,n_head=4,vocab_size=101,n_positions=32,resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,use_cache=False,bos_token_id=None,eos_token_id=None,attn_implementation='eager')
    model=GPT2LMHeadModel(config).float().train();params=list(model.parameters());names=[n for n,_ in model.named_parameters()]
    ids=[module._classify_param(n)for n in names];chunks=[[p for p,c in zip(params,ids)if c==i]for i in range(3)]
    assert len(params)==148 and all(p.device.type=='cpu'for p in params)
    tokens=torch.randint(0,101,(4,8),generator=torch.Generator().manual_seed(31000))
    loss=model(input_ids=tokens,labels=tokens,use_cache=False).loss;loss.backward();reference=[p.grad.clone()for p in params];model.zero_grad(set_to_none=True)
    node=next(n for n in ast.walk(ast.parse(source))if isinstance(n,ast.FunctionDef)and n.name=='_chunked_backward')
    literal=ast.get_source_segment(source,node);namespace={'torch':torch,'chunk_params':chunks}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact-candidate-chunked-backward>','exec'),namespace)
    namespace['_chunked_backward'](model(input_ids=tokens,labels=tokens,use_cache=False).loss)
    full_error=max(float((p.grad-g).abs().max())for p,g in zip(params,reference))
    full_matches=all(torch.allclose(p.grad,g,atol=1e-6,rtol=1e-4)for p,g in zip(params,reference));model.zero_grad(set_to_none=True)
    original=[p.detach().clone()for p in params];steps=[]
    class Interleaving:
        def __iter__(self):
            for i,chunk in enumerate(chunks):
                steps.append('before-backward:'+str(i));yield chunk;steps.append('after-backward:'+str(i))
                if i==0:
                    with torch.no_grad():
                        for p in chunk:p.add_(p.grad,alpha=-.001)
                    steps.append('early-update:0')
    namespace['chunk_params']=Interleaving();failure=None
    try:namespace['_chunked_backward'](model(input_ids=tokens,labels=tokens,use_cache=False).loss)
    except RuntimeError as exc:failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        with torch.no_grad():
            for p,value in zip(params,original):p.copy_(value)
    return {'cpu_model':'GPT2 12layers/hidden16/heads4/vocab101/4x8tokens FP32 eager; synthetic smaller model only',
        'parameters':len(params),'elements':sum(p.numel()for p in params),'chunk_parameter_counts':[len(c)for c in chunks],
        'classifier_exact':True,'helper_source_sha256':hashlib.sha256(literal.encode()).hexdigest(),
        'positive_all_segments_before_update_max_abs_grad_error':full_error,'positive_all_full_gradients_match_original_tolerance':full_matches,'interleaving':steps,'early_update_then_next_backward_exception':failure,
        'weights_restored':all(torch.equal(p.detach(),v)for p,v in zip(params,original)),
        'scope':'Exact original classifier and _chunked_backward helper; CPU autograd positive control plus one injected scheduling order (update first chunk before next backward). Not full candidate binding/build/setup, no CUDA Graph, DDP/NCCL, original full model/quality, or historical GPU failure. Error demonstrates saved parameter consumer dependency; CUDA replay behavior remains unmeasured.'}

def main(args):
    assert sha(args.candidate)==CODE and not args.output.exists()
    assert not torch.cuda.is_initialized()
    original_available=torch.cuda.is_available
    source=args.candidate.read_text();spec=importlib.util.spec_from_file_location('exact_eef828',args.candidate);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args.output.mkdir(parents=True)
    with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA initialization forbidden by CPU fixture')):
        ordered=exact_fast_case(module,True);deferred=exact_fast_case(module,False);consumer=cpu_saved_parameter_case(module,source)
    api=inspect.getsource(dist.all_reduce)
    (args.output/'installed-all-reduce-source.txt').write_text(api)
    report={'at':datetime.now(timezone.utc).isoformat(),'candidate_sha256':CODE,'candidate_bytes':args.candidate.stat().st_size,'fixture_sha256':sha(__file__),
        'torch_version':torch.__version__,'torch_cuda_initialized':torch.cuda.is_initialized(),'is_available_restored':torch.cuda.is_available is original_available,
        'installed_all_reduce_file':inspect.getsourcefile(dist.all_reduce),'installed_all_reduce_source_sha256':hashlib.sha256(api.encode()).hexdigest(),
        'ordered_completion_positive_control':ordered,'deferred_collective_completion':deferred,'saved_parameter_consumer':consumer,
        'limits':'Once-only CPU acquisition. Exact candidate unchanged. No GPU/model-source/real distributed collective/reference controls/full-quality run. Synthetic Work/event ordering is not a measured NCCL schedule; small CPU model counterexample is not a historical GPU trace. No repair implementation supplied or performance/acceptance conclusion.'}
    (args.output/'result.json').write_text(json.dumps(report,indent=2))
    assert ordered['max_weight_error']==0 and deferred['max_weight_error']>0
    assert consumer['positive_all_full_gradients_match_original_tolerance']
    assert not torch.cuda.is_initialized() and sha(args.candidate)==CODE
    print(json.dumps({'result_sha256':sha(args.output/'result.json'),'ordered_error':ordered['max_weight_error'],'deferred_error':deferred['max_weight_error'],'consumer':consumer,'cuda_initialized':torch.cuda.is_initialized()}))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
