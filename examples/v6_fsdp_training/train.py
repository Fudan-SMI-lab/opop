"""Seeded GPT-2 Medium FP32 training with PyTorch's default FSDP2 policy.

One invocation is zero_grad, full rank-local forward/loss, backward with
distributed gradient averaging, and one SGD update. Quality is outside timing.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import GPT2Config, GPT2LMHeadModel


def policy():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def build_model(spec, device):
    torch.manual_seed(spec["seed"])
    config = GPT2Config(**spec["model"], resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False)
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).to(device=device, dtype=torch.float32).train()


def inputs(spec, rank, device):
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"] + 100 + rank)
    return torch.randint(spec["model"]["vocab_size"],
        (spec["batch_per_rank"], spec["sequence_length"]), generator=generator).to(device)


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def prepare(spec, device):
    model = build_model(spec, device)
    # Official bottom-up composable sharding. No forced serialization, small
    # buckets, manual per-layer barriers, or disabled default prefetch.
    for block in model.transformer.h:
        fully_shard(block)
    fully_shard(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec["learning_rate"], momentum=0., weight_decay=0., foreach=False)
    return {"model": model, "optimizer": optimizer, "input_ids": inputs(spec, dist.get_rank(), device),
        "initial": {n: local(p).detach().cpu().clone() for n, p in model.named_parameters()}, "spec": spec}


@torch.no_grad()
def reset(state):
    # Full initial values, outside the timer; no empty_cache or forced GC.
    for name, parameter in state["model"].named_parameters():
        local(parameter).copy_(state["initial"][name])
    state["optimizer"].zero_grad(set_to_none=True)


def step(state):
    state["optimizer"].zero_grad(set_to_none=True)
    result = state["model"](input_ids=state["input_ids"], labels=state["input_ids"], use_cache=False)
    result.loss.backward()
    state["optimizer"].step()
    return {"loss": result.loss.detach(), "logits": result.logits.detach()}


def observe(state, result):
    return {"loss": result["loss"].cpu(), "logits": result["logits"].cpu(),
        "gradients": {n: local(p.grad).detach().cpu().clone() for n,p in state["model"].named_parameters()},
        "parameters": {n: local(p).detach().cpu().clone() for n,p in state["model"].named_parameters()}}


def reference_shard(full, parameter):
    if not isinstance(parameter, DTensor):
        return full
    placements = parameter.placements
    if len(placements) != 1 or not placements[0].is_shard():
        raise ValueError("Independent fixture needs a reviewed mapping for this layout")
    return torch.chunk(full, dist.get_world_size(), dim=placements[0].dim)[dist.get_rank()]


def compare(actual, expected, quality):
    assert actual.shape == expected.shape
    a, b = actual.detach(), expected.detach()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"passed": finite and torch.allclose(a,b,atol=quality["atol"],rtol=quality["rtol"]),
        "elements": a.numel(), "max_abs_error": float((a-b).abs().max()) if a.numel() else 0.}


def verify(state, output):
    """Independent unsharded serial-global-batch trajectory, two full updates.

    Reference differentiates both rank inputs with 1/world loss weighting and
    does not call FSDP or the sharded step(). Gradients and post-update parameters
    are compared in their actual DTensor layout, not assumed flat slices.
    """
    spec = state["spec"]; device = state["input_ids"].device
    reference = build_model(spec, device)
    optimizer = torch.optim.SGD(reference.parameters(), lr=spec["learning_rate"], momentum=0., weight_decay=0., foreach=False)
    refs = dict(reference.named_parameters()); records=[]
    reset(state)
    for index in range(spec["quality"]["continuous_updates"]):
        optimizer.zero_grad(set_to_none=True); expected=None
        for rank in range(dist.get_world_size()):
            ids=inputs(spec,rank,device)
            out=reference(input_ids=ids,labels=ids,use_cache=False)
            (out.loss/dist.get_world_size()).backward()
            if rank==dist.get_rank():
                expected={"loss":out.loss.detach().clone(),"logits":out.logits.detach().clone()}
            del out
        optimizer.step()
        actual=step(state); checks={key:compare(actual[key],expected[key],spec["quality"]) for key in actual}
        for name,p in state["model"].named_parameters():
            r=refs[name]
            checks["gradient:"+name]=compare(local(p.grad),reference_shard(r.grad,p),spec["quality"])
            checks["parameter:"+name]=compare(local(p),reference_shard(r,p),spec["quality"])
        assert set(refs)==set(dict(state["model"].named_parameters()))
        negative = not compare(actual["logits"]+1.,expected["logits"],spec["quality"])["passed"]
        row={"step":index,"checks":checks,"negative_rejected":negative,
            "tied_embedding":state["model"].transformer.wte.weight is state["model"].lm_head.weight}
        row["passed"]=all(c["passed"] for c in checks.values()) and negative and row["tied_embedding"]
        records.append(row)
        # Durable full tensors for later independent adapter fidelity. Only
        # written on the full readiness run, never copied to a local machine.
        if output is not None:
            torch.save(observe(state,actual),output/f"source-step-{index}-rank-{dist.get_rank()}.pt")
        del actual,expected
    del reference,optimizer,refs
    reset(state)
    return records


def measured(state):
    reset(state);dist.barrier();torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter_ns();result=step(state);torch.cuda.synchronize();end=time.perf_counter_ns()
    # Result is still alive while the memory metric is sampled.
    row={"step_ms":(end-begin)/1e6,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
         "allocated_bytes":torch.cuda.memory_allocated()}
    del result
    return row


def main():
    import os
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--api-fixture',action='store_true');parser.add_argument('--profile',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    policy();torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl');spec=json.loads(Path(__file__).with_name('workload.json').read_text())
    if args.api_fixture:
        spec=copy.deepcopy(spec);spec['model']={'n_layer':2,'n_embd':128,'n_head':4,'vocab_size':256,'n_positions':64};spec['sequence_length']=16
    assert dist.get_world_size()==spec['world_size']
    state=prepare(spec,device);quality=verify(state,None if args.api_fixture else args.output)
    rank=dist.get_rank();record={'rank':rank,'spec':spec,'quality':quality,'scope':'API fixture only' if args.api_fixture else 'Full source readiness, not optimization acceptance',
        'layout':{n:{'shape':list(p.shape),'local_shape':list(local(p).shape),'placements':str(p.placements)} for n,p in state['model'].named_parameters()}}
    passed=torch.tensor(int(all(r['passed'] for r in quality)),device=device);dist.all_reduce(passed,op=dist.ReduceOp.MIN)
    record['passed']=bool(passed.item());(args.output/f'quality-rank-{rank}.json').write_text(json.dumps(record,indent=2))
    if not record['passed']:
        dist.destroy_process_group();raise RuntimeError('Source fidelity failed; retain all comparisons and original tolerance')
    for _ in range(spec['warmup']):measured(state)
    samples=[measured(state) for _ in range(spec['repeats'])];all_samples=[None]*dist.get_world_size();dist.all_gather_object(all_samples,samples)
    record['samples']=samples;record['max_rank_samples']={k:[max(s[i][k] for s in all_samples) for i in range(spec['repeats'])] for k in samples[0]}
    record['metrics']={k:statistics.median(v) for k,v in record['max_rank_samples'].items()}
    record['communication_profiles']=[]
    if args.profile:
        for index in range(spec['profile_repeats']):
            reset(state);dist.barrier();torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('source.complete_training_step'):
                    result=step(state);torch.cuda.synchronize()
            path=args.output/f'profile-{index}-rank-{rank}.json';prof.export_chrome_trace(str(path));record['communication_profiles'].append(str(path));del result,prof
    (args.output/f'result-rank-{rank}.json').write_text(json.dumps(record,indent=2));dist.destroy_process_group()


if __name__=='__main__':main()
