"""Execution-policy candidate for the GPT-2 124M FP32 DDP accumulated_step project.

Replaces project.invoke with the same measured step (zero_grad, four
forward/backward micro-batches with DDP no_sync on the first three, one SGD
optimizer.step(), closing synchronize) but re-sequences the step tail so that
the loss stack/sum and the per-parameter SGD add_ launches are enqueued on a
consumer CUDA stream while the final micro-batch's DDP bucket allreduces are
still in flight. Collective count, payload, ReduceOp, gradients and all
update/loss values are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase as _phase
except Exception:  # optional instrumentation; keep a clean timing path

    @contextmanager
    def _phase(name):
        yield


PROJECT_INVOKE_CALL_ID = "5d7d6bb13691b02f533f62b0"


class _TailPlan:
    def __init__(self, device):
        self.device = device
        self.usable = False
        self.stream = None
        self.losses_ready = None
        self.grads_ready = None
        if device.type == "cuda":
            try:
                self.stream = torch.cuda.Stream(device=device)
                self.losses_ready = torch.cuda.Event()
                self.grads_ready = torch.cuda.Event()
                self.usable = True
            except Exception:
                self.usable = False


def _make_run(plan):
    def run_policy(state):
        model = state["model"]
        optimizer = state["optimizer"]
        inputs = state["inputs"]
        device = state["device"]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        total = len(inputs)
        for i, tokens in enumerate(inputs):
            is_final = i + 1 == total
            context = model.no_sync() if isinstance(model, DDP) and not is_final else nullcontext()
            with context, _phase("accumulate_micro_step_final_allreduce" if is_final else "accumulate_micro_step_local"):
                loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / total
                losses.append(loss.detach())
                if is_final and plan.usable:
                    plan.losses_ready.record()
                loss.backward()
        if plan.usable:
            plan.grads_ready.record()
            stream = plan.stream
            stream.wait_event(plan.losses_ready)
            with torch.cuda.stream(stream):
                micro_losses = torch.stack(losses)
                total_loss = micro_losses.sum()
            stream.wait_event(plan.grads_ready)
            with torch.cuda.stream(stream), _phase("optimizer_step"):
                optimizer.step()
            torch.cuda.synchronize(device)
        else:
            micro_losses = torch.stack(losses)
            total_loss = micro_losses.sum()
            with _phase("optimizer_step"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        assert torch.isfinite(total_loss)
        return {"loss": total_loss, "micro_losses": micro_losses}

    return run_policy


def build(context):
    state = context["state"]
    device = state.get("device")
    if not isinstance(device, torch.device):
        device = torch.device(device if device is not None else "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    plan = _TailPlan(device)
    return {PROJECT_INVOKE_CALL_ID: _make_run(plan)}
