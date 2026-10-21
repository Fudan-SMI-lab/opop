"""Whole-bundle execution-policy replacement for the complete-workload project
call 5d7d6bb13691b02f533f62b0 (group 3c57c8731ec33ed76cbb).

Repair of trial-f684195486 (revision 43b0f446...): that attempt had the same
single direct project.invoke boundary and the same plan - no_sync micro-batches
as captured CUDA graph replays, eager synchronized final micro-batch through
the unmodified DDP reducer - but graph capture died in the parent's own forward
at transformers 5.16.1 masking_utils.py:603 (eager_mask): 
torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
raises "Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned" because torch.tensor(..., device=cuda) builds a
pageable CPU scalar and copies it host-to-device. Every invocation then fell
back to the parent (route fallback:parent:capture_failed:RuntimeError), the
objective regressed (65.19/81.10 endpoint kernel sums, metric 73.14 vs
reference 66.55) and the step constraint failed (145.55/144.25 vs limit
144.80).

This candidate fixes the capture path instead of changing the plan:

1. The captured/warmed segments call the underlying module with an explicit
   precomputed 4D causal attention mask. transformers 5.16.1
   masking_utils._preprocess_mask_arguments returns an already-prepared 4D mask
   as-is (create_causal_mask early exit), so eager_mask and its uncapturable
   torch.tensor(0.0, device=cuda) are never executed inside capture, while the
   eager attention path just adds the same 4D mask
   (attn_weights + mask). The mask is built out of capture as
   where(lower_triangular_bool, 0, finfo(float32).min), verified against the
   installed stack: torch.equal to create_causal_mask's own output, and the
   masked forward reproduces the parent forward's loss, logits and every
   gradient bit-exactly on the real GPT2Config(attn_implementation="eager").
2. During the capture (warmup) invocation the mask path is first validated on
   the live current inputs (two no_grad forwards compared with torch.equal);
   any mismatch aborts the graph path before any capture attempt.
3. Each segment is executed eagerly once on a side stream (primed in-place
   gradient buffers) before capture, so library lazy init and any eager-side
   failure happen outside the capture region and a failure leaves no
   half-captured state.

Everything else is the previously CPU-verified (two-rank gloo) design: the
first invocation runs the pure parent reference; the second invocation
(documented untimed warmup) captures one graph per no_sync micro-batch into a
shared pool and finishes through replay + the eager final micro-batch; later
invocations are replay-only. Per invocation the static input buffers are
refreshed from the current inputs and gradients accumulate in place into
persistent per-parameter buffers primed and zeroed before each replay. The
final synchronized micro-batch runs eagerly through the unmodified DDP wrapper
(same 13 fp32 bucket allreduces on the default NCCL process group, reducer,
bucket_cap_mb=25, gradient_as_bucket_view=True untouched), followed by the SGD
step and the single final torch.cuda.synchronize, with
optimizer.zero_grad(set_to_none=True) inside the timed boundary exactly as the
parent. Any failure (mask mismatch, warmup, capture, replay, shape change)
permanently routes the whole invocation through the parent reference with a
distinct label; the collective-bearing final micro-batch and optimizer step are
never re-run by a fallback. No barriers, no extra collectives, no comm hooks,
no environment or process-group changes.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from kernel_optimizer.project.discovery import phase
except ImportError:
    def phase(name):
        return nullcontext()

KNOWN_PROJECT_CALL = "5d7d6bb13691b02f533f62b0"

_LABEL_FIRST = "eager:parent:first_invocation"
_LABEL_CAPTURE = "capture:mask_passthrough_graphs+eager_final_micro"
_LABEL_REPLAY = "replay:mask_passthrough_graphs+eager_final_micro"


class _SetupFailure(RuntimeError):
    """Setup-stage failure (mask equivalence check or eager warmup) that
    aborts the graph path before any capture attempt; carries the exact
    fallback label suffix."""


def _project_call_id(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    if KNOWN_PROJECT_CALL in targets and KNOWN_PROJECT_CALL in references:
        return KNOWN_PROJECT_CALL
    for call_id, spec in targets.items():
        if (
            isinstance(spec, dict)
            and spec.get("kind") == "project"
            and spec.get("target") == "project.invoke"
            and call_id in references
        ):
            return call_id
    return None


def _build_causal_mask(sequence_length, device, dtype):
    """Bit-exact replica of transformers 5.16.1 create_causal_mask/eager_mask
    output for the parent's calls (attention_mask=None, no cache, no padding):
    float tensor of shape (1, 1, q, kv) with 0.0 where attention participates
    and finfo(dtype).min where masked. Verified torch.equal against
    create_causal_mask on the installed stack; must be built outside capture
    (it materializes scalar constants)."""
    positions = torch.arange(sequence_length, device=device)
    keep = positions[:, None] >= positions[None, :]
    keep = keep.view(1, 1, sequence_length, sequence_length)
    min_dtype = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, sequence_length, sequence_length), device=device, dtype=dtype)
    mask = torch.where(keep, mask, torch.full_like(mask, min_dtype))
    return mask


def build(context):
    context = context or {}
    record = context.get("record_route") or (lambda *_: None)
    references = context.get("references") or {}
    state = context.get("state") or {}

    call_id = _project_call_id(context)
    reference = references.get(call_id) if call_id is not None else None
    if call_id is None or reference is None:
        return {}

    model = state.get("model")
    raw_model = state.get("raw_model") or model
    inputs = state.get("inputs")
    if (
        not isinstance(model, DDP)
        or not isinstance(raw_model, torch.nn.Module)
        or state.get("optimizer") is None
        or not isinstance(inputs, (list, tuple))
        or len(inputs) < 2
        or not torch.cuda.is_available()
    ):
        return {}

    book = {
        "failed": None,
        "model": None,
        "raw_model": None,
        "params": None,
        "buffers": None,
        "graphs": None,
        "static_masks": None,
        "static_inputs": None,
        "static_losses": None,
    }

    def _fallback(state, label):
        record(call_id, label)
        return reference(state)

    def _prime_and_clear(params, buffers):
        for parameter, buffer in zip(params, buffers):
            parameter.grad = buffer
        try:
            torch._foreach_zero_(buffers)
        except Exception:
            for buffer in buffers:
                buffer.zero_()

    def _check_shapes(inputs):
        graphs = book["graphs"]
        static_inputs = book["static_inputs"]
        static_masks = book["static_masks"]
        if len(inputs) - 1 != len(graphs):
            return False
        for index in range(len(graphs)):
            if (
                inputs[index].shape != static_inputs[index].shape
                or inputs[index].dtype != static_inputs[index].dtype
                or inputs[index].device != static_inputs[index].device
                or static_masks[index].shape[-1] != inputs[index].shape[-1]
            ):
                return False
        return True

    def _run_segments(inputs):
        graphs = book["graphs"]
        static_inputs = book["static_inputs"]
        static_losses = book["static_losses"]
        losses = []
        for index, graph in enumerate(graphs):
            static_inputs[index].copy_(inputs[index], non_blocking=True)
            graph.replay()
            losses.append(static_losses[index].detach().clone())
        return losses

    def _verify_mask_equivalence(raw_model, tokens, mask):
        """Two eager no_grad forwards on the live current inputs: parent-style
        (internal create_causal_mask) versus explicit 4D mask passthrough.
        Rejects the graph path unless loss and logits are bit-identical."""
        with torch.no_grad():
            parent_out = raw_model(input_ids=tokens, labels=tokens, use_cache=False)
            masked_out = raw_model(
                input_ids=tokens, labels=tokens, use_cache=False, attention_mask=mask
            )
        return torch.equal(parent_out.loss, masked_out.loss) and torch.equal(
            parent_out.logits, masked_out.logits
        )

    def _warm_segments(raw_model, masks, static_inputs, total):
        """Eager warmup of the exact captured code path on a side stream with
        primed gradient buffers (in-place accumulation flavor), so lazy
        library init and any eager-side failure happen outside capture."""
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for index in range(len(static_inputs)):
                output = raw_model(
                    input_ids=static_inputs[index],
                    labels=static_inputs[index],
                    use_cache=False,
                    attention_mask=masks[index],
                )
                segment_loss = output.loss / total
                segment_loss.backward()
        torch.cuda.current_stream().wait_stream(warm)

    def _capture_segments(raw_model, optimizer, inputs):
        params = book["params"]
        if book["buffers"] is None:
            book["buffers"] = [torch.zeros_like(parameter) for parameter in params]
        buffers = book["buffers"]
        segments = len(inputs) - 1
        total = len(inputs)
        dtype = next(raw_model.parameters()).dtype
        device = inputs[0].device
        static_masks = [
            _build_causal_mask(int(tokens.shape[-1]), device, dtype) for tokens in inputs[:segments]
        ]
        for index in range(segments):
            try:
                equivalent = _verify_mask_equivalence(raw_model, inputs[index], static_masks[index])
            except Exception as exc:
                raise _SetupFailure("setup_failed:mask_check_error:" + type(exc).__name__) from exc
            if not equivalent:
                raise _SetupFailure("setup_failed:mask_not_bit_exact")
        optimizer.zero_grad(set_to_none=True)
        _prime_and_clear(params, buffers)
        static_inputs = [inputs[index].detach().clone() for index in range(segments)]
        try:
            _warm_segments(raw_model, static_masks, static_inputs, total)
        except Exception as exc:
            raise _SetupFailure("setup_failed:warmup_failed:" + type(exc).__name__) from exc
        _prime_and_clear(params, buffers)
        static_losses = []
        graphs = []
        pool = torch.cuda.graph_pool_handle()
        for index in range(segments):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                output = raw_model(
                    input_ids=static_inputs[index],
                    labels=static_inputs[index],
                    use_cache=False,
                    attention_mask=static_masks[index],
                )
                segment_loss = output.loss / total
                segment_loss.backward()
            static_losses.append(segment_loss)
            graphs.append(graph)
        for parameter, buffer in zip(params, buffers):
            if parameter.grad is not buffer:
                raise RuntimeError("out-of-place gradient accumulation detected")
        book["graphs"] = graphs
        book["static_masks"] = static_masks
        book["static_inputs"] = static_inputs
        book["static_losses"] = static_losses
        return _run_segments(inputs)

    def _replay_segments(optimizer, inputs):
        if not _check_shapes(inputs):
            raise RuntimeError("workload_shape_changed")
        optimizer.zero_grad(set_to_none=True)
        _prime_and_clear(book["params"], book["buffers"])
        return _run_segments(inputs)

    def _eager_final_micro(model, inputs):
        tokens = inputs[-1]
        loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss / len(inputs)
        loss.backward()
        return loss.detach()

    def _discard_graphs():
        book["graphs"] = None
        book["static_masks"] = None
        book["static_inputs"] = None
        book["static_losses"] = None
        book["buffers"] = None

    def replacement(state):
        if book["failed"] is not None:
            return _fallback(state, "fallback:parent:steady_state:" + book["failed"])

        model = state.get("model")
        raw_model = state.get("raw_model") or model
        optimizer = state.get("optimizer")
        inputs = state.get("inputs")
        if (
            not isinstance(model, DDP)
            or not isinstance(raw_model, torch.nn.Module)
            or optimizer is None
            or not isinstance(inputs, (list, tuple))
            or len(inputs) < 2
        ):
            return _fallback(state, "fallback:parent:unsupported_signature")

        if book["params"] is None:
            result = reference(state)
            try:
                book["model"] = model
                book["raw_model"] = raw_model
                book["params"] = list(raw_model.parameters())
            except Exception:
                book["failed"] = "snapshot_failed"
            record(call_id, _LABEL_FIRST)
            return result

        if model is not book["model"] or raw_model is not book["raw_model"]:
            return _fallback(state, "fallback:parent:model_replaced")

        if book["graphs"] is not None:
            failure = None
            with phase("gradient_accumulation"):
                try:
                    losses = _replay_segments(optimizer, inputs)
                except Exception as exc:
                    book["failed"] = "replay_failed:" + type(exc).__name__
                    failure = "fallback:parent:" + book["failed"]
                if failure is None:
                    losses.append(_eager_final_micro(model, inputs))
                    loss = torch.stack(losses).sum()
            if failure is not None:
                return _fallback(state, failure)
            with phase("optimizer_step"):
                optimizer.step()
                torch.cuda.synchronize()
            record(call_id, _LABEL_REPLAY)
            return {"loss": loss.detach(), "micro_losses": losses}

        failure = None
        with phase("gradient_accumulation"):
            try:
                losses = _capture_segments(raw_model, optimizer, inputs)
            except Exception as exc:
                _discard_graphs()
                if isinstance(exc, _SetupFailure):
                    book["failed"] = str(exc)
                else:
                    book["failed"] = "capture_failed:" + type(exc).__name__
                failure = "fallback:parent:" + book["failed"]
            if failure is None:
                losses.append(_eager_final_micro(model, inputs))
                loss = torch.stack(losses).sum()
        if failure is not None:
            return _fallback(state, failure)
        with phase("optimizer_step"):
            optimizer.step()
            torch.cuda.synchronize()
        record(call_id, _LABEL_CAPTURE)
        return {"loss": loss.detach(), "micro_losses": losses}

    return {call_id: replacement}
