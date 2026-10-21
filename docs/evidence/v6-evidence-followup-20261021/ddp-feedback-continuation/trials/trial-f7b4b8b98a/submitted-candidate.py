"""Whole-bundle execution-policy replacement for the complete-workload project
call 5d7d6bb13691b02f533f62b0 (group 3c57c8731ec33ed76cbb).

Repair of trial-93aa5d5fc1 (trial-93aa5d5fc1-param-4fabdd0b): same single
direct project.invoke boundary, same plan - three no_sync micro-batches as
captured CUDA graph replays plus an eager synchronized final micro-batch
through the unmodified DDP reducer - and quality/constraint already passed,
but every replay invocation fell back to the parent with route
fallback:parent:capture_failed:RuntimeError, so the paired gain was parent
vs parent noise (0.41 vs 6.67 ms bound) and the bundle was rejected for
paired_gain_not_above_noise.  candidate_diagnostics-4123e6d6 records the exact
exception at candidate.py line 262 (segment_loss.backward() inside
torch.cuda.graph capture):

  RuntimeError: During CUDA graph capture, autograd node
  'torch::autograd::AccumulateGrad' has a stale reference to the default
  stream (stream 0) from warmup. This will invalidate the capture because
  cudaStreamWaitEvent on the default stream pulls a non-capturing stream into
  the graph. To fix, either: (a) Run warmup on the same stream that capture
  will use, or (b) Delete references to the loss / autograd graph (e.g.
  `del loss`) before capture, or (c) Call
  torch.autograd.graph.set_override_stale_capture_stream(True) ...

The stale default-stream references come from invocation-1's parent backward
on stream 0 and from the old code warming on an unrelated side stream while
torch.cuda.graph captured on its own internal stream.  This candidate applies
the error message's own fixes, all verified present in the measured worker
(torch 2.13.0+cu129, /root/autodl-tmp/dev-venv/bin/python):

(a) ONE explicit side stream S is used for both the eager warmup
    (with torch.cuda.stream(S)) and the capture of every segment, passed as
    torch.cuda.graph(graph, pool=pool, stream=S) - the documented pattern for
    a shared pool, which torch 2.13's graph docstring states explicitly.
(c) torch.autograd.graph.set_override_stale_capture_stream(True) is enabled
    around the capture section so remaining stale default-stream AccumulateGrad
    references are redirected to the capturing stream, and restored to the
    documented process-initial False in a finally block.
(b) warmup loss/output/autograd-graph references are dropped (del) before
    capture, and the captured static losses are stored detached.

Everything else keeps the previously validated design: explicit precomputed
4D causal mask (bypasses the transformers 5.16.1 masking_utils.py eager_mask
unpinned CPU-scalar copy that killed trial-f684195486's captures;
_preprocess_mask_arguments returns an already-prepared 4D mask as-is, and the
eager attention path just adds the mask, attn_weights + attention_mask);
bit-exactness re-verified on the live current inputs before any capture
attempt; eager warmup outside capture surfacing lazy init; per-parameter
primed in-place gradient buffers (124,439,808 fp32 elements, about 497.8 MB);
one graph per no_sync micro-batch in a shared pool replayed in capture order;
static input buffers refreshed from the current inputs on every invocation;
eager final micro-batch through the unmodified DDP wrapper (same 13 fp32
bucket allreduces, reducer, bucket_cap_mb=25, gradient_as_bucket_view=True,
broadcast_buffers=False untouched); SGD step and the single final
torch.cuda.synchronize; optimizer.zero_grad(set_to_none=True) inside the
timed boundary exactly as the parent.  Any failure (mask mismatch, warmup,
capture, replay, shape change) permanently routes the whole invocation
through the parent reference with a distinct label; the collective-bearing
final micro-batch and optimizer step are never re-run by a fallback, and both
ranks always execute the same eager 13-allreduce final micro-batch, so
collective participation is rank-symmetric under mixed success/failure.

CPU-verified semantics on the installed stack (two-rank gloo DDP with
gradient_as_bucket_view=True, bucket_cap_mb=25, broadcast_buffers=False):
the interleaving pattern (prime standalone per-parameter grad buffers ->
three raw-module backwards -> one synced DDP backward -> SGD step) reproduces
the parent pattern's loss, micro losses, every gradient and every parameter
bit-exactly, across consecutive invocations with weight resets in between;
the 4D mask forward is bit-identical to the mask=None forward (logits, loss,
gradients; mask tensor torch.equal to create_causal_mask's output).
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
_LABEL_CAPTURE = "capture:stale_stream_repaired+eager_final_micro"
_LABEL_REPLAY = "replay:stale_stream_repaired+eager_final_micro"

_SET_OVERRIDE = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)


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
    record = context.get("record_route", lambda *_: None)
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

    def _warm_segments(raw_model, masks, static_inputs, total, stream):
        """Eager warmup of the exact captured code path on the SAME explicit
        stream that capture will use (stale-stream fix (a)), so library lazy
        init, stream-liveness warmup and any eager-side failure happen outside
        capture on the eventual capturing stream.  Loss and output references
        are dropped before capture (stale-stream fix (b))."""
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for index in range(len(static_inputs)):
                output = raw_model(
                    input_ids=static_inputs[index],
                    labels=static_inputs[index],
                    use_cache=False,
                    attention_mask=masks[index],
                )
                segment_loss = output.loss / total
                segment_loss.backward()
                del output, segment_loss
        torch.cuda.current_stream().wait_stream(stream)

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
        capture_stream = torch.cuda.Stream()
        try:
            _warm_segments(raw_model, static_masks, static_inputs, total, capture_stream)
        except Exception as exc:
            raise _SetupFailure("setup_failed:warmup_failed:" + type(exc).__name__) from exc
        _prime_and_clear(params, buffers)
        static_losses = []
        graphs = []
        pool = torch.cuda.graph_pool_handle()
        try:
            if _SET_OVERRIDE is not None:
                # Stale-stream fix (c): redirect remaining stale default-stream
                # AccumulateGrad references (from invocation-1's parent
                # backward on stream 0) to the capturing stream.
                _SET_OVERRIDE(True)
            for index in range(segments):
                graph = torch.cuda.CUDAGraph()
                # Stale-stream fix (a): capture on the same explicit stream
                # that warmed the segment, as required for the shared pool.
                with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                    output = raw_model(
                        input_ids=static_inputs[index],
                        labels=static_inputs[index],
                        use_cache=False,
                        attention_mask=static_masks[index],
                    )
                    segment_loss = output.loss / total
                    segment_loss.backward()
                static_losses.append(segment_loss.detach())
                graphs.append(graph)
                del output, segment_loss
        finally:
            if _SET_OVERRIDE is not None:
                _SET_OVERRIDE(False)
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
