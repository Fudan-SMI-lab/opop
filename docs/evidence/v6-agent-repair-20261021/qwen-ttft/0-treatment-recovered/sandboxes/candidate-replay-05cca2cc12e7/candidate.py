"""Candidate: CUDA-graph replay of the parent prefill region, with
eager-decision-preserving capture and a bit-exactness gate.

Repair of trial-af78dd3283 (rejected: per-step logits max abs error 5.0625,
41.5% mismatched elements on both continuous steps; tokens still matched).

Measured root cause (previous-feedback supplemental_evidence): capturing the
unchanged parent flips transformers' tracing probe.  In the installed
Transformers 5.16.1, ``transformers.utils.import_utils.is_tracing()`` includes
``is_cuda_stream_capturing()`` (``torch.cuda.is_current_stream_capturing()``,
transformers/utils/import_utils.py:1776-1797).  During stream capture it
returns True, so ``masking_utils._ignore_causal_mask_sdpa`` returns False
(masking_utils.py:259-260), a boolean 4D mask is materialized, and
``sdpa_attention_forward`` then sees ``attention_mask is not None``:
``use_gqa_in_sdpa`` becomes False, K/V are ``repeat_kv``-expanded to
[1,32,256,128], and SDPA runs with ``is_causal=False`` plus the bool mask.
The eager parent instead runs mask=None, GQA enabled, is_causal=True.  The
divergent kernel path differs by 0.0009765625 in the first attention output
and compounds to 5.0625 in the logits.  The same diagnostic record shows the
repair direction: preserving the eager attention decision during capture let
the graph execute with zero error over two complete generations.

This candidate therefore:

1. Keeps the previous attempt's boundary, signature guard, cache-snapshot
   machinery and fallback structure (those parts were validated: installation,
   direct module binding at the region root, coverage 2/2 and routing all
   passed; only numerics failed).

2. Wraps ONLY the graph-capture window in a temporary, restored patch of
   ``transformers.utils.import_utils.is_cuda_stream_capturing`` -> False, so
   every Python-level decision the parent makes while being captured (mask
   skip, GQA, causal flag) is exactly the decision the eager parent makes for
   this call.  No dynamo/jit tracing is active in this workload, so eager
   ``is_tracing()`` is False; the patch makes capture observe the same truth.
   The patch is scoped to the ``with torch.cuda.graph(...)`` block (try/finally
   restore); it is not a global mask-policy change, and decode/other calls
   never see it.  If the probe cannot be resolved in the installed layout, an
   ``is_tracing``-level fallback is used; if that also fails, capture proceeds
   unpatched and the verification gate below decides.

3. Adds a setup-time bit-exactness gate: immediately after capture it replays
   the graph and compares (``torch.equal``) the replayed logits and layer-0
   K/V against a fresh, unpatched eager run of the same parent on the live
   input.  Replay of the identical captured kernel sequence on the same
   pointers must be bit-identical; any mismatch permanently disables the graph
   and the parent serves every call, so the accepted graph path can never
   degrade quality relative to the parent.

The replacement binds the phase-guarded prefill call of ``Qwen3ForCausalLM``
(call id ``900c23e99c808d2c9bda215a``, kwargs ``input_ids=[1,256]`` int64 CUDA,
``use_cache=True``, ``logits_to_keep=1``).  Capture happens lazily inside the
first real adapter invocation, with the evaluator's dispatch modes and the
adapter's prefill phase active, per inputs/binding-contract.md.  Every
invocation copies the current ``input_ids`` into the static buffer, replays,
and returns a ``CausalLMOutputWithPast`` whose logits are a fresh clone and
whose ``past_key_values`` is a fresh ``DynamicCache`` viewing the graph-owned
static K/V buffers, so the untouched parent decode loop extends it with
``torch.cat`` exactly as before (``DynamicLayer.update`` reassigns new cat
results; it never writes in place into the static buffers).

Fallback labels distinguish unsupported signature
(``prefill_parent_signature_fallback``), setup/capture/verification failure
(``prefill_setup_failure_parent_fallback`` /
``prefill_graph_verification_failed_parent_fallback``) and runtime replay
failure (``prefill_parent_runtime_fallback``).  No weights are transformed,
no fixture-dependent outputs are cached (replay recomputes from the live input
copy on every invocation), and decode-phase calls are untouched.
"""

from __future__ import annotations

import sys

import torch

try:  # worker always has transformers; kept defensive for import-time safety
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import CausalLMOutputWithPast
except Exception:  # pragma: no cover
    DynamicCache = None
    CausalLMOutputWithPast = None

PREFILL_CALL = "900c23e99c808d2c9bda215a"

_BATCH = 1
_SEQ = 256

_KNOWN_KWARGS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "use_cache",
        "logits_to_keep",
    }
)
_NONE_ONLY_KWARGS = (
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
)

_WARMUP_ITERS = 3


class _VerificationError(RuntimeError):
    """Graph replay did not reproduce the eager parent bit-exactly."""


def _eager_false(*_args, **_kwargs):
    return False


class _EagerDecisionPatch:
    """Make transformers' tracing probes report the eager truth during capture.

    Preferred single patch point: ``is_cuda_stream_capturing`` inside
    ``transformers.utils.import_utils`` - the only stream-capture term of
    ``is_tracing()``.  With no dynamo/jit tracing active (this workload),
    ``is_tracing()`` is False eagerly, so binding this probe to False makes
    every capture-time decision identical to the eager parent's decision.
    Restored unconditionally in ``__exit__``.
    """

    def __init__(self):
        self._patches = []  # (module, attribute name, original value)

    def _resolve(self):
        import transformers.utils.import_utils as import_utils

        probe = getattr(import_utils, "is_cuda_stream_capturing", None)
        if callable(probe):
            return [(import_utils, "is_cuda_stream_capturing", probe)]

        # Layouts without the dedicated probe: neutralize is_tracing() itself
        # at its definition site and in modules that imported it by name.
        patches = []
        base = getattr(import_utils, "is_tracing", None)
        if callable(base):
            patches.append((import_utils, "is_tracing", base))
        masking_utils = sys.modules.get("transformers.masking_utils")
        if masking_utils is not None:
            imported = getattr(masking_utils, "is_tracing", None)
            if callable(imported):
                patches.append((masking_utils, "is_tracing", imported))
        return patches

    def __enter__(self):
        try:
            for module, name, original in self._resolve():
                setattr(module, name, _eager_false)
                self._patches.append((module, name, original))
        except Exception:
            self._restore()
        return self

    def _restore(self):
        while self._patches:
            module, name, original = self._patches.pop()
            try:
                setattr(module, name, original)
            except Exception:  # pragma: no cover - keep unwinding
                pass

    def __exit__(self, *exc):
        self._restore()
        return False


class _GraphPrefill:
    """CUDA-graph replay engine for the region-root prefill call."""

    def __init__(self, model, parent, record_route, call_id):
        self.call_id = call_id
        self.parent = parent
        self._route = record_route
        self.device = next(model.parameters()).device
        self.vocab = int(getattr(model.config, "vocab_size", 0))
        self.mode = "init"
        self.graph = None
        self.static_in = None
        self.static_logits = None
        self._cache_cls = None
        self._cache_base = None
        self._layer_protos = None

    # -- signature guard ------------------------------------------------
    def _guard(self, args, kwargs):
        """Return input_ids when the call matches the observed signature."""
        try:
            if args:
                return None
            for key in kwargs:
                if key not in _KNOWN_KWARGS and kwargs.get(key) is not None:
                    return None
            for key in _NONE_ONLY_KWARGS:
                if kwargs.get(key) is not None:
                    return None
            use_cache = kwargs.get("use_cache", None)
            if not (use_cache is True or use_cache is None):
                return None
            logits_to_keep = kwargs.get("logits_to_keep", None)
            if not isinstance(logits_to_keep, int) or isinstance(logits_to_keep, bool):
                return None
            if logits_to_keep != 1:
                return None
            ids = kwargs.get("input_ids")
            if not (
                torch.is_tensor(ids)
                and ids.dim() == 2
                and tuple(ids.shape) == (_BATCH, _SEQ)
                and ids.dtype == torch.long
                and ids.is_cuda
                and ids.device == self.device
            ):
                return None
            return ids
        except Exception:
            return None

    # -- structure checks -----------------------------------------------
    def _check_structure(self, logits, cache_obj):
        layers = list(getattr(cache_obj, "layers", []) or [])
        if not torch.is_tensor(logits):
            raise RuntimeError("graph prefill: capture produced no logits tensor")
        if tuple(logits.shape) != (_BATCH, 1, self.vocab):
            raise RuntimeError("graph prefill: unexpected captured logits shape")
        if cache_obj is None or not layers:
            raise RuntimeError("graph prefill: capture produced no KV cache")
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not (torch.is_tensor(keys) and torch.is_tensor(values)):
                raise RuntimeError("graph prefill: cache layer without tensors")
            if keys.shape[-2] != _SEQ or values.shape[-2] != _SEQ:
                raise RuntimeError("graph prefill: cache layer with wrong length")

    # -- lazy capture inside the live invocation ------------------------
    def _capture(self, ids):
        static_in = torch.empty_like(ids)
        static_in.copy_(ids)

        # Side-stream eager warmup of the parent (cuBLAS/SDPA workspace and
        # kernel selection state), per the CUDA-graph recipe.  The side stream
        # is not capturing, so the parent already follows its eager decisions.
        # Results are discarded.
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(_WARMUP_ITERS):
                self.parent(input_ids=static_in, use_cache=True, logits_to_keep=1)
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()

        # Eager reference on the live input, with every probe unpatched and
        # the stream not capturing: exactly the execution the framework's
        # reference/quality path performs.  Used only for the bit-exactness
        # gate below; its KV cache is discarded after verification.
        probe = self.parent(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()

        # Capture the parent itself.  During this window only, transformers'
        # stream-capture tracing probe reports False so the parent's Python
        # control flow (mask skip, GQA, causal flag) matches its eager path -
        # the validated repair of the 5.0625-maxabs divergence.
        graph = torch.cuda.CUDAGraph()
        with _EagerDecisionPatch():
            with torch.cuda.graph(graph):
                out = self.parent(input_ids=static_in, use_cache=True, logits_to_keep=1)

        logits = getattr(out, "logits", None)
        cache_obj = getattr(out, "past_key_values", None)
        self._check_structure(logits, cache_obj)

        # Verification gate: replay must reproduce the unpatched eager parent
        # bit-for-bit (same kernels, same pointers, deterministic kernels).
        static_in.copy_(ids, non_blocking=True)
        graph.replay()
        torch.cuda.synchronize()
        probe_layers = list(getattr(probe.past_key_values, "layers", []) or [])
        ok = torch.equal(logits, probe.logits)
        if ok and probe_layers:
            ok = torch.equal(cache_obj.layers[0].keys, probe_layers[0].keys) and torch.equal(
                cache_obj.layers[0].values, probe_layers[0].values
            )
        if not ok:
            raise _VerificationError(
                "graph prefill: replay did not reproduce the eager parent bit-exactly"
            )

        # Snapshot the cache structure (class, non-layer state, per-layer
        # attribute dictionaries).  Tensor attributes keep pointing at the
        # graph-owned static KV buffers; each invocation gets fresh container
        # objects so decode-time updates never mutate this snapshot.
        layers = list(cache_obj.layers)
        self._cache_cls = type(cache_obj)
        self._cache_base = {k: v for k, v in cache_obj.__dict__.items() if k != "layers"}
        self._layer_protos = [(type(layer), dict(layer.__dict__)) for layer in layers]

        self.static_in = static_in
        self.static_logits = logits
        self.graph = graph
        self.mode = "graph"

    # -- output construction --------------------------------------------
    def _fresh_cache(self):
        cache = object.__new__(self._cache_cls)
        base = dict(self._cache_base)
        layers = []
        for layer_cls, layer_dict in self._layer_protos:
            layer = object.__new__(layer_cls)
            layer.__dict__.update(layer_dict)
            layers.append(layer)
        base["layers"] = layers
        cache.__dict__.update(base)
        return cache

    def _outputs(self):
        # Clone so the returned logits never alias the graph-owned buffer
        # across invocations; the adapter slices/argmaxes this immediately.
        logits = self.static_logits.clone()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=self._fresh_cache(),
            hidden_states=None,
            attentions=None,
        )

    def _teardown(self):
        self.graph = None
        self.static_in = None
        self.static_logits = None
        self._cache_cls = None
        self._cache_base = None
        self._layer_protos = None

    # -- replacement entry point ------------------------------------------
    def __call__(self, *args, **kwargs):
        ids = self._guard(args, kwargs)
        if ids is None:
            self._route(self.call_id, "prefill_parent_signature_fallback")
            return self.parent(*args, **kwargs)

        if self.mode == "init":
            try:
                self._capture(ids)
                self._route(self.call_id, "prefill_cuda_graph_capture_verify_replay")
                return self._outputs()
            except _VerificationError:
                self._teardown()
                self.mode = "parent"
                self._route(self.call_id, "prefill_graph_verification_failed_parent_fallback")
                return self.parent(*args, **kwargs)
            except Exception:
                self._teardown()
                self.mode = "parent"
                self._route(self.call_id, "prefill_setup_failure_parent_fallback")
                return self.parent(*args, **kwargs)

        if self.mode == "graph":
            try:
                self.static_in.copy_(ids, non_blocking=True)
                self.graph.replay()
                self._route(self.call_id, "prefill_cuda_graph_replay")
                return self._outputs()
            except Exception:
                self.mode = "parent"
                self._teardown()
                self._route(self.call_id, "prefill_parent_runtime_fallback")
                return self.parent(*args, **kwargs)

        self._route(self.call_id, "prefill_parent_runtime_fallback")
        return self.parent(*args, **kwargs)


def build(context):
    """Install the prefill-region CUDA-graph replay replacement (rank-local)."""
    record_route = context.get("record_route") or (lambda *_: None)
    modules = context.get("modules") or {}
    models = context.get("models") or {}
    references = context.get("references") or {}

    replacements = {}
    call_id = PREFILL_CALL
    model = modules.get(call_id) or models.get("model")
    parent = references.get(call_id)
    if model is None or parent is None or CausalLMOutputWithPast is None or DynamicCache is None:
        # Nothing safely replaceable on this rank: return an empty (legal)
        # mapping; the parent keeps serving every call.
        return replacements

    try:
        engine = _GraphPrefill(model, parent, record_route, call_id)
    except Exception:

        def _setup_failed(*args, **kwargs):
            record_route(call_id, "prefill_setup_failure_parent_fallback")
            return parent(*args, **kwargs)

        replacements[call_id] = _setup_failed
        return replacements

    replacements[call_id] = engine
    return replacements
