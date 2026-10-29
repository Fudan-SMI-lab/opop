"""Diagnostic collector entry for the supplied failed candidate evaluation.

Contract (inputs/interface.md): run(context) imports the ORIGINAL evaluator
entry named by parameters.diagnostic_context.original_entry and calls its
original_function exactly once with the unchanged context. The original
computation, bindings, quality protocol, precision policies, reset, completion
boundaries, thresholds and clean repetitions are preserved; original exceptions
propagate unchanged after evidence has been persisted; a successful original
return stays successful with only an added optional `diagnostic` index field.

Observation scope: the candidate bundle installation window only. The supplied
failure record (inputs/failure.json) shows two internally caught AssertionErrors
from the candidate's own setup self-checks (torch.testing.assert_close at
opop_candidates/trial-0db61e9938.py:468/537 and :629) whose comparison operands
and input/state provenance were never captured. This collector records bounded
summaries of those operands and of every parent reference call made during
installation, then restores all instrumentation in finally blocks. Generated
artifacts are written only under context["output"]. Full design, source
mappings, controls and limitations: diagnostic-plan.json in this directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

EVIDENCE_SCHEMA = "project.diagnostic-evidence.v1"
_MAX_REFERENCE_CALLS = 2048
_MAX_ASSERT_SITES = 128
_MAX_TOP_VIOLATIONS = 64
_MAX_STACK_FRAMES = 12
_MAX_TOKENS_LISTED = 8
_MSG_CLIP = 300
_TORCH_TESTING_PREFIX = None


def _clip(value, limit):
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<clipped>"


def _sha256_file(path):
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        return None


def _file_identity(path):
    if path is None:
        return None
    path = Path(path)
    entry = {"path": str(path)}
    if path.is_file():
        entry["sha256"] = _sha256_file(path)
        entry["bytes"] = int(path.stat().st_size)
    else:
        entry["absent"] = True
    return entry


def _sanitize(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return {"__repr__": _clip(repr(value), 200)}


def _write_evidence_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_sanitize(payload), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _torch_testing_prefix():
    global _TORCH_TESTING_PREFIX
    if _TORCH_TESTING_PREFIX is None:
        import os
        import torch
        _TORCH_TESTING_PREFIX = str(Path(torch.testing.__file__).resolve().parent) + os.sep
    return _TORCH_TESTING_PREFIX


def _describe_operand(value):
    import torch
    if not isinstance(value, torch.Tensor):
        return {"kind": type(value).__name__, "repr": _clip(repr(value), 200)}
    info = {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "numel": int(value.numel()),
            "requires_grad": bool(value.requires_grad)}
    if value.numel() and value.is_floating_point():
        with torch.no_grad():
            host = value.detach().to("cpu", torch.float32)
            finite = bool(torch.isfinite(host).all().item())
            info["all_finite"] = finite
            if finite:
                info["min"] = float(host.min().item())
                info["max"] = float(host.max().item())
            else:
                info["nan_count"] = int(torch.isnan(host).sum().item())
                info["inf_count"] = int(torch.isinf(host).sum().item())
    return info


def _flat_argmax_info(host_flat):
    import torch
    if host_flat is None or host_flat.numel() < 2:
        return None
    top = torch.topk(host_flat, 2)
    return {"flat_argmax": int(top.indices[0]), "top_value": float(top.values[0]),
            "second_flat_index": int(top.indices[1]), "second_value": float(top.values[1]),
            "top1_minus_top2_margin": float(top.values[0] - top.values[1])}


def _comparison_stats(actual, expected, atol, rtol, req_atol, req_rtol):
    import torch
    out = {"call_tolerance": {"atol": atol, "rtol": rtol},
           "request_quality_tolerance": ({"atol": req_atol, "rtol": req_rtol}
                                         if req_atol is not None and req_rtol is not None else None)}
    a = actual.detach().to("cpu", torch.float32).reshape(-1)
    b = expected.detach().to("cpu", torch.float32).reshape(-1)
    if a.numel() != b.numel():
        out["numel_mismatch"] = {"actual": int(a.numel()), "expected": int(b.numel())}
        return out, a, b
    diff = (a - b).abs()
    finite = torch.isfinite(diff)
    out["numel"] = int(a.numel())
    out["nonfinite_difference_count"] = int((~finite).sum().item())
    if a.numel():
        out["max_abs_error"] = float(diff.max().item())
        out["mean_abs_error"] = float(diff.mean().item())
    if atol is not None and rtol is not None:
        allowed = float(atol) + float(rtol) * b.abs()
        bad = (diff > allowed) & finite
        out["violates_call_tolerance_count"] = int(bad.sum().item())
        if bool(bad.any()) and int(bad.sum().item()) <= 1_000_000:
            idx = torch.nonzero(bad, as_tuple=False).reshape(-1)
            order = idx[diff[idx].argsort(descending=True)][:_MAX_TOP_VIOLATIONS]
            out["top_call_tolerance_violations"] = [
                {"flat_index": int(i), "actual": float(a[i]), "expected": float(b[i]),
                 "abs_error": float(diff[i]), "allowed": float(allowed[i])} for i in order]
    if req_atol is not None and req_rtol is not None:
        allowed_req = float(req_atol) + float(req_rtol) * b.abs()
        bad_req = (diff > allowed_req) & finite
        out["violates_request_quality_count"] = int(bad_req.sum().item())
        if bool(bad_req.any()) and int(bad_req.sum().item()) <= 1_000_000:
            idx = torch.nonzero(bad_req, as_tuple=False).reshape(-1)
            order = idx[diff[idx].argsort(descending=True)][:_MAX_TOP_VIOLATIONS]
            out["top_request_quality_violations"] = [
                {"flat_index": int(i), "actual": float(a[i]), "expected": float(b[i]),
                 "abs_error": float(diff[i]), "allowed": float(allowed_req[i])} for i in order]
    return out, a, b


def _describe_input_ids(value):
    import torch
    if not isinstance(value, torch.Tensor):
        return {"kind": type(value).__name__}
    flat = value.detach().to("cpu").reshape(-1)
    info = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
            "numel": int(flat.numel()), "tokens_head": [int(x) for x in flat[:_MAX_TOKENS_LISTED].tolist()]}
    if flat.numel() > _MAX_TOKENS_LISTED:
        info["tokens_tail"] = [int(x) for x in flat[-_MAX_TOKENS_LISTED:].tolist()]
    return info


def _describe_past(past):
    if past is None:
        return None
    info = {"type": type(past).__name__, "object_id": id(past)}
    try:
        info["seq_length"] = int(past.get_seq_length())
    except Exception:
        info["seq_length"] = None
    layers = getattr(past, "layers", None)
    if layers is not None:
        info["layers"] = len(layers)
        keys = getattr(layers[0], "keys", None) if len(layers) else None
        if keys is not None:
            info["first_layer_keys_shape"] = list(keys.shape)
            info["first_layer_keys_dtype"] = str(keys.dtype)
    return info


class _Recorder:
    def __init__(self, request_quality, diagnostic_context, context_echo):
        quality = request_quality if isinstance(request_quality, dict) else None
        self.req_atol = float(quality["atol"]) if quality and "atol" in quality else None
        self.req_rtol = float(quality["rtol"]) if quality and "rtol" in quality else None
        self.diagnostic_context = diagnostic_context
        self.context_echo = context_echo
        self.windows = []
        self.active = None
        self.errors = []

    def note_error(self, where, error):
        self.errors.append({"where": where, "type": type(error).__name__,
                            "message": _clip(str(error), _MSG_CLIP)})

    def open_window(self, bundle_id, bundle_file):
        window = {"bundle": bundle_id, "bundle_file": _file_identity(bundle_file),
                  "reference_calls": [], "reference_call_total": 0, "reference_call_overflow": 0,
                  "assert_close_sites": [], "assert_close_total": 0, "assert_close_overflow": 0,
                  "instrumentation": "observed"}
        self.windows.append(window)
        self.active = window
        return window

    def close_window(self):
        self.active = None

    def mark_window_uninstrumented(self):
        if self.active is not None:
            self.active["instrumentation"] = "uninstrumented_after_setup_error"

    def make_reference_observer(self, ident, spec, function):
        spec_info = {key: spec.get(key) for key in ("kind", "target", "phase")} if isinstance(spec, dict) else None

        def observed_reference(*args, **kwargs):
            window = self.active
            record = None
            if window is not None:
                window["reference_call_total"] += 1
                if len(window["reference_calls"]) < _MAX_REFERENCE_CALLS:
                    past = kwargs.get("past_key_values")
                    record = {"seq": window["reference_call_total"], "call": ident, "spec": spec_info,
                              "kwargs": {"input_ids": _describe_input_ids(kwargs.get("input_ids")),
                                         "past_key_values": _describe_past(past),
                                         "use_cache": kwargs.get("use_cache"),
                                         "logits_to_keep": kwargs.get("logits_to_keep"),
                                         "other_kwarg_names": sorted(
                                             set(kwargs) - {"input_ids", "past_key_values",
                                                            "use_cache", "logits_to_keep"})},
                              "input_past_object_id": id(past) if past is not None else None}
                    window["reference_calls"].append(record)
                else:
                    window["reference_call_overflow"] += 1
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                if record is not None:
                    record["error"] = {"type": type(error).__name__,
                                       "message": _clip(str(error), _MSG_CLIP)}
                raise
            if record is not None:
                logits = getattr(result, "logits", None)
                past_out = getattr(result, "past_key_values", None)
                import torch
                record["result"] = {"type": type(result).__name__,
                                    "logits": ({"shape": list(logits.shape), "dtype": str(logits.dtype)}
                                               if isinstance(logits, torch.Tensor) else None),
                                    "past_key_values": _describe_past(past_out),
                                    "returned_past_same_object_as_input": (
                                        past_out is kwargs.get("past_key_values")
                                        if past_out is not None or kwargs.get("past_key_values") is not None
                                        else None)}
            return result

        return observed_reference

    def patch_assert_close(self):
        import torch
        original = torch.testing.assert_close

        def observed_assert_close(*args, **kwargs):
            window = self.active
            record = None
            try:
                if window is not None:
                    record = self._record_assert_entry(window, args, kwargs)
            except Exception as error:
                self.note_error("assert_close_summary", error)
            try:
                outcome = original(*args, **kwargs)
            except Exception as error:
                if record is not None:
                    record["outcome"] = {"raised": {"type": type(error).__name__,
                                                    "message": _clip(str(error), _MSG_CLIP)}}
                raise
            if record is not None:
                record["outcome"] = {"passed": True}
            return outcome

        torch.testing.assert_close = observed_assert_close

        def restore():
            torch.testing.assert_close = original

        return restore

    def _record_assert_entry(self, window, args, kwargs):
        window["assert_close_total"] += 1
        if len(window["assert_close_sites"]) >= _MAX_ASSERT_SITES:
            window["assert_close_overflow"] += 1
            return None
        frames = []
        frame = sys._getframe(2)
        depth = 0
        while frame is not None and depth < _MAX_STACK_FRAMES:
            frames.append({"file": frame.f_code.co_filename, "line": frame.f_lineno,
                           "function": frame.f_code.co_name})
            frame = frame.f_back
            depth += 1
        prefix = _torch_testing_prefix()
        own_file = str(Path(__file__).resolve())
        site = None
        for entry in frames:
            if entry["file"] == own_file or entry["file"].startswith(prefix):
                continue
            site = {"file": entry["file"], "line": entry["line"], "function": entry["function"]}
            break
        bundle_identity = window.get("bundle_file") or {}
        bundle_file = bundle_identity.get("path")
        candidate_stack = ([{"file": entry["file"], "line": entry["line"],
                             "function": entry["function"]}
                            for entry in frames
                            if bundle_file and entry["file"] == bundle_file] or
                           [{"file": entry["file"], "line": entry["line"],
                             "function": entry["function"]}
                            for entry in frames
                            if entry["file"].startswith("opop_candidates") or
                            Path(entry["file"]).parent.name == "opop_candidates"])
        atol = kwargs.get("atol")
        rtol = kwargs.get("rtol")
        comparison = None
        argmax = None
        operands = []
        import torch
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            operands = [_describe_operand(args[0]), _describe_operand(args[1])]
            try:
                stats, a_flat, b_flat = _comparison_stats(args[0], args[1], atol, rtol,
                                                          self.req_atol, self.req_rtol)
                comparison = stats
                if args[0].ndim <= 2 and args[0].numel() > 1:
                    mine = _flat_argmax_info(a_flat)
                    reference = _flat_argmax_info(b_flat)
                    argmax = {"actual": mine, "expected": reference,
                              "flat_argmax_match": (None if mine is None or reference is None
                                                    else mine["flat_argmax"] == reference["flat_argmax"])}
            except Exception as error:
                self.note_error("comparison_stats", error)
        else:
            operands = [_clip(repr(item), 200) for item in args[:2]]
        record = {"seq": window["assert_close_total"], "site": site,
                  "candidate_stack": candidate_stack,
                  "kwargs": {"atol": atol, "rtol": rtol,
                             "equal_nan": kwargs.get("equal_nan"),
                             "msg": (_clip(kwargs["msg"], 200)
                                     if "msg" in kwargs and kwargs["msg"] is not None else None),
                             "other_kwarg_names": sorted(set(kwargs) - {"atol", "rtol", "equal_nan", "msg"})},
                  "operands": operands, "comparison": comparison, "argmax": argmax,
                  "outcome": None, "observer_error": None}
        window["assert_close_sites"].append(record)
        return record

    def report(self, outcome, instrumentation):
        try:
            import torch
            torch_version = torch.__version__
        except Exception:
            torch_version = None
        try:
            import transformers
            transformers_version = transformers.__version__
        except Exception:
            transformers_version = None
        return {
            "schema_version": EVIDENCE_SCHEMA,
            "context": self.context_echo,
            "diagnostic_context": self.diagnostic_context,
            "runtime": {"torch": torch_version, "transformers": transformers_version,
                        "python": sys.version.split()[0]},
            "instrumentation": {**instrumentation, "errors": self.errors,
                                "caps": {"reference_calls_per_window": _MAX_REFERENCE_CALLS,
                                         "assert_sites_per_window": _MAX_ASSERT_SITES,
                                         "top_violations_per_site": _MAX_TOP_VIOLATIONS,
                                         "stack_frames_per_site": _MAX_STACK_FRAMES}},
            "install_windows": self.windows,
            "original_outcome": outcome,
            "limitations_ref": "diagnostic-plan.json (limitations)",
        }


def _build_observed_session_class(recorder):
    from kernel_optimizer.project.binding import BindingSession

    class _ObservedBindingSession(BindingSession):
        def install(self, file, parameters, **kwargs):
            original_references = self.references
            try:
                recorder.open_window(self.bundle_id, file)
                observers = {ident: recorder.make_reference_observer(ident, self.specs.get(ident), function)
                             for ident, function in original_references.items()}
                restore_assert = recorder.patch_assert_close()
            except Exception as error:
                recorder.note_error("instrumentation_setup", error)
                recorder.mark_window_uninstrumented()
                recorder.close_window()
                return super().install(file, parameters, **kwargs)
            try:
                self.references = observers
                return super().install(file, parameters, **kwargs)
            finally:
                self.references = original_references
                restore_assert()
                recorder.close_window()

    return _ObservedBindingSession


def _load_original_entry(path, function_name):
    spec = importlib.util.spec_from_file_location("opop_diagnostic_original_entry", Path(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import original entry: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    if not callable(function):
        raise ValueError(f"original entry has no callable {function_name}")
    return function


def _persist(path, payload):
    try:
        _write_evidence_json(path, payload)
        if path.exists():
            return {"path": str(path), "sha256": _sha256_file(path),
                    "bytes": int(path.stat().st_size)}
        return {"path": str(path), "absent": True}
    except Exception as error:
        print(f"[diagnostic] evidence write failed: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return {"path": str(path), "write_failed": {"type": type(error).__name__,
                                                    "message": _clip(str(error), _MSG_CLIP)}}


def run(context):
    parameters = context["parameters"]
    diagnostic = parameters.get("diagnostic_context")
    if not isinstance(diagnostic, dict) or not diagnostic.get("original_entry"):
        raise ValueError("parameters.diagnostic_context.original_entry is required")
    original_entry = _load_original_entry(diagnostic["original_entry"],
                                          diagnostic.get("original_function", "run"))
    request = parameters.get("request", {})
    rank = context.get("rank", 0)
    context_echo = {"rank": rank, "world_size": context.get("world_size"),
                    "device": str(context.get("device")), "arm": diagnostic.get("arm"),
                    "action": parameters.get("action"),
                    "bundles_declared": [b.get("id") for b in parameters.get("bundles", [])],
                    "request_identity": {
                        "source_root": request.get("source_root"), "entry": request.get("entry"),
                        "workload_id": (request.get("workload") or {}).get("id"),
                        "quality": request.get("quality"),
                        "warmup": request.get("warmup"), "repeats": request.get("repeats"),
                        "continuous_steps": request.get("continuous_steps"),
                        "measurement_lifetime": request.get("measurement_lifetime")}}
    recorder = _Recorder(request.get("quality"), {
        "original_entry": _file_identity(diagnostic["original_entry"]),
        "original_function": diagnostic.get("original_function", "run"),
        "independent_reference": _file_identity(diagnostic.get("independent_reference")),
        "arm": diagnostic.get("arm")}, context_echo)
    identity = {
        "adapter": _file_identity(parameters.get("adapter")),
        "bundles": [{"id": b.get("id"), "file": _file_identity(b.get("file")),
                     "specs": _file_identity(b.get("specs_path")) if b.get("specs_path") else None}
                    for b in parameters.get("bundles", [])],
        "reference": _file_identity(Path(parameters.get("reference_dir", context["output"])) /
                                    f"reference-rank-{rank}.pt"),
        "collector": _file_identity(__file__)}
    instrumentation = {"status": "dormant", "framework_patch": None}
    restore_class = None
    try:
        import kernel_optimizer.project.binding as binding_module
        original_class = getattr(binding_module, "BindingSession", None)
        if original_class is None:
            instrumentation["framework_patch"] = "BindingSession attribute absent"
        else:
            restore_class = (lambda: setattr(binding_module, "BindingSession", original_class))
            binding_module.BindingSession = _build_observed_session_class(recorder)
            instrumentation["status"] = "installed"
            instrumentation["framework_patch"] = (
                "kernel_optimizer.project.binding.BindingSession replaced by a subclass whose "
                "install() wraps self.references with provenance observers and patches "
                "torch.testing.assert_close for the duration of the original install call only; "
                "both are restored in finally")
    except Exception as error:
        recorder.note_error("framework_patch", error)
        instrumentation["status"] = "failed"
        instrumentation["framework_patch"] = _clip(f"{type(error).__name__}: {error}", _MSG_CLIP)
        if restore_class is not None:
            restore_class()
            restore_class = None
    outcome = {"original_returned": False, "error_type": None, "error": None, "traceback": None}
    evidence_path = Path(str(context["output"])) / f"diagnostic-evidence-rank-{rank}.json"
    try:
        result = original_entry(context)
        outcome["original_returned"] = True
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=_clip(str(error), 2000),
                       traceback=_clip(traceback.format_exc(), 8000))
        evidence = recorder.report(outcome, {**instrumentation, "identity": identity})
        _persist(evidence_path, evidence)
        raise
    finally:
        if restore_class is not None:
            restore_class()
    outcome["result_keys"] = sorted(result.keys()) if isinstance(result, dict) else type(result).__name__
    evidence = recorder.report(outcome, {**instrumentation, "identity": identity})
    written = _persist(evidence_path, evidence)
    if isinstance(result, dict) and "diagnostic" not in result:
        result["diagnostic"] = {
            "evidence": written, "arm": diagnostic.get("arm"),
            "instrumentation": instrumentation["status"],
            "schema": EVIDENCE_SCHEMA,
            "note": ("diagnostic observations only; timings and metrics in this result are not "
                     "acceptance scores; see diagnostic-plan.json for limitations")}
    return result
