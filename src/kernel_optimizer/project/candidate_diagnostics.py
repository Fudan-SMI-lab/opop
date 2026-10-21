"""Observe swallowed candidate exceptions and Boolean guards in a separate worker."""
from collections import Counter, deque
from contextlib import AbstractContextManager
import linecache
from pathlib import Path
import sys
import traceback


class CandidateTrace(AbstractContextManager):
    """Trace only supplied candidate files; never retain frames, locals or tensors."""

    def __init__(self, bundles):
        self.files = {str(Path(b["file"]).resolve()): b["id"] for b in bundles}
        self.events = Counter()
        self.phase = "setup"
        self.previous = None
        self.recent = {}

    def _trace(self, frame, event, arg):
        filename = frame.f_code.co_filename
        if filename not in self.files:
            return None
        # CPython can report a with/finally cleanup line for a False return.
        # A small source-line history identifies the actual guard without locals.
        ident = id(frame)
        if event == "call":
            self.recent[ident] = deque(maxlen=12)
        elif event == "line":
            self.recent.setdefault(ident, deque(maxlen=12)).append(frame.f_lineno)
            return self._trace
        history = tuple(self.recent.get(ident, ()))
        if event == "return":
            self.recent.pop(ident, None)
        if event == "exception":
            kind, error, exception_tb = arg
            try:
                message = str(error)
            except Exception:
                message = "<exception message unavailable>"
            detail = (kind.__name__, message)
            stack = tuple((entry.filename, entry.name, entry.lineno, entry.line or "")
                          for entry in traceback.extract_tb(exception_tb))
        elif event == "return" and arg is False:
            detail = ("bool", "False")
            stack = ()
        else:
            return self._trace
        self.events[(self.phase, filename, frame.f_code.co_name, frame.f_lineno, event, *detail, history, stack)] += 1
        return self._trace

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)
        self.recent.clear()
        return False

    def report(self):
        return {
            "schema_version": "project.candidate-diagnostics.v1",
            "events": [
                {"phase": phase, "bundle": self.files[file], "file": file,
                 "function": function, "line": line, "source": linecache.getline(file, line).strip(),
                 "recent_lines": [{"line": n, "source": linecache.getline(file, n).strip()} for n in history],
                 "event": event, "type": kind, "message": message, "count": count,
                 **({"traceback": [{"file": f, "function": fn, "line": n, "source": src}
                                   for f, fn, n, src in stack]} if stack else {})}
                for (phase, file, function, line, event, kind, message, history, stack), count in self.events.items()
            ],
            "interpretation": (
                "Separate intrusive Python diagnostic, including candidate installation and continuous runs. "
                "Exceptions may be intentionally handled; False returns may be normal predicates. "
                "These are source locations for investigation, not proof of failure or causal attribution. "
                "Only candidate installer files on the worker's current Python thread are traced; "
                "native code and other threads are outside this observation. Propagated exception tracebacks "
                "can locate imported helpers without tracing their execution or retaining frames/locals."
            ),
        }


def reference_sites(run, bundle_id):
    """Locate fully referenced sites, including partial fallback inside a joint bundle."""
    sites = []
    for rank in run.get("ranks", []):
        coverage = rank.get("result", {}).get("coverage", {}).get(bundle_id, {})
        references = coverage.get("reference_invocations", {})
        for call, count in coverage.get("counts", {}).items():
            if references.get(call, 0) >= count > 0:
                sites.append({"rank": rank.get("rank"), "bundle": bundle_id, "call": call,
                              "invocations": count, "reference_invocations": references[call]})
    return sites


def should_trace(run, bundle_id, profiling):
    """Reference use is a diagnostic trigger, not a correctness/acceptance rule."""
    policy = profiling.get("candidate_diagnostics", "on_reference")
    if policy not in {"off", "on_reference", "always"}:
        raise ValueError("candidate_diagnostics must be off, on_reference or always")
    if policy == "always":
        return True
    if policy == "off":
        return False
    return bool(reference_sites(run, bundle_id))
