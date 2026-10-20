"""Observe swallowed candidate exceptions and Boolean guards in a separate worker."""
from collections import Counter, deque
from contextlib import AbstractContextManager
import linecache
from pathlib import Path
import sys


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
            kind, error, _ = arg
            try:
                message = str(error)
            except Exception:
                message = "<exception message unavailable>"
            detail = (kind.__name__, message)
        elif event == "return" and arg is False:
            detail = ("bool", "False")
        else:
            return self._trace
        self.events[(self.phase, filename, frame.f_code.co_name, frame.f_lineno, event, *detail, history)] += 1
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
                 "event": event, "type": kind, "message": message, "count": count}
                for (phase, file, function, line, event, kind, message, history), count in self.events.items()
            ],
            "interpretation": (
                "Separate intrusive Python diagnostic, including candidate installation and continuous runs. "
                "Exceptions may be intentionally handled; False returns may be normal predicates. "
                "These are source locations for investigation, not proof of failure or causal attribution. "
                "Only candidate installer files on the worker's current Python thread are traced; "
                "native code, other threads and imported helper files are outside this observation."
            ),
        }


def should_trace(run, bundle_id, profiling):
    """Reference use is a diagnostic trigger, not a correctness/acceptance rule."""
    policy = profiling.get("candidate_diagnostics", "on_reference")
    if policy not in {"off", "on_reference", "always"}:
        raise ValueError("candidate_diagnostics must be off, on_reference or always")
    if policy == "always":
        return True
    if policy == "off":
        return False
    for rank in run.get("ranks", []):
        coverage = rank.get("result", {}).get("coverage", {}).get(bundle_id, {})
        counts = coverage.get("counts", {})
        references = coverage.get("reference_invocations", {})
        if counts and all(references.get(call, 0) >= count > 0 for call, count in counts.items()):
            return True
    return False
