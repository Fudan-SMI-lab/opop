import importlib
from pathlib import Path
from types import TracebackType
from typing import Self

from .runner_records import RunnerError


class TorchTrace:
    def __init__(self) -> None:
        self.api = importlib.import_module("torch.profiler")
        self.activities = [self.api.ProfilerActivity.CPU, self.api.ProfilerActivity.CUDA]
        # Scoped wrappers capture shape/stride/dtype without profiler-held activation references.
        self.trace = self.api.profile(activities=self.activities, record_shapes=False)

    def __enter__(self) -> Self:
        self.trace.__enter__()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.trace.__exit__(exc_type, exc, traceback)

    def active(self, enabled: bool) -> None:
        self.trace.toggle_collection_dynamic(enabled, self.activities)

    def region(self, label: str):
        return self.api.record_function(label)

    def export(self, path: Path) -> None:
        self.trace.export_chrome_trace(str(path))

    def records(self):
        associated, intervals = [], []
        for event in self.trace.events():
            if str(event.device_type).endswith("CUDA"):
                intervals.append({"kernel": event.name, "start_us": event.time_range.start,
                                  "end_us": event.time_range.end, "device": event.device_index})
            parent = event
            while parent is not None and not parent.name.startswith("c3-site:"):
                parent = parent.cpu_parent
            if parent is None:
                continue
            for kernel in event.kernels:
                associated.append({"scope": parent.name, "cpu_event_id": event.id,
                                   "kernel": kernel.name, "cuda_duration_us": kernel.duration,
                                   "device": kernel.device})
        if not intervals:
            raise RunnerError("CUDA profiler produced no device events; cannot claim a device profile")
        return {"site_kernels": associated, "cuda_intervals": intervals}
