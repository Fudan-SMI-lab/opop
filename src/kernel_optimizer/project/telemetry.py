"""Optional device-level NVML observations, kept separate from project timing.

PCIe throughput is already a rate over NVML's 20 ms sample window. It is not
differenced or relabelled as per-collective wire bytes. No counter is configured.
"""
import ctypes as C
import ctypes.util
import json
from pathlib import Path
import re
import subprocess
import threading
import time


class NVMLFailure(Exception):
    def __init__(self, code, function):
        self.code = code
        super().__init__(f"{function} returned NVML status {code}")


class Memory(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in ("total", "free", "used")]


class Utilization(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ("gpu", "memory")]


class LinkControl(C.Structure):
    _fields_ = [(name, C.c_uint) for name in ("units", "pktfilter")]


class DeviceTelemetry:
    def __init__(self, uuid):
        self.uuid = uuid
        self.library = C.CDLL(ctypes_library())
        self._call("nvmlInit_v2", ())
        self.handle = C.c_void_p()
        try:
            self._call("nvmlDeviceGetHandleByUUID", (C.c_char_p, C.POINTER(C.c_void_p)), uuid.encode(), C.byref(self.handle))
            version = C.create_string_buffer(96)
            self._call("nvmlSystemGetNVMLVersion", (C.c_char_p, C.c_uint), version, len(version))
            self.version = version.value.decode()
            # Discover actual link IDs, rather than assuming a machine/model link count.
            self.link_probe = subprocess.run(["nvidia-smi", "nvlink", "-s", "-i", uuid], capture_output=True, text=True, timeout=15)
            self.links = sorted(set(map(int, re.findall(r"Link\s+(\d+)\s*:", self.link_probe.stdout))))
        except Exception:
            self.close()
            raise

    def _call(self, name, types, *args):
        function = getattr(self.library, name)
        function.argtypes, function.restype = types, C.c_int
        result = function(*args)
        if result:
            raise NVMLFailure(result, name)

    def close(self):
        self._call("nvmlShutdown", ())

    def reading(self, name, unit, kind, get, window=None):
        begin = time.monotonic_ns()
        row = {"name": name, "unit": unit, "kind": kind, "scope": "device", "entity": self.uuid,
               "provenance": "NVML read-only API", "collector_version": self.version,
               "clock_domain": "host monotonic_ns", "read_start_ns": begin, "window": window}
        try:
            row.update(status="observed", value=float(get()))
        except (NVMLFailure, AttributeError) as exc:
            status = {3: "not_supported", 4: "permission_denied"}.get(getattr(exc, "code", None), "failed")
            row.update(status=status, value=None, reason=str(exc))
        row["read_end_ns"] = time.monotonic_ns()
        return row

    def unsigned(self, name, *args):
        value = C.c_uint()
        self._call(name, (C.c_void_p, *(C.c_uint for _ in args), C.POINTER(C.c_uint)), self.handle, *args, C.byref(value))
        return value.value

    def struct(self, name, cls):
        value = cls()
        self._call(name, (C.c_void_p, C.POINTER(cls)), self.handle, C.byref(value))
        return value

    def sample(self):
        rows = [self.reading("pcie."+direction, "KB/s (NVML)", "rate",
                    lambda counter=counter: self.unsigned("nvmlDeviceGetPcieThroughput", counter),
                    {"native_duration_ms": 20, "alignment": "driver sampling window; exact boundaries unavailable"})
                for direction, counter in (("tx", 0), ("rx", 1))]
        rows += [self.reading("device."+field+"_utilization", "percent", "ratio",
                    lambda field=field: getattr(self.struct("nvmlDeviceGetUtilizationRates", Utilization), field),
                    {"alignment": "device-dependent NVML utilization period; not aligned to operator ranges"})
                 for field in ("gpu", "memory")]
        rows.append(self.reading("device.memory_used", "bytes", "gauge",
                                lambda: self.struct("nvmlDeviceGetMemoryInfo", Memory).used))
        return rows

    def link_counters(self):
        rows = []
        status, reason = "unavailable", "no link IDs reported by nvidia-smi"
        if not self.links:
            try:
                state = self.unsigned("nvmlDeviceGetNvLinkState", 0)
                reason = f"first link state is {state}, but link inventory is unavailable"
            except (NVMLFailure, AttributeError) as exc:
                status = "not_supported" if getattr(exc, "code", None) == 3 else "unavailable"
                reason = str(exc)
        for link in self.links:
            control = LinkControl()
            try:
                self._call("nvmlDeviceGetNvLinkUtilizationControl", (C.c_void_p, C.c_uint, C.c_uint, C.POINTER(LinkControl)),
                           self.handle, link, 0, C.byref(control))
                rx, tx = C.c_ulonglong(), C.c_ulonglong()
                started = time.monotonic_ns()
                self._call("nvmlDeviceGetNvLinkUtilizationCounter", (C.c_void_p, C.c_uint, C.c_uint, C.POINTER(C.c_ulonglong), C.POINTER(C.c_ulonglong)),
                           self.handle, link, 0, C.byref(rx), C.byref(tx))
                unit = {0: "cycles", 1: "packets", 2: "bytes"}.get(control.units, "unknown")
                rows.append({"link": link, "status": "observed" if unit != "unknown" else "unknown", "rx": rx.value, "tx": tx.value,
                    "unit": unit, "kind": "counter", "filter": control.pktfilter, "read_start_ns": started, "read_end_ns": time.monotonic_ns()})
            except (NVMLFailure, AttributeError) as exc:
                rows.append({"link": link, "status": "unavailable", "reason": str(exc)})
        return {"status": "observed" if any(r["status"] == "observed" for r in rows) else status,
                "reason": None if rows else reason,
                "entity": self.uuid, "links": rows, "probe": self.link_probe.stdout+self.link_probe.stderr,
                "probe_returncode": self.link_probe.returncode,
                "scope": "per-link cumulative counters with observed units/filter; deprecated APIs may be unsupported; never reconfigure them"}


def ctypes_library():
    return ctypes.util.find_library("nvidia-ml") or ("nvml.dll" if __import__("os").name == "nt" else "libnvidia-ml.so.1")


def probe(uuid):
    device = None
    try:
        device = DeviceTelemetry(uuid)
    except (OSError, AttributeError, NVMLFailure, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    try:
        return {"status": "observed", "fields": device.sample(), "nvlink": device.link_counters(),
                "scope": "capability probe outside workload; probe rates are not workload traffic"}
    except (OSError, AttributeError, NVMLFailure, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    finally:
        if device:
            device.close()


def capture(adapter, state, operation, synchronize, context, request):
    uuid = context["binding"].get("requested_uuid")
    if not uuid:
        return {"valid": True, "telemetry": {"status": "not_applicable", "reason": "CPU task"}}
    device, unavailable = None, None
    try:
        device = DeviceTelemetry(uuid)
    except (OSError, AttributeError, NVMLFailure, subprocess.TimeoutExpired) as exc:
        unavailable = str(exc)
    if context["world_size"] > 1:
        import torch
        ready = torch.tensor(int(device is not None), device=context["device"])
        torch.distributed.all_reduce(ready, op=torch.distributed.ReduceOp.MIN)
        if not ready.item():
            if device:
                device.close()
            return {"valid": True, "telemetry": {"status": "unavailable",
                    "reason": unavailable or "telemetry unavailable on another rank; no partial distributed workload executed"}}
    elif device is None:
        return {"valid": True, "telemetry": {"status": "unavailable", "reason": unavailable}}
    try:
        interval = float(request.get("profiling", {}).get("telemetry_interval_s", 0.02))
        if interval <= 0:
            raise ValueError("telemetry interval must be positive")
        windows = []
        repeats = int(request.get("profiling", {}).get("telemetry_repeats", 3))
        if repeats < 1:
            raise ValueError("telemetry repeats must be positive")
        for index in range(repeats):
            adapter.reset(state)
            synchronize()
            if callable(getattr(adapter, "before_run", None)):
                adapter.before_run(state)
            elif context["world_size"] > 1:
                import torch
                torch.distributed.barrier()
            samples, failures, stop = [], [], threading.Event()
            def sample_loop():
                while not stop.is_set():
                    try:
                        samples.append(device.sample())
                    except Exception as exc:
                        failures.append(f"{type(exc).__name__}: {exc}")
                        break
                    stop.wait(interval)
            before = device.link_counters()
            started = time.monotonic_ns()
            sampler = threading.Thread(target=sample_loop, daemon=True); sampler.start()
            try:
                operation(); synchronize()
                ended = time.monotonic_ns()
            finally:
                stop.set(); sampler.join()
            after = device.link_counters()
            windows.append({"repeat": index, "start_ns": started, "end_ns": ended, "wall_ms": (ended-started)/1e6,
                "samples": samples, "sampling_failures": failures, "nvlink_before": before, "nvlink_after": after})
        telemetry = {"status": "observed", "rank": context["rank"], "device": uuid, "windows": windows,
            "interval_s": interval, "clock_domain": "host monotonic_ns", "kind": "separate diagnostic run",
            "device_coverage": {"sampled": [uuid], "not_sampled_by_this_rank": [d for d in request.get("devices", []) if d != uuid]},
            "interpretation": "Device-wide samples include all device activity. Native rate windows may cross workload boundaries. Rates are not differenced or integrated into purported wire bytes; GPU utilization is sampled active time, not SM occupancy; memory utilization is activity, not capacity. No per-group allocation or causal critical-path claim."}
        path = Path(context["output"])/f"telemetry-rank-{context['rank']}.json"
        from kernel_optimizer.runtime.execution import write_json
        write_json(path, telemetry)
        return {"valid": True, "telemetry": telemetry, "raw_ref": str(path)}
    finally:
        if device:
            device.close()
