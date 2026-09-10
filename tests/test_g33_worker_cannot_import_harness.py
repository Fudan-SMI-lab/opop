"""G10's Triton ceilings were never measured in a real run, and nothing said so (G33).

WHAT WAS FOUND, 2026-09-10, while recalibrating box 3 for G32. A freshly measured calibration came
back with all four `*_triton_tflops` at 0.0. The raw worker output held the reason --
`triton_ceiling_error: ModuleNotFoundError: No module named 'kernel_optimizer'` -- but `Calibration`
has no field for that key, so pydantic dropped it and the cached file showed four clean zeros. Box 1's
cached calibration has the same four zeros.

So G10 -- whose entire finding is that cuBLAS is the wrong roof, over-reporting fp32 headroom by 19%
on box 1 and being EXCEEDED at fp16/bf16 (109.6% / 107.9%), which reads downstream as "saturated,
stop optimizing" -- has never once been in effect in a real run on either box. It worked only in the
standalone probe, which runs in the harness's own interpreter and therefore imports
`gpu.tritonmm` fine.

TWO INDEPENDENT DEFECTS, and both need fixing because either alone leaves the hole open:

  (1) THE WORKER CANNOT IMPORT THE HARNESS. It is documented as stdlib+torch+triton, and that is
      still the rule for what it may DEPEND on -- but two measurements import a harness module:
      `gpu.tritonmm` for the Triton ceilings and `evaluation.statics` for the SASS counters. Its
      PYTHONPATH carried kernelbench and an optional extra dir, never the harness's own `src`.

  (2) THE ZERO WAS INVISIBLE. Zero is a legitimate value (a box with no Triton), so it cannot raise
      -- but a calibration must be able to say "we never asked" rather than let a reader conclude
      "Triton cannot reach the roof on this card". Those are opposite conclusions with identical
      numbers. This is the same silent-zero shape as G10's own and G26's, one layer further out.

Fixing only (1) would leave the next such failure just as invisible; fixing only (2) would report a
gap on every run without closing it.
"""
from __future__ import annotations

from pathlib import Path

from kernel_optimizer.config import GpuConcurrencyConfig, WslConfig
from kernel_optimizer.evaluation.calibration import flag_suspect


def _client(tmp_path):
    from kernel_optimizer.gpu.worker_client import WslGpuWorker

    return WslGpuWorker(WslConfig(), GpuConcurrencyConfig(), jobs_dir=tmp_path / "jobs")


def _pythonpath(tmp_path) -> str:
    """The PYTHONPATH the worker will actually run with, read off the real command builder.

    Which of the two places to read is decided by the ROUTE, not by whether the key is present.
    Checking `env` first looked equivalent and was not: on the WSL route `env` is the whole
    `os.environ`, so a PYTHONPATH set in the shell that launched pytest is returned instead of the
    one the worker gets -- and it looks like a perfectly ordinary value. That is the same
    credible-wrong-reading shape this file exists to catch, hit while writing the file.
    """
    w = _client(tmp_path)
    argv, env = w._build_command(tmp_path / "j.json", tmp_path / "j.out.json")
    if Path(argv[0]).name.lower() != "wsl.exe":
        return env["PYTHONPATH"]
    # WSL route: the variables are a `VAR=x` prefix inside the shell command, not in the env.
    inner = argv[-1]
    for tok in inner.split():
        if tok.startswith("PYTHONPATH="):
            return tok[len("PYTHONPATH="):]
    raise AssertionError("no PYTHONPATH in the shell command: %r" % (inner,))


def test_the_worker_can_import_the_harness(tmp_path):
    """The measurement that was silently lost has to be able to import its module.

    Asserted on the path the worker is really launched with, not on a constant: `_build_command` is
    the single place that decides, and it differs by platform.

    The check is "some entry actually CONTAINS a `kernel_optimizer` package", not "some entry ends in
    src". The first version of this test used the latter and PASSED on the unfixed code, because
    `kernelbench_src` is itself `.../KernelBench/src` -- a vacuous assertion of exactly the kind that
    lets a defect ship green. Verified by reverting the fix: this now fails, that did not.
    """
    pp = _pythonpath(tmp_path)
    entries = [p for p in pp.split(":") if p]
    # Resolve back to host paths: the WSL route rewrites the prefix to /mnt/<drive>/...
    def _host(entry: str) -> Path:
        if entry.startswith("/mnt/") and len(entry) > 6:
            return Path("%s:/%s" % (entry[5], entry[7:]))
        return Path(entry)

    assert any((_host(e) / "kernel_optimizer" / "__init__.py").exists() for e in entries), (
        "no PYTHONPATH entry contains the `kernel_optimizer` package, so `from "
        "kernel_optimizer.gpu.tritonmm import reachable_tflops` raises inside a broad except and "
        "all four Triton-reachable ceilings degrade to 0.0 -- measured live on both boxes, with "
        "G10 therefore inert in every real run. PYTHONPATH was: %s" % pp)


def test_the_harness_src_is_appended_last(tmp_path):
    """Precedence, separately from presence: kernelbench must not be shadowed."""
    pp = _pythonpath(tmp_path)
    entries = [p for p in pp.split(":") if p]

    def _host(entry: str) -> Path:
        if entry.startswith("/mnt/") and len(entry) > 6:
            return Path("%s:/%s" % (entry[5], entry[7:]))
        return Path(entry)

    harness = [i for i, e in enumerate(entries)
               if (_host(e) / "kernel_optimizer" / "__init__.py").exists()]
    assert harness and harness[-1] == len(entries) - 1, (
        "the harness src is not the LAST PYTHONPATH entry, so it can shadow a module kernelbench "
        "or a pip --target dir provides: %s" % pp)


def test_kernelbench_still_comes_first(tmp_path):
    """Negative control: the addition must not shadow what was already there.

    kernelbench and any `pip --target` dir have to keep precedence -- the harness's src is appended
    last precisely so a name collision cannot change which kernelbench is imported.
    """
    pp = _pythonpath(tmp_path)
    entries = [p for p in pp.split(":") if p]
    assert len(entries) >= 2, "PYTHONPATH lost an entry: %s" % pp
    assert "KernelBench" in entries[0], (
        "kernelbench is no longer the first PYTHONPATH entry, so the harness's src could shadow a "
        "module it provides: %s" % pp)


def test_an_unmeasured_triton_ceiling_is_flagged():
    """A silent zero must become a stated gap. Drives the real `flag_suspect`."""
    notes = flag_suspect(1.686, 2.0, {
        "fp32_triton_tflops": 0.0, "tf32_triton_tflops": 0.0,
        "fp16_triton_tflops": 0.0, "bf16_triton_tflops": 0.0,
        "triton_ceiling_error": "ModuleNotFoundError: No module named 'kernel_optimizer'",
    })
    joined = " ".join(notes)
    assert "Triton-reachable" in joined, (
        "a calibration with all four Triton ceilings at 0.0 and a recorded import error reported "
        "nothing suspect, so the file reads as a clean measurement and a reader cannot distinguish "
        "'Triton cannot reach the roof here' from 'we never asked'")
    assert "kernel_optimizer" in joined, "the recorded cause was dropped from the note"


def test_a_partial_measurement_is_flagged_by_precision():
    """Naming WHICH precision is missing matters: bf16 is genuinely absent on old cards."""
    notes = flag_suspect(1.686, 2.0, {
        "fp32_triton_tflops": 18.0, "tf32_triton_tflops": 105.0,
        "fp16_triton_tflops": 186.9, "bf16_triton_tflops": 0.0,
    })
    joined = " ".join(notes)
    assert "bf16" in joined and "fp32" not in joined.split("(")[-1].split(")")[0], (
        "the note does not name the missing precision, so a card that genuinely lacks bf16 looks "
        "the same as a broken measurement: %s" % joined)


def test_a_complete_calibration_is_not_flagged():
    """Negative control. Flagging every calibration would make `suspect` meaningless."""
    notes = flag_suspect(1.686, 2.0, {
        "fp32_triton_tflops": 18.02, "tf32_triton_tflops": 105.02,
        "fp16_triton_tflops": 186.93, "bf16_triton_tflops": 187.45,
    })
    assert notes == [], (
        "a calibration with all four Triton ceilings measured was flagged as suspect, which trains "
        "the reader to ignore the field: %s" % notes)


def test_the_old_two_argument_call_still_works():
    """`worker_result` is optional, so no caller breaks and the DRAM check is unchanged."""
    assert flag_suspect(1.686, 2.0) == [], "a healthy DRAM ceiling was flagged"
    notes = flag_suspect(0.5, 2.0)
    assert notes and "throttled" in notes[0], (
        "the pre-existing throttle check stopped firing when the Triton check was added")


def test_the_assembler_actually_passes_the_worker_result():
    """The check is wired, not merely available.

    `flag_suspect` gained an OPTIONAL parameter, so `calibration_from_worker` calling it with two
    arguments still type-checks, still returns a clean list, and leaves the gap exactly as invisible
    as before. An optional parameter with no caller is the dormant-defect shape G26 hit, where
    `DevicePeaks` had no `l2_bytes` field at all and the gate reading it could never fire.

    Driven through the real assembly path on a worker result that reproduces the observed failure.
    """
    from kernel_optimizer.gpu.calibrate import calibration_from_worker

    cal = calibration_from_worker({
        "ok": True, "device_name": "NVIDIA A800 80GB PCIe", "capability": [8, 0], "sm_count": 108,
        "torch_version": "2.8.0+cu128", "driver_version": "12.8", "triton_version": "3.4.0",
        "dram_tbs": 1.686, "fp32_tflops": 19.0, "yardsticks": [],
        "fp32_triton_tflops": 0.0, "tf32_triton_tflops": 0.0,
        "fp16_triton_tflops": 0.0, "bf16_triton_tflops": 0.0,
        "triton_ceiling_error": "ModuleNotFoundError: No module named 'kernel_optimizer'",
    })
    assert any("Triton-reachable" in s for s in cal.suspect), (
        "the assembled Calibration records nothing suspect for a worker result whose four Triton "
        "ceilings all failed to measure, so the cached file reads as a clean measurement -- which "
        "is precisely how this went unnoticed on two boxes. suspect was: %s" % cal.suspect)
