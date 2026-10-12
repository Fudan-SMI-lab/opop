"""The device gate accepts two independent observations of a launch, and refuses to confuse them.

WHY THERE ARE TWO
-----------------
The gate's job is to establish that the row's number came from an AUTHORED DEVICE KERNEL rather than
a host fallback or a dummy. Its original single observation was the declared kernel's name appearing
in a `torch.profiler` CUDA trace. That is a real proof -- but it is an instrument, and on box4 the
instrument has been measured failing while the kernel demonstrably ran: eight probes (B1-B8) ruled
out every reconstructable cause, and in the live F4 lane three re-executions inside ONE proof each
returned `cuda_events == 0` while that same proof recorded `compiled`, `launches=3`, a TTIR store to
the declared output operand with provenance from real inputs, pointer identity with the returned
tensor, correct numerics, and a FAILING skip control.

So the fallback route is a different observation, not a weaker one: the launch is counted inside the
patched `JITFunction.run` (only for a real `CompiledKernel`, only when not a warmup), and suppressing
that kernel's stores makes the output wrong. That last part is a counterfactual -- a host fallback, a
dummy kernel or an unrelated computation cannot pass it, because their output would be unchanged when
the kernel's stores are suppressed.

THE ASYMMETRY THESE TESTS PIN
-----------------------------
The fallback applies ONLY when the profiler collected no CUDA activity at all. If it collected CUDA
activity and the declared name is absent, that absence is positive evidence against the declaration
and nothing rescues it. Without that asymmetry the second route would be a hole rather than a
second observation, so it is tested in both directions.
"""

import pytest

from examples.c3_qwen3.device_gate import launch_route, require_device_proof
from examples.c3_qwen3.device_records import KernelEvidence, TraceAttempt
from examples.c3_qwen3.runner_records import RunnerError


def evidence(**overrides) -> KernelEvidence:
    """A proof that passes everything; each test breaks exactly one thing."""
    base = dict(backend="triton", source_file="operators.py", entry="k", compiled=True,
                kernel_name="rmsnorm_kernel", launches=1, cuda_names=("rmsnorm_kernel",),
                output_used=True, stored_output_args=(0,), loaded_input_args=(1, 2),
                skip_control_rejected=True, local_quality_passed=True)
    return KernelEvidence(**{**base, **overrides})


def attempts(*cuda_counts: int, launch_api: bool = True) -> tuple[TraceAttempt, ...]:
    return tuple(TraceAttempt(index=i, events=7, cuda_events=n, launch_calls=1,
                              kernel_seen=n > 0, launch_api_seen=launch_api)
                 for i, n in enumerate(cuda_counts))


def test_a_named_kernel_in_the_cuda_trace_is_accepted_by_the_trace_route() -> None:
    proof = evidence(trace_attempts=attempts(1))
    assert launch_route(proof) == "cuda_trace"
    require_device_proof(proof)


def test_a_silent_profiler_falls_back_to_the_causal_chain() -> None:
    # Given: the live failure shape -- every attempt collected CPU events only, but those CPU events
    # DID include a CUDA launch API call, the launch was counted, and the skip control failed (so the
    # output depended on this kernel).
    proof = evidence(cuda_names=(), launches=3, trace_attempts=attempts(0, 0, 0))
    # Then: the causal chain carries it. The kernel is not rejected for an instrument's failure.
    assert launch_route(proof) == "causal_chain"
    require_device_proof(proof)


def test_the_fallback_needs_the_profilers_own_launch_record() -> None:
    # Given: the profiler saw no device activity AND no launch API call either -- it recorded nothing
    # about a launch at all. Then only our own patched counter claims one happened, and the fallback
    # would rest entirely on our instrumentation with no independent corroboration.
    proof = evidence(cuda_names=(), launches=3, trace_attempts=attempts(0, 0, launch_api=False))
    assert launch_route(proof) is None
    with pytest.raises(RunnerError, match="nor a CUDA launch API call"):
        require_device_proof(proof)


def test_a_working_profiler_that_did_not_see_the_kernel_is_still_a_rejection() -> None:
    # Given: CUDA activity WAS collected (some other kernel), but not the declared one. That absence
    # is positive evidence the declaration did not run -- the opposite situation from a silent
    # profiler -- so the fallback must not apply, even though the causal fields all look fine.
    proof = evidence(cuda_names=("someone_elses_kernel",), trace_attempts=attempts(1, 1))
    assert launch_route(proof) is None
    with pytest.raises(RunnerError, match="CUDA-trace-confirmed"):
        require_device_proof(proof)


def test_one_collecting_attempt_disables_the_fallback_for_the_whole_proof() -> None:
    # Given: collection worked on at least one attempt, so this device/process/proof demonstrably can
    # collect. A later empty attempt cannot then be read as "the instrument is broken".
    proof = evidence(cuda_names=(), trace_attempts=attempts(0, 2, 0))
    assert launch_route(proof) is None
    with pytest.raises(RunnerError):
        require_device_proof(proof)


def test_a_silent_profiler_does_not_rescue_a_kernel_whose_output_is_unused() -> None:
    # Given: a silent profiler AND a skip control that passed -- suppressing the kernel's stores left
    # the output correct, so the output did not come from this kernel. This is exactly the dummy/
    # fallback case the gate exists to catch, and the second route must not admit it.
    proof = evidence(cuda_names=(), skip_control_rejected=False, trace_attempts=attempts(0, 0))
    assert launch_route(proof) is None
    with pytest.raises(RunnerError, match="neither the CUDA trace nor the skip control"):
        require_device_proof(proof)


def test_a_silent_profiler_does_not_rescue_a_kernel_that_never_launched() -> None:
    proof = evidence(cuda_names=(), launches=0, trace_attempts=attempts(0))
    with pytest.raises(RunnerError, match="was not compiled and launched"):
        require_device_proof(proof)


def test_a_silent_profiler_does_not_rescue_an_uncompiled_declaration() -> None:
    proof = evidence(cuda_names=(), compiled=False, trace_attempts=attempts(0))
    with pytest.raises(RunnerError, match="was not compiled and launched"):
        require_device_proof(proof)


def test_neither_route_waives_numerics_or_output_participation() -> None:
    # The two routes decide only whether the kernel RAN. Everything downstream still applies to both,
    # so a silent profiler cannot be used to slip past correctness or participation. The numerics
    # failure surfaces the measured detail rather than a generic label, which is what a reader needs.
    with pytest.raises(RunnerError, match="mismatch at 3 positions"):
        require_device_proof(evidence(cuda_names=(), local_quality_passed=False,
                                      quality_detail="mismatch at 3 positions", trace_attempts=attempts(0)))
    with pytest.raises(RunnerError, match="local correctness failed"):
        require_device_proof(evidence(cuda_names=(), local_quality_passed=False,
                                      quality_detail=None, trace_attempts=attempts(0)))
    for broken in ({"output_used": False}, {"stored_output_args": ()}, {"loaded_input_args": ()}):
        with pytest.raises(RunnerError, match="output participation absent"):
            require_device_proof(evidence(cuda_names=(), trace_attempts=attempts(0), **broken))
        with pytest.raises(RunnerError, match="output participation absent"):
            require_device_proof(evidence(trace_attempts=attempts(1), **broken))


def test_the_route_is_recorded_on_the_published_row() -> None:
    # A reader of the final table must be able to tell which observation carried each row, rather
    # than assuming the trace did.
    from kernel_optimizer.models.core import ParamSet
    from kernel_optimizer.models.device_operator import DeviceKernelDeclaration
    from examples.c3_qwen3.device_records import FixtureDescription, FixtureResult, LocalReport, StageIdentity
    from examples.c3_qwen3.fast_formal import row_evidence
    from examples.c3_qwen3.fast_records import Artifact
    from pathlib import Path

    stage = StageIdentity(profile="c3_fast_device", stage="final", framework_id="F")
    declaration = DeviceKernelDeclaration(backend="triton", source_file="operators.py", entry="k",
                                          output_arg_indices=(0,))
    description = FixtureDescription(site_id="s", module_path="m", goal="ttft", phase="prefill",
        decode_step=0, batch_size=1, prompt_length=8, tensors=(), source="src", source_sha256="h",
        contract_sha256="c", corpus_sha256="p")
    for cuda_names, counts, expected in ((("rmsnorm_kernel",), (1,), "cuda_trace"), ((), (0, 0), "causal_chain")):
        proof = evidence(cuda_names=cuda_names, trace_attempts=attempts(*counts))
        report = LocalReport(stage=stage, valid=True, bundle_sha256="b", params_sha256="p",
            source_hashes={}, declared_kernels=(declaration,), counts={}, wall_ms=1.0,
            raw_path=Path("receipt.json"),
            fixtures=(FixtureResult(fixture_identity="i", description=description, quality_passed=True,
                                    device_evidence=(proof,)),))
        artifact = Artifact(author="agent", version=0, session_id="s", bundle=Path("bundle.json"),
            bundle_sha256="b", source_hashes={"operators.py": "h"}, params=ParamSet(values={}),
            params_sha256="p", kernels=(declaration,))
        row = row_evidence("ttft", artifact, (report,), None, stage, "base", None, None)
        assert row.device_proven and row.kernels[0].launch_route == expected
