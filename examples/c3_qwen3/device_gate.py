from .device_records import KernelEvidence
from .runner_records import RunnerError


def launch_route(evidence: KernelEvidence) -> str | None:
    """Which independent observation establishes that the declared kernel actually ran.

    Returns "cuda_trace", "causal_chain", or None when neither holds.

    TWO ROUTES, AND WHY THERE HAS TO BE MORE THAN ONE
    -------------------------------------------------
    The original gate accepted exactly one observation: the declared kernel's name appearing among
    the CUDA-side events of a `torch.profiler` trace. That is a real proof when the profiler
    collected anything -- but it is an INSTRUMENT, and this instrument has been measured failing
    while the kernel demonstrably ran. Eight probes on box4 (B1-B8) ruled out every reconstructable
    cause: a profiling server, the candidate's bytes, the helper RPC path, several proves through one
    evaluator, a fresh profiler per session, the read position, each layer of the context stack, and
    the agent transport thread's lifetime. In the live F4 lane, three re-executions inside a single
    proof each returned `cuda_events == 0` while that same proof recorded `compiled`, `launches=3`,
    a TTIR store to the declared output operand with provenance from real inputs, pointer identity
    between that operand and the returned tensor, correct numerics, AND a failing skip control.

    So the second route is not a relaxation, it is a different set of observations that together are
    at least as demanding:

      cuda_trace    the declared kernel's name is in the trace's CUDA events. Proves a kernel with
                    that name executed on the device. It does NOT by itself prove the output came
                    from it -- which is why the participation checks below apply to both routes.

      causal_chain  THREE independent facts must hold together:
                    (a) the profiler's own CPU side recorded a CUDA launch API call
                        (`launch_api_seen`: cuLaunchKernelEx / cudaLaunchKernel / ...), so the
                        instrument agrees a launch was issued to the driver even though it returned
                        no device-activity records;
                    (b) the launch was counted at the launch site, inside the patched
                        `JITFunction.run`, only when Triton returned a real `CompiledKernel` and the
                        call was not a warmup;
                    (c) suppressing that kernel's stores makes the output WRONG
                        (`skip_control_rejected`). This is a counterfactual: if the returned value
                        did not depend on this kernel's execution, removing its stores would leave
                        the value unchanged. A host fallback, a dummy kernel or an unrelated
                        computation all fail it.

    THE ASYMMETRY THAT KEEPS THIS SOUND
    -----------------------------------
    The fallback is permitted ONLY when the profiler collected no CUDA activity at all -- i.e. when
    the instrument is known to have failed. If it DID collect CUDA activity and the declared name is
    absent, that absence is positive evidence the declared kernel did not run, and no fallback
    applies. `trace_attempts` is what makes the distinction available: an attempt with
    `cuda_events > 0` says collection worked on this device, in this process, for this proof.
    """
    collected = any((attempt.cuda_events or 0) > 0 for attempt in evidence.trace_attempts) or bool(evidence.cuda_names)
    if evidence.kernel_name and evidence.kernel_name in evidence.cuda_names:
        return "cuda_trace"
    if collected:
        # The profiler worked and did not see the declared kernel. That is a verdict, not a gap.
        return None
    launch_api = any(attempt.launch_api_seen for attempt in evidence.trace_attempts)
    if launch_api and evidence.launches >= 1 and evidence.skip_control_rejected and evidence.compiled:
        return "causal_chain"
    return None


def require_device_proof(evidence: KernelEvidence) -> None:
    """Reject anything that is not an authored device kernel actually producing the output.

    Three things must hold, and none of them is relaxed by `launch_route`'s second route:
      1. the declaration compiled to a real Triton kernel and ran (see `launch_route`);
      2. the operator's local numerics passed;
      3. the output participates -- the kernel stores to its declared output operand, that store
         depends on real inputs, the operand is the tensor returned, and suppressing the kernel
         breaks the result.
    """
    if not evidence.compiled or evidence.launches < 1 or not evidence.kernel_name:
        raise RunnerError("declared device kernel was not compiled and launched")
    route = launch_route(evidence)
    if route is None:
        if evidence.cuda_names or any((a.cuda_events or 0) > 0 for a in evidence.trace_attempts):
            raise RunnerError("declared device kernel was not compiled and CUDA-trace-confirmed launched")
        if not any(a.launch_api_seen for a in evidence.trace_attempts):
            raise RunnerError("declared device kernel launched but the profiler recorded neither device "
                              "activity nor a CUDA launch API call, so nothing corroborates the launch")
        raise RunnerError("declared device kernel launched but neither the CUDA trace nor the skip "
                          "control could confirm it produced the output")
    if evidence.local_quality_passed is False:
        raise RunnerError(evidence.quality_detail or "device probe local correctness failed")
    if (not evidence.output_used or not evidence.stored_output_args or not evidence.loaded_input_args
            or not evidence.skip_control_rejected):
        raise RunnerError("device output participation absent: dummy/unrelated/fallback computation")
