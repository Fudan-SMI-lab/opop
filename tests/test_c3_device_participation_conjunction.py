"""The four participation facts are a conjunction because each one alone is evadable — measured.

WHY THIS FILE EXISTS SEPARATELY FROM `test_c3_device_gate_two_routes.py`
-----------------------------------------------------------------------
That file pins the launch-route asymmetry. This one pins something different and easier to lose: that
the gate must require ALL FOUR output-participation facts, because a candidate can satisfy any subset
of them while its returned value comes from the host.

Those field vectors are not invented for this test. They are the values B10 measured on a real 4090
(`results/c3-fast-kernel-iteration/B5-profiler/B10-0921-061450/b10-result.json`) from five candidates
that all launch the SAME Triton kernel and differ only in what they do with its output — so each row
below is a shape a real adversarial candidate produced, not a hypothetical:

  arm                stored  loaded  output_used  skip_ctl  route (working prof.)  verdict
  honest             [1]     [0]     True         True      cuda_trace             accepted
  host_fallback      [1]     [0]     False        False     cuda_trace             rejected
  discarded_output   [1]     [0]     False        False     cuda_trace             rejected
  dummy_constant     []      []      True         False     cuda_trace             rejected
  unrelated_kernel   []      []      False        False     cuda_trace             rejected

Read the first three rows together: `host_fallback` and `discarded_output` produce a GENUINE TTIR store
to the declared output operand with provenance from real inputs, identical to the honest arm's. So
`stored_output_args` and `loaded_input_args` do not discriminate at all here, and a gate resting on
them would admit both. `dummy_constant` goes further and achieves pointer identity with the returned
tensor (`output_used=True`) while the host overwrote the value. Only the counterfactual separates all
four from honest work.

Note the route column: under a working profiler every arm is carried by `cuda_trace`, because every
arm really did launch the declared kernel. The route says whether the kernel RAN, never whether the
output came from it — so a rejection here must come from the participation check, and these tests
assert it does. Under a silent profiler the same arms are rejected one step earlier, at the route
itself, because the skip control gates `causal_chain` as well.

If a future change makes any one of these facts sufficient on its own, these tests fail.
"""

import json
from pathlib import Path

import pytest

from examples.c3_qwen3.device_gate import launch_route, require_device_proof
from examples.c3_qwen3.device_records import KernelEvidence, TraceAttempt
from examples.c3_qwen3.runner_records import RunnerError

MEASURED = (Path(__file__).resolve().parents[1] / "results" / "c3-fast-kernel-iteration" /
            "B5-profiler" / "B10-0921-061450" / "b10-result.json")

# The arms as B10 measured them: (stored, loaded, output_used, skip_control_rejected).
ARMS: dict[str, tuple[tuple[int, ...], tuple[int, ...], bool, bool]] = {
    "honest": ((1,), (0,), True, True),
    "host_fallback": ((1,), (0,), False, False),
    "discarded_output": ((1,), (0,), False, False),
    "dummy_constant": ((), (), True, False),
    "unrelated_kernel": ((), (), False, False),
}


def evidence(arm: str, *, silent: bool) -> KernelEvidence:
    """One arm's measured field vector, under a working profiler or a silent one.

    `silent` reproduces the instrument failure the second route exists for: no device-activity
    records, but the profiler's own CPU side still recorded the launch API call. Both are tested
    because an adversarial candidate does not get to choose which one it meets.
    """
    stored, loaded, output_used, skipped = ARMS[arm]
    attempts = tuple(TraceAttempt(index=i, events=11, cuda_events=0 if silent else 1, launch_calls=1,
                                  kernel_seen=not silent, launch_api_seen=True)
                     for i in range(3 if silent else 1))
    return KernelEvidence(backend="triton", source_file="operators.py", entry="kernel", compiled=True,
                          kernel_name="kernel", launches=3 if silent else 1,
                          cuda_names=() if silent else ("kernel",),
                          output_used=output_used, stored_output_args=stored, loaded_input_args=loaded,
                          skip_control_rejected=skipped, local_quality_passed=True,
                          trace_attempts=attempts)


@pytest.mark.parametrize("silent", [False, True], ids=["working_profiler", "silent_profiler"])
def test_the_honest_arm_is_accepted_on_both_routes(silent: bool) -> None:
    # Given: the only arm whose returned tensor is the one the kernel wrote.
    proof = evidence("honest", silent=silent)
    # Then: accepted, and by the route that matches which observation was available -- so a real
    # kernel is never failed for the instrument's failure, and never silently credited to the trace.
    assert launch_route(proof) == ("causal_chain" if silent else "cuda_trace")
    require_device_proof(proof)


@pytest.mark.parametrize("arm", [a for a in ARMS if a != "honest"])
@pytest.mark.parametrize("silent", [False, True], ids=["working_profiler", "silent_profiler"])
def test_every_non_producing_arm_is_rejected_on_both_routes(arm: str, silent: bool) -> None:
    # Given: a candidate that launches a real device kernel but whose returned value does not come
    # from it -- host fallback, dummy constant, discarded output, or unrelated device work.
    with pytest.raises(RunnerError):
        require_device_proof(evidence(arm, silent=silent))


def test_a_genuine_store_from_real_inputs_is_not_sufficient() -> None:
    # This is the finding that makes the conjunction load-bearing rather than defensive. `host_fallback`
    # and `discarded_output` present EXACTLY the TTIR facts the honest arm does: a store to the
    # declared output operand whose dataflow depends on real inputs. Anything that judged on the IR
    # alone would admit both, so the assertion is on the equality as much as on the rejection.
    # Measured under a WORKING profiler on purpose: the launch route is then satisfied (the kernel
    # really ran), so the rejection can only come from the participation check.
    honest, fallback = evidence("honest", silent=False), evidence("host_fallback", silent=False)
    assert (fallback.stored_output_args, fallback.loaded_input_args) == \
           (honest.stored_output_args, honest.loaded_input_args) == ((1,), (0,))
    assert launch_route(fallback) == "cuda_trace"
    with pytest.raises(RunnerError, match="output participation absent"):
        require_device_proof(fallback)


def test_pointer_identity_with_the_returned_tensor_is_not_sufficient() -> None:
    # `dummy_constant` passes the declared operand out as the returned tensor -- so `output_used` is
    # genuinely True -- after the host has overwritten the constant the kernel wrote into it.
    proof = evidence("dummy_constant", silent=False)
    assert proof.output_used and not proof.skip_control_rejected
    assert launch_route(proof) == "cuda_trace"
    with pytest.raises(RunnerError, match="output participation absent"):
        require_device_proof(proof)


def test_the_skip_control_is_what_separates_the_arms() -> None:
    # The discriminating field, stated as a property of the whole measured set rather than of one row:
    # the counterfactual is true for exactly the arm whose output really came from the kernel.
    assert {arm for arm in ARMS if ARMS[arm][3]} == {"honest"}


def test_a_silent_profiler_rejects_the_same_arms_one_step_earlier() -> None:
    # When the instrument fails, `causal_chain` needs the skip control too, so the non-producing arms
    # never reach the participation check. Both rejection points matter: a candidate does not choose
    # which profiler state it meets, and the gate must not be sound in only one of them.
    for arm in (a for a in ARMS if a != "honest"):
        proof = evidence(arm, silent=True)
        assert launch_route(proof) is None, arm
        with pytest.raises(RunnerError, match="neither the CUDA trace nor the skip control"):
            require_device_proof(proof)


def test_the_field_vectors_match_what_was_measured_on_hardware() -> None:
    # Keeps this file honest: the constants above are a transcription of a real run, so if the record
    # is present it must agree. Skipped rather than failed when the results tree is not checked out,
    # since the gate's behaviour is what the other tests assert and that needs no fixture.
    if not MEASURED.exists():
        pytest.skip(f"measured B10 record not present at {MEASURED}")
    record = json.loads(MEASURED.read_text(encoding="utf-8"))
    assert record["verdict"] == "CONTROL_PASSED_every_non_producing_kernel_rejected"
    assert not record["leaked_arms"] and not record["honest_rejected"]
    for key, observed in record["arms"].items():
        arm, lane = key.split("/")
        assert (tuple(observed["stored_output_args"]), tuple(observed["loaded_input_args"]),
                observed["output_used"], observed["skip_control_rejected"]) == ARMS[arm], key
        assert observed["valid"] is (arm == "honest"), key
        # A working profiler saw the declared kernel for EVERY arm -- they all really launched it --
        # so the route is `cuda_trace` regardless of verdict. Only the silent lane distinguishes.
        expected_route = ("cuda_trace" if lane == "normal" else
                          ("causal_chain" if arm == "honest" else None))
        assert observed["route"] == expected_route, key
