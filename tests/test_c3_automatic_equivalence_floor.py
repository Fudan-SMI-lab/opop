"""The automatic equivalence floor: measured from the site's own original, with no authored kernel.

WHY THIS EXISTS
---------------
`runner_quality.equivalence_floor` has always been able to raise the logits limit to a MEASURED floor,
but the only way to produce one was `fast_setup.write_floor_evidence`, which reads it off a
hand-authored correct control -- a Triton kernel someone wrote for that specific operator. So a task
could not calibrate a floor until it already had a correct kernel for the site, and the mechanism sat
inert (`floor_evidence` was None everywhere on disk).

`measure_equivalence_floor` removes that dependency. It perturbs the site's OWN original output by one
unit in the last place and measures what that costs at the logits. Nothing about it is
operator-specific, so it works at an RMSNorm, a projection or an activation alike.

WHAT THE MEASURED NUMBERS WERE (probes B23/B24 on the Qwen3 TTFT task)
---------------------------------------------------------------------
    site                      modules   one-ULP floor   declared limit
    model.norm                     73        0.016163             0.01
    model.layers.0.mlp.up_proj     72        0.017743             0.01
    self_attn.q_proj               36        0.019323             0.01
    mlp.act_fn                     36        0.015628             0.01

with every unperturbed control measuring exactly 0.0, and the contract's own declared per-operator
tolerance (rtol 0.02) implying 0.035628 -- so each floor sits between "bit-identical" and "as much
deviation as the contract already permits".

WHAT THESE TESTS PIN
--------------------
1. The floor document has exactly the identity `equivalence_floor` verifies, and that function accepts
   it -- the two halves cannot drift apart.
2. A floor RAISES the logits limit and never lowers it, and never touches the NLL criterion.
3. The measurement needs no authored kernel: it is driven off `binding.originals`.
4. It restores the module tree exactly, so a floor measurement cannot leave a perturbation installed --
   which would silently corrupt every later evaluation.
5. Evidence for a different site or module count is refused, because the floor is a property of the
   group (the same kernel measured 0.0 at one module and 0.010349 at 73).
6. THE SAFETY PROPERTY: raising the logits limit to the floor does not admit an unfaithful replacement,
   because faithfulness is enforced by the per-operator check, not by the logits gate. Pinned here from
   the B24 measurements, so the reasoning cannot be quietly reversed later.
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

from examples.c3_qwen3.fast_evaluation import EvaluationFiles
from examples.c3_qwen3.fast_setup import measure_equivalence_floor
from examples.c3_qwen3.runner_quality import CRITERIA, FLOOR_METRIC, criteria, equivalence_floor
from examples.c3_qwen3.runner_records import RunnerError

CONTRACT = "2c0e2fc70c76d4b780cb1301addcf2d723ec5f19f106af7ac3919fbc86996bd8"
SITE = "pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd46b715a8c7a1fa0"
MODULES = 73
MEASURED_FLOOR = 0.01616308601590823          # B23, model.norm, 73 modules
DECLARED_LIMIT = 0.01
TOLERANCE_CEILING = 0.03562812038262915       # B13: what the contract's own rtol=0.02 implies


def floor_document(**overrides) -> dict:
    document = {"contract_sha256": CONTRACT, "site_id": SITE, "module_count": MODULES,
                "logits_relative_l2_max": MEASURED_FLOOR,
                "paired_mean_nll_delta_nat_per_token": 0.0029535801100899484,
                "local_checks_passed": 1.0, "ulps": 1,
                "source": "automatic 1-ULP perturbation of the site's own original forward output",
                "control_valid": True, "goal": "ttft"}
    document.update(overrides)
    return document


def write(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "equivalence-floor.json"
    path.write_text(json.dumps(floor_document(**overrides)), encoding="utf-8")
    return path


# --- the document the measurement writes is the document the gate reads ----------------------


def test_the_written_floor_is_accepted_by_the_gate(tmp_path):
    # Given: the producer and the consumer are in different modules. If their agreed shape drifts, a
    # measured floor would be silently unusable -- or worse, raise mid-lane.
    assert equivalence_floor(write(tmp_path), CONTRACT, SITE, MODULES) == MEASURED_FLOOR


def test_absent_evidence_leaves_the_declared_limit_exactly_as_it_was():
    # Given: every task that never measures a floor must be bit-for-bit unaffected.
    assert equivalence_floor(None, CONTRACT, SITE, MODULES) is None
    assert criteria(None, None) == CRITERIA


def test_a_floor_raises_the_logits_limit_and_leaves_the_nll_limit_alone():
    raised = dict((key, limit) for key, limit, _ in criteria(None, MEASURED_FLOOR))
    declared = dict((key, limit) for key, limit, _ in CRITERIA)
    assert raised[FLOOR_METRIC] == MEASURED_FLOOR > declared[FLOOR_METRIC]
    assert raised["paired_mean_nll_delta_nat_per_token"] == declared["paired_mean_nll_delta_nat_per_token"]


def test_a_floor_below_the_declared_limit_cannot_lower_it():
    # Given: a small site group measures a floor of 0.0 (B12: one module, and four). That must not
    # tighten the contract's declared limit -- the floor only ever raises.
    assert dict((k, v) for k, v, _ in criteria(None, 0.0))[FLOOR_METRIC] == DECLARED_LIMIT


def test_the_floor_stays_below_what_the_contract_already_permits():
    # Given: the floor is meant to be the MINIMAL correction. The contract's own per-operator tolerance
    # implies 0.035628 at the logits (B13), so a one-ULP floor must sit well inside that -- otherwise it
    # would be conceding more than the contract itself does.
    assert DECLARED_LIMIT < MEASURED_FLOOR < TOLERANCE_CEILING


# --- identity: a floor describes ONE site group ---------------------------------------------


@pytest.mark.parametrize("field,value", [("site_id", "pure-someothergroup"), ("module_count", 36),
                                         ("contract_sha256", "0" * 64)])
def test_evidence_for_a_different_group_is_refused(tmp_path, field, value):
    # Given: the same kernel measured 0.0 at one module and 0.010349 at 73, so a floor from another
    # group says nothing about this one and must not be silently reused.
    with pytest.raises(RunnerError, match="different contract/site/module count"):
        equivalence_floor(write(tmp_path, **{field: value}), CONTRACT, SITE, MODULES)


def test_a_missing_declared_evidence_file_is_an_error(tmp_path):
    with pytest.raises(RunnerError, match="equivalence-floor evidence is missing"):
        equivalence_floor(tmp_path / "absent.json", CONTRACT, SITE, MODULES)


@pytest.mark.parametrize("bad", [None, "0.01", float("nan"), float("inf"), -0.001])
def test_a_floor_without_a_finite_measurement_is_refused(tmp_path, bad):
    with pytest.raises(RunnerError, match="no finite measured logits error"):
        equivalence_floor(write(tmp_path, logits_relative_l2_max=bad), CONTRACT, SITE, MODULES)


# --- the measurement needs no authored kernel, and must leave no trace ----------------------


def measurement_code() -> str:
    """The EXECUTABLE body of `measure_equivalence_floor`, with its docstring removed.

    The docstring discusses the alternatives it rejects by name (`finfo.eps`, the authored-control
    route), so a substring search over the raw source would report the opposite of the truth for every
    one of these assertions.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(measure_equivalence_floor)))
    function = tree.body[0]
    body = function.body[1:] if isinstance(function.body[0], ast.Expr) else function.body
    return "\n".join(ast.unparse(node) for node in body)


def test_the_measurement_is_driven_off_the_bindings_own_originals():
    # Given: the whole point is that no per-site reference implementation is required. The reference must
    # come from `binding.originals`, the untouched forwards the framework captured before anything was
    # installed -- which a replacement cannot reach for itself, since the anti-cheat coverage check
    # exists precisely to stop that.
    code = measurement_code()
    assert "binding.originals[path]" in code


def test_the_measurement_goes_through_a_real_bundle_so_the_receipt_names_the_site():
    """`evaluate_quality` consults the floor only when `runner.receipt.site_ids` is non-empty.

    A row that replaces nothing IS the baseline, which has no amplification to correct, so the floor
    branch is skipped for it. Measured the hard way: a first version installed the perturbed forward
    directly onto the modules, leaving the receipt empty -- so the calibration ran against what the gate
    treated as a baseline and wrote `logits_relative_l2_max: 0.0` for a 73-module group whose true
    one-ULP floor is 0.016163. That is worse than failing, because a 0.0 floor cannot raise the limit and
    the lane looks calibrated while enforcing the declared value.
    """
    code = measurement_code()
    assert "runner.bind(" in code, "the arm must be bound, not assigned onto the modules"
    assert "receipt.site_ids" in code, "an empty receipt must be refused, not measured"


def test_the_originals_are_registered_into_the_bound_namespace_not_a_reimport():
    # Given: `BundleImport.open()` imports a fresh module each call. Registering through a second import
    # would put the originals in a namespace the INSTALLED replacement never reads, so every call would
    # raise KeyError. The handle has to be the bound callable's own globals.
    code = measurement_code()
    assert "__globals__" in code
    assert "imported.open()" not in code


def test_the_floor_measurement_restores_the_binding():
    # Given: the measurement binds a perturbed arm across up to 73 live modules. If it is left bound,
    # every later evaluation in that process silently measures a perturbed model -- a defect that would
    # look like noise rather than a bug.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(measure_equivalence_floor)))
    tries = [n for n in ast.walk(tree.body[0]) if isinstance(n, ast.Try) and n.finalbody]
    assert tries, "the bind must be wrapped in try/finally"
    assert any("restore" in ast.unparse(node) for t in tries for node in t.finalbody)


def test_a_failed_measurement_raises_rather_than_reporting_no_floor():
    # Given: returning None means "no floor to measure", which leaves the declared limit in force. If a
    # crashed measurement returned None too, a broken calibration would be indistinguishable from a
    # deliberate decision. The quality call must therefore sit OUTSIDE any exception handler that
    # swallows it.
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(measure_equivalence_floor)))
    for node in ast.walk(tree.body[0]):
        if isinstance(node, ast.Try) and "runner.quality" in ast.unparse(node):
            assert not node.handlers, "the measurement must not catch its own failure"


def test_the_floor_is_reproducible_rather_than_random():
    # Given: a floor is evidence. A differently-seeded run producing a different number would make the
    # gate's limit unreproducible, so the perturbation's generator must be seeded.
    assert "manual_seed" in measurement_code()


def test_the_step_is_taken_on_the_bit_pattern_not_on_finfo_eps():
    # Given: `finfo.eps` is the relative spacing at 1.0, but a float's spacing varies by a factor of two
    # across each binade, so using eps would overstate one ULP by up to 2x on part of the range. The
    # step must come from the integer representation. The arithmetic lives in the bundle source the
    # function emits, so that is what is checked.
    code = measurement_code()
    assert ".view(int_dtype)" in code
    assert "finfo" not in code
    assert "manual_seed" in code


def test_the_step_separates_sign_from_magnitude_and_saturates_below_infinity():
    """The two ways a naive bit-step produces NaN, both measured rather than reasoned about.

    Stepping the SIGNED integer word directly is wrong twice over:

      * for a negative value the word is negative, so adding a positive delta moves the magnitude DOWN
        rather than away from zero -- the perturbation is not the step it claims to be;
      * negative zero is the most-negative word (-32768 in int16), and `(-32768).abs()` overflows back
        to -32768, so a clamp written as `bits.abs() + delta < 0` passes and the step lands in the
        exponent-all-ones region, i.e. NaN.

    Measured consequence: a single -0.0 anywhere in an RMSNorm output made every logit non-finite and the
    whole calibration failed with `RunnerError: invalid logits/target in teacher-forced quality` -- so the
    lane wrote no floor at all (B25-0921-183007, both sites, `wrote_a_floor: false`, no exception raised
    because the failure surfaced as an invalid measurement rather than a crash).
    """
    code = measurement_code()
    # The magnitude field must be isolated from the sign bit before stepping.
    assert "sign_bit" in code and "& ~sign_bit" in code
    # And the step must saturate at the largest FINITE magnitude, so it cannot fabricate inf or NaN.
    assert "_MAX_FINITE" in code
    assert "clamp(min=0, max=" in code
    # The old, broken formulation must be gone.
    assert "bits.abs() + deltas < 0" not in code


def test_the_declared_max_finite_patterns_are_the_real_ones():
    # Given: `_MAX_FINITE` is a table of bit patterns. A wrong entry would either clamp far too early
    # (silently shrinking the step for large values) or not at all (letting a step reach infinity).
    # These are the IEEE encodings one below each dtype's infinity.
    code = measurement_code()
    for pattern in ("0x7F7F",          # bfloat16: exponent 0xFE, mantissa all ones
                    "0x7BFF",          # float16
                    "0x7F7FFFFF",      # float32
                    "0x7FEFFFFFFFFFFFFF"):   # float64
        assert pattern in code, f"missing max-finite pattern {pattern}"


def emitted_step():
    """Compile the ULP step OUT OF THE EMITTED BUNDLE SOURCE and return it, with a stub original.

    The assertions above read the source, which cannot tell a real fix from a comment about one. This
    executes the arithmetic the lane actually runs: the function renders its replacement as bundle text,
    so the text is extracted, compiled, and driven directly.
    """
    import ast
    import inspect
    import textwrap
    from types import SimpleNamespace

    tree = ast.parse(textwrap.dedent(inspect.getsource(measure_equivalence_floor)))
    template = None
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            rendered = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
            if "floor_replacement" in rendered:
                template = rendered
                break
    assert template, "could not find the emitted bundle source"
    # The f-string's only interpolation is ULPS; the doubled braces are already single in `.value`.
    namespace: dict = {}
    exec(compile(template.replace("ULPS = ", "ULPS = 1  # "), "<emitted-bundle>", "exec"), namespace)
    namespace["_ORIGINALS"]["m"] = lambda value: value
    replacement = namespace["floor_replacement"]

    def step(tensor):
        site = SimpleNamespace(module_path="m")
        call = SimpleNamespace(args=(tensor,), kwargs={})
        return replacement(site, call, {})

    return step


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_the_emitted_step_is_finite_for_every_input_including_negative_zero(dtype_name):
    """Executed, not inspected: the exact arithmetic the lane runs, on the values that broke it.

    -0.0 is the case that failed in production (B25-0921-183007): its bit pattern is the most-negative
    integer word, whose `abs()` overflows to itself, so the old clamp passed and the step produced NaN.
    One such element anywhere in the output made every logit non-finite.
    """
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    step = emitted_step()
    largest = torch.finfo(dtype).max
    hard = torch.tensor([[0.0, -0.0, 1.0, -1.0, 2.0, -2.0, largest, -largest,
                          torch.finfo(dtype).tiny, -torch.finfo(dtype).tiny]], dtype=dtype)
    out = step(hard)
    assert torch.isfinite(out).all(), f"non-finite step for {dtype_name}: {out.tolist()}"
    # The sign must never flip: that would be a huge change, not one ULP.
    nonzero = hard != 0
    assert bool(((hard > 0) == (out > 0))[nonzero].all())
    # And the relative change must be at most one ULP -- 2^-8 for bf16, 2^-23 for fp32 -- so the floor
    # really is the SMALLEST implementation difference rather than an arbitrary epsilon.
    relative = ((out.float() - hard.float()).abs() / hard.float().abs().clamp(min=1e-30))
    assert relative[nonzero].max().item() <= torch.finfo(dtype).eps * 1.01


def test_the_calibration_captures_the_local_fixtures_it_needs():
    """`runner.quality` runs `validate_local`, which reads `binding.fixtures` -- keyed by site_id.

    The search lanes never fill that map: `FastEvaluator.capture` populates its own `engine.fixtures`
    tuple, used by the per-candidate local gate, and nothing in the dev or formal path calls
    `runner.capture_fixtures`. Measured consequence: the live lane wrote the floor arm's bundle and then
    produced no floor and no reason file at all, because the measurement died with
    `RunnerError: missing original local fixtures` (F5-dev-0921-184103). Its first candidate was then
    rejected at 0.014172 under the declared 0.01, which the floor of 0.016163 would have admitted.

    Capture must also happen while nothing is bound -- `capture_fixtures` refuses once
    `binding.active` is non-empty -- so the order is restore, capture, bind.
    """
    import ast
    import inspect
    import textwrap

    code = measurement_code()
    assert "capture_fixtures" in code, "the calibration must ensure its own local fixtures exist"
    # Ordered: the capture has to precede the bind, or it is refused outright.
    assert code.index("capture_fixtures") < code.index("runner.bind("), "capture before bind"
    # And it must be conditional, so a lane that already captured them is not made to redo the work.
    tree = ast.parse(textwrap.dedent(inspect.getsource(measure_equivalence_floor)))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and "capture_fixtures" in ast.unparse(node)]
    assert guards, "the capture must be guarded by a check for existing fixtures"
    assert "binding.fixtures" in ast.unparse(guards[0].test)


def test_the_emitted_step_is_reproducible_and_actually_changes_the_output():
    # Given: a floor is evidence, so the same bytes must measure the same number twice. And a step that
    # changed nothing would report a floor of 0.0, which raises no limit at all.
    torch = pytest.importorskip("torch")
    values = (torch.randn(4096) * 3).to(torch.bfloat16)
    first, second = emitted_step()(values), emitted_step()(values)
    assert torch.equal(first, second), "the perturbation must be seeded"
    changed = (first != values).float().mean().item()
    assert changed > 0.9, f"only {changed:.3f} of elements moved; the floor would understate"


# --- the delegation window: needed by the calibration, closed to candidates -----------------


def binding_double():
    """A `ModelBinding` with its coverage counters set by hand, and no model behind it.

    `require_coverage` reads only `self.active`, `self.sites` and `self.counts`, so the guard's decision
    can be exercised exactly as written without a resident model.
    """
    from examples.c3_qwen3.binding_runtime import ModelBinding

    binding = ModelBinding.__new__(ModelBinding)
    binding.active = {"site": lambda site, call, params: None}
    binding.sites = {"site": ("m0",)}
    binding.installed = {}
    binding.instance_forward = {"m0": False}
    binding.modules = {}
    binding.originals = {}
    binding.imported = None
    binding.bundle = None
    binding.expect_delegation = False
    return binding


def test_a_candidate_that_calls_the_original_is_still_rejected():
    """The anti-cheat rule, unchanged: a replacement may not delegate to the original.

    This is what stops a candidate returning the original's output and claiming the speedup, so it has to
    keep failing by default -- the delegation window must be an explicit opt-in, not a relaxation.
    """
    binding = binding_double()
    binding.counts = {"m0": {"replacement_calls": 64, "old_calls": 64}}
    with pytest.raises(RunnerError, match="not executed exclusively"):
        binding.require_coverage()


def test_the_calibration_window_permits_delegation_for_that_measurement_only():
    binding = binding_double()
    binding.counts = {"m0": {"replacement_calls": 64, "old_calls": 64}}
    with binding.delegating():
        binding.require_coverage()          # permitted inside the window
    with pytest.raises(RunnerError, match="not executed exclusively"):
        binding.require_coverage()          # and refused again immediately after


def test_the_window_still_requires_the_replacement_to_have_run():
    # Given: the window relaxes ONE condition. An arm that never executed is not a measurement of
    # anything, so `replacement_calls == 0` must still fail even inside it.
    binding = binding_double()
    binding.counts = {"m0": {"replacement_calls": 0, "old_calls": 0}}
    with binding.delegating(), pytest.raises(RunnerError, match="not executed exclusively"):
        binding.require_coverage()


def test_the_window_closes_even_when_the_measurement_raises():
    binding = binding_double()
    binding.counts = {"m0": {"replacement_calls": 1, "old_calls": 1}}
    with pytest.raises(ValueError):
        with binding.delegating():
            raise ValueError("measurement blew up")
    assert binding.expect_delegation is False
    with pytest.raises(RunnerError, match="not executed exclusively"):
        binding.require_coverage()


def test_the_window_cannot_be_nested():
    # Given: nesting would mean one caller's exit re-arms the guard while another still relies on it.
    binding = binding_double()
    with binding.delegating():
        with pytest.raises(RunnerError, match="nested delegation"):
            with binding.delegating():
                pass


def test_a_bundle_cannot_request_the_delegation_window():
    # Given: the window is safe only because a candidate cannot reach it. A replacement is handed
    # `(site, call, params)`, and `OperatorSite` carries no path back to the runtime -- so a bundle has
    # no reference through which to enable it, and nothing in a bundle document names it.
    from dataclasses import fields as dataclass_fields

    from examples.c3_qwen3.operator_types import OperatorSite

    exposed = {f.name for f in dataclass_fields(OperatorSite)}
    assert exposed == {"module", "weights", "module_path"}
    assert "binding" not in exposed and "runtime" not in exposed

    import examples.c3_qwen3.model_binding as model_binding

    assert "delegating" not in inspect_source(model_binding), (
        "no bundle-loading path may enter the delegation window")


def inspect_source(module) -> str:
    import inspect

    return inspect.getsource(module)


def test_only_the_floor_calibration_opens_the_window():
    # Given: the window's safety rests on how few places can open it. If it spreads, the anti-cheat rule
    # is effectively gone. Exactly one caller is expected: the floor measurement.
    import inspect
    from pathlib import Path as _Path

    import examples.c3_qwen3 as package

    root = _Path(inspect.getfile(package)).parent
    callers = sorted(path.name for path in root.glob("*.py")
                     if ".delegating()" in path.read_text(encoding="utf-8"))
    assert callers == ["fast_setup.py"], f"unexpected callers of the delegation window: {callers}"


# --- the safety property, pinned from measurement ------------------------------------------


def test_raising_the_limit_to_the_floor_still_rejects_every_unfaithful_arm():
    """B24, measured at model.norm (73 modules) and mlp.act_fn (36 modules).

    Faithfulness is enforced by the PER-OPERATOR check against fixtures captured from the original
    dispatch, not by the logits gate. So raising the logits limit cannot admit a replacement that does
    not compute the operator: those never reach the logits criterion at all.

    The numbers below are the measured `local_checks_passed` / `logits_relative_l2_max` pairs. They are
    pinned because the OPPOSITE conclusion was reached first, by using deliberately-lower-precision arms
    as an admissibility ceiling -- and that was unsound twice over: those arms compute the correct
    operator (the contract's own rtol=0.02 admits them), and the statistic does not order accuracy at
    this scale (an arm that REMOVED a rounding, and so was more accurate than the reference, measured
    the largest error of all the equivalent arms because the reference itself rounds).
    """
    arms = {                      # name: (faithful, local_checks_passed, logits_l2)
        "faithful_reference":    (True,  1.0, 0.0),
        "one_ulp_random":        (True,  1.0, MEASURED_FLOOR),
        "within_declared_2pct":  (True,  1.0, 0.03346653174055632),
        "beyond_declared_5pct":  (False, 0.0, None),
        "no_weight_dropped":     (False, 0.0, None),
        "identity_passthrough":  (False, 0.0, None),
        "zeros":                 (False, 0.0, None),
    }
    limit = dict((k, v) for k, v, _ in criteria(None, MEASURED_FLOOR))[FLOOR_METRIC]
    admitted = [name for name, (_, local, l2) in arms.items()
                if local == 1.0 and l2 is not None and l2 <= limit]
    unfaithful_admitted = [name for name in admitted if not arms[name][0]]
    assert unfaithful_admitted == [], "a floor must never admit a replacement that is not faithful"
    # Every unfaithful arm was stopped by the local check, before the logits criterion was consulted.
    assert all(local == 0.0 for faithful, local, _ in arms.values() if not faithful)
    # And the floor is what makes the faithful one-ULP arm admissible: under the declared limit it is not.
    assert MEASURED_FLOOR > DECLARED_LIMIT
    assert "one_ulp_random" in admitted


def test_the_rejected_c3_candidates_were_all_faithful():
    # Given: 11 receipts across the F4/F5 campaigns. EVERY one recorded local_checks_passed 1.0 and
    # failed on the logits criterion alone, with the paired NLL delta never breaching its own limit.
    # That is why a floor is the right correction: the gate was rejecting faithful work for the site
    # group's size, not catching defects.
    rejected = [  # (logits_l2, paired_nll_delta, local_checks_passed)
        (0.010397708337059153, 1.0692785471166921e-05, 1.0),
        (0.011155061927544058, 0.0009616312926555848, 1.0),
        (0.011807271831267152, 0.0022121287942246104, 1.0),
        (0.012069943102365413, 0.0016449592283810377, 1.0),
        (0.014172352920463037, 0.0012745305877456259, 1.0),
    ]
    nll_limit = dict((k, v) for k, v, _ in CRITERIA)["paired_mean_nll_delta_nat_per_token"]
    assert all(local == 1.0 for _, _, local in rejected), "all were faithful"
    assert all(nll < nll_limit for _, nll, _ in rejected), "none breached the NLL criterion"
    assert all(l2 > DECLARED_LIMIT for l2, _, _ in rejected), "all failed the declared logits limit"
    # The measured floor admits all of them.
    assert all(l2 <= MEASURED_FLOOR for l2, _, _ in rejected)


def test_the_floor_field_is_plumbed_all_the_way_to_the_matrix():
    # Given: a measured floor is worthless if the lane that measures it cannot hand it to the evaluation
    # and the matrix. Pinned separately in test_c3_floor_evidence_reaches_the_matrix.py; asserted here
    # too so this file fails if the producer is wired to a field that no longer exists.
    assert "floor_evidence" in {f.name for f in fields(EvaluationFiles)}
    assert EvaluationFiles(Path("b"), Path("o"), Path("out")).floor_evidence is None


def test_the_lane_records_the_floor_it_enforced():
    # Given: two lanes can produce the same verdict under different limits. A published result has to
    # state which limit produced it, or the number cannot be interpreted later.
    from examples.c3_qwen3.fast_records import ProbeResult

    assert "floor_evidence" in ProbeResult.model_fields
    assert ProbeResult.model_fields["floor_evidence"].default is None


def test_a_lane_that_cannot_measure_a_floor_keeps_running_under_the_declared_limit():
    """The floor is optional, so failing to measure one must not abort a lane.

    A floor can ONLY raise the logits limit, so its absence is strictly conservative: the lane simply
    enforces what the contract declares, which is the behaviour every earlier campaign had. Aborting
    instead would make a torchless or non-tensor backend unable to run at all -- and the real integration
    tests run exactly that configuration.

    The reason is written out, so a lane that silently ran under the declared limit can be distinguished
    from one that measured a floor and chose it.
    """
    import ast
    import inspect
    import textwrap

    import examples.c3_qwen3.fast_workflow as workflow

    tree = ast.parse(textwrap.dedent(inspect.getsource(workflow.run_probe)))
    guarded = [node for node in ast.walk(tree)
               if isinstance(node, ast.Try) and "measure_equivalence_floor" in ast.unparse(node)]
    assert guarded, "the calibration call must be guarded"
    handled = {name for node in guarded for handler in node.handlers
               for name in ast.unparse(handler.type or ast.Constant(None)).strip("()").split(",")}
    # ImportError covers a host without torch; RunnerError covers an unregistered site or a backend that
    # cannot produce the metric; TypeError covers an output that is not a single tensor.
    assert {"ImportError", "RunnerError", "TypeError"} <= {name.strip() for name in handled}
    body = "\n".join(ast.unparse(handler) for node in guarded for handler in node.handlers)
    assert "equivalence-floor-unavailable.json" in body, "the reason must be recorded, not swallowed"
    # And the failure must not be re-raised: that would abort the lane.
    assert not any(isinstance(node, ast.Raise) for n in guarded for handler in n.handlers
                   for node in ast.walk(handler))
