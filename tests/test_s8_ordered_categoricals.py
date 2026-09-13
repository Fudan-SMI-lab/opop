"""S8 / item 3.1: the sampler is told which knobs have an ORDER.

The measurements these tests encode:
  - the premise, from the real corpus: mean |latency gap between ADJACENT values| over mean |gap
    between DISTANT values| has p50 = 0.735 < 1, so neighbouring settings behave more alike;
  - the predicate's coverage, from 9 events.jsonl files and 999 domain declarations: 858 `int`
    (211 of 216 distinct knobs have >= 3 values), 141 `str` (5 names, all precision switches), 0
    `float`, 0 numeric-valued `str`, 0 boolean-valued `int`;
  - the effect, measured with the shipping optuna 4.9.0 before any of this was written
    (`scripts/probes/is_categorical_distance_func_functional.py`, 8-rung ladder, +-16% noise, 120
    trials x 12 seeds): a SCRAMBLED rung order loses 34.1 of 120 near-optimum draws (12/12 seeds), and
    the ordered arm beats plain by +4.5 near-best draws and 12.1% mean cost (12/12 seeds).

The control that matters is scrambled-vs-ordered, NOT ordered-vs-plain: a CONSTANT distance function
does not reproduce plain TPE either (-22/600 near-best), because plain builds a count-based
categorical distribution while distance mode builds a kernel over distances. So a test that only
showed "distance differs from plain" would be showing the estimator change, not the ordering.
"""

from __future__ import annotations

import ast
import io
import warnings
from pathlib import Path

from kernel_optimizer.models.core import (
    Constraint,
    LatencyStats,
    ParamDomain,
    ParameterSpace,
    ParamSet,
    TrialRecord,
)
from kernel_optimizer.tuning import ordered_domains as od
from kernel_optimizer.tuning.tpe import OptunaTPETuner

SRC = Path("src/kernel_optimizer")


def _dom(name: str, kind: str, choices: list) -> ParamDomain:
    return ParamDomain(name=name, kind=kind, choices=choices)


# ---------------------------------------------------------------------------------------------
# the predicate: which knobs have an order, read off the VALUES
# ---------------------------------------------------------------------------------------------


def test_a_geometric_int_ladder_is_ordered():
    assert od.ordered_choices(_dom("BLOCK_N", "int", [16, 32, 64, 128])) == [16.0, 32.0, 64.0, 128.0]


def test_a_precision_switch_is_not_ordered():
    """The 141 `str` declarations in the corpus are 5 names, all precision/mode switches. Their
    "first" and "last" choices are an artifact of the order the agent wrote them, so a distance over
    them would hand the sampler a gradient over unrelated code paths.
    """
    assert od.ordered_choices(_dom("COMPUTE_DTYPE", "str", ["fp16", "bf16", "tf32", "ieee"])) is None


def test_the_predicate_reads_the_choices_and_not_the_declared_kind():
    """`kind` is the AGENT's declaration. `guard.py:113` does enforce the declared type, but a reader
    that depends on that enforcement inherits any hole in it -- and the question here is not what the
    agent called the knob, it is whether the values lie on a line. Measured: 0 such knobs in the
    corpus, so this is a guard against a case that has not happened yet rather than a fix for one
    that has.
    """
    # Declared str, values numeric => ordered.
    assert od.ordered_choices(_dom("BLOCK_M", "str", ["16", "32", "64"])) == [16.0, 32.0, 64.0]
    # Declared int, values not numeric => NOT ordered. (`kind` alone would have admitted it.)
    assert od.ordered_choices(_dom("MODE", "int", ["a", "b", "c"])) is None


def test_a_boolean_domain_is_never_ordered():
    """`float(True)` is 1.0, which would make an on/off switch look like an axis. Same exclusion as
    `wall_attribution._as_num`, and reusing that rule rather than writing a second one: two
    implementations of "is this ordered" are free to disagree, and only one of them is the rule the
    wall attribution used.
    """
    assert od.ordered_choices(_dom("USE_TMA", "int", [False, True, 1])) is None


def test_fewer_than_three_choices_gets_no_distance():
    """Two values give one gap, and a distance over one gap cannot tell a smooth axis from two
    isolated labels. 5 of the corpus's 216 distinct knobs are in this class.
    """
    assert od.ordered_choices(_dom("STAGES", "int", [2, 4])) is None
    assert od.ordered_choices(_dom("STAGES", "int", [2, 4, 8])) is not None


def test_a_duplicated_value_gets_no_distance():
    """Two identical rungs make the distance between two DIFFERENT labels zero, which tells the
    sampler they are interchangeable. Refuse rather than dedupe: a domain with a repeated choice is
    malformed, and silently repairing it would hide that from the space's own validation.
    """
    assert od.ordered_choices(_dom("BLOCK_N", "int", [16, 32, 32, 64])) is None


def test_an_empty_or_missing_choice_list_is_not_ordered():
    assert od.ordered_choices(_dom("X", "int", [])) is None


# ---------------------------------------------------------------------------------------------
# the distance itself: rungs, not raw values
# ---------------------------------------------------------------------------------------------


def test_the_distance_counts_rungs_not_raw_value_gaps():
    """|2048 - 1024| = 1024 while |32 - 16| = 16. A raw arithmetic distance would tell the sampler
    that the top of a geometric ladder is a distant, sparsely-explored region when it is one knob
    click away -- an artifact of the encoding rather than a property of the kernel.
    """
    d = od.rung_distance_for(_dom("BLOCK", "int", [16, 32, 64, 128, 256]))
    assert d(64, 128) == 1.0
    assert d(16, 32) == 1.0, "one click at the bottom must equal one click at the top"
    assert d(16, 256) == 4.0
    assert d(64, 64) == 0.0


def test_the_distance_is_symmetric_and_non_negative():
    """Optuna requires non-negativity. Symmetry is not required by the API but a non-symmetric
    distance would make "64 is near 128" and "128 is near 64" different claims.
    """
    d = od.rung_distance_for(_dom("BLOCK", "int", [16, 32, 64, 128]))
    for a in (16, 32, 64, 128):
        for b in (16, 32, 64, 128):
            assert d(a, b) >= 0.0
            assert d(a, b) == d(b, a)


def test_the_distance_sorts_the_choices_rather_than_trusting_their_order():
    """The contract says domains are ordered cheap-to-expensive, but the rung index must not DEPEND on
    that: a space whose choices arrive shuffled would otherwise get a distance that calls 16 and 128
    adjacent, which is worse than no distance at all.
    """
    d = od.rung_distance_for(_dom("BLOCK", "int", [128, 16, 64, 32]))
    assert d(16, 32) == 1.0
    assert d(16, 128) == 3.0


def test_an_unknown_value_is_far_rather_than_identical():
    """A value not in the map cannot happen for a draw from this domain, but `suggest_categorical` is
    also replayed with values from an enqueued anchor. Returning 0.0 there would declare an unknown
    value identical to everything, which is the `a-constant-reading-is-a-broken-probe` shape.
    """
    d = od.rung_distance_for(_dom("BLOCK", "int", [16, 32, 64]))
    assert d(999, 16) == 3.0
    assert d("nonsense", 16) == 3.0


def test_only_ordered_domains_appear_in_the_dict():
    """An unordered knob is OMITTED, not given a constant distance. Optuna applies its default
    (all choices equidistant) to any knob absent from the dict, which is the right model for a
    precision switch; a constant function would instead move that knob into the distance-kernel path,
    and the probe measured that path to differ from the default even when every distance is equal.
    """
    doms = [_dom("BLOCK_N", "int", [16, 32, 64]),
            _dom("COMPUTE_DTYPE", "str", ["fp16", "tf32"]),
            _dom("NUM_WARPS", "int", [2, 4, 8])]
    got = od.distance_funcs(doms)
    assert sorted(got) == ["BLOCK_N", "NUM_WARPS"]


def test_the_snapshot_names_both_sides():
    """`n_ordered` alone is unreadable: 5 of 5 and 5 of 12 are the same numerator with opposite
    meanings, and the omitted names are what a later reader checks the predicate against.
    """
    doms = [_dom("BLOCK_N", "int", [16, 32, 64]),
            _dom("COMPUTE_DTYPE", "str", ["fp16", "tf32", "ieee"])]
    snap = od.snapshot(doms)
    assert snap == {"n_domains": 2, "n_ordered": 1,
                    "ordered": ["BLOCK_N"], "unordered": ["COMPUTE_DTYPE"]}


# ---------------------------------------------------------------------------------------------
# the wiring: off by default, and ON must actually change the draws
# ---------------------------------------------------------------------------------------------

LADDER_SPACE = ParameterSpace(
    space_id="sp", candidate_id="c", source_sha="x",
    domains=[_dom("BLOCK", "int", [16, 32, 64, 128, 256, 512, 1024, 2048]),
             _dom("NUM_WARPS", "int", [2, 4, 8]),
             _dom("COMPUTE_DTYPE", "str", ["fp16", "bf16", "tf32"])],
    constraints=[Constraint(expr="BLOCK >= 16", rationale="always true; keeps the guard exercised")],
)
# 8 x 3 x 3 = 72 configurations. The budget below has to stay UNDER that: the tuner deduplicates
# draws, so a budget at or above the space size makes every arm cover the whole space and the only
# difference left is ordering. Below it, the arms differ in WHAT they sampled, which is the thing a
# prior is supposed to change.
_SPACE_SIZE = 72


def _drive(space: ParameterSpace, *, ordered: bool, n: int = 40, seed: int = 0) -> list[dict]:
    """Run the REAL tuner and return the drawn configurations in order.

    The objective is monotone in BLOCK so the ordering has something to exploit, and the tuner is the
    production one -- a test that reimplemented the ask loop would pass against a tuner that never
    passed the distance functions through, which is `a-test-that-copies-the-loop-does-not-test-it`.

    `n` stays under the space size for the reason above; the first version of this test asked 60 of a
    24-configuration space, got 24, and failed on the length rather than on the behaviour.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # optuna's own Future/Experimental warnings
        tuner = OptunaTPETuner(space, guard_ok=lambda p: True, budget=n, seed=seed,
                               ordered_categoricals=ordered)
        drawn: list[dict] = []
        while True:
            asked = tuner.ask()
            if asked is None:
                break
            trial_id, params = asked
            drawn.append(dict(params.values))
            rungs = space.domains[0].choices.index(params.values["BLOCK"])
            ms = float(len(space.domains[0].choices) - rungs)
            tuner.tell(trial_id, TrialRecord(
                trial_id=trial_id, candidate_id="c", space_id="sp", params=params,
                status="complete",
                latency_ms=LatencyStats(mean=ms, std=0.0, min=ms, max=ms, n_samples=20,
                                        median=ms)))
    return drawn


def test_off_by_default():
    """Same rule as every other v3 stage: it changes what the sampler draws, so a run with it on is
    not comparable with the finished ones.
    """
    from kernel_optimizer.config import AppConfig

    assert AppConfig().v3.ordered_categoricals.enabled is False


def test_the_switch_off_takes_the_old_path_literally():
    """Not "produces the same result" -- takes the SAME path. With the switch off the distance dict
    must never be built, so an unordered space and an ordered one are the same sampler, and no
    deprecated optuna argument is exercised at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = OptunaTPETuner(LADDER_SPACE, guard_ok=lambda p: True, budget=4, seed=0)
    assert off.ordered_categoricals is False
    # And the log entry the orchestrator writes is None in that case -- asserted on the orchestrator,
    # which is where it is derived; see `test_the_orchestrator_journals_the_snapshot_only_when_on`.


def test_the_switch_on_actually_changes_what_is_drawn():
    """The variant must change behaviour, or it is not a variant.

    This is the assertion that would have caught the one failure mode worth fearing here: optuna
    DEPRECATED `categorical_distance_func` in the version this project pins (4.9.0) and removes it in
    5.0.0. An argument accepted and silently ignored would have produced a treatment arm
    byte-identical to the control, with nothing in the log to say so.
    """
    plain = _drive(LADDER_SPACE, ordered=False)
    ordered = _drive(LADDER_SPACE, ordered=True)
    assert len(plain) == len(ordered) == 40
    assert plain != ordered, \
        "the distance functions changed nothing -- optuna may have stopped honouring them"


def test_the_ordering_concentrates_draws_nearer_the_optimum():
    """Not just "different" -- better, in the direction the prior claims.

    A prior that changed the draws at random would pass the difference test above while being worth
    nothing. Measured on the synthetic ladder in
    `scripts/probes/is_categorical_distance_func_functional.py`: +4.5 near-best draws and 12.1% lower
    mean cost against plain, 12 of 12 seeds, and 34.1 near-best draws better than the same kernel with
    a SCRAMBLED rung order. Asserted here over several seeds rather than one, because a single seed
    would make this a coin flip dressed as a measurement.
    """
    rungs = LADDER_SPACE.domains[0].choices

    def mean_rung(drawn: list[dict]) -> float:
        return sum(rungs.index(d["BLOCK"]) for d in drawn) / len(drawn)

    seeds = range(6)
    wins = sum(1 for s in seeds
               if mean_rung(_drive(LADDER_SPACE, ordered=True, seed=s))
               > mean_rung(_drive(LADDER_SPACE, ordered=False, seed=s)))
    assert wins >= 4, (
        f"the ordering concentrated nearer the optimum on only {wins} of 6 seeds; the objective here "
        f"is monotone toward the LAST rung, so a working prior should draw higher rungs more often")


def test_the_switch_on_reports_which_knobs_got_an_order():
    """The snapshot is derived by the ORCHESTRATOR from the config and the space, not stored on the
    tuner. Reason, and it was a real failure rather than a preference: it lived on the tuner first,
    and two existing tests whose `_Tuner` stub had no such attribute raised on the log line -- every
    object that can stand in for a tuner would have to remember it, and one that forgot would journal
    a silent None. So the assertion is on the predicate the orchestrator calls, on the same domains.
    """
    snap = od.snapshot(list(LADDER_SPACE.domains))
    assert snap["ordered"] == ["BLOCK", "NUM_WARPS"]
    assert snap["unordered"] == ["COMPUTE_DTYPE"]


def test_the_orchestrator_journals_the_snapshot_only_when_the_switch_is_on():
    """"asked for, nothing qualified" and "not asked for" are different states, and the LOG is where
    that distinction lives: the sampler gets None either way, because an empty dict and None are the
    same sampler and passing {} would emit optuna's deprecation warning for a call that asked nothing.

    Asserted on the orchestrator's own source, because the property is about which value reaches the
    event: gated on the config switch, and fed by the same predicate the sampler used.
    """
    text = io.open(SRC / "control" / "orchestrator.py", encoding="utf-8").read()
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_tune")
    body = ast.get_source_segment(text, fn) or ""
    assert "ordered_domains.snapshot" in body, \
        "the journalled snapshot must come from the same predicate the sampler used"
    assert "if self.cfg.v3.ordered_categoricals.enabled else None" in body, \
        "the log entry must be None when the switch is off, or a run without S8 gains a new key"
    assert body.count('"ordered_categoricals": ordered') == 2, \
        "both TUNING_DONE branches (a best was found, and none was) must carry the field"


def test_a_space_with_no_ordered_knob_still_journals_that_it_tried():
    """"nothing qualified" must be a DICT saying so, not an absence. `n_ordered: 0` says the predicate
    ran and found nothing; a missing key says the switch was off, and those call for opposite reads.
    """
    space = ParameterSpace(
        space_id="sp", candidate_id="c", source_sha="x",
        domains=[_dom("COMPUTE_DTYPE", "str", ["fp16", "bf16", "tf32"])],
        constraints=[])
    assert od.snapshot(list(space.domains)) == {"n_domains": 1, "n_ordered": 0,
                                               "ordered": [], "unordered": ["COMPUTE_DTYPE"]}
    # ...and the sampler is handed None rather than {} in that case, so no deprecated optuna argument
    # is exercised for a call that asked for nothing.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        OptunaTPETuner(space, guard_ok=lambda p: True, budget=4, seed=0,
                       ordered_categoricals=True)
        assert not any("categorical_distance_func" in str(x.message) for x in w), \
            "an all-unordered space must not trigger the deprecated argument at all"


def test_the_ordering_does_not_shrink_or_reorder_any_domain():
    """A distance is a PRIOR, never a restriction. The user's standing constraint is that the search
    space is not narrowed to control cost, and every value must stay drawable.

    Checked on the AST, not on the source text. A text scan for "choices =" fires on the DOCSTRING
    that explains why there is no such assignment, and a guard that forbids naming the thing it
    forbids gets "fixed" by deleting the explanation -- the failure recorded as
    `source-text-assertions-can-encode-the-bug`, and one this test hit on its first run.
    """
    text = io.open(SRC / "tuning" / "ordered_domains.py", encoding="utf-8").read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        # No assignment to, or deletion of, anything named `choices` / `domains`.
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for t in targets:
            name = getattr(t, "id", None) or getattr(t, "attr", None)
            assert name not in ("choices", "domains"), \
                f"ordered_domains must not rebind {name!r} -- a space is read, never narrowed"
        # No mutating method call on anything derived from a domain. `nums.append` is fine and has to
        # be: the predicate builds a fresh local list of numbers. What must never appear is a mutation
        # of `choices`, of `domain.<anything>`, or of the local the choices were read into -- those
        # would edit the space the tuner is about to sample from.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in ("remove", "pop", "clear", "insert", "extend", "append",
                                      "sort", "reverse"):
                continue
            recv = node.func.value
            recv_name = getattr(recv, "id", None) or getattr(recv, "attr", None)
            assert recv_name not in ("choices", "domains", "domain", "values"), \
                f"ordered_domains must not mutate {recv_name}.{node.func.attr}() -- " \
                f"a space is read, never narrowed"
    # And behaviourally: every value is still reachable when the budget allows the whole space.
    drawn = _drive(LADDER_SPACE, ordered=True, n=_SPACE_SIZE)
    seen = {d["BLOCK"] for d in drawn}
    assert seen == set(LADDER_SPACE.domains[0].choices), \
        f"a value became unreachable under the ordering: missing " \
        f"{set(LADDER_SPACE.domains[0].choices) - seen}"


def test_the_predicate_is_the_same_rule_the_wall_attribution_uses():
    """Two implementations of "is this knob ordered" are free to disagree, and then a wall could be
    attributed on a knob the sampler treats as unordered (or the reverse). Checked on behaviour over
    the cases where a second implementation would plausibly differ, not on source text.
    """
    from kernel_optimizer.evaluation import wall_attribution as wa

    for value in (16, 32.5, "128", "fp16", True, False, None, [1]):
        assert (od._as_num(value) is None) == (wa._as_num(value) is None), \
            f"the two orderedness rules disagree on {value!r}"


def test_the_tuner_passes_the_distance_to_the_sampler_and_not_somewhere_else():
    """A structural guard on the one line that can silently stop working. If a refactor drops the
    `categorical_distance_func=` keyword, the switch would still flip, the snapshot would still be
    journalled, and nothing would reach the sampler.
    """
    text = io.open(SRC / "tuning" / "tpe.py", encoding="utf-8").read()
    tree = ast.parse(text)
    init = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    call = next(n for n in ast.walk(init)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "TPESampler")
    kwargs = {k.arg for k in call.keywords}
    assert "categorical_distance_func" in kwargs, \
        "TPESampler is constructed without the distance argument; the switch would be inert"
