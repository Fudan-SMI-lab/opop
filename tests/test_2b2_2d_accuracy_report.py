"""2b(2) + 2d: the per-dimension accuracy table, the four miss shapes, and `under-converged`.

Fixtures are shaped from the real `EXPECTATIONS_RECONCILED` payload (arm 2,
run-l3-43-20260912-214326): `payload.reconciliation.per_dimension[]` with
`dimension/expected/actual/match/before/after/delta/rel/unit/why`, plus `dimensions_unpredicted`
and `dimensions_unmeasured`. Copied from the emitter rather than invented -- a fixture shaped to match
the reader proves only that the reader reads itself.
"""

from __future__ import annotations

from kernel_optimizer.reporting.accuracy_report import accuracy_lines

DIMS = ("candidate_aten_bytes", "candidate_aten_ops", "n_regs", "n_spills",
        "occupancy", "peak_alloc_bytes", "shared_bytes", "threads_launched")


def _row(dim, expected, actual, match, rel=0.5, delta=-1.0):
    return {"dimension": dim, "expected": expected, "actual": actual, "match": match,
            "before": 100.0, "after": 50.0, "delta": delta, "rel": rel, "unit": "u",
            "why": "because"}


def _recon(cid, rows, *, unpredicted=(), unmeasured=(), family="fam-1", rnd=1):
    return {"type": "EXPECTATIONS_RECONCILED",
            "ts": 1000.0,
            "payload": {"candidate_id": cid, "family_id": family, "round": rnd,
                        "n_declared": len(rows),
                        "reconciliation": {"hypothesis_id": "H1", "per_dimension": list(rows),
                                           "hits": sum(1 for r in rows if r["match"] == "hit"),
                                           "misses": sum(1 for r in rows if r["match"] == "miss"),
                                           "vacuous": sum(1 for r in rows
                                                          if r["match"] == "vacuous"),
                                           "dimensions_unpredicted": list(unpredicted),
                                           "dimensions_unmeasured": list(unmeasured),
                                           "caveat": "ATTRIBUTION CAVEAT: ..."}}}


def _trials(cid, latencies, t0=0.0):
    """Completed trials for a candidate, in order, so best-so-far can be traced."""
    out = []
    for i, ms in enumerate(latencies):
        out.append({"type": "TRIAL_DONE", "ts": t0 + i,
                    "payload": {"trial": {"candidate_id": cid, "status": "complete",
                                          "latency_ms": {"mean": ms, "median": ms, "std": 0.1,
                                                         "min": ms, "max": ms, "n_samples": 20},
                                          "profile": {"shared_bytes": 1024}}}})
    return out


def test_the_section_is_absent_when_nothing_was_reconciled():
    """A run with no ledger must read exactly as before."""
    assert accuracy_lines([]) == []
    assert accuracy_lines([{"type": "TRIAL_DONE", "payload": {}}]) == []


def test_reads_events_of_either_shape():
    """`report.py` passes objects with `.type`/`.payload`; probes pass dicts. A reader handling only
    one silently returns an empty section for the other, which reads as "no ledger" -- this
    project's recurring failure mode."""
    ev = _recon("c1", [_row("n_regs", "up", "up", "hit")])

    class Ev:
        def __init__(self, d):
            self.type, self.payload, self.ts = d["type"], d["payload"], d.get("ts", 0.0)

    as_dicts = accuracy_lines([ev])
    as_objs = accuracy_lines([Ev(ev)])
    assert as_dicts and as_objs
    assert as_dicts == as_objs


def test_the_table_is_per_dimension_and_keeps_dimensions_apart():
    """The whole point: one row per dimension, each with its OWN accuracy. A pooled number would
    simultaneously overstate shared_bytes and understate aten_ops."""
    evs = [
        _recon("c1", [_row("candidate_aten_ops", "down", "down", "hit"),
                      _row("shared_bytes", "down", "flat", "miss")]),
        _recon("c2", [_row("candidate_aten_ops", "down", "down", "hit"),
                      _row("shared_bytes", "up", "flat", "miss")]),
    ]
    out = "\n".join(accuracy_lines(evs))
    assert "`candidate_aten_ops` | 2 | 2 | **100%**" in out
    assert "`shared_bytes` | 2 | 0 | **0%**" in out


def test_a_dead_lever_is_counted_separately_from_a_direction_flip():
    """Two miss shapes with different fixes: "the action never reached the dimension" against "the
    model of the mechanism is inverted". Pooling them makes the fix unaddressable."""
    evs = [_recon("c1", [_row("n_regs", "up", "flat", "miss"),
                         _row("n_spills", "down", "up", "miss")])]
    out = "\n".join(accuracy_lines(evs))
    assert "死杠杆" in out
    assert "方向反了" in out
    assert "说 down、实测 up" in out
    # And the table must attribute each to its own dimension, not to a shared bucket.
    assert "`n_regs` | 1 | 0 | **0%** | 1 | 0 | 0 |" in out
    assert "`n_spills` | 1 | 0 | **0%** | 0 | 0 | 1 |" in out


def test_under_converged_fires_when_the_child_was_still_improving_at_the_end():
    """2d's positive case. The best refresh lands in the final 10% of the pass, so "the dimension did
    not move" is our sampling limit rather than the agent's error."""
    evs = [_recon("kid", [_row("peak_alloc_bytes", "down", "flat", "miss")])]
    # 10 trials whose best arrives LAST -- still improving when the budget ended.
    evs += _trials("kid", [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.2, 2.0])
    out = "\n".join(accuracy_lines(evs))
    assert "其中 1 条疑似「未测够」" in out
    assert "`peak_alloc_bytes` | 1 | 0 | **0%** | 1 | 1 |" in out


def test_under_converged_does_NOT_fire_when_the_child_had_settled():
    """The negative control. A pass whose best stopped improving early is converged, and its dead
    lever is the agent's miss -- re-labelling it would launder a real miss."""
    evs = [_recon("kid", [_row("peak_alloc_bytes", "down", "flat", "miss")])]
    # best arrives at trial 1 of 10 and never improves again.
    evs += _trials("kid", [2.0, 9.0, 8.5, 8.4, 8.3, 8.2, 8.1, 8.0, 7.9, 7.8])
    out = "\n".join(accuracy_lines(evs))
    assert "其中 0 条疑似「未测够」" in out


def test_under_converged_is_NOT_gated_on_a_trial_COUNT():
    """The defect that made this label structurally dead, pinned so it cannot come back.

    The first version also required "<= 20 completed trials". Measured across the three step-3 runs,
    dead-lever children have 27 to 70 completed trials, so the label could NEVER fire while the
    report printed "0 suspected under-measured" as though it had looked. A count was the wrong proxy
    anyway: 69 trials that stop improving at 47% is converged; 29 trials still improving at 93% is not.

    So: a child with MANY trials that is still improving must still be flagged.
    """
    evs = [_recon("kid", [_row("occupancy", "down", "flat", "miss")])]
    evs += _trials("kid", [float(70 - i) for i in range(68)] + [1.5, 1.0])   # 70 trials, best last
    out = "\n".join(accuracy_lines(evs))
    assert "其中 1 条疑似「未测够」" in out, (
        "a 70-trial child still improving at the end was not flagged -- a trial-count gate is back")


def test_under_converged_is_a_relabel_and_never_a_promotion_to_hit():
    """It marks doubt about a miss; it must not change the accuracy. Otherwise 2d would quietly
    inflate the very number the section exists to report honestly."""
    rows = [_row("occupancy", "down", "flat", "miss")]
    settled = [_recon("kid", rows)] + _trials("kid", [2.0] + [8.0] * 9)
    improving = [_recon("kid", rows)] + _trials("kid", [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                                                       1.5, 1.0])
    a = "\n".join(accuracy_lines(settled))
    b = "\n".join(accuracy_lines(improving))
    assert "`occupancy` | 1 | 0 | **0%**" in a
    assert "`occupancy` | 1 | 0 | **0%**" in b, (
        "under-converged changed the accuracy -- it must re-label the miss, not promote it")
    assert "其中 0 条" in a and "其中 1 条" in b


def test_the_implementation_rate_condition_travels_with_every_number():
    """Each accuracy holds only over the hypotheses the agent chose to implement (8-15%). Without
    that sentence the table reads as "the agent does not understand shared memory", which is an
    over-generalisation the data does not support."""
    out = "\n".join(accuracy_lines([_recon("c1", [_row("shared_bytes", "down", "flat", "miss")])]))
    assert "8–15%" in out
    assert "不能" in out and "过度概括" in out


def test_an_impossible_miss_shape_is_not_reported_as_a_zero_finding():
    """When the agent declares EVERY dimension, "moved but never mentioned" cannot occur -- and
    printing "0 unforeseen effects" would claim a finding where the shape had no room to happen.
    Measured on all three step-3 runs: 8 of 8 dimensions declared in every round."""
    rows = [_row(d, "down", "down", "hit") for d in DIMS]
    out = "\n".join(accuracy_lines([_recon("c1", rows)]))
    assert "不可能" in out
    assert "这不是「查过了没有」" in out


def test_an_unforeseen_side_effect_is_reported_when_it_DOES_occur():
    """The positive control for the branch above: given a genuinely unmentioned dimension that moved,
    the section must name it. Otherwise the "impossible" message could be hiding a real finding."""
    rows = [_row("n_regs", "up", "up", "hit")]
    out = "\n".join(accuracy_lines([_recon("c1", rows, unpredicted=("shared_bytes",))]))
    assert "未预料的副作用" in out
    assert "`shared_bytes`×1" in out
    assert "不可能" not in out


def test_unmeasured_declarations_stay_out_of_the_denominator():
    """A declaration with no reading on one side is neither a hit nor a miss. Counting it either way
    would move the accuracy on the strength of a measurement nobody took."""
    evs = [_recon("c1", [_row("n_regs", "up", "up", "hit"),
                         _row("occupancy", "down", "unknown", "unmeasured")],
                  unmeasured=("occupancy",))]
    out = "\n".join(accuracy_lines(evs))
    assert "合计准确率 100%" in out, "an unmeasured declaration was counted in the denominator"
    assert "无读数" in out


def test_vacuous_declarations_are_reported_but_not_graded():
    """`vacuous` is the agent declining to declare. It belongs in the table (a dimension it never
    commits on is a fact about the agent) but not in the accuracy."""
    evs = [_recon("c1", [_row("n_regs", "up", "up", "hit"),
                         _row("shared_bytes", "unknown", "down", "vacuous")])]
    out = "\n".join(accuracy_lines(evs))
    assert "合计准确率 100%" in out
    assert "空洞 1 条" in out
