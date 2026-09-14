"""v4.1 §9: the six paper fixtures + the positive controls the review requires.

Every negative assertion here has a companion positive control that would fail loudly if
the mechanism under test were disconnected (probe-needs-a-positive-control discipline).
Fixture values come from the spec's own constructions (N={100,100}/F={80,80} etc.), not
invented to match the reader.
"""

from __future__ import annotations

import math

import pytest

from kernel_optimizer.conditional import gates
from kernel_optimizer.conditional.gates import evaluate_gates, reversed_interval
from kernel_optimizer.conditional.probe import DBudget, retry_prefix
from kernel_optimizer.conditional.scanner import ConditionalScanner
from kernel_optimizer.conditional.tokens import SourceContrast, TokenStore
from kernel_optimizer.models.core import ParamDomain, ParameterSpace


def _space(axes: int = 4, choices: int = 4) -> ParameterSpace:
    return ParameterSpace(
        space_id="sp-test", candidate_id="cand-test", version=1, source_sha="deadbeef",
        domains=[ParamDomain(name=f"K{i}", kind="int",
                             choices=[16 * (2 ** j) for j in range(choices)])
                 for i in range(axes)],
        constraints=[])


def _contrast(f_lat, n_lat, axis="K0", f_value=16, n_value=32, cid="sc-1",
              full=True) -> SourceContrast:
    c = SourceContrast(
        contrast_id=cid, comparable_key=f"id|pk|{axis}|F={f_value}|N={n_value}|C4",
        execution_identity="cand-test|triton|v4.1", candidate_id="cand-test",
        space_id="sp-test", axis=axis, partner_projection="pk",
        partner_values={"K1": 16, "K2": 16, "K3": 16},
        f_value=f_value, n_value=n_value, f_lat=list(f_lat), n_lat=list(n_lat),
        full=full)
    if full and len(f_lat) >= 2 and len(n_lat) >= 2:
        c.mint = evaluate_gates(list(f_lat), list(n_lat))
    return c


# ---------------------------------------------------------------- fixture 1 + 2: direction


class TestDirectionMint:
    def test_fixture1_negative_wallward_mints_one_inward_token(self):
        """N={100,100}, F={80,80}: toward-wall = −25%, cannot pass the outward gate;
        I_rev = +20% passes inward. Exactly ONE token, direction inward; the original
        wallward g_d/y are not redirected."""
        m = evaluate_gates([80.0, 80.0], [100.0, 100.0])
        assert m.direction == "inward"
        assert m.outward.passed is False and m.inward.passed is True
        # the spec's own arithmetic: toward-wall −25%, reversed +20%
        assert m.outward.interval[0] == pytest.approx(-0.25)
        assert m.inward.interval[0] == pytest.approx(0.20)

    def test_fixture2_positive_wallward_mints_one_outward_token(self):
        """F={100,100}, N={80,80}: same geometry, mirrored latencies → outward."""
        m = evaluate_gates([100.0, 100.0], [80.0, 80.0])
        assert m.direction == "outward"
        assert m.outward.passed is True and m.inward.passed is False
        assert m.outward.interval[0] == pytest.approx(0.20)

    def test_positive_control_gates_are_connected(self):
        """A flat contrast passes NEITHER gate — proves the pass branch above is not a
        constant-True read."""
        m = evaluate_gates([100.0, 100.0], [100.0, 100.0])
        assert m.direction is None

    def test_rho_gate_unresolved_when_spread_exceeds_gain(self):
        """High replicate spread lifts tau above the gain: no token."""
        m = evaluate_gates([100.0, 80.0], [100.0, 80.0])
        assert m.direction is None
        assert m.rho_f is not None and m.rho_f > 0.04

    def test_reversed_interval_algebra(self):
        lo, hi = reversed_interval((-0.25, -0.25))
        assert lo == pytest.approx(0.20) and hi == pytest.approx(0.20)

    def test_single_replicate_is_unknown_not_zero(self):
        """rho with one value is UNKNOWN (None) — never imputed to 0 (v4.1 §5)."""
        assert gates.rho([100.0]) is None
        m = evaluate_gates([80.0], [100.0, 100.0])
        assert m.direction is None and "UNKNOWN" in m.reason


# ---------------------------------------------------------------- fixture 3: supersession


class TestSupersession:
    def test_fixture3_newer_full_negative_revokes_old_outward(self):
        store = TokenStore()
        old = _contrast([100.0, 100.0], [80.0, 80.0], cid="sc-old")
        tok_old = store.register(old)
        assert tok_old is not None and tok_old.direction == "outward"
        newer = _contrast([80.0, 80.0], [100.0, 100.0], cid="sc-new")
        tok_new = store.register(newer)
        assert tok_old.state == "revoked"
        assert tok_new is not None and tok_new.direction == "inward"

    def test_fixture3_unknown_quality_does_not_mint(self):
        store = TokenStore()
        c = _contrast([80.0], [100.0, 100.0], cid="sc-u")
        c.mint = evaluate_gates(c.f_lat, c.n_lat)
        assert store.register(c) is None

    def test_fixture3_provisional_never_supersedes(self):
        store = TokenStore()
        old = _contrast([100.0, 100.0], [80.0, 80.0], cid="sc-old")
        tok_old = store.register(old)
        prov = _contrast([80.0, 80.0], [100.0, 100.0], cid="sc-prov", full=False)
        assert store.register(prov) is None
        assert tok_old.state == "pending"      # provisional shape did not suppress it

    def test_admission_consumes_and_never_refunds(self):
        store = TokenStore()
        tok = store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        store.consume(tok.token_id, "admitted E1")
        assert tok.state == "consumed"
        with pytest.raises(ValueError):
            store.consume(tok.token_id, "again")


# ---------------------------------------------------------------- fixture 4: budgets


class TestBudgets:
    def test_fixture4_b20_inherited_e2_pays_but_cold_start_cannot_c4(self):
        space = _space()
        # inherited token in a continuation space with B=20 (Q=2): E2 pays.
        store = TokenStore()
        tok = store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        tok.action_kind = "E2"
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=20,
                                tokens=store, initial_phase="E")
        block = sc.next_block([], fits_screen=None)
        assert block is not None and block.kind == "E2" and len(block.points) == 2
        # cold start with B=20 (Q=2): C4 needs 4 — cannot admit.
        sc2 = ConditionalScanner(space, "cand-test", "triton", budget_b=20,
                                 tokens=TokenStore())
        from kernel_optimizer.conditional.probe import ConditionedWall
        wall = ConditionedWall(kind="hard", axis="K0", partner_key="pk",
                               partner_values={"K1": 16, "K2": 16, "K3": 16},
                               point_map={}, f_value=16, n_value=32, refused_value=64)
        assert sc2.next_block([wall]) is None

    def test_failed_block_consumes_no_refund(self):
        space = _space()
        store = TokenStore()
        tok = store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=40,
                                tokens=store, initial_phase="E")
        block = sc.next_block([])
        assert block is not None and block.kind == "E1"
        q_after = sc.q_remaining()
        sc.tell(block, block.points[0], None, ok=False)   # the action fails
        assert sc.q_remaining() == q_after                 # no refund
        assert tok.state == "consumed"

    def test_b_remaining_gates_admission(self):
        space = _space()
        store = TokenStore()
        store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=40,
                                tokens=store, initial_phase="E")
        assert sc.next_block([], b_remaining=0) is None


# ---------------------------------------------------------------- fixture 5: bootstrap


class TestBootstrap:
    def test_fixture5_timeout_plus_retry_success_never_refreezes_d(self):
        b = DBudget()
        b.record(170.0, timed_out=True, failed=False, n_points=48)   # first: timeout
        assert b.retried and not b.stopped
        b.after_retry(30.0, timed_out=False, failed=False)           # retry: success
        assert b.stopped                        # post-retry stop, whatever the outcome
        assert b.bootstrap_spent_s == pytest.approx(200.0)
        assert b.frozen_d is None               # NEVER re-frozen to 60 from the retry
        assert b.admit(10.0) is False

    def test_first_success_freezes_and_continues(self):
        b = DBudget()
        b.record(100.0, timed_out=False, failed=False, n_points=48)
        assert b.frozen_d == pytest.approx(200.0)
        assert not b.stopped and b.admit(90.0) is True    # can continue inside D
        assert b.admit(150.0) is False                    # but not beyond it

    def test_non_timeout_failure_stops_without_retry(self):
        b = DBudget()
        b.record(50.0, timed_out=False, failed=True, n_points=48)
        assert b.stopped and not b.retried

    def test_retry_prefix_is_frozen_order_half(self):
        planned = [{"i": i} for i in range(48)]
        prefix = retry_prefix(planned)
        assert prefix == planned[:24]           # order preserved, never result-selected


# ---------------------------------------------------------------- fixture 6: identifiability


class TestIdentifiability:
    def test_fixture6_axis_pair_counts(self):
        assert math.comb(8, 2) == 28
        assert math.comb(12, 2) == 66
        assert math.comb(16, 2) == 120

    def test_fixture6_single_source_rr_equals_gain(self):
        """One eligible source: RR target == gain target — recorded, and NOT evidence
        against C2 (the decision record says differed=False)."""
        space = _space()
        store = TokenStore()
        store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=40,
                                tokens=store, initial_phase="E")
        block = sc.next_block([])
        assert block is not None
        rec = sc.eligibility_log[-1]
        assert rec.n_eligible == 1 and rec.rr_target == rec.gain_target
        assert rec.differed is False

    def test_two_sources_gain_ranking_can_differ_from_rr(self):
        """Positive control for the decision record: with two eligible sources of
        different gains the ranking is exercised for real."""
        space = _space()
        store = TokenStore()
        store.register(_contrast([100.0, 100.0], [80.0, 80.0], axis="K0", cid="sc-a"))
        store.register(_contrast([100.0, 100.0], [60.0, 60.0], axis="K1",
                                 f_value=16, n_value=32, cid="sc-b"))
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=80,
                                tokens=store, initial_phase="E")
        block = sc.next_block([])
        assert block is not None
        rec = sc.eligibility_log[-1]
        assert rec.n_eligible == 2
        # same span stratum → gain ranking picks the steeper source (K1: 40% gain)
        assert rec.gain_target is not None
        chosen_axis = block.axis
        assert chosen_axis == "K1"


# ---------------------------------------------------------------- direction geometry


class TestEGeometry:
    def _scanner_with_token(self, direction: str):
        space = _space(axes=2, choices=6)      # K0: 16,32,64,128,256,512
        store = TokenStore()
        if direction == "outward":
            c = _contrast([100.0, 100.0], [80.0, 80.0], f_value=32, n_value=64)
        else:
            c = _contrast([80.0, 80.0], [100.0, 100.0], f_value=32, n_value=64)
        c.partner_values = {"K1": 16}
        store.register(c)
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=40,
                                tokens=store, initial_phase="E")
        return sc, store

    def test_outward_extends_beyond_n(self):
        sc, _ = self._scanner_with_token("outward")
        block = sc.next_block([])
        assert block is not None
        assert block.points[0].values["K0"] == 128      # beyond N=64, toward the wall
        assert block.points[0].direction == "outward"

    def test_inward_extends_beyond_f(self):
        """The v4-review construction: improvement observed on the inner side → the next
        point is BEYOND F away from the wall (x0 side), not the wall side."""
        sc, _ = self._scanner_with_token("inward")
        block = sc.next_block([])
        assert block is not None
        assert block.points[0].values["K0"] == 16       # beyond F=32, away from the wall
        assert block.points[0].direction == "inward"

    def test_screen_gap_is_never_crossed(self):
        space = _space(axes=2, choices=6)
        store = TokenStore()
        c = _contrast([100.0, 100.0], [80.0, 80.0], f_value=32, n_value=64)
        c.partner_values = {"K1": 16}
        store.register(c)
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=40,
                                tokens=store, initial_phase="E")
        from kernel_optimizer.conditional.scanner import ConditionalScannerKeys
        gap_key = ConditionalScannerKeys.point_key({"K0": 128, "K1": 16})
        block = sc.next_block([], fits_screen={gap_key: False})
        # 128 is refused; stepping across to 256 is forbidden → fall to in-interval,
        # and with none unmeasured the block cannot form.
        assert block is None or block.points[0].values["K0"] != 256


# ---------------------------------------------------------------- phase table


class TestPhaseTable:
    def test_cc_flip_to_e_and_ec_fallback_to_c(self):
        space = _space()
        from kernel_optimizer.conditional.probe import ConditionedWall
        wall = ConditionedWall(kind="hard", axis="K0", partner_key="pk",
                               partner_values={"K1": 16, "K2": 16, "K3": 16},
                               point_map={}, f_value=16, n_value=32, refused_value=64)
        # requested C, served C → phase E afterwards.
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=80,
                                tokens=TokenStore())
        block = sc.next_block([wall])
        assert block is not None and block.kind == "C4" and sc.phase == "E"
        # requested E with nothing eligible, served C (fallback) → phase C afterwards.
        sc2 = ConditionalScanner(space, "cand-test", "triton", budget_b=80,
                                 tokens=TokenStore(), initial_phase="E")
        block2 = sc2.next_block([wall])
        assert block2 is not None and block2.kind == "C4" and sc2.phase == "C"

    def test_priority_token_sets_phase_c_explicitly(self):
        space = _space()
        store = TokenStore()
        tok = store.register(_contrast([100.0, 100.0], [80.0, 80.0]))
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=80,
                                tokens=store, initial_phase="E")
        sc.priority_token_id = tok.token_id
        block = sc.next_block([])
        assert block is not None and block.kind == "E1"
        assert sc.phase == "C"                  # explicit C, not a flip of requested


# ---------------------------------------------------------------- C4 role integrity


class TestC4Protocol:
    def test_g_d_and_y_are_role_keyed_not_arrival_keyed(self):
        """Results arrive in randomized execution order; the discovery prediction must
        still be computed from the F_d/N_d ROLES."""
        space = _space()
        from kernel_optimizer.conditional.probe import ConditionedWall
        wall = ConditionedWall(kind="hard", axis="K0", partner_key="pk",
                               partner_values={"K1": 16, "K2": 16, "K3": 16},
                               point_map={}, f_value=16, n_value=32, refused_value=64)
        store = TokenStore()
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=80, tokens=store)
        block = sc.next_block([wall])
        assert block is not None and block.kind == "C4"
        lat = {"F_d": 100.0, "N_d": 80.0, "F_v": 102.0, "N_v": 82.0}
        for p in block.points:                  # arrival = whatever order admission chose
            sc.tell(block, p, lat[p.role], ok=True)
        c = block.contrast
        assert c.g_d == pytest.approx(1.0 - math.exp(math.log(80.0) - math.log(100.0)))
        assert c.y == pytest.approx(1.0 - math.exp(math.log(82.0) - math.log(102.0)))
        assert c.mint is not None and c.mint.direction == "outward"

    def test_incomplete_c4_is_provisional_and_mints_nothing(self):
        space = _space()
        from kernel_optimizer.conditional.probe import ConditionedWall
        wall = ConditionedWall(kind="hard", axis="K0", partner_key="pk",
                               partner_values={"K1": 16, "K2": 16, "K3": 16},
                               point_map={}, f_value=16, n_value=32, refused_value=64)
        sc = ConditionalScanner(space, "cand-test", "triton", budget_b=80,
                                tokens=TokenStore())
        block = sc.next_block([wall])
        lat = {"F_d": 100.0, "N_d": 80.0, "F_v": 102.0, "N_v": 82.0}
        tokens = []
        for i, p in enumerate(block.points):
            ok = i != 2                          # one slot fails
            tokens.append(sc.tell(block, p, lat[p.role] if ok else None, ok=ok))
        assert all(t is None for t in tokens)
        assert block.contrast.full is False
