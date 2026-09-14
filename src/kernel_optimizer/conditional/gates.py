"""v4.1 §5 + §4.1: the print gate, the direction-threaded token mint, and interval algebra.

Pure functions only — no GPU, no I/O, no clock. Everything here is unit-testable against the
six paper fixtures in v4.1 §9.

Definitions (v4.1 §5, inherited v3 §5):
    d = log N − log F          (F = inner/away-from-wall end, N = wall-side end)
    g = 1 − exp(−d)? — NO: g = 1 − exp(d̄) where d̄ = mean(log F) − mean(log N)… — the sign
    conventions bite, so this module fixes ONE canonical form and every consumer uses it:

    c   = mean(log F) − mean(log N)         (c > 0 ⇔ toward-wall improves)
    I   = [1 − max(N)/min(F),  1 − min(N)/max(F)]      envelope interval of the gain toward wall
    ρ_e = (max(e) − min(e)) / geomean(e)    per-end replicate spread (requires ≥2 fresh values)
    τ   = max(4%, ρ_F, ρ_N)
    outward gate:  lower(I) > τ
    I_rev = [−U/(1−U), −L/(1−L)]  where I=[L,U]  (exact algebra for swapping F/N sets)
    inward gate:   lower(I_rev) > τ

Token mint (v4.1 §4.1 — the direction-threading fix that closed the v4 review's blocker 1):
    after fresh-complete + quality, compute J_out = I and J_in = I_rev; ONLY the direction whose
    lower(J) > τ mints the one-shot token, carrying `direction`. The two gates are mutually
    exclusive under positive latencies (out needs max(N) < min(F); in needs max(F) < min(N)),
    so "at most one token per source contrast" holds by construction — asserted anyway, loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Direction = Literal["outward", "inward"]

BASE_TAU = 0.04  # the 4% floor, a global preregistered constant (v4.1 §5)


def _geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rho(end: list[float]) -> float | None:
    """Replicate spread of one end. None (UNKNOWN) unless ≥2 positive fresh values.

    UNKNOWN is not 0 (v4.1 §5): a single measurement has unestimated noise, and the caller
    must not print a certified slope or mint a token from it.
    """
    if len(end) < 2 or any(x <= 0 for x in end):
        return None
    return (max(end) - min(end)) / _geomean(end)


@dataclass(frozen=True)
class GateResult:
    """One direction's gate evaluation. Immutable — the source contrast is frozen at C4
    completion and never redirected after results are seen (v4.1 §4.1 rule 1)."""

    interval: tuple[float, float]          # I for outward, I_rev for inward
    tau: float
    passed: bool
    center: float                          # 1 − exp(−c) resp. reversed; for the brief only


@dataclass(frozen=True)
class MintResult:
    direction: Direction | None            # None = no token
    outward: GateResult
    inward: GateResult
    rho_f: float | None
    rho_n: float | None
    quality_ok: bool
    reason: str                            # human-readable, for the event payload


def interval_toward_wall(f_end: list[float], n_end: list[float]) -> tuple[float, float]:
    """I = [1 − max(N)/min(F), 1 − min(N)/max(F)]. Positive lower bound ⇔ every pairing of
    the observed replicates says toward-wall improved."""
    lo = 1.0 - max(n_end) / min(f_end)
    hi = 1.0 - min(n_end) / max(f_end)
    return (lo, hi)


def reversed_interval(interval: tuple[float, float]) -> tuple[float, float]:
    """I_rev = [−U/(1−U), −L/(1−L)] — exact under F/N set swap; percentages must never be
    naively negated (v4.1 §4.1 step 2). Guards the U→1 pole: caps at 1 − 1e-12."""
    lo, hi = interval
    cap = 1.0 - 1e-12
    lo_c, hi_c = min(lo, cap), min(hi, cap)
    return (-hi_c / (1.0 - hi_c), -lo_c / (1.0 - lo_c))


def evaluate_gates(f_end: list[float], n_end: list[float]) -> MintResult:
    """The full §4.1 mint decision for one fresh-complete source contrast.

    `f_end` / `n_end` are the fresh latency replicates at the inner (F) and wall-side (N)
    geometry ends, in ms, AS FROZEN BEFORE THE FIRST SLOT (never renamed by speed).
    """
    if len(f_end) < 2 or len(n_end) < 2:
        empty = GateResult((0.0, 0.0), BASE_TAU, False, 0.0)
        return MintResult(None, empty, empty, rho(f_end), rho(n_end), False,
                          "quality: fewer than 2 fresh replicates on an end -> rho UNKNOWN, no gate")
    if any(x <= 0 for x in f_end + n_end):
        empty = GateResult((0.0, 0.0), BASE_TAU, False, 0.0)
        return MintResult(None, empty, empty, None, None, False,
                          "quality: non-positive latency replicate")

    rho_f, rho_n = rho(f_end), rho(n_end)
    assert rho_f is not None and rho_n is not None
    tau = max(BASE_TAU, rho_f, rho_n)

    c = (sum(math.log(x) for x in f_end) / len(f_end)
         - sum(math.log(x) for x in n_end) / len(n_end))
    i_out = interval_toward_wall(f_end, n_end)
    i_in = reversed_interval(i_out)

    out_gate = GateResult(i_out, tau, i_out[0] > tau, 1.0 - math.exp(-c))
    in_gate = GateResult(i_in, tau, i_in[0] > tau, 1.0 - math.exp(c))

    # Mutually exclusive by algebra; a violation is a defect in THIS module, so it must be
    # loud (a silent both-pass would mint two tokens from one contrast).
    if out_gate.passed and in_gate.passed:
        raise AssertionError(
            f"both direction gates passed — algebraically impossible: I={i_out}, "
            f"I_rev={i_in}, tau={tau}")

    if out_gate.passed:
        return MintResult("outward", out_gate, in_gate, rho_f, rho_n, True,
                          f"outward gate: lower(I)={i_out[0]:.4f} > tau={tau:.4f}")
    if in_gate.passed:
        return MintResult("inward", out_gate, in_gate, rho_f, rho_n, True,
                          f"inward gate: lower(I_rev)={i_in[0]:.4f} > tau={tau:.4f}")
    return MintResult(None, out_gate, in_gate, rho_f, rho_n, True,
                      f"unresolved: lower(I)={i_out[0]:.4f}, lower(I_rev)={i_in[0]:.4f} "
                      f"<= tau={tau:.4f}")


def ordering_score(f_end: list[float], n_end: list[float], direction: Direction) -> float:
    """The §4.2 direction-aware ordering score — positive for the direction that passed its
    gate: outward uses mean(logF)−mean(logN), inward uses mean(logN)−mean(logF)."""
    c = (sum(math.log(x) for x in f_end) / len(f_end)
         - sum(math.log(x) for x in n_end) / len(n_end))
    return c if direction == "outward" else -c
