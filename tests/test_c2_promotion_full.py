"""Pure section-5 decisions using real worker record models, without GPU jobs."""

from dataclasses import replace
from typing import Literal, assert_never

import pytest

from kernel_optimizer.models.core import LatencyStats, ParamSet, TrialRecord
from scripts.experiments.c2_promotion_full import (
    ConfirmationPair,
    PromotionDecision,
    PromotionEvidence,
    decide_promotion,
)


def block(value: float) -> TrialRecord:
    return TrialRecord(
        trial_id="full", candidate_id="candidate", space_id="space",
        params=ParamSet(values={}), status="complete",
        latency_ms=LatencyStats(mean=value, median=value, std=0.0,
                                min=value, max=value, n_samples=100),
    )


def evidence(parent: tuple[float, ...], child: tuple[float, ...]) -> PromotionEvidence:
    return PromotionEvidence(
        parent_screen=tuple(block(value) for value in parent),
        child_screen=tuple(block(value) for value in child),
    )


@pytest.mark.parametrize(("parent", "child", "expected"), [
    ((10., 11., 12.), (7., 8., 9.), PromotionDecision("child")),
    ((7., 8., 9.), (10., 11., 12.), PromotionDecision("parent")),
    ((10., 10., 10.), (9.99, 9.99, 9.99), PromotionDecision("child")),
    ((10., 11., 12.), (9., 10., 11.), PromotionDecision("request_pair", 1)),
    ((10., 11., 12.), (8., 9., 10.), PromotionDecision("request_pair", 1)),
    ((10., 10., 10.), (10., 10., 10.), PromotionDecision("request_pair", 1)),
])
def test_screen_decision_when_ranges_vary(
    parent: tuple[float, ...], child: tuple[float, ...], expected: PromotionDecision,
) -> None:
    # Given: exactly three complete full blocks on each arm.
    inputs = evidence(parent, child)
    # When: deciding before any confirmation.
    result = decide_promotion(inputs)
    # Then: overlap/touch confirms, clear wins have no percentage gate.
    assert result == expected


@pytest.mark.parametrize(("quick", "expected"), [
    ((5., 6.), PromotionDecision("request_pair", 1)),
    ((6., 5.), PromotionDecision("child")),
    ((5., 5.), PromotionDecision("child")),
    ((None, 5.), PromotionDecision("child")),
    ((5., None), PromotionDecision("child")),
    ((float("nan"), 5.), PromotionDecision("child")),
    ((5., float("inf")), PromotionDecision("child")),
    ((0., 5.), PromotionDecision("child")),
])
def test_quick_conflict_when_scores_are_strict_or_unknown(
    quick: tuple[float | None, float | None], expected: PromotionDecision,
) -> None:
    # Given: a clear full child win and optional native quick latency scores.
    inputs = replace(evidence((10., 11., 12.), (7., 8., 9.)),
                     parent_quick_ms=quick[0], child_quick_ms=quick[1])
    # When: checking strict opposite direction.
    result = decide_promotion(inputs)
    # Then: ties and unknown scores cannot invent a direction.
    assert result == expected


def test_confirmation_when_quick_favors_child_but_full_favors_parent() -> None:
    # Given: separated full ranges and the opposite quick ranking.
    inputs = replace(evidence((7., 8., 9.), (10., 11., 12.)),
                     parent_quick_ms=6., child_quick_ms=5.)
    # When: deciding from the screens.
    result = decide_promotion(inputs)
    # Then: the opposite conflict direction also requests pair one.
    assert result == PromotionDecision("request_pair", 1)


@pytest.mark.parametrize(("pair", "expected"), [
    ((12., 8.), PromotionDecision("request_pair", 2)),
    ((8., 12.), PromotionDecision("parent")),
    ((10., 10.), PromotionDecision("parent")),
])
def test_first_pair_when_positive_negative_or_tied(
    pair: tuple[float, float], expected: PromotionDecision,
) -> None:
    # Given: tied screens and a first paired block.
    inputs = replace(evidence((9., 10., 11.), (9., 10., 11.)),
                     confirmations=(ConfirmationPair(block(pair[0]), block(pair[1])),))
    # When: deciding after one pair.
    result = decide_promotion(inputs)
    # Then: a first pair can retain parent but never promote child.
    assert result == expected


def test_second_pair_requested_when_negative_first_has_child_cumulative_win() -> None:
    # Given: a negative first pair, but all four blocks still favor the child.
    inputs = replace(evidence((10., 11., 12.), (8., 9., 10.)),
                     confirmations=(ConfirmationPair(block(9.), block(10.)),))
    # When: applying the conjunction for early stopping.
    result = decide_promotion(inputs)
    # Then: the negative pair alone cannot stop confirmation.
    assert result == PromotionDecision("request_pair", 2)


@pytest.mark.parametrize(("pairs", "expected"), [
    (((12., 8.), (12., 8.)), "child"),
    (((12., 8.), (8., 12.)), "parent"),
    (((8., 12.), (12., 8.)), "parent"),
    (((12., 8.), (10., 10.)), "parent"),
])
def test_final_decision_when_pairs_agree_or_contradict(
    pairs: tuple[tuple[float, float], ...], expected: str,
) -> None:
    # Given: overlapping original ranges, unchanged by appending blocks.
    inputs = replace(evidence((9., 10., 11.), (9., 10., 11.)), confirmations=tuple(
        ConfirmationPair(block(parent), block(child)) for parent, child in pairs))
    # When: deciding after both pairs.
    result = decide_promotion(inputs)
    # Then: only two strict child wins plus cumulative improvement promote.
    assert result.outcome == expected
    assert result.next_pair_index is None


@pytest.mark.parametrize(("parent", "child", "expected"), [
    ((1., 2., 3.), (2., 3., 4.), "parent"),
    ((1., 4., 100.), (2., 3., 100.), "child"),
    ((1., 4., 5.), (2., 4., 5.), "parent"),
])
def test_all_screen_blocks_affect_final_cumulative_median(
    parent: tuple[float, ...], child: tuple[float, ...], expected: str,
) -> None:
    # Given: identical strict child-winning pairs but different original screens.
    inputs = replace(evidence(parent, child), confirmations=(
        ConfirmationPair(block(11.), block(10.)),
        ConfirmationPair(block(13.), block(12.)),
    ))
    # When: computing the median over all five values per arm.
    result = decide_promotion(inputs)
    # Then: screens remain decisive, unlike pair-only or best-three selection.
    assert result.outcome == expected


@pytest.mark.parametrize("invalid", [
    None, block(0.), block(-1.), block(float("nan")), block(float("inf")),
    block(1.).model_copy(update={"status": "fail"}),
    block(1.).model_copy(update={"latency_ms": None}),
    block(1.).model_copy(update={"latency_ms": LatencyStats(
        mean=1., median=1., std=0., min=1., max=1., n_samples=99)}),
    block(1.).model_copy(update={"latency_ms": LatencyStats(
        mean=1., std=0., min=1., max=1., n_samples=100)}),
])
@pytest.mark.parametrize("location", ["parent", "child", "pair_parent", "pair_child"])
def test_failed_when_any_required_block_is_invalid(
    invalid: TrialRecord | None, location: Literal["parent", "child", "pair_parent", "pair_child"],
) -> None:
    # Given: invalid or absent screen/confirmation evidence.
    inputs = evidence((10., 11., 12.), (7., 8., 9.))
    match location:
        case "parent":
            inputs = replace(inputs, parent_screen=(invalid, *inputs.parent_screen[1:]))
        case "child":
            inputs = replace(inputs, child_screen=(invalid, *inputs.child_screen[1:]))
        case "pair_parent":
            inputs = replace(inputs, confirmations=(ConfirmationPair(invalid, block(8.)),))
        case "pair_child":
            inputs = replace(inputs, confirmations=(ConfirmationPair(block(12.), invalid),))
        case unreachable:
            assert_never(unreachable)
    # When: attempting a decision.
    result = decide_promotion(inputs)
    # Then: invalid evidence never promotes or requests more measurements.
    assert result == PromotionDecision("failed")


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_failed_when_screen_count_is_not_three(count: int) -> None:
    # Given: an incomplete or oversized screen.
    inputs = evidence((10.,) * count, (8., 8., 8.))
    # When: deciding.
    result = decide_promotion(inputs)
    # Then: screen counts are exact, not a best-three selection.
    assert result == PromotionDecision("failed")


def test_failed_when_more_than_two_pairs_supplied() -> None:
    # Given: evidence exceeding the four-job confirmation budget.
    inputs = replace(evidence((9., 10., 11.), (9., 10., 11.)),
                     confirmations=(ConfirmationPair(block(12.), block(8.)),) * 3)
    # When: deciding.
    result = decide_promotion(inputs)
    # Then: excess evidence is rejected rather than silently truncated.
    assert result == PromotionDecision("failed")
