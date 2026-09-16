import pytest

from kernel_optimizer.evaluation.wall_attribution import Verdict, Wall, summarize
from kernel_optimizer.reporting.wall_report import wall_lines


@pytest.fixture
def wall() -> Wall:
    return Wall(
        param="BLOCK_M", refused_value=512.0, ran_values=[64.0, 128.0, 256.0],
        side="high", monotone=True, tail_gain_pct=16.7, limit=101376,
    )


def _render(wall: Wall, *, legacy: bool = False) -> list[str]:
    payload = wall.payload()
    if legacy:
        payload = {k: v for k, v in payload.items()
                   if not k.startswith(("origin_", "n_origins_"))}
    return wall_lines([{
        "type": "RESOURCE_WALL_ATTRIBUTED",
        "payload": {
            "candidate_id": "cand-origin", "n_refused_configs": 1,
            "walls_found": 1, "counts": summarize([wall]), "walls": [payload],
        },
    }])


def _row(lines: list[str]) -> list[str]:
    rows = [line for line in lines
            if line.startswith("| `cand-origin` |") and line.count("|") == 12]
    assert len(rows) == 1, "observed origin missing from the attribution table"
    return [cell.strip() for cell in rows[0].split("|")[1:-1]]


@pytest.mark.parametrize("secondary", ["attributed", "not_attributed", "undecidable"])
def test_secondary_is_rendered_when_primary_is_missing(wall: Wall, secondary: Verdict) -> None:
    # Given a recorded secondary probe but no primary answer.
    wall.origin_verdicts = {"theta_top2": secondary}
    wall.origin_max_shared = {"theta_top2": 122880}
    # When the real report section renders the event.
    lines = _render(wall)
    # Then secondary evidence is not discarded or substituted for the primary.
    row = _row(lines)
    assert row[4] == "未测"
    assert row[5].strip("*") == ("1/1" if secondary == "attributed" else "0/1")
    assert row[10] == "未测"
    assert not any(line.startswith("- `cand-origin`") for line in lines)
    if secondary == "attributed":
        assert row[6:9] == ["122880(在 theta_top2)", "101376", "1.21x"]


@pytest.mark.parametrize("primary", [None, "undecidable"])
def test_summary_does_not_claim_primary_fits_when_primary_is_unknown(
    wall: Wall, primary: Verdict | None,
) -> None:
    # Given secondary attribution and no decided primary verdict.
    wall.verdict = primary
    wall.origin_verdicts = {"theta_top2": "attributed"}
    if primary is not None:
        wall.origin_verdicts["theta_star"] = primary
    # When rendering the actual summary.
    lines = _render(wall)
    # Then unknown is not reported as a measured negative at theta*.
    summary = next(line for line in lines if "至少在一个高性能点" in line)
    assert "墙 1 个" in summary
    assert "并不触墙" not in summary
    assert "未确认触墙" in summary
    assert any("ATTRIBUTED 0" in line for line in lines)


@pytest.mark.parametrize("primary", ["attributed", "not_attributed", "undecidable"])
def test_multiple_origins_keep_primary_and_default_separate(wall: Wall, primary: Verdict) -> None:
    # Given a default refusal, a primary answer and two distinct secondary answers.
    wall.verdict = primary
    wall.second_origin = "attributed"
    wall.max_shared = 118784
    wall.origin_verdicts = {
        "default": "attributed", "theta_top2": "not_attributed",
        "theta_top3": "attributed", "theta_star": primary,
    }
    wall.origin_max_shared = {
        "default": 200000, "theta_top2": 65536,
        "theta_top3": 122880, "theta_star": 118784,
    }
    # When rendering the real report.
    lines = _render(wall)
    # Then default is excluded from counts/footprints and primary retains its own verdict.
    row = _row(lines)
    assert row[4].strip("*") == ("ATTRIBUTED" if primary == "attributed" else primary)
    assert row[5] == ("2/3" if primary == "attributed" else "1/3")
    assert row[6] == ("118784" if primary == "attributed" else "122880(在 theta_top3)")
    assert row[10].startswith("attributed(一致)" if primary == "attributed"
                              else "**attributed(不一致")


@pytest.mark.parametrize("verdict", ["attributed", "not_attributed", "undecidable"])
def test_dictionary_primary_and_default_render_without_scalar_mirrors(wall: Wall, verdict: Verdict) -> None:
    # Given dictionary answers recorded before the compatibility mirrors.
    wall.origin_verdicts = {"theta_star": verdict, "default": verdict}
    wall.origin_max_shared = {"theta_star": 122880, "default": 200000}
    # When rendering the report.
    lines = _render(wall)
    # Then the named origins retain their identities and explicit answers.
    row = _row(lines)
    assert row[4].strip("*") == ("ATTRIBUTED" if verdict == "attributed" else verdict)
    assert row[10] == f"{verdict}(一致)"


def test_default_only_observation_is_not_a_high_performance_wall(wall: Wall) -> None:
    # Given only a default-origin refusal, with the primary still unknown.
    wall.origin_verdicts = {"default": "attributed"}
    wall.origin_max_shared = {"default": 200000}
    # When rendering the report.
    lines = _render(wall)
    # Then default evidence is visible but supplies neither a fast-point hit nor a primary comparison.
    row = _row(lines)
    assert row[4:6] == ["未测", "0/0"]
    assert "200000" not in row[6]
    assert row[10] == "attributed"
    assert not any("A3" in line for line in lines)


@pytest.mark.parametrize("legacy", [False, True], ids=["empty-dictionary", "scalar-only"])
@pytest.mark.parametrize("verdict", ["attributed", "not_attributed", "undecidable"])
def test_scalar_fallback_preserves_verdict_and_footprint(wall: Wall, legacy: bool, verdict: Verdict) -> None:
    # Given an old scalar record, or its empty-dictionary equivalent.
    wall.verdict = verdict
    wall.max_shared, wall.over_ratio = 122880, 1.21
    wall.second_origin = "not_attributed"
    # When rendering the report.
    lines = _render(wall, legacy=legacy)
    # Then the scalar evidence is retained, rather than converted to a zero-origin result.
    row = _row(lines)
    assert row[4].strip("*") == ("ATTRIBUTED" if verdict == "attributed" else verdict)
    assert row[5].strip("*") == ("1/1" if verdict == "attributed" else "0/1")
    assert row[6:9] == ["122880", "101376", "1.21x"]
    assert "not_attributed" in row[10]


@pytest.mark.parametrize("legacy", [False, True], ids=["empty-dictionary", "scalar-only"])
def test_unprobed_wall_stays_out_of_verdict_table(wall: Wall, legacy: bool) -> None:
    # Given no recorded origin observations at all.
    wall.monotone, wall.tail_gain_pct = False, -5.0
    # When rendering the report.
    lines = _render(wall, legacy=legacy)
    # Then the old unprobed classification remains, without an invented verdict.
    assert not any(line.startswith("| `cand-origin` |") for line in lines)
    assert any(line.startswith("- `cand-origin` BLOCK_M:") for line in lines)
