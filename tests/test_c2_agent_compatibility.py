"""T1 characterization, not a claim that future explicit profiles already exist."""

from pathlib import Path

import pytest

from kernel_optimizer.agents.task_rewriter import TaskRewriteResult
from tests.c2_contract_capture import Route, capture_at, cpu_stack, load_anchor, structured_files

pytest_plugins = ["tests.c3_search_fakes"]


@pytest.mark.parametrize("route,anchor", [("generic", "8526a13-generic"), ("legacy", "8526a13-legacy")])
def test_provider_contract_when_current_factory_is_used(tmp_path: Path, route: Route, anchor: str) -> None:
    # Given: independent named baseline, captured before production changes.
    expected = load_anchor(anchor)
    # When: real factory, seeding, helper context and invoke reach the recording provider.
    actual = capture_at(tmp_path / "case", route)
    # Then: exact schema (including titles), machine inputs and call order are preserved.
    assert actual.output_schema == expected.output_schema
    assert structured_files(actual) == structured_files(expected)
    assert actual.model == expected.model and actual.agent == expected.agent
    assert actual.sequence == ("create_session", "prompt")
    assert set(actual.files) == set(expected.files)
    for name in actual.files:
        if name.endswith(".py"):
            assert actual.files[name] == expected.files[name]


def test_semantic_additions_are_detected_when_comparing_named_direct_anchors() -> None:
    # Given: old direct and current generic are deliberately different contracts.
    old = load_anchor("cda1130-direct")
    current = load_anchor("8526a13-generic")
    # When: inspect complete provider schemas and parsed seed data, without deleting fields.
    old_fields = old.output_schema["properties"]
    new_fields = current.output_schema["properties"]
    old_seed = structured_files(old)["analysis/task_response.json"]
    new_seed = structured_files(current)["analysis/task_response.json"]
    # Then: a silent global rollback or normalization would lose this negative witness.
    assert isinstance(old_fields, dict) and isinstance(new_fields, dict)
    assert isinstance(old_seed, dict) and isinstance(new_seed, dict)
    assert set(old_fields) == {"candidate_file", "space"}
    assert set(new_fields) - set(old_fields) == {"bundle_file", "site_groups", "recommended_configs"}
    assert set(new_seed) - set(old_seed) == {"bundle_sources", "bundle_document"}
    assert old.output_schema != current.output_schema and old_seed != new_seed


def test_legacy_machine_contract_when_compared_to_cda_anchor() -> None:
    # Given
    old = load_anchor("cda1130-legacy")
    current = load_anchor("8526a13-legacy")
    # When
    compared = (old.output_schema, structured_files(old), old.sequence)
    # Then
    assert compared == (current.output_schema, structured_files(current), current.sequence)


def test_two_field_response_when_parsed_by_current_generic() -> None:
    # Given
    response = '{"candidate_file":"child.py","space":null}'
    # When
    parsed = TaskRewriteResult.model_validate_json(response)
    # Then: acceptance compatibility does not imply model-facing schema identity.
    assert parsed.candidate_file == "child.py" and parsed.space is None
    assert parsed.bundle_file is None and parsed.site_groups == {} and parsed.recommended_configs == []
