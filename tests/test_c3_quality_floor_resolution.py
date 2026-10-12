"""`evaluate_quality` resolves the floor for the row it is judging, or refuses to guess.

WHY A STUBBED RUNNER AND NOT JUST `criteria()`
---------------------------------------------
`test_c3_quality_criteria_from_contract.py` tests the resolution function in isolation. This drives
`evaluate_quality` itself, because two of the three ways this can go wrong live in the CALL, not in the
function it calls, and I got both wrong on the first attempt:

  1. THE BASELINE ROW. It replaces nothing, so `receipt.site_ids` is empty. Passing `site_id=""` and
     `module_count=0` into the evidence check made it raise "different contract/site/module count" --
     which would have failed all nine baseline cells of the published matrix the moment floor evidence
     existed, and the baseline is the column every gain is computed against.
  2. A MULTI-SITE BUNDLE. Taking `site_ids[0]` and summing modules across every group would apply a
     floor measured for one amount of amplification to a candidate that has a different amount. The
     evidence would not describe what is being judged, so the only honest response is to refuse.

The floor never reaches `_verdict` unless the identity matches, so these tests stop before the real
oracle: what matters is which criteria were resolved and whether the call refused, both of which are
decided before any logits are read. A deliberately unreadable oracle path ends the evaluation right
after that decision.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.c3_qwen3.runner_quality import FLOOR_METRIC, evaluate_quality
from examples.c3_qwen3.runner_records import Contract

FLOOR_73 = 0.010348571942351894
CONTRACT_SHA = "c" * 64
SITE = "pure-727f184a"
MODULES = tuple(f"model.layers.{i}.input_layernorm" for i in range(73))


def contract() -> Contract:
    return Contract.model_validate({
        "interface_version": "c3-fast-inputs-v1.0.0",
        "assets": {"repo_id": "r", "revision": "v", "local_path_both_hosts": "/p",
                   "corpus_file": "c.json", "corpus_sha256": "h"},
        "goals": [], "runtime_versions": {},
        "shared_policy": {"quality": {
            "local_operator_rtol": 0.02, "local_operator_atol": 0.02,
            "logits_relative_l2_max_per_prompt": 0.01,
            "logits_relative_l2_denominator_floor": 1e-12,
            "paired_mean_nll_delta_max_nat_per_token": 0.02}}})


def runner(tmp_path: Path, site_ids: tuple[str, ...], sites: dict[str, tuple[str, ...]]):
    """The smallest surface `evaluate_quality` touches before it reads the oracle."""
    receipts: list[dict] = []

    def raw(kind: str, data: dict) -> Path:
        receipts.append({"kind": kind, **data})
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
        return path

    binding = SimpleNamespace(
        sites=sites, verify_installed=lambda: None, restore=lambda: None,
        validate_local=lambda: None, coverage=lambda: {}, trace=[])
    backend = SimpleNamespace(weights_stamp=lambda: "s", forward_calls=0, reset=lambda: None)
    return SimpleNamespace(
        prepared=SimpleNamespace(contract=contract(), contract_sha256=CONTRACT_SHA, corpus_sha256="p"),
        receipt=SimpleNamespace(site_ids=site_ids), binding=binding, backend=backend,
        close=lambda: None, _raw=raw, receipts=receipts)


def floor_file(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "equivalence-floor.json"
    path.write_text(json.dumps({"contract_sha256": CONTRACT_SHA, "site_id": SITE,
                                "module_count": 73, "logits_relative_l2_max": FLOOR_73,
                                **overrides}), encoding="utf-8")
    return path


def limits(receipt: dict) -> dict[str, float]:
    return {c["metric"]: c["limit"] for c in receipt["criteria"]}


def evaluate(tmp_path: Path, site_ids, sites, floor: Path | None):
    """Run the real function; the unreadable oracle stops it just after the floor decision."""
    stub = runner(tmp_path, site_ids, sites)
    report = evaluate_quality(stub, ("calibration-calibration-00",), tmp_path / "absent-oracle.json", floor)
    return report, stub.receipts[-1]


def test_the_baseline_row_needs_no_floor_and_is_not_broken_by_one(tmp_path) -> None:
    # Given: the baseline, which binds no site at all, and floor evidence for the 73-module group.
    report, receipt = evaluate(tmp_path, (), {}, floor_file(tmp_path))
    # Then: the evaluation reaches the oracle (and fails there, on the missing file) rather than dying
    # on the floor identity check -- and it judged against the contract's declared limits.
    assert receipt["equivalence_floor"] is None
    assert limits(receipt)[FLOOR_METRIC] == 0.01
    assert report.detail is not None and "different contract/site" not in report.detail


def test_a_matching_single_site_row_gets_the_measured_floor(tmp_path) -> None:
    # Given: the row actually under test -- one site group of exactly the 73 modules the evidence names.
    report, receipt = evaluate(tmp_path, (SITE,), {SITE: MODULES}, floor_file(tmp_path))
    # Then: the floor is applied and recorded, so the receipt states which limit the verdict used.
    assert receipt["equivalence_floor"] == FLOOR_73
    assert limits(receipt)[FLOOR_METRIC] == FLOOR_73
    assert receipt["criteria_source"] == "contract"
    assert limits(receipt)["paired_mean_nll_delta_nat_per_token"] == 0.02
    assert report.detail is not None  # the absent oracle, not the floor


def test_a_module_count_that_differs_from_the_evidence_is_refused(tmp_path) -> None:
    # Given: the same site id, but the binding covers 36 modules rather than the 73 measured. The
    # amplification differs, so this floor does not describe this row.
    report, _ = evaluate(tmp_path, (SITE,), {SITE: MODULES[:36]}, floor_file(tmp_path))
    assert report.detail is not None and "different contract/site/module count" in report.detail
    assert not report.valid


def test_a_multi_site_bundle_with_floor_evidence_is_refused(tmp_path) -> None:
    # Given: two active site groups and a floor measured for one. Using the first group's identity
    # would apply a floor for the wrong amount of amplification.
    other = "pure-otherside"
    report, _ = evaluate(tmp_path, (SITE, other), {SITE: MODULES, other: ("model.norm",)},
                         floor_file(tmp_path))
    assert report.detail is not None and "covers one site group" in report.detail
    assert not report.valid


def test_a_multi_site_bundle_without_floor_evidence_is_fine(tmp_path) -> None:
    # The refusal above is about the EVIDENCE, not about multi-site bundles: with no floor declared
    # there is nothing to misapply, so the declared limits apply and the run proceeds.
    other = "pure-otherside"
    report, receipt = evaluate(tmp_path, (SITE, other), {SITE: MODULES, other: ("model.norm",)}, None)
    assert receipt["equivalence_floor"] is None
    assert limits(receipt)[FLOOR_METRIC] == 0.01
    assert report.detail is not None and "covers one site group" not in report.detail


def test_the_receipt_names_the_criteria_source_without_a_policy(tmp_path) -> None:
    # A contract that declares no policy must be visible as such in the receipt, so a reader can tell
    # a module default from a contract-declared limit.
    stub = runner(tmp_path, (SITE,), {SITE: MODULES})
    stub.prepared = SimpleNamespace(
        contract=Contract.model_validate({
            "interface_version": "c3-task-local-v1.0.0",
            "assets": {"repo_id": "r", "revision": "v", "local_path_both_hosts": "/p",
                       "corpus_file": "c.json", "corpus_sha256": "h"},
            "goals": [], "runtime_versions": {}}),
        contract_sha256=CONTRACT_SHA, corpus_sha256="p")
    evaluate_quality(stub, ("calibration-calibration-00",), tmp_path / "absent-oracle.json", None)
    assert stub.receipts[-1]["criteria_source"] == "module_default"
    assert limits(stub.receipts[-1])[FLOOR_METRIC] == 0.01
