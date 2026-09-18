"""Fixed two-wave composition, independent heldout, deadlines and complete-denominator gates."""

import json
import subprocess
import sys

import pytest

from kernel_optimizer.store.run_store import RunStore
from tests.c2_opportunity_fakes import campaign, opportunity_module, prepared


@pytest.mark.parametrize("mode", ["expand", "confirm"])
def test_two_waves_use_original_parents_and_fresh_independent_heldout(campaign, mode):
    # Given: both exact P1 IDs, with defaults different from the selected point.
    fixture = campaign
    fixture.mode[0] = mode
    heldout_module = opportunity_module("heldout")
    inputs_module = opportunity_module("inputs")
    opportunities, heldouts = [], []
    expected = [("level3:43", "G0"), ("level3:43", "C2"), ("level3:21", "G0"), ("level3:21", "C2"),
                ("level3:21", "C2"), ("level3:21", "G0"), ("level3:43", "C2"), ("level3:43", "G0")]
    # When: independently invoke each prescribed slot, then the two same-host heldout pairs.
    for wave in (1, 2):
        for slot in ("A0", "A1", "B0", "B1"):
            result = fixture.run(wave, slot)
            opportunities.append(result)
            assert (result.task, result.arm) == expected[len(opportunities) - 1]
            assert result.shared_id == fixture.parents[result.task].identity()
            assert result.sampler_seed == (0 if wave == 1 else 2)
            assert result.information.asked == 80 and result.information.expanded_count == 1
            assert result.information.promotion_policy == "full"
            assert result.incumbent.space.space_id == result.information.child.selected_trial.space_id
            assert result.status == "complete" and result.deadline_unix_s == 15400
            assert result.proposal_source.is_file() and result.mechanism_status == "not_run"
            assert result.provider_cost is None
            envelope = json.loads(result.responses.read_text())
            assert envelope["probe_calls"] == (12 if result.arm == "C2" else 0)
            if result.arm == "G0":
                assert envelope["responses"] == []
            native = fixture.root / f"wave{wave}" / slot / "information/retune"
            manifest = json.loads((native / "manifest.json").read_text())
            assert manifest["inputs"]["sampler_seed"] == (0 if wave == 1 else 2)
            assert manifest["inputs"]["space_expansions_per_candidate"] == 1
        for slot in ("A0", "B0"):
            pair = [r for r in opportunities if r.wave == wave and r.slot.startswith(slot[0])]
            paths = {r.arm: fixture.root / f"wave{wave}" / r.slot / "result.json" for r in pair}
            source_bytes = [p.read_bytes() for p in paths.values()]
            inputs = heldout_module.HeldoutInputs(**fixture.inputs(wave, slot).model_dump(), g0=paths["G0"], c2=paths["C2"])
            output = fixture.root / "heldout" / f"wave{wave}" / slot
            result = heldout_module.run_heldout(inputs, inputs_module.CampaignRun(fixture.cfg, output, 15400,
                now=lambda: fixture.clock[0], clock=lambda: 900000 + fixture.clock[0]))
            heldouts.append(result)
            assert [p.read_bytes() for p in paths.values()] == source_bytes
            phases = [e.payload["phase"] for e in RunStore.open(output).iter_events() if e.type == "LOCAL_EVAL_STARTED"]
            assert phases == [f"heldout_{i}_{arm}" for i, order in enumerate(
                (("parent", "G0", "C2"), ("C2", "G0", "parent"), ("G0", "C2", "parent"))) for arm in order]
        fixture.clock[0] += 100
    # Then: budgets count all spaces, G0 has no fresh guidance, and heldout is never a screen reuse.
    assert len(opportunities) == 8 and sum(r.information.asked for r in opportunities) == 640
    assert len([r for r in fixture.roles if r[0] == "BottleneckReport"]) == 8
    assert len([r for r in fixture.roles if r[0] == "ParameterizationResult" and not r[1]]) == 8
    assert sum(fresh for title, expansion, fresh in fixture.roles if title == "RewriteResult") == 4
    assert len([j for j in fixture.jobs if j[0] == "acquisition"]) == 48
    assert len([j for j in fixture.jobs if j[0] in {"parent", "G0", "C2"}]) == 36
    assert len([j for j in fixture.jobs if j[1] == 100]) == (116 if mode == "confirm" else 84)
    summary_module = opportunity_module("summary")
    summary = summary_module.summarize(opportunities, heldouts)
    assert len(summary.opportunities) == 8 and len(summary.pairs) == 4 and summary.status == "pass"
    assert all(row.g0_ratio == 90 / 101 and row.c2_ratio == 80 / 101 for row in summary.pairs)
    assert summary_module.summarize(opportunities, heldouts[:-1]).status == "inconclusive"
    failed_c2 = [r.model_copy(update={"status": "failed"}) if r.arm == "C2" else r for r in opportunities]
    assert summary_module.summarize(failed_c2, heldouts).status == "fail"
    one_failed_c2 = [r.model_copy(update={"status": "failed"}) if r.wave == 1 and r.slot == "A1" else r for r in opportunities]
    resolved = summary_module.summarize(one_failed_c2, heldouts)
    assert resolved.status == "pass" and resolved.wins == 3 and len(resolved.pairs) == 4
    failed_g0 = [r.model_copy(update={"status": "failed"}) if r.arm == "G0" else r for r in opportunities]
    assert summary_module.summarize(failed_g0, heldouts).status == "pass"
    with pytest.raises(ValueError):
        summary_module.summarize([*opportunities[:-1], opportunities[-1].model_copy(update={"deadline_unix_s": 15401})], heldouts)


@pytest.mark.parametrize("mode", ["flat", "reject", "changed", "invalid"])
def test_negative_opportunities_are_not_redrawn_and_old_space_source_survives(campaign, mode):
    # Given: real native eligibility/no-opportunity, rejection or a worse expanded body.
    campaign.mode[0] = mode
    # When: run exactly one fixed C2 opportunity.
    result = campaign.run(1, "A1")
    # Then: retain truthful outcome/space identity and never regenerate a replacement.
    assert len([r for r in campaign.roles if r[0] == "RewriteResult"]) == 1
    assert result.status == ("failed" if mode == "invalid" else "complete")
    assert result.information.expanded_count == int(mode == "changed")
    if mode == "invalid":
        assert result.incumbent.selected == "parent"
    else:
        assert result.incumbent.space.space_id == result.information.child.spaces[0].space_id
        native = campaign.root / "wave1/A1/information/retune/report/selected.py"
        assert result.incumbent.source.read_bytes() == native.read_bytes()


def test_expired_wave_two_is_written_without_work_or_budget_reset(campaign):
    # Given: the original campaign deadline has already passed before wave2 admission.
    campaign.clock[0] = 15401
    # When: invoke a mandatory late second-wave row.
    result = campaign.run(2, "A0")
    # Then: the same deadline survives and no provider/worker is started.
    assert result.status == "censored" and result.deadline_unix_s == 15400
    assert result.incumbent is None and not campaign.jobs and not campaign.roles
    assert result.drain_s == 0 and (campaign.root / "wave2/A0/result.json").is_file()


def test_exhausted_agent_failure_retains_parent_when_upstream_is_resolved(campaign):
    # Given: an exhausted AgentCallError with a valid three-block parent screen.
    campaign.mode[0] = "service_error"
    # When: the one allowed opportunity reaches that external boundary.
    result = campaign.run(1, "A1")
    # Then: preserve the resolved failed opportunity and its actual parent, without redrawing.
    assert result.information.status == "failed" and result.information.child is None
    assert len(result.information.parent_finals) == 3 and result.information.studies == []
    assert result.status == "failed" and result.incumbent.selected == "parent"
    assert result.incumbent.params == campaign.parents[result.task].parent.params
    assert result.model_calls_started == 1 and result.model_calls_finished == 0
    assert result.provider_cost is None and result.responses.is_file()


@pytest.mark.parametrize("failure", ["invalid", "service_error"])
def test_failed_c2_parent_gets_own_heldout_blocks_and_cannot_win_noise(campaign, failure):
    # Given: a resolved invalid C2 proposal retaining the original parent.
    g0 = campaign.run(1, "A0")
    campaign.mode[0] = failure
    c2 = campaign.run(1, "A1")
    campaign.mode[0] = "expand"
    module, inputs_module = opportunity_module("heldout"), opportunity_module("inputs")
    inputs = module.HeldoutInputs(**campaign.inputs(1, "A0").model_dump(),
        g0=campaign.root / "wave1/A0/result.json", c2=campaign.root / "wave1/A1/result.json")
    # When: all three roles receive independent new measurements.
    result = module.run_heldout(inputs, inputs_module.CampaignRun(campaign.cfg, campaign.root / "heldout", 15400,
        now=lambda: campaign.clock[0], clock=lambda: 900000 + campaign.clock[0]))
    # Then: retained-parent measurements are distinct, and its favorable fake noise cannot pass C2.
    assert result.status == "complete" and c2.incumbent.selected == "parent"
    assert len({t.trial_id for t in result.parent_finals + result.c2_finals}) == 6
    summary = opportunity_module("summary").summarize([g0, c2], [result])
    assert summary.pairs[0].comparison.status == "fail"


def test_upstream_censor_and_partial_native_work_stay_censored(campaign):
    # Given: a real exhausted generation result, with alternate upstream integrity states.
    campaign.mode[0] = "service_error"
    info = campaign.run(1, "A1").information
    from scripts.experiments.c2_retune_studies import RetuneStudy
    classify = opportunity_module().outcome_status
    # When / Then: explicit censoring, incomplete parent evidence and orphaned native work cannot resolve.
    assert classify(info.model_copy(update={"status": "censored"})) == "censored"
    assert classify(info.model_copy(update={"parent_finals": info.parent_finals[:2]})) == "censored"
    partial = RetuneStudy(candidate_id="child", space_id="partial", asked=7)
    assert classify(info.model_copy(update={"studies": [partial]})) == "censored"


def test_retained_parent_export_error_remains_pipeline_censored(campaign, monkeypatch):
    # Given: resolved generation failure followed by unavailable source/helper export.
    campaign.mode[0] = "service_error"
    def unavailable(*args):
        raise OSError("fixture export failure")
    monkeypatch.setattr(opportunity_module(), "export_incumbent", unavailable)
    # When: exporting the known retained parent fails at the filesystem boundary.
    result = campaign.run(1, "A1")
    # Then: unavailable heldout evidence is still censored, not a resolved quality failure.
    assert result.information.status == "failed" and result.status == "censored"
    assert result.incumbent is None


@pytest.mark.parametrize("mode", ["heldout_deadline", "heldout_invalid"])
def test_heldout_partial_or_failed_blocks_remain_inconclusive(campaign, mode):
    # Given: frozen same-host participants from the real consumer.
    g0, c2 = campaign.run(1, "A0"), campaign.run(1, "A1")
    module, inputs_module = opportunity_module("heldout"), opportunity_module("inputs")
    inputs = module.HeldoutInputs(**campaign.inputs(1, "A0").model_dump(),
        g0=campaign.root / "wave1/A0/result.json", c2=campaign.root / "wave1/A1/result.json")
    campaign.mode[0] = mode
    # When: independent full jobs hit a cutoff or correctness failure.
    result = module.run_heldout(inputs, inputs_module.CampaignRun(campaign.cfg, campaign.root / "heldout", 15400,
        now=lambda: campaign.clock[0], clock=lambda: 900000 + campaign.clock[0]))
    # Then: no missing measurement is filled with old parent scores.
    assert result.status == ("censored" if mode == "heldout_deadline" else "failed")
    assert opportunity_module("summary").summarize([g0, c2], [result]).status == "inconclusive"


@pytest.mark.parametrize("change", [{"wave": 3}, {"slot": "X0"}, {"seed": 1}, {"budget": 8}])
def test_input_overrides_are_rejected_before_work(campaign, change):
    # Given / When: user input tries to change the fixed protocol.
    module = opportunity_module("inputs")
    with pytest.raises(ValueError):
        module.OpportunityInputs.model_validate({**campaign.inputs(1, "A0").model_dump(), **change})
    # Then: no external boundary was reached.
    assert not campaign.jobs and not campaign.roles


@pytest.mark.parametrize("change", ["parent", "reference", "evaluation"])
def test_original_identity_and_configuration_mismatch_rejected(campaign, change):
    # Given: an invalid original-parent binding or measurement contract.
    inputs = campaign.inputs(1, "A0")
    if change == "parent":
        data = json.loads(inputs.shared.read_text())
        data["parent"]["candidate_id"] = "not-original-P1"
        inputs.shared.write_text(json.dumps(data))
    elif change == "reference":
        inputs.reference.write_text("def wrong_reference(): return 0\n")
    else:
        campaign.cfg.evaluation.correctness_trials = 3
    # When: the actual runner validates before creating runtime/services.
    with pytest.raises(ValueError):
        campaign.run(1, "A0")
    # Then: no expensive work and no opportunity output has been created.
    assert not campaign.jobs and not campaign.roles and not (campaign.root / "wave1/A0").exists()


def test_real_cli_help_and_invalid_deadline_without_services(campaign):
    # Given: a valid staged wrapper but no valid service config is needed.
    module = opportunity_module()
    wrapper = campaign.root / "input.json"
    wrapper.write_text(campaign.inputs(1, "A0").model_dump_json())
    # When: invoke real module processes for help and invalid temporal input.
    help_run = subprocess.run([sys.executable, "-m", module.__name__, "--help"], capture_output=True, timeout=30)
    bad_runs = []
    for change, deadline in (({}, "nan"), ({}, "inf"), ({"wave": 3}, "15400"), ({"slot": "Z9"}, "15400")):
        wrapper.write_text(json.dumps({**campaign.inputs(1, "A0").model_dump(mode="json"), **change}))
        bad_runs.append(subprocess.run([sys.executable, "-m", module.__name__, "--config", "missing", "--inputs", str(wrapper),
            "--output", str(campaign.root / "denied"), "--deadline-unix-s", deadline], capture_output=True, timeout=30))
    # Then: parser/boundary exits cannot launch services or create a run.
    assert help_run.returncode == 0 and all(r.returncode == 1 and r.stderr.startswith(b"ValidationError:") for r in bad_runs)
    assert not (campaign.root / "denied").exists()
