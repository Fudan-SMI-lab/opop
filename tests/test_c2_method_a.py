"""Selected exploratory A through real loop/information/native adapters."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.experiments import c2_method_program
from scripts.experiments.c2_method_protocol import Deadline
from tests.c2_method_a_fakes import a_fixture, prepared


def a_module():
    assert (Path(__file__).parents[1] / "scripts/experiments/c2_method_a.py").is_file()
    return importlib.import_module("scripts.experiments.c2_method_a")


@pytest.mark.parametrize("slot", ["A0", "A1", "B0", "B1"])
def test_phase_a_cli_runs_two_native_rounds(a_fixture, slot):
    # Given: original P1 and explicitly contaminated core authorization.
    fixture = a_fixture
    inputs = fixture.wrapper(slot)
    config = fixture.root / "config.json"
    config.write_text(fixture.cfg.model_dump_json())
    output = fixture.root / slot
    # When: the actual CLI dispatch composes the existing loop and adapters.
    code = c2_method_program.main(["--phase", "A", "--slot", slot, "--config", str(config),
                                   "--inputs", str(inputs), "--output", str(output)])
    # Then: two B40 opportunities, no extra finals, correct seeds and source bytes.
    assert code == 0, [r["error"] for r in json.loads((output / "result.json").read_text(encoding="utf-8"))["loop"]["rounds"]]
    result = a_module().AResult.model_validate_json((output / "result.json").read_text())
    assert result.status == "valid" and result.source_fidelity == "failed" and not result.clean_core_pass
    assert len(result.loop.rounds) == 2 and [r.sampler_seed for r in result.loop.rounds] == [1, 2]
    assert [r.asked for r in result.loop.rounds] == [40, 40]
    assert len(fixture.roles) == 6 and len([j for j in fixture.jobs if j[2] == 100]) == 12
    assert len([j for j in fixture.jobs if j[1] == "acquisition"]) == (24 if slot in {"A1", "B1"} else 0)
    assert all(fresh == (slot in {"A1", "B1"}) for _, role, fresh in fixture.roles if role != "ParameterizationResult")
    assert fixture.histories == [[], []]
    assert "w0" in result.current.parent.params.values
    for number in (1, 2):
        native = output / "trajectory" / f"round-{number}" / "opportunity/retune"
        manifest = json.loads((native / "manifest.json").read_text())
        assert manifest["inputs"]["sampler_seed"] == number
        assert manifest["inputs"]["evaluation_seed"] == 0
        assert manifest["staged_source"] == "inputs/parameterized.py" and manifest["staged_reference"] == "reference.py"
        assert "old_axis" not in (native / "inputs/parameterized.py").read_text()
        assert (native / "inputs/source.py").is_file()


@pytest.mark.parametrize("mode", ["invalid", "slow", "deadline", "native_failure", "short", "final_metadata"])
def test_a_retains_parent_feedback_or_censors_without_redraw(a_fixture, mode):
    # Given: an actual rejected/slow first child or a deterministic expired scheduler.
    module = a_module()
    fixture = a_fixture
    fixture.mode[0] = mode
    inputs = module.AInputs.model_validate_json(fixture.wrapper("A1").read_text())
    run = module.ARun(fixture.cfg, fixture.root / "A1", Deadline(0, lambda: fixture.clock[0], 9000), 1000)
    # When: execute the same two-round loop.
    result = module.run_a(inputs, run)
    # Then: candidate failures consume rounds and carry only their own attempted feedback.
    assert len(result.loop.rounds) == 2
    if mode in {"deadline", "native_failure", "short", "final_metadata"}:
        assert result.status == "censored" and all(r.status == "censored" for r in result.loop.rounds), [r.error for r in result.loop.rounds]
        assert all(r.elapsed_ready_s is None for r in result.loop.rounds)
        if mode == "short":
            assert result.loop.rounds[0].asked == 4 and len(fixture.roles) == 3
    else:
        assert len(fixture.roles) == 6
        assert len(fixture.histories[1]) == 1 and fixture.histories[1][0]["id"] == "H1"
        assert result.loop.rounds[0].selected == "parent"
        assert "w0" in result.current.parent.params.values
        assert result.loop.rounds[1].elapsed_ready_s >= result.loop.rounds[0].elapsed_ready_s


@pytest.mark.parametrize("change", ["fidelity", "core_results", "parent"])
def test_a_rejects_unacknowledged_or_wrong_parent_before_work(a_fixture, change):
    # Given: an invalid scientific authorization or a nonoriginal parent.
    module = a_module()
    fixture = a_fixture
    path = fixture.wrapper("A0")
    data = json.loads(path.read_text())
    if change == "fidelity":
        data["source_fidelity"] = "passed"
    elif change == "core_results":
        data["core_results"] = []
    else:
        snapshot = Path(data["shared"])
        parent = json.loads(snapshot.read_text())
        parent["source"] += "\n"
        snapshot.write_text(json.dumps(parent))
    # When: parsing and running the proposed branch input.
    with pytest.raises(ValueError):
        inputs = module.AInputs.model_validate(data)
        module.run_a(inputs, module.ARun(fixture.cfg, fixture.root / "denied", Deadline(0, lambda: 0, 9000), 1000))
    # Then: no experiment service or worker was invoked.
    assert not fixture.roles and not fixture.jobs and not (fixture.root / "denied").exists()


def test_real_cli_rejects_clean_fidelity_claim_without_services(a_fixture):
    # Given: a wrapper that incorrectly labels the authorizing core clean.
    fixture = a_fixture
    path = fixture.wrapper("A0")
    data = json.loads(path.read_text())
    data["source_fidelity"] = "passed"
    path.write_text(json.dumps(data))
    # When: execute the actual module in a fresh Python process.
    result = subprocess.run([sys.executable, "-m", "scripts.experiments.c2_method_program", "--phase", "A",
        "--slot", "A0", "--config", str(fixture.root / "absent.json"), "--inputs", str(path),
        "--output", str(fixture.root / "denied")], capture_output=True, text=True, timeout=30, check=False)
    # Then: boundary rejection occurs before config loading or any experiment service.
    assert result.returncode == 1 and result.stderr.partition(":")[0] == "ValidationError"
    assert not (fixture.root / "denied").exists()
