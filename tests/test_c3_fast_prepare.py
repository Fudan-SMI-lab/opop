import importlib
import json
from pathlib import Path

import pytest

from examples.c3_qwen3.runner_prepare import prepare

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "results/c3-fast-kernel-iteration/setup"


def api():
    assert (ROOT / "examples/c3_qwen3/fast_prepare.py").is_file(), "explicit fast preparation is absent"
    return importlib.import_module("examples.c3_qwen3.fast_prepare")


@pytest.mark.parametrize("phase", ["development", "formal_search"])
def test_phase_prepare_when_only_development_records_allowed(phase: str, monkeypatch) -> None:
    # Given: sealed corpus reads are forbidden during both agent-facing phases.
    module = api()
    original = Path.read_bytes
    def read(path):
        assert path.name != "corpus.json" or path.parent.name != "heldout", "sealed corpus was opened"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    access = module.FastAccess(profile="c3_fast_device", phase=phase)
    # When: explicit opt-in preparation is requested.
    prepared = module.prepare_fast(SETUP / "contract.json", SETUP / "preflight.json", access)
    # Then: no raw/final IDs or recipe enter the agent-safe facet.
    assert len(prepared.task.prompts_by_id) == 16
    assert sum(len(p.input_ids) for p in prepared.task.prompts_by_id.values()) == 21120
    assert all(not g.final_prompt_groups for g in prepared.task.contract.goals)
    facet = json.dumps(prepared.agent_context("single"))
    assert "fast-heldout" not in facet
    assert "input_ids" not in facet
    assert "rendered_text" not in facet
    assert "construction_seed" not in facet


def test_final_prepare_when_formal_completion_missing(monkeypatch) -> None:
    # Given: opening any sealed data would already breach the phase boundary.
    module = api()
    original = Path.read_bytes
    def read(path):
        assert path.parent.name != "heldout"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    # When / Then: final access is denied before opening its corpus.
    with pytest.raises(RuntimeError, match="formal completion"):
        module.prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
                            module.FastAccess(profile="c3_fast_device", phase="final"))


def test_legacy_prepare_when_new_inputs_are_not_opted_in() -> None:
    # Given / When: default legacy preparation still uses the historical contract.
    old = ROOT / "results/c3-qwen3-4b-operator-goals"
    prepared = prepare(old / "contract.json", old / "preflight.json")
    # Then: old58 records remain valid and the new contract is not silently admitted.
    assert len(prepared.prompts_by_id) == 58
    with pytest.raises(RuntimeError, match="contract bytes"):
        prepare(SETUP / "contract.json", SETUP / "preflight.json")


def test_fast_load_when_preparation_is_reentered_without_legacy_pin(tmp_path) -> None:
    # Given: an explicit fast phase and an injected existing numerical backend.
    from tests.c3_tiny_backend import TinyBackend
    module = api()
    prepared = module.prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
                                   module.FastAccess(profile="c3_fast_device", phase="formal_search"))
    loads = []
    def factory(spec, device):
        loads.append(spec.revision)
        return TinyBackend()
    # When: fast load reparses the same phase, never default old preparation.
    resident = module.load_fast(prepared, "cpu", backend_factory=factory)
    # Then: exactly one instance and only permitted16 records.
    assert len(loads) == 1
    assert len(resident.prepared.prompts_by_id) == 16
    resident.close()
