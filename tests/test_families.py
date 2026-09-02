"""Family manager tests: signatures, dedup, novelty gate, active set."""

from kernel_optimizer.control.families import (
    FamilyManager,
    NoveltyRejection,
    similarity,
    structural_signature,
)
from kernel_optimizer.models.core import ParamSet

KERNEL_A = '''import torch
PARAMS = {"TILE": 64}

class ModelNew(torch.nn.Module):
    """Row-per-block reduction."""
    def forward(self, x):
        return x.sum(dim=1) * PARAMS["TILE"]
'''

# Same structure as A, different PARAMS values + docstring/comment noise.
KERNEL_A2 = '''import torch
# a comment
PARAMS = {"TILE": 128}

class ModelNew(torch.nn.Module):
    """Different docstring, same structure."""
    def forward(self, x):
        return x.sum(dim=1) * PARAMS["TILE"]
'''

KERNEL_B = '''import torch
PARAMS = {"CHUNK": 32, "WARPS": 4}

class Helper:
    def split(self, x, n):
        return torch.chunk(x, n, dim=0)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h = Helper()
    def forward(self, x):
        parts = self.h.split(x, PARAMS["CHUNK"])
        acc = torch.zeros_like(parts[0].sum(dim=1))
        for p in parts:
            acc = acc + p.sum(dim=1)
        return acc * PARAMS["WARPS"]
'''


def test_signature_invariant_to_params_and_docstrings():
    assert structural_signature(KERNEL_A) == structural_signature(KERNEL_A2)


def test_signature_differs_for_different_structure():
    assert structural_signature(KERNEL_A) != structural_signature(KERNEL_B)


def test_similarity_bounds():
    assert similarity(KERNEL_A, KERNEL_A) == 1.0
    assert similarity(KERNEL_A, KERNEL_B) < 0.85


def test_register_and_dedup():
    mgr = FamilyManager()
    c1 = mgr.register_candidate(KERNEL_A, "seed", [], "triton", "sum rows")
    assert c1 is not None
    dup = mgr.register_candidate(KERNEL_A2, "seed", [], "triton", "same again")
    assert dup is None  # structural duplicate


def test_rewrite_inherits_family():
    mgr = FamilyManager()
    c1 = mgr.register_candidate(KERNEL_A, "seed", [], "triton", "sum rows")
    c2 = mgr.register_candidate(KERNEL_B, "rewrite", [c1.candidate_id], "triton", "chunked")
    assert c2.family_id == c1.family_id
    assert mgr.families[c1.family_id].member_ids == [c1.candidate_id, c2.candidate_id]


def test_seed_gets_new_family():
    mgr = FamilyManager()
    c1 = mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    c2 = mgr.register_candidate(KERNEL_B, "seed", [], "triton", "b")
    assert c1.family_id != c2.family_id


def test_novelty_gate_rejects_similar():
    mgr = FamilyManager()
    mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    result = mgr.accept_novel_seed(KERNEL_A2, "triton", "same thing", "totally new!")
    assert isinstance(result, NoveltyRejection)
    assert result.reason == "duplicate_signature"


def test_novelty_gate_accepts_different():
    mgr = FamilyManager()
    mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    result = mgr.accept_novel_seed(KERNEL_B, "triton", "chunked", "different decomposition")
    assert not isinstance(result, NoveltyRejection)
    assert result.origin == "novelty"


def test_novelty_gate_family_budget():
    mgr = FamilyManager(max_families_total=1)
    mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    result = mgr.accept_novel_seed(KERNEL_B, "triton", "b", "new")
    assert isinstance(result, NoveltyRejection)
    assert result.reason == "family_budget"


def test_active_set_capped_and_sorted():
    mgr = FamilyManager(max_families_active=1, max_families_total=3)
    c1 = mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    c2 = mgr.register_candidate(KERNEL_B, "seed", [], "triton", "b")
    mgr.update_best(c1.family_id, c1.candidate_id, ParamSet(values={"TILE": 64}), 5.0)
    mgr.update_best(c2.family_id, c2.candidate_id,
                    ParamSet(values={"CHUNK": 32, "WARPS": 4}), 3.0)
    active = mgr.active_families()
    assert len(active) == 1
    assert active[0].family_id == c2.family_id  # better incumbent first


def test_update_best_only_improves():
    mgr = FamilyManager()
    c1 = mgr.register_candidate(KERNEL_A, "seed", [], "triton", "a")
    p = ParamSet(values={"TILE": 64})
    assert mgr.update_best(c1.family_id, c1.candidate_id, p, 5.0)
    assert not mgr.update_best(c1.family_id, c1.candidate_id, p, 6.0)
    assert mgr.families[c1.family_id].best.latency_ms == 5.0
