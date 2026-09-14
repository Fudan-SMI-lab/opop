"""v4.1 seed pairing: `run.seed_candidates_dir` replaces the generator with a shared,
frozen candidate set — the step-4 fix (zero shared candidates made the pair unreadable).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel_optimizer.config import AppConfig


SEED_SRC = '''PARAMS = {"BLOCK": 16}

class ModelNew:
    pass
'''


class _Store:
    def __init__(self):
        self.events = []

    def replay(self):
        class S:
            steps_done = set()
            candidates = {}
        return S()

    def append(self, event_type, payload):
        self.events.append((event_type, payload))


def test_seeds_imported_from_dir_in_sorted_order(tmp_path: Path):
    """The mechanism registers files (sorted, capped) and journals SEEDS_IMPORTED —
    exercised through the real _generate_seeds with everything else stubbed."""
    from kernel_optimizer.control.orchestrator import Orchestrator

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "b_second.py").write_text(SEED_SRC, encoding="utf-8")
    (seeds / "a_first.py").write_text(SEED_SRC.replace("16", "32"), encoding="utf-8")

    cfg = AppConfig()
    cfg.run.seed_candidates_dir = seeds
    orch = object.__new__(Orchestrator)          # bypass __init__: only what the method reads
    orch.cfg = cfg
    orch.store = _Store()
    orch.runs = {}
    registered = []

    def _reg(source, origin, parents, backend, summary):
        # Must return the CANDIDATE, not None: `None` is `_register`'s real signal for
        # "refused, structural-signature duplicate", and a stub returning it (which
        # `list.append`'s return value silently did) makes a successful import look like
        # a rejected one.
        registered.append((source, origin, backend, summary))
        cand = object()
        orch.runs[f"c{len(registered)}"] = cand
        return cand

    orch._register = _reg
    orch._step_done = lambda key: None

    orch._generate_seeds()

    assert len(registered) == 2
    assert "32" in registered[0][0]              # a_first.py first (sorted order)
    assert registered[0][1] == "seed"
    imported = [e for e in orch.store.events if e[0] == "SEEDS_IMPORTED"]
    assert len(imported) == 1
    assert imported[0][1]["files"] == ["a_first.py", "b_second.py"]
    assert imported[0][1]["rejected"] == []


def test_a_refused_seed_stops_the_run(tmp_path: Path):
    """A frozen seed set exists so both arms search the SAME candidates. A signature
    collision drops one silently — in BOTH arms, so no config or parity check sees it —
    and the pair then costs 24 GPU-hours to answer a weaker question than it claims.
    The run must refuse to start instead."""
    from kernel_optimizer.control.orchestrator import Orchestrator

    seeds = tmp_path / "seeds"
    seeds.mkdir()
    (seeds / "a.py").write_text(SEED_SRC, encoding="utf-8")
    (seeds / "b_dup.py").write_text(SEED_SRC, encoding="utf-8")

    cfg = AppConfig()
    cfg.run.seed_candidates_dir = seeds
    orch = object.__new__(Orchestrator)
    orch.cfg = cfg
    orch.store = _Store()
    orch.runs = {}
    seen: list[str] = []

    def _reg(source, origin, parents, backend, summary):
        seen.append(summary)
        if len(seen) > 1:                         # the duplicate, as families would judge it
            return None
        cand = object()
        orch.runs["c1"] = cand
        return cand

    orch._register = _reg
    orch._step_done = lambda key: None

    with pytest.raises(RuntimeError, match="refused registration"):
        orch._generate_seeds()
    # The journal is written BEFORE the raise, so the rejection is on disk for the
    # post-mortem rather than only in the traceback.
    imported = [e for e in orch.store.events if e[0] == "SEEDS_IMPORTED"]
    assert len(imported) == 1
    assert imported[0][1]["rejected"] == ["b_dup.py"]


def test_empty_dir_fails_loudly(tmp_path: Path):
    from kernel_optimizer.control.orchestrator import Orchestrator

    seeds = tmp_path / "empty"
    seeds.mkdir()
    cfg = AppConfig()
    cfg.run.seed_candidates_dir = seeds
    orch = object.__new__(Orchestrator)
    orch.cfg = cfg
    orch.store = _Store()
    orch.runs = {}
    orch._step_done = lambda key: None
    with pytest.raises(RuntimeError, match="no .py files"):
        orch._generate_seeds()


def test_default_is_generator_path():
    assert AppConfig().run.seed_candidates_dir is None
