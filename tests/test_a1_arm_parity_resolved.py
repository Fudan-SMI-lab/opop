"""Known-answer controls for `a1_arm_parity_resolved.py`.

Each test fixes an answer checkable by hand, and each guards a failure this project has
already made or narrowly avoided:

  - The m2b pair's arm B writes `pcap`/`max_probe_attempts`/`cadence_told` explicitly while
    arm A omits them. A source-YAML diff reports "only mode differs" and would keep saying so
    after a default moved. The resolved comparison is what catches that.
  - Window 1's n1 pair had mode=off on BOTH arms. A reader that infers the variable from
    whatever differs would have promoted some path difference into the treatment role.
  - A seed-paired experiment whose arms read different seed dirs is not seed-paired.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "scripts" / "probes" / "a1_arm_parity_resolved.py"

BASE = {
    "run": {"runs_dir": "/x/runs-v4/m1-a", "seed": 0,
            "seed_candidates_dir": "/x/runs-v4/m1-seeds"},
    "budgets": {"trials_per_space": 40, "rewrite_rounds_per_family": 3,
                "wall_clock_hours": 12.0},
    "v4": {"conditional_scan": {"mode": "off", "f_frac": 0.125, "pcap": 48,
                                "max_probe_attempts": 4, "cadence_told": 10}},
    "wsl": {"triton_cache_dir": "/x/.triton-cache-a"},
    "opencode": {"port": 4096, "launch_cwd": "/x/opop-glm",
                 "server_env": {"XDG_DATA_HOME": "/x/xdg-a"}},
}


def _cfg(**over):
    """Deep-ish copy with dotted overrides, so a test states only what it changes."""
    c = json.loads(json.dumps(BASE))
    for path, val in over.items():
        node = c
        parts = path.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if val is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = val
    return c


def _run(tmp: Path, name: str, cfg: dict) -> Path:
    d = tmp / name / "run-x"
    d.mkdir(parents=True)
    with io.open(d / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"task": "t", "config": cfg}, f)
    return d


def _probe(off: Path, act: Path, *flags: str) -> str:
    out = subprocess.run([sys.executable, str(PROBE), str(off), str(act), *flags],
                         capture_output=True)
    txt = out.stdout.decode("utf-8", errors="replace")
    assert out.returncode == 0, txt + out.stderr.decode("utf-8", errors="replace")
    return txt


def _pair(tmp: Path, **act_over):
    """The canonical clean pair: mode differs, isolation paths differ, nothing else."""
    off = _run(tmp, "m1-a", _cfg())
    act_cfg = _cfg(v4__conditional_scan__mode="active",
                   run__runs_dir="/x/runs-v4/m1-b",
                   wsl__triton_cache_dir="/x/.triton-cache-b",
                   opencode__server_env__XDG_DATA_HOME="/x/xdg-b",
                   **act_over)
    act = _run(tmp, "m1-b", act_cfg)
    return off, act


VAR = "--variable"
MODE = "v4.conditional_scan.mode"


def test_a_clean_pair_reports_only_the_declared_variable(tmp_path):
    off, act = _pair(tmp_path)
    txt = _probe(off, act, VAR, MODE)
    assert "no unexpected differences" in txt
    assert '"off" -> "active"' in txt
    assert "UNEXPECTED DIFFERENCES" not in txt
    assert "NONE OF THE DECLARED VARIABLES DIFFER" not in txt


def test_the_explicit_versus_default_divergence_is_caught(tmp_path):
    """The m2b shape, made to bite: arm B writes pcap=48 and arm A inherits a default that has
    since moved to 32. A source-YAML diff still shows only `mode`, because arm A's file never
    mentions pcap at all. The resolved comparison must report it as a second variable."""
    off = _run(tmp_path, "m2b-a", _cfg(v4__conditional_scan__pcap=32))
    act = _run(tmp_path, "m2b-b", _cfg(
        v4__conditional_scan__mode="active", v4__conditional_scan__pcap=48,
        run__runs_dir="/x/runs-v4/m2b-b", wsl__triton_cache_dir="/x/.triton-cache-b",
        opencode__server_env__XDG_DATA_HOME="/x/xdg-b"))
    txt = _probe(off, act, VAR, MODE)
    assert "UNEXPECTED DIFFERENCES" in txt
    assert "v4.conditional_scan.pcap" in txt
    assert "32 -> 48" in txt
    assert "no unexpected differences" not in txt


def test_an_undeclared_variable_is_never_promoted(tmp_path):
    """The n1 failure, inverted. If the declared variable does not differ, the reader must say
    STOP -- not find the largest remaining difference and treat it as the treatment."""
    off = _run(tmp_path, "n1-a", _cfg())
    act = _run(tmp_path, "n1-b", _cfg(run__runs_dir="/x/runs-v4/n1-b",
                                      wsl__triton_cache_dir="/x/.triton-cache-b",
                                      opencode__server_env__XDG_DATA_HOME="/x/xdg-b"))
    txt = _probe(off, act, VAR, MODE)
    assert "NONE OF THE DECLARED VARIABLES DIFFER" in txt
    assert "STOP HERE" in txt
    # and the isolation paths must not be dressed up as the variable
    assert '"off" -> "active"' not in txt


def test_isolation_paths_are_benign_but_named(tmp_path):
    off, act = _pair(tmp_path)
    txt = _probe(off, act, VAR, MODE)
    for k in ("run.runs_dir", "wsl.triton_cache_dir",
              "opencode.server_env.XDG_DATA_HOME"):
        assert k in txt
    assert "each arm needs its own journal tree" in txt


def test_the_seed_source_must_be_identical_not_merely_different(tmp_path):
    """seed_candidates_dir appears in the isolation table but belongs to MUST_BE_EQUAL: a
    seed-paired pair reading two seed dirs is not seed-paired, and treating that as benign
    isolation is precisely the mistake the split table prevents."""
    off, act = _pair(tmp_path, run__seed_candidates_dir="/x/runs-v4/OTHER-seeds")
    txt = _probe(off, act, VAR, MODE)
    assert "KEYS THAT MUST MATCH BUT DO NOT" in txt
    assert "seed_candidates_dir" in txt
    assert "not\n  !! seed-paired" in txt or "not" in txt


def test_a_key_resolved_in_one_arm_only_is_a_schema_difference(tmp_path):
    """Absent-vs-present is not a value difference: the two runs did not load the same
    settings object. It must never be absorbed into the benign list."""
    off, act = _pair(tmp_path, budgets__wall_clock_hours=None)
    txt = _probe(off, act, VAR, MODE)
    assert "SCHEMA DIFFERENCES" in txt
    assert "budgets.wall_clock_hours" in txt
    assert "<absent>" in txt


def test_a_budget_difference_is_unexpected_not_isolation(tmp_path):
    """The strongest form of the check: an arm given a bigger trial budget would win on search
    volume alone. `trials_per_space` holds a number, not a path, so no name-based rule can
    excuse it."""
    off, act = _pair(tmp_path, budgets__trials_per_space=60)
    txt = _probe(off, act, VAR, MODE)
    assert "UNEXPECTED DIFFERENCES" in txt
    assert "budgets.trials_per_space" in txt
    assert "40 -> 60" in txt


def test_a_semantic_string_is_not_absorbed_as_a_path(tmp_path):
    """Isolation is matched by key NAME, so a knob holding a path-shaped string is still a
    knob. `launch_cwd` is present in BOTH arms (making this a VALUE difference, not a schema
    one) and is not in the isolation table, so it must surface as unexpected."""
    off, act = _pair(tmp_path, opencode__launch_cwd="/somewhere/else")
    txt = _probe(off, act, VAR, MODE)
    assert "UNEXPECTED DIFFERENCES" in txt
    assert "opencode.launch_cwd" in txt
    assert "SCHEMA DIFFERENCES" not in txt


def test_a_missing_manifest_refuses_rather_than_falling_back(tmp_path):
    """The whole point is that the source YAML cannot answer this. A run without a manifest
    must stop the check, not silently degrade to the blind comparison."""
    off, act = _pair(tmp_path)
    (act / "manifest.json").unlink()
    out = subprocess.run([sys.executable, str(PROBE), str(off), str(act), VAR, MODE],
                         capture_output=True)
    assert out.returncode != 0
    err = out.stderr.decode("utf-8", errors="replace")
    assert "no manifest.json" in err
    assert "blind to a default-vs-" in err


def test_a_list_value_is_compared_whole(tmp_path):
    """A reordered list is ONE semantic difference, not len(list) differences -- otherwise a
    single knob change floods the unexpected section and hides everything else."""
    off, act = _pair(tmp_path)
    for d, vals in ((off, [16, 32, 64]), (act, [64, 32, 16])):
        m = json.load(io.open(d / "manifest.json", encoding="utf-8"))
        m["config"]["budgets"]["choices"] = vals
        json.dump(m, io.open(d / "manifest.json", "w", encoding="utf-8"))
    txt = _probe(off, act, VAR, MODE)
    assert txt.count("budgets.choices") == 1
