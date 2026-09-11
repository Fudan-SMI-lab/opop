"""Is the S2 switch REACHING THE AGENT, and are the two renderings mutually exclusive?

    python scripts/check_arm_prompt_separation.py <run_dir> <label> [vector|label]

Exit 0 = every prompt this run handed an agent carries the arm's own rendering and only that one.
Exit 1 = a document carries both, neither, or the wrong one -- and in every case the run continues
normally, which is why this has to be checked rather than assumed.

MEASURED live on the two L3:43 arms: box 1 5/5 documents carry the LABEL and 0 carry the vector; box 2
8/8 carry the VECTOR and 0 carry the label; `prompt_mode` in the events agrees with the documents on
both. See docs/result-s2-switch-reaches-the-agent.md.

`DIMENSION_STATE.prompt_mode` records what the orchestrator INTENDED. This reads the document the
agent was actually handed -- `analysis/bottleneck.md` in each analyst sandbox -- because the failure
mode being guarded against is silent: a run whose switch never took effect proceeds normally, and
every latency comparison between the arms then measures nothing.

Checked in BOTH directions. A one-directional test ("the treatment has the vector") passes on a
document carrying the vector AND the label, which is the case J2-3 forbids: the treatment arm would
then be more informed for two reasons at once.

WHY THE ANALYST SANDBOX AND NOT THE REWRITER'S. `_dimension_digest` returns prompt text only in
`vector` mode and that text goes to the ANALYST's prompt; the analyst's report then becomes the
rewriter's `analysis/bottleneck.json`. The rewriter receives the treatment indirectly, and its own
sandbox shows 2 of 8 dimension names with no `resource_state` key -- correct behaviour. Reading the
rewriter sandbox instead would report "the vector never arrived" on a working run.

The two markers are the SECTION HEADINGS `_bottleneck_doc` emits, so a rename there must break this
check rather than silently pass it -- which is why they are matched as text rather than inferred.
"""
import json
import os
import sys
from pathlib import Path

# The repo whose emitters the markers must still appear in. Normally this script's own parent, but
# a revert-check copies the script to a temp directory to patch it -- and then `parents[1]` points at
# that temp directory, `src/` is absent, and the marker positive control fires for EVERY variant. That
# made all five variants report "caught by" all six cases, which is a uniform-failure signature rather
# than discrimination. KOPT_REPO_ROOT lets the harness say where the real source is.
ROOT = Path(os.environ.get("KOPT_REPO_ROOT") or Path(__file__).resolve().parents[1])

# The exact section headings the emitters write. VECTOR: evaluation/digest.py:252. LABEL:
# agents/modules.py:241, an f-string whose prefix this is.
VECTOR = "## Resource state, one line per dimension"
LABEL = "## Verdict"
# Where each marker must still be found in src/, so a rename in the emitter breaks this check loudly
# instead of turning it into a test that passes by finding nothing.
_MARKER_SOURCES = {
    VECTOR: "src/kernel_optimizer/evaluation/digest.py",
    LABEL: "src/kernel_optimizer/agents/modules.py",
}


def _markers_still_exist() -> list[str]:
    """Positive control: a marker no emitter writes any more would make every arm read 'NEITHER'."""
    missing = []
    for marker, rel in _MARKER_SOURCES.items():
        p = ROOT / rel
        if not p.exists():
            missing.append("%s (emitter file %s is gone)" % (marker, rel))
            continue
        if marker not in p.read_text(encoding="utf-8", errors="replace"):
            missing.append("%r no longer appears in %s" % (marker, rel))
    return missing

run = Path(sys.argv[1])
label = sys.argv[2]
want = sys.argv[3] if len(sys.argv) > 3 else None   # "vector" | "label" | None

_gone = _markers_still_exist()
if _gone:
    # Without this, a renamed heading makes every document read NEITHER, and "the agent got no
    # resource statement" is indistinguishable from "this script is looking for the wrong string".
    # The recorded rule: a probe that cannot fail loudly is worse than no probe.
    print("PROBE BROKEN -- the marker(s) this check matches on are not in the emitting source any "
          "more, so every result below would be an artefact of a stale string:")
    for g in _gone:
        print("    %s" % g)
    raise SystemExit(2)

# The mode the orchestrator recorded, so an inconsistency between intent and delivery is visible.
modes = {}
ev = run / "events.jsonl"
if ev.exists():
    for line in ev.open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "DIMENSION_STATE":
            m = (e.get("payload") or {}).get("prompt_mode")
            if m:
                modes[m] = modes.get(m, 0) + 1

docs = sorted((run / "sandboxes").glob("analyst-*/analysis/bottleneck.md"))
if not docs:
    print("%-6s NO analyst bottleneck.md on disk -- nothing was handed to an agent yet" % label)
    raise SystemExit(1)

rows = []
for d in docs:
    t = d.read_text(encoding="utf-8", errors="replace")
    rows.append((d.parent.parent.name, VECTOR in t, LABEL in t, len(t.splitlines())))

n_vec = sum(1 for _, v, _, _ in rows if v)
n_lab = sum(1 for _, _, l, _ in rows if l)
n_both = sum(1 for _, v, l, _ in rows if v and l)
n_neither = sum(1 for _, v, l, _ in rows if not v and not l)
print("%-6s prompt_mode in events: %s   analyst docs on disk: %d" % (label, modes or "-", len(rows)))
print("%-6s   carrying the VECTOR: %d   the LABEL: %d   BOTH: %d   NEITHER: %d"
      % (label, n_vec, n_lab, n_both, n_neither))
for name, v, l, n in rows:
    print("%-6s   %-20s vector=%-5s label=%-5s %d lines" % (label, name, v, l, n))

bad = []
if n_both:
    bad.append("%d doc(s) carry BOTH -- the digest must REPLACE the label (J2-3), not sit beside "
               "it, or the arm differs in two ways at once" % n_both)
if n_neither:
    bad.append("%d doc(s) carry NEITHER -- the agent got no resource statement at all" % n_neither)
if want == "vector" and n_vec != len(rows):
    bad.append("expected every doc to carry the vector; %d of %d do" % (n_vec, len(rows)))
if want == "label" and n_lab != len(rows):
    bad.append("expected every doc to carry the label; %d of %d do" % (n_lab, len(rows)))
if want and modes and want not in modes:
    bad.append("events record prompt_mode %s but this arm was asserted as %s" % (list(modes), want))
for b in bad:
    print("%-6s !! %s" % (label, b))
print("%-6s => %s" % (label, "OK" if not bad else "MISMATCH"))
raise SystemExit(1 if bad else 0)
