"""Does every event name my readers key on actually EXIST in the emitting code?

Five reader bugs of one shape were found in `check_wrapup.py` alone, all of them "a wrong key path
returns a clean zero rather than an error":

    FINAL_REEVAL_DONE                     an event that does not exist
    BASELINE_DONE.payload["latency_ms"]   nested one level under `baseline`
    per-record S3 precision               the field is top-level, beside `records`
    WALL_CLOCK_REACHED payload shape      used to infer which loop, which it cannot
    Reconciliation.verdict/outcome/status  three field names that do not exist -- and the FIXTURE
                                          invented them too, so the suite was green

The last one is why this script exists. A green test suite is not evidence when the fixture was
built from the reader's expectations rather than from the emitter: both agree on keys that are not
there. The only external check is the source of truth -- `store.append("NAME", ...)` in src/ -- so
this compares every literal event name in scripts/ and tests/ against that set.

    python scripts/check_event_names.py

Exit 0 = no reader is keyed SOLELY on a name src/ never emits. Exit 1 = at least one is, which means
that branch never runs and reads zero events of that type forever.

Two distinctions the output keeps, because collapsing either would make the script lie:

  * A dead name ALTERNATED with a real one (`t in ("REWRITE_PRODUCED", "REWRITE_ROUND_DONE")`) still
    counts the real events. Dead weight, not a defect -- reported, exit 0.
  * A dead name that is the SOLE key for its branch makes that branch unreachable. Defect, exit 1.

WHAT THIS CANNOT DO, stated because a probe that overstates its reach is worse than none: it checks
NAMES, not payload PATHS. `BASELINE_DONE` exists and reading `payload["latency_ms"]` still returns
nothing. Nesting has to be pinned by a test holding a verbatim payload, which is what
tests/test_check_wrapup.py does. This closes the cheaper half of the class.

Takes an optional `--root DIR` so the checker can be pointed at a synthetic tree holding a KNOWN
answer. Without that there is no way to verify the checker itself: run against the live repo it
prints whatever it prints and a reader has to take it on faith, which is the same "green suite is
not evidence" trap. scripts/revert_check_event_names.py uses it.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Names that are not orchestrator events: pydantic field names, param knobs, arm labels or other
# upper-case literals that happen to match the shape. Listed explicitly rather than filtered by a
# heuristic, so a genuinely missing event can never be excused as "probably a knob".
#
# MEASURED: with the patterns narrowed to event positions, exactly 0 of these 36 currently fire --
# the tightening did the work the exemption list was written for. Kept anyway, because the list is
# free and the alternative is a reader adding `t == "COMPUTE_DTYPE"`-shaped code and getting a false
# DEFECT; but the count of exclusions that actually fired is now PRINTED, so a future reader can see
# whether this list is load-bearing instead of assuming it.
_NOT_EVENTS = {
    # param knobs, which appear in readers that group trials by config
    "COMPUTE_DTYPE", "DOT_MODE", "NUM_WARPS", "NUM_STAGES", "GROUP_M", "GEMM_GROUP_M",
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "GEMM_BM", "GEMM_BN", "GEMM_BK", "QKV_BLOCK_N",
    "PREC", "DTYPE", "SPLIT",
    # arm / report labels
    "CONTROL", "TREATMENT", "PASS", "FAIL", "PARTIAL", "NOEVID", "UNVERIF", "DEFECT",
    "BORROWED", "ALL", "SHARED", "OPTIN", "SHARED_OPTIN",
    # our own section headings and verdict words
    "RANGE", "POOLED", "CONSTANT", "CRASH", "NOT", "YES", "NO",
}

# Only literals in an EVENT POSITION count. A first version matched every upper-case string in the
# file and drowned the signal: knob names, signal names, verdict words, env vars, 60+ false hits, so
# a real missing event would never have been noticed. These patterns are the places a reader actually
# names an event type -- comparison against the `type` field, membership in a set of types, or a
# `store.append` in a test double.
_USE_PATTERNS = (
    # e.get("type") == "X" / e["type"] == "X" / ev.type == "X"   (and != and reversed operands)
    re.compile(r'(?:\.get\(\s*"type"\s*\)|\[\s*"type"\s*\]|\bev\.type|\btyp\b|\bt\b)'
               r'\s*[!=]=\s*"([A-Z][A-Z0-9_]{3,})"'),
    re.compile(r'"([A-Z][A-Z0-9_]{3,})"\s*[!=]=\s*(?:\.get\(\s*"type"\s*\)|\bev\.type)'),
    # in ("X", "Y") / in {"X", "Y"} following a type expression, plus the type-set literal form
    re.compile(r'(?:\.get\(\s*"type"\s*\)|\bev\.type|\btyp\b|\bt\b)'
               r'\s+(?:not\s+)?in\s*[\(\{\[]([^)\}\]]*)[\)\}\]]'),
    # {"type": "X", ...} -- how a test builds an event
    re.compile(r'"type"\s*:\s*"([A-Z][A-Z0-9_]{3,})"'),
    # store.append("X", ...) in a test double or a script that writes events
    re.compile(r'store\.append\(\s*"([A-Z][A-Z0-9_]+)"'),
    # grep -c "X" in a shell string inside a monitor command
    re.compile(r'grep\s+-c\s+"?([A-Z][A-Z0-9_]{5,})"?'),
)

_INNER_RE = re.compile(r'"([A-Z][A-Z0-9_]{3,})"')
_NAME_RE = re.compile(r'^[A-Z][A-Z0-9_]{3,}$')
_EMIT_RE = re.compile(r'store\.append\(\s*"([A-Z][A-Z0-9_]+)"')
# Some events are appended via a variable or an f-string; catch the `type=` / `ev.type ==` forms too
# so a real event name is not reported missing because of how it is written.
_TYPE_CMP_RE = re.compile(r'(?:ev\.type|e\.get\("type"\))\s*==\s*"([A-Z][A-Z0-9_]+)"')


def emitted_names(src: Path) -> set[str]:
    names: set[str] = set()
    for p in sorted(src.rglob("*.py")):
        text = p.read_text(encoding="utf-8", errors="replace")
        names |= set(_EMIT_RE.findall(text))
        names |= set(_TYPE_CMP_RE.findall(text))
    return names


def _names_in(hit: str) -> list[str]:
    """Event names inside one regex capture.

    The `in (...)` pattern captures the whole tuple body, so it needs splitting back out -- and a
    body with no quoted name yields NOTHING, because that same pattern also matches numeric tuples
    like `for t in (600.0, 1500.0, ...)` in tests/test_g20_idle_abort.py. Reporting those as missing
    events was noise that buried the real hits.
    """
    if '"' in hit:
        return _INNER_RE.findall(hit)
    return [hit] if _NAME_RE.match(hit) else []


def used_names(paths: list[Path]) -> tuple[dict[str, set[Path]], int]:
    out: dict[str, set[Path]] = {}
    excluded = 0
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in _USE_PATTERNS:
            for hit in pat.findall(text):
                for name in _names_in(hit):
                    if name in _NOT_EVENTS:
                        excluded += 1
                        continue
                    out.setdefault(name, set()).add(p)
    return out, excluded


def _is_guarded(path: Path, name: str, emitted: set[str]) -> bool:
    """Is every use of `name` in `path` an alternation that also names a real event?

    Cheap and deliberately conservative: it looks at the LINE, and a line naming another emitted
    event beside this one is treated as an alternation. A false "guarded" would hide a defect, so
    the other name must be on the same line rather than merely nearby -- and a single unguarded
    use is enough to call the whole file unguarded.
    """
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"%s"' % name not in line:
            continue
        others = {n for n in _INNER_RE.findall(line) if n != name}
        if not (others & emitted):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT),
                    help="tree holding src/ + scripts/ + tests/ (default: this repo)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    emitted = emitted_names(root / "src")
    if len(emitted) < 15:
        # Positive control: this project has dozens of event types, so a tiny set means the
        # extraction itself broke and every "missing" below would be an artefact. A probe that
        # cannot fail loudly is the recorded failure mode this guards against.
        print("PROBE BROKEN -- only %d event names found in %s, expected dozens. Every result "
              "below would be an artefact of a failed extraction, not a finding."
              % (len(emitted), (root / "src")))
        print("found: %s" % sorted(emitted))
        return 2

    targets = sorted(list((root / "scripts").glob("*.py")) + list((root / "tests").glob("*.py")))
    targets = [p for p in targets if p.name != Path(__file__).name]
    used, excluded = used_names(targets)

    print("event names emitted by src/: %d" % len(emitted))
    print("upper-case literals used by scripts/ + tests/: %d (%d use(s) dropped by the %d-name "
          "non-event exemption list)" % (len(used), excluded, len(_NOT_EVENTS)))

    missing = {n: fs for n, fs in used.items() if n not in emitted}
    # A revert-check names the WRONG event on purpose: that is the documented bug it injects, and
    # flagging it would train the reader to ignore this script's output. Exempt only that use --
    # the same name in a real reader is still reported.
    for name in list(missing):
        real = {p for p in missing[name] if not p.name.startswith("revert_check_")}
        if real:
            missing[name] = real
        else:
            del missing[name]
    if not missing:
        print("\nEVERY event name used by a reader is emitted somewhere in src/.")
        print("Reminder: this checks NAMES, not payload PATHS. `BASELINE_DONE` exists and reading")
        print("`payload[\"latency_ms\"]` still returns nothing -- nesting needs a verbatim-payload")
        print("test, which is what tests/test_check_wrapup.py holds.")
        return 0

    print("\n%d name(s) used by a reader are NOT emitted anywhere in src/:" % len(missing))
    hard = 0
    for name in sorted(missing):
        where = ", ".join(sorted(p.relative_to(root).as_posix() for p in sorted(missing[name])))
        near = difflib.get_close_matches(name, sorted(emitted), n=2, cutoff=0.6)
        hint = ("  did you mean %s?" % " / ".join(near)) if near else ""
        guarded = all(_is_guarded(p, name, emitted) for p in sorted(missing[name]))
        tag = "  [harmless: alternated with a real name]" if guarded else "  [DEFECT: sole key]"
        if not guarded:
            hard += 1
        print("  %-34s used in %s%s%s" % (name, where, hint, tag))
    if not hard:
        print("\nAll of the above are alternations that also name a real event, so no branch is "
              "unreachable. Dead weight, not a defect.")
        return 0
    print("\n%d of them is/are the SOLE key for a branch, so that branch never runs: a reader keyed "
          "on a non-existent event counts zero of them forever, and zero is usually the value a "
          "criterion is checked against." % hard)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
