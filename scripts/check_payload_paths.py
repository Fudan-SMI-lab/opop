"""Does every payload path my readers walk actually EXIST in the events on disk?

`check_event_names.py` closed the cheap half of the reader-bug class: an event NAME that src/ never
emits. This closes the expensive half, which is where four of the five real bugs actually were:

    BASELINE_DONE.payload["latency_ms"]        the field is under payload["baseline"]
    TRIAL_DONE.payload["params"]               under payload["trial"], and prints all-None
    S3 per-record precision                    the field is top-level, beside `records`
    CONVERGENCE_DECIDED.payload["verdict"]     under payload["decision"]

Every one is the same shape: several emitters write `{"<key>": obj.model_dump()}`, so the interesting
fields sit ONE LEVEL DOWN, and reading them off the payload returns a clean None rather than raising.
A table of Nones looks like a run that measured nothing.

WHY THIS READS REAL EVENTS RATHER THAN INFERRING A SCHEMA. The wrapper keys could be recovered from
src/ by resolving each `model_dump()` back to its pydantic class, but then the checker's idea of the
shape and the reader's idea would both be derived from my reading of the code -- which is exactly the
trap `a-fixture-invented-to-match-the-reader-proves-nothing` records. An `events.jsonl` written by a
real run is the one description of the shape that no reader's expectations went into.

    python scripts/check_payload_paths.py <run_dir> [<run_dir> ...]

Exit 0 = no reader reads a key off a payload that actually lives one level down. Exit 1 = at least
one does, and that read returns None on every event of that type forever.

WHAT THIS CANNOT DO, stated because a probe that overstates its reach is worse than none:

  * it only knows the events PRESENT in the runs it is given. A reader keyed on an event no supplied
    run contains is not checked, and the output says how many such readers there were rather than
    passing them silently.
  * it only catches the ONE-LEVEL displacement -- a key that exists exactly one level down under a
    wrapper. A misspelled key that exists nowhere is invisible here (nothing to match it against),
    and so is a two-level displacement.
  * it is a TEXTUAL association between an event name and a subscript in the same reader, not a
    dataflow analysis. It can therefore be wrong in the harmless direction (a false hit on a reader
    that handles several events in one function) which is why every hit prints the line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A subscript or .get() on a payload-ish variable. The names are the ones this project's readers
# actually use for the payload dict, which is worth stating: a reader that calls it something else
# is not checked, and that is a limit rather than a pass.
_PAYLOAD_VARS = ("p", "pl", "payload", "pay")
_READ_RE = re.compile(
    r'\b(%s)\s*(?:\.get\(\s*"(\w+)"|\[\s*"(\w+)"\s*\])' % "|".join(_PAYLOAD_VARS))
# Which event a reader is inside. Same event positions as check_event_names.py.
_EVENT_RE = re.compile(r'"([A-Z][A-Z0-9_]{3,})"')
_TYPE_TEST_RE = re.compile(
    r'(?:\.get\(\s*"type"\s*\)|\[\s*"type"\s*\]|\bev\.type|\btyp\b|\bt\b)\s*(?:[!=]=|\s+in\s+)')


def shapes(run_dirs: list[Path]) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]], int]:
    """(top-level payload keys per event, one-level-down keys per event per wrapper, n events read).

    Unioned across every supplied run, because one run need not exercise every optional field.
    """
    top: dict[str, set[str]] = defaultdict(set)
    down: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    n = 0
    for d in run_dirs:
        f = d / "events.jsonl"
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                t = e.get("type")
                p = e.get("payload")
                if not isinstance(t, str) or not isinstance(p, dict):
                    continue
                n += 1
                top[t] |= set(p)
                for k, v in p.items():
                    if isinstance(v, dict):
                        down[t][k] |= set(v)
    return dict(top), {k: dict(v) for k, v in down.items()}, n


def _reader_blocks(text: str) -> list[tuple[int, str, list[tuple[int, str]]]]:
    """[(line_no, event_name, [(line_no, line), ...])] -- each branch keyed on an event type.

    A block runs from the line naming the event until the next line that tests `type` again, so the
    payload reads attributed to an event are the ones lexically inside its own branch.

    A `continue`/`return` guard ENDS the block too. Without that, `if e["type"] not in (A, B):
    continue` -- a guard that keeps ONLY A and B -- attributes the whole rest of the function to A
    and B, and every payload read after it is judged against the wrong event's shape. Measured:
    audit_noise_floor_rejections.py:198 was reported as a TRIAL_DONE defect on exactly that
    mis-attribution, when it sits in a SPACE_REJECTED reader where `candidate_id` IS top-level.
    """
    lines = text.splitlines()
    starts: list[tuple[int, list[str]]] = []
    for i, line in enumerate(lines):
        if _TYPE_TEST_RE.search(line):
            names = _EVENT_RE.findall(line)
            if names:
                starts.append((i, names))
    out = []
    for j, (i, names) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(lines)
        # A negated guard (`not in`, `!=`) followed by continue/return selects what SURVIVES rather
        # than opening a branch, so its block is the rest of the scope -- bounded only by the next
        # type test. A POSITIVE branch, by contrast, ends at its own bare exit.
        #
        # "Bare" is load-bearing: `return p["params"]` is the branch's WORK, not an exit guard, and
        # an earlier version treated any return as the end of the block. Measured against the four
        # real bugs, that truncated three of them away and the checker reported clean -- caught only
        # because the positive control injects known defects instead of trusting a clean run.
        negated = ("not in" in lines[i]) or ("!=" in lines[i])
        if not negated:
            for k in range(i + 1, end):
                if re.match(r'\s*(continue|return|break)\s*$', lines[k]):
                    end = k
                    break
        body = [(k + 1, lines[k]) for k in range(i, end)]
        for name in names:
            out.append((i + 1, name, body))
    return out


def check(paths: list[Path], top: dict[str, set[str]], down: dict[str, dict[str, set[str]]],
          root: Path) -> tuple[list[str], int]:
    hits: list[str] = []
    unknown_events = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for _, event, body in _reader_blocks(text):
            if event not in top:
                unknown_events += 1
                continue
            for line_no, line in body:
                # A COMMENT is not a read. A reader that documents the bug it fixed --
                # "Reading `p["verdict"]` (one level too shallow) printed..." -- would otherwise be
                # reported as still having it, which punishes exactly the code that recorded the
                # lesson. Measured on scripts/prog.py:81, whose comment describes the fixed defect.
                code = line.split("#", 1)[0] if "#" in line else line
                for m in _READ_RE.finditer(code):
                    key = m.group(2) or m.group(3)
                    if key in top[event]:
                        continue
                    # A read that ALREADY falls back to the nested location on the same line --
                    # `pay.get("candidate_id") or t.get("candidate_id")` -- is defensive code, not a
                    # defect: it gets the right answer. Measured: audit_dead_knob_values.py:44 does
                    # exactly this and was reported as a defect by the first version.
                    if re.search(r'\bor\s+\w+\s*(?:\.get\(\s*"%s"|\[\s*"%s"\s*\])'
                                 % (re.escape(key), re.escape(key)), line):
                        continue
                    # Not a top-level key. Does it exist EXACTLY one level down, under a wrapper?
                    for wrapper, sub in down.get(event, {}).items():
                        if key in sub:
                            hits.append(
                                "%s:%d  %s.payload[%r] -- %r is not a payload key; it lives under "
                                "payload[%r]\n        %s" % (
                                    path.relative_to(root).as_posix(), line_no, event, key, key,
                                    wrapper, line.strip()))
                            break
    return hits, unknown_events


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories holding events.jsonl")
    ap.add_argument("--root", default=str(ROOT), help="tree holding scripts/ + tests/")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    top, down, n_events = shapes([Path(r) for r in args.runs])
    wrapped = sum(len(v) for v in down.values())
    # Positive control on the SHAPES, not the event count. A real run has thousands of events but a
    # one-event-per-type fixture is legitimately tiny, and what the check actually needs is enough
    # WRAPPER keys to compare against: with none, every reader below is reported clean and "clean"
    # means "nothing was compared". Counting raw events instead would reject a valid fixture while
    # accepting a thousand events that all happen to be TRIAL_DONE.
    if wrapped < 8 or len(top) < 5:
        print("PROBE BROKEN -- read %d event(s) across %d run(s), yielding %d event type(s) and %d "
              "wrapper key(s). There is not enough shape to check against, so a clean result below "
              "would mean 'nothing was compared', not 'no defects'."
              % (n_events, len(args.runs), len(top), wrapped))
        return 2

    targets = sorted(list((root / "scripts").glob("*.py")) + list((root / "tests").glob("*.py")))
    targets = [p for p in targets
               if p.name != Path(__file__).name and not p.name.startswith("revert_check_")]
    hits, unknown = check(targets, top, down, root)

    print("payload shapes learned from %d event(s) over %d run(s): %d event type(s), %d of them "
          "with a nested wrapper" % (n_events, len(args.runs), len(top), len(down)))
    print("reader branches keyed on an event no supplied run contains: %d (NOT checked -- supply a "
          "run that has them, or they stay unverified)" % unknown)
    # Name the wrappers, because they are the whole risk surface and a reader should not have to
    # guess which events have one.
    for event in sorted(down):
        for wrapper, sub in sorted(down[event].items()):
            if len(sub) >= 3:
                print("    %-28s payload[%r] holds %d field(s), e.g. %s" % (
                    event, wrapper, len(sub), ", ".join(sorted(sub)[:4])))
    if not hits:
        print("\nEVERY payload key a reader reads is a real top-level key of that event.")
        return 0
    print("\n%d payload read(s) go one level too shallow -- each returns None on every event of "
          "that type:" % len(hits))
    for h in hits:
        print("  " + h)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
