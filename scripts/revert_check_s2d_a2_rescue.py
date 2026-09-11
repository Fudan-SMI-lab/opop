# -*- coding: utf-8 -*-
"""Revert-check the A2 rescued-expectations fix. Every variant must be CAUGHT.

Both directions, because this fix touches a path where the model's own validated answer is
absent and so has failure modes on each side:

  * the expectations are lost again -> box 3's outcome, a round with zero S2d evidence and
    nothing saying so;
  * unvalidated content is admitted -> the rescue becomes a way to put junk in the ledger and
    then into the next round's prompt, which the rescue contract forbids.

Anchors are explicit line lists joined with chr(10); never written through a shell heredoc, which
rewrites Python escapes.
"""
from __future__ import annotations
import io, os, shutil, subprocess, sys, tempfile

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)
MOD = os.path.join("src", "kernel_optimizer", "agents", "modules.py")
TESTS = ["tests/test_s2d_a2_rescued_expectations.py", "tests/test_agent_schemas.py"]

VARIANTS = [
    # ---- the expectations go missing again ---------------------------------------------
    ("sidecar_never_read", MOD,
     ['        declared = self._rescue_expectations(sb)'],
     ['        declared = {}'],
     "box 3's exact outcome: files recovered, declarations silently lost, ledger empty"),

    ("matched_on_full_path_so_nothing_is_found", MOD,
     ['            exps = declared.get(f.rsplit("/", 1)[-1]) or []'],
     ['            exps = declared.get(f) or []'],
     "the agent writes 'rw_1.py' and list_outputs returns 'rewrites/rw_1.py', so every lookup "
     "misses and the recovery reads as 'the agent declared nothing'"),

    ("sidecar_keys_not_normalised", MOD,
     ['                out[key.rsplit("/", 1)[-1]] = kept'],
     ['                out[key] = kept'],
     "an agent that wrote a path instead of a name loses its declarations"),

    ("expectations_not_passed_to_the_candidate", MOD,
     ['                expectations=exps,'],
     ['                expectations=[],'],
     "read, validated, and then dropped on the floor"),

    ("prompt_no_longer_asks_for_the_sidecar", MOD,
     ['**Also write the same expectations to `rewrites/expectations.json`**, as'],
     ['Do not write any sidecar file. As'],
     "the reader works but no agent ever writes the file: a positive control that cannot fire"),

    # ---- unvalidated content gets in ---------------------------------------------------
    ("no_pydantic_validation", MOD,
     ['                try:',
      '                    exp = ResourceExpectation.model_validate(item)',
      '                except Exception:  # noqa: BLE001 — skip the entry, keep the rest',
      '                    continue'],
     ['                if True:',
      '                    exp = ResourceExpectation.model_construct(**item)'],
     "a magnitude or an illegal direction reaches the ledger; `extra=forbid` is bypassed and the "
     "agent reasons from a number nobody checked"),

    ("vocabulary_check_removed", MOD,
     ['                if unknown_dimensions([exp.dimension]):',
      '                    continue'],
     ['                if False:',
      '                    continue'],
     "a dimension nothing measures enters the ledger, so it can never be reconciled"),

    ("malformed_json_raises_and_costs_the_round", MOD,
     ['        try:',
      '            raw = json.loads(sb.read_output("rewrites/expectations.json"))',
      '        except Exception:  # noqa: BLE001 — a malformed or absent sidecar must not cost the round',
      '            return {}'],
     ['        if True:',
      '            raw = json.loads(sb.read_output("rewrites/expectations.json"))'],
     "a truncated OR ABSENT sidecar turns a recoverable transport failure into a lost round -- the "
     "exact thing rescue_from_sandbox exists to prevent. This one clause now covers both cases, "
     "since the dead `sb.exists()` pre-check was removed"),

    ("non_dict_payload_accepted", MOD,
     ['        if not isinstance(raw, dict):',
      '            return {}'],
     ['        if False:',
      '            return {}'],
     "a JSON list or string reaches the .items() call and raises inside the rescue"),

    # An `a_missing_sidecar_is_read_anyway` variant was written here and REMOVED. It patched out an
    # `sb.exists()` pre-check and was NOT CAUGHT -- correctly, because the check was dead code: the
    # `except Exception` below already returns {} for the FileNotFoundError that `read_output`
    # raises on an absent file, so removing the guard changes no behaviour. The guard is now gone
    # from the source rather than pinned by a variant that proves nothing. See
    # `a-variant-that-changes-no-behaviour-is-not-a-variant`; the same call was made for
    # `max_families_active < 2` in families.py. `test_the_files_are_still_recovered_when_there_is_
    # no_sidecar` covers the behaviour either way, and `malformed_json_raises_and_costs_the_round`
    # above is what pins the except clause itself.
]


def run(cwd):
    env = dict(os.environ, PYTHONPATH=os.path.join(cwd, "src"))
    p = subprocess.Popen([sys.executable, "-m", "pytest"] + TESTS + ["-q"], cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o, _ = p.communicate()
    return p.returncode, o.decode("utf-8", "replace")


base = tempfile.mkdtemp(prefix="rc-a2-")
work = os.path.join(base, "opop")
shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "runs",
                                                          "runs-*", ".venv", "sandboxes"))
rc, out = run(work)
if rc != 0:
    print("BASELINE FAILS:"); print(out[-3000:]); raise SystemExit(2)
print("baseline: %s" % out.strip().splitlines()[-1])
bad = 0
for name, rel, old_l, new_l, why in VARIANTS:
    path = os.path.join(work, rel)
    orig = io.open(path, encoding="utf-8").read()
    old, new = NL.join(old_l), NL.join(new_l)
    if old not in orig:
        print("  %-48s INVALID -- anchor absent" % name); bad += 1; continue
    patched = orig.replace(old, new, 1)
    if patched == orig:
        print("  %-48s INVALID -- no bytes changed" % name); bad += 1; continue
    io.open(path, "w", encoding="utf-8", newline=NL).write(patched)
    try:
        rc, out = run(work)
    finally:
        io.open(path, "w", encoding="utf-8", newline=NL).write(orig)
    tail = out.strip().splitlines()[-1] if out.strip() else "(none)"
    if rc == 0:
        print("  %-48s **NOT CAUGHT** -- %s" % (name, why)); print("      %s" % tail); bad += 1
    else:
        print("  %-48s CAUGHT  (%s)" % (name, tail))
shutil.rmtree(base, ignore_errors=True)
print()
print("%d not caught or invalid" % bad)
raise SystemExit(1 if bad else 0)
