"""Revert-check for the S7 tests: does each guard fail when its behaviour is removed?

Same discipline as `revert_check_2e.py`, `revert_check_s8.py` and `revert_check_item2.py`. The
behaviours worth guarding here are the ones whose removal produces a PLAUSIBLE run rather than a crash:
a cadence counted on the wrong clock still recomputes; an unbounded target still enqueues something;
a suggestion built from the wrong origin is still a legal configuration. S7 changes the trial sequence,
so a silent defect here does not produce an error -- it produces a 12h arm whose numbers cannot be
attributed to anything.

Run from the v3 worktree root:  python scripts/probes/revert_check_s7.py
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SG = "src/kernel_optimizer/tuning/slope_guide.py"
TPE = "src/kernel_optimizer/tuning/tpe.py"
CFG = "src/kernel_optimizer/config.py"
ORCH = "src/kernel_optimizer/control/orchestrator.py"

VARIANTS = [
    # --- the cadence ---------------------------------------------------------------------------
    ("the cadence fires on every trial", SG,
     "        if self.recompute_every <= 0:\n"
     "            return False\n"
     "        return n_told > 0 and n_told % self.recompute_every == 0",
     "        return n_told > 0",
     "the dose becomes 1 suggestion per trial instead of per 10 -- the mechanism would claim a "
     "large share of a 40-trial budget on a signal that holds 53% of the time"),

    ("a zero cadence divides by zero instead of disabling", SG,
     "        if self.recompute_every <= 0:\n            return False\n",
     "",
     "recompute_every: 0 is the natural way to switch the cadence off from a YAML, and it would "
     "raise ZeroDivisionError mid-tuning instead"),

    ("the cadence is due at zero told trials", SG,
     "        return n_told > 0 and n_told % self.recompute_every == 0",
     "        return n_told % self.recompute_every == 0",
     "the first recompute would run against an empty stats table, burning a recompute to journal "
     "a zero"),

    ("the cadence reads asked instead of told", ORCH,
     "            if guide is not None and guide.due(tuner.n_told):",
     "            if guide is not None and guide.due(tuner._asked):",
     "with constant_liar there are asked-but-untold trials at any moment, so the recompute would "
     "run against a stats table that has not moved"),

    ("told is derived instead of counted", TPE,
     "        self._told += 1",
     "",
     "n_told would stay 0 forever and the recompute would never fire -- a mechanism that is "
     "switched on, journals a snapshot, and does nothing"),

    # --- what gets proposed --------------------------------------------------------------------
    ("the refused value itself can be proposed", SG,
     "            if refused_value is not None:\n"
     "                if side == \"high\" and n >= refused_value:\n"
     "                    continue\n"
     "                if side == \"low\" and n <= refused_value:\n"
     "                    continue\n",
     "",
     "the compiler already refused it: the trial would return infeasible_shared_memory, so the "
     "mechanism would manufacture its own evidence that the wall exists"),

    ("the refusal bound is applied across sides", SG,
     "            if knob in bound and bound[knob][0] == side:",
     "            if knob in bound:",
     "a low-side refusal would cap a high-side proposal, excluding values the compiler never "
     "objected to -- a narrowing of the space, which this mechanism must never do"),

    ("targets are resolved before the dedup", SG,
     "        rows.sort(key=lambda r: r[2], reverse=True)",
     "        rows.sort(key=lambda r: r[2], reverse=False)",
     "the SHALLOWEST wall would get the bounded trial budget -- the allocation decision inverted"),

    # The variant that USED to be here -- "an already-drawn value can be proposed again" -- has been
    # deleted, because that behaviour is now the SHIPPING behaviour. It was a defect, and the probe
    # `s7_why_it_declines.py` measured its cost: requiring the target VALUE to be one no trial had drawn
    # left 1 enqueued point in 224 recomputes (1 of 35 candidates) where dropping the requirement gives 13
    # points and 6 of 35. A treatment arm inserting one trial in 760 could not be told from its control.
    # See `_toward_wall`'s docstring for why the reasoning behind it was wrong on both counts.
    #
    # What replaces it is the variant BELOW, which reinstates the old rule and must be caught: a guard has
    # to exist against the requirement coming back, since it looks eminently reasonable in review.
    ("the value-level dedup is reinstated", SG,
     "        target = nums[0][1]",
     "        target = next((c for _, c in nums if str(c) not in drawn), None)\n"
     "        if target is None:\n"
     "            return None",
     "measured: the value-level requirement cuts the mechanism from 13 enqueued points to 1 over 224 "
     "recomputes, because a hard wall means the range was already swept and the domains hold a median "
     "of 4 choices"),

    ("a value outside the declared choices can be proposed", SG,
     "        for c in domain.choices:\n"
     "            n = wall_attribution._as_num(c)\n"
     "            if n is None:\n"
     "                continue",
     "        for c in list(domain.choices) + [max((wall_attribution._as_num(x) or 0)\n"
     "                                             for x in domain.choices) * 2]:\n"
     "            n = wall_attribution._as_num(c)\n"
     "            if n is None:\n"
     "                continue",
     "Optuna does NOT raise on a queued value outside the distribution -- it warns and samples "
     "that knob normally, silently turning a suggestion into an ordinary draw"),

    ("the origin is any trial instead of the incumbent", SG,
     "            if best is None or ms < best[0]:\n                best = (ms, dict(params))",
     "            if best is None:\n                best = (ms, dict(params))",
     "from the optimum 6 of 6 walls attribute to a single knob, from a slow corner only 1 of 6 -- "
     "the suggestion would probe a point where the walled knob is not the binding one"),

    ("the origin uses the luckiest sample", SG,
     "            ms = _robust_of(t)",
     "            ms = (t.latency_ms.min if getattr(t, \"latency_ms\", None) is not None\n"
     "                  else _robust_of(t))",
     "min reports the luckiest sample (+9.8% to +156% biased at n=20) and would name a different "
     "winner than the objective the rest of the framework selects on"),

    ("robust_ms is looked up by name instead of reproduced", SG,
     "        for key in (\"median\", \"mean\"):",
     "        for key in (\"robust_ms\",):",
     "robust_ms is a @property and never serialized, so every replayed trial would read as "
     "untimed and the guide would silently suggest nothing"),

    ("a failed draw does not count as measured", SG,
     "    for t in trials:\n"
     "        for knob, value in _params_of(t).items():\n"
     "            out.setdefault(str(knob), set()).add(str(value))",
     "    for t in trials:\n"
     "        if _status_of(t) != \"complete\":\n"
     "            continue\n"
     "        for knob, value in _params_of(t).items():\n"
     "            out.setdefault(str(knob), set()).add(str(value))",
     "a configuration the compiler refused has been asked about; re-proposing it spends a trial "
     "to receive the same refusal"),

    ("an incumbent from another space is used anyway", SG,
     "        for d in self.space.domains:\n"
     "            if d.name not in theta:\n"
     "                return None\n"
     "            value = theta[d.name]\n"
     "            if value not in d.choices:\n"
     "                return None\n"
     "            base[d.name] = value",
     "        for d in self.space.domains:\n"
     "            if d.name in theta:\n"
     "                base[d.name] = theta[d.name]",
     "reachable after an expansion re-tune; Optuna raises on neither defect, so the suggestion "
     "would silently stop being 'one knob varied from the optimum'"),

    ("the proposal is not anchored on the incumbent", SG,
     "            if here is not None:\n"
     "                if side == \"high\" and n <= here:\n"
     "                    continue\n"
     "                if side == \"low\" and n >= here:\n"
     "                    continue\n",
     "",
     "'toward the wall' becomes 'any undrawn value on that side of the range': a low-side wall "
     "proposes the knob's TOP value and a high-side wall proposes a value BELOW the optimum, both "
     "logged as a step toward the wall"),

    ("the walls are computed before the incumbent", SG,
     "        walls = self._walls(stats, trials, base)",
     "        walls = self._walls(stats, trials, {})",
     "the anchor would be silently empty for every knob, which is the same defect as removing it "
     "but reached from the call site instead of the predicate"),

    # --- the shared slope filter ---------------------------------------------------------------
    ("the slope filter is re-implemented instead of called", SG,
     "            worth, _ = wall_attribution.select_for_probing(walls, -1)",
     "            worth = [w for w in walls if w.tail_gain_pct > 0.0]",
     "a private copy of the worthiness predicate can drift from the one the report and the prompt "
     "agree on -- and this variant already differs: it drops the monotone requirement"),

    ("the slope filter is dropped entirely", SG,
     "            worth, _ = wall_attribution.select_for_probing(walls, -1)",
     "            worth = list(walls)",
     "a wall whose latency WORSENS toward it is real but worthless; spending a trial toward it is "
     "the sampling analogue of spending a rewrite on a non-problem"),

    ("the probe cap is applied instead of the dose cap", SG,
     "            worth, _ = wall_attribution.select_for_probing(walls, -1)",
     "            worth, _ = wall_attribution.select_for_probing(walls, 0)",
     "0 means 'no walls' to select_for_probing, so the hard criterion would never fire while "
     "still appearing to be consulted"),

    # --- the dose ------------------------------------------------------------------------------
    ("the dose cap is not enforced", SG,
     "            if len(out) >= cap:\n                break",
     "",
     "12 of 34 early walls VANISH by the end of tuning, so an uncapped mechanism could spend a "
     "large share of the budget on knobs the final measurement does not support"),

    ("a zero cap still enqueues", SG,
     "        cap = self.max_enqueued_per_recompute\n        if cap <= 0:\n            return []",
     "        cap = self.max_enqueued_per_recompute or 10**9",
     "max_enqueued_per_recompute: 0 is the natural way to price the cadence alone, and it would "
     "become unbounded instead"),

    ("the same knob is pushed by both criteria", SG,
     "            if knob in seen:\n                continue\n            seen.add(knob)",
     "            seen.add(knob)",
     "two criteria naming one knob is agreement, not two independent reasons to spend two trials"),

    # --- parity: the budget, the guard, the space -----------------------------------------------
    ("the guard is bypassed for an enqueued point", TPE,
     "        if not self.guard_ok(params):\n            return \"guard_rejected\"\n",
     "",
     "S7 would override the candidate's own declared feasibility, and the forced point could then "
     "be measured and counted as the winner"),

    ("an already-drawn point is enqueued silently", TPE,
     "        if key in self._seen:\n            return \"already_drawn\"\n",
     "",
     "it would come back from ask(), be told PRUNED by the dedup branch and consume one of the "
     "bounded re-asks, while the log recorded a successful enqueue and no trial appeared"),

    ("drawn_keys hands out the live set", TPE,
     "        return set(self._seen)",
     "        return self._seen",
     "a caller that mutated it would change what ask() treats as a duplicate, and the extra "
     "PRUNED draws would look like the sampler exhausting the space"),

    # --- the switches ---------------------------------------------------------------------------
    ("S7 defaults ON", CFG,
     "    enabled: bool = False\n\n    # How many FINISHED trials between recomputes.",
     "    enabled: bool = True\n\n    # How many FINISHED trials between recomputes.",
     "the trial sequence would change without being asked, and no finished run would be "
     "comparable"),

    ("the soft criterion defaults ON inside the sampler", CFG,
     "    use_soft_wall: bool = False",
     "    use_soft_wall: bool = True",
     "a spill wall has no independent probe confirmation, so it must not steer the sampler "
     "unasked"),

    ("the soft criterion ignores item 2's switch", ORCH,
     "            use_soft_wall=(cfg.use_soft_wall and self.cfg.v3.soft_wall.enabled),",
     "            use_soft_wall=cfg.use_soft_wall,",
     "an unconfirmed signal would enter the tuning loop while the mechanism that finds it is off, "
     "and in a run with both on the two contributions would be inseparable"),

    ("the guide is built even when the switch is off", ORCH,
     "        cfg = self.cfg.v3.slope_guide\n        if not cfg.enabled:\n            return None",
     "        cfg = self.cfg.v3.slope_guide",
     "off must be the LITERAL old path; a guide that exists is consulted by _tune's due() check"),

    # --- the journal ----------------------------------------------------------------------------
    ("the step is journalled only when something was enqueued", ORCH,
     "            self.store.append(\"SLOPE_GUIDE_STEP\", {",
     "            if not accepted:\n                return\n"
     "            self.store.append(\"SLOPE_GUIDE_STEP\", {",
     "'the guide suggested it and the tuner refused it' is the fact P4 needs, and its absence "
     "reads as the switch being off"),

    ("refusals are not journalled", ORCH,
     "                \"refused\": refused,\n",
     "",
     "'the mechanism fired' and 'the sampler drew the point' are different claims, and only the "
     "log can separate them afterwards"),

    ("the guide can end a candidate", ORCH,
     "        except Exception as exc:  # noqa: BLE001 -- a sampling hint must never end a candidate\n"
     "            self.store.append(\"SLOPE_GUIDE_FAILED\", {",
     "        except ValueError as exc:\n"
     "            self.store.append(\"SLOPE_GUIDE_FAILED\", {",
     "a bookkeeping defect would surface as a candidate defect mid-tuning, with trials already "
     "spent"),

    ("the snapshot is dropped from the no-best branch", ORCH,
     "                \"best_ms\": None, \"snapshot\": tuner.snapshot(), \"deweight\": deweight,\n"
     "                \"ordered_categoricals\": ordered,\n"
     "                \"slope_guide\": slope,",
     "                \"best_ms\": None, \"snapshot\": tuner.snapshot(), \"deweight\": deweight,\n"
     "                \"ordered_categoricals\": ordered,",
     "a space where every trial failed is exactly where the mechanism's effect matters most, and "
     "it would silently drop the record"),

    ("the snapshot loses its skip counters", SG,
     "            \"n_skipped_no_wall\": self.n_skipped_no_wall,\n"
     "            \"n_skipped_no_value_toward_wall\": self.n_skipped_no_value_toward_wall,\n"
     "            \"n_skipped_already_proposed\": self.n_skipped_already_proposed,\n"
     "            \"n_skipped_incomplete_incumbent\": self.n_skipped_incomplete_incumbent,\n",
     "",
     "n_suggested: 0 alone cannot tell 'the signal is not useful' from 'the mechanism never got "
     "to fire' -- which is exactly the distinction P4 rests on"),
]


def run_tests():
    p = subprocess.run(["uv", "run", "--offline", "--extra", "test", "--quiet",
                        "python", "-m", "pytest",
                        "tests/test_s7_slope_guide.py", "tests/test_item2_soft_wall.py",
                        "tests/test_2e_wall_attribution.py", "tests/test_s8_ordered_categoricals.py",
                        "tests/test_tpe.py", "tests/test_reused_trial_artifact.py",
                        "-q", "--no-header", "-x", "-p", "no:warnings"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return p.returncode, p.stdout.decode("utf-8", "replace")


code, out = run_tests()
print(f"BASELINE: {'PASS' if code == 0 else 'FAIL'}  {out.strip().splitlines()[-1]}")
if code != 0:
    print(out[-4000:])
    sys.exit("baseline must pass before reverting anything")

results = []
for label, path, old, new, why in VARIANTS:
    text = io.open(path, encoding="utf-8").read()
    n = text.count(old)
    if n != 1:
        results.append((label, "PATCH-NOT-UNIQUE", f"pattern found {n} times", why))
        print(f"\n[{label}]\n  !! pattern found {n} times -- variant not applied, "
              f"so its 'pass' would be meaningless")
        continue
    backup = tempfile.mktemp(suffix=".bak")
    shutil.copy2(path, backup)
    try:
        io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(old, new, 1))
        code, out = run_tests()
        last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
        verdict = "CAUGHT" if code != 0 else "NOT CAUGHT"
        results.append((label, verdict, last, why))
        print(f"\n[{label}]\n  {verdict}: {last}\n  guards: {why}")
    finally:
        shutil.copy2(backup, path)
        os.unlink(backup)

print("\n" + "=" * 78)
bad = [r for r in results if r[1] != "CAUGHT"]
for label, verdict, last, why in results:
    print(f"  {verdict:18s} {label}")
print(f"\n{len(results) - len(bad)}/{len(results)} variants caught")
if bad:
    print("UNGUARDED BEHAVIOUR -- each of these can be removed with the suite still green:")
    for label, verdict, last, why in bad:
        print(f"  - {label}: {why}")

code, out = run_tests()
print(f"\nRESTORED: {'PASS' if code == 0 else 'FAIL'}  {out.strip().splitlines()[-1]}")
sys.exit(1 if bad or code != 0 else 0)
