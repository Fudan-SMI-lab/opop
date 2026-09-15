"""What did the first rewrites target, and did any of them receive WALL TEXT?

WHY THE DISTINCTION IS THE WHOLE MECHANISM. C2's claim is that a tuning-derived SLOPE, attached to a
resource limit that truncated a knob's range, steers the structural rewrite. Two other things also reach
the rewriter and look similar in the output:

  * the analyst's BottleneckReport -- hypotheses about what limits the kernel, produced from tuning
    statistics WITHOUT any wall attribution;
  * the tuning stats themselves (`at_boundary`, effect sizes).

So a rewrite that says "reduce shared memory" is NOT evidence the mechanism fired. Neither arm has a
single attributed wall in this pair, which means NO rewrite here can have received wall text -- and the
recorded `step3-arm3-beat-arm1-but-2e-did-not-produce-the-winner` case is exactly what happens when that
distinction is skipped: the winning family had never received wall text, so only a mechanism result
could be claimed.

This prints, per rewrite: which family and parent, the change summary, whether any wall text was
delivered for that family, and what the analyst had reported -- so the source of each idea is
attributable rather than assumed.

    python scripts/probes/rewrite_provenance.py <run_dir> [<run_dir> ...]

TAKES THE RUN DIRS AS ARGUMENTS, and that is a fix rather than a style choice: the first version had
`BASE` and `RUN` as module constants pinned to the S7 pair, so pointing it at any other pair silently
re-read S7. The same shape made `gpu_pinning_check.py` report a healthy pair's own workers as a foreign
tenant. A probe whose subject is baked in answers a question nobody asked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# No BASE / RUN constants: they were what pinned this probe to one pair. Deleted rather than left unused
# so nothing can drift back to reading them.


def main() -> int:
    runs = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not runs:
        print("USAGE: rewrite_provenance.py <run_dir> [<run_dir> ...]")
        print("Refusing to default to a hardcoded pair: the previous version pinned the S7 run dirs as")
        print("constants, so every later pair silently got S7's answer.")
        return 2
    for run in runs:
        arm = Path(run).parent.name or Path(run).name
        ev = []
        for line in (Path(run) / "events.jsonl").open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
        print("=" * 78)
        print("%s   (%s)" % (arm, Path(run).name))

        # Which families ever had an ATTRIBUTED wall? That is the only thing that can carry wall text.
        walled_families: dict[str, list[str]] = {}
        fam_of_cand: dict[str, str] = {}
        parents_of_cand: dict[str, list[str]] = {}
        for e in ev:
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            p = e.get("payload") or {}
            c = p.get("candidate") or p
            if c.get("candidate_id"):
                fam_of_cand[str(c["candidate_id"])] = str(c.get("family_id") or "?")
                # THE ONLY PLACE LINEAGE LIVES. `REWRITE_PRODUCED` carries candidate_id,
                # family_id, hypothesis_id, change_summary and expectations -- no parents.
                # Reading parents off the rewrite event yields [] and silently degrades
                # every verdict to family-level matching, which is exactly what this probe
                # must not do: a brief is delivered per (candidate, space), so "a sibling in
                # the family was briefed" is a much weaker claim than "this rewrite's parent
                # was", and the two must not be reported as the same thing.
                parents_of_cand[str(c["candidate_id"])] = [
                    str(x) for x in (c.get("parent_ids") or [])]
        for e in ev:
            if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
                continue
            p = e.get("payload") or {}
            fid = fam_of_cand.get(str(p.get("candidate_id")), "?")
            for w in (p.get("walls") or []):
                if not isinstance(w, dict):
                    continue
                if (w.get("monotone") and (w.get("tail_gain_pct") or 0) > 0
                        and w.get("verdict") == "attributed"):
                    walled_families.setdefault(fid, []).append(str(w.get("param")))
        print("  [v3 path] families with an ATTRIBUTED wall: %s" % (walled_families or "NONE"))

        # THE v4.1 PATH, which the v3 read above cannot see. v4.1 replaced the legacy wall texts with
        # a CONDITIONED BRIEF, journalled as CONDITIONED_BRIEF_DELIVERED per (candidate, space) --
        # there is no RESOURCE_WALL_ATTRIBUTED at all when v4 runs in active mode. Reading only the v3
        # line against a v4.1 run therefore prints "NONE" and would be taken as "no rewrite could have
        # been wall-steered", which is the exact false negative this project keeps paying for: the
        # argument's scope must match the code's scope, and a probe that knows one generation of a
        # mechanism silently reports the other generation's success as a failure.
        #
        # A brief is delivered per SPACE, so it is indexed by candidate: the rewriter for candidate X
        # sees X's brief. `n_chars` is kept because a delivered-but-empty brief would be a different
        # finding from no delivery.
        briefed_cands: dict[str, list[int]] = {}
        for e in ev:
            if e.get("type") != "CONDITIONED_BRIEF_DELIVERED":
                continue
            p = e.get("payload") or {}
            cid = str(p.get("candidate_id"))
            briefed_cands.setdefault(cid, []).append(int(p.get("n_chars") or 0))

        # WHAT THE BRIEF ACTUALLY CONTAINED, per candidate. A brief can carry a real
        # conditioned hard wall and still say "slope UNRESOLVED at this noise level" -- the
        # v4.1 brief is written to disclose that. Counting such a delivery as evidence for
        # "the tuning SLOPE steered the rewrite" would claim the half of C2 the brief itself
        # declines to support, so the two halves are reported separately: a wall was
        # delivered, and/or a slope that passed its gate was.
        #
        # Read from SCAN_BLOCK_DONE (the mint decisions) rather than by grepping the brief
        # text: the direction is a field, and matching prose would break the moment the
        # wording changes.
        minted_axes: dict[str, list[str]] = {}
        unresolved_axes: dict[str, list[str]] = {}
        for e in ev:
            if e.get("type") != "SCAN_BLOCK_DONE" or (e.get("payload") or {}).get("kind") != "C4":
                continue
            p = e["payload"]
            cid, axis = str(p.get("candidate_id")), str(p.get("axis"))
            if p.get("direction"):
                minted_axes.setdefault(cid, []).append("%s/%s" % (axis, p["direction"]))
            elif p.get("full"):
                unresolved_axes.setdefault(cid, []).append(axis)
        briefed_families: dict[str, list[str]] = {}
        for cid, sizes in briefed_cands.items():
            fid = fam_of_cand.get(cid, "?")
            briefed_families.setdefault(fid, []).append(
                "%s(%s)" % (cid[:13], ",".join(str(s) for s in sizes)))
        print("  [v4.1 path] families with a CONDITIONED BRIEF: %s"
              % (briefed_families or "NONE"))
        if briefed_cands and not walled_families:
            print("              (v3 line reads NONE by construction in active mode -- the legacy"
                  " wall texts are REPLACED by the brief, not run alongside it)")

        # Index the analyst's hypotheses by (candidate, id) so a rewrite's `hypothesis_id` can be
        # resolved to the reasoning it followed. This is what makes provenance a MATCH rather than an
        # assertion: the rewrite names H1, and H1's own text says where its resource reasoning came from.
        hyp_by_id: dict[str, dict] = {}
        for e in ev:
            if e.get("type") != "BOTTLENECK_REPORTED":
                continue
            p = e.get("payload") or {}
            rep = p.get("report") or p
            for h in (rep.get("hypotheses") or []):
                if isinstance(h, dict) and h.get("id"):
                    hyp_by_id.setdefault("%s/%s" % (p.get("candidate_id"), h["id"]), h)

        rewrites = [e for e in ev if e.get("type") == "REWRITE_PRODUCED"]
        if not rewrites:
            print("  no REWRITE_PRODUCED yet")
            continue
        for e in rewrites:
            p = e.get("payload") or {}
            print("  --- REWRITE_PRODUCED seq=%s" % e.get("seq"))
            print("      payload keys: %s" % ", ".join(sorted(p.keys())))
            for k in ("family_id", "candidate_id", "parent_ids", "parent_id", "hypothesis_id",
                      "backend", "origin"):
                if k in p:
                    print("      %-14s %s" % (k, json.dumps(p[k], ensure_ascii=False)[:120]))
            for k in ("change_summary", "approach_summary", "difference_claim"):
                if p.get(k):
                    print("      %-14s %s" % (k, str(p[k])[:400]))
            fid = str(p.get("family_id") or fam_of_cand.get(str(p.get("candidate_id")), "?"))
            # Resolve the hypothesis this rewrite claims to implement. The parent candidate is not in
            # the payload, so match on the id across every candidate's report and say when it is
            # ambiguous rather than picking one.
            hid = str(p.get("hypothesis_id") or "")
            matches = {k: v for k, v in hyp_by_id.items() if hid and k.endswith("/" + hid)}
            if matches:
                print("      follows hypothesis %s from %d report(s):" % (hid, len(matches)))
                for k, h in list(matches.items())[:1]:
                    print("        %s" % k)
                    print("        change: %s" % str(h.get("change") or "")[:200])
                    print("        effect: %s" % str(h.get("expected_effect") or "")[:200])
                if len(matches) > 1:
                    print("        (the id appears in %d reports; the payload does not name the parent,"
                          % len(matches))
                    print("         so which one is AMBIGUOUS -- do not attribute it to one candidate)")
            elif hid:
                print("      hypothesis_id %s matches NO analyst hypothesis -- the rewriter invented it"
                      % hid)
            # Two generations of the mechanism, and the verdict must name which one applied.
            # Reading only the v3 condition against a v4.1 active run says "CANNOT have been
            # steered" about a rewrite whose prompt DID carry a conditioned brief -- the most
            # damaging direction this probe can be wrong in, because that sentence is exactly
            # the recorded finding it would be contradicting.
            parent_cids = parents_of_cand.get(str(p.get("candidate_id")), [])
            briefed_parent = [c for c in parent_cids if c in briefed_cands]
            if fid in walled_families:
                print("      [v3] WALL TEXT WAS AVAILABLE for this family (walls: %s) -- check the "
                      "prompt artefact to confirm it was delivered" % ", ".join(walled_families[fid]))
            elif briefed_parent:
                minted = [m for c in briefed_parent for m in minted_axes.get(c, [])]
                unres = [u for c in briefed_parent for u in unresolved_axes.get(c, [])]
                print("      [v4.1] A CONDITIONED BRIEF was delivered for this rewrite's parent (%s)"
                      % ", ".join(c[:13] for c in briefed_parent))
                if minted:
                    print("      parent's GATE-PASSING slopes: %s" % ", ".join(minted))
                    print("      => the STRONGEST attributable case: a wall AND a directed slope were")
                    print("         available. Now check the change_summary names that axis and"
                          " direction;")
                    print("         resource language alone is not evidence (the analyst produces the"
                          " same words).")
                else:
                    print("      parent's slopes: NONE PASSED THE GATE%s"
                          % (" (unresolved on %s)" % ", ".join(sorted(set(unres))) if unres else ""))
                    print("      => a conditioned WALL reached this rewrite, but NOT a usable slope:")
                    print("         the brief itself says 'slope UNRESOLVED at this noise level'."
                          " Report this as")
                    print("         wall-informed, NOT as slope-steered -- claiming the latter asserts"
                          " the half of")
                    print("         C2 the brief declines to support.")
            elif fid in briefed_families:
                print("      [v4.1] a brief reached this FAMILY but not this rewrite's own parent"
                      " (parents: %s)" % (", ".join(parent_cids) or "NONE RECORDED"))
                print("      => attribution is WEAKER than a parent-level match; say so rather than"
                      " claiming the mechanism fired.")
            else:
                print("      NO WALL TEXT AND NO CONDITIONED BRIEF for family %s => this rewrite"
                      % fid)
                print("      CANNOT have been steered by either mechanism. Any resource reasoning in")
                print("      it comes from the analyst's BottleneckReport or the tuning stats. Do not")
                print("      report it as the C2 mechanism firing.")

        # What the analyst said, since that is the alternative source of the same-sounding ideas.
        # Field names taken from a dumped record, not guessed: a hypothesis carries `id`, `change`,
        # `expected_effect` and `risk`. A first version looked for `statement`/`text`/`summary` and
        # printed three empty lines per candidate -- a plausible-looking blank that read as "the analyst
        # said nothing" when in fact it had produced the reasoning the rewrite followed.
        for e in ev:
            if e.get("type") != "BOTTLENECK_REPORTED":
                continue
            p = e.get("payload") or {}
            rep = p.get("report") or p
            hyps = rep.get("hypotheses")
            if not (isinstance(hyps, list) and hyps):
                continue
            print("  analyst hypotheses for %s (the NON-wall source of resource reasoning):"
                  % (p.get("candidate_id") or "?"))
            for h in hyps[:3]:
                if not isinstance(h, dict):
                    continue
                missing = [k for k in ("change", "expected_effect") if not h.get(k)]
                if missing:
                    print("      %s | FIELDS MISSING %s -- keys present: %s"
                          % (str(h.get("id") or "?")[:6], missing, ", ".join(sorted(h.keys()))))
                    continue
                print("      %-4s change: %s" % (str(h.get("id") or "?")[:4],
                                                 str(h["change"])[:150]))
                print("           effect: %s" % str(h["expected_effect"])[:150])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
