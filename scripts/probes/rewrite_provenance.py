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

Run on box4:
  PYTHONPATH=/root/autodl-tmp/work/opop/src \
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/rewrite_provenance.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"


def main() -> int:
    for arm in ("s7-treatment", "s7-control"):
        ev = []
        for line in (BASE / arm / RUN / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass
        print("=" * 78)
        print(arm)

        # Which families ever had an ATTRIBUTED wall? That is the only thing that can carry wall text.
        walled_families: dict[str, list[str]] = {}
        fam_of_cand: dict[str, str] = {}
        for e in ev:
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            p = e.get("payload") or {}
            c = p.get("candidate") or p
            if c.get("candidate_id"):
                fam_of_cand[str(c["candidate_id"])] = str(c.get("family_id") or "?")
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
        print("  families with an ATTRIBUTED wall: %s" % (walled_families or "NONE"))

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
            if fid in walled_families:
                print("      WALL TEXT WAS AVAILABLE for this family (walls: %s) -- check the prompt "
                      "artefact to confirm it was delivered" % ", ".join(walled_families[fid]))
            else:
                print("      NO ATTRIBUTED WALL for family %s => this rewrite CANNOT have been steered"
                      % fid)
                print("      by wall text. Any resource reasoning in it comes from the analyst's")
                print("      BottleneckReport or the tuning stats, NOT from 2e. Do not report it as")
                print("      the C2 mechanism firing.")

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
