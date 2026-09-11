"""Candidate families: structural signatures, dedup, novelty gate, active set."""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import tokenize
import uuid

from pydantic import BaseModel

from kernel_optimizer.models.core import (
    BestRecord,
    Candidate,
    Family,
    ParamSet,
    ProfileRecord,
    sha256_text,
)


class NoveltyRejection(BaseModel):
    reason: str
    detail: str


def _normalize_source(source: str) -> str:
    """Strip comments/blank lines for text similarity."""
    out: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        result = []
        last_end = (1, 0)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                continue
            result.append(tok)
        out_src = tokenize.untokenize(result)
        for line in out_src.splitlines():
            if line.strip():
                out.append(line.rstrip())
        return "\n".join(out)
    except (tokenize.TokenError, IndentationError):
        return source


class _SignatureTransformer(ast.NodeTransformer):
    """Zero PARAMS values and drop docstrings so the signature is structural."""

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "PARAMS":
                if isinstance(node.value, ast.Dict):
                    node.value.values = [
                        ast.Constant(value=0) for _ in node.value.values
                    ]
        return node

    def _strip_docstring(self, node):
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_docstring(node)


def structural_signature(source: str, backend: str | None = None) -> str:
    """Hash of the candidate's structure: AST with docstrings dropped and PARAMS zeroed.

    `backend` is part of the structure, not incidental to it. Without it, two candidates
    implementing the same algorithm in Triton and in CUDA C++ hash differently only by
    accident of syntax -- and the reverse case is worse: a candidate whose *idea* is "the same
    approach, expressed in CUDA so the launch path can be hand-written" is a genuinely
    different structure with a different performance ceiling, yet `accept_novel_seed` would
    reject it as a `duplicate_signature` if the AST happened to match, and `register_candidate`
    would silently drop it.

    That matters now the profiler is backend-neutral: CUDA/CUTLASS/CuTe candidates carry real
    resources, so the search can actually use them -- but only if the family machinery treats
    the backend as a structural axis. Optional argument (not required) so the 20 runs of
    recorded signatures stay reproducible when replayed without one.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntaxerror:" + sha256_text(source)[:32]
    tree = _SignatureTransformer().visit(tree)
    digest = hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()
    if backend:
        # Prefixed rather than folded into the hash so a signature stays legible in the event
        # log: "cuda:9f3a..." says at a glance why two candidates are not duplicates.
        return f"{backend}:{digest}"
    return digest


def similarity(a_source: str, b_source: str) -> float:
    return difflib.SequenceMatcher(
        None, _normalize_source(a_source), _normalize_source(b_source)
    ).ratio()


class FamilyManager:
    # CLASS-level defaults, so a manager built with `__new__` (several tests construct one that way
    # to drive `active_families` in isolation) has correct, inert values rather than raising
    # AttributeError from inside the selection method. Two of those tests broke when the reservation
    # was added, and the alternative -- a `getattr` fallback inside `active_families` -- would hide a
    # genuinely mis-wired manager, which is the failure mode recorded on the prescreen deadline.
    #
    # The set default is a FROZENSET, not a set: a shared mutable class attribute would let one
    # manager's marks leak into another's, and `frozenset` makes any accidental `.add` on the class
    # default raise instead of silently succeeding. `__init__` replaces it with a real `set`, and the
    # reservation reads it only when the bool above is True -- which a `__new__`-built manager never
    # has.
    reserve_round_for_reconciled: bool = False
    families_with_a_ledger: frozenset[str] | set[str] = frozenset()

    def __init__(self, max_families_active: int = 2, max_families_total: int = 3,
                 novelty_max_similarity: float = 0.85,
                 max_families_total_hard: int | None = None,
                 reserve_round_for_reconciled: bool = False):
        self.max_families_active = max_families_active
        self.max_families_total = max_families_total
        # Absolute ceiling on families ever created, to bound novelty growth once
        # dead families stop counting against the budget (improvement E).
        self.max_families_total_hard = (max_families_total_hard
                                        if max_families_total_hard is not None
                                        else max_families_total * 2)
        self.novelty_max_similarity = novelty_max_similarity
        # S2d(c): let rule 1 yield ONE slot to a family that already HAS a ledger, so a ledger can
        # actually reach a rewriter prompt. Off by default -- it changes the search order, so it
        # must be a declared arm of an experiment and never a silent default. See `active_families`.
        self.reserve_round_for_reconciled = reserve_round_for_reconciled
        # Families that have a ledger entry. A SET OF IDS AND NOTHING ELSE: this module may know
        # THAT a family has a ledger, never what the ledger says. J2d-8 bans a declared direction
        # from ranking, allocation and acceptance, and it is enforced as a text ban on the selection
        # modules -- so this field is named for what it holds rather than for the event that fills
        # it. That is also the more accurate name: an entry with `n_declared=0` reconciles nothing
        # yet is still a ledger the next prompt can carry.
        #
        # The text ban was NOT relaxed to accommodate this. What protects the invariant is
        # behavioural: `test_selection_cannot_see_what_the_ledger_SAYS` drives this method with
        # hit-only and miss-only ledgers and asserts the slate is identical.
        self.families_with_a_ledger: set[str] = set()
        self.candidates: dict[str, Candidate] = {}
        self.families: dict[str, Family] = {}
        self._sources: dict[str, str] = {}  # candidate_id -> source

    def productive_family_count(self) -> int:
        """Families that still count against the novelty budget: those that are
        active or have produced a correct incumbent. A family that was dropped
        (nothing correct, already frozen) is dead and should NOT consume a slot —
        otherwise a batch of failed seeds permanently blocks novelty exploration
        (the level3:43 failure, improvement E)."""
        return sum(1 for f in self.families.values()
                   if f.status == "active" or f.best is not None)

    # -- registration -----------------------------------------------------------

    def register_candidate(
        self,
        source: str,
        origin: str,
        parent_ids: list[str],
        backend: str,
        approach: str,
    ) -> Candidate | None:
        """Register a seed/rewrite/repair candidate. Returns None on exact-dup."""
        sig = structural_signature(source, backend)
        for existing in self.candidates.values():
            if existing.structural_signature == sig:
                return None  # exact structural duplicate

        if origin == "seed" or not parent_ids:
            family_id = f"fam-{uuid.uuid4().hex[:8]}"
        else:
            family_id = self.candidates[parent_ids[0]].family_id

        candidate = Candidate(
            candidate_id=f"cand-{uuid.uuid4().hex[:8]}",
            family_id=family_id,
            parent_ids=parent_ids,
            origin=origin,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            source_sha=sha256_text(source),
            structural_signature=sig,
            approach_summary=approach,
        )
        self.candidates[candidate.candidate_id] = candidate
        self._sources[candidate.candidate_id] = source
        if family_id not in self.families:
            self.families[family_id] = Family(
                family_id=family_id, anchor_candidate_id=candidate.candidate_id,
                member_ids=[candidate.candidate_id],
            )
        else:
            self.families[family_id].member_ids.append(candidate.candidate_id)
        return candidate

    def accept_novel_seed(self, source: str, backend: str, approach: str,
                          claim: str) -> Candidate | NoveltyRejection:
        """Novelty gate: distinct signature AND low similarity to every anchor."""
        if self.productive_family_count() >= self.max_families_total:
            return NoveltyRejection(
                reason="family_budget",
                detail=f"already {self.productive_family_count()} productive families")
        if len(self.families) >= self.max_families_total_hard:
            return NoveltyRejection(
                reason="family_budget_hard",
                detail=f"hit hard cap of {self.max_families_total_hard} families total")
        # Backend included: a novel seed expressing the same approach in a DIFFERENT backend is
        # a genuinely different structure with a different ceiling, and must not be rejected as
        # a duplicate_signature. This gate is where Loop D would have refused exactly that.
        sig = structural_signature(source, backend)
        for family in self.families.values():
            anchor_src = self._sources.get(family.anchor_candidate_id, "")
            anchor = self.candidates[family.anchor_candidate_id]
            if anchor.structural_signature == sig:
                return NoveltyRejection(reason="duplicate_signature",
                                        detail=f"identical to {family.family_id} anchor")
            sim = similarity(source, anchor_src)
            if sim >= self.novelty_max_similarity:
                return NoveltyRejection(
                    reason="too_similar",
                    detail=f"similarity {sim:.2f} to {family.family_id} anchor "
                           f"(claim: {claim[:100]})",
                )
        candidate = self.register_candidate(source, "novelty", [], backend, approach)
        if candidate is None:
            return NoveltyRejection(reason="duplicate_signature", detail="exact dup")
        return candidate

    # -- queries / updates ---------------------------------------------------------

    def source_of(self, candidate_id: str) -> str:
        return self._sources[candidate_id]

    def family_of(self, candidate_id: str) -> Family:
        return self.families[self.candidates[candidate_id].family_id]

    def active_families(self) -> list[Family]:
        """Which families get a rewrite round now, at most max_families_active.

        Selection is deliberately NOT "the K lowest-latency incumbents". That is the
        early-pruning failure this project exists to avoid: a family's current latency
        reflects how good its *initial parameterization* happens to be, which does not
        predict how much a structural rewrite can still win. Measured on L3:43
        (run-l3-43-20260904-093730): the best-ranked family stalled at [19.6, 19.6,
        19.6] across three rounds while the second-ranked one went [19.5, 17.9, 17.9]
        and produced the run's winner. Ranking on latency alone would have spent the
        budget on the stalled branch; with max_families_active=1 it would have deleted
        the winner outright. In the two round-2 L3 runs, half the families (2 of 4)
        never received a single rewrite round for this reason.

        So:
        1. Every family that has never had a rewrite round goes first. A branch may not
           be dropped before it has been given one chance to show its headroom.
        2. The rest are ordered by IMPROVEMENT SLOPE — how much the last round actually
           gained — not by absolute latency. A family still moving keeps its budget; a
           stalled one yields to a fresher branch even if it currently holds a better
           number.
        3. Absolute latency is only the final tie-break, among families that are
           equally unproven and equally stalled.

        A family with NO correct candidate (`best is None`) is excluded outright, and
        that exclusion is load-bearing rather than cosmetic. Such a family cannot be
        rewritten at all -- `_do_rewrite` needs a correct parent to materialize -- so
        occupying one of the `max_families_active` slots with it costs a real rewrite
        round. Worse, `_rewrite_round` freezes it *without* setting `progressed`, and
        the outer loop reads `not progressed` as "nothing left to do anywhere" and
        freezes every remaining active family. So two empty families filling both slots
        ended a run that still had budget: run-l3-21-20260905-071312 stopped at 2.05h of
        12h with a 15.5 ms incumbent and 4 of 6 rewrite rounds unspent, both empty
        families having been activated for the first time in the same second the run
        finished. The threshold is exactly max_families_active -- L3:48 with ONE empty
        family ran 6.08h and used [3,1,0,3] rounds; L3:21 with TWO used [0,1,0,1].

        Excluding them here is the general statement of the invariant: a family that
        cannot be structurally rewritten does not compete for rewrite budget. Empty
        families are still frozen (in `_rewrite_round`, when reached, and by the
        outer loop's sweep); they simply stop ending other families' search. If EVERY
        family is empty this returns [], `progressed` stays False and the run ends --
        which is the correct outcome, since there is then genuinely nothing to rewrite.

        S2d(c) RESERVATION, off by default. Rule 1 is unconditional, and that is exactly why S2d(c)
        -- "a ledger improves the NEXT rewrite" -- has never been administered: measured
        across every run this project has made, 9 of 9 families in finished runs plus both live arms
        received exactly ONE round each, so the ledger argument was `[]` on every rewriter call ever
        made, in both arms, regardless of the switch. The arithmetic is off by one family every
        time: 4 seed families and 3 rounds fit in 12 h, so the never-rewritten queue never empties.

        With `reserve_round_for_reconciled`, ONE of the `max_families_active` slots may go to a
        family that already has a ledger, provided at least one unproven family still gets a slot.
        Only SET MEMBERSHIP is read -- never a hit, a miss or a direction (J2d-8). That preserves rule 1's protected case -- the recorded L3:43 run where ranking on
        latency would have deleted the eventual winner -- because an unproven family is still
        activated in the same call; it only declines to fill EVERY slot with unproven families.

        Deliberately a switch and not the new default: it changes the order in which families are
        rewritten, so a run with it on is not comparable with the three finished runs or either
        paired arm. It is the arm of an experiment, not a fix.
        """
        active = [f for f in self.families.values()
                  if f.status == "active" and f.best is not None]

        def rank(f: Family) -> tuple[int, float, float]:
            unproven = 0 if f.rewrite_rounds_used == 0 else 1
            return (unproven, -self._improvement_pct(f), self._incumbent(f))

        active.sort(key=rank)
        chosen = active[: self.max_families_active]
        if not self.reserve_round_for_reconciled:
            return chosen

        # A family WITH A LEDGER that rule 1 pushed out, best (lowest) incumbent first among them.
        waiting = [f for f in active[self.max_families_active:]
                   if f.family_id in self.families_with_a_ledger]
        if not waiting:
            return chosen
        # Swap out the LAST chosen unproven family, not the first: the head of the queue is the
        # one rule 1 most wants activated, and there must still be an unproven family in the
        # result or the reservation has become the very rule it is meant to bend.
        unproven_idx = [i for i, f in enumerate(chosen) if f.rewrite_rounds_used == 0]
        if len(unproven_idx) < 2:
            # Only one unproven family in the slate: yielding it would leave zero, which is the
            # early-pruning failure. Better to postpone S2d(c) by a round than to reintroduce it.
            #
            # This single guard also covers `max_families_active == 1`, where `chosen` holds one
            # family and so `unproven_idx` can never reach 2. An explicit `max_families_active < 2`
            # check above was written first and then REMOVED: its revert-variant changed no
            # behaviour, which is the recorded `a-variant-that-changes-no-behaviour-is-not-a-variant`
            # signal that the branch was dead rather than that the test was weak. Keeping an
            # untestable branch is worse than not having it.
            return chosen
        out = unproven_idx[-1]
        return chosen[:out] + chosen[out + 1:] + [waiting[0]]

    @staticmethod
    def _incumbent(f: Family) -> float:
        return f.best.latency_ms if f.best else float("inf")

    @staticmethod
    def _improvement_pct(f: Family) -> float:
        """Percent gained in the most recent completed rewrite round (0 if stalled).

        best_history holds the family's incumbent after each round, so the last step
        is the freshest evidence of remaining headroom.
        """
        hist = f.best_history
        if len(hist) < 2:
            return 0.0
        prev, cur = hist[-2], hist[-1]
        if not prev or prev <= 0 or cur is None:
            return 0.0
        return max(0.0, (prev - cur) / prev * 100.0)

    def update_best(self, family_id: str, candidate_id: str, params: ParamSet,
                    latency_ms: float, profile: ProfileRecord | None = None) -> bool:
        """Record a new family best if it beats the incumbent. Monotonic.

        `profile` is the winning trial's resource profile (G27). It is stored rather than dropped
        because the round-level conversion verdict needs the resource state on BOTH sides of a
        rewrite; without it, `conversion_verdict` received None twice and could only ever report
        latency movement, making `no_conversion` unreachable in production.

        Defaulted so existing callers keep working, but every in-tree caller passes it: a silent
        None here reproduces exactly the defect this parameter exists to fix.
        """
        family = self.families[family_id]
        if family.best is None or latency_ms < family.best.latency_ms:
            family.best = BestRecord(candidate_id=candidate_id, params=params,
                                     latency_ms=latency_ms, profile=profile)
            return True
        return False

    def record_round(self, family_id: str, best_latency_ms: float) -> None:
        self.families[family_id].best_history.append(best_latency_ms)

    def record_round_not_evaluated(self, family_id: str) -> None:
        """A round that spent budget without evaluating any rewrite.

        Deliberately does NOT touch `best_history`. Appending the unchanged incumbent would
        make the history flat, and `family_verdict` reads a flat history as `converged` -- so a
        family whose rewriter simply never answered gets reported as having exhausted its
        structural headroom. Measured on run-l1-42-20260907-193510: fam-50ba7c87 was reported
        `frozen_converged` with history [5.54, 5.54] having never evaluated a single rewrite.
        """
        self.families[family_id].rounds_not_evaluated += 1

    def lineage_tree(self) -> dict:
        return {
            fid: {
                "anchor": f.anchor_candidate_id,
                "status": f.status,
                "best_ms": f.best.latency_ms if f.best else None,
                "history": f.best_history,
                # Distinguish "spent its rewrite budget" from "never got one". Both
                # end as frozen_budget, so without this the report reads as though
                # every branch was explored: in both round-2 L3 runs, 2 of 4 families
                # had rewrite_rounds_used == 0 and never invoked the rewriter at all.
                "rewrite_rounds_used": f.rewrite_rounds_used,
                # How many of those rounds evaluated nothing. A `converged` status is only
                # meaningful when the flat history behind it came from measured rewrites.
                "rounds_not_evaluated": f.rounds_not_evaluated,
                "explored": f.rewrite_rounds_used > 0,
                "members": [
                    {
                        "id": cid,
                        "origin": self.candidates[cid].origin,
                        "parents": self.candidates[cid].parent_ids,
                        "approach": self.candidates[cid].approach_summary,
                    }
                    for cid in f.member_ids
                ],
            }
            for fid, f in self.families.items()
        }
