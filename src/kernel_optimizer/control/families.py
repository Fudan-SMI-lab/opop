"""Candidate families: structural signatures, dedup, novelty gate, active set."""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import tokenize
import uuid

from pydantic import BaseModel

from kernel_optimizer.models.core import BestRecord, Candidate, Family, ParamSet, sha256_text


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


def structural_signature(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntaxerror:" + sha256_text(source)[:32]
    tree = _SignatureTransformer().visit(tree)
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def similarity(a_source: str, b_source: str) -> float:
    return difflib.SequenceMatcher(
        None, _normalize_source(a_source), _normalize_source(b_source)
    ).ratio()


class FamilyManager:
    def __init__(self, max_families_active: int = 2, max_families_total: int = 3,
                 novelty_max_similarity: float = 0.85,
                 max_families_total_hard: int | None = None):
        self.max_families_active = max_families_active
        self.max_families_total = max_families_total
        # Absolute ceiling on families ever created, to bound novelty growth once
        # dead families stop counting against the budget (improvement E).
        self.max_families_total_hard = (max_families_total_hard
                                        if max_families_total_hard is not None
                                        else max_families_total * 2)
        self.novelty_max_similarity = novelty_max_similarity
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
        sig = structural_signature(source)
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
        sig = structural_signature(source)
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
        """
        active = [f for f in self.families.values() if f.status == "active"]

        def rank(f: Family) -> tuple[int, float, float]:
            unproven = 0 if f.rewrite_rounds_used == 0 else 1
            return (unproven, -self._improvement_pct(f), self._incumbent(f))

        active.sort(key=rank)
        return active[: self.max_families_active]

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
                    latency_ms: float) -> bool:
        family = self.families[family_id]
        if family.best is None or latency_ms < family.best.latency_ms:
            family.best = BestRecord(candidate_id=candidate_id, params=params,
                                     latency_ms=latency_ms)
            return True
        return False

    def record_round(self, family_id: str, best_latency_ms: float) -> None:
        self.families[family_id].best_history.append(best_latency_ms)

    def lineage_tree(self) -> dict:
        return {
            fid: {
                "anchor": f.anchor_candidate_id,
                "status": f.status,
                "best_ms": f.best.latency_ms if f.best else None,
                "history": f.best_history,
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
