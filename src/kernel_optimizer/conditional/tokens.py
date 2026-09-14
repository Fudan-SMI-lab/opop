"""v4.1 §4.1: source contrasts, one-shot tokens, and the new-vs-old source discipline.

A SOURCE CONTRAST is one completed C4 protocol: four fresh trials over two axis endpoints
(F = inner, N = wall-side), frozen geometry, discovery pair + validation pair. Its identity is
two-layered (v4.1 §4.1):

  * `contrast_id`   — immutable, unique per C4 execution; used for UNIQUE CONSUMPTION.
  * `comparable_key` — execution identity + exact partners + axis + frozen F/N values +
    protocol; used for NEW-VS-OLD comparison. "Different contrast_id" never means
    "never supersedes" — a newer FULL C4 on the same comparable_key supersedes the older one.

Supersession rules (v4.1 §4.1):
  * newer full C4 whose toward-wall reading turns negative → revoke the older outward
    eligibility; if its J_in qualifies it mints its OWN inward token.
  * newer full C4 with unknown/contradictory quality → suppress older eligibility.
  * provisional shapes / E2 degraded contrasts NEVER supersede a full C4.
  * different F/N values = a different conditioned comparison → different comparable_key,
    no blanket "newest on the axis wins".

Token semantics: minted only by `gates.evaluate_gates` picking a direction; at most one per
contrast (enforced by the gates' exclusivity); consumed at ADMISSION (subsequent failure or
cancellation does not refund); E output never mints recursively; unconsumed tokens survive
into same-candidate continuation spaces (execution identity matches; ruling ⑥ forbids
cross-candidate inheritance).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from kernel_optimizer.conditional.gates import Direction, MintResult, ordering_score

TokenState = Literal["pending", "consumed", "revoked"]


@dataclass
class SourceContrast:
    contrast_id: str
    comparable_key: str
    execution_identity: str
    candidate_id: str
    space_id: str
    axis: str
    partner_projection: str
    partner_values: dict[str, Any]
    f_value: Any                       # inner endpoint (frozen before first slot)
    n_value: Any                       # wall-side endpoint (frozen before first slot)
    f_lat: list[float] = field(default_factory=list)   # fresh replicates, arrival order
    n_lat: list[float] = field(default_factory=list)
    g_d: float | None = None           # discovery-pair prediction (frozen at discovery deadline)
    y: float | None = None             # validation-pair holdout
    mint: MintResult | None = None
    full: bool = True                  # False = provisional/E2-degraded; never supersedes

    def choice_span(self, declared: list[Any]) -> int:
        """Stratum key for §4.2 ordering: |index(N) − index(F)| over the DECLARED choices.
        Cross-axis units are not comparable, so ordering stratifies by this span first."""
        try:
            return abs(declared.index(self.n_value) - declared.index(self.f_value))
        except ValueError:
            return 0

    def gain_score(self) -> float | None:
        """Direction-aware ordering score (§4.2): positive for the minted direction."""
        if self.mint is None or self.mint.direction is None:
            return None
        return ordering_score(self.f_lat, self.n_lat, self.mint.direction)


@dataclass
class Token:
    token_id: str
    contrast_id: str
    comparable_key: str
    execution_identity: str
    candidate_id: str
    axis: str
    direction: Direction
    action_kind: Literal["E1", "E2", "E4"] = "E1"   # registered at mint; not downgraded later
    state: TokenState = "pending"
    consumed_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return {"token_id": self.token_id, "contrast_id": self.contrast_id,
                "axis": self.axis, "direction": self.direction,
                "action_kind": self.action_kind, "state": self.state}


class TokenStore:
    """Per-CANDIDATE store (cross-space within the candidate; ruling ⑥ scopes it).

    Held by the orchestrator across a candidate's spaces so a continuation space can consume
    an inherited pending token, paying with its own Q/B (v4.1 §0).
    """

    def __init__(self) -> None:
        self.contrasts: list[SourceContrast] = []
        self.tokens: list[Token] = []

    # ------------------------------------------------------------------ registration

    def register(self, contrast: SourceContrast) -> Token | None:
        """Record a completed contrast; apply supersession; mint at most one token.

        Returns the minted token, or None (no direction passed, or quality failed).
        """
        self.contrasts.append(contrast)
        if contrast.full:
            self._supersede_older(contrast)
        if not contrast.full:
            return None                     # provisional: records evidence, never mints
        mint = contrast.mint
        if mint is None or mint.direction is None:
            return None
        token = Token(
            token_id=f"tk-{uuid.uuid4().hex[:8]}",
            contrast_id=contrast.contrast_id,
            comparable_key=contrast.comparable_key,
            execution_identity=contrast.execution_identity,
            candidate_id=contrast.candidate_id,
            axis=contrast.axis,
            direction=mint.direction,
        )
        self.tokens.append(token)
        return token

    def _supersede_older(self, newer: SourceContrast) -> None:
        """A newer FULL C4 on the same comparable_key suppresses older pending tokens.

        The older token is revoked whatever the newer outcome was (negative, unknown or a
        different direction): the newest full evidence on the same conditioned comparison is
        the one that stands. The newer contrast's own mint (if any) arrives via `register`.
        """
        for tok in self.tokens:
            if (tok.state == "pending"
                    and tok.comparable_key == newer.comparable_key
                    and tok.contrast_id != newer.contrast_id):
                tok.state = "revoked"
                tok.consumed_reason = f"superseded by newer full C4 {newer.contrast_id}"

    # ------------------------------------------------------------------ consumption

    def pending(self, execution_identity: str) -> list[Token]:
        return [t for t in self.tokens
                if t.state == "pending" and t.execution_identity == execution_identity]

    def consume(self, token_id: str, reason: str) -> None:
        """ADMISSION consumes — including a subsequently failed or cancelled action."""
        for t in self.tokens:
            if t.token_id == token_id:
                if t.state != "pending":
                    raise ValueError(f"token {token_id} already {t.state}")
                t.state = "consumed"
                t.consumed_reason = reason
                return
        raise KeyError(token_id)

    def contrast_of(self, token: Token) -> SourceContrast:
        for c in self.contrasts:
            if c.contrast_id == token.contrast_id:
                return c
        raise KeyError(token.contrast_id)

    def snapshot(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        for t in self.tokens:
            states[t.state] = states.get(t.state, 0) + 1
        return {"n_contrasts": len(self.contrasts),
                "n_full": sum(1 for c in self.contrasts if c.full),
                "tokens_by_state": states}
