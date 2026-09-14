"""v4.1 §2: typed identity, partner projection, applicability.

Every conditioned fact in the scan layer is keyed by WHERE it was measured: the full typed
configuration minus the axis under study (the "partner projection") plus the execution identity
(candidate / source / workload / toolchain / device / protocol). Two values that print the same but
differ in type (`1` vs `1.0` vs `"1"` vs `True`) are different points — a JSON round-trip must not
merge them, which is why the tag is the python type name and bool is checked before int
(bool is an int subclass; `isinstance(True, int)` is True).
"""

from __future__ import annotations

import hashlib
from typing import Any


def typed_repr(value: Any) -> str:
    """Canonical `type:repr` string for one parameter value. Stable across JSON round-trips
    only if the caller re-parses to the declared type first; this function never coerces."""
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, int):
        return f"int:{value!r}"
    if isinstance(value, float):
        return f"float:{value!r}"
    return f"{type(value).__name__}:{value!r}"


def config_key(values: dict[str, Any]) -> str:
    """Digest of the FULL typed configuration."""
    body = ";".join(f"{k}={typed_repr(values[k])}" for k in sorted(values))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def partner_projection(values: dict[str, Any], axis: str) -> str:
    """Digest of everything EXCEPT `axis` — the frozen-partner identity a conditioned fact
    is attached to. Field-for-field equality is required for 'currently applicable'
    (v4.1 §2); there is no `stale_within_noise` — latency similarity never substitutes."""
    body = ";".join(f"{k}={typed_repr(values[k])}" for k in sorted(values) if k != axis)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def execution_identity(candidate_id: str, space_id: str, backend: str,
                       protocol: str = "v4.1") -> str:
    """The non-parameter half of a conditioned fact's identity. Same candidate + backend +
    protocol; the SPACE id is deliberately NOT part of it — a same-candidate expansion space
    keeps the identity so tokens can be inherited (v4.1 §0), while a different candidate
    never can (ruling ⑥: no cross-candidate transfer)."""
    del space_id
    return f"{candidate_id}|{backend}|{protocol}"
