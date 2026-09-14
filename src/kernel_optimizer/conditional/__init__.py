"""v4.1: the conditional measurement layer (U/W probes + C/E four-slot scan).

Spec: docs/implementation-conditional-scan-v4.1.md (sign-off:
docs/review-implementation-conditional-scan-v4.1.md). Modules:

  identity — typed configuration keys, partner projections, execution identity
  gates    — print gate (I/rho/tau), directional J_out/J_in token mint, I_rev algebra
  tokens   — source contrasts, one-shot tokens, new-vs-old supersession
  probe    — U/W compile-only probe planning, conditioned hard/soft walls, D budget
  scanner  — C/E phase machine, C4 protocol, E1/E2/E4 geometry+ordering, decision records

Everything here is pure planning/bookkeeping; the orchestrator owns dispatch and the GPU.
All of it sits behind `cfg.v4.conditional_scan.enabled` (off = the literal v3 path).
"""
