# Measurement: 55% of constraints reject nothing — and that is almost entirely fine

Noticed while checking the clean L3:21 rerun's first published space, which contained:

```
BLOCK_SIZE * 0 <= MAX_SHARED_BYTES_OPTIN
```

Always true. That looked like a whole class of defect worth chasing — constraints written to
look like protection while protecting nothing. Measuring it turned the alarming number into a
small one, and the process is the point of recording it.

## The scary number, and why it is wrong

Evaluating every published constraint on its own against its own space's grid:

```
constraints across all published spaces:              519
that reject NOTHING over their own grid (vacuous):    287  (55%)
spaces containing >= 1 such constraint:               130
```

The most common by far is `NUM_WARPS * 32 <= MAX_THREADS_PER_BLOCK`, 82 times, plus another
~40 of the same shape under prefixed names (`QKV_`, `ATTN_`, `PV_`, `DW_`, …).

That constraint is **correct and worth having**. `MAX_THREADS_PER_BLOCK` is 1024, so it binds
at `NUM_WARPS = 32`, and no agent has offered a warp count above 16. It rejects nothing
*because the domain never reaches its limit* — which is protection that happens not to be
needed at the current range, not a bug. It would start binding the moment an expansion widened
the warp knob far enough, which is exactly when it matters.

Same for the `% 16 == 0` family (26 occurrences): agents offer only powers of two ≥ 16 for those
tiles, so the divisibility rule never fires. Redundant against the choices, not false.

So "rejects nothing over the current grid" is the wrong test. It measures the *domain*, not the
constraint.

## The real number: 1.3%

The right test is whether a constraint could **ever** bind. Re-evaluating each one with its
numeric domains artificially widened to powers of two up to 65536:

```
constraints total:                                    519
STRUCTURALLY vacuous (reject nothing even then):        7  (1.3%)
```

All seven:

```
2x  0 <= MAX_SHARED_BYTES_OPTIN
1x  BLOCK_SIZE * 0 <= MAX_SHARED_BYTES_OPTIN
1x  NUM_WARPS >= 1
1x  NUM_STAGES >= 0
1x  EXPAND_NUM_STAGES >= 1 and FUSED_NUM_STAGES >= 1
1x  CHUNK > 0 and BLOCK_P > 0 and NUM_STAGES > 0
```

## And even those seven are not a defect

The three `MAX_SHARED_BYTES_OPTIN` ones all belong to **elementwise copy kernels** that use no
shared memory at all:

```
l3-21-0905-153452  cand-d086960b  sp-b174cc79   uses tl.dot: False   num_stages: False
l3-21-0905-153452  cand-d086960b  sp-25531280   uses tl.dot: False   num_stages: False
l3-21-0905-160156  cand-e9b995d0  sp-4ff7f575   uses tl.dot: False   num_stages: False
```

A kernel that stages nothing has a shared-memory footprint of zero, so `0 <= LIMIT` is a
*truthful* statement of its requirement. It is the honest answer to a prompt that asks for a
shared-memory bound, from a kernel that needs none. The remaining four (`NUM_WARPS >= 1`,
`NUM_STAGES >= 0`, and two positivity conjunctions) are restatements of the domain's own
minimum — noise in the rationale, zero cost at runtime.

Note the first two rows are `cand-d086960b`, the delegating candidate from the abandoned run
(`finding-candidate-delegates-to-baseline-compiler.md`). Its space is vacuous-by-construction
because its "kernel" was a no-op copy. The vacuous constraint was a *symptom* of that defect,
not an independent one — which is the second time this shape of thing has pointed at the real
problem sideways, after `inert_space`.

## Nothing changed, deliberately

No fix, for three reasons:

1. **The rate of genuine emptiness is 1.3%, and every instance is either truthful or harmless.**
   There is no measurable cost: the guard evaluates a handful of cheap expressions per sampled
   config.
2. **A "reject vacuous constraints" gate would fire on the 55%, not the 1.3%.** Any check cheap
   enough to run at publish time evaluates against the *current* grid, so it would reject the
   `NUM_WARPS * 32 <= 1024` bounds that are correct and that
   `finding-expansion-drops-inherited-constraints.md` just spent a fix restoring. That would
   be actively harmful — and it is the same trap as measuring `min`-direction expansion by its
   current-domain yield.
3. **The prompt already asks for the right thing.** `_render_prompt` step 3 tells the agent to
   derive the arithmetic from the kernel's own body and warns that a plausible-looking formula
   copied from elsewhere is worse than none. An elementwise kernel writing `0 <= LIMIT` is
   *complying* with that instruction.

What this does buy is a check I can run: `scripts/audit_vacuous_constraints.py` reports both
numbers, and the structural one is the one to watch. If it climbs above a few percent — or if a
`MAX_SHARED_BYTES` bound ever appears as structurally vacuous on a kernel that **does** use
`tl.dot` or `num_stages` — that is a real finding, because it would mean a staging kernel
declared a bound that cannot constrain it.
