# Measurement: the failed-hypothesis channel changes what the rewriter proposes, 9 of 11 — and 0 of 9 won

`cand-e3a5da01` (14:45, `fam-4aea322a` round 3) is the first rewrite whose `approach_summary`
visibly reacts to a recorded failure:

> "Uses a distinct row-microtiled QKV decomposition … **Unlike the failed narrow-column/warp-group
> rewrite**, it does not partition output columns or serialize column subtiles."

That is a direct reference to `cand-aa016dfe`, the H1 implementation refuted at 11.8 vs 11.0
(`result-analyst-hypothesis-refuted-by-control.md`). Worth recording because most of this session's
findings are defects, and this is a piece of machinery working exactly as designed.

## The mechanism, and it is already correct

`orchestrator.py:1077-1088`: when a round fails to improve (`best_after >= best_before`), the
source candidate's hypotheses are appended to `failed_hypotheses[family_id]` and journalled as
`HYPOTHESES_FAILED`. The rewriter receives them as `history/failed_hypotheses.json`
(`modules.py:629`) with the instruction that they are "changes already tried that did NOT help".

On disk for `fam-4aea322a`, written at 14:02:50 — the same second the flat round was recorded:

```
H1: Reshape the persistent QKV matmul so a CTA covers more rows without holding a 256x128
    accumulator: use a 256x32 or 256x64 logical tile, partition rows...
H2: Reduce the flash-attention pipeline's per-stage storage by staging K and V in reused buffers...
H3: Add per-kernel precision knobs rather than the current global COMPUTE_DTYPE...
```

`cand-e3a5da01` then proposes a 256×128 tile serialising **four 64×128 row slices** — same
256-row goal, opposite decomposition (row-wise rather than column-wise), and it says so. The
feedback loop closed without any intervention.

## The base rate: 9 of 11 react, and none of the 9 won

The `HYPOTHESES_FAILED` **journalling** is new (`7db5bf1`, 09-05), so earlier runs show 0 events —
but the in-memory dict has always been populated, and the trigger condition is reconstructible from
`FAMILY_ROUND_RECORDED`. Every rewrite issued while its family had a non-empty dict:

| | n |
|---|---|
| rewrites issued with failed-hypothesis context | **11** |
| summary explicitly contrasts with a prior attempt | **9 (82%)** |
| of the 9 measurable, child beat the family incumbent | **0** |

```
run                    candidate        cites   child  fam_best
l3-21-0903-071357      cand-6f467ced    True     24.6    24.6
l3-21-0903-071357      cand-18529f35    False    24.7    24.6
l3-43-0903-020233      cand-49a2d908    True     29.3    29.3
l3-43-0903-020233      cand-674372d7    True     36.8    31.6
l3-43-0903-020233      cand-05a299d9    True     56.1    31.6
l3-43-0903-145357      cand-dce0040b    True     21.3    19.4
l3-43-0903-145357      cand-2246d8ea    False    19.7    19.5
l3-43-0904-093730      cand-c36d7820    True     21.0    19.6
l3-43-0904-093730      cand-d257924a    True     22.3    19.6
l3-43-0905-091705      cand-e3a5da01    True     (pending) 11.0
l3-48-0905-010737      cand-ef6f0748    True     (pending) 2.09
```

So the channel reliably changes the *proposal* and has never yet produced a *win*.

## The confound, which makes the second number nearly uninterpretable

The obvious comparison:

```
WITH failed-hyp context    n=9   median -7.1%   beat incumbent 0/9
WITHOUT (first round)      n=23  median -1.9%   beat incumbent 6/23
```

**This does not show the context is harmful, and I will not report it that way.** The two groups
differ in *round number*, not only in context: a rewrite has failed-hypothesis context precisely
because its family already had a non-improving round, i.e. it is a round-2-or-later rewrite. And
round 3 has an independent **0-for-13** record
(`finding-converged-stop-kind-is-unreachable.md`) for reasons that have nothing to do with what the
rewriter was told — by that point the family's easy structural wins are taken and the incumbent it
must beat is the best of everything tried so far.

The two effects are inseparable in this data. `0 of 9` is consistent with "the context does not
help", with "late rounds are hard", and with both. What would separate them is a round-2 rewrite
issued *without* the context as a control, and the harness never does that — correctly, since
withholding known-failed information to make a cleaner measurement would be a worse search.

## What is actually established

1. **The channel works mechanically.** The event is journalled at the right moment, survives
   resume, reaches the rewriter, and demonstrably shifts the proposal — 9 of 11, with explicit
   contrastive language. That was worth verifying because
   `finding-parameterizer-lacks-triton-pitfalls-doc.md` found a sandbox file that was silently
   never delivered; this one is delivered and consumed.
2. **It does not rescue a stalled family.** Whatever the cause, no family in the record has been
   turned around by a rewrite that had failed-hypothesis context. Combined with the seeding
   counterfactual (`result-seeded-history-counterfactual.md`), where 6 of 14 families would have
   frozen at `used=2` and **all 6 gained exactly 0.00% in their actual round 3**, the picture is
   consistent: by the time this context exists, the family is usually done.
3. **That is an argument for the seeding fix, not against the channel.** If a stalled family froze
   one round earlier, the round this context informs would mostly not be spent at all — and the two
   families it would *keep* funded are the ones with a real slope, where the context has its best
   chance.

## Nothing to change

The channel is correct as built. The one thing I would *not* do is make it more aggressive (e.g.
forbidding the rewriter from touching a previously-failed axis): `cand-e3a5da01` is attacking the
same 256-row goal H1 failed at, by a different decomposition, and that is a legitimate second
attempt rather than a repeat — the H1 refutation established that 256 rows *are* reachable and the
gain was not there, which does not prove no decomposition reaches it profitably.

Two open items to watch: `cand-e3a5da01` and `cand-ef6f0748` are the two pending children above. If
either beats its incumbent, the `0 of 9` becomes `1 of 10` and the channel has its first win.

Reproduce with `scripts/audit_failed_hypothesis_channel.py`.
