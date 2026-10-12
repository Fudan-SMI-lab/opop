"""Rank the sites a goal could be optimised at, from the goal's own measured profile.

WHY THIS EXISTS
---------------
The site used to be a constant (`C3_SITE_MODULE`, defaulting to one RMSNorm instance). That made the
search space a hand-picked guess, and it made it the SAME guess for all three goals -- even though the
three goals profile differently. Measured on the frozen Qwen3 profiles, the composite MLP site carries
146,793 us of associated CUDA work for `ttft` but only 17,258 us for `single`: the same module is the
biggest site under one objective and a minor one under another. Picking per goal is therefore part of
the goal-specific claim, not a convenience.

WHAT "ELIGIBLE" MEANS HERE
--------------------------
Nothing is re-implemented. A site is eligible exactly when `build_target` accepts it, so every gate
stays in one place: the leaf/composite rule, the KV-cache purity rule, the geometry rule (one
representative for `ttft`, two for the decode goals) and the "no nesting" rule. This module only
enumerates observed module paths, asks `build_target`, and keeps what it returns. A site that would
fail later cannot be ranked first here, because it never becomes an option.

RANKING
-------
The criterion, best first, is MEASURED OPTIMISATION SPACE, then size as a fallback:

  1. A site a short probe lane actually improved, ranked by the fractional gain it measured
     (`probed_gain_frac`, descending). This is the direct answer to "how much headroom does the goal
     have here", measured rather than guessed.
  2. A site with no probe, ranked by `associated_cuda_us` -- the measured CUDA time of the kernels a
     replacement would subsume, read from the goal's own profile -- with sites known REACHABLE ahead
     of unmeasured ones.
  3. A site measured to have no usable headroom -- either a probe found no gain, or the B20/B21
     equivalence evidence says no correct replacement can pass -- ranked last but kept selectable.

`associated_cuda_us` ALONE is the wrong criterion, and that was measured rather than reasoned. Ranking
purely by size puts `mlp.up_proj` first (88,822 us for ttft, 6.3x the published RMSNorm site), but B20
and B21 then measured what a numerically equivalent Triton GEMM actually scores on the frozen quality
gate at three projection sites, each against a `torch.nn.functional.linear` control that read exactly
0.0:

    up_proj    72 modules   equivalent GEMM 0.013889   (wrong-arm control 0.019014)   limit 0.01
    down_proj  36 modules   equivalent GEMM 0.012321
    q_proj     36 modules   equivalent GEMM 0.011955   (wrong-arm control 0.017937)

Every equivalent formulation fails, and halving the group from 72 to 36 modules moved the number only
from 0.013889 to 0.012321 -- so this is NOT the site-count amplification B12 characterised. Within one
site the three arms (different BLOCK_K, different tile grid, `input_precision="ieee"`) agreed to
sixteen digits, which rules out the K-reduction order too. What remains is that a Triton GEMM's bf16
output differs from cuBLAS's bf16 output at all, and no GEMM can avoid owning its reduction: the dot
product over K *is* the operator.

The accepted RMSNorm kernel passed at 0.0 by a route a GEMM cannot take -- it left `mean` and `rsqrt`
to ATen and authored only elementwise kernels whose single-rounding `mul.rn.f32` / `cvt.rn.bf16.f32`
sequences are bit-identical to eager. So "how much CUDA time does this site carry" and "can a correct
replacement of this site be ACCEPTED and made FASTER" are different questions, and a lane sent to a
site that fails the second one burns its whole window rediscovering that.

`reachability` (see `Reachability`) is the document that carries this measured per-site evidence: the
binary can-a-correct-replacement-pass verdict from B20/B21, and -- when a probe has run -- the
fractional gain a short lane measured at the site. Both change ORDER only; a site with a failure verdict
or a zero-gain probe is ranked last rather than dropped, so a campaign can still be pointed at one
deliberately and the record shows why it was not chosen. No limit is adjusted anywhere: this changes
which site is searched, never what counts as correct.

DISJOINTNESS
------------
`BindingRuntime.register_site` refuses a module that already belongs to another group, so two goals
may share one site ID but may not hold OVERLAPPING different ones. Choosing per goal independently can
produce exactly that (one goal picks `...mlp`, another picks `...mlp.down_proj`), and it would surface
as a matrix that cannot register its rows. `resolve_overlaps` settles it by promoting the contained
site to the containing one, which keeps both goals' work inside a site each was measured on.
"""

import json
from pathlib import Path

from .fast_records import TargetSpec
from .fast_target import build_target, load_profile
from .runner_records import FrozenRecord, GoalId, RunnerError


class SiteOption(FrozenRecord):
    """One site a goal could be optimised at, with the measurement that ranked it."""

    module_path: str
    site_id: str
    module_paths: tuple[str, ...]
    associated_cuda_us: float
    exclusive_fraction: float | None
    composite: bool
    # Measured evidence about whether a CORRECT replacement of this site can pass the quality gate.
    # None means nothing has been measured for it, which is the normal case and ranks on size alone.
    reachable: bool | None = None
    reachability_detail: str | None = None
    # Measured optimisation space: the fractional gain a short probe lane achieved at this site for this
    # goal (`(baseline-best)/baseline` for minimise, symmetric for maximise). None means no probe ran,
    # which ranks on size. A probe that ran but found no meaningful win records 0.0, which ranks the
    # site among the exhausted ones -- "we looked and there is no headroom here" is itself a result.
    probed_gain_frac: float | None = None


class Reachability(FrozenRecord):
    """Measured evidence that a numerically equivalent replacement of a site can, or cannot, pass.

    This is deliberately shaped like `TargetSpec.reference_noise_us` and the equivalence floor: a
    property of a site that must come from a MEASUREMENT rather than from a hand-written list. The
    evidence document names the sites by `site_id`, so evidence collected for one group can never be
    read as evidence about a different one.

    Written by the B20/B21 probes, which bind numerically equivalent replacements at a site and record
    what the production `runner.quality` returns, always beside a torch control that must read exactly
    0.0. A site with `reachable: false` is one where every equivalent formulation exceeded the declared
    limit -- so no correct candidate can be accepted there, and a lane sent there would spend its whole
    window discovering that.
    """

    site_id: str
    reachable: bool
    detail: str | None = None
    measured_l2: float | None = None
    control_l2: float | None = None
    limit: float | None = None
    # The fractional gain a short probe lane measured at this site for this goal, when one ran. This is
    # the MEASURED optimisation space -- the direct evidence for "how much headroom does the goal have
    # here" -- and it is what ranks a site above a merely-bigger one. Absent (None) means no probe ran
    # and the site ranks on size; present-and-0.0 means a probe found no meaningful win. It is keyed by
    # `site_id` for a specific `goal`, so a document mixes evidence for several goals and each goal reads
    # only its own rows.
    goal: GoalId | None = None
    probed_gain_frac: float | None = None
    probed_baseline: float | None = None
    probed_best: float | None = None
    probe_detail: str | None = None


def load_reachability(path: Path | None) -> "SiteEvidence":
    """Read measured site evidence (binary reachability and/or probe gain), for ranking.

    Absent evidence is not an error: every site then ranks on size alone, which is the behaviour before
    any of this was measured. A malformed document IS an error, because silently ignoring it would rank
    a site as unknown when its evidence says it cannot be used.

    A row may be goal-agnostic (`goal` unset -- the B20/B21 can-a-correct-replacement-pass verdict,
    which is a property of the operator) or goal-specific (`goal` set -- the fractional gain a short
    probe measured for that goal at that site). One document holds both, and one site may have several
    rows; `SiteEvidence.for_site` merges the applicable ones for a given goal.
    """
    if path is None:
        return SiteEvidence(())
    if not path.is_file():
        raise RunnerError(f"declared site-reachability evidence is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("sites") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise RunnerError("site-reachability evidence must hold a list of per-site records")
    return SiteEvidence(tuple(Reachability.model_validate(row) for row in rows))


class SiteEvidence:
    """Measured per-site evidence, queryable by (site_id, goal).

    A binary reachability row (`goal is None`) applies to every goal; a probe row (`goal` set) applies
    only to its goal. `for_site` returns the single record that should inform one site's rank for one
    goal: the goal-specific probe row when present, layered over the goal-agnostic reachability verdict,
    so a site can be both "a correct kernel can pass here" AND "a probe measured 3.1% headroom for ttft"
    at once. Kept as a small class rather than a dict because one site_id legitimately carries several
    rows once probes for different goals exist -- a dict keyed by site_id alone could not hold them.
    """

    def __init__(self, rows: "tuple[Reachability, ...] | list[Reachability]") -> None:
        self._rows = tuple(rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SiteEvidence) and self._rows == other._rows

    def for_site(self, site_id: str, goal: GoalId | None = None) -> Reachability | None:
        """The evidence that should rank `site_id` for `goal`, or None if nothing applies.

        A goal-specific probe row is layered over any goal-agnostic reachability row for the same site,
        so the merged record carries both the pass/fail verdict and the measured gain. When no probe row
        exists the goal-agnostic verdict is returned unchanged; when neither exists, None.
        """
        agnostic = next((r for r in self._rows if r.site_id == site_id and r.goal is None), None)
        specific = None
        if goal is not None:
            specific = next((r for r in self._rows if r.site_id == site_id and r.goal == goal), None)
        if specific is None:
            return agnostic
        if agnostic is None:
            return specific
        # Layer the probe's gain onto the reachability verdict without losing either side.
        return agnostic.model_copy(update={"goal": goal, "probed_gain_frac": specific.probed_gain_frac,
            "probed_baseline": specific.probed_baseline, "probed_best": specific.probed_best,
            "probe_detail": specific.probe_detail})


class SiteChoice(FrozenRecord):
    goal: GoalId
    chosen: SiteOption
    # Every option considered, best first, so a reader can see what the choice was made against
    # instead of taking the winner on trust.
    considered: tuple[SiteOption, ...]
    # Set when `resolve_overlaps` moved this goal off its own best option, with the reason.
    promoted_from: str | None = None


def observed_modules(profile) -> tuple[str, ...]:
    """Every module path the profile observed during prefill, deepest first.

    Prefill is the phase `build_target` selects its representative from, so a path with no observed
    prefill call could never become a site anyway.
    """
    paths = {c.module_path for c in profile.calls if c.phase == "prefill" and c.module_path}
    return tuple(sorted(paths, key=lambda p: (-p.count("."), p)))


def candidate_sites(profile_path: Path, *, allow_composite: bool = False,
                    reachability: "SiteEvidence | dict[str, Reachability] | None" = None,
                    goal: GoalId | None = None) -> tuple[SiteOption, ...]:
    """Ask `build_target` about every observed module, keep the ones it accepts, rank them.

    One `build_target` call per module path, all sharing ONE parse of the profile -- these files are
    4.86 MB and there are ~550 observed modules, so re-reading per module is the difference between
    seconds and many minutes. It is CPU-only work on an already-recorded profile, so it spends no
    profile budget and no GPU time; a rejected module costs one exception.

    Sites are de-duplicated by `site_id`: the 36 MLP instances of one family all produce the SAME site
    (the family group), and listing it 36 times would say nothing new. The shallowest module path is
    kept as the representative so the recorded choice reads as `model.layers.0.mlp`.

    The order (see `rank_key`) is: sites a probe MEASURED with the most gain first, then sites with no
    probe (reachable-first, then by size), then sites with no usable headroom (a zero-gain probe or a
    failure verdict). `goal` selects which goal's probe rows apply; a site that cannot pass or was probed
    to nothing is kept in the list rather than dropped, so it can still be selected deliberately and the
    record shows what was known about it. See the module docstring for the measurements.
    """
    evidence = _as_evidence(reachability)
    profile = load_profile(profile_path)
    modules = observed_modules(profile)
    options: dict[str, SiteOption] = {}
    for module_path in modules:
        try:
            target = build_target(profile_path, module_path, allow_composite=allow_composite,
                                  profile=profile)
        except RunnerError:
            continue
        except (OSError, ValueError, SyntaxError, TypeError, ArithmeticError, RecursionError):
            # A module whose recorded source cannot even be parsed is not a site; it is also not a
            # reason to abandon the ranking, so it is skipped like any other rejection.
            continue
        known = evidence.for_site(target.brief.site_id, goal)
        option = SiteOption(module_path=module_path, site_id=target.brief.site_id,
                            module_paths=target.brief.module_paths,
                            associated_cuda_us=target.associated_cuda_us,
                            exclusive_fraction=target.exclusive_fraction,
                            composite=any(p.startswith(module_path + ".") for p in modules),
                            reachable=None if known is None else known.reachable,
                            reachability_detail=None if known is None else known.detail,
                            probed_gain_frac=None if known is None else known.probed_gain_frac)
        existing = options.get(option.site_id)
        if existing is None or (len(option.module_path), option.module_path) < (
                len(existing.module_path), existing.module_path):
            options[option.site_id] = option
    return tuple(sorted(options.values(), key=rank_key))


def _as_evidence(reachability: "SiteEvidence | dict[str, Reachability] | None") -> "SiteEvidence":
    """Accept a `SiteEvidence`, a legacy `{site_id: Reachability}` dict, or None, and return evidence.

    The dict form is still accepted so a caller that already keyed evidence by site_id keeps working;
    it can only carry one row per site (the goal-agnostic reachability verdict), which is exactly what
    that shape ever held.
    """
    if reachability is None:
        return SiteEvidence(())
    if isinstance(reachability, SiteEvidence):
        return reachability
    return SiteEvidence(tuple(reachability.values()))


def rank_key(option: SiteOption) -> tuple:
    """Sort key: most measured gain first, then unprobed by reachability+size, then exhausted last.

    Three tiers, best first:
      0. A probe measured a MEANINGFUL win here (`probed_gain_frac > 0`) -- ordered by that gain,
         descending. Measured optimisation space is the criterion the user asked for, so it wins over a
         site that is merely bigger or merely known-reachable.
      1. No probe ran -- ordered as before: measured-reachable, then unmeasured, then measured
         unreachable; and within a reachability tier, by descending `associated_cuda_us`.
      2. A probe ran and found NO usable headroom (`probed_gain_frac == 0.0`), OR the equivalence
         evidence says the site is unreachable -- ranked last but still selectable.

    Separated from `candidate_sites` so the ordering is one testable rule rather than an inline lambda.
    """
    gain = option.probed_gain_frac
    if gain is not None and gain > 0:
        return (0, -gain, len(option.module_paths), option.module_path)
    if gain == 0.0 or option.reachable is False:
        return (2, {True: 0, None: 1, False: 2}[option.reachable], -option.associated_cuda_us,
                len(option.module_paths), option.module_path)
    reach_tier = {True: 0, None: 1, False: 2}[option.reachable]
    return (1, reach_tier, -option.associated_cuda_us, len(option.module_paths), option.module_path)


def contains(outer: SiteOption, inner: SiteOption) -> bool:
    """Does every module of `inner` sit inside the subtree of some module of `outer`?

    This is the relation `register_site` cares about. It is asked on the MODULE SETS rather than on
    the selected paths, because the group is what gets registered.
    """
    return all(any(p == q or p.startswith(q + ".") for q in outer.module_paths) for p in inner.module_paths)


def overlaps(a: SiteOption, b: SiteOption) -> bool:
    if a.site_id == b.site_id:
        return False  # the same group registered once is explicitly allowed
    return any(p == q or p.startswith(q + ".") or q.startswith(p + ".")
               for p in a.module_paths for q in b.module_paths)


def resolve_overlaps(choices: dict[GoalId, SiteChoice]) -> dict[GoalId, SiteChoice]:
    """Make the chosen sites either identical or disjoint, by promoting contained sites.

    When two goals pick sites where one contains the other, the contained goal is moved to the
    containing site -- it is still a site that goal's own profile accepted (it appears in its
    `considered` list), and it strictly contains the work that goal ranked first, so the goal is not
    being pointed at something it never measured.

    A pair that overlaps WITHOUT one containing the other cannot be fixed this way. Rather than invent
    a site, the later goal falls back to its best option that is disjoint from (or identical to) the
    ones already settled, and `promoted_from` records that it moved.
    """
    order = sorted(choices, key=lambda g: -choices[g].chosen.associated_cuda_us)
    settled: dict[GoalId, SiteChoice] = {}
    for goal in order:
        choice = choices[goal]
        conflict = next((s for s in settled.values() if overlaps(s.chosen, choice.chosen)), None)
        if conflict is None:
            settled[goal] = choice
            continue
        if contains(conflict.chosen, choice.chosen):
            promoted = next((o for o in choice.considered if o.site_id == conflict.chosen.site_id), None)
            if promoted is not None:
                settled[goal] = choice.model_copy(update={"chosen": promoted,
                    "promoted_from": f"{choice.chosen.module_path}: contained by {conflict.goal}'s site"})
                continue
        alternative = next((o for o in choice.considered
                            if not any(overlaps(s.chosen, o) for s in settled.values())), None)
        if alternative is None:
            raise RunnerError(f"goal {goal} has no site disjoint from the goals already chosen")
        settled[goal] = choice.model_copy(update={"chosen": alternative,
            "promoted_from": f"{choice.chosen.module_path}: overlaps {conflict.goal}'s site"})
    return settled


def select_sites(profiles: dict[GoalId, Path], *, allow_composite: bool = False,
                 forced: str | None = None,
                 reachability: "Path | SiteEvidence | dict[str, Reachability] | None" = None) -> dict[GoalId, SiteChoice]:
    """The per-goal site choice: each goal's highest-ranked eligible site, made mutually registrable.

    `forced` pins every goal to one module path instead, which is what reproduces an earlier campaign
    or answers "what would this site have done" -- the choice is then recorded as a forced one with the
    ranking it beat, so the table still says what else was on the table. A forced site is used even if
    the evidence says it is unreachable, because that is what being asked for it means; the option's own
    `reachable` field still carries the warning.

    A forced path is resolved by BUILDING at that path, not by looking for it among the ranked options'
    labels. Many module paths map to one site (all 73 RMSNorm instances of the published site do), and
    the ranking keeps just one of them as the label -- `model.norm`, which is the shortest. Matching on
    the label would have refused `model.layers.33.post_attention_layernorm`, the exact path the
    published campaign used, as "not eligible".

    `reachability` is measured evidence (a path to a document, or an already-loaded `SiteEvidence`) about
    each site's optimisation space: the B20/B21 can-a-correct-replacement-pass verdict, and the
    fractional gain a short probe measured PER GOAL. Its only effect is on ORDER -- a site a probe found
    the most headroom for sorts first, a site with no headroom or an unreachable verdict sorts last.
    Because probe gains are per goal, each goal's ranking reads only its own probe rows. Nothing about
    the gate itself changes.
    """
    evidence = (load_reachability(reachability) if reachability is None or isinstance(reachability, Path)
                else _as_evidence(reachability))
    choices: dict[GoalId, SiteChoice] = {}
    for goal, profile_path in profiles.items():
        options = candidate_sites(profile_path, allow_composite=allow_composite, reachability=evidence,
                                  goal=goal)
        if not options:
            raise RunnerError(f"goal {goal} has no eligible site in {profile_path}")
        if forced is None:
            chosen = options[0]
        else:
            try:
                target = build_target(profile_path, forced, allow_composite=allow_composite)
            except RunnerError as exc:
                raise RunnerError(f"forced site {forced} is not eligible for goal {goal}: {exc}") from exc
            ranked = next((o for o in options if o.site_id == target.brief.site_id), None)
            known = evidence.for_site(target.brief.site_id, goal)
            chosen = ranked if ranked is not None else SiteOption(module_path=forced,
                site_id=target.brief.site_id, module_paths=target.brief.module_paths,
                associated_cuda_us=target.associated_cuda_us,
                exclusive_fraction=target.exclusive_fraction, composite=False,
                reachable=None if known is None else known.reachable,
                reachability_detail=None if known is None else known.detail,
                probed_gain_frac=None if known is None else known.probed_gain_frac)
            # Record the path that was actually asked for, so the campaign's site matches the request
            # even when the ranking labelled the same site by a different member.
            chosen = chosen.model_copy(update={"module_path": forced})
        choices[goal] = SiteChoice(goal=goal, chosen=chosen, considered=options)
    if forced is not None:
        return choices
    return resolve_overlaps(choices)


def build_selected_target(profile_path: Path, choice: SiteChoice, *, allow_composite: bool) -> TargetSpec:
    """Build the goal's target at its chosen site, and verify it is the site that was ranked.

    The re-check is not ceremony: `select_sites` and the campaign each call `build_target`, and if
    those two ever disagreed the campaign would search a site the ranking never measured.
    """
    target = build_target(profile_path, choice.chosen.module_path, allow_composite=allow_composite)
    if target.brief.site_id != choice.chosen.site_id:
        raise RunnerError("selected site changed between ranking and target construction")
    return target
