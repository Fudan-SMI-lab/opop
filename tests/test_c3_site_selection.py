"""Per-goal site selection: rank the sites a goal could be optimised at, keep them registrable.

WHY THIS MATTERS
----------------
The site used to be one hardcoded module path, shared by all three goals. That made the search space a
guess, and the SAME guess under three different objectives -- while the measured profiles disagree
about which site is largest (the composite MLP carries 146,793 us for `ttft` against 17,258 us for
`single`). Choosing per goal from each goal's own profile is part of the goal-specific claim.

WHAT THESE TESTS PIN
--------------------
1. Eligibility is delegated: a site is an option exactly when `build_target` accepts it, so the four
   gates keep living in one place. Nothing here re-implements them.
2. Ranking is by measured `associated_cuda_us`, descending, deterministically.
3. The chosen sites are mutually REGISTRABLE -- identical or disjoint -- because
   `BindingRuntime.register_site` refuses a module that belongs to another group, and a matrix whose
   rows cannot register loses every optimised cell (the defect `register_row_sites` was added for).
4. `build_target` still fingerprints the profile PATH when handed a pre-parsed profile, so the
   performance shortcut cannot produce a spec that describes different bytes.

These run against the real frozen Qwen3 profile, so the module tree and family keys are production
ones rather than a mock of what they might look like.
"""

import json
from pathlib import Path

import pytest

from examples.c3_qwen3.fast_sites import (SiteChoice, SiteOption, candidate_sites, contains,
                                          load_reachability, observed_modules, overlaps, rank_key,
                                          resolve_overlaps, select_sites)
from examples.c3_qwen3.fast_target import build_target, load_profile, require_target
from examples.c3_qwen3.runner_records import RunnerError

RESULTS = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration"


def profile_path() -> Path:
    """The frozen short device profile, wherever this checkout keeps it.

    A whole profile is required: `build_target` parses `cuda_intervals` and `site_kernels`, and the
    locally mirrored copies of some campaigns are truncated.
    """
    for candidate in sorted(RESULTS.glob("**/short-device-profile-*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8")).get("data") or {}
        except (ValueError, OSError):
            continue
        if data.get("valid") and data.get("cuda_intervals") and data.get("site_kernels"):
            return candidate
    pytest.skip("no complete short device profile is mirrored in this checkout")


def option(module_path: str, paths: tuple[str, ...], us: float = 1.0,
           reachable: bool | None = None, probed_gain_frac: float | None = None) -> SiteOption:
    return SiteOption(module_path=module_path, site_id="pure-" + module_path, module_paths=paths,
                      associated_cuda_us=us, exclusive_fraction=None, composite=False,
                      reachable=reachable, probed_gain_frac=probed_gain_frac)


# --- probed optimisation space: MEASURED gain outranks size and mere reachability -----------


def test_a_measured_gain_outranks_a_bigger_unprobed_site():
    # Given: a small site a probe measured 3.1% headroom at, and a huge unprobed site. The user's rule
    # is "rank by MEASURED optimisation space, not size", so the probed site must come first even though
    # it is a fraction of the size.
    probed_small = option("norm", ("norm",), us=14_041.1, probed_gain_frac=0.031)
    big_unprobed = option("up_proj", ("up_proj",), us=88_821.9)
    ordered = sorted([big_unprobed, probed_small], key=rank_key)
    assert [o.module_path for o in ordered] == ["norm", "up_proj"]


def test_more_measured_gain_outranks_less():
    # Given: two probed sites. The one with the larger measured gain wins, regardless of size.
    less = option("a", ("a",), us=99.0, probed_gain_frac=0.02)
    more = option("b", ("b",), us=1.0, probed_gain_frac=0.05)
    assert [o.module_path for o in sorted([less, more], key=rank_key)] == ["b", "a"]


def test_a_zero_gain_probe_ranks_below_an_unprobed_site():
    # Given: a probe that ran and found NO usable headroom (0.0). "We looked and there is nothing here"
    # must rank the site below one nobody has probed yet -- otherwise a proven dead end would be chosen
    # over an unexplored, possibly-better one.
    exhausted = option("a", ("a",), us=99.0, probed_gain_frac=0.0)
    unprobed = option("b", ("b",), us=1.0)
    assert [o.module_path for o in sorted([exhausted, unprobed], key=rank_key)] == ["b", "a"]


def test_probe_evidence_reorders_selection_per_goal(tmp_path):
    # Given: a probe measured headroom at DIFFERENT sites for two goals. Each goal's ranking must read
    # only its own probe rows, so the two goals can end up choosing different sites from the same profile
    # -- the whole point of goal-specific site selection.
    up = build_target(profile_path(), "model.layers.0.mlp.up_proj").brief.site_id
    down = build_target(profile_path(), "model.layers.0.mlp.down_proj").brief.site_id
    evidence = {"sites": [
        {"site_id": up, "reachable": True, "goal": "ttft", "probed_gain_frac": 0.08,
         "probe_detail": "short ttft lane won 8%"},
        {"site_id": down, "reachable": True, "goal": "single", "probed_gain_frac": 0.06,
         "probe_detail": "short single lane won 6%"},
    ]}
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    ev = load_reachability(path)
    ttft = candidate_sites(profile_path(), reachability=ev, goal="ttft")
    single = candidate_sites(profile_path(), reachability=ev, goal="single")
    assert ttft[0].site_id == up            # ttft's own probe put up_proj first
    assert single[0].site_id == down        # single's own probe put down_proj first
    assert ttft[0].probed_gain_frac == 0.08
    assert single[0].probed_gain_frac == 0.06


def test_a_probe_row_layers_over_the_reachability_verdict(tmp_path):
    # Given: a site with BOTH a goal-agnostic reachability verdict and a goal-specific probe gain. The
    # merged evidence must carry both -- the pass/fail verdict AND the measured headroom -- so neither
    # source is lost when the other is present.
    site = build_target(profile_path(), "model.norm").brief.site_id
    evidence = {"sites": [
        {"site_id": site, "reachable": True, "detail": "RMSNorm passes at 0.0"},
        {"site_id": site, "goal": "ttft", "reachable": True, "probed_gain_frac": 0.045},
    ]}
    path = tmp_path / "layered.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    merged = load_reachability(path).for_site(site, "ttft")
    assert merged is not None and merged.reachable is True and merged.probed_gain_frac == 0.045


def test_a_probe_row_for_another_goal_does_not_apply(tmp_path):
    # Given: probe evidence measured for `single`. Ranking `ttft` must NOT read it -- optimisation space
    # is a per-goal quantity, and borrowing one goal's headroom for another is exactly the mistake the
    # per-goal design removes.
    site = build_target(profile_path(), "model.norm").brief.site_id
    evidence = {"sites": [{"site_id": site, "goal": "single", "reachable": True, "probed_gain_frac": 0.09}]}
    path = tmp_path / "otherg.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    ev = load_reachability(path)
    assert ev.for_site(site, "ttft") is None            # ttft sees nothing
    assert ev.for_site(site, "single").probed_gain_frac == 0.09


# --- reachability: measured evidence changes the ORDER, never the gate -----------------------


def test_rank_key_puts_reachable_first_then_unmeasured_then_unreachable():
    # Given: a small site measured REACHABLE and a huge one measured UNREACHABLE. Raw size would put
    # the huge one first and send a lane somewhere no correct candidate can be accepted.
    reachable_small = option("norm", ("norm",), us=14_041.1, reachable=True)
    unmeasured_mid = option("o_proj", ("o_proj",), us=18_382.2, reachable=None)
    unreachable_big = option("up_proj", ("up_proj",), us=88_821.9, reachable=False)
    ordered = sorted([unreachable_big, unmeasured_mid, reachable_small], key=rank_key)
    assert [o.module_path for o in ordered] == ["norm", "o_proj", "up_proj"]


def test_size_still_breaks_ties_inside_one_reachability_tier():
    small = option("a", ("a",), us=10.0, reachable=True)
    large = option("b", ("b",), us=99.0, reachable=True)
    assert [o.module_path for o in sorted([small, large], key=rank_key)] == ["b", "a"]


def test_an_unreachable_site_is_demoted_not_dropped(tmp_path):
    # Given: a site that cannot pass must remain SELECTABLE -- a campaign may want to point at it
    # deliberately, and dropping it would hide what was known about it.
    evidence = {"sites": [{"site_id": build_target(profile_path(), "model.layers.0.mlp.up_proj").brief.site_id,
                           "reachable": False, "measured_l2": 0.013889, "control_l2": 0.0, "limit": 0.01,
                           "detail": "every equivalent GEMM exceeds the limit"}]}
    path = tmp_path / "reach.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    options = candidate_sites(profile_path(), reachability=load_reachability(path))
    by_path = {o.module_path: o for o in options}
    assert "model.layers.0.mlp.up_proj" in by_path          # still present
    assert by_path["model.layers.0.mlp.up_proj"].reachable is False
    assert options[-1].reachable is False                    # and ranked last
    assert "every equivalent GEMM" in (by_path["model.layers.0.mlp.up_proj"].reachability_detail or "")


def test_the_measured_evidence_file_reorders_the_real_ranking():
    # Given: the real evidence written from B20/B21. Without it the top pick is the 72-module GEMM
    # site that cannot pass; with it the top pick is the site measured to pass at exactly 0.0.
    evidence = load_reachability(RESULTS / "B5-profiler/site-reachability.json")
    assert evidence, "the measured evidence file must be present"
    blind = candidate_sites(profile_path())
    informed = candidate_sites(profile_path(), reachability=evidence)
    assert blind[0].module_path == "model.layers.0.mlp.up_proj"
    assert informed[0].reachable is True
    assert informed[0].module_path == "model.norm"


def test_absent_evidence_leaves_the_ranking_exactly_as_it_was():
    # Given: no measurement. Every site ranks on size alone, which is the behaviour before any of this
    # was measured -- so a task that never measures reachability is unaffected.
    assert candidate_sites(profile_path()) == candidate_sites(profile_path(), reachability={})


def test_a_missing_evidence_file_is_an_error_not_a_silent_empty(tmp_path):
    # Given: a declared path that does not exist. Treating it as "no evidence" would rank a site as
    # unknown when its evidence says it cannot be used.
    with pytest.raises(RunnerError, match="site-reachability evidence is missing"):
        load_reachability(tmp_path / "absent.json")


def test_malformed_evidence_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sites": {"not": "a list"}}), encoding="utf-8")
    with pytest.raises(RunnerError, match="list of per-site records"):
        load_reachability(path)


def test_evidence_is_keyed_by_site_id_so_it_cannot_describe_another_group(tmp_path):
    # Given: evidence for a site_id that is not in this profile. It must simply not apply to anything,
    # rather than attaching to whichever site happens to be nearby.
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"sites": [{"site_id": "pure-someothergroup", "reachable": False}]}),
                    encoding="utf-8")
    options = candidate_sites(profile_path(), reachability=load_reachability(path))
    assert all(o.reachable is None for o in options)


def test_a_forced_site_is_used_even_when_measured_unreachable():
    # Given: being asked for a site explicitly. The evidence must warn, not override -- reproducing an
    # earlier campaign has to stay possible.
    evidence = load_reachability(RESULTS / "B5-profiler/site-reachability.json")
    choices = select_sites({"ttft": profile_path()}, forced="model.layers.0.mlp.up_proj",
                           reachability=evidence)
    assert choices["ttft"].chosen.module_path == "model.layers.0.mlp.up_proj"
    assert choices["ttft"].chosen.reachable is False


def test_a_preparsed_profile_produces_the_identical_target() -> None:
    # Given: the ranking parses the profile once and passes it to every `build_target` call. That
    # shortcut must not change the target at all, or the ranking would measure one thing and the
    # campaign would search another.
    path = profile_path()
    fresh = build_target(path, "model.layers.0.input_layernorm")
    shared = build_target(path, "model.layers.0.input_layernorm", profile=load_profile(path))
    assert fresh.model_dump() == shared.model_dump()


def test_a_preparsed_target_still_fingerprints_the_file_on_disk() -> None:
    # Given: `profile_sha256` is what `require_target` re-checks before any candidate is admitted. It
    # must describe the file, not the object that was handed in.
    path = profile_path()
    target = build_target(path, "model.layers.0.input_layernorm", profile=load_profile(path))
    require_target(target, noise=False)   # raises if the fingerprint does not match the path


def test_every_ranked_site_is_one_build_target_accepts() -> None:
    # Given: eligibility must not be re-implemented here. Every option the ranking returns has to be
    # rebuildable through the real gate at its own module path.
    options = candidate_sites(profile_path())
    assert options
    for candidate in options[:5]:
        rebuilt = build_target(profile_path(), candidate.module_path)
        assert rebuilt.brief.site_id == candidate.site_id
        assert rebuilt.associated_cuda_us == candidate.associated_cuda_us


def test_ranking_is_descending_by_measured_cuda_time() -> None:
    options = candidate_sites(profile_path())
    values = [o.associated_cuda_us for o in options]
    assert values == sorted(values, reverse=True)


def test_sites_are_deduplicated_by_site_id() -> None:
    # Given: 36 sibling instances of one family all produce the SAME group. Listing that site 36 times
    # would make the ranking look like a list of instances rather than a list of choices.
    options = candidate_sites(profile_path())
    assert len({o.site_id for o in options}) == len(options)


def test_the_composite_opt_in_adds_options_and_never_removes_any() -> None:
    # Given: the opt-in is strictly a relaxation. Every leaf site must still be on the table, so
    # enabling it cannot silently drop a site that carries published numbers.
    leaf = candidate_sites(profile_path())
    both = candidate_sites(profile_path(), allow_composite=True)
    assert {o.site_id for o in leaf} <= {o.site_id for o in both}
    assert len(both) > len(leaf)


def test_the_composite_mlp_outranks_the_rmsnorm_leaf_on_the_ttft_profile() -> None:
    # Given: the measured reason for changing site at all. With wrappers admitted, the top-ranked site
    # must be a bigger one than the RMSNorm leaf the published row used.
    both = candidate_sites(profile_path(), allow_composite=True)
    leaf = candidate_sites(profile_path())
    assert both[0].associated_cuda_us > leaf[0].associated_cuda_us


def test_a_forced_site_must_still_be_eligible() -> None:
    # Given: reproducing an earlier campaign pins the site. Pinning an INELIGIBLE one has to fail
    # loudly rather than silently fall back to the ranked choice.
    profiles = {"ttft": profile_path()}
    with pytest.raises(RunnerError, match="not eligible"):
        select_sites(profiles, forced="model.layers.0.mlp")   # a wrapper, with the opt-in off


def test_a_forced_site_records_what_it_beat() -> None:
    # Given: a forced choice is still a choice. The ranking it overrode must stay on the record, or the
    # campaign cannot say what else was available.
    choices = select_sites({"ttft": profile_path()}, forced="model.layers.0.input_layernorm")
    assert choices["ttft"].chosen.module_path == "model.layers.0.input_layernorm"
    assert len(choices["ttft"].considered) > 1


def test_forcing_the_published_site_path_works_although_the_ranking_labels_it_differently() -> None:
    # Given: many module paths map to ONE site -- all 73 RMSNorm instances of the published site do --
    # and the ranking keeps only the shortest as its label. Resolving a forced path by matching that
    # label refused `model.layers.33.post_attention_layernorm`, the exact path the published campaign
    # ran, as ineligible. The forced path must be resolved by building at it.
    published = "model.layers.33.post_attention_layernorm"
    choices = select_sites({"ttft": profile_path()}, forced=published)
    assert choices["ttft"].chosen.module_path == published
    assert choices["ttft"].chosen.site_id == build_target(profile_path(), published).brief.site_id


def test_containment_is_computed_on_module_sets() -> None:
    outer = option("a.mlp", ("a.mlp", "b.mlp"))
    inner = option("a.mlp.down", ("a.mlp.down", "b.mlp.down"))
    assert contains(outer, inner)
    assert not contains(inner, outer)


def test_the_same_site_id_twice_is_not_an_overlap() -> None:
    # Given: `register_site` explicitly allows three goals to share one site; only DIFFERENT
    # overlapping groups are refused. Treating the shared case as a conflict would promote goals away
    # from a site they legitimately share.
    a = option("a.mlp", ("a.mlp",))
    assert not overlaps(a, a)


def test_a_contained_site_is_promoted_to_the_containing_one() -> None:
    # Given: one goal picks the MLP wrapper and another picks a child of it. `register_site` would
    # refuse the second, so the contained goal moves UP to the site that already covers its work.
    big = option("a.mlp", ("a.mlp",), us=100.0)
    small = option("a.mlp.down", ("a.mlp.down",), us=10.0)
    choices = {"ttft": SiteChoice(goal="ttft", chosen=big, considered=(big,)),
               "single": SiteChoice(goal="single", chosen=small, considered=(small, big))}
    settled = resolve_overlaps(choices)
    assert settled["single"].chosen.site_id == big.site_id
    assert "contained by" in (settled["single"].promoted_from or "")
    assert settled["ttft"].promoted_from is None


def test_overlapping_choices_that_cannot_be_promoted_fall_back_to_a_disjoint_site() -> None:
    # Given: two groups that intersect without either containing the other, and the containing site is
    # not on the smaller goal's list. It must move to its best DISJOINT option rather than to a site it
    # never measured.
    a = option("x", ("m1", "m2"), us=100.0)
    b = option("y", ("m2", "m3"), us=50.0)
    c = option("z", ("m9",), us=5.0)
    choices = {"ttft": SiteChoice(goal="ttft", chosen=a, considered=(a,)),
               "single": SiteChoice(goal="single", chosen=b, considered=(b, c))}
    settled = resolve_overlaps(choices)
    assert settled["single"].chosen.site_id == c.site_id
    assert "overlaps" in (settled["single"].promoted_from or "")


def test_no_disjoint_alternative_fails_loudly() -> None:
    a = option("x", ("m1", "m2"), us=100.0)
    b = option("y", ("m2",), us=50.0)
    choices = {"ttft": SiteChoice(goal="ttft", chosen=a, considered=(a,)),
               "single": SiteChoice(goal="single", chosen=b, considered=(b,))}
    with pytest.raises(RunnerError, match="no site disjoint"):
        resolve_overlaps(choices)


def test_resolution_keeps_the_largest_goal_on_its_own_choice() -> None:
    # Given: promotion must be deterministic and must never move the goal with the most work at stake.
    big = option("a.mlp", ("a.mlp",), us=100.0)
    small = option("a.mlp.down", ("a.mlp.down",), us=10.0)
    for order in ({"ttft": big, "single": small}, {"single": small, "ttft": big}):
        choices = {g: SiteChoice(goal=g, chosen=o, considered=(o, big)) for g, o in order.items()}
        settled = resolve_overlaps(choices)
        assert settled["ttft"].chosen.site_id == big.site_id
        assert settled["single"].chosen.site_id == big.site_id


def test_observed_modules_are_prefill_only_and_deepest_first() -> None:
    profile = load_profile(profile_path())
    modules = observed_modules(profile)
    assert modules
    depths = [m.count(".") for m in modules]
    assert depths == sorted(depths, reverse=True)
    prefill = {c.module_path for c in profile.calls if c.phase == "prefill"}
    assert set(modules) <= prefill
