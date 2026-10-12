"""A composite (wrapper) site: the opt-in that makes cross-KIND fusion requestable.

WHY THIS EXISTS
---------------
`build_target` originally refused any module with an observed child ("choose a narrow leaf pure-tensor
boundary, not a decoder/model wrapper"). That confined every site to ONE operator family repeated down
the model's depth, so a candidate could fuse a family's own stages (the accepted ttft kernel fused 8
ATen kernels into 2) but could never be asked to fuse across operator kinds.

`allow_composite=True` admits a wrapper as the site. Measured on the frozen Qwen3 profile, that turns
`model.layers.N.mlp` into a site whose associated CUDA time is 146,793 us against the RMSNorm leaf's
14,041 us -- 10.45x -- because a replacement of the wrapper subsumes `down_proj(act_fn(gate_proj(x)) *
up_proj(x))`: three GEMMs, an activation and an elementwise multiply behind one pure-tensor boundary.

WHAT MUST STILL REFUSE, AND WHY
-------------------------------
The leaf rule is one of four gates and the only one that was merely conservative. The purity gate is
load-bearing: `pure_tensors.tensor_leaves` raises on a `CacheValue`, so a module receiving the KV cache
can never be captured as a fixture -- and with no fixture there is no local proof, no `validate_local`
and no device-proven row. Admitting such a site would not widen the search, it would move the failure
further from its cause. So `self_attn`, the decoder layer and `model` are refused EVEN WITH the opt-in,
and the refusal says why.

These tests drive the real `build_target` against the real frozen profile written by the native
profiler, so the module tree, the family keys and the recorded sources are the production ones.
"""

import json
from pathlib import Path

import pytest

from examples.c3_qwen3.fast_target import build_target, subtree_is_pure
from examples.c3_qwen3.runner_records import RunnerError

RESULTS = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration"


def profile_path() -> Path:
    """The frozen short device profile, wherever this checkout keeps it.

    A whole profile is required: `build_target` parses `cuda_intervals` and `site_kernels`, and the
    locally mirrored copies of some campaigns are truncated. Any profile that parses is equivalent for
    these assertions, so the first one that loads is used.
    """
    for candidate in sorted(RESULTS.glob("**/short-device-profile-*.json")):
        try:
            data = json.loads(candidate.read_text(encoding="utf-8")).get("data") or {}
        except (ValueError, OSError):
            continue
        if data.get("valid") and data.get("cuda_intervals") and data.get("site_kernels"):
            return candidate
    pytest.skip("no complete short device profile is mirrored in this checkout")


def test_a_wrapper_is_refused_by_default_so_existing_targets_are_unchanged() -> None:
    # Given: the MLP block, which has observed children. With the opt-in off the original message must
    # still appear -- every target built before this change has to keep building identically.
    with pytest.raises(RunnerError, match="narrow leaf pure-tensor boundary"):
        build_target(profile_path(), "model.layers.0.mlp")


def test_a_leaf_site_is_identical_whether_the_flag_is_set_or_not() -> None:
    # Given: the RMSNorm leaf actually used by the published ttft row. The flag must not change a leaf
    # site at all, or enabling it would silently move a site that already carries published numbers.
    path = profile_path()
    off = build_target(path, "model.layers.0.input_layernorm")
    on = build_target(path, "model.layers.0.input_layernorm", allow_composite=True)
    assert off.brief.site_id == on.brief.site_id
    assert off.brief.module_paths == on.brief.module_paths
    assert off.associated_cuda_us == on.associated_cuda_us
    assert off.brief.reference_source == on.brief.reference_source


def test_the_mlp_block_builds_as_a_composite_site_of_the_wrappers_alone() -> None:
    # When: the opt-in admits the MLP wrapper.
    target = build_target(profile_path(), "model.layers.0.mlp", allow_composite=True)

    # Then: the group is the 36 wrapper instances -- one per decoder layer -- and NOT a wrapper plus
    # its own children. A nested group would have the binding replace a child whose original
    # `require_coverage` then expects to have run.
    assert len(target.brief.module_paths) == 36
    assert all(p.endswith(".mlp") for p in target.brief.module_paths)
    assert not any(a != b and a.startswith(b + ".") for a in target.brief.module_paths
                   for b in target.brief.module_paths)


def test_a_composite_brief_carries_its_children_or_the_kernel_cannot_be_written() -> None:
    # Given: a wrapper's own forward only names its children (`self.down_proj(self.act_fn(...))`), so
    # the reference is useless on its own -- a candidate cannot reproduce, let alone fuse, what it
    # cannot see.
    target = build_target(profile_path(), "model.layers.0.mlp", allow_composite=True)
    reference = target.brief.reference_source

    # Then: every child is named, with its own observed metadata, and its class is defined.
    for child in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        assert f"self.{child}:" in reference, f"composite brief omits child {child}"
    assert "Definition of Linear" in reference
    assert "Definition of SiLUActivation" in reference
    # The differing weight shapes have to be visible: gate/up project 2560->9728 and down comes back,
    # so a fused kernel cannot be written from one shape.
    assert "9728" in reference and "2560" in reference
    # And it is substantially richer than the leaf brief it replaces.
    leaf = build_target(profile_path(), "model.layers.0.input_layernorm")
    assert len(reference) > 4 * len(leaf.brief.reference_source)


def test_a_repeated_child_class_is_defined_once_not_once_per_instance() -> None:
    # Given: an MLP block has three `nn.Linear` children. Emitting the class per child repeated
    # torch's whole `Linear` body three times -- measured at 200 redundant lines of a 341-line brief,
    # which is prompt budget spent on nothing and, live, a 1500 s transport timeout on the first draw.
    reference = build_target(profile_path(), "model.layers.0.mlp",
                            allow_composite=True).brief.reference_source

    assert reference.count("Definition of Linear") == 1
    # `reset_parameters` appears once per emitted Linear body, so it counts the duplication directly.
    assert reference.count("def reset_parameters") == 1
    # All three instances are still individually described, which is what the dedup must not cost.
    for child in ("gate_proj", "up_proj", "down_proj"):
        assert f"self.{child}:" in reference


def test_a_composite_site_is_credited_with_the_kernels_it_would_subsume() -> None:
    # Given: `site_kernels` attributes a kernel to the scope that launched it, so an MLP wrapper's own
    # scope sees only its elementwise multiply while the three GEMMs sit under the children's scopes.
    # Crediting only the wrapper's own scope measured 8,582 us -- LESS than the RMSNorm leaf's 14,041 --
    # which inverts the ranking this number exists to provide.
    path = profile_path()
    composite = build_target(path, "model.layers.0.mlp", allow_composite=True)
    leaf = build_target(path, "model.layers.0.input_layernorm")

    assert composite.associated_cuda_us > leaf.associated_cuda_us
    # The children's GEMM kernels have to be in the declared set, since a fused replacement removes them.
    assert len(composite.cuda_kernel_names) > 1


@pytest.mark.parametrize("module", ["model.layers.0.self_attn", "model.layers.0", "model"])
def test_a_cache_carrying_subtree_is_refused_even_with_the_opt_in(module: str) -> None:
    # Given: these forwards receive a KV cache, which `tensor_leaves` raises on. Such a site could
    # never be captured as a fixture, so it could never carry a device proof -- admitting it would
    # produce a site that fails later, further from the cause.
    with pytest.raises(RunnerError, match="receives a KV cache"):
        build_target(profile_path(), module, allow_composite=True)


def test_purity_is_judged_from_the_recorded_forward_source() -> None:
    # Given: the profile strips CacheValue arguments before describing tensors, so a cache is invisible
    # in the recorded shapes and must be detected in the module's own observed source.
    from examples.c3_qwen3.fast_target import DeviceProfile

    data = DeviceProfile.model_validate_json(profile_path().read_bytes()).data
    assert subtree_is_pure(data, "model.layers.0.mlp") == (True, None)
    pure, why = subtree_is_pure(data, "model.layers.0.self_attn")
    assert pure is False and why is not None and "KV cache" in why
    # A module that was never observed cannot be judged pure by default.
    pure, why = subtree_is_pure(data, "model.layers.0.does_not_exist")
    assert pure is False and why is not None
