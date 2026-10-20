from types import SimpleNamespace as Object

from kernel_optimizer.project.aggregation import lineage_resolver


def test_survival_shared_membership_and_replaced_boundary_have_distinct_lineage():
    previous = Object(groups=(Object(id="direct", calls=("survivor", "retired")),
                              Object(id="shared", calls=("survivor",)),
                              Object(id="fused", calls=("old-child",))))
    calls = {"survivor": {"parents": []}, "new-child": {"parents": ["new-boundary"]},
             "new-boundary": {"parents": []}, "unrelated": {"parents": ["unknown"]}}
    resolver = lineage_resolver(previous, calls,
        [Object(calls=("new-boundary",), groups=("fused", "absent-from-previous-registry"))])
    assert resolver(["survivor"]) == ("direct", "shared")
    assert resolver(["new-child"]) == ("fused",)
    assert resolver(["new-boundary", "survivor"]) == ("direct", "fused", "shared")
    assert resolver(["unrelated"]) == ()
