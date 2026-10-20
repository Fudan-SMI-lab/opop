from copy import deepcopy

from examples.v6_validation.run_interaction import interaction


def test_interaction_retains_negative_combination_and_unmeasured_effects():
    def measured(gains):
        return {"evaluation": {"quality": "pass"}, "measurement": {"valid": True,
                "paired_gains": gains, "noise_bounds": [0.1, 0.2], "framework_source": "same", "constraints": []}}
    results = {"A": measured([2, 3]), "B": measured([1, 2]), "AB": measured([-4, -3])}
    report = interaction(results)
    assert report["gain_interaction"] == -7.5
    assert report["observed_combination_range"] == [-9, -6]
    assert report["range_expanded_by_endpoint_noise"][1] < 0
    bad = deepcopy(results)
    bad["B"]["evaluation"]["quality"] = "fail"
    assert interaction(bad)["status"] == "uninterpretable"
    bad = deepcopy(results)
    bad["AB"]["measurement"]["framework_source"] = "changed"
    assert interaction(bad)["status"] == "uninterpretable"
    assert interaction({"AB": results["AB"]})["status"] == "uninterpretable"
