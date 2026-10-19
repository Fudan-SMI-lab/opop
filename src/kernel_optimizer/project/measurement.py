"""Project acceptance uses paired clean samples and fixed original constraints."""
from __future__ import annotations

import statistics
from kernel_optimizer.project.constraints import constraints_for_runs


def valid_run(run):
    return run["status"] == "passed" and bool(run["ranks"]) and all(r.get("result", {}).get("valid", False) for r in run["ranks"])


def metric(run, name, reducer="max"):
    values = [r["result"]["metrics"][name] for r in run["ranks"]]
    reducers = {"max": max, "min": min, "sum": sum, "mean": statistics.mean}
    return float(reducers[reducer](values))


def paired_decision(pairs, objective, anchor):
    direction, name = objective["direction"], objective["name"]
    reducer = objective.get("rank_aggregation", "max" if direction == "minimize" else "min")
    threshold = float(objective.get("improvement_threshold", 0))
    reasons, gains, noise, constraints = [], [], [], []
    if not all(valid_run(run) for pair in pairs for run in pair.values()):
        return {"valid": False, "reasons": ["invalid_paired_endpoint"], "objective": None, "reference": None}
    before = [metric(p["parent"], name, reducer) for p in pairs]
    after = [metric(p["candidate"], name, reducer) for p in pairs]
    for pair, a, b in zip(pairs, before, after, strict=True):
        gain = a-b if direction == "minimize" else b-a
        gains.append(gain)
        deviations = []
        for run in pair.values():
            for rank in run["ranks"]:
                samples = rank["result"].get("metric_samples", {}).get(name)
                if samples is None:
                    samples = [s[name] for s in rank["result"]["samples"]]
                center = statistics.median(samples)
                deviations.append(statistics.median(abs(v-center) for v in samples))
        bound = float(objective.get("noise_factor", 2)) * max(deviations, default=0)
        noise.append(bound)
        if gain <= max(threshold, bound):
            reasons.append("paired_gain_not_above_noise")
    constraints = constraints_for_runs([pair["candidate"] for pair in pairs], objective, anchor)
    if any(not row["passed"] for row in constraints):
        reasons.append("original_baseline_constraint_failed")
    return {"valid": True, "objective": statistics.median(after), "reference": statistics.median(before),
            "parent_values": before, "candidate_values": after, "paired_gains": gains, "noise_bounds": noise,
            "constraints": constraints, "reasons": sorted(set(reasons)),
            "protocol": "AB then BA in fresh workers; median within each endpoint; every pair must exceed its MAD-derived noise bound"}
