"""Feasibility against the immutable original project baseline."""
import operator


def constraints_for_runs(runs, objective, anchor):
    from kernel_optimizer.project.measurement import metric
    operations = {"le": operator.le, "lt": operator.lt, "ge": operator.ge, "gt": operator.gt, "eq": operator.eq}
    constraints = []
    for constraint in objective.get("constraints", []):
        field = constraint["metric"]
        aggregate = constraint.get("rank_aggregation", "min" if constraint["op"] in {"ge", "gt"} else "max")
        order = constraint.get("aggregation_order", "samples_then_ranks")
        try:
            reference = metric(anchor, field, aggregate, order)
            limit = reference * float(constraint["baseline_ratio"]) if "baseline_ratio" in constraint else float(constraint["threshold"])
            values = [metric(run, field, aggregate, order) for run in runs]
            passed = all(operations[constraint["op"]](value, limit) for value in values)
            constraints.append({"metric": field, "values": values, "limit": limit, "original_reference": reference, "passed": passed})
        except (KeyError, TypeError, ValueError):
            constraints.append({"metric": field, "passed": False, "reason": "constraint measurement or definition unavailable"})
    return constraints
