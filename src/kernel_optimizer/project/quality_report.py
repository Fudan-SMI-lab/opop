"""Keep output quality, actual execution coverage and worker failures distinct."""


def quality_and_coverage(runs):
    ranks = [rank for run in runs for rank in run.get("ranks", [])]
    checks = [check for rank in ranks for check in rank.get("result", {}).get("checks", [])]
    failed = any(not c["passed"] for c in checks) or any(
        rank.get("result", {}).get("math_policy_unchanged") is False for rank in ranks)
    known = bool(runs) and all(run.get("status") == "passed" and run.get("ranks") for run in runs) and all(
        rank.get("status", "passed") == "passed" and rank.get("result", {}).get("checks") for rank in ranks)
    quality = "fail" if failed else "pass" if known else "unknown"
    missing = []
    for index, run in enumerate(runs):
        for rank in run.get("ranks", []):
            for bundle, coverage in rank.get("result", {}).get("coverage", {}).items():
                absent = set(coverage["required"]) - {c for c, n in coverage["counts"].items() if n > 0}
                if absent:
                    missing.append({"endpoint": index, "rank": rank["rank"], "bundle": bundle, "calls": sorted(absent)})
    return quality, missing
