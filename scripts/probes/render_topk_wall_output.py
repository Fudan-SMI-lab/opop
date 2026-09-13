"""Render the K=1 and K=3 outputs side by side so a human can read what actually changed."""
import sys
from pathlib import Path

from kernel_optimizer.evaluation import wall_attribution as wa
from kernel_optimizer.models.reports import ParamStat, TuningStats
from kernel_optimizer.reporting.wall_report import wall_lines


def wall():
    stats = TuningStats(candidate_id="c", space_id="s", n_complete=10, n_fail=0,
                        param_stats=[ParamStat(name="BLOCK_N", best_value="64",
                                               latency_by_value={"16": 3.82, "32": 3.38,
                                                                 "64": 3.18})])
    w = wa.find_walls(stats, [{"BLOCK_N": 128}])[0]
    w.limit = 101376
    return w


out = []

# --- K=1, today's behaviour
w = wall()
w.verdict = "attributed"
w.max_shared, w.over_ratio = 122880, 122880 / 101376
w.second_origin = "not_attributed"
w.origin_verdicts = {"theta_star": "attributed", "default": "not_attributed"}
w.origin_max_shared = {"theta_star": 122880, "default": 40960}
out.append("=" * 100)
out.append("K=1  (probe_top_k default) -- must read exactly as before top-K existed")
out.append("=" * 100)
out.append(wa.for_prompt([w]) or "(none)")

# --- K=3, wall holds at 2 of 3
w3 = wall()
w3.verdict = "attributed"
w3.max_shared, w3.over_ratio = 122880, 122880 / 101376
w3.second_origin = "not_attributed"
w3.origin_verdicts = {"theta_star": "attributed", "theta_top2": "attributed",
                      "theta_top3": "not_attributed", "default": "not_attributed"}
w3.origin_max_shared = {"theta_star": 122880, "theta_top2": 119808, "theta_top3": 65536,
                        "default": 40960}
out.append("")
out.append("=" * 100)
out.append("K=3, wall holds at 2 of the 3 fastest points")
out.append("=" * 100)
out.append(wa.for_prompt([w3]) or "(none)")

# --- K=3, holds ONLY away from theta* -- previously discarded entirely
w4 = wall()
w4.verdict = "not_attributed"
w4.max_shared, w4.over_ratio = 65536, 65536 / 101376
w4.origin_verdicts = {"theta_star": "not_attributed", "theta_top2": "attributed",
                      "theta_top3": "attributed"}
w4.origin_max_shared = {"theta_star": 65536, "theta_top2": 122880, "theta_top3": 118784}
out.append("")
out.append("=" * 100)
out.append("K=3, fits at theta* but refused at #2 and #3 -- previously DISCARDED")
out.append("=" * 100)
out.append(wa.for_prompt([w4]) or "(none)")

# --- report section
def ev(walls, counts):
    return {"type": "RESOURCE_WALL_ATTRIBUTED",
            "payload": {"candidate_id": "cand-dc87a93a", "n_refused_configs": 12,
                        "walls_found": len(walls), "walls_probed": len(walls),
                        "walls_worthless": 0, "n_probes": 12, "probe_total_s": 5.9,
                        "counts": counts, "walls": [x.payload() for x in walls]}}


out.append("")
out.append("=" * 100)
out.append("REPORT SECTION, K=3 (one wall at 2/3, one that fits at theta* only)")
out.append("=" * 100)
out.extend(wall_lines([ev([w3, w4], wa.summarize([w3, w4]))]))

Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")
