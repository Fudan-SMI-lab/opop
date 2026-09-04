"""Pre-flight audit: will the fixes queued since the L3:48 run started actually fire?

Several are driver-side or config-side, so they take effect only on the NEXT run and get
exactly one chance to be exercised. A typo or a wiring gap would waste a multi-hour L3:21
run before anyone noticed. This checks each one at the source level: no GPU, no imports of
torch, safe to run while an experiment is in flight.

Usage:  python scripts/preflight_next_run.py
Exit 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((ok, name, detail))


def src(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def main() -> int:
    orch = src("src/kernel_optimizer/control/orchestrator.py")
    base = src("src/kernel_optimizer/agents/base.py")
    mods = src("src/kernel_optimizer/agents/modules.py")
    report = src("src/kernel_optimizer/reporting/report.py")

    # --- 36e7d17: every agent is told the real device -------------------------------
    from kernel_optimizer.config import load_config
    cfg = load_config("configs/experiments_l3.yaml")
    check("device name reaches agents", cfg.device.name not in ("", "unknown"),
          f"device.name={cfg.device.name!r}")
    # The gate must NOT have moved: that is the user's decision, not a side effect.
    check("gate untouched (pass_frac 0.99)",
          cfg.evaluation.relaxed_pass_frac == 0.99 and cfg.evaluation.cosine_min == 0.99985,
          f"pass_frac={cfg.evaluation.relaxed_pass_frac} cosine_min={cfg.evaluation.cosine_min}")

    # --- ea9a3a2: reused measurements are journalled --------------------------------
    tune = orch.split("def _tune")[1].split("def _run_trial")[0]
    check("reuse journalling wired",
          '"reused_measurement": True' in tune and "measured_cache.get" in tune,
          "TRIAL_DONE carries reused_measurement for cache hits")

    # --- e56ea0f: repair events record the edit, not only the reasoning -------------
    rep_ev = orch.split('self.store.append("REPAIR_PRODUCED"')[1].split("})")[0]
    check("repair event records change_summary + source_sha",
          "change_summary" in rep_ev and "source_sha" in rep_ev, "both fields present")

    # --- fc51f18: candidate_id on agent calls --------------------------------------
    started = base.split('self.store.append("AGENT_CALL_STARTED"')[0][-900:]
    ok_generic = 'getattr(inputs, "candidate_id", None)' in started
    passed_everywhere = all(
        "candidate_id=" in seg[:700]
        for cls in ("RepairInputs(", "AnalystInputs(")
        for seg in orch.split(cls)[1:]
    )
    helper = orch.split("def _parameterize_agent_call")[1].split("\n    def ")[0]
    check("candidate_id attribution wired",
          ok_generic and passed_everywhere and "candidate_id=cand_id" in helper,
          "read generically in base.invoke; passed by repair/analyst/parameterizer sites")

    # --- 4030458: K skips hard-edge knobs and no-op expansions ----------------------
    sel = orch.split("def boundary_knobs_to_expand")[1].split("\n@dataclass")[0]
    exp = orch.split("def _maybe_expand_space")[1].split("def _expand_directive_text")[0]
    check("K skips hard hardware edges",
          "HARD_EDGE" in sel and "_at_hard_edge" in sel,
          "NUM_WARPS/NUM_STAGES at 1 are not requested in the min direction")
    check("K rejects a no-op expansion before re-tuning",
          "no_new_choices" in exp
          and exp.find("no_new_choices") < exp.find("self._tune("),
          "delivered domains compared against previous; bails out pre-re-tune")

    # --- 7db5bf1: failed hypotheses persisted --------------------------------------
    rw = orch.split("def _rewrite_round")[1].split("def _do_rewrite")[0]
    check("HYPOTHESES_FAILED journalled",
          'self.store.append("HYPOTHESES_FAILED"' in rw and "best_after >= best_before" in rw,
          "fires when a round fails to improve the family")

    # --- bea65f8: report is honest on an unfinished run -----------------------------
    check("report reconstructs an unfinished run",
          "_reconstruct_summary" in report and "provisional" in report,
          "no longer claims 'no correct candidate survived' mid-run")
    check("report marks the expanded space",
          "expanded_spaces" in report and "(expanded space)" in report,
          "a K re-tune is distinguishable from a duplicate line")

    # --- c08041f / c462a93: repair feedback quality ---------------------------------
    par = orch.split("def _parameterize_with_repair")[1].split("def _tune")[0]
    check("repair history pairs a diagnosis with what it caused",
          'repair_history[-1]["failure_detail"] = verdict.detail' in par,
          "detail filled in on the NEXT iteration")
    # The SPACE_REJECTED append specifically -- not every verdict.detail in the function.
    # repair_history entries use a deliberate [:600] cap (c08041f) and must not be
    # mistaken for the head-truncation bug this check watches for.
    rej_appends = [seg for seg in orch.split("self.store.append(")
                   if "REJECTED" in seg[:60] and "verdict.detail" in seg[:900]]
    check("rejection events keep the verdict tail",
          bool(rej_appends)
          and all("error_excerpt(verdict.detail" in s[:900] for s in rej_appends)
          and not any("verdict.detail[:" in s[:900] for s in rej_appends),
          f"{len(rej_appends)} verdict-bearing rejection append(s), all tail-preserving")

    # --- prompts actually reach the agent ------------------------------------------
    check("repair prompt carries ref + prior attempts",
          "ref_line" in mods and "prior_line" in mods and "_rejected_repairs_doc" in mods,
          "seeded and rendered")

    # --- minimal witness = the fp16 corner (docs/finding-minimal-witness-forces-fp16) --
    val = src("src/kernel_optimizer/paramspace/validation.py")
    wit = val.split("for label, params in ((\"default\", default_params)")[-1]
    check("out-of-range cheap corner falls back, not rejects",
          "_next_witness" in wit
          and 'if label == "minimal" and _looks_out_of_range(result):' in wit
          and "def _next_witness" in val and "def _looks_out_of_range" in val,
          "a minimal witness failing with the OVERFLOW signature tries the next feasible "
          "config; an ordinary mismatch still rejects")
    check("witness fallback is bounded",
          "max_witness_retries" in val
          and "attempted >= self.max_witness_retries" in val,
          "each retry is a real GPU quick test, so the walk cannot be exhaustive")
    check("witness rejection names the failing config",
          "[{label} witness config {params.values}]" in val
          and "DEFAULT config passed" in val,
          "repair is told it is the cheap corner, not a globally broken kernel")
    # The fallback must not weaken acceptance: two distinct sources must still pass.
    check("anti-inertness preserved",
          'reason="inert_space"' in val
          and "if mat_src == witness_sources_default:" in wit,
          "an alternative identical to the default is not accepted as a second witness")

    width = max(len(n) for _, n, _ in RESULTS)
    bad = 0
    for ok, name, detail in RESULTS:
        if not ok:
            bad += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name:{width}}  {detail}")
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
