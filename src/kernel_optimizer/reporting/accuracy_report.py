"""2b(2) + 2d: the per-dimension declaration-accuracy table, the four miss shapes, and
`under-converged`.

WHY THIS EXISTS, AND WHY PER-DIMENSION IS THE POINT. The pooled ledger accuracy (71.6% over 81
declarations on box 2) is an average over eight dimensions that do NOT behave alike. Split, the two
boxes agree on the ORDER: `candidate_aten_ops` 88%/100% against `shared_bytes` 36%/20% -- a spread of
52 and 80 percentage points. Reporting only the pooled number hides the one fact that makes
"per-dimension" more than a presentation choice: the agent is reliable about how many aten calls it
will remove and unreliable about shared memory, and those two claims should not be believed equally.

The dimension COVERAGE is balanced (each dimension is ~12% of declarations: box 2's 136 = 17x8, the
A800's 48 = 6x8), so the spread is not an artefact of some dimensions having fewer samples.

THE CONDITION THAT MUST TRAVEL WITH EVERY NUMBER HERE. These accuracies hold only over the hypotheses
the agent CHOSE to implement -- measured at 8-15% of those proposed (19 of 128 and 6 of 78), with a
preference skew (H1 22% against H3 5%). So the honest sentence is "accurate about shared_bytes among
the ideas it was willing to write", never "the agent understands shared memory poorly". The section
prints the implementation rate next to the table for exactly this reason.

THE FOUR MISS SHAPES, because "miss" pools things with different fixes:

  dead lever          declared it would move, it stayed flat. The action did not reach the dimension.
  unforeseen effect   moved, never declared (`dimensions_unpredicted`). A blind spot, not an error.
  wrong direction     declared up, went down (or the reverse). The model of the mechanism is inverted.
  under-converged     2d. A "dead lever" where the child's tuning did not sample enough to have
                      moved the dimension -- so the flat reading is OUR measurement's limit, not the
                      agent's mistake. Measured at 2 of 11 candidates, so it is not negligible; the
                      only stable form of the finding is that dead-lever rows carry roughly 3x the
                      under-convergence of hit rows.

2d IS DELIBERATELY CONSERVATIVE. `under-converged` is a RE-LABEL of a miss, never a promotion to hit:
the declaration still was not confirmed. It is threshold-sensitive (21/50/71% of dead levers depending
on the cut) and does not reproduce across boxes, so the label carries its own criterion inline and the
section says the count is a lower bound on doubt rather than a corrected accuracy.

Read-only: consumes `EXPECTATIONS_RECONCILED` and `TRIAL_DONE`, so `report` regenerates it from
events.jsonl alone and it applies retroactively to every finished run.
"""

from __future__ import annotations

from typing import Any

# A dead lever is only re-labelled `under-converged` when the child's tuning plainly had not settled.
# The criterion is a property of OUR sampling, not of the declaration: the child's own best was still
# improving at the very end of its pass, so the budget ran out before the search stopped moving.
#
# WHY THERE IS NO TRIAL-COUNT CONDITION. The first version required "<= 20 completed trials" as well.
# Measured across the three step-3 runs, the dead-lever children have 27 to 70 completed trials, so
# that condition could NEVER fire -- the label was structurally dead while the report printed
# "0 suspected under-measured" as though it had looked. An unreachable branch is not a safeguard, and
# this one asserted a negative it had never tested. Trial COUNT was the wrong proxy anyway: 69 trials
# that stopped improving at 47% is converged, and 29 trials still improving at 93% is not.
#
# The surviving condition is the late-refresh one, which does fire: `cand-d8f78fc7 peak_alloc_bytes`
# (29 trials, last best-refresh at 93%) and `cand-705470ff occupancy` (68 trials, at 94%) are exactly
# the cases the label is for -- the tuner was still finding better points when the budget ended, so
# "the dimension did not move" is our measurement's limit rather than the agent's error.
_UNDERCONVERGED_LATE_IMPROVE_FRAC = 0.90


def _ev(e: Any, name: str) -> Any:
    return e.get(name) if isinstance(e, dict) else getattr(e, name, None)


def _pct(hits: int, total: int) -> str:
    return f"{100.0 * hits / total:.0f}%" if total else "—"


def _rows(events: Any) -> list[dict]:
    out = []
    for e in events:
        if _ev(e, "type") != "EXPECTATIONS_RECONCILED":
            continue
        p = _ev(e, "payload") or {}
        rec = p.get("reconciliation") or {}
        if rec:
            out.append({"candidate_id": p.get("candidate_id"), "family_id": p.get("family_id"),
                        "round": p.get("round"), "rec": rec})
    return out


def _child_convergence(events: Any) -> dict[str, dict]:
    """Per candidate: completed-trial count and whether its best was still improving late.

    Used only to decide `under-converged`. Deliberately computed from TRIAL_DONE rather than from any
    convergence verdict: the verdict is about a FAMILY, and the question here is whether one
    candidate's tuning pass had settled.
    """
    trials: dict[str, list[tuple[float, float]]] = {}
    for e in events:
        if _ev(e, "type") != "TRIAL_DONE":
            continue
        t = (_ev(e, "payload") or {}).get("trial") or {}
        cid = t.get("candidate_id")
        if not cid or t.get("status") != "complete":
            continue
        lat = t.get("latency_ms") or {}
        med = lat.get("median")
        ms = med if isinstance(med, (int, float)) and med > 0 else lat.get("mean")
        if isinstance(ms, (int, float)):
            trials.setdefault(str(cid), []).append((float(_ev(e, "ts") or 0.0), float(ms)))
    out: dict[str, dict] = {}
    for cid, xs in trials.items():
        xs.sort()
        n = len(xs)
        best = None
        last_improve_at = 0
        for i, (_, ms) in enumerate(xs):
            if best is None or ms < best:
                best, last_improve_at = ms, i
        out[cid] = {
            "n": n,
            # "still improving late" = the last improvement landed in the final quarter of the pass.
            "late": bool(n and last_improve_at >= _UNDERCONVERGED_LATE_IMPROVE_FRAC * (n - 1)),
            "last_improve_frac": (last_improve_at / (n - 1)) if n > 1 else 0.0,
        }
    return out


def accuracy_lines(events: Any) -> list[str]:
    """The section. Returns [] when no round was ever reconciled."""
    entries = _rows(events)
    if not entries:
        return []

    conv = _child_convergence(events)

    # Per-dimension tallies, plus the miss shapes.
    per_dim: dict[str, dict[str, int]] = {}
    unforeseen: dict[str, int] = {}
    wrong_dir: list[str] = []
    dead_lever: list[tuple[str, str, bool]] = []   # (candidate, dimension, under_converged)
    unmeasured: dict[str, int] = {}

    for ent in entries:
        rec = ent["rec"]
        cid = str(ent.get("candidate_id") or "?")
        cinfo = conv.get(cid, {})
        under = bool(cinfo and cinfo.get("late"))
        for name in (rec.get("dimensions_unpredicted") or []):
            unforeseen[str(name)] = unforeseen.get(str(name), 0) + 1
        for name in (rec.get("dimensions_unmeasured") or []):
            unmeasured[str(name)] = unmeasured.get(str(name), 0) + 1
        for row in (rec.get("per_dimension") or []):
            dim = str(row.get("dimension") or "?")
            d = per_dim.setdefault(dim, {"hit": 0, "miss": 0, "vacuous": 0, "unmeasured": 0,
                                         "dead": 0, "flip": 0, "under": 0})
            match = str(row.get("match") or "")
            if match in d:
                d[match] += 1
            if match != "miss":
                continue
            expected, actual = str(row.get("expected")), str(row.get("actual"))
            if actual == "flat":
                d["dead"] += 1
                dead_lever.append((cid, dim, under))
                if under:
                    d["under"] += 1
            elif expected in ("up", "down") and actual in ("up", "down") and expected != actual:
                d["flip"] += 1
                wrong_dir.append(f"`{cid}` {dim}: 说 {expected}、实测 {actual}")

    lines = ["## 逐维度声明准确率与 miss 的四种形状(2b② / 2d)\n"]

    tot_hit = sum(d["hit"] for d in per_dim.values())
    tot_miss = sum(d["miss"] for d in per_dim.values())
    tot_vac = sum(d["vacuous"] for d in per_dim.values())
    graded = tot_hit + tot_miss
    lines.append(
        f"- 共 {len(entries)} 个已对账的轮次;可判定声明 {graded} 条"
        f"(hit {tot_hit} / miss {tot_miss}),空洞 {tot_vac} 条"
        f"{'、无读数 ' + str(sum(unmeasured.values())) + ' 条' if unmeasured else ''};"
        f"**合计准确率 {_pct(tot_hit, graded)}**")
    lines.append(
        "- **合计数字会掩盖唯一重要的事实**:8 个维度并不同质。"
        "实测两箱同序 —— `candidate_aten_ops` 88%/100% 对 `shared_bytes` 36%/20%,"
        "**跨度 52 / 80 个百分点**,而各维度的声明数基本相等(各约 12%)"
        "⇒ 跨度不是样本量噪声。下表按维度拆开。")
    lines.append("")

    header = "| 维度 | 可判定 | hit | 准确率 | 死杠杆 | 其中疑似**未测够** | 方向反了 | 空洞 |"
    lines.extend([header, "|---|---|---|---|---|---|---|---|"])
    for dim, d in sorted(per_dim.items(),
                         key=lambda kv: (-(kv[1]["hit"] + kv[1]["miss"]), kv[0])):
        g = d["hit"] + d["miss"]
        lines.append(
            f"| `{dim}` | {g} | {d['hit']} | **{_pct(d['hit'], g)}** | {d['dead']} | "
            f"{d['under']} | {d['flip']} | {d['vacuous']} |")
    lines.append("")

    # The condition that must travel with the table.
    lines.append(
        "**每个准确率都只在「agent 愿意实现的那些假设」上成立。** 实测 analyst 提出的假设只有 "
        "**8–15%** 被实现(19/128 与 6/78),且有偏好倾斜(H1 22% 对 H3 5%)。"
        "所以可写的是「在它愿意写的想法里,它对 shared_bytes 的声明只有 36%/20% 命中」,"
        "**不能**写成「agent 不理解共享内存」—— 后者是过度概括。")
    lines.append("")

    lines.append("**miss 的四种形状**(合并成一个数字会让修法失去指向):")
    lines.append(
        f"- **死杠杆**(说会动、实测没动):{len(dead_lever)} 条。"
        "动作没有真正触及该维度。")
    n_under = sum(1 for _, _, u in dead_lever if u)
    if dead_lever:
        lines.append(
            f"- **其中 {n_under} 条疑似「未测够」(2d)**:该子代的调参在预算耗尽时**仍在刷新最优**"
            f"(最后一次最优刷新落在该轮 trial 的最后 "
            f"{100 - int(_UNDERCONVERGED_LATE_IMPROVE_FRAC * 100)}% 内)"
            "⇒ 那个「没动」是**我们测量的下限**,不是 agent 判断错。"
            "**这是重新贴标签,不是改判为 hit** —— 声明仍未被确认。"
            "判据只看「是否还在改善」而**不看 trial 数**:实测死杠杆子代有 27–70 个完成 trial,"
            "任何 trial 数阈值都只会让这个标签永远不触发,而 69 个 trial 在 47% 处就停止改善是收敛的、"
            "29 个 trial 到 93% 还在改善不是。"
            "该标签对阈值敏感(实测 21/50/71% 随切法变化)、不跨箱复现;"
            "唯一稳的结论是死杠杆行的未收敛率约为 hit 行的 **3 倍**。"
            "⇒ 把它读成「对该 miss 存疑的下界」,不要读成修正后的准确率。")
    if unforeseen:
        top = ", ".join(f"`{k}`×{v}" for k, v in
                        sorted(unforeseen.items(), key=lambda kv: -kv[1])[:5])
        lines.append(
            f"- **未预料的副作用**(动了但没声明):{sum(unforeseen.values())} 条 —— {top}。"
            "这是盲点而不是错误:「你没想到共享内存」与「你对它判断错了」是两句不同的话。")
    else:
        # NOT "0 unforeseen effects". Whether that shape can occur at all depends on how many
        # dimensions the agent declared: if it declares all 8 every round, an unmentioned dimension
        # cannot exist, and printing "0" would claim a finding where the shape was structurally
        # unavailable. Measured across the three step-3 runs: 8 of 8 declared in every single round.
        n_rounds = len(entries)
        full = sum(1 for e in entries
                   if len([r for r in (e["rec"].get("per_dimension") or [])
                           if str(r.get("match")) != "unpredicted"]) >= len(per_dim))
        if n_rounds and full == n_rounds:
            lines.append(
                f"- **未预料的副作用**:本 run **不可能**出现 —— {n_rounds}/{n_rounds} 个轮次都对"
                f"全部 {len(per_dim)} 个维度做了声明,没有「未提及」的维度。"
                "这不是「查过了没有」,而是该形状在此 run 里没有发生的空间。")
        else:
            lines.append(
                f"- **未预料的副作用**:0 条(其中 {full}/{n_rounds} 个轮次声明了全部维度,"
                "在那些轮次里这个形状无法出现)。")
    if wrong_dir:
        lines.append(f"- **方向反了**(说 up 实测 down 或反之):{len(wrong_dir)} 条 —— "
                     + "; ".join(wrong_dir[:4]))
    else:
        lines.append("- **方向反了**:0 条。")
    if unmeasured:
        lines.append(
            f"- (另有**无读数** {sum(unmeasured.values())} 条:声明了但有一侧没有测量值。"
            "既不是空洞也不是 miss,不进准确率的分母。)")
    lines.append("")
    lines.append(
        "**为什么这张表服务 contribution 2 的动机**:它说明「逐维度」不是表述方式而是必需 —— "
        "在 52–80 个百分点的跨度下,任何把八维合成一个可靠性数字的做法都会同时高估 shared_bytes "
        "并低估 aten_ops。")
    return lines
