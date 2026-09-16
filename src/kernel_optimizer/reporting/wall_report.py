"""The 2e report section: which knob each shared-memory wall belongs to, and what it costs.

Reads only `RESOURCE_WALL_ATTRIBUTED` events, so `report` still regenerates purely from
events.jsonl.

Three things this section must make visible, because each of them has already misled a reader of the
raw numbers once:

1. THE VERDICT IS CONDITIONAL. Measured on box 2, the same ablation attributes 6 of 6 walls when it
   starts from the candidate's optimum and 1 of 6 when it starts from the space's default corner. A
   line reading "BLOCK_M is capped by shared memory" is therefore false as written; it is capped at
   THIS point, given what the other knobs are set to. Every row here carries the origin, and a row
   whose second origin disagrees says so in the row rather than in a footnote.

2. A WALL IS NOT AUTOMATICALLY AN OPPORTUNITY. Three of box 2's six walls have latency getting WORSE
   toward the wall, the worst at -54.8%: real walls, worth nothing. They are reported separately, not
   filtered away, because "no wall" and "six worthless walls" are different states and the
   difference is the whole reason the slope filter exists.

3. HOW FAR OVER THE LIMIT DECIDES WHETHER A REWRITE CAN HELP. 1.21x is one knob step; 6.48x is not
   reachable by freeing shared memory at all. Reporting "infeasible" without the ratio invites a
   rewrite that cannot succeed.
"""

from __future__ import annotations

from typing import Any


def _rows(events: Any) -> list[dict]:
    out = []
    for ev in events:
        t = ev.get("type") if isinstance(ev, dict) else getattr(ev, "type", None)
        if t != "RESOURCE_WALL_ATTRIBUTED":
            continue
        p = (ev.get("payload") if isinstance(ev, dict) else getattr(ev, "payload", None)) or {}
        out.append(p)
    return out


def _fmt_val(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _ev(ev: Any, name: str) -> Any:
    return ev.get(name) if isinstance(ev, dict) else getattr(ev, name, None)


def _same_value(a: Any, b: Any) -> bool:
    """Compare a knob value across records where one side may be `128` and the other `128.0`.

    The wall record's `refused_value` arrives as a float (it comes from the numeric coercion in
    `find_walls`), while a trial's `params.values` holds whatever the space declared. A plain `==`
    silently answers False for the very case this section exists to report, and the failure looks
    exactly like a real negative: "the rewrite did not free it".
    """
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def _origin_counts(w: dict) -> tuple[int, int]:
    """`(attributed, probed)` over HIGH-PERFORMANCE origins, for a wall payload.

    Falls back to theta*'s own verdict for a record written before top-K existed, where the only
    origin ever recorded IS theta*. A fallback of `(0, 0)` would make every finished run's walls read
    as "attributed at 0 points" -- the shape of defect where a new reader turns old evidence into a
    negative.
    """
    verdicts = w.get("origin_verdicts")
    if isinstance(verdicts, dict) and verdicts:
        fast = {k: v for k, v in verdicts.items() if k != "default"}
        return sum(1 for v in fast.values() if v == "attributed"), len(fast)
    return (1 if w.get("verdict") == "attributed" else 0), (1 if w.get("verdict") else 0)


def _delivered(w: dict) -> bool:
    """Whether this wall was rendered for the rewriter.

    Must mirror `Wall.is_attributed`: a wall refused at the 2nd and 3rd fastest points but accepted at
    the very fastest one IS delivered, and A3 asking about a different set than the one delivered
    would answer a question nobody asked.
    """
    return _origin_counts(w)[0] >= 1


def _attributing_footprint(w: dict) -> tuple[Any, Any, str | None]:
    """`(max_shared, over_ratio, origin_name)` from an origin where the compiler REFUSED.

    Mirrors `Wall.attributing_footprint`, and for the same reason: a row that claims a wall must not
    quote a footprint measured where the configuration FIT. Without this, a wall refused at the 2nd
    and 3rd fastest points but accepted at the 1st renders as "编译器要求 65536,上限 101376,0.65x"
    -- three numbers that say the opposite of the verdict beside them, which is worse than no row.

    theta* first when it attributed, so a row that was correct before top-K reads identically.
    """
    limit = w.get("limit")
    verdicts = w.get("origin_verdicts") or {}
    shares = w.get("origin_max_shared") or {}
    if not isinstance(verdicts, dict) or not verdicts:
        return w.get("max_shared"), w.get("over_ratio"), None
    names = ["theta_star"] + [n for n in verdicts if n not in ("theta_star", "default")]
    for name in names:
        if verdicts.get(name) != "attributed":
            continue
        ms = shares.get(name)
        if not isinstance(ms, (int, float)):
            continue
        ratio = (float(ms) / float(limit)) if isinstance(limit, (int, float)) and limit else None
        return int(ms), ratio, (None if name == "theta_star" else name)
    return w.get("max_shared"), w.get("over_ratio"), None


def freed_lines(events: Any) -> list[str]:
    """A3: for each ATTRIBUTED wall, did a later rewrite in the same lineage FREE that dimension?

    This is the section the 2e report was missing. Everything above says which knob a wall belongs
    to and what it would be worth; none of it says whether the rewrite that the wall text steered
    actually bought the range back. That is the last verb in the mechanism, and it is the one an
    ablation of the wall-attribution arm has to answer.

    FREED means: a DESCENDANT of the walled candidate has a COMPLETED trial holding the knob at the
    value the compiler refused. Descendant, not "same family": a family can contain candidates that
    are not children of the walled one, and those say nothing about this wall.

    Four outcomes, not two, because two of them are routinely mistaken for failure:

      FREED           the compiler accepted what it refused before.
      仍被拒          the child still gets `infeasible_shared_memory` at that value.
      维度已消失      the child never samples that knob at all -- the rewriter restructured the kernel
                      and the knob ceased to exist. Reported apart from both verdicts on purpose:
                      counted as failure it slanders a rewrite that may have removed the constraint
                      by removing the tile loop; counted as success it credits one for deleting the
                      evidence.
      未曾尝试        the knob exists but no trial sampled that value. TPE is not a uniform sampler,
                      so absence is not refusal -- already recorded once as
                      `trial-failure-rate-is-not-space-density`.

    `shared_bytes` is reported AT THE WALL'S VALUE, not at the child's optimum, and that distinction
    is the whole reason this reads as evidence. Measured on arm 3: the two children of the one
    attributed wall report 36864 B and 73728 B at their own optima -- the second identical to its
    parent's, which invites "the footprint did not change". At BLOCK_N=128 itself both fell (77824 B,
    and 45056..98304 B) from the refused configuration's 122880 B, under the 101376 B limit. The
    optimum sits whereever the tuner preferred, so its footprint says nothing about feasibility at
    the wall.
    """
    payloads = _rows(events)
    attributed: list[tuple[str, dict]] = []
    for p in payloads:
        for w in (p.get("walls") or []):
            if _delivered(w):
                attributed.append((str(p.get("candidate_id") or "?"), w))
    if not attributed:
        return []

    # Lineage and trials, both read from the same events the rest of the report replays from.
    children: dict[str, list[str]] = {}
    for ev in events:
        if _ev(ev, "type") != "CANDIDATE_REGISTERED":
            continue
        c = (_ev(ev, "payload") or {}).get("candidate") or {}
        cid = c.get("candidate_id")
        if not cid:
            continue
        for parent in (c.get("parent_ids") or []):
            children.setdefault(str(parent), []).append(str(cid))

    trials: dict[str, list[dict]] = {}
    for ev in events:
        if _ev(ev, "type") != "TRIAL_DONE":
            continue
        t = (_ev(ev, "payload") or {}).get("trial") or {}
        cid = t.get("candidate_id")
        if cid:
            trials.setdefault(str(cid), []).append(t)

    def descendants(cid: str) -> list[str]:
        out: list[str] = []
        stack = list(children.get(cid, []))
        while stack:
            x = stack.pop(0)
            out.append(x)
            stack.extend(children.get(x, []))
        return out

    lines = ["### 被投递的墙,后续改写是否真的解开了它(A3)\n"]
    lines.append(
        "判据是**后代在被拒取值上有跑通的 trial**,且共享内存要在**墙的那个取值处**下降 —— "
        "不是在后代自己的最优点处(最优点落在 tuner 偏好的取值上,其占用与该墙是否可行无关)。")
    lines.append("")
    header = "| 被墙候选 | knob | 被拒值 | 后代 | 结果 | 该取值处 shared_bytes | 上限 |"
    lines.extend([header, "|---|---|---|---|---|---|---|"])
    any_row = False
    for cid, w in attributed:
        knob = w.get("param")
        refused = w.get("refused_value")
        limit = w.get("limit")
        kids = descendants(cid)
        if not kids:
            lines.append(
                f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | —— | "
                f"**该墙尚无后代**(改写未产生/未注册) | —— | {limit} |")
            any_row = True
            continue
        for kid in kids:
            saw_knob = False
            completed: list[int] = []
            refused_again = False
            for t in trials.get(kid, []):
                vals = (t.get("params") or {}).get("values") or {}
                if knob not in vals:
                    continue
                saw_knob = True
                if not _same_value(vals[knob], refused):
                    continue
                if t.get("status") == "complete":
                    sb = (t.get("profile") or {}).get("shared_bytes")
                    if isinstance(sb, int):
                        completed.append(sb)
                    else:
                        completed.append(-1)
                elif t.get("failure_kind") == "infeasible_shared_memory":
                    refused_again = True
            if completed:
                real = [s for s in completed if s >= 0]
                span = (f"{min(real)}..{max(real)}" if real and min(real) != max(real)
                        else (str(real[0]) if real else "未记录"))
                under = (isinstance(limit, (int, float)) and real and max(real) <= limit)
                verdict = "**FREED**" + ("(占用已降到上限内)" if under
                                         else "(但占用仍在上限处 ⇒ 另有原因)")
                lines.append(f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | `{kid}` | "
                             f"{verdict} | {span}({len(completed)} 个 trial) | {limit} |")
            elif refused_again:
                lines.append(f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | `{kid}` | "
                             f"仍被拒 | —— | {limit} |")
            elif not saw_knob:
                lines.append(f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | `{kid}` | "
                             f"维度已消失(改写后无此 knob) | —— | {limit} |")
            else:
                lines.append(f"| `{cid}` | {knob} | **{_fmt_val(refused)}** | `{kid}` | "
                             f"未曾尝试(TPE 非均匀采样,缺席≠被拒) | —— | {limit} |")
            any_row = True
    lines.append("")
    lines.append(
        "**这一节不能单独证明 2e 有效**:「改写本来就会顺手降共享内存」会给出同样的观测。"
        "要归因必须对照一个**关掉 2e 的同配置臂**,在它自己的 `TuningStats` 上离线跑 `find_walls`"
        "(`scripts/probes/a3_counterfactual_arm_without_2e.py`)。")
    return lines if any_row else []


def soft_wall_lines(events: Any) -> list[str]:
    """The item-2 section: register-spill walls, with the denominators that make them readable.

    Returns [] when the soft wall never ran, so a run without it reads exactly as before.

    THE DENOMINATORS ARE THE SECTION. A count of walls cannot be interpreted on its own here, for two
    measured reasons:

      * a candidate whose fastest trial does not spill has NO wall to find at the point the agent
        rewrites from -- 103 of 153 candidates in the local corpus, a structural non-event rather than
        a negative result. "Not applicable" and "applicable, no wall" must not collapse.
      * applicability is not a constant across corpora: 43% (24 of 56) on the five backed-up runs,
        23.5% (36 of 153) locally, and 0 of 8 on the L3:48 runs. So the rate has to be computed from
        THIS run's own numbers and never quoted from another.

    And every row says it was not probe-confirmed. The hard wall's attribution is re-checked by an
    independent question to the compiler; this is one observation of an already-measured field.
    """
    rows = []
    for ev in events:
        t = _ev(ev, "type")
        if t != "RESOURCE_SOFT_WALL":
            continue
        rows.append((_ev(ev, "payload") or {}))
    if not rows:
        return []

    n = len(rows)
    applicable = [p for p in rows if p.get("applicable")]
    hit = [p for p in applicable if (p.get("walls") or [])]
    zero_spill = sum(1 for p in rows
                     if not p.get("applicable") and "does not spill" in str(p.get("reason") or ""))
    n_walls = sum(len(p.get("walls") or []) for p in rows)

    lines = ["## 寄存器溢出墙(软墙,item 2)\n"]
    lines.append(
        f"- 候选数 {n};**最优点确实溢出(可适用)的 {len(applicable)} 个**"
        + (f"({100.0 * len(applicable) / n:.0f}%)" if n else "")
        + f";其中找到墙的 {len(hit)} 个,共 {n_walls} 条 (候选, knob)")
    lines.append(
        f"- **不适用 {n - len(applicable)} 个,其中 {zero_spill} 个的最优 trial 根本不溢出** —— "
        "这不是阴性结果:最优点不溢出时,任何 knob 都不可能在该点被溢出截断。"
        "「不适用」与「适用但无墙」是两种状态,必须分开读。")
    lines.append(
        "- **适用率不是常数**:五个已备份 run 上是 43%(24/56),本机语料 23.5%(36/153),"
        "L3:48 的 8 个候选上是 0 —— 所以这一行的比例只能用**本 run 自己的**数字,不能引用别处的。")
    lines.append(
        "- **每一条都未经独立探针确认**:硬墙的归因由编译器的第二次独立提问复核(实测最优点 6/6),"
        "软墙只有一次已测字段的观测,且没有廉价探针可加"
        "(「会不会溢出」不是编译器能脱离编译单独回答的是非题)。")
    lines.append("")

    body: list[str] = []
    for p in rows:
        cid = str(p.get("candidate_id") or "?")
        for w in (p.get("walls") or []):
            spills = w.get("spills_by_value") or []
            vals = w.get("ran_values") or []
            span = (f"{spills[0]:.0f} → {spills[-1]:.0f}" if spills else "?")
            body.append(
                f"| `{cid}` | {w.get('param')} | "
                f"{', '.join(_fmt_val(v) for v in vals)} | {span} | "
                f"**{_fmt_val(w.get('onset_value'))}** | "
                f"{float(w.get('tail_gain_pct') or 0.0):+.1f}% | "
                f"{w.get('limiter_at_best') or '未测'} |")
    if body:
        lines.extend(["| 候选 | knob | 已测取值 | 溢出槽位 | 起始取值 | 尾部斜率 | 最优点限制者 |",
                      "|---|---|---|---|---|---|---|", *body, ""])
    else:
        lines.append("**本 run 没有找到任何溢出墙。** 见上一行的分母判断这是「不适用」还是「无墙」。")
        lines.append("")
    return lines


def wall_lines(events: Any) -> list[str]:
    """The section. Returns [] when neither wall mechanism ran, so a run without them reads as before.

    The two mechanisms are INDEPENDENTLY switched, so the early return has to check both: with only the
    soft wall on, `_rows` is empty and returning [] here would drop a section that has content -- the
    shape of defect where a new reader silently reports nothing for a run that produced something.
    """
    payloads = _rows(events)
    if not payloads:
        return soft_wall_lines(events)

    lines = ["## 共享内存墙归因(2e)\n"]

    total_found = sum(int(p.get("walls_found") or 0) for p in payloads)
    total_probed = sum(int(p.get("walls_probed") or 0) for p in payloads)
    total_worthless = sum(int(p.get("walls_worthless") or 0) for p in payloads)
    total_capped = sum(int(p.get("walls_skipped_by_cap") or 0) for p in payloads)
    probe_s = sum(float(p.get("probe_total_s") or 0.0) for p in payloads)
    n_probes = sum(int(p.get("n_probes") or 0) for p in payloads)
    # The INPUT, which decides whether "0 walls" is a finding or an empty table. Measured live on
    # arm 3: two candidates with 12 and 8 refusals produced 0 walls, because every refused value had
    # also RUN successfully in some other combination -- no value was truncated out of its range. A
    # summary that printed only "found 0" is indistinguishable from "this card never refused
    # anything", and those call for opposite conclusions: the first says the mechanism had input and
    # nothing to say, the second says the arm was on the wrong hardware.
    total_refused = sum(int(p.get("n_refused_configs") or 0) for p in payloads)

    counts = {"attributed": 0, "not_attributed": 0, "undecidable": 0, "attributed_any_origin": 0}
    for p in payloads:
        for k, v in (p.get("counts") or {}).items():
            if k in counts:
                counts[k] += int(v)

    lines.append(
        f"- 候选数 {len(payloads)};**被硬件拒绝的配置 {total_refused} 条**(2e 的唯一输入);"
        f"发现墙 {total_found} 个,其中**斜率为正、值得归因**的 "
        f"{total_probed} 个,**顶住但不值钱**的 {total_worthless} 个"
        + (f",受诊断上限跳过 {total_capped} 个" if total_capped else ""))
    if total_refused and not total_found:
        # Not a defect, and the report must say so in words rather than leaving a reader to infer it
        # from two numbers. Base rate measured on box 2's two completed L3:43 runs: 6 of 24
        # candidates had any wall and only 3 of 24 had a probe-worthy one, so a run with refusals and
        # no walls is the common case, not a broken mechanism.
        lines.append(
            f"- **有输入但没有墙**:{total_refused} 条拒绝里没有任何 knob 的被拒值落在它已测范围之外 —— "
            "每个被拒的取值在别的组合下都跑通过,所以没有维度被**截断**。"
            "这是机制的正常输出而不是缺陷(box2 两个已完成 run 的基线:24 个候选里 6 个有墙、"
            "只有 3 个有值得探针的墙)。")
    elif not total_refused:
        lines.append(
            "- **没有任何输入**:本 run 没有被硬件拒绝的配置,所以 2e 无从归因。"
            "这通常意味着卡的共享内存上限对该任务不紧(A800 的 166912 B 对 4090 的 101376 B)。")
    lines.append(
        f"- 归因结果:**ATTRIBUTED {counts['attributed']}**、"
        f"not attributed {counts['not_attributed']}、undecidable {counts['undecidable']}"
        "(以上三项都是**在最优点 θ\\* 处**的裁决,与历史 run 同底可比)")
    # The top-K difference, reported as its own line rather than folded into the counts above. The
    # gap between the two IS the measurement of what relaxing "only at the optimum" bought: a wall the
    # compiler refuses at the 2nd and 3rd fastest configurations but accepts at the 1st used to be
    # discarded, even though those points sit inside this project's own re-evaluation noise
    # (+-2-4%, unstable sign) around theta*.
    extra = counts["attributed_any_origin"] - counts["attributed"]
    if counts["attributed_any_origin"] or extra:
        primary_fits = sum(1 for p in payloads for w in (p.get("walls") or [])
                           if _delivered(w) and w.get("verdict") == "not_attributed")
        comparison = ("并不触墙**(只在第 2/3 快的点上触墙)" if primary_fits == extra
                      else "未确认触墙**(仅在其他高性能点确认)")
        tail = (f",其中 **{extra} 个在 θ\\* 处{comparison}"
                f"—— 这 {extra} 个正是放宽「仅最优点」约束新增的"
                if extra > 0 else "(与 θ\\* 处的裁决完全一致)")
        lines.append(
            f"- **至少在一个高性能点上成立的墙 {counts['attributed_any_origin']} 个**{tail}")
    if n_probes:
        lines.append(
            f"- 探针成本:{n_probes} 个变体共 {probe_s:.1f} s"
            f"(每个 {probe_s / n_probes:.2f} s)—— 批处理共享一次进程启动")
    lines.append(
        "- **每一条归因都是有条件的**:它成立于该候选的**实测高性能参数点**处。"
        "从空间默认配置出发,同样的改动往往并不触墙(实测 6 个墙里只有 1 个一致),"
        "因为默认配置各维都取最小。**不要把它读成该 knob 的固有上限。**")
    lines.append("")

    header = ("| 候选 | knob | 已测取值 | 被拒值 | θ\\* 处归因 | 高性能点 | 编译器要求 | 上限 | 倍数 "
              "| 尾部斜率 | 默认配置 origin |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    body: list[str] = []
    worthless: list[str] = []
    for p in payloads:
        cid = str(p.get("candidate_id") or "?")
        for w in (p.get("walls") or []):
            ran = ", ".join(_fmt_val(v) for v in (w.get("ran_values") or []))
            row = (f"| `{cid}` | {w.get('param')} | {ran} | "
                   f"**{_fmt_val(w.get('refused_value'))}** | ")
            verdicts = w.get("origin_verdicts") or {}
            verdict = verdicts.get("theta_star", w.get("verdict"))
            n_hit, n_probed = _origin_counts(w)
            if verdict is None and not verdicts:
                # No recorded origin: keep unprobed walls separate from measured verdicts.
                worthless.append(
                    f"- `{cid}` {w.get('param')}:已测 {ran} → 被拒 "
                    f"{_fmt_val(w.get('refused_value'))},尾部 "
                    f"{float(w.get('tail_gain_pct') or 0.0):+.1f}%"
                    f"{'(非单调)' if not w.get('monotone') else ''} —— 顶住但不值钱")
                continue
            max_shared, over, from_origin = _attributing_footprint(w)
            # The θ* verdict stays in its own column so the number is comparable with the finished
            # runs; the count is the column that says whether the wall survived away from θ*, and the
            # footprint column names its origin whenever that origin is NOT θ* -- otherwise a reader
            # would take the bytes for θ*'s and see a figure under the limit next to "ATTRIBUTED".
            row += (f"{'**ATTRIBUTED**' if verdict == 'attributed' else verdict or '未测'} | "
                    f"{'**' if n_hit and n_hit == n_probed else ''}{n_hit}/{n_probed}"
                    f"{'**' if n_hit and n_hit == n_probed else ''} | "
                    f"{max_shared}{f'(在 {from_origin})' if from_origin else ''} | "
                    f"{w.get('limit')} | "
                    f"{f'{float(over):.2f}x' if isinstance(over, (int, float)) else '?'} | "
                    f"{float(w.get('tail_gain_pct') or 0.0):+.1f}% | ")
            so = verdicts.get("default", w.get("second_origin"))
            if so is None:
                row += "未测 |"
            elif verdict is None:
                row += f"{so} |"
            elif so == verdict:
                row += f"{so}(一致) |"
            else:
                row += f"**{so}(不一致 ⇒ 依赖其他 knob)** |"
            body.append(row)

    if body:
        lines.extend([header, sep, *body, ""])
    if worthless:
        lines.append("**发现但未归因的墙**(斜率非正或超出诊断上限):")
        lines.extend(worthless)
        lines.append("")

    # The comparison that makes the section worth reading: the analyst's own claims about the same
    # thing. Only the agreement counts are shown here -- the claims themselves are already in the
    # bottleneck-report section, and repeating them would bury the comparison.
    attributed = {(str(p.get("candidate_id")), w.get("param"))
                  for p in payloads for w in (p.get("walls") or [])
                  if _delivered(w)}
    if attributed:
        claims = 0
        confirmed = 0
        for ev in events:
            t = ev.get("type") if isinstance(ev, dict) else getattr(ev, "type", None)
            if t != "BOTTLENECK_REPORTED":
                continue
            p = (ev.get("payload") if isinstance(ev, dict) else getattr(ev, "payload", None)) or {}
            cid = str(p.get("candidate_id") or "")
            rep = p.get("report") or {}
            for lim in (rep.get("parameter_limits") or []):
                claims += 1
                if (cid, lim.get("param")) in attributed:
                    confirmed += 1
        if claims:
            lines.append(
                f"**对比 analyst 自己的 `parameter_limits`**:共 {claims} 条声明,"
                f"其中与本节归因一致的 {confirmed} 条"
                f"({100.0 * confirmed / claims:.0f}%)。"
                "该字段由 agent 填写、harness 从不核对,所以两者不一致时**本节是实测的那一方**。")
            lines.append("")
    # A3 last, because it is the only part that depends on events AFTER the wall was found: it needs
    # the rewrite and its trials to exist. Appended here rather than at report.py's call site so
    # `wall_lines` stays the single entry point and a run without 2e still emits nothing.
    lines.extend(freed_lines(events))
    # The soft wall LAST and as its own section, never merged into the table above. The hard wall's
    # rows carry an independent compiler confirmation and the soft wall's do not, and a reader who took
    # one for the other would have over-read the weaker evidence. Appended here so `wall_lines` stays
    # the single entry point -- a run with the soft wall on and 2e off still emits the section, because
    # `soft_wall_lines` reads its own event type and returns [] when there is none.
    lines.extend(soft_wall_lines(events))
    return lines
