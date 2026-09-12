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


def wall_lines(events: Any) -> list[str]:
    """The section. Returns [] when 2e never ran, so a run without it reads exactly as before."""
    payloads = _rows(events)
    if not payloads:
        return []

    lines = ["## 共享内存墙归因(2e)\n"]

    total_found = sum(int(p.get("walls_found") or 0) for p in payloads)
    total_probed = sum(int(p.get("walls_probed") or 0) for p in payloads)
    total_worthless = sum(int(p.get("walls_worthless") or 0) for p in payloads)
    total_capped = sum(int(p.get("walls_skipped_by_cap") or 0) for p in payloads)
    probe_s = sum(float(p.get("probe_total_s") or 0.0) for p in payloads)
    n_probes = sum(int(p.get("n_probes") or 0) for p in payloads)

    counts = {"attributed": 0, "not_attributed": 0, "undecidable": 0}
    for p in payloads:
        for k, v in (p.get("counts") or {}).items():
            if k in counts:
                counts[k] += int(v)

    lines.append(
        f"- 候选数 {len(payloads)};发现墙 {total_found} 个,其中**斜率为正、值得归因**的 "
        f"{total_probed} 个,**顶住但不值钱**的 {total_worthless} 个"
        + (f",受诊断上限跳过 {total_capped} 个" if total_capped else ""))
    lines.append(
        f"- 归因结果:**ATTRIBUTED {counts['attributed']}**、"
        f"not attributed {counts['not_attributed']}、undecidable {counts['undecidable']}")
    if n_probes:
        lines.append(
            f"- 探针成本:{n_probes} 个变体共 {probe_s:.1f} s"
            f"(每个 {probe_s / n_probes:.2f} s)—— 批处理共享一次进程启动")
    lines.append(
        "- **每一条归因都是有条件的**:它成立于该候选的**最优参数点**处。"
        "从空间默认配置出发,同样的改动往往并不触墙(实测 6 个墙里只有 1 个一致),"
        "因为默认配置各维都取最小。**不要把它读成该 knob 的固有上限。**")
    lines.append("")

    header = ("| 候选 | knob | 已测取值 | 被拒值 | 归因 | 编译器要求 | 上限 | 倍数 "
              "| 尾部斜率 | 第二 origin |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    body: list[str] = []
    worthless: list[str] = []
    for p in payloads:
        cid = str(p.get("candidate_id") or "?")
        for w in (p.get("walls") or []):
            ran = ", ".join(_fmt_val(v) for v in (w.get("ran_values") or []))
            row = (f"| `{cid}` | {w.get('param')} | {ran} | "
                   f"**{_fmt_val(w.get('refused_value'))}** | ")
            verdict = w.get("verdict")
            if verdict is None:
                # Found but never probed: it failed the slope filter or the cap. Kept out of the
                # main table so the reader does not mistake "not probed" for "not attributed".
                worthless.append(
                    f"- `{cid}` {w.get('param')}:已测 {ran} → 被拒 "
                    f"{_fmt_val(w.get('refused_value'))},尾部 "
                    f"{float(w.get('tail_gain_pct') or 0.0):+.1f}%"
                    f"{'(非单调)' if not w.get('monotone') else ''} —— 顶住但不值钱")
                continue
            over = w.get("over_ratio")
            row += (f"{'**ATTRIBUTED**' if verdict == 'attributed' else verdict} | "
                    f"{w.get('max_shared')} | {w.get('limit')} | "
                    f"{f'{float(over):.2f}x' if isinstance(over, (int, float)) else '?'} | "
                    f"{float(w.get('tail_gain_pct') or 0.0):+.1f}% | ")
            so = w.get("second_origin")
            if so is None:
                row += "未测 |"
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
                  if w.get("verdict") == "attributed"}
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
    return lines
