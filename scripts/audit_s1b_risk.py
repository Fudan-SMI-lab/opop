"""S1b 的风险:过度拦截会不会发生,以及误判的概率是多少 —— 全部用实测回答。

前一轮得到「跨空间汇总 + 下限 8 + 见通过即撤回 ⇒ 省 249 个失败 trial、误杀 0」。那个 0 是在
**4 条规则**上观测到的 0,不是在 1842 个点上;而且下限 8 是我从同一份语料里挑出来的。这个脚本测
六件事,每一件都可能否掉 S1b:

  1 校准曲线  P(该取值后来其实会通过 | 已累计 k 次失败且 0 次通过)。这是「误判概率」的直接定义,
              按阈值给出,不依赖我挑的那个 8。
  2 余量      对**后来真的通过了**的取值,它在首次通过前累计了多少次失败。若有取值卡在 7,
              则 8 就在悬崖边上,换个采样顺序就会越界。
  3 置信上界  0/4 与 0/249 分别能把误杀率压到多少(Clopper-Pearson 单侧)。0 不等于安全。
  4 汇总范围  跨空间是否等于**跨候选**?根因(未补偿的 dot / PREC 门控)是**候选自己的**代码性质,
              跨候选汇总在原理上不成立。测:每候选几个空间、触发的规则由几个候选的证据拼成。
  5 阈值稳定性 用 4 个 run 选出「最小的零误杀 N」,拿到第 5 个 run 上用 —— 这正是我批评
              「N=12 是事后挑的」时用的判据,现在对 S1b 自己用一次。
  6 生效时长  错误规则从触发到撤回之间活了多久(暴露窗口)。

一律用 (knob, value) 作汇总键,run 边界清零 —— 与判据一致。
"""
import json
import sys
import glob
from collections import defaultdict
from pathlib import Path
from statistics import median


def run_trials(run_dir: Path):
    """一个 run 的全部 trial,按 events.jsonl 的 append-only 顺序,产出 (space_id, candidate_id, cfg, ok)。

    只保留 complete 与 correctness_mismatch —— 这两类是 S1b 的池子。shared-memory 不可行与
    runtime_error 不在其中(前者已被编译期预筛处理,后者可能是非确定性的)。
    """
    ev = run_dir / "events.jsonl"
    if not ev.exists():
        return
    for line in ev.open(encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        p = e.get("payload") or {}
        tr = p.get("trial") or p
        vals = (tr.get("params") or {}).get("values") or {}
        sid = tr.get("space_id")
        cid = tr.get("candidate_id") or ""
        if not sid or not vals:
            continue
        status, kind = tr.get("status"), tr.get("failure_kind")
        cfg = {k: str(v) for k, v in vals.items()}
        if status == "complete":
            yield sid, cid, cfg, True
        elif kind == "correctness_mismatch":
            yield sid, cid, cfg, False


def key_histories(runs, pooled=True, by_candidate=False):
    """每个汇总键的结局序列。pooled=False 时键含 space_id;by_candidate 时键含 candidate_id。"""
    hist = defaultdict(list)          # (run, key) -> [ok, ok, ...] 按顺序
    contributors = defaultdict(set)   # (run, key) -> {candidate_id}
    for r in runs:
        for sid, cid, cfg, ok in run_trials(Path(r)):
            for knob, v in cfg.items():
                if pooled:
                    k = (cid, knob, v) if by_candidate else (knob, v)
                else:
                    k = (sid, knob, v)
                hist[(r, k)].append(ok)
                contributors[(r, k)].add(cid)
    return hist, contributors


def first_pass_index(seq):
    """该键首次通过前累计了多少次失败;从不通过则返回 None。"""
    for i, ok in enumerate(seq):
        if ok:
            return i
    return None


def cp_upper(n, alpha=0.05):
    """0 次事件 / n 次机会 的单侧 Clopper-Pearson 上界。"""
    if n <= 0:
        return 1.0
    return 1.0 - alpha ** (1.0 / n)


runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))
if not runs:
    print("没有找到 run-l3-* —— 这是语料的问题,不是结论")
    sys.exit(1)
print("语料:%d 个 run\n" % len(runs))

# ---------------------------------------------------------------- 1 & 2 校准与余量
for label, pooled in (("单空间", False), ("跨空间汇总", True)):
    hist, _ = key_histories(runs, pooled=pooled)
    # p = 首次通过前的失败数;None = 从不通过
    ps, totals = {}, {}
    for rk, seq in hist.items():
        ps[rk] = first_pass_index(seq)
        totals[rk] = sum(1 for ok in seq if not ok)
    print("=== %s:误判概率随阈值变化 ===" % label)
    print("%-6s %-12s %-12s %-12s %s" % ("阈值k", "会触发", "其中误判", "误判概率", "0 误判时的 95% 上界"))
    print("-" * 74)
    for k in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        fires = [rk for rk in hist
                 if (ps[rk] is None and totals[rk] >= k) or (ps[rk] is not None and ps[rk] >= k)]
        wrong = [rk for rk in fires if ps[rk] is not None]
        rate = (100.0 * len(wrong) / len(fires)) if fires else None
        ub = cp_upper(len(fires)) * 100.0 if fires else None
        print("%-6d %-12d %-12d %-12s %s" % (
            k, len(fires), len(wrong),
            ("%.1f%%" % rate) if rate is not None else "n/a",
            ("%.1f%%" % ub) if (ub is not None and not wrong) else ("—" if wrong else "n/a")))
    survivors = sorted(p for p in ps.values() if p is not None)
    print()
    print("余量:后来**真的通过**的取值,在首次通过前累计了多少次失败")
    if survivors:
        hi = [p for p in survivors if p >= 5]
        print("  n=%d,中位 %.0f,最大 **%d**;>=5 次的有 %d 个,>=8 次的有 %d 个"
              % (len(survivors), median(survivors), max(survivors), len(hi),
                 len([p for p in survivors if p >= 8])))
        print("  分布 (失败数: 个数):", ", ".join(
            "%d:%d" % (v, survivors.count(v)) for v in sorted(set(survivors))[:14]))
    print()

# ---------------------------------------------------------------- 4 汇总范围
print("=== 汇总范围:跨空间是不是等于跨候选? ===")
for r in runs:
    per_cand = defaultdict(set)
    for sid, cid, _c, _ok in run_trials(Path(r)):
        per_cand[cid].add(sid)
    n_sp = sum(len(v) for v in per_cand.values())
    print("  %-30s 候选 %2d 个,空间 %2d 个,每候选空间数中位 %.1f"
          % (r.rsplit("/", 1)[-1][:30], len(per_cand), n_sp,
             median([len(v) for v in per_cand.values()]) if per_cand else 0))
print()
hist_p, contrib = key_histories(runs, pooled=True)
for k in (6, 8, 12):
    fires = []
    for rk, seq in hist_p.items():
        p = first_pass_index(seq)
        tot = sum(1 for ok in seq if not ok)
        if (p is None and tot >= k) or (p is not None and p >= k):
            fires.append(rk)
    multi = [rk for rk in fires if len(contrib[rk]) > 1]
    print("  阈值 %2d:触发 %d 条,其中 **%d 条的证据来自多个候选**(跨候选规则)"
          % (k, len(fires), len(multi)))
    for rk in sorted(multi, key=lambda x: -len(contrib[x]))[:5]:
        print("      %s=%s ← %d 个候选" % (rk[1][0], rk[1][1], len(contrib[rk])))
print()
print("  同一判据但只在**同一候选**内汇总:")
hist_c, _ = key_histories(runs, pooled=True, by_candidate=True)
for k in (6, 8, 12):
    fires = 0
    wrong = 0
    saved = 0
    for rk, seq in hist_c.items():
        p = first_pass_index(seq)
        tot = sum(1 for ok in seq if not ok)
        if (p is None and tot >= k) or (p is not None and p >= k):
            fires += 1
            if p is not None:
                wrong += 1
            saved += (tot - k) if p is None else max(0, p - k)
    print("      阈值 %2d:触发 %d 条、误判 %d 条、可省失败 trial 约 %d 个" % (k, fires, wrong, saved))
print()

# ---------------------------------------------------------------- 5 阈值稳定性
print("=== 阈值稳定性:用 4 个 run 选 N,在第 5 个 run 上验(对 S1b 自己用一次我批评别人的判据)===")
print("%-30s %-10s %-12s %-12s %s" % ("留出的 run", "选出的 N*", "留出 run 触发", "误判", "可省失败"))
print("-" * 82)
agree = []
for held in runs:
    train = [r for r in runs if r != held]
    hist_t, _ = key_histories(train, pooled=True)
    # 训练集上最小的零误判 N
    nstar = None
    for k in range(1, 25):
        fires = wrong = 0
        for rk, seq in hist_t.items():
            p = first_pass_index(seq)
            tot = sum(1 for ok in seq if not ok)
            if (p is None and tot >= k) or (p is not None and p >= k):
                fires += 1
                if p is not None:
                    wrong += 1
        if fires > 0 and wrong == 0:
            nstar = k
            break
    if nstar is None:
        print("%-30s %-10s %s" % (held.rsplit("/", 1)[-1][:30], "无", "训练集上没有任何零误判且非空的 N"))
        continue
    agree.append(nstar)
    hist_h, _ = key_histories([held], pooled=True)
    fires = wrong = saved = 0
    for rk, seq in hist_h.items():
        p = first_pass_index(seq)
        tot = sum(1 for ok in seq if not ok)
        if (p is None and tot >= nstar) or (p is not None and p >= nstar):
            fires += 1
            if p is not None:
                wrong += 1
            saved += (tot - nstar) if p is None else max(0, p - nstar)
    print("%-30s %-10d %-12d %-12d %d" % (
        held.rsplit("/", 1)[-1][:30], nstar, fires, wrong, saved))
print()
if agree:
    print("  选出的 N* 集合:%s ⇒ %s" % (
        sorted(agree),
        "稳定(全部相同)" if len(set(agree)) == 1 else "**不稳定,阈值随语料变动**"))
print()

# ---------------------------------------------------------------- 6 生效时长
print("=== 暴露窗口:一条错误规则从触发到撤回之间,活了多少次该取值的出现 ===")
print("%-8s %-12s %-14s %s" % ("阈值k", "错误规则数", "窗口(该取值出现次数)", "明细"))
print("-" * 74)
for k in (3, 6, 8, 12):
    windows = []
    for rk, seq in hist_p.items():
        p = first_pass_index(seq)
        if p is not None and p >= k:
            windows.append((p - k, rk[1]))
    if not windows:
        print("%-8d %-12d %s" % (k, 0, "无错误规则"))
        continue
    ws = sorted(w for w, _ in windows)
    print("%-8d %-12d %-14s %s" % (
        k, len(windows), "中位 %.0f / 最大 %d" % (median(ws), max(ws)),
        ", ".join("%s=%s(+%d)" % (n[0], n[1], w) for w, n in sorted(windows, reverse=True)[:4])))
print()
print("窗口 = 触发之后、首次通过之前该取值还出现了几次。**这是重放能看到的暴露**;")
print("重放看不到的是**降权本身会把那次通过往后推**(顺序被固定住了)—— 见输出末尾的未验证项。")
