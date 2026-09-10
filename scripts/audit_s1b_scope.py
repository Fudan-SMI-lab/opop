"""上一轮暴露了一个原理性缺陷:"跨空间汇总"实际上跨了**候选**。这里测有原则的那一版。

风险测量的发现:阈值 8 时 4 条规则里 **3 条的证据来自多个候选**,`DOT_PRECISION=tf32` 那条更是
把 **10 个候选**的失败混在一起。而 S1b 的根因是「**候选自己**把低精度放在未补偿的 dot / PREC 门控
整个算法」—— 那是**那个候选的代码性质**。跨候选汇总在原理上不成立,它之所以在这份语料上「看起来
对」,是因为多数候选恰好犯同一个错。

所以把汇总键改成 `(candidate_id, knob, value)` —— 仍然跨该候选的多个空间(那是打破单空间算术
天花板所必需的),但**不跨候选**。用与前一轮完全相同的前瞻式模拟对比三种汇总范围,使
saved / mis-hit 直接可比(前一轮 by_candidate 的数字是 hindsight 估计,不是模拟)。

同时给出每种范围的 Clopper-Pearson 上界:0/4 与 0/33 是非常不同的证据强度,这一点比 saved 的
大小更能决定该选哪个。
"""
import json
import sys
import glob
from collections import defaultdict
from pathlib import Path


def run_events(run_dir: Path):
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
        sid, cid = tr.get("space_id"), tr.get("candidate_id") or ""
        if not sid or not vals:
            continue
        status, kind = tr.get("status"), tr.get("failure_kind")
        cfg = {k: str(v) for k, v in vals.items()}
        if status == "complete":
            yield sid, cid, cfg, True
        elif kind == "correctness_mismatch":
            yield sid, cid, cfg, False


def simulate(runs, floor, scope):
    """前瞻式:先按已触发的规则计分,再更新证据。scope = space | candidate | run。"""
    saved = mis = 0
    fired = set()
    retracted = set()
    for r in runs:
        fails = defaultdict(int)
        passes = defaultdict(int)
        dead = set()
        for sid, cid, cfg, ok in run_events(Path(r)):
            for knob, v in cfg.items():
                if scope == "space":
                    k = (sid, knob, v)
                elif scope == "candidate":
                    k = (cid, knob, v)
                else:
                    k = (knob, v)
                if k in dead:
                    if ok:
                        mis += 1
                    else:
                        saved += 1
            for knob, v in cfg.items():
                if scope == "space":
                    k = (sid, knob, v)
                elif scope == "candidate":
                    k = (cid, knob, v)
                else:
                    k = (knob, v)
                if ok:
                    passes[k] += 1
                    if k in dead:
                        dead.discard(k)
                        retracted.add((r, k))
                else:
                    fails[k] += 1
                if passes[k] == 0 and fails[k] >= floor and k not in dead:
                    dead.add(k)
                    fired.add((r, k))
    return saved, mis, len(fired), len(retracted)


def cp_upper(n, alpha=0.05):
    return 1.0 if n <= 0 else 1.0 - alpha ** (1.0 / n)


runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))
SCOPES = (("每空间独立", "space"), ("每候选(跨其空间)", "candidate"), ("整个 run(跨候选)", "run"))

print("=== 三种汇总范围,同一前瞻式模拟 ===")
print("%-20s %-7s %-8s %-11s %-11s %-9s %s" % (
    "汇总范围", "下限N", "触发", "省下失败", "误杀通过", "撤回", "误杀率(0 时给 95% 上界)"))
print("-" * 104)
for label, scope in SCOPES:
    for floor in (6, 8, 12):
        saved, mis, fired, retr = simulate(runs, floor, scope)
        tot = saved + mis
        if mis:
            rate = "%.2f%%" % (100.0 * mis / tot)
        else:
            rate = "0%% (≤%.1f%%, n=%d)" % (cp_upper(fired) * 100.0, fired)
        print("%-20s %-7d %-8d %-11d %-11d %-9d %s" % (label, floor, fired, saved, mis, retr, rate))
    print()

print("上界是按**触发的规则数**算的(每条规则是一次独立的判断机会)。0/4 与 0/33 的证据强度差一个")
print("数量级:前者的 95% 上界高到无法排除「误杀率其实很高」,后者可以。")
print()

print("=== 阈值稳定性:每种范围各做一次留一(用 4 个 run 选最小零误杀 N,在第 5 个上用)===")
print("%-20s %-24s %-10s %s" % ("汇总范围", "选出的 N*", "是否稳定", "留出 run 上的表现(误杀/省下)"))
print("-" * 96)
for label, scope in SCOPES:
    picks, outcomes = [], []
    for held in runs:
        train = [r for r in runs if r != held]
        nstar = None
        for k in range(1, 25):
            _s, m, f, _r = simulate(train, k, scope)
            if f > 0 and m == 0:
                nstar = k
                break
        if nstar is None:
            picks.append(None)
            outcomes.append("训练集无可选 N")
            continue
        picks.append(nstar)
        s, m, f, _r = simulate([held], nstar, scope)
        outcomes.append("%d/%d" % (m, s))
    good = [p for p in picks if p is not None]
    stable = "稳定" if good and len(set(good)) == 1 else "**不稳定**"
    print("%-20s %-24s %-10s %s" % (label, str(picks), stable, "  ".join(outcomes)))
print()
print("留一是我用来否掉「N=12 是事后挑的」的同一条判据,现在对 S1b 自己用。若 N* 随语料变动,")
print("那么把某个具体的 N 写进代码就是在挑一个恰好在这份语料上安全的数。")
