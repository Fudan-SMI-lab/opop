"""两个还能翻转 S1b 的风险,以及降权强度 k 的选择 —— 重放能答的部分,答到底。

前三轮定了汇总范围(per-candidate)与阈值(N=7 留一稳定)。剩下两个风险,重放**可以**部分回答:

  A 降权强度 k 决定「误判之后多久能被纠正」。摘除 = k 无穷 = 永不纠正;k=4 时该取值仍有约 1/4
    的原份额。测:一条错误规则触发后,该取值在剩余 trial 里还会出现几次(=纠正机会),以及降权到
    1/k 后这些机会还剩多少期望次数。若期望 < 1,则「保留非零概率」在 40-trial 预算内是空话。

  B 省下的预算流向哪里。S1b 的收益全部建立在「省下的 trial 会变成别的真实评测」。已知 63/64 空间
    跑满 40 trial ⇒ 有消费者;但**被降权的取值占该空间采样份额多少**决定了省下的量级是否真如
    saved 所示。测:触发时刻该键剩余的出现次数,按空间归一。

  C 最坏情况的形状。不看均值,看**单个空间最惨的那次**:一条规则最多能吃掉某空间多少比例的
    剩余预算,以及若它是错的,该空间会不会因此错过它的最优点。后者用「该空间的最优 trial 是否
    使用了被降权的取值」直接判 —— 这是过度拦截最坏后果的可测形式。
"""
import json
import sys
import glob
from collections import defaultdict
from pathlib import Path
from statistics import median


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
        lat = tr.get("latency_ms") or {}
        ms = lat.get("median")
        if not isinstance(ms, (int, float)):
            ms = lat.get("mean")
        status, kind = tr.get("status"), tr.get("failure_kind")
        cfg = {k: str(v) for k, v in vals.items()}
        if status == "complete":
            yield sid, cid, cfg, True, (float(ms) if isinstance(ms, (int, float)) and ms > 0 else None)
        elif kind == "correctness_mismatch":
            yield sid, cid, cfg, False, None


FLOOR = 7          # 留一选出、五个 run 一致的阈值
runs = sorted(glob.glob(sys.argv[1] + "/run-l3-*"))

# 每个 run 走一遍,记录每条规则触发时刻之后该键的剩余出现,并按空间统计份额
print("=== A/B 触发之后还剩多少机会(per-candidate 汇总,N=%d)===" % FLOOR)
print("%-8s %-14s %-16s %-16s %s" % ("", "规则数", "触发后该键剩余出现", "占该空间剩余预算", "降权 1/k 后期望仍被采到"))
print("-" * 96)

all_rules = []          # (run, key, remaining_appearances, share_of_space_remaining, was_wrong)
for r in runs:
    ev = list(run_events(Path(r)))
    fails = defaultdict(int)
    passes = defaultdict(int)
    dead = {}
    per_space_total = defaultdict(int)
    for sid, _cid, _cfg, _ok, _ms in ev:
        per_space_total[sid] += 1
    seen_in_space = defaultdict(int)
    for idx, (sid, cid, cfg, ok, _ms) in enumerate(ev):
        seen_in_space[sid] += 1
        for knob, v in cfg.items():
            k = (cid, knob, v)
            if ok:
                passes[k] += 1
                dead.pop(k, None)
            else:
                fails[k] += 1
            if passes[k] == 0 and fails[k] >= FLOOR and k not in dead:
                dead[k] = idx
                # 该键在触发之后还出现几次,以及它属于哪些空间
                rest = [(s2, c2, cfg2, ok2) for (s2, c2, cfg2, ok2, _m) in ev[idx + 1:]
                        if c2 == cid and cfg2.get(knob) == v]
                spaces_touched = {s2 for s2, _c, _g, _o in rest}
                # 该键触发后所在空间的剩余 trial 总量
                rem_budget = sum(per_space_total[s] - seen_in_space[s] for s in spaces_touched) or 1
                wrong = any(o for _s, _c, _g, o in rest)
                all_rules.append((r, (cid, knob, v), len(rest), len(rest) / rem_budget, wrong))

for kk in (2, 4, 8):
    rem = [n for _r, _k, n, _s, _w in all_rules]
    shares = [s for _r, _k, _n, s, _w in all_rules]
    exp = [n / kk for n in rem]
    print("%-8s %-14d %-16s %-16s %s" % (
        "k=%d" % kk, len(all_rules),
        "中位 %.0f / 最大 %d" % (median(rem) if rem else 0, max(rem) if rem else 0),
        "中位 %.1f%%" % (100.0 * median(shares)) if shares else "n/a",
        "中位 %.1f 次%s" % (median(exp) if exp else 0,
                          "  ← <1 则纠正机会不足" if exp and median(exp) < 1 else "")))
print()
wrong_rules = [x for x in all_rules if x[4]]
print("其中**错误**规则(触发后该取值真的通过过):%d 条 / %d" % (len(wrong_rules), len(all_rules)))
for r, k, n, s, _w in wrong_rules[:6]:
    print("   %s  %s=%s  触发后出现 %d 次,占该空间剩余 %.1f%%"
          % (r.rsplit("/", 1)[-1][:24], k[1], k[2], n, 100.0 * s))
print()

# ---------------------------------------------------------------- C 最坏后果
print("=== C 最坏后果:被降权的取值,是不是该空间最优点用的取值? ===")
print("这是过度拦截最严重的可测形式 —— 不是「少测几个点」,而是「最优点被挡住」。")
print()
print("%-26s %-16s %-30s %s" % ("run", "空间", "该空间最优点", "最优点是否用了被降权的取值"))
print("-" * 104)
hits = 0
checked = 0
for r in runs:
    ev = list(run_events(Path(r)))
    # 每空间最优
    best = {}
    for sid, cid, cfg, ok, ms in ev:
        if ok and ms is not None and (sid not in best or ms < best[sid][0]):
            best[sid] = (ms, cfg, cid)
    # 该 run 的规则(用最终状态判定,偏保守:任何曾触发过的键都算)
    fails = defaultdict(int)
    passes = defaultdict(int)
    fired = set()
    for sid, cid, cfg, ok, _ms in ev:
        for knob, v in cfg.items():
            k = (cid, knob, v)
            if ok:
                passes[k] += 1
            else:
                fails[k] += 1
            if passes[k] == 0 and fails[k] >= FLOOR:
                fired.add(k)
    for sid, (ms, cfg, cid) in sorted(best.items()):
        checked += 1
        clash = [(kn, vv) for kn, vv in cfg.items() if (cid, kn, vv) in fired]
        if clash:
            hits += 1
            print("%-26s %-16s %-30s %s" % (
                r.rsplit("/", 1)[-1][:26], sid[:16], "%.4f ms" % ms,
                "**是** " + ", ".join("%s=%s" % c for c in clash)))
print()
print("检查了 %d 个空间的最优点,其中 **%d 个**用到了会被降权的取值。" % (checked, hits))
if hits == 0:
    print("零命中 = 在这份语料上,S1b 从未挡住任何空间的最优点。这是它最强的一条安全证据,")
    print("而且比「误杀率」更贴近实际后果:误杀一个平庸的通过点几乎无代价,挡住最优点才是灾难。")
print()
print("=== 重放**答不了**的部分(必须写明)===")
print("1. 降权会改变采样顺序,因此「触发后该取值还会出现 N 次」在真实运行里不成立 —— 真实的 N")
print("   更小(被降权了),所以纠正机会比上表更少,而省下的量也与上表不同。方向:两者都被高估。")
print("2. TPE 的代理模型对降权的反应完全未测(全部重放用随机采样器)。")
print("3. 最优点检查用的是**已记录**的最优,而非真实最优 —— 省下的预算本可能找到更好的点。")
