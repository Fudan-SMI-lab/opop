"""P5 的批量预筛只在"影响 shared 的 knob 组合数"可枚举时成立。数一下真实空间的规模。

如果某个候选的 tile/stage/dtype 子网格有几千个点,一次性全筛就不可行,方案要退化成"按需筛+缓存"。
从真实 run 的 SPACE_PUBLISHED 里读出每个空间的域,只取影响 shared 的那些 knob
(名字里含 BLOCK/STAGE/WARP/DTYPE/CHUNK/TILE),算它们的笛卡尔积大小。
"""
import json, io, glob, math, re

PAT = re.compile(r"BLOCK|STAGE|WARP|DTYPE|CHUNK|TILE|SPLIT", re.I)
for path in sorted(glob.glob("/root/autodl-tmp/opop-workspace/opop-glm/runs-l3/"
                             "run-l3-*/events.jsonl")):
    run = path.split("/")[-2]
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            try: e = json.loads(line)
            except Exception: continue
            if e.get("type") != "SPACE_PUBLISHED": continue
            p = e.get("payload") or {}
            sp = p.get("space") or p
            doms = sp.get("domains") or []
            full = 1
            sub = 1
            nsub = 0
            for d in doms:
                n = len(d.get("choices") or [])
                if n <= 0: continue
                full *= n
                if PAT.search(d.get("name") or ""):
                    sub *= n; nsub += 1
            rows.append((sp.get("candidate_id") or p.get("candidate_id"),
                         len(doms), full, nsub, sub))
    if not rows: continue
    print(f"\n### {run}")
    print(f"  {'candidate':16s} {'knobs':>5s} {'full grid':>12s} "
          f"{'shared knobs':>12s} {'shared grid':>12s}")
    for cid, nk, full, nsub, sub in rows:
        print(f"  {str(cid):16s} {nk:5d} {full:12,d} {nsub:12d} {sub:12,d}")
    subs = [r[4] for r in rows]
    print(f"  shared-subgrid: median {sorted(subs)[len(subs)//2]:,}  max {max(subs):,}")
    print(f"  at 7 ms marginal, the max costs {max(subs)*0.007:.1f}s to screen exhaustively")
