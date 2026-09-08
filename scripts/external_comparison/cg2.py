"""交叉验证:手算 shared 低估是否也出现在 L3:48 的候选上(另一个任务、另一个 agent 写的约束)。

上一版失败的原因是我从 device_caches 里捞编译产物,而 kernel 从未被调用所以缓存是空的。
改用 P1 的正路:JITFunction.warmup(..., grid=...) 只编译不启动,直接读 metadata.shared。
需要真实参数签名,所以先真跑一次 forward 让 Triton 编译,然后读缓存 —— 一次真实调用最可靠。
"""
import importlib.util, pathlib, re, itertools
import torch, triton

W = pathlib.Path("/root/autodl-tmp/ext-eval/ours")
BASE = (W / "our_l3_48_best.py").read_text()
dev = torch.device("cuda")
B, L, H, P, N, BLK = 2048, 128, 8, 64, 16, 64

def build(over):
    s = BASE
    for k, v in over.items():
        lit = f"'{v}'" if isinstance(v, str) else str(v)
        s, n = re.subn(rf"'{k}': [^,\n]+", f"'{k}': {lit}", s)
        assert n == 1, f"{k} x{n}"
    return s

def true_shared(mod):
    best = 0
    for v in vars(mod).values():
        if not isinstance(v, triton.runtime.JITFunction):
            continue
        for cache in (v.device_caches or {}).values():
            entries = cache[0] if isinstance(cache, (list, tuple)) else cache
            for k in (entries or {}).values():
                md = getattr(k, "metadata", None)
                best = max(best, getattr(md, "shared", 0) or 0)
    return best

print(f"{'dtype':6s} {'BL':>4s} {'BP':>4s} {'stg':>3s} {'TRUE':>8s} {'handcount':>10s} {'ratio':>6s}")
rows = []
for dt in ("fp16", "tf32"):
    for bl, bp, stg in itertools.product((32, 64), (32, 64), (2, 4)):
        over = {"COMPUTE_DTYPE": dt, "BC_CACHE_DTYPE": dt if dt == "fp16" else "fp32",
                "BLOCK_L": bl, "BLOCK_P": bp, "NUM_STAGES": stg}
        p = W / f"cg2_{dt}_{bl}_{bp}_{stg}.py"
        p.write_text(build(over))
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            m = mod.ModelNew(B, L, H, P, N, BLK).to(dev).eval()
            x = torch.rand(B, L, H, P, device=dev)
            with torch.no_grad():
                m(x)                      # one real call -> Triton compiles -> cache filled
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{dt:6s} {bl:4d} {bp:4d} {stg:3d}  FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        t = true_shared(mod)
        w = 2 if dt == "fp16" else 4
        hand = stg * (bl * bp + 2 * bl * N) * w
        if t:
            rows.append((t, hand))
            print(f"{dt:6s} {bl:4d} {bp:4d} {stg:3d} {t:8d} {hand:10d} {hand/t:6.2f}")
        del m, x
        torch.cuda.empty_cache()

if rows:
    import statistics
    rs = [h / t for t, h in rows]
    print(f"\n  n={len(rows)} handcount/TRUE median {statistics.median(rs):.2f} "
          f"min {min(rs):.2f} max {max(rs):.2f}  "
          f"under(<0.95) in {sum(1 for r in rs if r < 0.95)}/{len(rs)}")
else:
    print("\n  no rows")
