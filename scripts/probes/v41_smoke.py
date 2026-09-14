"""V0 smoke: the v4.1 conditional layer end-to-end against a REAL GPU worker, no agents.

What it exercises, in order (each step prints PASS/FAIL loudly):
  1. a real compile-probe batch through prescreen_batch on a real Triton candidate
     (materialized axis points under frozen partners) -> conditioned answers recorded
  2. C4 admission on a derived (or synthetic-fallback) wall -> 4 fresh enqueues into a
     REAL OptunaTPETuner -> 4 real latency measurements via the worker -> mint decision
  3. an E action if minted: direction geometry + fresh enqueue
  4. the FRESH-INTENT positive control on the live tuner: an ordinary re-enqueue of a
     measured point is REFUSED while a fresh re-enqueue is re-drawn and flagged
     (the fake-A/A blocker, live)

Usage (on a GPU box, from the repo root):
  PYTHONPATH=src <orch-venv>/bin/python scripts/probes/v41_smoke.py \
      --config configs/<box config>.yaml --task level1:19

Exits non-zero on any FAIL.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", default="level1:1")
    args = parser.parse_args()

    from kernel_optimizer.conditional.probe import ConditionedWall, ProbePlanner
    from kernel_optimizer.conditional.scanner import ConditionalScanner
    from kernel_optimizer.conditional.tokens import TokenStore
    from kernel_optimizer.config import load_config
    from kernel_optimizer.evaluation.correctness import latency_from_result
    from kernel_optimizer.models.core import (
        LatencyStats,
        ParamDomain,
        ParameterSpace,
        ParamSet,
        TrialRecord,
    )
    from kernel_optimizer.paramspace import materializer
    from kernel_optimizer.store.run_store import RunStore
    from kernel_optimizer.tasks.kernelbench import parse_task_arg
    from kernel_optimizer.tuning.tpe import OptunaTPETuner
    from kernel_optimizer.wiring import build_gpu_stack, load_task

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""),
              flush=True)
        if not ok:
            failures.append(name)

    cfg = load_config(Path(args.config)) if args.config else load_config()
    level, pid = parse_task_arg(args.task)
    task = load_task(cfg, level, pid)
    run_id = f"v41-smoke-{int(time.time())}"
    store = RunStore.create(Path(cfg.run.runs_dir), run_id, {"purpose": "v41 smoke"})
    _, evaluator, _, _ = build_gpu_stack(cfg, store)
    check("wiring/task setup", True, f"{task.name} -> {store.run_dir}")

    # Matches level1:1 (4096x4096 square matmul, 64MB tensors — level1:19's 6.4GB input
    # blew both the probe deadline and 24GB VRAM). Real tile knobs => real shared-memory
    # behaviour, so the probe layer sees genuine fit/refused variation. input_precision
    # "ieee": tf32 would fail the correctness gate on a 4096-deep dot.
    candidate_src = '''import torch
import triton
import triton.language as tl

PARAMS = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "N_STAGES": 2}

@triton.jit
def _mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & ((rk[None, :] + k) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((rk[:, None] + k) < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK * sak
        b_ptrs += BK * sbn * 0 + BK * sbk
    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        M, K = a.shape
        K2, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(M, PARAMS["BLOCK_M"]), triton.cdiv(N, PARAMS["BLOCK_N"]))
        _mm[grid](a, b, c, M, N, K,
                  a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                  c.stride(0), c.stride(1),
                  BM=PARAMS["BLOCK_M"], BN=PARAMS["BLOCK_N"], BK=PARAMS["BLOCK_K"],
                  num_stages=PARAMS["N_STAGES"], num_warps=4)
        return c
'''
    space = ParameterSpace(
        space_id="sp-smoke", candidate_id="cand-smoke", version=1, source_sha="smoke",
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64, 128, 256]),
                 ParamDomain(name="BLOCK_N", kind="int", choices=[32, 64, 128, 256]),
                 ParamDomain(name="BLOCK_K", kind="int", choices=[32, 64]),
                 ParamDomain(name="N_STAGES", kind="int", choices=[1, 2, 4])],
        constraints=[])

    # ---- 1. probe layer against the real worker ----
    print("\n== step 1: real compile-probe batch ==", flush=True)
    planner = ProbePlanner(space, cfg.device.max_shared_bytes_optin,
                           legal=lambda v: True, pcap=12)
    incumbent = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "N_STAGES": 2}
    planned = planner.plan_batch(incumbent)
    check("planner produced a batch", len(planned) > 0, f"{len(planned)} points")

    probe_dir = store.run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)
    paths, kept = [], []
    for i, values in enumerate(planned):
        src = materializer.materialize(candidate_src, ParamSet(values=values))
        p = probe_dir / f"pt{i:02d}.py"
        p.write_text(src, encoding="utf-8")
        paths.append(p)
        kept.append(values)
    t0 = time.time()
    evaluator.prescreen_batch(task, paths, tag="v41smoke", backend="triton")
    elapsed = time.time() - t0

    def collect():
        res, n = {}, 0
        for p in paths:
            entry = evaluator.screen_cache_entry(p.read_text(encoding="utf-8"), "triton")
            if entry is None or not entry.get("ok"):
                res[str(p)] = {"ok": False, "reason": "not answered"}
            else:
                res[str(p)] = entry
                n += 1
        return res, n

    results, answered = collect()
    if answered == 0:
        # The FIRST worker job on a cold box pays process start + CUDA context + imports
        # and can blow the 30+3n deadline; the real pipeline's DBudget retries a smaller
        # batch for exactly this case. The smoke retries the same batch once, warm.
        print("cold-start batch returned 0; one warm retry (mirrors DBudget retry)",
              flush=True)
        t0 = time.time()
        evaluator.prescreen_batch(task, paths, tag="v41smoke-r", backend="triton")
        elapsed = time.time() - t0
        results, answered = collect()
    planner.record(results, kept)
    check("probe batch answered", answered > 0, f"{answered}/{len(paths)} in {elapsed:.1f}s")
    # shared-bytes is the wall criterion and is filled at warmup=True; n_regs/n_spills
    # need ptxas and are legitimately None here (a known probe property, not a defect).
    shared_ok = any(any(k.get("shared") is not None for k in a.kernels)
                    for a in planner.answers.values())
    check("shared-bytes present (wall criterion)", shared_ok)
    walls = planner.walls(incumbent)
    print(f"conditioned walls: {[w.payload() for w in walls] or '(none; expected for add)'}",
          flush=True)

    # ---- 2. scanner + real measurements ----
    print("\n== step 2: C4 through the real tuner + worker ==", flush=True)
    tuner = OptunaTPETuner(space, guard_ok=lambda p: True, budget=12, seed=0)
    scanner = ConditionalScanner(space, "cand-smoke", "triton", budget_b=80,
                                 tokens=TokenStore(), seed=0)  # Q=8: room for C4+E1
    wall = walls[0] if walls else ConditionedWall(
        kind="soft", axis="BLOCK_M", partner_key="pk-smoke",
        partner_values={"BLOCK_N": 64, "BLOCK_K": 32, "N_STAGES": 2}, point_map={},
        f_value=32, n_value=64, refused_value=None)
    block = scanner.next_block([wall])
    check("C4 admitted", block is not None and block.kind == "C4",
          f"axis={getattr(block, 'axis', None)}")
    if block is None:
        print(f"FAILURES: {failures}")
        return 1

    def record_of(tid: str, params: ParamSet, ms: float) -> TrialRecord:
        return TrialRecord(
            trial_id=tid, candidate_id="cand-smoke", space_id="sp-smoke",
            params=params, status="complete",
            latency_ms=LatencyStats(mean=ms, median=ms, std=0.0, min=ms, max=ms,
                                    n_samples=20))

    def measure(values: dict) -> float | None:
        src = materializer.materialize(candidate_src, ParamSet(values=values))
        kpath = probe_dir / f"m-{abs(hash(str(sorted(values.items())))) % 10**8}.py"
        kpath.write_text(src, encoding="utf-8")
        res = evaluator.quick_test(task, kpath, tag="v41smoke-m", backend="triton")
        lat = latency_from_result(res)
        return lat.robust_ms if lat is not None else None

    for pt in block.points:
        reason = tuner.enqueue_fresh(ParamSet(values=dict(pt.values)))
        check(f"fresh enqueue {pt.role}", reason is None, str(reason))
    by_key: dict[str, list] = {}
    for pt in block.points:
        by_key.setdefault(ParamSet(values=dict(pt.values)).key(), []).append(pt)
    served, minted = 0, None
    for _ in range(12):
        if served >= 4:
            break
        got = tuner.ask()
        if got is None:
            break
        tid, params = got
        pts = by_key.get(params.key())
        ms = measure(dict(params.values))
        if ms is None:
            check("worker measurement", False, f"no latency for {params.values}")
            break
        tuner.tell(tid, record_of(tid, params, ms))
        if pts:
            pt = pts.pop(0)
            check(f"draw {pt.role} flagged fresh", tuner.is_fresh(tid))
            minted = scanner.tell(block, pt, ms, ok=True)
            served += 1
    check("all four C4 slots measured", served == 4, f"served={served}")
    c = block.contrast
    print(f"C4: full={c.full} g_d={c.g_d} y={c.y} "
          f"mint={c.mint.direction if c.mint else None}", flush=True)
    check("C4 role-keyed completion", bool(c.full and c.g_d is not None and c.y is not None))

    # ---- 3. E action if a token minted ----
    if minted is not None:
        eblock = scanner.next_block([])
        check("E1 admitted via completion hook",
              eblock is not None and eblock.kind == "E1",
              f"direction={eblock.points[0].direction if eblock else None}")
    else:
        print("no token minted (flat axis => unresolved is legitimate); "
              "E geometry is covered by the unit fixtures", flush=True)

    # ---- 4. live fresh-intent control ----
    print("\n== step 4: live fresh-intent control ==", flush=True)
    measured_params = ParamSet(values=dict(block.points[0].values))
    check("ordinary re-enqueue refused (positive control)",
          tuner.enqueue(measured_params) == "already_drawn")
    check("fresh re-enqueue accepted", tuner.enqueue_fresh(measured_params) is None)
    got = tuner.ask()
    ok = (got is not None and got[1].key() == measured_params.key()
          and tuner.is_fresh(got[0]))
    check("fresh re-enqueue re-drawn and flagged", ok)

    print(f"\n{'=' * 50}", flush=True)
    if failures:
        print(f"SMOKE FAILED: {failures}")
        return 1
    print("SMOKE PASSED: probe batch, C4 mint, fresh intent all live on this GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
