"""Do two boxes' Triton compilers agree on the RESOURCE MAP? Digest a whole sweep and compare.

WHY A SWEEP AND NOT ONE KERNEL. The question this answers -- "can the toolchains be unified so a
control run's two arms may sit on different boxes" -- cannot be settled by version numbers, and it
cannot be settled by one agreeing configuration either. The resource map is measured to be
non-monotone (13 non-monotone slices; BK 16->32 dropped 58 registers while 32->64 added 87 and hit
the 255 cap) and not separable (0 of 10 one-step deltas agreed), so two compilers can agree on a
tile and disagree two tiles away. What has to match is the map over the space the tuner samples.

WHY THE DIGEST IS NOT VACUOUS. A digest over rows that are all identical, or all failures, matches
trivially. Measured on the 4090s at the time this was written: 162 configurations, 122 DISTINCT
(regs, shared, spills) triples, registers spanning 38..255, shared up to 98304 B, 19 rows with
spills, 6 rows failing to launch. The script prints those counts alongside the digest so a reader
can see the comparison had something to compare.

THE NEGATIVE CONTROL IS THE POINT. A probe that only ever returns good news is indistinguishable
from a correct one -- this project has already read five negative "results" off a broken probe. So
this script is only trustworthy when a box that SHOULD differ does: the A800 (sm_80, Triton 3.4.0,
166912 B of shared memory) returns a different digest and 0 launch failures where the 4090s
(101376 B) have 6. If the A800 ever matched a 4090, the script would be broken, not the boxes
identical.

Usage -- run in a directory that is NOT /tmp (a stale /tmp/nt.py on one box shadowed the stdlib's
ntpath and crashed the interpreter before main() ran):

    mkdir -p ~/probe-clean && cd ~/probe-clean
    <venv>/bin/python sweep_resource_map.py

Then compare the DIGEST lines across boxes. Equal digest => same compiler decisions on every
configuration => the two boxes are interchangeable for a paired run. Unequal => they are not, and
`sweep_out.txt` says which rows differ.
"""

from __future__ import annotations

import hashlib
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A, B_, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        PREC: tl.constexpr):
    """A plain tiled GEMM. Deliberately ordinary: the point is to exercise the compiler's
    register/shared allocator across tile shapes, warp counts and dot precisions -- the four axes
    the tuner actually samples -- not to be a good kernel."""
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.)
        b = tl.load(B_ + rk[:, None] * N + rn[None, :],
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.)
        acc += tl.dot(a, b, input_precision=PREC)
    tl.store(C + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def main() -> int:
    if not torch.cuda.is_available():
        print("NO CUDA -- nothing measured, and an absent reading must not read as agreement")
        return 2

    print("torch %s triton %s %s" % (torch.__version__, triton.__version__,
                                     torch.cuda.get_device_name(0)))
    M = N = K = 512
    A = torch.randn(M, K, device="cuda")
    B_ = torch.randn(K, N, device="cuda")
    C = torch.empty(M, N, device="cuda")

    rows: list[str] = []
    for BM in (32, 64, 128):
        for BN in (32, 64, 128):
            for BK in (16, 32, 64):
                for warps in (2, 4, 8):
                    for prec in ("tf32", "ieee"):
                        try:
                            c = _mm[(M // BM, N // BN)](
                                A, B_, C, M, N, K, BM=BM, BN=BN, BK=BK, PREC=prec,
                                num_warps=warps)
                            rows.append("%d,%d,%d,%d,%s,%d,%d,%d" % (
                                BM, BN, BK, warps, prec,
                                c.n_regs, c.metadata.shared, c.n_spills))
                        except Exception as exc:  # noqa: BLE001 -- a config that cannot launch is
                            # DATA, not an error: which tiles are infeasible is part of the map, and
                            # it is where the two card families legitimately differ.
                            rows.append("%d,%d,%d,%d,%s,FAIL,%s" % (
                                BM, BN, BK, warps, prec, type(exc).__name__))

    body = "\n".join(rows)
    triples = {tuple(r.split(",")[5:8]) for r in rows if "FAIL" not in r}
    regs = sorted(int(r.split(",")[5]) for r in rows if "FAIL" not in r)
    shared = sorted(int(r.split(",")[6]) for r in rows if "FAIL" not in r)
    spills = sum(1 for r in rows if "FAIL" not in r and int(r.split(",")[7]) > 0)

    print("configs: %d  launch failures: %d" % (len(rows), sum(1 for r in rows if "FAIL" in r)))
    # Printed so a matching digest can be read as a real claim rather than taken on trust.
    print("spread (a flat sweep would make the digest vacuous): %d distinct (regs,shared,spills) "
          "triples, regs %d..%d, shared %d..%d, %d rows with spills"
          % (len(triples), regs[0], regs[-1], shared[0], shared[-1], spills))
    if len(triples) < 10:
        print("!! FEWER THAN 10 DISTINCT TRIPLES -- this sweep is too flat for its digest to mean "
              "much; widen the axes before comparing boxes")
    print("DIGEST %s" % hashlib.sha256(body.encode()).hexdigest()[:32])
    with open("sweep_out.txt", "w", encoding="utf-8") as f:
        f.write(body)
    print("per-config rows written to sweep_out.txt (diff this between boxes to see WHICH rows "
          "differ when the digests do not match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
