"""Do the reasoning-effort tiers actually differ on glm-5.3? A properly powered test.

An earlier 3-sample run was inconclusive: within-tier spread (high: 175-961 reasoning tokens)
swamped between-tier differences, so it could not say whether "xhigh" is honoured or dropped.

This raises n and reports the distribution, calling the endpoint directly (the opencode layer
is verified separately by probe_agent_call.py). The question is narrow:

    Does reasoning_effort="xhigh" behave like the deepest tier, or like the default?

If xhigh is silently dropped, its distribution should match the omitted-parameter default.
If it maps to max, it should match max. `low` is the control: it must be clearly the smallest,
or the parameter is not being applied at all and the whole comparison is void.

Reads the API key from the repo config; never prints it.
"""
from __future__ import annotations

import json
import statistics
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CONFIG = Path(r"D:\Pyhon_projects\opop\.opencode\opencode.jsonc")
REPEATS = 5
TIERS = ("low", "high", "max", "xhigh", None)

PROMPT = (
    "A Triton kernel tiles a matmul with BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=3, "
    "fp32 inputs, and shared memory BLOCK_K*(BLOCK_M+BLOCK_N)*4 bytes per stage against a "
    "101376-byte limit. Separately, num_warps=4 gives 128 threads and each thread may use at "
    "most 255 registers. Work out: (a) total shared memory and whether it fits, (b) the "
    "largest num_stages that fits, (c) whether doubling BLOCK_M keeps it fitting at "
    "num_stages=2. Show every arithmetic step."
)


def load_zhipu() -> tuple[str, str]:
    lines = [ln for ln in CONFIG.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("//")]
    cfg = json.loads("\n".join(lines))
    z = cfg["provider"]["zhipuai"]["options"]
    return z["baseURL"], z["apiKey"]


def call(args) -> tuple[str, int | None]:
    base, key, effort = args
    body = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4000,
        "stream": False,
        "temperature": 0.0,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    label = effort or "(omitted)"
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return label, None
    details = (data.get("usage") or {}).get("completion_tokens_details") or {}
    return label, details.get("reasoning_tokens")


def main() -> int:
    base, key = load_zhipu()
    print(f"endpoint: {base}")
    print(f"{REPEATS} samples per tier, temperature 0, multi-step arithmetic prompt\n")

    jobs = [(base, key, t) for t in TIERS for _ in range(REPEATS)]
    got: dict[str, list[int]] = {}
    # Serial: concurrent requests hit the provider's rate limit and returned nothing.
    for j in jobs:
        label, rt = call(j)
        if rt is not None:
            got.setdefault(label, []).append(rt)

    print(f"{'tier':12s} {'n':>3s} {'median':>8s} {'mean':>8s} {'min':>6s} {'max':>6s}")
    for t in TIERS:
        label = t or "(omitted)"
        v = sorted(got.get(label, []))
        if not v:
            print(f"{label:12s}   0   (all calls failed)")
            continue
        print(f"{label:12s} {len(v):3d} {statistics.median(v):8.0f} "
              f"{statistics.mean(v):8.0f} {min(v):6d} {max(v):6d}")

    low = got.get("low") or []
    xh = got.get("xhigh") or []
    mx = got.get("max") or []
    de = got.get("(omitted)") or []
    print()
    if low and mx:
        lo_m, mx_m = statistics.median(low), statistics.median(mx)
        print(f"control: low median {lo_m:.0f} vs max median {mx_m:.0f} -> "
              f"{'parameter IS applied' if lo_m < mx_m * 0.7 else 'NO clear tier effect'}")
        if not lo_m < mx_m * 0.7:
            print("  Without a clear low-vs-max separation the xhigh question cannot be")
            print("  answered from token counts at all.")
            return 0
    if xh and mx and de:
        x, m, d = statistics.median(xh), statistics.median(mx), statistics.median(de)
        print(f"xhigh median {x:.0f}   max median {m:.0f}   default median {d:.0f}")
        print(f"  |xhigh-max| = {abs(x - m):.0f}   |xhigh-default| = {abs(x - d):.0f}")
        print(f"  -> xhigh looks like {'MAX' if abs(x - m) < abs(x - d) else 'the DEFAULT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
