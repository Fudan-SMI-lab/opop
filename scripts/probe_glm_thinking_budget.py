"""Is 32000 a thinking budget attached to glm-5.3's `reasoning_effort` tier?

All three glm-5.3 generator attempts in `run-l3-21-20260906-084636` stopped at exactly
`output + reasoning == 32000` with `finish: "length"`, mid-reasoning, before ever emitting a
tool call. The endpoint accepts `max_tokens: 120000` for a trivial prompt, and opencode sends
no token-limit field at all, so 32000 is neither a blanket request cap nor something our side
asks for. The remaining explanation is a per-tier thinking budget.

This decides it from outside opencode: the same deliberately hard prompt at each tier, with
`max_tokens` far above 32000 every time. If the stop lands at 32000 regardless of `max_tokens`
and moves with the tier, the budget is the tier's.

Streamed, because a non-streaming `max` call on this prompt exceeds a 900s read timeout (the
first version of this probe died that way). Streaming also keeps the socket alive between
chunks, so the only timeout that matters is the gap between chunks, not total wall time.

Tiers run cheapest-first: `high` is the one that decides whether lowering the tier is even a
viable option, so it should not be gated behind a 20-minute `max` call.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")
import httpx  # noqa: E402

CFG = pathlib.Path(r"D:\Pyhon_projects\opop\v2-glm\.opencode\opencode.jsonc")
cfg = json.loads(
    "\n".join(
        l for l in CFG.read_text(encoding="utf-8").splitlines()
        if not l.strip().startswith("//")
    )
)
opts = cfg["provider"]["zhipuai"]["options"]
URL = opts["baseURL"].rstrip("/") + "/chat/completions"
KEY = opts["apiKey"]

# Deliberately open-ended and arithmetic-heavy: the kind of prompt that made the real
# generator call spend 32k tokens planning an MBConv fusion before writing anything.
HARD = (
    "Plan, in full detail and without writing any code, a Triton implementation of a "
    "fused EfficientNet MBConv block for input (10,112,224,224) fp32 on an RTX 5080 "
    "(sm_120, 16GB): 1x1 expand 112->672 + BatchNorm(train-mode batch statistics) + "
    "ReLU6, then depthwise 5x5 stride 2 pad 2 + BN + ReLU6, then 1x1 project 672->192 "
    "+ BN. For each of four structurally different fusion strategies, work out the exact "
    "tile shapes, the shared-memory bytes per program, the register pressure, the total "
    "bytes moved, and the arithmetic intensity, and compare them numerically. Be "
    "exhaustive and show every calculation."
)


def call(effort: str | None, max_tokens: int) -> dict:
    body = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": HARD}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if effort is not None:
        body["reasoning_effort"] = effort

    usage: dict = {}
    finish: list[str] = []
    chunks = 0
    # 300s between chunks is generous; a live stream emits far more often than that.
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        with client.stream(
            "POST",
            URL,
            json=body,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                return {"_http": resp.status_code, "_body": resp.read()[:400].decode("utf-8", "replace")}
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                chunks += 1
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish.append(ch["finish_reason"])
    return {"usage": usage, "finish": finish, "chunks": chunks}


print(f"{'effort':>8} {'max_tokens':>11} {'completion':>11} {'reasoning':>10}  finish")
for effort, mt in [("high", 120000), ("max", 120000), ("max", 40000), ("low", 120000)]:
    d = call(effort, mt)
    if "_http" in d:
        print(f"{effort:>8} {mt:>11}  HTTP {d['_http']}: {d['_body'][:120]}")
        continue
    u = d["usage"] or {}
    comp = u.get("completion_tokens")
    rea = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(f"{effort:>8} {mt:>11} {str(comp):>11} {str(rea):>10}  {d['finish']}")
    sys.stdout.flush()
