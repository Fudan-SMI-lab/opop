"""Is 32000 a thinking budget attached to glm-5.3's `reasoning_effort` tier?

All three glm-5.3 generator attempts in `run-l3-21-20260906-084636` stopped at exactly
`output + reasoning == 32000` with `finish: "length"`, mid-reasoning, before ever emitting a
tool call. The endpoint accepts `max_tokens: 120000` for a trivial prompt, and opencode sends
no token-limit field at all, so 32000 is neither a blanket request cap nor something our side
asks for. The remaining explanation is a per-tier thinking budget.

This decides it from outside opencode: the same deliberately hard prompt at each tier, with
`max_tokens` far above 32000 every time. If the stop lands at 32000 regardless of `max_tokens`
and moves with the tier, the budget is the tier's.

Non-streaming with a long timeout and a transport retry. Three earlier shapes of this probe
all died in the HTTP layer rather than producing a number, which is worth recording so nobody
repeats them:

  * non-streaming, 900s read timeout  -> TimeoutError (a max-tier call on this prompt is longer)
  * streamed, 300s between-chunk       -> ReadTimeout (the endpoint BUFFERS reasoning; there is
                                          no incremental traffic to satisfy a short gap timeout)
  * streamed, 1800s between-chunk      -> RemoteProtocolError, incomplete chunked read

So: no streaming (nothing is gained when the server buffers anyway), 2400s, and one retry on a
transport error. Each attempt costs roughly $0.15 and up to 20 minutes.

Two calls, both with `max_tokens: 40000` — above the observed 32000 stop, so `max_tokens` cannot
be the binding constraint:

  * `max`  — if it stops at exactly 32000, the ceiling is the tier's, not the request's.
  * `high` — if it stops lower, the budget tracks the tier; if it completes, `high` fits and
    lowering the tier is a viable option for the arm.

Deliberately NOT tested with a real agent call: `agent-smoke` on an L3 task would take the GPU
lock, and the two arms' `runs_dir` differ so their locks are different files that do not see
each other. While a gpt run is live that would corrupt its timings, which costs far more than
this probe answers.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
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


def call_once(effort: str | None, max_tokens: int) -> dict:
    body = {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": HARD}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    with httpx.Client(timeout=httpx.Timeout(2400.0, connect=30.0)) as client:
        resp = client.post(
            URL,
            json=body,
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            return {"_http": resp.status_code, "_body": resp.text[:400]}
        d = resp.json()
    return {
        "usage": d.get("usage") or {},
        "finish": [c.get("finish_reason") for c in d.get("choices") or []],
    }


def call(effort: str | None, max_tokens: int) -> dict:
    """One transport retry: three earlier probe shapes died in the HTTP layer, not upstream."""
    last: dict = {}
    for attempt in (1, 2):
        try:
            return call_once(effort, max_tokens)
        except (httpx.HTTPError, httpx.RemoteProtocolError) as exc:
            last = {"_transport": f"{type(exc).__name__}: {str(exc)[:200]}", "_attempt": attempt}
            print(f"   attempt {attempt} transport failure: {last['_transport']}", flush=True)
    return last


print(f"{'effort':>8} {'max_tokens':>11} {'completion':>11} {'reasoning':>10}  finish", flush=True)
for effort, mt in [("max", 40000), ("high", 40000)]:
    d = call(effort, mt)
    if "_http" in d:
        print(f"{effort:>8} {mt:>11}  HTTP {d['_http']}: {d['_body'][:120]}", flush=True)
        continue
    if "_transport" in d:
        print(f"{effort:>8} {mt:>11}  gave up after 2 transport failures", flush=True)
        continue
    u = d["usage"] or {}
    comp = u.get("completion_tokens")
    rea = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(f"{effort:>8} {mt:>11} {str(comp):>11} {str(rea):>10}  {d['finish']}", flush=True)
