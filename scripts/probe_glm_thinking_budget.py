"""Is 32000 a thinking budget attached to glm-5.3's `reasoning_effort` tier?

All three glm-5.3 generator attempts in `run-l3-21-20260906-084636` stopped at exactly
`output + reasoning == 32000` with `finish: "length"`, mid-reasoning, before ever emitting a
tool call. The endpoint accepts `max_tokens: 120000` for a trivial prompt, so 32000 is not a
blanket request cap.

Two hypotheses remain:

  A. the harness/opencode sends `max_tokens: 32000`;
  B. `reasoning_effort: "max"` grants a 32000-token thinking budget the model spends in full
     on a hard prompt, and truncation lands on the budget, not on a request field.

This decides between them from outside opencode: the same deliberately hard prompt at each
effort tier, with `max_tokens` set far above 32000 in every case. If the stop lands at 32000
regardless of `max_tokens`, and moves when the tier moves, it is B.

Costs a few cents per tier.
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

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
        "stream": False,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read()[:500].decode("utf-8", "replace")}


print(f"{'effort':>8} {'max_tokens':>11} {'completion':>11} {'reasoning':>10} {'finish':>10}")
for effort, mt in [("max", 120000), ("max", 40000), ("high", 120000), ("low", 120000)]:
    d = call(effort, mt)
    if "_http_error" in d:
        print(f"{effort:>8} {mt:>11}  HTTP {d['_http_error']}: {d['_body'][:120]}")
        continue
    u = d.get("usage") or {}
    comp = u.get("completion_tokens")
    rea = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    fin = [c.get("finish_reason") for c in d.get("choices", [])]
    print(f"{effort:>8} {mt:>11} {comp:>11} {rea:>10} {str(fin):>10}")
