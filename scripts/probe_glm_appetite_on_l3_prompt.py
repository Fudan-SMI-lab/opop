"""How many tokens does glm-5.3 actually want for the L3:21 generator task?

The question survived two failed config placements (`provider...models.glm-5.3.options.maxTokens`
and `agent.build.maxTokens`, both silently dropped), so it cannot currently be answered through
opencode: every agent call is capped at opencode's 32000 default and truncates there.

It CAN be answered by going around opencode. This sends the real generator prompt straight to the
zhipuai endpoint with `max_tokens: 100000` and reports where the model stops. What that measures
and what it does not:

  * MEASURES the model's reasoning appetite on this task -- the number the config question is
    really about, and the thing that decides whether a 100k cap would even help.
  * DOES NOT reproduce the agent loop: no tools, so the model cannot read the sandbox files or
    write candidates. It will plan and then answer in prose. The token count is the datum; the
    content is not a candidate.

Because the prompt normally tells the agent to read files it cannot open here, the file-reading
instructions are replaced by the file CONTENTS inline, so the model has the same information the
real generator had after its two successful tool calls (ref.py, the contract, the device doc, the
eval semantics, the triton pitfalls).

Cost roughly $0.5 and up to ~25 min at GLM's measured ~4400 tok/min. Non-streaming with a long
read timeout; three earlier probe shapes died in the HTTP layer (see
scripts/probe_glm_thinking_budget.py for that history).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")
import httpx  # noqa: E402

V2 = pathlib.Path(r"D:\Pyhon_projects\opop\v2")
GLM_ARM = pathlib.Path(r"D:\Pyhon_projects\opop\v2-glm")
KB = pathlib.Path(r"D:\Pyhon_projects\opop\KernelBench")

cfg = json.loads("\n".join(
    l for l in (GLM_ARM / ".opencode" / "opencode.jsonc").read_text(encoding="utf-8").splitlines()
    if not l.strip().startswith("//")))
opts = cfg["provider"]["zhipuai"]["options"]
URL = opts["baseURL"].rstrip("/") + "/chat/completions"
KEY = opts["apiKey"]

ref = (KB / "KernelBench" / "level3" / "21_EfficientNetMBConv.py").read_text(encoding="utf-8")
contract = (V2 / "src" / "kernel_optimizer" / "agents" / "prompts"
            / "candidate_contract.md").read_text(encoding="utf-8")
pitfalls = (V2 / "src" / "kernel_optimizer" / "agents" / "prompts"
            / "triton_pitfalls.md").read_text(encoding="utf-8")

PROMPT = f"""You are optimizing a GPU operator from KernelBench.

The reference PyTorch implementation (task: 21_EfficientNetMBConv, level 3):

```python
{ref}
```

The candidate contract you must follow:

{contract}

Triton rules that are compiler/correctness hard constraints:

{pitfalls}

Target device: RTX 5080 Laptop (sm_120), 16 GB, 255 regs/thread, 49152 B static shared
(101376 B opt-in), 1024 threads/block.

Evaluation semantics: the reference is evaluated in TRAIN mode, so BatchNorm uses BATCH
statistics, not running stats. Match that exactly.

Write 4 candidate kernel implementations, each in its own file `candidates/cand_1.py`,
`candidates/cand_2.py`, ... Each must follow the contract exactly (ModelNew + a PARAMS dict
of tunable knobs).

CRITICAL: the candidates must differ in COMPUTATIONAL APPROACH, not just in parameter
defaults or code style.

When done, answer with JSON:
{{"candidates": [{{"file": "candidates/cand_1.py", "backend": "triton",
  "approach_summary": "<1-2 sentences>", "structural_axes": ["<axis>", ...]}}, ...]}}
"""

print(f"prompt: {len(PROMPT)} chars")
body = {
    "model": "glm-5.3",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 100000,
    "reasoning_effort": "max",
    "stream": False,
}
t0 = time.time()
with httpx.Client(timeout=httpx.Timeout(2400.0, connect=30.0)) as client:
    r = client.post(URL, json=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    elapsed = time.time() - t0
    print(f"HTTP {r.status_code} after {elapsed/60:.1f} min")
    if r.status_code != 200:
        print(r.text[:600])
        raise SystemExit(1)
    d = r.json()

u = d.get("usage") or {}
comp = u.get("completion_tokens")
rea = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
fin = [c.get("finish_reason") for c in d.get("choices") or []]
print(f"\ncompletion_tokens : {comp}")
print(f"reasoning_tokens  : {rea}")
print(f"answer tokens     : {(comp or 0) - (rea or 0)}")
print(f"finish_reason     : {fin}")
print(f"wall clock        : {elapsed/60:.1f} min  ({(comp or 0)/max(elapsed,1)*60:.0f} tok/min)")
print(f"\nverdict: {'STOPPED ON ITS OWN' if 'length' not in fin else 'STILL TRUNCATED at 100000'}")
msg = (d.get("choices") or [{}])[0].get("message") or {}
txt = msg.get("content") or ""
print(f"answer length: {len(txt)} chars")
print("answer head:", txt[:400].replace("\n", " "))
