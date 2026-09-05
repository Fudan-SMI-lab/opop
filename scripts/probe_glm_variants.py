"""Probe the zhipuai endpoint directly: does glm-5.3 accept reasoning_effort "xhigh"?

The user asked for variant `xhigh`. The repo's own opencode config declares only
low/high/max for glm-5.3, and its comment says the API accepts exactly those; `xhigh`
appears in the config tree only for GPT models (`.omo/omo_gpt.jsonc`). Before concluding
the request is unsatisfiable, ask the endpoint.

Sends the same minimal chat completion four times, with reasoning_effort xhigh / max /
high / omitted, and reports status plus which model the response claims. Reads the API key
from the repo config; never prints it.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CONFIG = Path(r"D:\Pyhon_projects\opop\.opencode\opencode.jsonc")


def load_zhipu() -> tuple[str, str]:
    lines = [ln for ln in CONFIG.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("//")]
    cfg = json.loads("\n".join(lines))
    z = cfg["provider"]["zhipuai"]["options"]
    return z["baseURL"], z["apiKey"]


def call(base: str, key: str, effort: str | None) -> None:
    body = {
        "model": "glm-5.3",
        "messages": [{"role": "user",
                      "content": "Reply with exactly: OK. Then state your model name."}],
        "max_tokens": 200,
        "stream": False,
    }
    if effort is not None:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    label = f"reasoning_effort={effort!r}" if effort is not None else "reasoning_effort omitted"
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = (msg.get("content") or "").strip().replace("\n", " ")
        usage = data.get("usage") or {}
        print(f"  {label:34s} HTTP {resp.status}  model={data.get('model')!r}")
        print(f"      reply: {text[:110]}")
        print(f"      usage: prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')} "
              f"reasoning={(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300].replace("\n", " ")
        print(f"  {label:34s} HTTP {e.code}  ERROR: {detail}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:34s} {type(exc).__name__}: {exc}")


def main() -> int:
    base, key = load_zhipu()
    print(f"endpoint: {base}\n")
    for effort in ("xhigh", "max", "high", None):
        call(base, key, effort)
    print("\nIf xhigh returns an error while max/high succeed, glm-5.3 has no xhigh tier")
    print("and the closest available setting is max (the config's deepest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
