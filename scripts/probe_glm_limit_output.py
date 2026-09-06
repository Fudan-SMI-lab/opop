"""Does `provider.zhipuai.models.glm-5.3.limit.output` lift opencode's 32000 output ceiling?

This supersedes scripts/probe_glm_token_cap_placement.py, whose result was INVALID: every
row of that table, INCLUDING the baseline, printed "(no upstream call)". A baseline that
makes no upstream request means the probe harness itself never got a turn to the provider,
so the five "dropped" placements were never actually tested.

What the config schema (https://opencode.ai/config.json, fetched live) says:

    Config.provider.<id>.models.<id>.limit = {
      "type": "object",
      "properties": {"context": {...}, "input": {...}, "output": {...}},
      "required": ["context", "output"],          <-- BOTH are mandatory
      "additionalProperties": false
    }

`limit.output` is the only output-token field in the whole schema; `maxTokens` and
`maxOutputTokens` do not appear anywhere in it. My earlier `model limit.output` row set
ONLY `output`, violating `required`, which invalidated the model entry -- that is the
`KeyError: 'id'` in the old table, i.e. my own malformed config, not a dropped route.

The arm's glm-5.3 entry currently has NO `limit` block, so opencode falls back to a
default output cap, which is the observed 32000.

Method: a local proxy stands in for the zhipuai endpoint and captures the exact JSON body
opencode sends upstream, then forwards it so the turn completes normally. One trivial PONG
call per variant (~seconds, ~free). The probe FAILS LOUDLY if the baseline captures nothing,
so a broken harness can never again be misread as a negative result.
"""
from __future__ import annotations

import json
import pathlib
import socket
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")
import httpx  # noqa: E402

GLM_ARM = pathlib.Path(r"D:\Pyhon_projects\opop\v2-glm")
# MUST live outside v2-glm: opencode merges every .opencode config from the filesystem root
# down to cwd, so a probe dir nested under v2-glm inherits that arm's REAL baseURL, which
# overrides the proxy override and sends the turn straight to zhipuai. That is why the
# first two probe generations captured nothing while still costing money.
TMP = pathlib.Path(r"D:\ClaudeCode\tmp\glm_captest")
UPSTREAM = "https://open.bigmodel.cn/api/coding/paas/v4"
LIMIT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "maxOutputTokens")
captured: list[dict] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw)
            captured.append({k: body.get(k) for k in LIMIT_KEYS if k in body}
                            or {"(no token-limit field)": True})
        except Exception:
            captured.append({"unparsed": len(raw)})
        req = urllib.request.Request(
            UPSTREAM + self.path, data=raw, method="POST",
            headers={k: v for k, v in self.headers.items()
                     if k.lower() in ("content-type", "authorization", "accept")},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data, code = r.read(), r.status
                ctype = r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data, code, ctype = e.read(), e.code, "application/json"
        except Exception:
            data, code, ctype = b'{"error":"proxy upstream failed"}', 502, "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


CAP = 200000        # asked-for output ceiling
CONTEXT = 400000    # `limit` requires `context` too; generous so it never clamps the ask


def _model(cfg: dict) -> dict:
    return cfg["provider"]["zhipuai"]["models"]["glm-5.3"]


ENV_VAR = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"

VARIANTS: list[tuple[str, object]] = [
    # Baseline is a CONTROL: it must capture a real upstream body, else the probe is broken.
    ("baseline (no limit block)", lambda c: None),
    ("limit{context,output}  <-- schema-valid",
     lambda c: _model(c).__setitem__("limit", {"context": CONTEXT, "output": CAP})),
    ("limit{output} only  (invalid, for the record)",
     lambda c: _model(c).__setitem__("limit", {"output": CAP})),
    # The binary computes  Math.min(model.limit.output, ENV ?? 32000)  -- so the env var is
    # a CEILING and config can only lower it. Both must be raised together.
    ("env var alone", "env"),
    ("env var + limit{context,output}", "env+limit"),
]


def load_arm_config() -> dict:
    raw = (GLM_ARM / ".opencode" / "opencode.jsonc").read_text(encoding="utf-8")
    return json.loads("\n".join(l for l in raw.splitlines()
                                if not l.strip().startswith("//")))


pport = free_port()
srv = HTTPServer(("127.0.0.1", pport), Proxy)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"proxy on 127.0.0.1:{pport} -> {UPSTREAM}\n")

tmp = TMP
(tmp / ".opencode").mkdir(parents=True, exist_ok=True)

rows: list[tuple[str, dict]] = []
print(f"{'placement':44s} token-limit field sent upstream")
for label, mutate in VARIANTS:
    env = dict(os.environ)
    cfg = load_arm_config()
    cfg["provider"]["zhipuai"]["options"]["baseURL"] = f"http://127.0.0.1:{pport}"
    if mutate == "env":
        env[ENV_VAR] = str(CAP)
    elif mutate == "env+limit":
        env[ENV_VAR] = str(CAP)
        _model(cfg)["limit"] = {"context": CONTEXT, "output": CAP}
    elif callable(mutate):
        mutate(cfg)
    (tmp / ".opencode" / "opencode.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    port = free_port()
    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port)], cwd=str(tmp),
        stdout=open(tmp / "srv.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, shell=True, env=env)
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180.0)
    seen: dict = {"(probe error)": True}
    try:
        for _ in range(90):
            try:
                if client.get("/config").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        captured.clear()
        # Verify the proxy override actually SURVIVED config merging before spending a turn.
        # If a parent .opencode reinstated the real baseURL, the request bypasses the proxy
        # and every capture below would be empty for a reason unrelated to token limits.
        live = client.get("/config").json()
        live_base = (((live.get("provider") or {}).get("zhipuai") or {})
                     .get("options") or {}).get("baseURL")
        if live_base != f"http://127.0.0.1:{pport}":
            seen = {"(proxy override lost)": str(live_base)}
            print(f"  {label:42s} {json.dumps(seen)}")
            continue
        live_limit = ((((live.get("provider") or {}).get("zhipuai") or {})
                       .get("models") or {}).get("glm-5.3") or {}).get("limit")
        sid = client.post("/session", json={"title": "cap-probe"}).json()["id"]
        r = client.post(f"/session/{sid}/message", json={
            "model": {"providerID": "zhipuai", "modelID": "glm-5.3"},
            "agent": "build",
            "parts": [{"type": "text", "text": "Reply with exactly: PONG"}],
        }, params={"directory": str(tmp)})
        seen = captured[0] if captured else {"(no upstream call)": True}
        if not captured:
            seen["http"] = r.status_code
            seen["body"] = r.text[:200]
        seen["resolved_limit"] = live_limit
        print(f"  {label:42s} {json.dumps(seen)}")
    except Exception as exc:
        seen = {"(probe error)": f"{type(exc).__name__}: {str(exc)[:70]}"}
        print(f"  {label:42s} {json.dumps(seen)}")
    finally:
        rows.append((label, seen))
        client.close()
        try:
            proc.terminate()
        except Exception:
            pass
        subprocess.run(["powershell", "-NoProfile", "-Command",
            f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty OwningProcess -Unique | "
            "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"],
            capture_output=True)

srv.shutdown()

base = dict(rows[0][1])
print()
if "(no upstream call)" in base or "(probe error)" in base:
    print("PROBE INVALID: the baseline never reached the provider, so no row means anything.")
    print("Fix the probe harness before reading any result below.  (This is exactly how")
    print("probe_glm_token_cap_placement.py produced five bogus negatives.)")
    raise SystemExit(2)

print("baseline reached upstream, so the captures are trustworthy.")
val = rows[1][1]
if any(k in val for k in LIMIT_KEYS):
    got = {k: v for k, v in val.items() if k in LIMIT_KEYS}
    print(f"LIVE ROUTE: limit{{context,output}} -> upstream {got}")
    print("Next: confirm end-to-end that a real turn now exceeds 32000 output tokens.")
else:
    print("limit{context,output} did NOT add a token-limit field to the upstream request.")
