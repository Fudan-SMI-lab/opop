"""SUPERSEDED AND INVALID -- use scripts/probe_glm_limit_output.py instead.

This probe's result table was worthless and its own output said so: EVERY row, including the
baseline control, printed "(no upstream call)". A baseline that never reaches the provider means
the harness was broken, so the five "dropped" placements measured nothing at all.

The bug: it wrote its throwaway config to `v2-glm/_captest`, INSIDE the arm's tree. opencode
merges every .opencode config from the filesystem root down to cwd, so the parent's real baseURL
was layered back over the proxy override and every turn went straight to zhipuai -- costing
~$0.024 per row while capturing nothing.

The answer it was looking for is not a config key at all. opencode computes the per-turn cap as
`min(model.limit.output, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX ?? 32000)`, so the setting is an
ENVIRONMENT VARIABLE with no config-file route. See docs/finding-glm-truncated-before-acting.md,
section "The cap's real route: an env var, not a config key".

Kept only as the record of a probe whose negative result and whose own breakage looked identical.
The replacement asserts its baseline reaches upstream and exits non-zero if it does not.

--- original docstring below ---

Which config key, if any, lifts opencode's 32000 output ceiling for a glm-5.3 turn?

Two placements have already been ruled out live, each costing a ~12 min L3 generator call:

  * `provider.zhipuai.models.glm-5.3.options.maxTokens` -> ignored, turn stopped at 32000
  * `agent.build.maxTokens`                             -> ignored, turn stopped at 32000

The reason the second failed is visible in opencode's own schema: `$AgentConfig.properties`
LISTS `maxTokens` but its spec is `null` -- a stub with no type, while sibling keys like
`steps` and `temperature` carry real specs. The resolved `/agent` view of `build` has no
`maxTokens` key at all. So it is accepted by the validator and dropped.

What is still live: `$AgentConfig.options` is `{"type": "object"}` -- a free-form passthrough
that DOES resolve on the build agent (`options: {}`). The underlying Vercel AI SDK names this
setting `maxOutputTokens` (confirmed in @ai-sdk/provider's index.d.ts: "Maximum number of
tokens to generate. maxOutputTokens?: number"). So the plausible remaining routes are
`agent.build.options.maxOutputTokens` and the provider-model `options.maxOutputTokens`.

Rather than burn another 12-minute L3 call per guess, this probes all remaining candidates
CHEAPLY: a trivial prompt whose only job is to reveal whether the request the provider
receives carries a token limit. A local proxy stands in for the zhipuai endpoint, captures
the exact JSON body opencode sends, forwards it upstream, and prints the token-limit fields.
Each variant costs one ~1-second PONG call instead of a full generator turn.

Technique borrowed from v2-glm/probe_effort_reaches_api.py, which established the same
proxy method for `reasoningEffort`.

Nothing outside this script's own temp directory is modified: it writes a throwaway config
under v2-glm/_captest and never touches the real arm's config.
"""
from __future__ import annotations

import json
import pathlib
import socket
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
            captured.append({k: body.get(k) for k in LIMIT_KEYS if k in body} or {"(none)": True})
        except Exception:
            captured.append({"unparsed": len(raw)})
        req = urllib.request.Request(
            UPSTREAM + self.path, data=raw, method="POST",
            headers={k: v for k, v in self.headers.items()
                     if k.lower() in ("content-type", "authorization", "accept")},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data, code = r.read(), r.status
                ctype = r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data, code, ctype = e.read(), e.code, "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


CAP = 100000
# Each variant is a patch applied to the throwaway config: (label, mutator).
VARIANTS: list[tuple[str, callable]] = [
    ("baseline (nothing set)", lambda c: None),
    ("agent.build.options.maxOutputTokens",
     lambda c: c.setdefault("agent", {}).setdefault("build", {})
                .setdefault("options", {}).__setitem__("maxOutputTokens", CAP)),
    ("agent.build.options.max_tokens",
     lambda c: c.setdefault("agent", {}).setdefault("build", {})
                .setdefault("options", {}).__setitem__("max_tokens", CAP)),
    ("model options.maxOutputTokens",
     lambda c: c["provider"]["zhipuai"]["models"]["glm-5.3"]["options"]
                .__setitem__("maxOutputTokens", CAP)),
    ("model options.max_tokens",
     lambda c: c["provider"]["zhipuai"]["models"]["glm-5.3"]["options"]
                .__setitem__("max_tokens", CAP)),
    ("model limit.output",
     lambda c: c["provider"]["zhipuai"]["models"]["glm-5.3"]
                .setdefault("limit", {}).__setitem__("output", CAP)),
]


def load_arm_config() -> dict:
    raw = (GLM_ARM / ".opencode" / "opencode.jsonc").read_text(encoding="utf-8")
    return json.loads("\n".join(l for l in raw.splitlines()
                                if not l.strip().startswith("//")))


pport = free_port()
srv = HTTPServer(("127.0.0.1", pport), Proxy)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"proxy on 127.0.0.1:{pport} -> {UPSTREAM}\n")

tmp = GLM_ARM / "_captest"
(tmp / ".opencode").mkdir(parents=True, exist_ok=True)

print(f"{'placement':38s} {'token-limit fields in the upstream request'}")
for label, mutate in VARIANTS:
    cfg = load_arm_config()
    cfg["provider"]["zhipuai"]["options"]["baseURL"] = f"http://127.0.0.1:{pport}"
    mutate(cfg)
    (tmp / ".opencode" / "opencode.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    port = free_port()
    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port)], cwd=str(tmp),
        stdout=open(tmp / "srv.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, shell=True)
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180.0)
    try:
        for _ in range(60):
            try:
                if client.get("/config").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        captured.clear()
        sid = client.post("/session", json={"title": f"cap-{label[:20]}"}).json()["id"]
        client.post(f"/session/{sid}/message", json={
            "model": {"providerID": "zhipuai", "modelID": "glm-5.3"},
            "agent": "build",
            "parts": [{"type": "text", "text": "Reply with exactly: PONG"}],
        }, params={"directory": str(tmp)})
        seen = captured[0] if captured else {"(no upstream call)": True}
        print(f"  {label:36s} {json.dumps(seen)}")
    except Exception as exc:
        print(f"  {label:36s} ERROR {type(exc).__name__}: {str(exc)[:80]}")
    finally:
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
print("\nA placement that shows a token-limit field is the live route; the others are dropped.")
