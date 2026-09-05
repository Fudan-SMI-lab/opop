"""Feasibility test: can the harness drive a real opencode agent on glm-5.3 at a given variant?

The user asked for a second experiment using `zhipuai/glm-5.3`, variant `xhigh`, from its own
config directory and its own opencode server. Four things have to hold, and each is checked
against a live server rather than reasoned about:

  1. A server launched from THIS directory sees only the zhipuai provider and offers glm-5.3.
  2. The variant can actually be selected on a prompt (the SDK types carry `model.variant`,
     but the harness's own `prompt()` sends only providerID/modelID).
  3. A real agent session returns usable output -- text and a structured JSON payload, since
     every harness agent depends on schema-shaped replies.
  4. The reply is genuinely produced by glm-5.3, confirmed from the session record rather
     than from the model's self-description (models are unreliable about their own identity).

Run from D:\Pyhon_projects\opop\v2-glm with the harness venv on the path:
  python probe_agent_call.py [variant]
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")

import httpx  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "xhigh"
MODEL = "zhipuai/glm-5.3"

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "shared_bytes": {"type": "integer"},
        "fits": {"type": "boolean"},
    },
    "required": ["answer", "shared_bytes", "fits"],
    "additionalProperties": False,
}

TASK = (
    "A Triton matmul uses BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_stages=3, fp32 inputs. "
    "Shared memory is BLOCK_K*(BLOCK_M+BLOCK_N)*4 bytes per stage. The limit is 101376 bytes. "
    "Compute the total shared memory in bytes, and whether it fits. "
    'Reply with ONLY a JSON object: {"answer": "<one sentence>", '
    '"shared_bytes": <int>, "fits": <true|false>}'
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    port = free_port()
    log = HERE / "probe-server.log"
    print(f"launching opencode serve on port {port}, cwd={HERE}")
    env = dict(os.environ)
    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port)],
        cwd=str(HERE), stdout=open(log, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, env=env, shell=True,
    )
    base = f"http://127.0.0.1:{port}"
    client = httpx.Client(base_url=base, timeout=900.0)
    try:
        # 1. server up, and which providers it sees
        for _ in range(60):
            try:
                if client.get("/config").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            print("server never became healthy; log tail:")
            print(log.read_text(encoding="utf-8")[-1500:])
            return 1

        provs = client.get("/config/providers").json()
        plist = provs.get("providers") or provs
        print("\n[1] providers this server sees:")
        found = None
        for p in plist if isinstance(plist, list) else []:
            models = list((p.get("models") or {}).keys())
            print(f"    {p.get('id')}: {len(models)} models")
            if p.get("id") == "zhipuai":
                m = (p.get("models") or {}).get("glm-5.3") or {}
                found = list((m.get("variants") or {}).keys())
        print(f"    glm-5.3 declared variants: {found}")
        if found is None:
            print("    FAIL: zhipuai/glm-5.3 not offered by this server")
            return 1
        variant_declared = VARIANT in found

        # 2+3. real agent session with the variant and a JSON schema
        sid = client.post("/session", json={"title": f"glm-probe-{VARIANT}"}).json()["id"]
        print(f"\n[2] session {sid}")
        body = {
            "model": {"providerID": "zhipuai", "modelID": "glm-5.3", "variant": VARIANT},
            "agent": "build",
            "parts": [{"type": "text", "text": TASK}],
            "format": {"type": "json_schema", "schema": SCHEMA, "retryCount": 2},
        }
        t0 = time.time()
        resp = client.post(f"/session/{sid}/message", json=body,
                           params={"directory": str(HERE)})
        dur = time.time() - t0
        print(f"    POST message -> HTTP {resp.status_code} in {dur:.1f}s")
        if resp.status_code != 200:
            print(f"    body: {resp.text[:600]}")
            return 1
        data = resp.json()
        info = data.get("info") or {}
        parts = data.get("parts") or []
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
        print(f"    reply text ({len(text)} chars): {text.strip()[:200]}")
        structured = info.get("structured") or data.get("structured")
        print(f"    structured payload: {json.dumps(structured)[:200] if structured else None}")
        if not structured:
            fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
            blob = fenced[-1] if fenced else (
                text[text.find("{"):text.rfind("}") + 1] if "{" in text else "")
            if blob:
                try:
                    structured = json.loads(blob)
                    print(f"    fenced-JSON fallback parsed: {structured}")
                except json.JSONDecodeError as exc:
                    print(f"    fallback parse failed: {exc}")

        # 4. which model actually served it, from the session record
        sess = client.get(f"/session/{sid}").json()
        smodel = sess.get("model") or {}
        tokens = sess.get("tokens") or {}
        print(f"\n[3] session record says model={smodel}")
        print(f"    tokens: {{k: v for k, v in tokens.items() if k != 'cache'}}"
              .replace("{k: v for k, v in tokens.items() if k != 'cache'}",
                       str({k: v for k, v in tokens.items() if k != "cache"})))
        msgs = client.get(f"/session/{sid}/message").json()
        rows = msgs if isinstance(msgs, list) else msgs.get("messages") or []
        for m in rows:
            mi = m.get("info") or m
            if mi.get("role") == "assistant":
                print(f"    assistant message model: providerID={mi.get('providerID')} "
                      f"modelID={mi.get('modelID')} variant={mi.get('variant')}")

        print("\n=== verdict ===")
        ok_json = bool(structured) and structured.get("shared_bytes") == 49152
        print(f"  server offers zhipuai/glm-5.3            : YES")
        print(f"  variant {VARIANT!r} declared in config      : "
              f"{'YES' if variant_declared else 'NO'}")
        print(f"  prompt with that variant accepted        : "
              f"{'YES' if resp.status_code == 200 else 'NO'}")
        print(f"  usable reply returned                   : {'YES' if text else 'NO'}")
        print(f"  schema-shaped JSON obtained             : {'YES' if structured else 'NO'}")
        print(f"  arithmetic correct (49152 bytes)        : {'YES' if ok_json else 'NO'}"
              f"   (got {structured.get('shared_bytes') if structured else None})")
        return 0
    finally:
        client.close()
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
                        "Select-Object -ExpandProperty OwningProcess -Unique | "
                        "ForEach-Object {{ Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }}"],
                       capture_output=True)
        print(f"\n(server stopped; log at {log})")


if __name__ == "__main__":
    sys.exit(main())
