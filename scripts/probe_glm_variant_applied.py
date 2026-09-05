"""Where, if anywhere, does the per-message variant show up?

The session record reports variant='default' for BOTH xhigh (undeclared) and max
(declared), so that field evidently reflects the session default rather than the variant
sent on a message. This dumps the assistant message record in full to find the field that
does carry it -- or to establish that nothing does, which would mean the variant cannot be
verified from the API and must be set another way.
"""
import json, os, re, socket, subprocess, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")
import httpx

HERE = Path(__file__).resolve().parent
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "max"

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

port = free_port()
proc = subprocess.Popen(["opencode", "serve", "--port", str(port)], cwd=str(HERE),
    stdout=open(HERE/"probe-server2.log","w",encoding="utf-8"),
    stderr=subprocess.STDOUT, shell=True)
client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=900.0)
try:
    for _ in range(60):
        try:
            if client.get("/config").status_code == 200: break
        except httpx.HTTPError: pass
        time.sleep(1)
    sid = client.post("/session", json={"title": f"vp-{VARIANT}"}).json()["id"]
    body = {"model": {"providerID":"zhipuai","modelID":"glm-5.3","variant":VARIANT},
            "agent":"build",
            "parts":[{"type":"text","text":"Reply with exactly: PONG"}]}
    r = client.post(f"/session/{sid}/message", json=body, params={"directory": str(HERE)})
    print(f"variant sent: {VARIANT!r}  HTTP {r.status_code}")
    info = (r.json().get("info") or {})
    print("\n--- response info keys ---")
    print(sorted(info.keys()))
    for k in ("providerID","modelID","variant","model","reasoningEffort","options"):
        if k in info: print(f"  {k} = {json.dumps(info[k])[:200]}")
    msgs = client.get(f"/session/{sid}/message").json()
    rows = msgs if isinstance(msgs,list) else (msgs.get("messages") or [])
    print(f"\n--- {len(rows)} messages; assistant records ---")
    for m in rows:
        mi = m.get("info") or m
        if mi.get("role") != "assistant": continue
        keep = {k:v for k,v in mi.items() if k in
                ("role","providerID","modelID","variant","model","reasoningEffort","options","tokens")}
        print(json.dumps(keep, ensure_ascii=False)[:600])
finally:
    client.close()
    try: proc.terminate()
    except Exception: pass
    subprocess.run(["powershell","-NoProfile","-Command",
        f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique | "
        "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"],
        capture_output=True)
