"""Does the config's reasoningEffort actually reach the zhipuai API?

The opencode session record does not expose the applied variant, so config correctness
cannot be confirmed from it. This intercepts the traffic instead: a tiny HTTP proxy stands
in for the zhipuai endpoint, records the request body opencode sends, forwards it upstream,
and returns the real response. If `reasoning_effort` appears in the captured body with the
configured value, the setting is reaching the model.
"""
import json, os, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import urllib.request, urllib.error
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Pyhon_projects\opop\v2\src")
import httpx

HERE = Path(__file__).resolve().parent
UPSTREAM = "https://open.bigmodel.cn/api/coding/paas/v4"
captured = []

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw)
            captured.append({"path": self.path, "model": body.get("model"),
                             "reasoning_effort": body.get("reasoning_effort"),
                             "thinking": body.get("thinking"),
                             "keys": sorted(body.keys())})
        except Exception:
            captured.append({"path": self.path, "unparsed": len(raw)})
        req = urllib.request.Request(UPSTREAM + self.path, data=raw, method="POST",
            headers={k: v for k, v in self.headers.items()
                     if k.lower() in ("content-type", "authorization", "accept")})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read(); code = r.status
                ctype = r.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as e:
            data = e.read(); code = e.code; ctype = "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

pport = free_port()
srv = HTTPServer(("127.0.0.1", pport), Proxy)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"proxy listening on 127.0.0.1:{pport}, forwarding to {UPSTREAM}")

# Point a throwaway config at the proxy, keeping everything else identical.
cfg = json.loads((HERE/".opencode"/"opencode.jsonc").read_text(encoding="utf-8"))
cfg["provider"]["zhipuai"]["options"]["baseURL"] = f"http://127.0.0.1:{pport}"
tmp = HERE/"_proxytest"
(tmp/".opencode").mkdir(parents=True, exist_ok=True)
(tmp/".opencode"/"opencode.jsonc").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

port = free_port()
proc = subprocess.Popen(["opencode", "serve", "--port", str(port)], cwd=str(tmp),
    stdout=open(tmp/"server.log","w",encoding="utf-8"), stderr=subprocess.STDOUT, shell=True)
client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=600.0)
try:
    for _ in range(60):
        try:
            if client.get("/config").status_code == 200: break
        except httpx.HTTPError: pass
        time.sleep(1)
    for variant in ("xhigh", "low", None):
        captured.clear()
        sid = client.post("/session", json={"title": f"pt-{variant}"}).json()["id"]
        model = {"providerID":"zhipuai","modelID":"glm-5.3"}
        if variant: model["variant"] = variant
        r = client.post(f"/session/{sid}/message", json={"model": model, "agent":"build",
            "parts":[{"type":"text","text":f"Reply with exactly: PONG-{variant}-{time.time()}"}]},
            params={"directory": str(tmp)})
        label = f"variant={variant!r}" if variant else "no variant sent"
        print(f"\n{label}: HTTP {r.status_code}, {len(captured)} upstream call(s)")
        for c in captured:
            print(f"   model={c.get('model')!r}  reasoning_effort={c.get('reasoning_effort')!r}")
finally:
    client.close()
    try: proc.terminate()
    except Exception: pass
    srv.shutdown()
    subprocess.run(["powershell","-NoProfile","-Command",
        f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique | "
        "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"],
        capture_output=True)
