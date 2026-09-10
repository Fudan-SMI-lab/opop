"""Can this box actually make a real agent call? (box environment verification)

`opencode --version` proving the binary exists is not the gate. Neither is the provider config
parsing. The question is whether a REAL prompt reaches the model and comes back THROUGH THE PATH THE
HARNESS ACTUALLY USES -- which is the only thing separating a configured box from one that fails on
its first agent call, 20 minutes into a run.

Two traps this deliberately walks into rather than around, because both have bitten this project:

  1. A sandbox's own `opencode.json` makes it a project root, which STOPS opencode's upward search
     for configuration. A repo-local provider is therefore INVISIBLE from inside a sandbox:
     `zhipuai/glm-5.3` failed `ProviderModelNotFoundError` on every attempt until its provider block
     was injected into the sandbox config. So a test that prompts from a bare directory can pass on
     a box where every real agent call would fail -- it would be resolving the provider by walking
     up to `.opencode/`, which the harness never gets to do.

  2. Every permission opencode may ask about must be named in that config. An omitted key defaults to
     `ask`, and an ask is fatal headless: the call sits idle until the ceiling kills it (28 min,
     observed). Using the harness's own PERMISSION_CONFIG is the only way to test the real thing.

Both are handled by building the sandbox with the harness's own SandboxFactory and feeding it the
same provider block via `sandbox_config_path` -- exactly what wiring.py does for a real run.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.agents.runtime import (  # noqa: E402
    AgentCallError,
    OpencodeClient,
    OpencodeServer,
    resolve_opencode,
)
from kernel_optimizer.agents.sandbox import SandboxFactory  # noqa: E402
from kernel_optimizer.config import OpencodeConfig  # noqa: E402
from kernel_optimizer.wiring import _sandbox_extra_config  # noqa: E402


class _Shim:
    """Minimal stand-in for AppConfig so the real `_sandbox_extra_config` can be reused.

    Reusing the harness's own function matters more than tidiness: if it ever changes which keys it
    copies into a sandbox, this verification changes with it instead of quietly testing the old
    behaviour.
    """

    def __init__(self, opencode):
        self.opencode = opencode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch-cwd", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sandbox-config", default=None,
                    help="the .opencode/opencode.jsonc holding the provider block")
    ap.add_argument("--work", default="/tmp/verify-agent-env")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    print("opencode binary : %s" % resolve_opencode())
    cfg = OpencodeConfig(
        launch_cwd=Path(args.launch_cwd),
        sandbox_config_path=Path(args.sandbox_config) if args.sandbox_config else None,
        server_env={"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "131072"},
        request_timeout_s=args.timeout,
    )

    # Build the sandbox the way a real call does, and SAY whether the provider block made it in --
    # its absence is the single most likely reason a fresh box fails, and it is silent otherwise.
    extra = _sandbox_extra_config(_Shim(cfg))
    providers = sorted((extra.get("provider") or {}).keys())
    want_provider = args.model.partition("/")[0]
    print("sandbox providers: %s" % (providers or "NONE"))
    if want_provider not in providers:
        print("  WARN: %r is not among the providers injected into the sandbox. If it is not in the"
              " user's GLOBAL opencode config either, this call will fail"
              " ProviderModelNotFoundError -- which is a config problem, not a network one."
              % want_provider)
    else:
        print("  ok: %r will resolve from inside a sandbox (upward search is blocked there)"
              % want_provider)

    root = Path(args.work)
    root.mkdir(parents=True, exist_ok=True)
    factory = SandboxFactory(root / "sandboxes", extra_config=extra)
    call_id = "verify-%d" % int(time.time())
    sandbox = factory.create(call_id)
    written = json.loads((sandbox.root / "opencode.json").read_text(encoding="utf-8"))
    print("sandbox         : %s" % sandbox.root)
    print("  permission keys: %s" % sorted((written.get("permission") or {}).keys()))

    srv = OpencodeServer(cfg, log_path=root / "opencode-server.log")
    t0 = time.time()
    try:
        base = srv.start()
    except AgentCallError as exc:
        print("FAIL: server did not come up: %s" % exc)
        return 1
    print("server up       : %s  (%.1fs)" % (base, time.time() - t0))
    print("opencode version: %s" % (srv.version() or "unknown"))

    client = OpencodeClient(base, timeout_s=cfg.request_timeout_s)
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "gpu_name": {"type": "string"},
        },
        "required": ["answer", "gpu_name"],
    }
    # A prompt with one trivial reasoning step AND one tool use: a model that can only talk still
    # fails a real agent call, where every module must read files and run commands.
    prompt = (
        "Run the shell command `nvidia-smi --query-gpu=name --format=csv,noheader` and read its "
        "output. Then reply with JSON only: {\"answer\": <2+2>, \"gpu_name\": \"<the GPU name you "
        "saw>\"}. Do not create any files."
    )
    try:
        sid = client.create_session(sandbox.root, "env-verify")
        print("session         : %s" % sid)
        print("prompting %s (idle-abort at %.0fs) ..." % (args.model, client.idle_abort_s))
        t1 = time.time()
        res = client.prompt(sid, prompt, model=args.model, schema=schema,
                            directory=sandbox.root)
        dt = time.time() - t1
        print("returned        : %.1fs  finish=%s  tokens=%s  cost=%s"
              % (dt, res.finish, res.tokens, res.cost))
        if res.error:
            print("server error    : %s" % str(res.error)[:400])
        got = res.structured or None
        print("structured      : %r" % (got,))
        if not got:
            print("text tail       : %r" % (res.text or "")[-400:])
            print()
            print("FAIL: no structured answer came back. The model was reached only if `tokens`"
                  " above is non-empty; check the text tail to tell a refusal from a transport"
                  " problem.")
            return 1
        ok_math = got.get("answer") == 4
        gpu = str(got.get("gpu_name") or "")
        print()
        if ok_math and gpu:
            print("VERDICT: READY -- a real agent call through the harness's own sandbox +"
                  " permission + provider path reached %s, ran a shell command, and returned a"
                  " valid structured answer." % args.model)
            print("  the agent itself read this GPU: %s" % gpu)
            return 0
        # Reached but imperfect: say precisely which half worked. Calling the box unconfigured here
        # would be wrong and would invent work.
        print("PARTIAL: the provider, credentials and permissions all work (a structured reply came"
              " back), but the content is off: answer=%r gpu_name=%r" % (got.get("answer"), gpu))
        print("  math_ok=%s  saw_gpu=%s -- if only the GPU name is empty, the shell permission is"
              " the thing to check." % (ok_math, bool(gpu)))
        return 0
    except Exception as exc:  # noqa: BLE001
        print()
        print("FAIL: %s: %s" % (type(exc).__name__, str(exc)[:700]))
        print("  an unreachable provider, a bad key, a wrong baseURL and a missing provider block"
              " all land here; %s has the server's own view." % srv.log_path)
        return 1
    finally:
        client.close()
        srv.stop()
        print("server stopped")


if __name__ == "__main__":
    raise SystemExit(main())
