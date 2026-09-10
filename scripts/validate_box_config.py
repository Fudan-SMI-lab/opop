"""Does this config's every path exist ON THIS BOX, and does the device block match this card?

A config that parses is not a config that runs. The failure this guards against is specific and has
happened: `load_config` reads exactly ONE file, so an omitted key silently falls back to a pydantic
default -- and on a Linux box those defaults are Windows paths. The run then dies minutes in, or
worse, does not die at all: a wrong `device:` block is written verbatim into every agent's
docs/device.md, so agents are told the wrong shared-memory limit and the wrong capability, and the
only symptom is worse kernels.

So this checks three things a YAML parse cannot:
  1. every path the config names exists, and the venv it names actually has torch+triton;
  2. the `device:` block matches what the GPU in this box reports RIGHT NOW;
  3. the provider config it points at is readable and holds the model's provider.

Run it before the first real run on any new box.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernel_optimizer.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    problems: list[str] = []
    warnings: list[str] = []

    def check_dir(label: str, p) -> None:
        if p is None:
            problems.append("%s is not set" % label)
        elif not Path(p).is_dir():
            problems.append("%s does not exist: %s" % (label, p))
        else:
            print("  ok   %-24s %s" % (label, p))

    def check_file(label: str, p) -> None:
        if p is None:
            problems.append("%s is not set" % label)
        elif not Path(p).is_file():
            problems.append("%s does not exist: %s" % (label, p))
        else:
            print("  ok   %-24s %s" % (label, p))

    print("PATHS")
    check_dir("kernelbench_root", cfg.kernelbench_root)
    check_dir("wsl.kernelbench_src", cfg.wsl.kernelbench_src)
    check_dir("opencode.launch_cwd", cfg.opencode.launch_cwd)
    check_file("sandbox_config_path", cfg.opencode.sandbox_config_path)
    # runs_dir is created on demand; its PARENT must exist.
    runs = Path(cfg.run.runs_dir)
    if not runs.parent.is_dir():
        problems.append("runs_dir's parent does not exist: %s" % runs.parent)
    else:
        print("  ok   %-24s %s (parent exists)" % ("run.runs_dir", runs))

    # The venv is the one that fails as a worker_crash on every job rather than at load.
    venv = Path(str(cfg.wsl.venv)).expanduser()
    py = venv / "bin" / "python"
    if not py.is_file():
        py = venv / "Scripts" / "python.exe"
    if not py.is_file():
        problems.append("wsl.venv has no interpreter: %s" % venv)
    else:
        print("  ok   %-24s %s" % ("wsl.venv", venv))
        import subprocess
        probe = ("import torch,triton,json;"
                 "print(json.dumps({'torch':torch.__version__,'triton':triton.__version__,"
                 "'cuda':torch.cuda.is_available(),"
                 "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                 "'cap':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,"
                 "'vram_gb':round(torch.cuda.get_device_properties(0).total_memory/2**30,1) if torch.cuda.is_available() else None,"
                 "'sms':torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else None,"
                 "'shared_optin':getattr(torch.cuda.get_device_properties(0),'shared_memory_per_block_optin',None)}))")
        out = subprocess.run([str(py), "-c", probe], capture_output=True, timeout=180)
        try:
            info = json.loads(out.stdout.decode().strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            problems.append("wsl.venv cannot import torch+triton: %s"
                            % (out.stderr.decode()[-400:] or out.stdout.decode()[-400:]))
            info = None

        if info:
            print()
            print("THIS BOX, AS THE VENV SEES IT")
            for k in ("torch", "triton", "cuda", "name", "cap", "vram_gb", "sms", "shared_optin"):
                print("  %-14s %s" % (k, info.get(k)))
            if not info.get("cuda"):
                problems.append("torch reports CUDA unavailable in wsl.venv")

            print()
            print("DEVICE BLOCK vs THIS CARD  (a mismatch is written into every agent's device.md)")
            dev = cfg.device
            real_name = info.get("name") or ""
            # Compare on the MODEL tokens, not the first word. Matching `real_name.split()[0]`
            # is vacuous: it is "NVIDIA" for essentially every card, so a config naming a 4090
            # passed this check while sitting on an A800 (caught in the negative control).
            # Require the distinctive tokens -- the ones that are not the vendor or a unit.
            _noise = {"nvidia", "geforce", "rtx", "gtx", "gb", "pcie", "sxm", "sxm4", "laptop", "gpu"}
            real_tokens = [t for t in real_name.lower().replace("-", " ").split()
                           if t not in _noise]
            cfg_low = dev.name.lower()
            missing = [t for t in real_tokens if t not in cfg_low]
            if real_tokens and missing:
                problems.append(
                    "device.name %r does not name this card (%r): missing %s. Agents read this "
                    "string verbatim." % (dev.name, real_name, missing))
            else:
                print("  ok   name      %s  ~  %s" % (dev.name, real_name))

            cap = info.get("cap") or []
            if cap:
                sm = "sm_%d%d" % (cap[0], cap[1])
                if sm not in dev.name:
                    problems.append("device.name says %r but this card is %s -- agents would be told "
                                    "the wrong ISA, which decides which tensor-core paths exist"
                                    % (dev.name, sm))
                else:
                    print("  ok   capability %s" % sm)

            real_vram = info.get("vram_gb")
            if isinstance(real_vram, (int, float)):
                # Tolerate rounding down (23 for 23.5, 79 for 79.3); flag a real gap.
                if abs(dev.vram_gb - real_vram) > max(1.5, 0.05 * real_vram):
                    problems.append("device.vram_gb=%s but this card has %s GiB"
                                    % (dev.vram_gb, real_vram))
                else:
                    print("  ok   vram_gb   %s  ~  %s" % (dev.vram_gb, real_vram))

            real_shared = info.get("shared_optin")
            if isinstance(real_shared, int) and real_shared > 0:
                if dev.max_shared_bytes_optin != real_shared:
                    # THE consequential one: understating it silently forbids feasible tiles,
                    # overstating it makes the feasibility screen admit tiles that cannot compile.
                    problems.append(
                        "device.max_shared_bytes_optin=%d but this card reports %d. This value "
                        "gates which tile sizes agents believe are feasible."
                        % (dev.max_shared_bytes_optin, real_shared))
                else:
                    print("  ok   shared_optin %d" % real_shared)
            else:
                warnings.append("this torch build does not expose shared_memory_per_block_optin, so "
                                "the most consequential device value could not be cross-checked")

    # The provider block: present, and covering the model actually configured.
    print()
    print("PROVIDER")
    try:
        from kernel_optimizer.wiring import _sandbox_extra_config

        class _Shim:
            def __init__(self, oc): self.opencode = oc

        extra = _sandbox_extra_config(_Shim(cfg.opencode))
        provs = sorted((extra.get("provider") or {}).keys())
        want = str(cfg.agents.default_model).partition("/")[0]
        print("  providers in sandbox config: %s" % (provs or "NONE"))
        if want not in provs:
            problems.append("default_model %r needs provider %r, which the sandbox config does not "
                            "define -- every agent call would fail ProviderModelNotFoundError"
                            % (cfg.agents.default_model, want))
        else:
            print("  ok   %r resolves from inside a sandbox" % want)
    except Exception as exc:  # noqa: BLE001
        problems.append("sandbox provider config unreadable: %s: %s" % (type(exc).__name__, exc))

    print()
    for w in warnings:
        print("WARN: %s" % w)
    if problems:
        print("NOT READY -- %d problem(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("READY: every path in %s exists on this box, the venv imports torch+triton with working "
          "CUDA," % args.config)
    print("  the device block matches this card, and the model's provider resolves inside a sandbox.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
