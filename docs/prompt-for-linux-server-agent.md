# Prompt: prepare a Linux GPU server for the KernelBench optimization experiments

Copy everything between the `---` markers to the agent working on the Linux server.

---

You are preparing a fresh Linux GPU server (one NVIDIA RTX 4090, sm_89, 24 GB) to run an
LLM-agent harness that searches for faster GPU kernels on KernelBench tasks. Your job is
**preparation, porting and verification only — do not start a real experiment.** Report
back what works, what you changed, and what you could not verify.

## Start here

```bash
git clone -b v2 https://github.com/Fudan-SMI-lab/opop.git
cd opop/v2
cat docs/deploy-on-a-linux-gpu-server.md
```

That document is your specification. It was written from a line-by-line audit of this
codebase and gives you: the seven places that must be ported (section A), environment setup
(B), nine acceptance tests (C), how to run and read an experiment (D), and security notes
(E). Follow it. Where it and this prompt disagree, the document is more precise — but read
this whole prompt first, because it tells you which failures are expected and which mean
you have a real problem.

## The single most important thing to understand

**The harness does not currently run on Linux.** It was built with the orchestrator on
Windows dispatching GPU work into WSL2. This is not a configuration problem — it is about
seven places in the source. If you try to run it before porting, it will fail, and some of
those failures look like something else entirely.

In particular, **`src/kernel_optimizer/agents/runtime.py:73` uses `shell=True` with a list
argv.** On POSIX that runs `/bin/sh -c "opencode"` and silently discards every argument, so
the server binds its default port instead of the one it was told, and the health check then
times out against a URL nothing is listening on. **It reads exactly like a network
problem.** No test catches this — the test that monkeypatches `Popen` never inspects argv.
If you find yourself debugging connectivity, check this first.

## What to do

1. **Port** the seven items in section A. They are: `to_wsl_path` (must become identity),
   the `wsl.exe` GPU dispatch (all GPU work goes through it), the per-job timeout kill, the
   three `shell=True` sites, the `taskkill` server shutdown, `doctor`'s WSL probe, and three
   tests that assert Windows-ness and should be *retired*, not repaired.

   When you replace the dispatch, note that `_build_command` currently relies on a shell for
   three things that Python does not do for you: `VAR=x cmd` env prefixing, `~` expansion in
   the venv and cache paths, and login-shell profile sourcing. Miss the `~` and every job
   dies on a literal `./~/...` path.

2. **Set up the environment** per section B: pin KernelBench at `423217d`, build the venv
   (torch 2.9 / cu129, triton 3.5 — cu129 covers sm_89), install opencode, and create the
   GLM provider config.

   **The provider config layout is not obvious and is the most common setup failure.**
   opencode resolves providers by walking *up* from the session's working directory, and
   each agent sandbox gets its own `opencode.json`, which makes it a project root and
   **stops** that upward search. So `launch_cwd`, `runs_dir` and `sandbox_config_path` must
   all be arranged as section B4 describes. Get it wrong and every agent call fails with
   `ProviderModelNotFoundError: Model not found: zhipuai/glm-5.3`.

   Also set `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX: "131072"` via `opencode.server_env`.
   There is no config-file key for it; the environment variable is the only route. At the
   32000 default, glm-5.3 at `reasoningEffort: max` spent its entire budget on reasoning and
   wrote **zero files** on three consecutive attempts, killing a whole run.

3. **Write a new config. Do not reuse the Windows ones.** `load_config` reads exactly ONE
   file — `configs/default.yaml` is *not* a base layer. Any key you omit falls back to a
   pydantic default, and those defaults are Windows paths (`D:/...`, `/mnt/d/...`). Five
   existing configs set no paths at all and would silently inherit them.

4. **Measure the device limits; do not copy them.** Every shipped config claims
   `RTX 5080 Laptop (sm_120)`, `vram_gb: 16`, `max_shared_bytes_optin: 101376`. Those values
   are written **verbatim into every agent sandbox** and are exposed as names inside
   agent-authored constraint expressions. A stale value does two kinds of damage: the guard
   admits configurations that then fail to compile, and **the model is actively told it is
   on Blackwell** and will emit tensor-core paths that do not exist on sm_89. Section C3 has
   a verified snippet, and names the two fields that snippet must NOT be trusted for.

5. **Run all nine acceptance tests in section C** and report each result. They are ordered
   so each failure is unambiguous.

## Expected failures — do not "fix" these

- **Three tests must fail after the port**: `test_to_wsl_path`,
  `test_wsl_paths_are_on_ext4_not_9p`, `test_setup_script_refuses_a_9p_venv`. They assert
  Windows/WSL-ness. Retire them with a note; repairing them would mean re-asserting what you
  just removed. Everything else should pass (255 pass / 9 skip on the Windows box).
- **Run pytest from `v2/`**. About 28 test sites read `src/...` and `configs/...` by relative
  path, so another working directory fails them en masse in a way that looks like a port bug.

## Constraints

- **Run one experiment per GPU.** The GPU lock is created per-run
  (`wiring.py:103` → `worker_client.py:116`), not machine-wide, so two concurrent runs do
  **not** mutually exclude their timing and both sets of numbers become worthless. Either
  respect this or move the lock to a fixed path as part of your port.
- **Do not commit secrets.** The opencode provider config holds a plaintext API key.
  `.gitignore` covers `*.jsonc` and `opencode.json`, but note that `sandbox_config_path`
  copies the whole provider block — key included — into every agent sandbox under `runs/`,
  and those copies are `.json`, which the `*.jsonc` rule does not match. `runs/` is already
  git-ignored; keep it that way.
- **Do not start a real 12-hour experiment.** Section C8's cheap L1 smoke is the end of your
  scope. The person who sent you this decides when a real run starts.

## A standard to hold yourself to

This codebase has been bitten repeatedly by checks that could not fail. Three examples from
its own history: a monitor invoked `python3`, which on that box was a stub that exited
without running, so it watched nothing while appearing armed; a liveness probe caught its
own `OSError` as "assume alive", so its death detector was permanently vacuous; and a test
re-implemented the loop it was supposed to test, so six green tests coexisted with a loop
spinning 2 million times.

So: **when you report that something works, say how you would have noticed if it did not.**
A negative result is only meaningful if the check would have failed loudly. Where you
cannot verify something, say so plainly instead of inferring it — an honest "I could not
test this" is far more useful than a confident guess.

## Report back

1. What you changed, by `file:line`, and why.
2. Each of the nine acceptance tests: pass / fail / not run, with the actual numbers
   (device properties, baseline latency, measured noise floor).
3. The measured `device:` block you wrote into the config.
4. Anything that surprised you or that you could not verify.
5. Whether the box is ready for a real experiment, and what you would want checked first.

---
