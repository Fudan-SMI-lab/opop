# Feasibility: a second experiment arm on `zhipuai/glm-5.3`

Asked: stand up a second experiment using `zhipuai/glm-5.3` at variant `xhigh`, in its own
directory with its own opencode server, and **verify it works with real agent calls** before
committing to it. Findings below are all from live calls, not from reading configs.

## Verdict: feasible, with one correction to the request

The arm works end to end. But **`glm-5.3` has no `xhigh` variant** — the repo's own config
declares `low` / `high` / `max`, and so does the live server:

```
glm-5.2:       variants=['high', 'max', 'off']
glm-5.3:       variants=['low', 'high', 'max']
glm-5.3-flash: variants=['low', 'high', 'max']
```

`xhigh` appears in this repo only in `.omo/omo_gpt.jsonc`, for GPT models. The `glm-5.2` entry
in `.opencode/opencode.jsonc` documents what the API does with it:
`xhigh→映射为max` — i.e. it maps to `max`.

**Checked rather than assumed.** The endpoint accepts `reasoning_effort: "xhigh"` with HTTP 200
(it does not reject unknown values), so acceptance proves nothing. Measuring reasoning-token
counts, 8 samples per tier, temperature 0, on a multi-step arithmetic prompt:

```
tier         n   median    mean   min    max
low          8      118     123   102    168
high         8     1443    1464  1040   1993
max          8     1022    1086   749   1858
xhigh        8      917    1027   744   1530
(omitted)    8     1072    1118   794   1481

control: low 118 vs max 1022 -> the parameter IS being applied
xhigh 917  vs  max 1022  vs  default 1072
  |xhigh-max| = 105     |xhigh-default| = 156
```

`xhigh` sits nearer `max` than the default, consistent with the documented mapping — but the
within-tier spread (`max` ranges 749–1858) is far wider than the gap between medians, so this
does **not** separate "maps to max" from "ignored, falls back to a default that happens to be
max-like". Two readings, one conclusion either way: the deepest tier `glm-5.3` offers is `max`.

An earlier 3-sample version of this was inconclusive and I nearly reported it as evidence;
`high` alone spanned 175–961 there. The n=8 run also failed entirely the first time because I
ran it concurrently — the provider rate-limited every request and only `low` returned. Both are
my errors, not the model's.

**So the config pins the depth two ways**, and does not depend on which reading is right:

```jsonc
"glm-5.3": {
  "options":  { "reasoningEffort": "max" },     // applies to EVERY call, variant or not
  "variants": { "low": …, "high": …, "max": …,
                "xhigh": { "reasoningEffort": "max" } }   // so "xhigh" is selectable
}
```

The model-level `options` block is the load-bearing part: the harness's `prompt()` sends only
`providerID`/`modelID` and no variant at all, so a variant-only setting would never be applied.

## The failure that had to be fixed first, and it is a general one

Three consecutive `agent-smoke` runs failed with:

```
AgentCallError: generator failed after 3 attempts: prompt failed 500
opencode.log: ProviderModelNotFoundError: Model not found: zhipuai/glm-5.3.
              Did you mean: glm-5.3, glm-5.3-flash?
```

The "did you mean" is misleading — the server does key the model as `glm-5.3` under provider
`zhipuai`, and a hand-built request with exactly the harness's `{providerID, modelID}` body
returns HTTP 200. The real cause is the **sandbox**:

- every agent call runs with `directory=<sandbox>`, and `SandboxFactory` writes an
  `opencode.json` there for permissions;
- that file makes the sandbox a **project root**, which stops opencode's upward search for
  configuration;
- so a provider declared in an ancestor directory is invisible from inside a sandbox.

`openai/gpt-5.6-sol` has never hit this because `openai` is in the user's **global**
`~/.config/opencode/opencode.jsonc`, which is always loaded. `zhipuai` exists only in project
config. Verified by construction: a sandbox-style config **with** the provider block returns
HTTP 200 and a real reply ("PONG", cost 0.0049); **without** it, `ProviderModelNotFound`.

Moving the run directory under `v2-glm` was not sufficient — I tried that first, and it still
failed, because the sandbox's own config halts the walk regardless of where the sandbox sits.

**The fix is a general mechanism, not a GLM special case.** `SandboxFactory` takes an
`extra_config` dict merged into each sandbox's `opencode.json`, supplied by two new
`OpencodeConfig` fields:

```yaml
opencode:
  sandbox_config_path: .../v2-glm/.opencode/opencode.jsonc   # provider block read from disk
  sandbox_extra_config: {}                                   # inline overrides
```

Only the `provider` key is copied — carrying `permission` or `plugin` across would change
unrelated behaviour. Reading from a path keeps the API key out of the experiment config and out
of git. A missing file raises rather than warning: an experiment whose every agent call fails is
worse than one that refuses to start. No model name, provider name, or task appears anywhere in
the harness.

**Side effect worth stating plainly:** a provider block carries `options.apiKey`, so every
sandbox this arm creates writes a plaintext credential to disk. Counted, not assumed — of the
three sandbox `opencode.json` files on disk, the one created after the fix carries a key and the
two from before do not. Run directories are gitignored (`runs/`, and `runs-glm/` added for the
abandoned first location), but they get archived and copied, so a GLM run directory should be
treated as secret-bearing. The gpt arm has no equivalent copies: its provider comes from the
global config, which the sandbox never has to be told about.

## The two arms differ in exactly three fields

Machine-checked by diffing the loaded configs:

```
.agents.default_model : gpt='openai/gpt-5.6-sol'   glm='zhipuai/glm-5.3'
.opencode.launch_cwd  : opop                       opop/v2-glm
.run.runs_dir         : v2/runs                    v2-glm/runs
```

Every budget and evaluation knob is identical, including `fp64_relative_gate: true`, so the
comparison is a model swap rather than a different experiment.

`launch_cwd` is what gives the arm its own server: the harness spawns `opencode serve` there on
a free port, so the two arms never share one.

## Isolation caveat worth knowing

A server launched from `v2-glm` still sees `anthropic`, `openai` and `bailian-coding-plan` from
the global config, plus `deepseek` etc. when launched anywhere under `opop` (opencode merges
every config it finds walking up). The arm is isolated in the sense that matters — its own
server, its own run directory, and `agents.default_model` decides which model every module uses
— but it is not a hermetic provider environment, and a config error naming another provider
would silently resolve instead of failing.

## What the end-to-end call actually did

`agent-smoke --module generator --task level1:19` against the GLM arm's config:

```
got 2 candidates (attempts=1, cost=$0.0074)
  cand_1.py: PARAMS ok {'BLOCK': 16384, 'NUM_WARPS': 8}
  cand_2.py: PARAMS ok {'BLOCK_M': 8, 'BLOCK_N': 2048, 'PROGRAMS_PER_SM': 16,
                        'NUM_WARPS': 8, 'NUM_STAGES': 3}
```

First attempt, no retries, both candidates satisfying the single-`PARAMS` contract. The event
log names the model — `AGENT_CALL_STARTED {'model': 'zhipuai/glm-5.3'}` — and the finish event
reports `reasoning: 226` tokens, i.e. the reasoning channel is live rather than the model
answering with thinking disabled. The sandbox's merged `opencode.json` reads back as
`providers: ['zhipuai']`, `glm-5.3 options: {'reasoningEffort': 'max'}`, so the depth pin is
present at the point of use, not just in the source config.

## Reproduce

```
python scripts/probe_glm_variants.py            # does the endpoint accept "xhigh"?
python scripts/probe_glm_tiers_powered.py       # reasoning-token distribution per tier (serial; ~20 min)
python scripts/probe_glm_agent_call.py xhigh    # live server + real agent call + structured output
kernel-opt --config configs/experiments_l3_glm.yaml agent-smoke --module generator --task level1:19
```
