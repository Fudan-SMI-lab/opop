# Root cause: glm-5.3 burns its whole token budget on reasoning and is cut off before acting

`run-l3-21-20260906-084636` — the glm-5.3 arm the feasibility check cleared — died 23 minutes
in, at its very first agent call:

```
AgentCallError: generator failed after 3 attempts: no parseable JSON found in your
response; emit a single fenced ```json block matching the required schema
```

That message is a **misdiagnosis produced by our own retry loop**. The model's JSON was never
the problem.

## What actually happened

The generator sandbox `sandboxes/generator-8f9c2567` contains `docs/`, `task/`,
`opencode.json` and nothing else — **no `candidates/` directory**. GLM wrote no files in 22
minutes. `artifacts/` is empty.

`events.jsonl` recorded one `AGENT_CALL_STARTED` and one `AGENT_CALL_FAILED`, and the failure
event carried only the corrective text — i.e. the wrong cause. The real trace had to be
recovered from opencode's own sqlite store (`~/.local/share/opencode/opencode.db`; the REST
message endpoint cannot serve these sessions back, because the stored `info.format` fails the
server's own `OutputFormatJsonSchema` validator on read):

```
attempt  output  reasoning  out+reason  finish   error
   1        225        13          238  tool-calls   -          <- read the 4 task files
   2        118        19          137  tool-calls   -          <- read triton_pitfalls + glob
   3         14     31986        32000  length   StructuredOutputError
   4         57     31943        32000  length   StructuredOutputError
   5          7     31993        32000  length   StructuredOutputError
```

Every failing turn stopped at **exactly 32000** `output + reasoning` tokens with
`finish: "length"`. The reasoning parts are 100379, 102550 and 102229 characters of MBConv
planning — FLOP counts, tile shapes, arithmetic intensity — that never reach a tool call. The
model deliberated until it was truncated, three times, spending **$0.4557** to write nothing.

Compare the L1:19 smoke test that cleared the arm as feasible:

```
   1        221        12          233  tool-calls
   2         60       405          465  tool-calls
   3       1123      4466         5589  tool-calls   <- wrote both candidate files here
   4         94         0           94  tool-calls
   5        316       226          542  tool-calls
```

Peak 5589 tokens against a 32000 ceiling. **The smoke test never approached the limit**, which
is exactly why it passed and told us nothing about L3. That is the error in my feasibility
verdict: I validated the transport, the provider config, the sandbox config mechanism and the
contract, on a prompt an order of magnitude easier than the one the experiment runs.

## Where the 32000 comes from: it is `max_tokens`, and it is ours to set

Not an upstream request cap — a direct call to the same endpoint with `max_tokens: 120000` and
`reasoning_effort: "max"` returns HTTP 200 normally. Not opencode's declared model limit either:
the live server reports `glm-5.3: limit {context: 1000000, output: 131072}`.

**I first concluded it was a per-tier thinking budget belonging to the provider. That was wrong.**
`scripts/probe_glm_thinking_budget.py`, same hard prompt, `reasoning_effort: "max"`, but
`max_tokens: 40000`:

```
effort  max_tokens  completion  reasoning  finish
   max       40000       40000      39952  ['length']
```

It stopped at **exactly 40000**, not 32000. So the ceiling tracks `max_tokens` precisely: the
model spends whatever budget it is given and gets truncated at the limit, whatever the limit is.
Raising it raises the stop.

Which means 32000 is a **default**, and nothing in this project's own configs sets it:
`v2-glm/.opencode/opencode.jsonc` has only `$schema` and `provider`, and the generated sandbox
`opencode.json` adds only `permission` and that provider block — no `maxTokens` anywhere. The
`maxTokens: 32000` visible in the live `/config` belongs to an unrelated agent definition
(`Hephaestus - Deep Agent`) in the user's global config, not to `build`, which resolves to
`{"mode": "subagent", "hidden": true}` with empty options. So the 32000 is opencode's own default
output cap for a request that names none.

That reframes the whole failure. It is not "GLM needs more thinking than the provider allows"; it
is **"we never told opencode how many tokens this agent may spend, and the default is too small
for an L3 prompt at `reasoningEffort: max`."**

### Why gpt-5.6-sol never hit it

Same default, nowhere near it. Across the live L3:43 run's 107 assistant turns:

```
finish reasons: {'tool-calls': 104, 'stop': 3}
max output+reasoning in a single turn: 6649    (vs the 32000 ceiling)
```

gpt-5.6-sol acts after a few thousand tokens of deliberation; glm-5.3 at `max` writes ~100k
characters of planning before its first tool call. The default was sufficient for one model and
not the other, which is exactly why 17 gpt runs gave no warning.

What remains certain either way: **`reasoningEffort: "max"` plus an L3-sized prompt put glm-5.3
over a 32000 ceiling 3 out of 3**, at the very first call, on the easiest of the three L3 tasks.

## Two harness defects this exposed, both model-agnostic

### 1. A truncation was reported to the agent as a formatting mistake

`AgentModule.invoke` had one branch for `result.structured is None`, and it always sent
"emit a single fenced ```json block". For a cut-off turn that describes a mistake the model did
not make, so it re-planned from scratch and failed identically — three times at full budget.

`PromptResult` now carries `finish`, and truncation gets feedback that can actually change the
outcome: it names the cut-off, quantifies the tokens spent, and asks the agent to stop planning
and act with the tools immediately. The ordinary malformed-answer branch is unchanged.

This is not GLM-specific. Any model, on a hard enough task, can be cut off mid-reasoning; the
old code would send it in a loop of identical failures at full cost.

#### Confirmed live on the real L3 prompt, and it is what fixed the arm

`agent-smoke --module generator --task level3:21` against the glm arm's config — the same prompt
that killed the run — now succeeds:

```
got 2 candidates (attempts=2, cost=$0.1584)
  cand_1.py: PARAMS ok {10 knobs}   NCHW layer-by-layer, tl.dot 1x1 conv
  cand_2.py: PARAMS ok { 8 knobs}   NHWC channels-last single pass
```

Per-turn, from opencode's store (`ses_f8ad6c04dffedOHyiDAbYLBzZm`):

```
 #   out  reason  out+rea    secs  tok/min  finish
 1   233      11      244     8.6     1697  tool-calls    read the task files
 2    63     348      411    13.0     1902  tool-calls    read triton_pitfalls
 3     5   31995    32000   729.4     2632  length        <- STILL truncated at 32000
     --- attempt 2, now carrying the truncation feedback ---
 4    80   12756    12836   255.3     3016  tool-calls    <- stopped at 12.8k and ACTED
 5  2786       0     2786    34.0     4918  tool-calls    wrote cand_1.py
 6  2805      13     2818    30.9     5471  tool-calls    wrote cand_2.py
 7    47     105      152     4.3     2141  tool-calls
 8   598     111      709    13.4     3166  tool-calls    emitted the JSON
```

Total wall clock **18 min 16 s**, cost **$0.1584** against the failed run's $0.4557.

Two things this settles:

- **The feedback change is load-bearing, not speculative.** Attempt 1 was truncated exactly as
  before; attempt 2, told it had been cut off and to act instead of planning, deliberated 12836
  tokens and went straight to the tools. Under the old feedback it would have been told its JSON
  was malformed and re-planned into the same wall — which is precisely how the run died.
- **A truncation is survivable, so it need not be fatal.** The cost is one wasted ~12 min attempt
  per call. Over the ~85 agent calls a 12 h L3 run makes that is ~8-9 h of pure overhead, which
  is a real threat to finishing four families x three rounds inside the budget — the arm is
  viable but slower than the gpt arm, and that asymmetry has to be stated when comparing them.

#### `maxTokens` in a provider model's `options` is ignored

I proposed raising the cap by putting `maxTokens: 100000` in the glm arm's
`provider.zhipuai.models.glm-5.3.options`. The smoke above ran WITH that setting, and turn 3
still stopped at exactly 32000 — so it never reached the API.

The reason is that `maxTokens` is not a model option. opencode's own config schema
(`https://opencode.ai/config.json`) lists it under `$AgentConfig.properties`
(`color, description, disable, hidden, maxSteps, mode, model, options, permission, prompt,
steps, temperature, tools, top_p, variant`), i.e. it belongs to an **agent** definition, not to a
provider's model block. `$ProviderConfig` has no such key, so the value was carried into the
sandbox config (verified) and silently dropped.

I verified the setting reached the sandbox and did not verify that opencode would honour it
there. Those are different checks, and only the second one mattered.

It also means the earlier arithmetic about needing `request_timeout_s` raised above 1200 s does
not apply on this path: the successful call took 18.2 min, and its single truncated turn took
729 s, both inside the existing limit.

### 2. The run's own trace could not diagnose it

`AGENT_CALL_FAILED` recorded only `module`, `call_id`, `final` and the corrective text, so the
persisted explanation was the wrong one and the intermediate attempts left no per-attempt
record at all. It now also records `finish`, `attempts`, `cost` and `tokens` — enough to tell
truncation from malformed output without opening a 17GB sqlite file.

Tests: `tests/test_improvements.py`
`::test_a_truncated_turn_is_told_it_was_cut_off_not_that_its_json_was_malformed`,
`::test_a_malformed_answer_still_gets_the_formatting_feedback`,
`::test_the_final_failure_event_records_how_the_last_turn_ended`.

Both changes are agent-side, so they reach a running experiment immediately
(`opop-v2-worker-vs-driver-fix-propagation`) — but no GLM run is in flight to benefit.

## The remedy, superseded: the arm already works without raising the cap

> **Superseded by the live smoke above.** This section proposed raising `maxTokens` via the
> sandbox config. That specific route does not work — `maxTokens` is an `$AgentConfig` key, not a
> provider model option, so the value is dropped. And it turned out not to be needed: with the
> truncation feedback in place, the arm produces two contract-valid candidates on the real L3:21
> prompt in 18.2 min for $0.1584, absorbing one truncated attempt on the way. The reasoning below
> about *sizing* a cap remains valid if the cap is ever set in the right place
> (`agent.build.maxTokens`), which is untested.

Because the ceiling is `max_tokens` and nothing here sets it, the fix is a config addition rather
than a model or tier change — the arm stays **glm-5.3 at `reasoningEffort: max`**, which is what
was asked for.

`OpencodeConfig.sandbox_extra_config` already exists for exactly this: it merges into every
sandbox's `opencode.json`, on top of the provider block read from `sandbox_config_path`. Adding a
`maxTokens` to the glm arm's model entry raises the cap at the point of use, changes nothing for
the gpt arm, and needs no code.

Sizing it from measurement rather than taste: the three failed attempts spent 31986 / 31943 /
31993 reasoning tokens **and were still not finished**, and at `max_tokens: 40000` the model
consumed 39952 and was *still* truncated. So GLM at `max` will spend whatever it is given on this
prompt, and the cap has to be set above what the task actually needs rather than above what the
model will happily consume. The model's declared output limit is 131072; the L1 smoke, where it
did finish, needed 5589. There is no measurement establishing where between 40000 and 131072 it
would stop on its own, so a first attempt should use a generous cap (the declared limit, or close
to it) and the run's own `AGENT_CALL_FINISHED` token records will then show what it really used.

### The `high` tier does not help, so lowering the tier is not a remedy

The same probe at `reasoning_effort: "high"`, `max_tokens: 40000`:

```
effort  max_tokens  completion  reasoning  finish
   max       40000       40000      39952  ['length']
  high       40000       40000      39991  ['length']
```

`high` spent **39991** — marginally *more* than `max` — and was truncated identically. So the
effort tier is not a thinking budget on this endpoint: it does not cap deliberation, and dropping
to `high` would have hit exactly the same wall. That removes option (2) below as a fix, and it
retroactively justifies not having tried it blind: it looked like the safe conservative move and
it would have failed the same way, at the same cost, while also changing what the experiment is.

It also means the earlier tier measurement in `feasibility-glm-5.3-second-arm.md` — where `high`
produced a *higher* median reasoning count than `max` (1443 vs 1022) — was not the anomaly it
looked like. On a prompt hard enough to matter, both tiers simply spend everything available.

### What raising the cap costs

From the failed run's own billing: $0.4557 for 96375 output+reasoning tokens, i.e. **$4.73 per
million**. Per generator/rewriter call, and scaled to the ~85 agent calls a 12 h L3 run makes:

```
cap  40000   $0.189/call   ~$16 per run
cap  80000   $0.378/call   ~$32 per run
cap 131072   $0.620/call   ~$53 per run
```

Those are worst-case figures assuming every call spends the whole cap, which is what GLM did on
three of three attempts. A cap is therefore also a budget decision, not only a correctness one.

The measured reality, from the successful smoke, is cheaper than any of those rows: **$0.1584 per
generator call** — one truncated 32000-token attempt plus a ~20000-token productive attempt. At
~85 calls that is roughly **$13 per run**, i.e. the truncation-and-recover path costs less than
letting the model spend a 100k cap freely would.

## Status: the arm is viable at `reasoningEffort: max`, with a throughput penalty

Measured, not assumed:

```
                       failed run          after the fix
outcome                dead at 1st call    2 valid candidates
attempts               3 (all truncated)   2 (1 truncated, 1 productive)
wall clock             23 min, then died   18 min 16 s
cost                   $0.4557             $0.1584
files written          0                   2
```

So the ordered options are now:

1. **Re-run the GLM arm as-is.** No config change needed; the truncation fix carries it. The
   penalty is ~12 min of wasted first attempt per agent call, which over ~85 calls is ~8-9 h of
   overhead against a 12 h budget — so the arm will likely complete fewer family-rounds than the
   gpt arm, and any model-to-model comparison must say so rather than compare final numbers as if
   the search effort were equal.
2. **Set `agent.build.maxTokens` (the correct key) and re-test.** If honoured, GLM's first attempt
   would not truncate and the ~12 min penalty disappears, bringing throughput closer to the gpt
   arm. Untested — one ~20 min, ~$0.3 probe would settle it.
3. ~~**Lower the tier to `high`.**~~ Measured: `high` truncates identically at 39991/40000.
4. **Drop GLM** and finish the two remaining gpt tasks.

Option 2 is worth doing before option 1, because it is cheap and it removes the one asymmetry
that would weaken the comparison the arm exists to make.

Superseded options, kept for the record:

1. ~~**Raise `maxTokens` in the provider model options.**~~ The value never reaches the API; see
   the `$AgentConfig` note above.

## Reproduce

```
python scripts/dump_opencode_session.py ses_f8bd092d3ffeOPjvtUcytpvhyu   # the real run
python scripts/dump_opencode_session.py ses_f8dd66365ffeUYOSETkrLAy42b   # the smoke test
python scripts/probe_glm_thinking_budget.py                                     # tier vs ceiling
```

Note: `v2-glm/runs/*/sandboxes/*/opencode.json` hold a plaintext zhipuai key, copied there by
the `sandbox_config_path` mechanism. Treat those run directories as secret-bearing.

## Ordering consequence, stated plainly

The user asked for the GLM arm to run **before** the remaining two gpt tasks. It did not: the
GLM run crashed at 09:09:48 and `scripts/run_chain.ps1` advanced to gpt `level3:43`
(`run-l3-43-20260906-091019`, live and healthy — baselines eager 41.8 / eager_tf32 29.0 /
torch_compile 35.4 / torch_compile_tf32 18.5, first tuning 19.8 ms). `level3:48` is queued
behind it. The requested ordering is therefore already broken, and re-running GLM now would
mean either interrupting a healthy run or accepting third place in the queue.
