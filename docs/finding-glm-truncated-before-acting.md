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

## The remedy: raise the cap, which keeps the arm's identity

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

The honest caveat: raising the cap is necessary but might not be sufficient. It removes the
truncation, and the new truncation feedback gives a cut-off agent a way to recover, but neither
makes GLM *choose* to act earlier. If it simply deliberates for 100k tokens and then writes four
candidates, the arm works and each generator call is expensive. If it deliberates without ever
converging, the arm fails for a different reason. That is not knowable without trying.

Remaining options, now that (3) is confirmed available:

1. **Raise `maxTokens` and re-run at `max`.** Preserves the experiment's identity. One config
   line. Recommended.
2. **Lower the tier to `high`.** Would also fit under a smaller budget, but changes the arm from
   "glm-5.3 max" to something else — an experiment-identity change, so the user's call.
3. **Drop GLM** and finish the two remaining gpt tasks.

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
