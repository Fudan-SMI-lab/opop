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

## Where the 32000 comes from

Not from an upstream request cap — a direct call to the same endpoint with
`max_tokens: 120000` and `reasoning_effort: "max"` returns HTTP 200 normally.

Not from opencode's declared model limit either. The live server reports
`glm-5.3: limit {context: 1000000, output: 131072}`, and our `prompt()` body sends no
token-limit field at all, so nothing on our side asks for 32000.

That leaves the provider: on this endpoint, `reasoning_effort: "max"` appears to come with a
~32000-token thinking budget that the model spends in full on a hard prompt.
`scripts/probe_glm_thinking_budget.py` measures whether the ceiling moves with the tier.

What is already certain, and sufficient for the operational decision, is that
**`reasoningEffort: "max"` and an L3-sized prompt put glm-5.3 over the ceiling reliably** —
3 out of 3, at the very first call, on the easiest of the three L3 tasks.

gpt-5.6-sol on the identical prompt has never done this in 17 runs.

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

## What this does NOT fix

The feedback change gives a truncated agent a chance to recover. It does not raise the ceiling,
and on this evidence GLM at `reasoningEffort: "max"` needs ~100k characters of reasoning before
it will write anything for an L3 task — so the recovery has to come from the model choosing to
act sooner. Whether that works is untested and cannot be assumed.

The options for actually running the GLM arm, in increasing order of confidence:

1. **Re-run as-is.** The new feedback may pull it out of the loop on attempt 2 or 3. Cheap to
   try, unproven, and a failure costs another ~$0.45 and ~25 min.
2. **Lower the tier to `high`.** If the ceiling tracks the effort tier, a smaller thinking
   budget both fits and forces earlier action. This is a config change in
   `v2-glm/.opencode/opencode.jsonc` (`options.reasoningEffort`), not a code change — but it
   means the arm is no longer "glm-5.3 max", which is what was asked for.
3. **Raise the ceiling.** Only viable if the cap is a request field opencode controls; the
   probe settles that.

Option 2 changes the experiment's identity, so it is the user's call, not mine.

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
