# C3 §4.3 — what the device proof actually establishes, and how it was measured

Written for the paper's methods text and for whoever checks it. Everything here is a measurement
taken on box4 (RTX 4090, torch 2.13.0+cu129, triton 3.7.1); the probe scripts are named so each claim
can be re-run.

## The claim §4.3 makes

Given a well-defined task and a correct evaluator, the framework produces **goal-specific** operator
optimisations: a kernel written for one serving objective is measured on all three, and the
off-diagonal cells show whether the gain was specific or a general speedup any objective would have
found. Four rows (baseline plus one per goal) × 3 goals × 3 held-out blocks = 36 cells.

Everything below exists to support one precondition of that claim: **each optimised row's number came
from an authored device kernel, not from a host fallback, a dummy kernel, or the baseline measured
twice.** That precondition is exactly what §4.1 failed — 3 of its 4 rows shared the baseline's bundle
hash — so it is checked mechanically rather than assumed.

## What a row must satisfy

`device_gate.require_device_proof`, applied per declared kernel:

1. **Compiled and launched.** The declaration resolved to a real `triton.runtime.jit.JITFunction` in a
   declared bundle source, Triton returned an actual `CompiledKernel`, and a non-warmup launch was
   counted — see "Two routes" below for how the launch is observed.
2. **Numerically correct.** The operator's output matches the captured reference for the fixture, and
   the fixture's mutable state matches too.
3. **The output participates.** Four separate facts, none of them inferable from the other three:
   - the kernel's TTIR contains a **store to the declared output operand** (`stored_output_args`);
   - that store's dataflow **depends on real inputs** (`loaded_input_args`), so a constant write does
     not qualify;
   - the declared operand is **pointer-identical** to a tensor the replacement returned
     (`output_used`);
   - **suppressing the kernel makes the output wrong** (`skip_control_rejected`): the same call is
     re-run with the output buffers filled with NaN and the launch downgraded to a warmup, and the
     result must then FAIL the reference check.

The fourth is a counterfactual and it is the strongest of the four: a host fallback that happens to
compute the right answer, a dummy kernel, or a kernel whose output is discarded all produce an
unchanged result when the kernel is suppressed, and are rejected.

## Two routes to "it launched", and why

The original gate accepted exactly one observation of the launch: the declared kernel's name appearing
among the **CUDA-side** events of a `torch.profiler` trace. That is a real proof — when the profiler
collects anything. It is also an instrument, and this instrument was measured failing while the kernel
demonstrably ran.

**The observation.** In the live F4 development lane, 14 of 15 proofs returned `cuda_names=[]` while
the same proof recorded `compiled=True`, `launches=3`, a TTIR store to the declared output with
provenance from real inputs, pointer identity with the returned tensor, correct numerics, **and a
failing skip control**. The one proof that did collect CUDA activity was the first in the process.

**What was ruled out.** Eight probes, each a falsification attempt with a stated verdict:

| probe | hypothesis | verdict |
|---|---|---|
| B1 | an `opencode serve` subprocess contends for CUPTI | disconfirmed — byte-identical reads with and without |
| B2 | the agent-written kernel's bytes | disconfirmed — both F0 and F1 bytes pass |
| B3 | the helper RPC / `search_helper` child process | disconfirmed — both pass, including a real child |
| B4 | several proofs through one evaluator | not reproduced — 3/3 pass |
| B5 | a fresh `torch.profiler` per proof | disconfirmed — 5/5 fresh sessions collect |
| B6 | reading `events()` inside the profiler context | **real defect**, 0 events in 12/12 — but not this one: the code already reads outside, and the failing reads carried 11–12 CPU events |
| B7 | each layer of `prove`'s context stack (`cuda.device`, `inference_mode`, patched `run`, `sys.setprofile`) | disconfirmed — all 6 arms collect in every session |
| B8 | the agent transport thread being created and joined | disconfirmed — all 5 arms collect 3/3 |

Re-executing did not help either: three attempts inside one proof each returned `cuda_events=0`, so
the state is persistent within a process rather than a missed asynchronous flush.

**The second route.** `device_gate.launch_route` therefore accepts either:

- `cuda_trace` — the declared kernel's name is among the trace's CUDA events; or
- `causal_chain` — **three** facts together: (a) the profiler's own CPU side recorded a CUDA **launch
  API** call (`cuLaunchKernelEx` / `cudaLaunchKernel` / …), so the instrument agrees a launch was
  issued to the driver; (b) the launch was counted at the launch site, inside the patched
  `JITFunction.run`, only for a real `CompiledKernel` and only when not a warmup; (c) the skip control
  **failed**, so the output demonstrably depended on this kernel executing.

(a) matters because without it the fallback would rest only on our own instrumentation. Every one of
the 15 live traces carried `cuLaunchKernelEx`, including all 14 whose device-activity records were
empty, so the profiler saw the launch being issued and lost only the device-side record of it.

**The asymmetry that keeps this sound.** The fallback applies **only when the profiler collected no
CUDA activity at all** for that proof — judged from `trace_attempts`, where any attempt with
`cuda_events > 0` proves collection worked on this device, in this process, for this proof. If the
profiler DID collect CUDA activity and the declared name is absent, that absence is positive evidence
against the declaration and nothing rescues it. Numerics and output participation apply unchanged to
both routes.

Each published row records which route carried it (`KernelEvidenceRow.launch_route`), and the rendered
table prints the breakdown, so no reader has to assume the trace did.

### The adversarial control on the second route (B10)

A relaxed gate is worth exactly what it can still reject, so the second route was measured against the
four things it exists to catch. `b10_probe.py` runs five candidates on a real 4090 through the real
observer, the real `prove` and the real `require_device_proof`; all five launch the **same** Triton
kernel, so the arms differ only in whether the returned value comes from it. Each runs twice — once
normally, and once through a `TorchLocalDevice` subclass that reproduces the measured instrument
failure (device-activity records dropped, `cuLaunchKernelEx` kept), which is what puts `causal_chain`
under test rather than letting the trace route quietly carry the verdict.

| arm | what it does | normal | CUDA-suppressed |
|---|---|---|---|
| `honest` | the kernel writes the returned tensor | **accepted** via `cuda_trace` | **accepted** via `causal_chain` |
| `host_fallback` | kernel runs into a scratch buffer; host computes the answer | rejected | rejected |
| `dummy_constant` | kernel writes a constant into its declared output; host overwrites it | rejected | rejected |
| `discarded_output` | correct kernel, but a host-computed copy is returned | rejected | rejected |
| `unrelated_kernel` | real device work on unrelated buffers; output is host arithmetic | rejected | rejected |

10/10 arms as expected, no leaks (`CONTROL_PASSED_every_non_producing_kernel_rejected`,
`B10-0921-061450/b10-result.json`). The decisive field is the counterfactual: `skip_control_rejected`
is `False` in all eight non-producing arms and `True` only in the two honest ones. The TTIR and pointer
facts are individually evadable — `host_fallback` and `discarded_output` both produce a genuine store
to the declared output from real inputs (`stored=[1]`, `loaded=[0]`), and `dummy_constant` achieves
pointer identity with the returned tensor (`output_used=True`) — which is why the gate requires all
four facts rather than any one of them.

Note what this control does *not* claim: it shows the causal chain rejects a kernel whose output is not
used. It cannot show that a **timing** taken around such a kernel would have been rejected, because
latency never comes from these attempts.

## Where an optimised row's gain actually comes from, and how much is left

The device proof establishes that a row's number came from a real kernel. It does not establish that the
*gain* came from that kernel — a replacement could be faster because it bypasses some wrapper, or the
arms could have been measured at different times. Those are separate questions and they were measured
separately, because the ttft lane's two measurements initially disagreed about the kernel's sign:

| measurement | reading |
|---|---|
| full model, end to end | 273.433 ms vs 282.818 ms baseline = **+3.32 %** (faster) |
| operator, paired microbenchmark | 230.24 µs vs 227.49 µs reference = **0.988×** (slower) |

The operator number is the better-controlled one — `evaluate_local` interleaves reference and candidate
on the same fixture in one process, 20 timed samples each — and it measured the accepted kernel slower
on **all four** of its configurations (0.9802–0.988×). So the end-to-end gain needed an explanation that
was not "the kernel is faster per call".

**B14 — is the gain the kernel, or the replacement path it travels through?** Three arms, interleaved in
one resident process and cycled rather than blocked so any drift spreads across all of them:

| arm | median TTFT | vs baseline |
|---|---|---|
| baseline (zero-site bundle, no replacement) | 277.498 ms | — |
| **`torch_passthrough`** — the reference arithmetic in torch, **no kernel at all** | **277.861 ms** | **−0.131 %** |
| `accepted_kernel` — the published bytes and params | 269.366 ms | **+2.931 %** |

The passthrough arm is the control that matters: it goes through the *same* binding, site and quality
path as the candidate and contains no authored kernel. It sat at baseline. So the gain is not the
replacement path, not the bypassed `@use_kernel_forward_from_hub` wrapper, and not Python-level
overhead — it is the kernel. Both non-baseline arms measured `logits_relative_l2_max = 0.0`.

Two further facts rule out drift independently: the lane measured its baseline **twice, in separate
processes 89 minutes apart**, and got 282.334 and 282.818 ms (0.17 % apart); and two *structurally
different* bundles independently landed on 273.386 and 273.433 ms — the same value to 0.02 %, which
drift does not reproduce.

**B15 — then why is it slower in isolation?** One prefill wave per arm, same process:

| | baseline | passthrough | accepted kernel |
|---|---|---|---|
| kernel-level GPU time | 312 166 µs | 313 121 µs | **303 704 µs (−3.007 %)** |
| kernel launches | 2 300 | 2 300 | **2 081 (−219)** |
| allocated during the wave | 45.98 GB | 45.98 GB | **38.85 GB (−7.13 GB)** |

The kernel removes **9.42 ms of kernel-level GPU time** against a **9.60 ms** measured end-to-end
difference, so the gain is fully accounted for by this operator's own GPU time. The per-group deltas
show the mechanism: `mul fp32` −5 949 µs/−146 launches, `pow` −3 498 µs/−73, `cast/copy`
−3 334 µs/−146, against the two authored kernels' own cost of +3 545 µs. It fuses the reference's **8
ATen kernels into 2 Triton kernels**, and what it saves is the round trip of the intermediate fp32
tensors ([1, 4096, 2560] fp32 = 41.9 MB each), not arithmetic. The isolated microbenchmark clones one
call with a warm allocator and no neighbours, so it structurally cannot see that saving — which makes
the *microbenchmark* the misleading measurement here, not the end-to-end number.

One arithmetic caveat for anyone re-deriving these totals: summing every profiler event with device
time **double counts**, because `aten::mul` and the `MulFunctor` kernel it launched each carry the same
device time at op level and kernel level, and it also folds in `Command Buffer Full`, which is a
CUDA-runtime wait on a saturated launch queue rather than a kernel. The table above uses kernel-level
groups only.

**B16 — how much is left at this site?** Headroom cannot be established by optimising harder, only by
measuring the limit, so a fourth arm returns its input unchanged — no normalisation, **numerically wrong
by construction**, never publishable, used purely to price the site:

| arm | median TTFT |
|---|---|
| baseline | 277.025 ms |
| accepted kernel | 269.061 ms |
| **identity (a free RMSNorm — wrong)** | **259.287 ms** |

So the whole RMSNorm site is worth **17.739 ms = 6.403 % of TTFT**; the accepted kernel captured
**7.964 ms = 2.875 %**, i.e. **45 % of the site**; and **3.53 % remains** as the hard ceiling for any
further kernel there. Beyond it, gains must come from a different site — the leaf families ranked by
inclusive time are mlp gate/up **40.01 %**, mlp down **19.95 %**, attention k_norm/q_norm 9.94 %,
q_proj 8.48 %, o_proj 8.28 %, and rmsnorm only **6.33 %**.

`ResidentRunner.measure` refuses to time a bound site that has not passed `quality`, which is what keeps
every published number honest, and the identity arm cannot pass it. B16 therefore sets
`quality_identity` **inside the probe** for that one arm: no framework gate is modified, the correct arms
in the same probe pass `quality` normally, and the identity arm yields a timing bound rather than a
result. It is reported as an instrument reading obtained by deliberately breaking the model.

### What the framework can and cannot be asked to fuse

Fusion is not limited per candidate: `LocalRequest.kernels` is `Field(min_length=1)` with **no upper
bound**, and `BundleDocument.sites` and `.files` are unbounded, so a candidate may declare many kernels
across many sites and sources. The accepted ttft candidate already ships two Triton kernels and, as
above, fuses eight ATen kernels into them.

A **site** is not a time fraction — it is the **set of module instances whose `forward` is replaced**
(`site_id = canonical_site(sorted(module_paths))`). The time it occupies is a property of that set. So
"why not target the bigger ones" is really "which sets will the framework accept", and a set has to
survive **four independent gates**, not one:

| gate | where | what it requires |
|---|---|---|
| **L** leaf | `fast_target.build_target` | no observed child module ("choose a narrow leaf pure-tensor boundary, not a decoder/model wrapper") |
| **P** pure | `pure_tensors.tensor_leaves`, reached from `device_capture.capture_target_fixture` → `require_pure` | no `CacheValue` anywhere in the call, and ≤ 64 MiB. **A module that receives the KV cache cannot be captured as a fixture at all — and with no fixture there is no local proof, no `validate_local`, and therefore no device-proven row.** |
| **G** geometry | `build_target` | exactly `1 if ttft else 2` distinct input geometries across the whole group, at decode steps {0} or {0, 127} |
| **S** one site | `fast_artifacts.source_gate` | the bundle's sites must be exactly `(target.brief.site_id,)` — one site **ID** per candidate, though one ID may cover many modules (the live one covers 73) |

Measured against the frozen profile (`b17_probe.py`), that gives a precise picture of what is blocked
and by which gate — and the wrappers differ, which is why one relaxation does not unlock them all:

| candidate site | share of leaf-inclusive CUDA time | blocked by |
|---|---|---|
| attention block | **17.65 %** | **L and P** — its forward takes `past_key_values: Cache`; also its 36 instances form **36 separate families**, because `layer_idx` enters `family_key` |
| decoder layer | 1.04 % | **L and P** (same cache argument) |
| **MLP block** | 3.03 % | **L only** — its input is one pure tensor `[1, 4096, 2560]`, and its 36 instances share **one** family |
| model | 0.00 % | L and P |

Sixteen sites pass all four gates today, and several are far larger than the one used: `up_proj`
15.71 %, `down_proj` 15.62 %, `gate_proj` 15.61 %, `q_norm` 7.01 %, `q_proj` 6.63 %, `o_proj` 6.48 %,
against `input_layernorm` at **2.42 %**. Targeting one of those needs **no code change** — only a
different module passed to `build_target`. The RMSNorm site was a development-time choice, not a
constraint.

So the honest statement of the fusion limit, corrected against this measurement:

- **Within-family fusion** is available and was used (8 ATen kernels → 2 Triton kernels).
- **The leaf rule is now relaxable, and doing so enables genuine cross-kind fusion for the MLP block.**
  `build_target(..., allow_composite=True)` admits a wrapper as the site, so the replacement subsumes
  its whole subtree: `down_proj(act_fn(gate_proj(x)) * up_proj(x))` is three GEMMs, an activation and an
  elementwise multiply behind one pure-tensor boundary. Measured on the frozen profiles, the composite
  MLP site carries **146,793 µs** (ttft), 17,258 µs (single) and 152,984 µs (multi) of associated CUDA
  work, against the RMSNorm leaf's 14,041 / 2,383 / 15,036 µs — **7.2× to 10.5×**.

  Three details are load-bearing, and each was wrong in a first draft:
  - **the children are excluded from the group** — the site is the 36 wrapper instances, never a wrapper
    plus its own children, or the binding would install a replacement on both and `require_coverage`
    would then demand that a replaced child's original still ran;
  - **the brief carries the children's sources** — a wrapper's forward only names `self.down_proj(...)`,
    so without them a candidate cannot reproduce the computation, let alone fuse it. Each child is
    listed with its own observed metadata (the weight shapes differ: gate/up project 2560→9728 while
    down projects back), and **each class body is emitted once**. Emitting it per child repeated
    torch's whole `Linear` class three times — 200 redundant lines of a 341-line brief, and in the
    first live attempt **two consecutive 1500 s transport timeouts** (C0 and C1), with the opencode
    server answering in 8 ms and the GPU idle, so it was generation that did not finish rather than a
    resource problem. Deduplicated, the brief is 6,750 bytes against 14,052 with nothing removed.
    A composite brief's size is a sampling cost, not a formatting detail.
  - **the site is credited with the kernels it would subsume** — `site_kernels` attributes a kernel to
    the scope that launched it, so the wrapper's own scope holds only the elementwise multiply while the
    three GEMMs sit under the children. Crediting only the wrapper measured 8,582 µs, *less* than the
    RMSNorm leaf's 14,041 µs, inverting the ranking the number exists to provide.

  It is off by default, which keeps every leaf site byte-identical (`site_id`, `module_paths`,
  `associated_cuda_us` and `reference_source` all unchanged, pinned by
  `tests/test_c3_composite_site.py`), and nothing on the C2/KernelBench path imports this module.
- **It does not enable attention.** `subtree_is_pure` refuses `self_attn`, the decoder layer and `model`
  **even with the opt-in**, because their forwards receive a KV cache. That needs the purity gate to
  admit a `CacheValue`, and that gate is the foundation of the whole fixture → local-proof →
  device-proof chain (it exists so a call can be cloned and replayed), plus `family_key` would have to
  stop splitting on `layer_idx` — its 36 instances are 36 separate families. Those changes weaken
  evidence rather than widen search, so the refusal is deliberate and says why.

A site switch also has to carry its own noise floor: the development target's measured A/A noise belongs
to the site it was measured on, so `run_formal_campaign.py` now keys that carry-over on `site_id` as
well as goal. With a different site the noise is left unset and each lane measures its own, which
`require_target(noise=True)` requires before any candidate is admitted (verified: it refuses).

## What is NOT claimed

- **No significance claim.** Three blocks per cell support a descriptive median, not a test. The
  rendered table prints each cell's block spread (max−min as a percent of its median) directly above
  the gains, and names the largest spread in the table, because a gain of comparable size is not
  distinguishable from re-measuring the same thing. Measured instance: one accepted ttft candidate
  scored **273.433 ms** at selection and **274.449 ms** at confirmation on *identical bytes* — 0.372 %
  apart, which makes its own gain either 3.32 % or 2.96 % depending on which measurement is read.
  Off-diagonal effects can be that small, so the spread is reported next to them rather than left out.
- **`speedups` / `same_precision_speedup` are unusable for arm differences** — KernelBench stores the
  mean with `.3g`, a ≈1.3 % resolution floor. Cell values come from the measured medians.
- **Cross-hardware generality is untested**: all four available GPUs are 4090s.
- **Task/evaluator construction is out of scope this round.** Building the task and evaluator from a
  model plus a natural-language goal is hand-produced here; the framework is entered at the operator
  optimisation step. The narrowed claim is the conditional one stated at the top.
- **A baseline-fallback row is not a result.** A goal whose lane accepted nothing is measured as
  baseline bytes and labelled `⚠ NOT device-proven` with its reason, in both the table and the
  evidence list.

## A defect that made a real row look unmeasurable

Worth recording because of how it presented. The first matrix run of campaign `F4-campaign-0921-053151`
lost **all nine cells of its one accepted row** to:

```
RunnerError: unresolved traced site: pure-727f184a931d6918df621c6385fe123d7dd248cb470c207fd...
```

`ModelBinding.bind` raises that when a bundle declares a site the binding was never told about, and only
`register_site` tells it. Every other lane registers through `FastEvaluator.capture`; the sealed final
lane has no capture step — `evaluate_local` refuses local feedback there outright — so `run_matrix` had
to register the sites itself and did not.

The presentation is what matters: baseline rows bind a **zero-site** bundle, so they passed. The table
therefore rendered, correctly labelled itself partial, and showed the optimised row as the one that
could not be measured — which reads like a kernel that fails under sealed evaluation rather than a
missing registration. The row itself was fine: the same bundle at the same site binds and measures
without complaint in B14 and B16.

The fix is `fast_matrix.register_row_sites`, which registers each accepted row's site from the **frozen**
target before the model is loaded — the same site the row was searched and proven on — and refuses two
ways rather than guessing: an accepted row with no frozen target, and two frozen targets disagreeing
about the modules behind one `site_id`. Baseline fallback rows register nothing, because they bind
nothing. Pinned by `tests/test_c3_matrix_registers_row_sites.py` (5 tests), and the matrix path never
re-scans the tree, so a finished campaign's matrix can be re-run against its own frozen framework
without redoing the lanes (`rerun_matrix.py`).


## A second defect that turned two thirds of the table into baseline rows

In campaign `F4-campaign-0921-053151` the `ttft` lane accepted a kernel and the other two did not, which
read as a property of the goals. It was not. Both decode lanes **drew a candidate that passed every
gate** — device proof, local correctness, full-model quality — and both were then discarded, because
`run_probe`'s formal branch ended the lane after the first *proven* candidate whether or not it beat the
baseline:

| lane | direction | baseline | candidate | signed gain | lane outcome |
|---|---|---|---|---|---|
| `ttft` | minimize (ms) | 282.818 | 273.433 | **+3.32 %** | accepted |
| `single` | maximize (tok/s) | 21.243 | 20.028 | **−5.72 %** | `baseline`, 2 draws, hours unspent |
| `multi` | maximize (tok/s) | 162.134 | 153.503 | **−5.32 %** | `baseline`, 2 draws, hours unspent |

Note the direction column: `single` and `multi` are **maximize** goals in tokens/s, so a *lower* number
is worse. A candidate that looks "close to baseline" in a raw table can be a 5 % regression. The
framework was right to refuse those two candidates; what was wrong is that refusing them also ended the
search, so each lane stopped after two draws with most of its window unused and its row fell back to
baseline bytes.

The fix (`fast_workflow.run_probe`) treats a non-improving candidate as a failed draw like any other: it
is recorded with its own status, handed back as `previous` on the rewriter's feedback path, and the lane
keeps sampling until its clock stops it. Two statuses are added so a reader can tell the cases apart —
`not_improved` (correct, did not beat baseline) and `not_confirmed` (first measurement won, the repeat
did not) — and neither is the same as `model_failed`, which means the kernel was wrong.

**The acceptance bar is unchanged.** A row is still selected only on a positive gain confirmed by a
second full-model measurement, and `prove_candidate` still runs first. The change can only stop a lane
from discarding budget. Pinned by
`tests/test_c3_lane_keeps_searching_after_a_slower_candidate.py` (6 tests), including one that asserts
the `ttft` shape — proven, faster, accept immediately — is byte-for-byte the old behaviour.


## Choosing the site per goal, from measurement

The site used to be one constant module path (`C3_SITE_MODULE`, defaulting to a single RMSNorm
instance), shared by all three goals. That made the search space a hand-picked guess, and the *same*
guess under three objectives that profile differently. `fast_sites.select_sites` now ranks every site a
goal's own profile makes eligible by `associated_cuda_us` — the measured CUDA time of the kernels a
replacement would subsume — and takes the top one.

Eligibility is not re-implemented: a site is an option exactly when `build_target` accepts it, so the
four gates stay in one place. Ranking is CPU-only work on an already-recorded profile (≈20 s for ~550
observed modules, one parse shared across every `build_target` call via its new `profile=` argument), so
it spends no profile budget and no GPU time.

Measured on the frozen Qwen3 profiles, leaf sites only (no composite opt-in):

| rank | site | ttft | single | multi |
|---|---|---|---|---|
| 1 | `mlp.up_proj` (72 modules — gate+up share a family) | **88,822 µs** | **11,281 µs** | **92,975 µs** |
| 2 | `mlp.down_proj` (36) | 44,294 | 5,584 | 46,243 |
| 3 | `self_attn.q_proj` (36) | 18,816 | 2,366 | 19,571 |
| 4 | `self_attn.o_proj` (36) | 18,382 | 2,593 | 19,196 |
| 5 | `model.norm` — **the published site** (73) | 14,041 | 2,383 | 15,036 |

The published RMSNorm site ranks **fifth**, at 16 % of the top leaf site's associated work — and
`the-c3-rmsnorm-site-is-45-percent-exhausted` had already priced that site's entire value at 6.40 % TTFT
with 2.88 % already taken. So the ranking's first pick is not a marginal improvement over the old
constant; it is a site with ~6× the room, reachable **without** the composite opt-in.

With `C3_ALLOW_COMPOSITE=1` the composite MLP wrapper outranks everything (146,793 / 17,258 / 152,984
µs), which is the cross-kind fusion site. It is not the default: a composite brief is much larger, and
that cost is measured (see the timeout section below).

One constraint is load-bearing and is enforced rather than hoped for: `BindingRuntime.register_site`
refuses a module that already belongs to another group, and the matrix registers all three rows' sites
in **one** binding. Two goals may therefore share one site but may not hold different overlapping ones.
`resolve_overlaps` settles that by promoting a contained site to the containing one (which strictly
covers the work the goal ranked first) or, failing that, to the goal's best disjoint option, recording
`promoted_from` either way. On the current profiles all three goals rank the same site first, so nothing
is promoted — but the check is what stops a future profile from producing a table whose rows cannot
bind. Pinned by `tests/test_c3_site_selection.py` (17 tests).

Setting `C3_SITE_MODULE` still forces one path for every goal, which is how an earlier campaign is
reproduced; the ranking it overrode is recorded next to the choice. A forced path is resolved by
*building* at it rather than by matching the ranking's label, because many paths map to one site (all 73
RMSNorm instances do) and the ranking keeps only the shortest — matching on the label rejected
`model.layers.33.post_attention_layernorm`, the exact path the published campaign ran, as ineligible.


## The transport ceiling is a sampling constraint, not a safety margin

`request_timeout_s` was 1500 s, a value calibrated on the C2/KernelBench population (n=982 successful
calls, slowest 1167 s, none over 1200 s). On the C3 model-operator population the same number was wrong
in both directions:

- the slowest **successful** C3 call took **1447.6 s** — and it is the call that produced the published
  ttft row, i.e. 52 s from being discarded;
- on the composite MLP site, **four consecutive draws** each burned the full 1500 s and returned
  nothing. One was reconstructed from its sandbox file mtimes: 811 s of reasoning before the first file,
  `valid=True` with device proof and numerics both passing at **t+1359 s**, final file at **t+1502 s** —
  two seconds past the cap. That was finished work cut off by the transport, not a failed candidate.

A ReadTimeout costs the *whole* ceiling and yields nothing, so those four draws spent 100 minutes to
produce zero candidates. The ceiling is therefore **3600 s**. The hang cost that 1500 priced is carried
by `idle_abort_frac`, which is a *fraction* of the ceiling: a call that produces nothing for 30 minutes
is still aborted, while a call that keeps writing files is left to finish. Both bounds are pinned in
`tests/test_improvements.py`, with the old reasoning kept beside the new so neither story can be
restored on the strength of the other.

Related: a failed agent call logs **two** `AGENT_CALL_FAILED` rows (per-attempt and final), so counting
rows doubles the apparent failure rate.


## Running the three lanes in parallel across two boxes

The three goal lanes are independent observations by construction — `run_formal_goal` loads the
**original** baseline as every lane's parent, so no lane can see another's kernel — and nothing ties a
row to the machine that found it: `evaluate_fast_model` re-checks an inherited proof by bundle hash,
params hash and `framework_id`, never by device identity. So lanes can be split across boxes and merged
for the matrix, which is measured **once, on one machine**, so every cell shares one device.

Four things must be identical for the halves to be halves of one campaign, and `run_split_matrix.py`
checks each before the model is loaded: the same `framework_id`, the same frozen target per goal, the
same contract and sealed corpus, and exactly one accepted lane per goal (two machines both accepting
`ttft` is refused rather than resolved by merge order).

Two practical constraints, both found by inspection before launching rather than by a failed run:

- **A fresh development epoch is required after any code or timeout change.** `identify_framework`
  hashes every `fast_*.py` (so adding `fast_sites.py` changes it) *and* `request_timeout_s` (so
  1500 → 3600 changes it), and `run_probe` refuses when generation and admission identities differ. An
  older `DEV_GO` cannot seal today's lanes however valid it was. What a dev epoch does **not** constrain
  is the site — `freeze_framework` wants a real `DEV_GO` plus three predeclared targets — so the epoch
  runs on the cheapest site while the lanes go wherever their profiles rank first (`run_dev_only.sh`).
- **Three resources are single-occupancy per box.** The approved config pins one opencode `port` with
  `port_attempts: 1` (fail-fast), one `CUDA_VISIBLE_DEVICES`, and one set of triton/XDG caches. A second
  lane process inheriting all three loses the port race and exits with a bare `ServeError`, times its
  kernels against the other lane's GPU work, and shares its compile cache. `C3_AGENT_PORT`,
  `C3_CUDA_INDEX` and `C3_RUNTIME_ROOT` give each lane its own (and the port override also raises
  `port_attempts` to 3, so a collision retries on an ephemeral port instead of dying). None of these are
  part of `framework_id` — verified: `agent_config_sha256` covers the rewriter module dump, `agent` and
  `request_timeout_s` only — which is what makes one campaign across two machines possible at all.
  `XDG_CONFIG_HOME` is deliberately *not* moved: it holds the provider credential and is read-only.


## The numerical acceptance criteria, and what they were measured to admit

The device proof above establishes that a row's number came from a real kernel. A separate set of
limits decides whether that kernel is *correct enough* to publish. Those limits were calibrated rather
than assumed, because on this task they turned out not to be consistent with each other.

The frozen contract declares both of these about the same replacement:

| limit | value | scope |
|---|---|---|
| `local_operator_rtol` / `local_operator_atol` | 0.02 | per element, at the operator |
| `logits_relative_l2_max_per_prompt` | 0.01 | end to end, at the logits |
| `paired_mean_nll_delta_max_nat_per_token` | 0.02 | end to end, at the logits |

**What the live lane did.** Four independent candidates — separate agent calls, separate source files,
different kernel structure — were all rejected at `logits_relative_l2_max` 0.011155, 0.011807, 0.014172
and 0.011155, while each one's paired NLL delta stayed 20× *inside* its own tolerance (0.00096–0.0022
against 0.02) and each passed the per-operator check. Two of those numbers are byte-identical: the
agent, told only `frozen teacher-forced logits/NLL gate failed`, re-derived the same kernel.

**B11 — is the limit reachable by a correct kernel?** Four hand-written Triton RMSNorms, each
numerically equivalent to the reference forward (differing only in reduction blocking, where the bf16
rounding falls, and `rsqrt` vs `1/sqrt`), measured through the production `evaluate_quality` against
the frozen oracle:

| arm | `logits_relative_l2` | paired NLL delta |
|---|---|---|
| baseline, no replacement (**control**) | **0.0** exactly | 0.0 |
| `two_pass_fp32` (best) | 0.010349 | 0.001254 |
| `single_block_fp32` | 0.011807 | 0.002212 |
| `recip_sqrt` | 0.012070 | 0.001645 |
| `fp32_weight_mul` | 0.015751 | 0.002755 |

The control measuring exactly 0.0 is what makes the rest attributable: the harness, the oracle and the
model contribute nothing. Every equivalent formulation exceeds 0.01; the best misses by 3.5 %.

**B12 — why?** The selected site group covers **73 RMSNorm modules**, not one. Binding the *same*
equivalent kernel to a nested prefix of them isolates the cause, since only the count changes:

| modules replaced | 1 | 4 | 16 | 36 | 73 |
|---|---|---|---|---|---|
| `logits_relative_l2` | **0.0** | **0.0** | 0.004489 | 0.004489 | **0.010349** |
| verdict | passes | passes | passes | passes | **rejected** |

Monotonic, and the kernel is identical at every point. So the rejection is amplification across the
site group, not a defect: this kernel is exact at one site and fails only because the group has 73.
(B9 measured the single-site error of such a rewrite at 1.7e-05, 573× below the limit.)

**But the limit is reachable — the framework found how.** The sixth candidate in the live lane passed
it *exactly*: `logits_relative_l2_max = 0.0`, bit-identical to the reference, at **273.386 ms against
the 282.334 ms baseline — a 3.17 % TTFT improvement**. Its approach is the thing B11's four arms all
missed: keep the fp32 **reduction** in torch (`squares.mean(-1)` and `torch.rsqrt`, so the variance
bits match the reference exactly) and author Triton kernels only for the elementwise squares and the
normalise-and-scale apply. The amplification comes from *redoing the reduction in Triton*, not from
replacing the operator, and a formulation that avoids it exists. Four of six draws failed on
self-contained reductions before one did not, which is what the search is for.

This corrects the broader reading of B11: its `LIMIT_IS_UNREACHABLE` verdict is sound for
equivalence-preserving kernels that own their reduction, and too broad as a statement about all
correct kernels.

**Does the statistic still separate right from wrong?** Yes, and mostly *upstream* of it. Two
deliberately wrong kernels at full site count — normalising by mean|x| instead of RMS, and halving the
normalised value — never reached the logits statistic at all: the per-operator check rejected them
first (`local_checks_passed: 0.0`, 1.1 % and 6.8 % of elements outside tolerance). A milder defect
(omitting epsilon) did reach it, at 0.013044 — above the equivalent arms but within their spread,
which is the honest limit of this statistic's resolution at k=73.

**What this does and does not license.** It licenses reporting that the logits constant, applied to a
73-module group, rejects correct work whose reduction lives in the kernel — and that a formulation
keeping the reduction in torch passes it bit-exactly. It does not license moving a limit to admit a
candidate. The change made is to how the criterion is *resolved*, with these measurements attached:
the limits are read from the contract that declares them (they were hardcoded here while the contract
declared the same numbers separately, so the two could silently diverge), and a floor **measured** on
the actual site group by the setup lane's own correct control may raise — never lower — the logits
limit, exactly as `TargetSpec.reference_noise_us` measures timing noise instead of assuming it. With no
floor evidence and the default multiple of 1.0, behaviour is byte-identical to the declared absolute
limit, which is what keeps every other task and the C2 KernelBench path (which never imports this
module) unaffected. The per-operator check, the NLL criterion and the device proof are unchanged.

The candidate that passed needed none of this: it was admitted under the declared 0.01 limit.

### The floor is now measured automatically, and measured to be safe (B20–B25)

The floor mechanism above sat inert for a practical reason: the only way to produce evidence was the
setup lane's **hand-authored** correct control, so a task needed a correct kernel for the site before it
could calibrate a limit for that site. B23 removes that. It perturbs the site's **own original output**
by exactly one unit in the last place — the smallest difference any non-bit-identical implementation can
have from the reference — and measures what that costs at the logits. Nothing in it is
operator-specific, so the same code calibrates an RMSNorm, a projection or an activation.

| site | modules | one-ULP floor | unperturbed control |
|---|---|---|---|
| `model.norm` | 73 | 0.016163 | **0.0** exactly |
| `model.layers.0.mlp.up_proj` | 72 | 0.017743 | **0.0** |
| `self_attn.q_proj` | 36 | 0.019323 | **0.0** |
| `mlp.act_fn` | 36 | 0.015628 | **0.0** |

Each floor sits between "bit-identical to the reference" and 0.035628, which is what the contract's
**own** declared per-operator tolerance (rtol 0.02) implies at the logits (B13). So it is the minimal
correction consistent with the contract rather than a concession.

**An earlier reading of this evidence was wrong, and the correction matters.** These floors were first
refused on the grounds that each exceeds the smallest l2 measured by a "deliberately wrong" arm at the
same site. That was unsound twice over:

* **The statistic does not order accuracy at this scale.** B11's `fp32_weight_mul` keeps the weight
  multiply in fp32 — it *removes* a rounding, so it is strictly **more** accurate than the reference —
  yet it measured the **largest** error of all four equivalent arms (0.015751), because the reference
  itself rounds to bf16 before that multiply. The statistic measures deviation from the reference's
  rounding **sequence**, not fidelity to the mathematical result. Consistently, the "wrong"
  `bf16_reduction` measured 0.012061, *below* `recip_sqrt`'s 0.012070.
* **Those arms are precision choices, not faithfulness failures.** `bf16_reduction`, `bf16_acc` and
  `bf16_intermediate` all compute the correct operator at lower internal precision;
  `bf16_reduction`'s paired NLL delta was 0.0018 nat/token against a 0.02 limit, and the contract's own
  `local_operator_rtol = 0.02` admits that much per-element deviation. Treating them as the ceiling
  imported a bit-exactness standard the contract never declares.

**B24 measured what actually guards faithfulness.** Seven arms per site, graded by what they *do* to the
operator rather than by what they measure:

| arm | faithful | `local_checks_passed` | `logits_relative_l2` |
|---|---|---|---|
| unperturbed reference | yes | **1.0** | 0.0 |
| one ULP | yes | **1.0** | 0.016163 / 0.015628 |
| scaled 1.8 % (inside declared rtol) | yes | **1.0** | 0.033467 / 0.022202 |
| scaled 5 % (outside declared rtol) | no | **0.0** | never reached |
| learned weight dropped | no | **0.0** | never reached |
| identity pass-through | no | **0.0** | never reached |
| zeros | no | **0.0** | never reached |

At both sites `unfaithful_arms_admitted_under_floor` is empty. The division of labour is therefore:

* **`local_operator_rtol = 0.02` — faithfulness.** Whether the replacement computes the operator at
  all. Every unfaithful arm dies here, before the logits criterion is consulted.
* **the measured floor — amplification.** What the *group's size* costs a faithful kernel.
* **0.035628 — the ceiling** the contract's own per-element tolerance already permits.

This is what makes raising the logits criterion to the floor safe, and it is pinned in
`tests/test_c3_automatic_equivalence_floor.py` so the reasoning cannot be silently reversed.

**Who it helps.** All 11 rejected C3 receipts across the F4/F5 campaigns recorded
`local_checks_passed: 1.0` and failed on the logits criterion **alone**, with the paired NLL never
breaching (worst 0.0022 against 0.02) — i.e. every rejected candidate was *faithful*. The four distinct
kernels measured 0.010398, 0.011155, 0.011807 and 0.014172, plus F5's C1 at 0.012070 (identical to
B11's `recip_sqrt`). The `model.norm` floor of 0.016163 admits every one of them. It also exceeds the
equivalent-GEMM values of 0.012–0.014 (B20/B21), so it brings the five projection sites inside reach —
including `mlp.up_proj` at 88,822 µs, **6.3×** the RMSNorm site and a leaf. That last consequence needs
its own end-to-end verification before being claimed.

**One implementation trap, recorded because it fails silently.** `evaluate_quality` consults the floor
only when `runner.receipt.site_ids` is non-empty, since a row replacing nothing *is* the baseline and
has no amplification to correct. A first version of `measure_equivalence_floor` installed the perturbed
forward directly onto the modules, which left the receipt empty — so the calibration measured what the
gate treated as a baseline and wrote `logits_relative_l2_max: 0.0` for the 73-module group. Because
`criteria` takes `max(declared, floor × multiple)`, a 0.0 floor raises nothing: the lane wrote a
well-formed evidence file, the gate accepted it, and the declared 0.01 stayed in force with no error
anywhere. The floor arm must therefore be written as a real bundle and bound through `runner.bind`, with
`receipt.site_ids` asserted non-empty. The originals are handed to it through
`binding.active[site_id].__globals__`, the namespace `bind` actually imported — calling
`imported.open()` again imports a second copy whose module state the installed replacement never reads.

**Three more failures worth recording, each caught only by running it (B25).** A probe that calls the
production function and compares against B23's independent number is what caught all three; source
review had passed every time.

1. **Negative zero produced NaN.** The step was first taken on the signed integer word directly. That is
   wrong twice: for a negative value the word is negative, so adding a positive delta moves the
   magnitude *toward* zero rather than away; and `-0.0` is the most-negative word (`-32768` in int16),
   whose `abs()` overflows back to itself, so a clamp written as `bits.abs() + delta < 0` passes and the
   step lands in the exponent-all-ones region. One `-0.0` anywhere in an RMSNorm output made **every**
   logit non-finite and the measurement failed with `invalid logits/target in teacher-forced quality`.
   The fix isolates the magnitude field from the sign bit and saturates at the largest finite magnitude,
   so a finite output can never be stepped into an infinity or a NaN.
2. **The anti-cheat guard forbade the framework's own calibration.** `require_coverage` rejects any
   replacement that invokes the original, which is exactly what stops a candidate claiming the
   original's work as its own speedup. But the floor arm *is* "the original's output, moved one ULP", so
   it necessarily trips `old_calls` and was refused at both sites with `replacement not executed
   exclusively` — while `local_checks_passed: 1.0` confirmed the arithmetic was right. Resolved with
   `ModelBinding.delegating()`, a window that relaxes **only** `old_calls`, still requires
   `replacement_calls > 0`, is entered from exactly one place (pinned by a test that globs the package),
   and cannot be reached from a bundle: a replacement receives only `(site, call, params)` and
   `OperatorSite` exposes no path back to the runtime.
3. **The search lanes never populate `binding.fixtures`.** `validate_local` reads that map, but
   `FastEvaluator.capture` fills its own `engine.fixtures` tuple for the per-candidate local gate and
   nothing in the dev or formal path calls `runner.capture_fixtures`. The live lane therefore wrote the
   arm's bundle and then produced no floor *and no reason file*, dying with `missing original local
   fixtures`; its first candidate was rejected at 0.014172 under the declared 0.01, which the floor
   would have admitted. The calibration now captures them itself, before binding (capture is refused
   once anything is bound) and only when absent.

**Verified end to end.** B25 drives the production `measure_equivalence_floor` and compares it against
B23's independent measurement:

| site | framework measured | B23 independent | `local_checks_passed` | baseline after restore | rejected candidates admitted |
|---|---|---|---|---|---|
| `model.norm` (73) | 0.016163086 | 0.016163086 | 1.0 | **0.0** | **5 / 5** |
| `mlp.act_fn` (36) | 0.015628396 | 0.015628396 | 1.0 | **0.0** | **5 / 5** |

Identical to the digit, the binding restored (re-measuring the baseline afterwards returns exactly 0.0,
so nothing was left installed), and a live dev lane then calibrated **0.016163086** for itself with no
authored reference kernel anywhere in the path.

## Re-running the checks

Each probe writes a JSON record with an explicit `verdict` field next to its measurements, so a reader
can confirm the verdict quoted above came from the run rather than from this prose. The records are
under `results/c3-fast-kernel-iteration/B5-profiler/B1*-<stamp>/`.

```
# why the CUDA trace failed -- the eight disconfirmations, on the free GPU
CUDA_VISIBLE_DEVICES=1 python /root/probe-clean/b5_probe.py   # fresh profiler per session
CUDA_VISIBLE_DEVICES=1 python /root/probe-clean/b6_probe.py   # read position (a real defect)
CUDA_VISIBLE_DEVICES=1 python /root/probe-clean/b7_probe.py   # context stack, layer by layer
CUDA_VISIBLE_DEVICES=1 python /root/probe-clean/b8_probe.py   # transport thread lifetime

# whether the gates hold -- these produced every calibration number in this document
CUDA_VISIBLE_DEVICES=1 python /root/probe-clean/b9_probe.py   # operator-level quality calibration
/root/probe-clean/run_b10.sh    # adversarial control: does the causal_chain route reject a fake
/root/probe-clean/run_b11.sh    # is the logits limit reachable by an equivalent kernel
/root/probe-clean/run_b12.sh    # amplification curve over site count + discrimination controls
/root/probe-clean/run_b13.sh    # what the contract's own per-operator tolerance implies at the logits
# site reachability under the DECLARED limit: which operator classes can pass 0.01 at all
/root/probe-clean/run_b20.sh <snapshot>              # up_proj, three equivalent Triton GEMMs
/root/probe-clean/run_b21.sh <snapshot> [sites...]   # four more GEMM sites
/root/probe-clean/run_b22.sh <snapshot>              # act_fn, the weightless elementwise site
# the automatic floor: measure it, prove it safe, and check the framework reproduces it
/root/probe-clean/run_b23.sh <snapshot> [sites...]   # one-ULP floor per site, with controls at 0.0
/root/probe-clean/run_b24.sh <snapshot> [sites...]   # does ANY gate still reject an unfaithful arm
/root/probe-clean/run_b25.sh <snapshot> [sites...]   # does fast_setup reproduce B23's number, or 0.0
# the floor's plumbing and safety properties, without a GPU
pytest tests/test_c3_automatic_equivalence_floor.py tests/test_c3_floor_evidence_reaches_the_matrix.py
# b11/b12/b13 load the 4B model, so they need a free GPU; each run_b*.sh pins CUDA_VISIBLE_DEVICES=1
# and reads the SEALED snapshot, which is why they are safe to run beside a live chain.

# where an optimised row's gain comes from, and how much of the site is left
/root/probe-clean/run_b14.sh    # is the gain the kernel or the replacement path (interleaved A/B)
/root/probe-clean/run_b15.sh    # mechanism: in-model CUDA time, launches, allocation per arm
python b15_report.py <B15-dir>/b15-result.json   # kernel-level totals, without the double counting
/root/probe-clean/run_b16.sh    # hard ceiling at the site, via a deliberately wrong identity arm

# attribution from the finished lane's own receipts, no GPU needed
python operator_attribution.py <lane>-evaluation/raw 73 <baseline_ms> <row_ms> <accepted_bundle_sha>
python ttft_wave_detail.py <lane>-evaluation/raw [<other-lane>/raw ...]   # per-wave, drift structure
python fusion_capability.py <profile.json>        # what the site definition excludes, as numbers
python b17_probe.py <profile.json> [<out.json>]   # all four site gates, per module class, no GPU
python b18_probe.py <profile.json> [<out.json>]   # does a composite site build; what still refuses
python b19_probe.py <raw-profile-dir> [<out.json>]  # composite MLP for all three goals, noise unset
python wrapper_detail.py <profile.json>           # the wrappers' observed forwards and family split
python ttft_headroom.py <profile.json> [<accepted-operators.py>]

# the gates' own behaviour, without a GPUpytest tests/test_c3_device_gate_two_routes.py \
       tests/test_c3_device_trace_retry.py \
       tests/test_c3_device_participation_conjunction.py \
       tests/test_c3_quality_gate_reports_its_criteria.py \
       tests/test_c3_quality_criteria_from_contract.py \
       tests/test_c3_quality_floor_resolution.py \
       tests/test_c3_floor_evidence_reaches_the_matrix.py -q

# the whole chain against an immutable snapshot (see snapshot_code.sh for why)
/root/autodl-tmp/run_chain.sh <revision> <dev_hours> <search_hours> <final_hours>
# force ONE site for every goal instead of ranking per goal (how an earlier campaign is reproduced).
# Unset => each goal's site is ranked from its own profile by measured associated CUDA time.
C3_SITE_MODULE=model.layers.0.mlp C3_ALLOW_COMPOSITE=1 /root/autodl-tmp/run_chain.sh <revision> ...

# rank the sites a goal could be optimised at, without running anything (CPU only, ~20 s per profile)
python -c "from examples.c3_qwen3.fast_sites import candidate_sites; \
  [print(f'{o.associated_cuda_us:12.1f} us n={len(o.module_paths):3d} {o.module_path}') \
   for o in candidate_sites('<profile.json>', allow_composite=False)[:10]]"

# THREE LANES IN PARALLEL, one dev epoch shared. Step 1: seal the framework (either box).
#   A fresh dev epoch is REQUIRED after any fast_*.py or request_timeout_s change -- identify_framework
#   hashes both, and run_probe refuses a mismatch. It does NOT have to use the lanes' site.
/root/autodl-tmp/run_dev_only.sh <revision> <dev_hours> <gpu_index> <agent_port>

# Step 2: the lanes, split however you like, each with its OWN port/GPU/cache root. Two on one box:
/root/autodl-tmp/run_lanes.sh <rev> <dev-epoch> ttft         7 0 42810 &
/root/autodl-tmp/run_lanes.sh <rev> <dev-epoch> single,multi 7 1 42910 &
#   ...or across two boxes; then copy the remote campaign dir to the SAME absolute path here
#   (receipts are recorded as absolute paths; run_split_matrix refuses if they do not resolve).
rsync -a $ROOT/F5-lanes-single-multi-<stamp> root@<matrix-box>:$ROOT/

# Step 3: merge and measure the 36 cells ONCE, on one box, so every cell shares one device.
/root/autodl-tmp/merge_and_matrix.sh <rev> <out-name> <campaign-dir> [<campaign-dir> ...]

# verify two boxes would seal the SAME framework before splitting anything across them
python /root/fid_probe.py     # framework_id + the four hashes behind it

# render the table from a finished matrix, or find out exactly why it cannot be rendered
python render_matrix_table.py <campaign>/matrix/matrix.json --markdown table.md

# pair agent STARTED/FINISHED events by call_id -- a failed call logs TWO FAILED rows (per-attempt
# and final), so counting rows overstates the failure rate
python agent_call_timing.py <dev>/F3-agents/events.jsonl <campaign>/lane-<goal>-agents/events.jsonl
```

## Cost of a draw, measured

Relevant to reading the table: how many candidates a lane could afford. Agent call durations on this
box, under the old `request_timeout_s = 1500`:

| lane | calls | successful durations (s) | timed out |
|---|---|---|---|
| development | 6 | 652.1, 472.8, 765.9, 613.9, **1447.6** | 1 × 1500.2 |
| formal `ttft` | 3 | 506.6, … | 1 × 1500.2 |
| development on the **composite MLP** site | 4 | — none | 4 × 1500.2 |

The call that produced the development lane's accepted candidate took **1447.6 s — 53 s inside the
timeout**, and on the larger composite site *every* draw hit the cap (one of them finishing its work at
t+1502 s). So the cap was a constraint on how many candidates a lane can draw, and on how large a site
can be searched at all, rather than a safety margin. It is now 3600 s; see the transport-ceiling section.
A timeout costs the full ceiling and yields nothing. The formal lanes deliberately do **not** inherit the
development winner: each re-searches from the baseline, because inheriting one kernel into all three
rows would make the off-diagonal cells meaningless.
