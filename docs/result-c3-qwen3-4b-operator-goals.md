# C3: Qwen3-4B manual-task operator-goal pilot

**Final analysis — numerical and source/session audits integrated; prospective T10 fix CPU-verified at revision 82daf15b460004707e55dbdebb789e2788921527. This report is archived separately from the unchanged historical results; the single final F1 review remains pending.**

## Result in brief

**PARTIAL C3-B evidence; GOAL DIFFERENTIATION NOT DEMONSTRATED; no new custom GPU kernel demonstrated.** Six structural opportunities recorded five failures and one accepted decoder-layer SOURCE rewrite. TTFT and single-request searches retained the unchanged baseline, not optimized winners.
The one fixed multi-selected artifact was independently measured under a repaired heldout harness: single-request throughput was **12.879% higher**, and finite eight-request throughput **9.095% higher**, than the original baseline medians.
Its **0.177% lower TTFT is within observed variation**. These are descriptive three-block results, not statistical significance, three distinct optimal winners, universal efficacy, or a new low-level GPU-kernel result.
The original harness failed before scoring all36 heldout blocks. Those null results remain intact; the valid matrix belongs to the separately recorded recovery revision.
Three search failures were **false framework postprocessor rejections**, not unchanged/bad final proposals: writable parent-source mirrors had been overwritten with child code. This defect limits the experiment's ability to test goal differentiation and cannot be used to conclude that the LLM failed to produce goal-specific computational changes.

## Frozen setup and scope

- Scope: **C3-B, manual-task-first**. Humans supplied the reviewed task/evaluator, workload and quality contract; existing provided-evaluator/native min/max search drove real agent rewriting. Automatic natural-language task construction, C3-A, was not tested.
- Model: `Qwen/Qwen3-4B`, checkpoint/tokenizer revision **`1cfa9a7208912126459214e8b04321603b3df60c`**, BF16, eager execution with SDPA. No alternate model, quantization or whole-model FP64 rescue.
- Runtime: Torch2.13.0+cu129 /CUDA12.9, Triton3.7.1, Transformers5.16.1, safetensors0.8.0, Optuna4.9.0 and Pydantic2.13.5. A used `kernel-opt-venv`; B used `orch-venv`.
- Hardware: independent RTX4090 cards, not pooled memory. Search used A0=TTFT, A1=single, B0=multi; B1 unused. Every recovered heldout block ran serially on A0, UUID `GPU-c896ae45-532b-2a1d-147d-ed72e221fa42`, driver580.105.08.
- Data: frozen58-prompt **synthetic** corpus:2 calibration,14 search,42 heldout prompts. Heldout content is disjoint from calibration/search; every row uses the same column/block prompts.
- Generation: non-thinking chat template, greedy/seed0, exact rendered input lengths and fixed output lengths; EOS does not stop generation early. No padding counted as useful tokens.
- One resident model per active card; baseline/candidate bindings restored sequentially. Tokenization, checkpoint loading and warmup are outside scored windows; each timed request starts with fresh KV state. Request-side allocation, prefill and token-ready synchronization remain in their defined windows.
- Approximately8GB of model assets and their acquisition/runtime preparation belong to the preparation ledger, not the main experiment window. T6's development norm replacement and integration controls validate the harness; they are not research-agent kernel-generation evidence.

| Column | Work per block | Native objective |
|---|---|---|
| TTFT |4 requests,4096 input /1 output each |Minimize median warm request-to-first-token wall, ms |
| Single |2 requests,256 input /128 output each |Maximize median `127/(last_ready-first_ready)`, tokens/s |
| Multi |One synchronous finite wave of8 requests,512 input /128 output each |Maximize `1024/shared_batch_wall`, tokens/s; includes prefill/completion |

Single still executes prefill although its decode-rate denominator excludes first-token latency. Multi is not online-server throughput/SLO, one forward multiplied by eight, or a stream-count metric. Raw records retain first-token, total-wall and full growing-KV traces.
Each candidate/shape receives the prescribed unscored warmup. Summary values are medians of three block scores, with min/max ranges; they are not pooled-request medians.

## Public implementation and recovery provenance

| Stage | Published revision | Actual scoped implementation |
|---|---|---|
| Protocol archive |[`8cfac7d`](https://github.com/Fudan-SMI-lab/opop/commit/8cfac7d5a13abafb25f7e658fbf1b6152a0ae0cf) |Authoritative manual-task contract before product/GPU work |
| Foundations |[`05ca48e`](https://github.com/Fudan-SMI-lab/opop/commit/05ca48e38cab38fef4128c0d73ec83f27011743a) |18 example modules plus4 tests/support files; binding, resident measurement, quality and manual evaluator |
| Bounded search |[`cc792df`](https://github.com/Fudan-SMI-lab/opop/commit/cc792df39c94aafb824e72cb2ce80c73e27f578e) |17 paths:3 core seams,9 task-local modules,5 tests/support; not a replacement search engine |
| Quality arithmetic/main |[`af77534`](https://github.com/Fudan-SMI-lab/opop/commit/af775349a778fa9c62e8fcda6f327578d05ad89d) |5 paths:3 arithmetic/caller files and2 test/replay files; exact search/original-heldout revision |
| Heldout recovery |[`1870e8e`](https://github.com/Fudan-SMI-lab/opop/commit/1870e8e612a1febfec3b77a63b2e091cbc7a6539) |2 paths:grouped-fixture matrix adapter and regression tests; parentaf77534 |

Original T8 expanded the selected36-module decoder group into separate fixture sites and hit `selected local fixtures exceed the 128 MiB resident budget` before cell admission. All36 original scores are null, not zero; this was a preparation integration failure, not demonstrated operator numerical failure or deadline censoring.
Recovery was predeclared and authorized before measurement, used a new output root and tested harness, and changed common fixture/execution grouping only. **64MiB/128MiB caps, operator/helper bytes, per-module assignments, params, inputs/oracles, thresholds, row order and original deadline did not change.**
One representative fixture per group/phase yielded2 fixtures in4 model forwards; full-model quality still exercised every selected module. Exact native fixture-byte usage was not persisted and is not replaced by synthetic test sizes.
Original matrix SHA256: `f66d72c69a9b513428fbe68bbe2ff33f88404d4fdf282bd2d761ed014ac3981a` (36 invalid/null). Recovery: `d14d11fe97b7e754c6b96219b57eead233de240c4e807073156fd5edb89ee4a8` (36 valid).

## Frozen artifacts and source interpretation

| Row | Selection identity / effective params | Interpretation |
|---|---|---|
| Baseline |`9e4ae897…` /`{}` |Original model; empty replacement map |
| TTFT |Same baseline bytes /`{}` |**BASELINE_FALLBACK** after two failures |
| Single |Same baseline bytes /`{}` |**BASELINE_FALLBACK** after two failures |
| Multi |`8e37a735…` /`{}` |Accepted SOURCE rewrite, `model.layers.0`–`model.layers.35` →`fused_decoder_layer` |

Baseline operators SHA256: `882e62a1165dc59b606568ef7c64f2994f0077b4b129a979e64cc62bfb490d7a`. Multi: `e4753894532cd479749a25774d46844f4760a8274f3956214934d38e04679380`.
Multi's recovered execution bundle `d80c70ba…` differs from frozen selection `8e37a735…` only through common instrumentation site labels; callable assignments/source/params remain the selected artifact. Empty params/space permitted **one parameter-free native trial**. TPE machinery was implemented, but substantial TPE parameter optimization did not take place in this run.
The completed [source audit](../.omo/evidence/v5-c3-qwen3-4b-operator-goals/T9-source-audit.md) and [machine ledger](../results/c3-qwen3-4b-operator-goals/source-audit.json) trace the148-line agent-written source to its selected and recovery copies; this is a real nonconfiguration SOURCE change.
It inlines decoder composition/Python dispatch, uses seven existing `F.linear` calls, eager RMS cast/pow/mean/rsqrt chains and SiLU/multiply, and retains existing rotary/cache.update/configured SDPA attention. Cached weight/module references and interface metadata are not output or answer caches. The accepted file defines no Triton/CUDA device kernel, extension or `torch.compile` path; “fused_decoder_layer” does not establish device fusion, nor does use of existing GPU operators mean CPU-only inference.
The goal/profile packet supplies an observable dispatch-reduction rationale, but its inclusive CPU timings overlap. “Mostly host dispatch” remains a hypothesis: no before/after low-level GPU trace proves identical launch sequences or quantifies a causal mediator percentage. Finite zero-error checks do not prove universal bitwise equivalence.

## Recovered heldout matrix

All cells use3 independent blocks on the same card. Entries below are **median [minimum, maximum]**; ms lower is better, tokens/s higher is better.

| Row | TTFT ms ↓ | Single tokens/s ↑ | Multi tokens/s ↑ |
|---|---|---|---|
| Baseline |284.781157 [283.122453,285.098008]|25.783499 [25.736718,26.329318]|196.880185 [193.766142,197.539852]|
| TTFT BASELINE_FALLBACK |284.124854 [283.728923,285.298542]|26.372324 [25.428999,26.494505]|195.102521 [193.842437,198.047351]|
| Single BASELINE_FALLBACK |284.581298 [284.071853,284.946991]|26.451264 [25.724591,26.488749]|197.778747 [193.011518,198.187867]|
| Multi fixed SOURCE rewrite |284.277034 [284.154466,284.825768]|29.104201 [28.896900,29.186193]|214.785688 [213.994827,215.745360]|

Direction-aware relative difference versus baseline median: TTFT=`100*(1-row/base)`; throughput=`100*(row/base-1)`. Positive means favorable, not necessarily a source gain.

| Row | TTFT difference % | Single difference % | Multi difference % |
|---|---:|---:|---:|
| Baseline |0.000000|0.000000|0.000000|
| TTFT BASELINE_FALLBACK |+0.230459|+2.283727|−0.902917|
| Single BASELINE_FALLBACK |+0.070180|+2.589889|+0.456401|
| Multi fixed SOURCE rewrite |+0.177021|+12.879174|+9.094619|

The byte-identical fallback rows are useful noise controls: their differences are not optimization wins. Multi's0.504123ms median TTFT reduction is smaller than baseline block variation and the TTFT fallback's0.656303ms median difference.
The multi artifact improves both throughput columns descriptively, rather than establishing a distinct best artifact for each requested goal. Even nonoverlapping observed throughput ranges across these blocks do not establish population-level significance or causality.

### All36 block scores

| Row | Column | Block0 | Block1 | Block2 |
|---|---|---:|---:|---:|
| Baseline |TTFT ms|283.122453|284.781157|285.098008|
| TTFT fallback |TTFT ms|283.728923|284.124854|285.298542|
| Single fallback |TTFT ms|284.581298|284.946991|284.071853|
| Multi rewrite |TTFT ms|284.154466|284.277034|284.825768|
| Baseline |Single tokens/s|25.736718|25.783499|26.329318|
| TTFT fallback |Single tokens/s|26.494505|25.428999|26.372324|
| Single fallback |Single tokens/s|26.488749|26.451264|25.724591|
| Multi rewrite |Single tokens/s|29.104201|29.186193|28.896900|
| Baseline |Multi tokens/s|197.539852|193.766142|196.880185|
| TTFT fallback |Multi tokens/s|193.842437|195.102521|198.047351|
| Single fallback |Multi tokens/s|193.011518|198.187867|197.778747|
| Multi rewrite |Multi tokens/s|215.745360|213.994827|214.785688|

Actual execution order: columns TTFT/single/multi; within each column block0 baseline/T/S/M, block1 M/S/T/baseline, block2 S/M/baseline/T. No duplicate/missing block, retry, post-hoc reselection or substituted search score. Rounded displays above; full precision in analysis JSON.

## Quality and numerical verification

All36 heldout quality records passed the frozen BF16 contract: local shape/dtype/finite/state checks and `rtol=atol=.02`; per-prompt pooled relative-L2≤.01; paired mean NLL delta≤.02nat/token. **Recorded L2 max0 and independently reaggregated NLL delta0 throughout; no FP64 model rescue.**
Quality used the original frozen baseline teacher-forced continuations,32 positions per prompt:168 prompt records/5376 positions across rows. Every multi-row quality record reports `36*32*prompt_count` replacement calls and zero old-target calls; representative fixtures did not replace full-model checking.
The independent CPU analysis recomputed every score from token-ready timestamps with maximum absolute discrepancy0.0, verified exact request/output counts and recorded cache progression (4096;256…383;512…639), and checked all36 admission times and72 unique raw quality/measurement links.
Paired NLL was recomputed from saved per-token candidate/reference losses. Full candidate logit vectors are **not** retained, so independent full-vector L2 or logits-to-NLL recomputation is not claimed. Reported L2 values/gates and their aggregates were checked.
All42 heldout prompt token arrays match baseline exactly across the four rows in this run. That observation does not replace the quality policy or imply general semantic equivalence outside this synthetic corpus.

## Search failures, budget and separate cost inventories

| Goal | Opportunity1 /2 | Slots used | Baselines | Self-tests | Native | Parent alignment | Search forwards |
|---|---|---:|---:|---:|---:|---:|---:|
| TTFT |failed1 /failed2|3|1|3|0|0|141|
| Single |failed7 /failed2|9|1|9|0|0|1996|
| Multi |failed6 /accepted4|10|1|8|1|1|1992|
| Total |5 failed /1 accepted|22/48|3|20|1|1|4129|

Six primary TaskRewriter calls/sessions (two per goal), zero explicit repairs or replacement draws. The recorded five failures remain failures; the audit distinguishes their causes rather than relabeling them successes.

| Trajectory | Confirmed outcome and interpretation |
|---|---|
| TTFT rounds1/2 |1500.208450s /1500.152025s prompt transport ReadTimeouts, then MessageAbortedError; neither returned StructuredOutput and no helper result was valid. These25-minute call limits are distinct from the90-minute goal deadline. Provider cause and what more time would achieve are unknown. |
| Single rounds1/2 |Both exact final returned computational bundles had valid helper results, but were falsely rejected after their child source replaced the writable comparison mirror. Baseline fallback stays frozen. |
| Multi round1→round2 |Round1 had the same false rejection despite a helper-valid returned attention/MLP bundle. Round2's decoder composition passed two helper tests, one aligned parent witness and one native trial, becoming the sole accepted artifact. |

`TaskRewriter.check_output` treated `candidate/bundle-sources.json` as the parent authority after agent writes, so it compared child to child. The immutable request snapshot still contained the distinct original parent. The source auditor reproduced3/3 rejections; substituting only the original parent read view made3/3 pass, and all three passed the downstream actual-parent structural validator. These were not three unmodified kernels or failures caused by structural normalization erasing real changes.
Agent input-metadata writes triggered the **framework comparison-reference integrity bug**; malicious intent is not established. A successful StructuredOutput capture was schema acknowledgment, not source acceptance or evidence of a second `check_output` invocation. Intermediate helper failures remain real and separate:20 helper requests yielded7 valid/13 invalid, including local KV, logits/NLL, DynamicCache-adapter and overlapping-site failures.
Private helper validity does not supply missing native/aligned-parent evaluation or establish improvement. Neither the diagnosis nor T10 can promote those bundles, replace fallback rows, retry beyond the original two rounds, reopen budget, or repair goal-specific selections retrospectively. The known guard defect prevents attributing the blocked full-flow outcomes solely to LLM capability.
There were25 admitted search evaluations (22 slots+3 baselines),12 valid/13 invalid. Private valid helper evaluations were not promoted when the returned artifact failed admission. Each opportunity used≤8 slots; unused slots were not topped up, and there were no C2 endpoint probes or extra execute-best.
**Search-only** multi score268.461921tokens/s versus fresh aligned parent244.647213 (initial unaligned baseline246.958335) selected the artifact. Those are not the recovered heldout214.785688/196.880185 values; workloads/instrumentation contexts must not be mixed.

| Recovery inventory | Count |
|---|---:|
| Fresh quality /measurement records |36 /36|
| Scored waves /completed requests |84 /168|
| Scored output tokens |15408 =48 TTFT+3072 single+12288 multi|
| Scored /warmup model forwards |4656 /514;6 warmup waves|
| Quality /fixture forwards |5376 /4|
| Total recovery model forwards |10550 =10546 attempt forwards+4 preparation forwards|

The failed original T8 admitted0 cells but performed model/fixture work whose forward count was not persisted. Thus4129 search plus10550 recovery is **not** a complete whole-campaign GPU inventory; readiness and unknown failure-preparation work remain separate.
Preparation also exposed slow quality arithmetic: native64-forward A/A quality wall fell from60.574330s to5.442113s after CPU vectorization of full-vocabulary reductions, with unchanged quality thresholds/positions. The observed11.13× wall ratio is evaluator engineering, **not an agent GPU-kernel gain** or an isolated native CPU/GPU time decomposition.

## Calendar time and provider ledger

Shared S=`1789849886.5547094` (2026-09-19 20:31:26.554709UTC); search cutoff S+9000; **original final deadline=`1789860686.5547094` (23:31:26.554709UTC), unchanged**. Goal-local90-minute admission limits also remained active.

| Recorded interval | Elapsed /boundary |
|---|---|
| TTFT /single /multi search walls, concurrent |3009.849 /2504.227 /1917.191s; do not sum as makespan |
| Initial S→client closure after failed T8 |3036.013s /50m36s, ending21:22:02.567712UTC |
| Original search join→heldout launch |4.610s; automatic, not a human review pause |
| Failed closure→recovery start |4295.000s /1h11m35s; repair/review/publication/deployment/wait calendar gap |
| Recovery process |22:33:37.567899–22:45:15.587656UTC;698.020s |
| Shared S→recovery end |8029.033s /2h13m49s, including the gap; not GPU busy time |
| Final deadline headroom /drain |2770.967s /0s |
| Plan activation12:58:09.432UTC→S /recovery end |7h33m17s /9h47m06s; preparation/approval/implementation/waits included |

Broader elapsed spans are not active labor or continuous compute; later evidence export, analysis and publication are outside those endpoints. Cleanup receipts report all owned processes closed and all four cards idle; T9 did not launch a new check/model.
The following reaggregates the existing scoped T7 ledger for `zhipuai/glm-5.3`, not a new session audit or verified provider invoice. Input/output/reasoning/cache-read fields stay separate.

| Goal | Assistant records | Input | Output | Reasoning | Cache-read | Cost-field sum |
|---|---:|---:|---:|---:|---:|---:|
| TTFT |107|309178|29919|134972|9848960|3.71909920|
| Single |127|348318|39839|122127|15257984|5.16737144|
| Multi |105|275991|25297|81927|10145984|3.49612884|
| Total |339|933487|95055|339026|35252928|12.38259948|

Assistant records are not provider HTTP request counts; cost fields are OpenCode estimates, currency/billing unverified, and aborted calls may limit completeness. Recovery added zero provider/search calls. T7 audited442 saved tool parts/211 bash commands:20 helper commands,19 reaching GPU work; no private bypass identified **within saved evidence**, not a claim about unrecorded activity.

## Reproduction evidence and limits

Local evidence root `results/c3-qwen3-4b-operator-goals/`: root `analysis.json` contains full-precision summaries/counts/provenance; original main exports preserve search attempts/selections and failed matrix; `recovery-1870e8e/host-A/heldout-recovery-1870e8e/` preserves recovered matrix, row identities and raw timestamps/quality losses. Original analyzer outputs remain untouched.
Contract SHA256 `9d1ef6a79bf99eb6ac0e639d82e894227b403124764ba4018531c9ddf179eaf8`; corpus `d60e5c108e79c8327a74199d5c093b117f9e64661cff4356e33a723dceb30a02`. Readiness retains10 oracle manifests/44 prompt references (2 calibration+42 heldout); prior receipts verify reference identities without duplicating weights/full heldout oracle payloads here.
Main evidence index has219 `files` entries; recovery export has111 artifacts, while its wider metadata/evidence index has128 entries. These nested scopes are **not additive**. T9 checked the two matrix hashes and numerical records, not broad old-C2/weight hashes. No raw data was rewritten.
Independent arithmetic used existing NumPy2.5.2 via `uv run --offline --no-project --python .venv-v5/Scripts/python.exe python -B -`; it imported no model and ran no new experiment. Detailed method/checks: `.omo/evidence/v5-c3-qwen3-4b-operator-goals/T9-analysis.md`.
This pilot supplies **partial C3-B evidence**: one real source-composition artifact with positive descriptive throughput observations under manual frozen tasks. It does **not** fulfill the stronger new-custom-GPU-kernel claim or establish goal differentiation, three optimal implementations, automatic task generation, statistical significance, online serving performance, broad language quality, or transfer. Thirty-six valid recovery blocks establish matrix completion, not overall framework/scientific success. Historical C2 data are not pooled.
**T10 prospective fix CPU-verified:** [82daf15b460004707e55dbdebb789e2788921527](https://github.com/Fudan-SMI-lab/opop/commit/82daf15b460004707e55dbdebb789e2788921527) captures immutable host-side parent strings before the base invoke loop in an agent-scoped ContextVar and restores its token in finally; editable mirrors are not comparison authority. Independent verification passed80 tests with90 existing warnings in44.38s and clean LSP on both files, including all3 exact exported-return CPU replays without importing/executing candidate functions or changing original failed statuses. Only `src/kernel_optimizer/agents/task_rewriter.py` and `tests/test_c3_rewriter_parent_snapshot.py` changed; helper-only changes, genuine nonchanges and concurrent/nested cleanup remain covered. See local `T10-parent-snapshot.md` and `T10-publish.md` receipts. This fix was not used by original search af77534 or recovery1870e8e and establishes no retrospective acceptance, new model quality, performance or selected winner; no permission platform, timeout fix, experiments or threshold relaxation was added.
**Publication status:** this report archives the completed analysis and actual prospective-fix provenance. Raw data and local receipts remain local evidence, not a published dataset. The existing single F1 review remains pending and will check data plus the scoped fix; no extra review wave or scientific run is authorized.
