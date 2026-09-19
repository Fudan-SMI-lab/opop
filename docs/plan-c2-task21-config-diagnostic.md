# Task21 rep0: bounded frozen-artifact configuration diagnostic

## Scope and question
User-authorized priority2 diagnostic, not a new candidate campaign. Archive this protocol before GPU work; parent reviews the archive/CPU staging before execution. Preserve deployed evaluator source `cda113070c8ba3a91f480f42b68d196187796cda`, old reports/raw/deadlines, and all source bodies. No models, TPE, redraw, body repair or storage ablation.

Question: does a small, predeclared configuration intervention on frozen C2 rep0 reduce its gap to a freshly measured frozen G0 rep0? Separate an observed search/configuration miss from remaining structural limitations; these cells cannot prove an intrinsic ceiling or attribute the entire gap to one mechanism.

**C2 selected already uses FP16 compute, RAW1_DTYPE=fp16 and SPLIT_K=1. Its raw2/raw3 allocations are hardcoded FP32, not selectable knobs.** Wider projection BLOCK_K and an alternate feasible joint point do not change those storage bodies. No raw2/raw3 FP16 ablation is authorized here. G0's FP16 storage/precasts/schedule are a different frozen structure, not a one-variable control for C2.

## Exact inputs and full parameters
Donors on HOST-B under `/root/autodl-tmp/c2-helper-pilot/main/B/wave1/`:

- G: `B0/information/retune/report/selected.py`, SHA-256 `fdd95ca58e696ba081a0f742c82fd49a600952ad88e29f3c7e83faeb56ce1f26`.
- C: `B1/information/retune/report/selected.py`, SHA-256 `c4982b91b5d0456c409ccdfda193b1b6931436f07862939702df404c471c3849`.
- Historical T: `B1/information/retune/events.jsonl`, native `TRIAL_DONE` for `tr-5dfdde92`, candidate `cand-3a36b550`, space `sp-f3f4954d`; matching `candidates/cand-3a36b550/trials/tr-5dfdde92.py` SHA-256 `e7afcc1470b6b94eb4065ca91f3ba276bd8101e82799dc0eefb2b1e958a95425` beneath that retune directory. Recover all values from this record, not a tuning-summary abbreviation. Its prior quick score is provenance only, not a new control.
- Reference: `/root/autodl-tmp/c2-helper-pilot/inputs/task21/reference/reference.py`, SHA-256 `24e555726c4856120cceaa8dad2dbbca798a56c0a99313c77fbe84c16e37c20b`; shape10x112x224x224, init112/192/5/2/6, TRAIN semantics.

G selected, complete14-key ParamSet:
```json
{"values":{"BLOCK_M":64,"BLOCK_N":128,"BLOCK_K":64,"DW_BLOCK_H":32,"DW_BLOCK_W":128,"EW_BLOCK":128,"FIN_BLOCK_C":32,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","STORE_DTYPE":"fp16","GEMM_WARPS":4,"GEMM_STAGES":3,"DW_WARPS":4,"EW_WARPS":2}}
```
C selected, complete19-key ParamSet:
```json
{"values":{"BLOCK_M":32,"BLOCK_N":128,"BLOCK_K":16,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":64,"STATS_BLOCK_C":16,"DW_BLOCK_H":16,"DW_BLOCK_W":32,"EW_BLOCK":1024,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","RAW1_DTYPE":"fp16","GEMM_WARPS":4,"GEMM_STAGES":2,"STATS_WARPS":2,"DW_WARPS":8,"EW_WARPS":4}}
```
T historical trial, complete19-key ParamSet:
```json
{"values":{"BLOCK_M":64,"BLOCK_N":128,"BLOCK_K":16,"SPLIT_M":32,"SPLIT_K":4,"STATS_BLOCK_M":256,"STATS_BLOCK_C":16,"DW_BLOCK_H":16,"DW_BLOCK_W":128,"EW_BLOCK":512,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","RAW1_DTYPE":"fp16","GEMM_WARPS":8,"GEMM_STAGES":1,"STATS_WARPS":2,"DW_WARPS":8,"EW_WARPS":4}}
```

## Six fixed cells and18-attempt order
| Cell | Frozen source | Complete parameter rule | Intended comparison |
|---|---|---|---|
| G | G | G selected exactly | Fresh same-card baseline |
| C | C | C selected exactly | Fresh selected-C2 control |
| K32 | C | C with only BLOCK_K=32 | K16→32 at selected partners |
| K64 | C | C with only BLOCK_K=64 | K16→64 at selected partners |
| T_SK1 | C | T with only SPLIT_K=1 | Compare directly to T_SK4 |
| T_SK4 | C | T exactly, SPLIT_K=4 | Sixth cell: necessary matched control for the SK switch |

T_SK1 versus C changes several partners and is **not** a single-variable comparison. Adding T_SK4 costs only three fixed full attempts and avoids using a historical quick measurement as its comparator. CPU checks must establish T source body equals C excluding PARAMS and its materialization reproduces the recorded trial hash.

All formal calls serial on HOST-B GPU0, UUID `GPU-9c819228-dcf4-f1da-f21e-6b77b1c58be1`, PCI52:00.0. Freeze these rounds before outcomes:
1. `G, C, K32, K64, T_SK1, T_SK4`
2. `T_SK4, T_SK1, K64, K32, C, G`
3. `K32, C, G, K64, T_SK4, T_SK1`

Each cell occurs once per round (balanced counts); the first two rounds are exact reversals and the third is a fixed interleaving, not a claim of perfect position balance. Execute **at most18 full attempts**, three per cell. Compile/correctness failures consume their scheduled attempt and remain failures; no replacements or adaptive cells. K64 is predeclared regardless of K32's outcome. Unexpected helper/infrastructure failure stops with explicit unmeasured cells, not an inline patch/restart.

## Formal execution contract and clock
Use the existing formal helper/operator route, absolute `/root/autodl-tmp/orch-venv/bin/python`, deployed `v5/src:v5` plus configured KernelBench PYTHONPATH, local `CUDA_DEVICE_ORDER=PCI_BUS_ID` / `CUDA_VISIBLE_DEVICES=0`. No agent/provider service. Recheck exclusive availability before launch; no concurrent profiling or other workload on the used card.

Generate context/usage via actual `self_test_context_factory` and live evaluator with the true reference. Every cell supplies its complete native `params.json` to `python -m kernel_optimizer.agents.self_test --context ... --candidate ... --params ... --mode full --output NEWDIR`. Fresh per-attempt output/materialization; no stale module dictionary. Keep seed0, full100 performance/5 correctness, FP32 reference, dual-witness relaxed policy, fraction0.99, cosine0.99985, FP64 rescue enabled and multipliers2/3 unchanged. Record helper validity, compiled/correct/formal_ok, null versus valid score, rescues, all samples, actual worker commands and submission times. Static jobs are counted separately from18 full attempts.

Declare actual S only at authorized execution, before the first call: **admission cutoff=S+3600s** includes optional profiling. Pass that cutoff through the existing formal-helper context scope. No new submission after cutoff; already-admitted work may drain under existing worker rules and must be reported. No extension. Setup/protocol/staging and post-run notes/export are separate from execution elapsed. Do not instantiate a clock during this protocol/CPU stage.

## Optional profile: bounded and secondary
Only after all scheduled formal attempts, and only if existing torch.profiler CUDA support is ready with at least120s admission budget remaining: at most **one trace each for selected G and selected C**, serialized, at most60s per trace including setup/warmup. At most three uninstrumented warmups and one instrumented forward per artifact, same fixed shape/mode/seed/weights/default point. Skip a selected artifact that failed formal validity. If capability is absent, setup is troublesome, or the bound is insufficient, skip and report unknown; no installation/Nsight/new profiling platform or retry.

Keep per-kernel CPU/CUDA durations and available ATen/cast/allocation events; allocator evidence is not measured DRAM traffic. Profiler-instrumented timing is not the formal ranking metric. Profile only these two selected points, not the other configurations; no profile-driven extra cell, source change or storage ablation.

## Reporting and stop
For each cell retain all three outcomes; a fully valid cell reports median-of-three block medians and complete block range. Failed/missing cells are not rescued by the fastest surviving block. Report K32/K64 versus C, T_SK1 versus its new T_SK4 comparator, and each resolved C2 point versus fresh G, with validity/rescue and uncertainty explicit. These finite contrasts can demonstrate a local configuration effect, not exhaustive search, unique structural causality, general efficacy or statistical significance. Any remaining gap may involve hardcoded storage/schedule/weights and untested partners; no body-ablation follow-on is automatic.

Remote root `/root/autodl-tmp/c2-task21-config-diagnostic/`; local raw `results/c2-task21-config-diagnostic/`. Freeze source/config/context/ParamSet/materialized hashes and order before launch; preserve old inputs/raw, export allowlisted evidence without credentials, verify touched donors only, and clean up only owned processes. Tracking `.omo/plans/v5-c2-task21-config-diagnostic.md`; receipt directory `.omo/evidence/v5-c2-task21-config-diagnostic/`. Stop after bounded execution, analysis and one compact final verification; no new generation/TPE campaign.
