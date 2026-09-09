# L3:43 第二轮结果 — 2.7628 ms final(bf16),首个 bf16 获胜者,首个三代改写获胜

Run: `run-l3-43-20260909-015247`,box 2(RTX 4090),commit `e840642`(首个天花板可见 + 后端可切换的 L3:43 run),agent 模型 `zhipuai/glm-5.3`。任务 KernelBench level3/43_MinGPTCausalAttention,train 模式。全部数字取自该 run 自身 `events.jsonl` 的 `RUN_FINISHED` 汇总与 `report/report.md`,非通知转述。

## 头条

| | ms | 来源 |
|---|---|---|
| **final_reeval median** | **2.7628** | 独立新进程复测,5/5 正确 |
| final_reeval mean | 2.76 | 同 job |
| tuned_ms(获胜 trial) | 2.7674 | 调参循环——复测快 0.2% |
| 上轮 incumbent(手工复测) | 3.0126 | run-l3-43-20260908-053708 |
| eager | 21.457 | 100 采样中位 |
| eager_tf32 | 18.239 | |
| torch_compile | 13.975 | |
| **torch_compile_tf32** | **10.985** | 最强基线 |

加速比(中位):**7.77× eager,5.06× torch_compile,3.98× torch_compile_tf32(同精度判定基线)**。获胜者以 bf16 计算,`beats_same_precision_baseline: true`。较上轮 3.0126 提升 **8.3%**,且这次是有 final_reeval 的正式收尾(上轮是主动终止 + 手工复测)。

## 获胜者:三代改写(Loop C 的最强证据)

`cand-173b12ae` 是一个 **rewrite**,谱系深度 3:

```
seed cand-9739eeca (3.591)  →  rewrite cand-16439def (2.822)  →  rewrite cand-173b12ae (2.767)
```

这是本项目首个由**多代连续改写**拿下的获胜者——之前 L3:21/L3:48 的获胜者都是种子,改写只改善了非获胜族。这一轮改写改善了**全部四个族**(3.195→3.174、3.713→2.945、3.591→2.767 两跳、3.318→2.997),Loop C 在这个任务上实打实起了作用。获胜配置:`COMPUTE_DTYPE=bf16, ACC_DTYPE=fp32`,ATTN tile 128×64/8 warps,GEMM tile 64×256×32/8 warps。

## 七点检查

1. **final_reeval_ms** = 2.7628 median,`final_reeval_ok: true`。
2. **墙钟结束**:`budget_exhausted` at 13.51 h(超支 12.6%);两条 `WALL_CLOCK_REACHED` 记录显示它在 round 2 的 fam-1402fc69 前中止了剩余候选。**第 3 个由墙钟结束的 run**(20 个里)。四族全 active,fam-f5d9d4cd 用了 2/5 改写轮,其余 1/5——预算未被改写轮约束,墙钟先到。
3. **fp64 救援**:获胜 trial 3/3 快速正确性经 fp64 相对臂通过(bf16 有 fp32 的指数范围,避开了 cosine 溢出类,但需相对门确认精度)。final_reeval `final_reeval_ok: true`。
4. **首个 bf16 获胜者,与 L3:21 完全相反**:此任务 bf16 **369 complete / 30 fail**(胜出精度),而 L3:21 上 bf16 是 0-for-100。同一个门、同一个多器,判决相反——这正是门按任务自适应、不做 dtype 禁令的价值(印证 memory `opop-all-three-tasks-floors-below-gate`:L3:43 的 bf16 比参考自身 ieee/tf32 一致性更高)。fp16 206✓/21✗、tf32 71✓/40✗、ieee 63✓/40✗。
5. **F5 prescreen**:129 个 infeasible_shared_memory trial 全部编译期拒绝(上轮同类失败是 180/1004),正确性半边完美。
6. **Loop D 零执行**:0 novelty。四族 < max_families_total=6,但墙钟在 novelty 被调度前耗尽——与 L3:21 同因(D 排在调度最末)。
7. **后端计数**:14 候选全 triton,0 `BACKEND_DECLARATION_MISMATCH`,0 CUDA。这是 D3 里说的两个非触发任务之一(causal attention 非 strict-IEEE dot-bound 主导),后端切换可用但未被选择——合法结果,真正的触发检验在 L3:48。

## 质量信号

- **获胜者分离良好**:std 0.0045 ms,per-trial CV 中位 0.3%(全项目最干净之一);近平局带名义 10/709,但带宽被少数高方差 trial 撑大,获胜者本身与场分离清晰。
- **agent 可靠性**:全程 0 AGENT_CALL_FAILED,1 repair,10 rewrites。
- **深度改写有效**:14 个候选里 10 个是 rewrite,最好的 5 个 tuned 值(2.767/2.822/2.945/2.997/3.079)全部来自 rewrite,没有一个种子进前五。

## 对比纪律

上轮 incumbent 3.0126 是主动终止后手工复测的中位数;本轮 2.7628 是正式 finalize 的 final_reeval 中位数。同任务、同箱(box 2)、同基线口径,**提升 8.3%**,tuned-vs-final 差 −0.2%。同卡 vs 外部 CUDA 的对比见 `results-summary-l3-vs-external.md`(外部 L3:43 为 strict-fp32,我方 bf16,分精度档另述)。
