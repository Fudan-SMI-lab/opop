# L3:48 在 A800 上的结果 — 0.9728 ms(bf16),8.23x 同精度,`excessive_speedup` 已复核

Run: `run-l3-48-20260911-052647`,box 3(**NVIDIA A800 80GB PCIe**,sm_80、108 SM、torch 2.8.0+cu128、
triton 3.4.0),agent 模型 `zhipuai/glm-5.3`,任务 KernelBench level3/48_Mamba2ReturnY,
`v3.diagnosis.mode = vector` + `expectation_ledger = true`(**处理臂配置**,但**不是**配对实验的一臂——
另一张卡、另一个任务,只能与本箱自身的历史比)。

全部数字取自该 run 自身 `events.jsonl` 与 `jobs/final-reeval-eval-5a4c820c.out.json`。

---

## 头条

| | 值 | 来源 |
|---|---|---|
| **final_reeval median** | **0.9728 ms** | 独立新进程复测,**5/5 正确性通过**(fresh inputs) |
| final_reeval mean | 0.981 ms | 同 job,100 采样,std 0.0918 |
| tuned_ms(获胜 trial) | 0.9585 ms | 复测比 tuned **慢 1.5%** |
| eager | 13.9597 ms | 100 采样中位 |
| eager_tf32 | 13.4349 | |
| torch_compile | 8.5832 | |
| **torch_compile_tf32(同精度最强基线)** | **8.0527** | |

加速比(mean 口径):**14.2712x eager / 8.7564x torch_compile / 8.2263x torch_compile_tf32**;
median 口径 14.35x / 8.82x / 8.28x。获胜精度 **bf16**,`beats_same_precision_baseline: true`。

获胜者 `cand-56411b63` 是**种子**(非改写),配置
`COMPUTE_DTYPE=bf16, DOT_MODE=split3, BLOCK_LEN=32, NUM_WARPS=2, NUM_STAGES=3`。
`DOT_MODE=split3` 是**三项分裂补偿 dot**(`ah·bh + ah·bl + al·bh`),正是 memory
`compensated-dot-passes-plain-dot-fails` 记录的那条通路 —— 低精度能过 fp64 相对门是因为它补偿了,
不是因为门松。本次 `fp64_rescued_trials = 0`,即**获胜者不靠相对臂救援**。

---

## `excessive_speedup_flag: True` 已复核 —— 接受

`RUN_FINISHED.payload.summary.best.excessive_speedup_flag = true`。落盘复核结论:**这是一个标记,
不是拒收,且这里应当接受。**

从 `jobs/final-reeval-eval-5a4c820c.out.json` 逐字段读到:

```
ok = True                      correct = True
trials_passed = 5              trials_total = 5
correctness_mode = dual_witness_relaxed
speedup_vs_ref_in_worker = 14.294246650252504
excessive_speedup = True
excessive_speedup_note = "14.3x vs reference (14.023 ms -> 0.981 ms) exceeds the 10x
    plausibility threshold, but all 5/5 correctness trials passed on fresh inputs;
    accepted and flagged for review"
fp64_gate_enabled = True       fp64_rescued_trials = 0
```

`worker_main.py:2442-2482` 的语义:门的目的是抓**跳过工作**(缓存输出、省略计算),表现为不可信的
加速;**通过全部正确性 trial 的候选已在新输入上产出了参考的数值**,硬拒会为"跑得快"这个罪名丢掉一个
已验证正确的 kernel。所以 `correct` 决定接受,加速只抬标记;`correct=False` 且超阈仍是硬失败
(计时作弊 fixture 按张量身份缓存,其 fresh-input 正确性过不了)。

**这个门曾经真的伤过我们**:L3:48 的 `cand-c18203b6` 有四个 trial 在 11.1x–13.9x 被拒,
`correct=True, trials_passed=3/3`,而邻近 8.95x 的点被接受 —— 判决由噪声落在 10x 哪一侧决定,
**且被丢掉的正是最快的点**,把报告的最优值系统性压低。修复(标记而非拒收)已记入 memory
`opop-v2-speed-guard-must-not-override-correctness`。

**本 run 内 `failure_kind = excessive_speedup` 的 trial 数:0**(320 个 TRIAL_DONE:196 complete /
98 correctness_mismatch / 22 infeasible_shared_memory / 3 runtime_error / 1 timeout)。所以门只在
final re-eval 抬了一次标记,没有拒掉任何测量。

**为什么 14.3x 本身是物理上讲得通的**,不是"太快所以可疑"。用本箱实测标定
(`CALIBRATION_LOADED`: DRAM **1.6858 TB/s**)与任务代价(`TASK_COST_MEASURED`:
compulsory 1350.57 MB、flop 30.367 GFLOP、reference materializes **40.5x** compulsory over 130 ops):

```
DRAM 下界(不可避免流量)  = 1350565888 / 1.6858e12 = 0.8011 ms
bf16 计算下界             = 30.367e9 / 233.96e12   = 0.1298 ms
=> 本任务在本卡上的物理最快 ≈ 0.8011 ms(带宽侧绑定)
=> 相对 eager 13.9597 ms 的物理最大加速 = 17.50x
```

我们拿到 0.9728 ms = **DRAM 屋顶的 82.4%**,即物理最大加速的 **82%**。**14.3x 不是超物理,
是逼近物理。** 加速的来源写在 task cost 里:参考实现搬运了 40.5x 它不可避免的流量(130 个 op 的
中间量写出再读回),融合掉这些就是这个任务的最大杠杆 —— 而获胜者的 `candidate_aten_ops = 1`
(参考 130),`candidate_aten_bytes = 1073.7 MB`(**低于** 1350.57 MB 的 task floor,因为 aten 层看不见
被融进 kernel 的流量,这正是"候选已融合"的指纹,检查器 8/8 如实报告、从不 clamp)。

---

## 跨卡对比:A800 vs 4090(同任务、同参考、同评测配置)

| | 4090 r1 | 4090 r2 | **A800** |
|---|---|---|---|
| run | `run-l3-48-20260907-202457` | `run-l3-48-20260909-115701` | `run-l3-48-20260911-052647` |
| final median | 1.41(mean) | 1.5616 | **0.9728** |
| 精度 | fp16 | fp16 | **bf16** |
| 同精度加速 | 9.2908x | 8.4516x | 8.2263x |
| 实测 DRAM 屋顶 | 0.9094 TB/s | 0.9094 | **1.6858** |
| 该卡 DRAM 下界 | 1.4851 ms | 1.4851 | **0.8011** |
| **达屋顶比例** | 105.3%(见下) | **95.1%** | **82.4%** |
| 墙钟 | 15.75 h | ~12 h | 12.383 h(事件跨度 13.67 h,含一次 resume) |
| `excessive_speedup_flag` | True | True | True |

**绝对更快、相对更远**:A800 的带宽是 4090 的 1.85x,而结果只快 1.61x ⇒ **把额外带宽转化成速度的
效率 71%**。4090 上 r2 已达自身屋顶 95.1%(r1 的 105.3% 说明那一版的 `compulsory_bytes` 口径与 r2
不同——r1 的 `TASK_COST_MEASURED` 未落在可读位置,不能直接比,只作参考),A800 还留 82.4% ⇒
**A800 上还有约 0.17 ms 的物理空间,4090 上几乎没有了。** 这与 memory
`l3-48-is-bandwidth-bound-at-90-percent`(4090 上 1.64 ms 已达 90.4%)一致,并把它推进到 95%。

三次 run 的 `excessive_speedup_flag` 全为 True(而两个 L3:43 臂全为 False)—— **10x 阈值对
L3:48 这个任务系统性偏低**,因为这个任务的参考实现本身浪费 40.5x 流量,融合后的合法加速天然 > 10x。
这是**任务性质**,不是候选可疑。记入 §"后续"。

---

## Loop C 只跑了一轮,且那一轮的 S2d 证据为零

| | |
|---|---|
| 候选 | 6(4 seed + 2 rewrite) |
| 发布空间 | 8 | 
| trial | 320 |
| `REWRITE_PRODUCED` | 2(同一族 `fam-710722dd`) |
| `EXPECTATIONS_RECONCILED` | 1,**`n_declared = 0`(空账本)** |
| `REPAIR_PRODUCED` | 11 |
| `AGENT_CALL_FAILED` | **2**(repair 与 rewriter,均 `prompt transport error (ReadTimeout): timed out`,各吃满 1500 s) |
| `RUN_INTERRUPTED` | 1(在 +1.277 h,`terminated by signal or Ctrl-C`) |
| 结束 | `WALL_CLOCK_REACHED` 12.383 h,`stopped_before_family: fam-e6fadf96` |

**那一轮的 S2d(a) 证据是零,原因已定位**:两个 rewrite 候选的 `change_summary` 都是
`[recovered from sandbox after a transport failure; the agent's own summary never arrived]`,
`hypothesis_id` 为空 —— 即 rewriter 的 HTTP 响应超时,候选文件由 `rescue_from_sandbox` 从沙箱救回。
**`rescue_from_sandbox` 只重建文件**,而 S2d 的 `expectations` 只存在于 JSON 响应里,所以救回的候选
没有声明 ⇒ 账本空 ⇒ `conversion = flat, gain = 0.0`。这是已记录的待修项(见 §后续 #4):
救援函数的 docstring 把 `hypothesis_id`/`change_summary` 称为 "narration the pipeline does not gate
on" —— 在 S2d 之前为真,**现在为假**。

**墙钟归属**:事件跨度 13.670 h 而 run 自报 `elapsed_hours = 12.383`,差 1.29 h 正是 resume 之前那段
(memory `a-resumed-runs-own-clock-restarts`)。两个最大的非 agent 空档是 0.83 h 与 0.70 h 的
`TRIAL_DONE → TRIAL_DONE`,与已记录的 A800 大 shared 配置 ptxas 长尾一致(A800
`max_shared_bytes_optin = 166912` vs 4090 `101376`,21% 的 trial 用了 4090 物理跑不了的区域)。
两次 agent 超时又吃掉 0.83 h。

---

## 七点检查

1. **final_reeval**:0.9728 median / 0.981 mean,`ok=True`,5/5 正确性,`fp64_rescued = 0`。
2. **墙钟结束**,`stopped_before_family` 明确 ⇒ 轮次是**时钟给的下界**,不是收敛。
3. **`excessive_speedup` 已复核并接受**(§2),run 内 0 个 trial 因它被拒。
4. **精度**:bf16 获胜且用补偿 dot;`correctness_mismatch` 98/320 = 30.6%,高于语料的 ~21%,与本任务
   的低精度敏感性一致(未修,属独立课题)。
5. **prescreen**:22 个 `infeasible_shared_memory` 全在编译期拦下;8 次 prescreen(每空间一次)。
6. **Loop D 零执行**:4 族 < `max_families_total = 6`,墙钟先到 —— 与 L3:21/L3:43 同因。
7. **S3/S4′ 落地**:64 条 dimension record / 8 次诊断,8/8 的 compute ceiling 指明精度,
   `unreachable ceilings` 如实报出三条(fp16 85.5% / bf16 85.4% / tf32 90.6% 的 Triton 可达比例),
   `below-floor` 8 条全是 `candidate_aten_bytes`(= 已融合的指纹),`readings exactly on the floor`
   6 条全是 `candidate_aten_ops = 1`(**定义下界本身,不是 clamp**)。

---

## 待办(不在本文档修)

1. **`excessive_speedup` 的 10x 常量对 L3:48 系统性偏低**:该任务参考浪费 40.5x 流量,合法融合加速
   天然 > 10x,三次 run 三次抬旗。修法必须**泛化**:阈值应从**任务自身的物理上界**导出
   (`compulsory_bytes / dram_tbs` 对 eager 的比,本例 17.50x),而不是按任务硬编码一个数。
   这样"跳过工作"仍会被抓(超过物理上界即不可能),而合法融合不再抬旗。
2. **`rescue_from_sandbox` 丢掉 S2d `expectations`**:需要 prompt 契约改动(让声明也落到沙箱文件),
   属高风险/较大改动,已记录待统一处理。
3. **A800 的大 shared 配置 ptxas 长尾**:治法是**给 prescreen 独立的短超时(60–120 s)**,
   **绝不可**降 `build_timeout_s`、**绝不可**按卡型收紧探索空间(见 memory
   `never-narrow-the-search-space-to-control-cost`)。
