# 双臂对照结果 — L3:43,vector+ledger 臂 2.84 ms vs label 臂 3.11 ms(−8.68%)

**日期** 2026-09-11 · **对照(control)** box 1 `run-l3-43-20260911-053020` · **处理(treatment)** box 2
`run-l3-43-20260911-052630` · 两臂均 RTX 4090、启动于 `dceda97`、agent 模型 `zhipuai/glm-5.3`、任务
KernelBench level3/43_MinGPTCausalAttention(train 模式)。

**所有数字取自两个 run 各自 `events.jsonl` 的落盘记录**(processed by `scripts/check_wrapup.py` 与
`scripts/rederive_per_candidate_ledger.py`),不取通知转述、不取 `report.md` 的二手汇总。box 2 的
`events.jsonl` 在分析前重新 scp 了一份(先前的副本停在 14:11,缺最后 3 h),两次读数的候选数由 8 变 9、
ledger 由 1 条变 3 条 —— **这份文档基于完整的 1061 行**。

---

## 1. 自变量:只有两个开关,且处理臂拿到的原始信息**更少**

`arm parity` 检查逐字段比对了两臂 config,差异恰好三项:

| key | control | treatment |
|---|---|---|
| `v3.diagnosis.mode` | `label` | `vector` |
| `v3.diagnosis.expectation_ledger` | `False` | `True` |
| `wsl.venv` | `/root/autodl-tmp/orch-venv` | `/root/autodl-tmp/kernel-opt-venv` |

第三项是路径形状的键,且**两臂记录了同一个 calibration identity**
`NVIDIA GeForce RTX 4090|8.9|128|2.13.0+cu129|12.9|3.7.1` —— 不同的解释器**路径**不是不同的解释器,
所以被 parity 检查显式豁免并打印出来,而不是静默忽略。

**关键事实(必须写进论文,否则读者会读反):处理臂收到的原始数字更少,不是更多。** 两臂的
`bottleneck.md` 都含相同的 task-cost 段与 machine-ceiling 段(逐字相同的 8 条屋顶行)。差别在候选级:

- **对照臂**:一行 `## Verdict: **resource_limited**` + **31–32 行原始 key-value**
  (`gpu_ms = 3.1861759424209595`、`occupancy = 0.1667`、`pct_of_compute_peak = 73.7`、
  `n_regs = 255`、`arithmetic_intensity = 990.439` …… 外加派生阈值),文件 59–64 行。
- **处理臂**:**0 行候选级原始数字**,换成 8 条 per-dimension 判断行(binding / near-binding /
  slack / NOT MEASURED / measured-but-NOT-RANKED)+ "How much room is left"(到 **FLOOR** 的距离,
  不是到屋顶)+ polarity/rankability 声明 + 分母出处(带 calibration identity 与测量日期),
  文件 52–53 行。

所以这个实验测的是**"结构化判断取代原始倾泻"**,而不是"多给信息"。它**不能**区分
"结构有帮助" 与 "原始倾泻有害" —— 需要第四臂(判断 + 距上界真实数值)才能分开,见 §7。

---

## 2. 搜索量对等(不是靠 config 相同推断的,是数出来的)

memory `equal-configs-do-not-imply-equal-search` 记录过:config 相同 ≠ 搜索量相同。这里是实数:

| | control | treatment |
|---|---|---|
| 发布空间数 | 16 | 16 |
| trial 总数 | **640** | **640** |
| 单空间最大 trial | 40(= `trials_per_space`) | 40 |
| 每候选空间数分布 | `{2: 7, 1: 2}` | `{2: 7, 1: 2}` |
| K 扩展次数 | 7 | 7 |
| complete / infeasible_shared / 其他 | 505 / 134 / 1 timeout | 497 / 142 / 1 correctness |
| agent 调用 | generator 1, parameterizer 20, analyst 16, rewriter 3 | 1 / 17 / 16 / 3 |
| agent 总时长 | 3.76 h | 3.85 h |
| `AGENT_CALL_FAILED` | 0 | 0 |
| resume | 无 | 无 |
| 事件跨度 | **12.47 h** | **12.02 h** |

**唯一的不对等是墙钟**:对照臂被 `WALL_CLOCK_REACHED` 在 12.459 h(超 12 h 预算 3.8%)截断于一个
candidate batch(`pipelined: 1, skipped: 1`),处理臂 12.013 h 干净收尾、无墙钟事件。**方向对处理臂
不利**(对照臂多跑了 0.45 h 仍更慢),所以这个不对等不能解释处理臂的领先。

---

## 3. 头条(J2-5,取 `final_reeval_ms` 不取 `tuned_ms`)

| | control (`label`) | treatment (`vector+ledger`) |
|---|---|---|
| **final_reeval mean** | **3.11** ms | **2.84** ms |
| final_reeval median | 3.0935 | 2.8344 |
| tuned_ms(获胜 trial) | 3.1329 | 2.8616 |
| 获胜精度 | bf16 | bf16 |
| `final_reeval_ok` | true | true |
| `excessive_speedup_flag` | **false** | **false** |
| eager / eager_tf32 | 6.9453x / 5.8842x | 7.5704x / 6.4437x |
| torch_compile | 4.5016x | 4.9296x |
| **torch_compile_tf32(同精度最强基线)** | **3.537x** | **3.8732x** |

**判决**:`PASS -- treatment 2.8400 ms vs control 3.1100 ms (-8.68%)`。J2-5 只要求处理臂**不更差**,
而它**更好且超出噪声底**,所以这是一次**朝处理臂方向的分离,不是平局**。

**噪声底的来源与选择**:检查器手上有两个候选口径 —— 借来的 2.35%(`1 - 0.9765`,一个
`frac_within_tol` **正确性**数字)与本次实测的 **0.75%** 延迟底(同 kernel 复测差,n=2 个已完成 run,
最宽 0.75%)。检查器**取更宽的 2.35%**,理由写在输出里:*延迟判决不得比延迟测量本身更严*。
8.68% 是 2.35% 的 3.7 倍、0.75% 的 11.6 倍。

**这一行的措辞曾是我自己的缺陷**:第一版检查器把它印成 `(-8.68%), within the 2.35% noise floor` ——
逻辑(单边 J2-5)是对的,措辞把结论说反了。已在 `68aa1b3` 修成三个分支(更差超底=FAIL / 更好超底=
报方向 / 不足底=不可分辨),并配 1 个单测 + 3 个 revert 变体。

---

## 4. 增益从哪来:不是种子更好,是改写更有效

两臂的种子起点几乎一样,**处理臂的最好种子还略差**:

| | control | treatment |
|---|---|---|
| 四个种子的 best | **3.1862** | **3.2128**(差 0.83%) |
| 四个种子的 mean / median | 3.7128 / 3.5858 | 3.9809 / 4.0862(更差) |
| 最好种子 → 最终 tuned | 3.1862 → 3.1329 = **−1.67%** | 3.2128 → 2.8616 = **−10.93%** |

**处理臂以更差的起点(种子 mean 差 7.2%)拿到更好的终点。** 逐轮 `latency_gain_pct` 与
`conversion_verdict`:

| | control | treatment |
|---|---|---|
| round 1 | `no_conversion` +1.671% | `improved` +10.932% |
| round 2 | `flat` +1.692% | `improved` +28.079% |
| round 3 | `improved` +9.136% | `improved` +26.374% |
| 中位增益 | **+1.69%** | **+26.37%** |

**这是本项目 Loop C 首次出现 3/3 全 `improved`。** 对照臂三轮里两轮低于 2.0% 的
`min_improvement_pct` 门(其中 round 1 是 `no_conversion`:资源改善了而延迟只动 1.671% ⇒ 那些资源
**不是**这个结构的限制项 —— 这是关于"限制不在哪"的证据,本身有价值)。

**但注意**:per-family / per-round 增益**对起点极其敏感,不能作跨 run 的头条**。同一指标下 v2 首个
L3:43 run 的 per-family 平均增益是 **55.80%**(因为它的种子差得多),远高于处理臂的 21.79% —— 这不
意味着 v2 的改写更强。我曾用这个比值得出过一个 19x 的差异结论,是错的。**跨 run 只比
`final_reeval_ms`。**

### 获胜候选与假设

| | control `cand-d02b0742` | treatment `cand-2d8eaf9a` |
|---|---|---|
| 谱系 | seed `cand-52e0e567` → rewrite(深度 2) | seed `cand-9a9ab3d2` → rewrite(深度 2) |
| 假设 | H1+H2+H3(exp2 softmax + native-dtype dot + 缓存权重 cast) | H1+H3(两个 `F.linear` 搬上单个 Triton GEMM,删掉 `x.to(dt)` 与 `out.to(fp32)` 各 ~201 MB 往返) |
| 获胜配置 | bf16/plain, BM=128 BN=32 W=4 S=2 | bf16/plain, BM=32 BN=16 W=2 S=2 + GEMM 64×128×32 W=8 S=2 G=8 |
| 最优 trial 的 aten 流量 | 2430.1 MB / 10 ops | **1624.8 MB / 10 ops** |
| regs / spills / shared / occ | 255 / 6 / 49152 / 0.167 | **155 / 0 / 17408 / 0.208** |

**两臂的改写朝不同方向走**:对照臂的六个改写全部围绕**寄存器与 spill**(它的 verdict 16/16 都是
`resource_limited`,`at_limit` 里 14/16 含 `regs=255/255`,13/16 含 `occupancy=17%`);处理臂的五个
改写有三个直接打 **aten 流量**(把 cuBLAS 投影搬上 Triton、把 scratch 存成 bf16)。

这和两臂各自看到的东西一致:处理臂的 `bottleneck.md` 里唯一带**数值 room-left** 的两行正是
`candidate_aten_bytes — room left: 2.13e+09 above a floor of 4.16e+08 (84% of the current reading)`
与 `candidate_aten_ops — room left: 14 above a floor of 1 (93%)`;而对照臂把 `candidate_aten_mib`
连同 31 个别的数字平铺在一起,没有"离下界还有多远"这一层。**处理臂拿到的是更少的数、更明确的
"还有多少空间"。**

---

## 5. S2d 期望账本:方向一致但**不显著**

journalled 的 ledger 是**汇总的**(pooled,`candidate_id=None`,一轮两个候选共用族 incumbent 的一次
conversion),这是已记录的缺陷 `docs/result-s2d-pooled-ledger-defect.md`。论文数字用
`scripts/rederive_per_candidate_ledger.py` 从同一份 `REWRITE_PRODUCED` 离线重导:

| | journalled(pooled) | 重导(per-candidate) |
|---|---|---|
| control | 6/10/0, 8/6/1, 10/3/3 | **23 hit / 15 miss / 1 vacuous** |
| treatment | 7/6/3, 10/5/1, 3/4/1 | **26 hit / 9 miss / 5 vacuous** |

命中率 **74.29%(处理) vs 60.53%(对照)**,**Fisher 单边 p = 0.1585**(双边 0.2260)——
**方向对但不显著**,n 太小。论文必须这样写,不能写成"账本提高了预测准确率"。

per-candidate 明细:

| arm | candidate | hyp | 自身 best | hit/miss/vac |
|---|---|---|---|---|
| control | cand-70cbf6bc | H2 | 3.1842 | 5/3/0 |
| control | cand-d02b0742 | H1+H2+H3 | 3.1329 | 3/5/0 |
| control | cand-d0dd9c40 | H1 | 3.3398 | **7/0/0** |
| control | cand-fdbae5d5 | H3 | 3.3039 | 1/6/1 |
| control | cand-33d090e1 | H1 | 3.4627 | 7/1/0 |
| control | cand-c5f157d2 | H3 | — | **UNMEASURED**(墙钟在 parameterizer 之后截断,该候选一个 trial 都没跑) |
| treatment | cand-2d8eaf9a | H1+H3 | 2.8616 | 3/2/3 |
| treatment | cand-3760b4d7 | H2 | 3.2031 | **7/1/0** |
| treatment | cand-b417c5cd | H1 | 3.3213 | 7/1/0 |
| treatment | cand-8e50a54a | H1+H2 | 2.8759 | 6/1/1 |
| treatment | cand-136dfbfb | H1 | 3.0730 | 3/4/1 |

**两个注意点**:(a) `vacuous` 是 agent **主动拒绝预测**(处理臂的获胜候选就有 3 个 `unknown`,理由写得
很具体:"两个新 GEMM kernel 与不变的 155-reg attention kernel 谁主导聚合值不可预测"),不是判错;
(b) pooled 汇总把处理臂 H2 这个**全项目最准的一次预测(7/8)**记成了带 5 个别人挣来的 miss。

### S2d(c) 在两臂都**未被施测**

`self.ledger` 按**族**存,只有一个族的**第二轮**改写才可能收到账本。两臂都是 **3 个族 × 1 轮**,
所以 `ledger_entries` 在**每一次** rewriter 调用上都是 `[]`。**两臂在 S2d(c) 上完全相同,任何延迟
差异都不是关于它的证据。** 必须写成 **UNTESTED**,不是 tested-and-null。这条结论已编码进
`check_wrapup.py` 的 `[3c]`(`check_ledger_reach`),配 4 个 revert 变体,以免报告时靠我的阅读。

---

## 6. 与 v2 最优候选的绝对比较:**平手,v3 没有赢**

| | run | ms(median) | 同精度加速 | 预算 | 候选 | 轮 |
|---|---|---|---|---|---|---|
| v2 最优 | `run-l3-43-20260909-015247`(box 2) | **2.7628** | 3.9855x | 13.51 h | 14 | 5 |
| v3 处理臂 | `run-l3-43-20260911-052630` | 2.8344 | 3.8732x | 12.01 h | 9 | 3 |
| v3 对照臂 | `run-l3-43-20260911-053020` | 3.0935 | 3.537x | 12.47 h | 10 | 3 |

处理臂比 v2 最优**慢 2.59%**(median 口径),落在 2.35% 底之外一点、在 4.73% 的旧口径之内 ——
诚实的说法是 **平手偏 v2 一侧**,而 v3 用了**少 1.5 h 预算、少 5 个候选、少 2 轮**。
**不能宣称 v3 超过 v2。** 对照臂则明确差于 v2(−12.0%)。

---

## 7. 全部局限(写进论文时必须逐条保留)

1. **n = 3 族/臂、1 个任务、1 张卡、1 个模型**。这是一次配对观察,不是有统计效力的实验。
2. **S2d(c) 未施测**(§5),不是测了没效。
3. **账本准确率 p = 0.1585,不显著**。
4. **对照臂墙钟超支 3.8% 且被截断**;方向对处理臂不利,但两臂预算不严格相等。
5. **ledger 落盘条目是 pooled 的**;论文数字来自离线重导,重导脚本本身曾把每个族的 parent 混池
   (导致 `cand-3760b4d7` 读成 1/7 而真值是 7/1,精确反转),已按族分键修复并复现 3/2/3 与 7/1/0。
6. **`launch_bound` 分支对约 40% 候选不可达**(复用测量不写 `.py`,`docs/result-reused-trial-has-no-source-file.md`)。
   两臂比例相近(3/7 vs 3/8)⇒ **不是臂间对等缺陷**,但报告 `launch_bound` 未触发时必须限定为
   "在探针跑到的候选上"。修复 `80a4c14` 是 driver 侧,**不作用于这三个 run**。
7. **实验本身不能区分"结构有帮助"与"原始倾泻有害"**(§1)。G9 第三次 run 探过相邻方向
   (`none` 7.87 / `verdict` 4.57 / `rich` 4.76 ms):`verdict` 与 `rich` 差 4.19% 而臂内跨度 4.72%
   ⇒ 结构有用、体积无用,但 G9 的 `rich` 只加了**无屋顶的量**、n=2/臂、且绕过了 analyst。
   要分开需要**第四臂**(结构化判断 + 距上界真实数值),必须新写渲染函数且**不得绕过
   `assert_no_raw_vector`**,必须排除 `pct_of_dram_peak`/`pct_of_compute_peak`(它们是 1/latency 的
   改写),并需要 ≥4–6 族/臂 ⇒ 又一对 12 h。
8. **对照臂有一个候选零 trial**(`cand-c5f157d2`,墙钟),所以对照臂的 ledger 是 5 个候选而非 6 个。
9. 外部对照 —— KernelPro 的原始计数器臂 1.77x vs 无反馈 3.35x(单边 Wilcoxon p=0.0007)、
   KernelBench few-shot 把 o1 的 L1 fast_1 从 10% 压到 6% —— 与本结果**方向一致**(原始数字堆叠有害),
   但都不是同一条控制变量,只能作为讨论。
