# v3 文档索引

**本分支(v3)的设计文档共七份 + 一份结果文档,阅读顺序见下。** v2 的分析/调研/结果文档也在本目录(数量多,按文件名前缀查:`finding-` 实测缺陷、`result-` 实验结果、`measurement-` 量化、`plan-` 计划、`research-` 调研)。

---

## 一、v3 设计文档:从哪份读起

**如果只读一份:`v3-plain-language-design-and-gaps.md`。** 它用白话讲清"框架现在怎么跑、v3 要做的六件事、缺口分三类、整体形态",不使用 S1/S2/A-1 这类代号。

| 顺序 | 文档 | 内容 | 什么时候读它 |
|---|---|---|---|
| 1 | **`v3-plain-language-design-and-gaps.md`** | 白话版:现状流程 + 六件事 + 缺口三类 + 三层形态 | **默认入口** |
| 2 | **`v3-design-v2-resource-visibility-and-change-impact.md`** | **第二版设计(当前有效)**:用户的重新框定、四条新实测、三层架构、三个缺口怎么填 | 要看架构决策与其实测依据 |
| 3 | `v3-design-resource-ratio-and-conversion-efficiency.md` | 第一版设计:核心思想、v2 差距、新颖性评估、异构卡 | 要看**思想**与**新颖性主张**(架构部分已被第 2 份取代) |
| 4 | `v3-implementation-plan.md` | 阶段 S1 / S1b / S2 / **S2c** / S2b / S3 / S4 / S5 / S6:改什么、判据、反向对照、失败退路 | 要动手实现 |
| 5 | `v3-flow-and-gap-register.md` | 端到端流程(对 `orchestrator.py` 复验)+ 缺口登记册(设计/测量/功效/已决定不做) | 要查某个缺口的状态 |
| 6 | `v3-prior-art-and-what-to-borrow.md` | 调研:六条可借鉴项、十种资源交换、不该借的、14 项待复核 | 要引用外部工作 |
| 7 | `v3-revision-no-scalarization-and-retire-risk.md` | 两处方向性否决的完整论证(拒绝合成、撤回取值黑名单) | 想知道为什么不做某件事 |

**冲突时的优先级**:第 7 份 > 第 2 份 > 其余。第 7 份记录的是用户的方向性决定,第 2 份是当前架构定稿。

---

## 二、v3 已完成的实测结果

| 文档 | 结论 |
|---|---|
| **`result-a1-no-ranking-rule-survives-its-controls.md`** | **缺口 A-1 没有排序规则可用**:最好规则 7.1% vs 最好对照 21.4%,两个规则都输给"永远挑同一维"。**注意:文档里"赢的动作是填空隙"这个机制解释已被同日的正对照推翻** |
| **`result-dram-and-compute-pressure-are-latency-restated.md`** | **两个发现,第二个更严重**:(1) 利用率判据在 5 个已知答案的 kernel 上误判 2 个(纯 L2 流式 kernel 读出 DRAM 屋顶 **287%**);(2) **`dram` 与 `compute` 两"维度"就是 1/延迟** —— byte_count/flop_count 是任务级常数,实测 66 份判决 `gpu_ms × achieved_tbs` 恒定到 0.07–0.36%。**真正独立于延迟的只有编译期维度** |
| **`result-wall-clock-is-always-the-binding-budget.md`** | **5/5 跑完的 L3 run 都是墙钟结束**,改写预算(上限 5 轮)只用掉 1–2 轮、族数(上限 6)只用掉 3–4 个;**loop C 只拿到墙钟的 7–8%**,62% 的 agent 时间花在逐候选的参数化+分析上(按候选收费 vs 按轮收费);Loop D 零执行的原因与先前记录的不同 |

**探针脚本**(`v3/scripts/`,全部零 GPU 或空窗期运行):

| 脚本 | 回答什么 | 结论 |
|---|---|---|
| `probe_resource_read_cost.py` | 一个资源点多贵、字段什么时候可读 | **614.6 ms 中位 / 899 ms 均值 = 1/21~1/30 个计时 trial**;`n_regs` 编译后 0/108、一次 0.4 ms launch 后 96/96 |
| `probe_resource_map.py` | 能不能用公式或"测一次到处用"省掉地图 | **都不能**:shared 线性拟合 0/96 精确命中(残差最大 121.4%);地图**不可分离**(1/8 一步移动增量一致,shared 跨度 24576 B、寄存器跨度 127) |
| `multibind_rule_check.py` | 多维绑定时该动哪一维 | 没有规则打得过对照(7.1% vs 21.4%) |
| **`multibind_latency_independent.py`** | **剔除 `dram`/`compute` 两个 1/延迟的列后,多维绑定还剩多少** | **还剩很多**:n=51,0.70 下 **72.5%**(旧计法 90%)、0.85 下 **25.5%**(旧 45%);最常见组合 **occupancy+registers** 本来就与延迟无关 ⇒ **A-1 的动机站得住,且与 S2c 所测维度重合** |
| `budget_allocation.py` | 预算实际花在哪个循环上 | **墙钟 5/5 绑定**;loop C 占墙钟 7–8%,parameterizer+analyst 占 agent 时间 62% |
| `probe_binding_criterion_control.py` | **利用率判据本身可信吗**(5 个已知答案的 kernel) | **误判 2/5**;L2 内流式 kernel 读出 DRAM 屋顶 287%;**并顺带查出 dram/compute 就是 1/延迟** |
| `spill_filter_check.py` | 该不该按寄存器溢出过滤配置 | **不该**:0/5 获胜配置溢出,但 11/175 近平局溢出最多 52 个、只差 +0.6%~+2.1% |
| `blind_spots.py` | 用户点的三个缺口有多严重 | 17 份报告 1 个标签;76% 报告有 ≥2 维同时接近上限;改写后资源漂移最多 79.9 KB / 137 regs |
| `knob_resource_map.py` | 旋钮→资源的映射在磁盘上有没有 | 有,且从未被用;shared 效应跨候选一致,寄存器效应**混杂** |
| `retire_risk.py` / `value_conditional.py` | 取值黑名单会误杀多少 | N=6 时误杀 **38.5%**;失败是**条件性**的而摘除是无条件的 |

---

## 三、v2 的关键实测文档(v3 的设计依据大量引用它们)

按"被引用频率"列前几份,其余按前缀查:

- `result-l3-48-rerun-verdict.md` — L3:48 两次 run 的对比(**r2 的 1.55 ms / tf32 0-for-172 那份写在 v2 checkout 的 docs/ 下,不在本分支**)
- `finding-unreachable-correctness-gate.md` — 正确性门的固定阈值问题(**改动需用户明确决定**)
- `opop-v2-tuning-inside-the-noise-floor` 系列 / `measurement-rejections-above-the-noise-floor.md` — 噪声底
- `finding-shared-memory-constraints-never-fire.md` / `measurement-vacuous-constraints.md` — agent 手写约束中位只有真值 32%
- `result-every-tensor-core-candidate-was-rejected.md` — L3:48 上 8/8 张量核候选被拒
- `finding-optimization-behind-a-dead-mode-branch.md` — 死分支事故(31 个 trial 全测 fallback、零报错)
- `measurement-predicted-gain-overshoots.md` — `predicted_gain_pct` n=21,8/21 符号错
- `research-counter-free-profiling-capability-tiers.md` — 无 counter 时能看到什么

---

## 四、v2 文档索引(历史,保留)

v2 的 `analysis/` 与 `research/` 子目录索引:

- `analysis/framework-diagnosis-and-improvements.md` — v2 效果分析(基于 events.jsonl 磁盘核实)
- `analysis/improvement-implementation-plan.md` — 第一轮 7 项改进(A–F + H1/H2/H3 已实施)
- `analysis/improvement-plan-round2.md` — 第二轮 J/K/L
- `research/kernelfoundry-findings.md` — Intel ISL KernelFoundry(ICML 2026)源码调研

## 五、不在本目录的关键文件

- `v2/README.md` — 项目入口 / 架构概览
- `v2/HANDOFF.md` — 跨 session 交接(**以这份为准**,根目录那份更早)
- `D:/ClaudeCode/data/plans/lively-cuddling-clarke.md` — 已批准的 M0–M5 实施计划
- 工作路径根的前史调研:`gpu_kernel_repo_poc_research.md`、`gpu_kernel_structure_search_research_report.md`、`v2-kernel-optimization-flow.md`

**注意**:仓库根目录有一个 **14 GB 的 `.db` 文件**。**任何递归 grep/find 必须限定子目录**(`v2/`、`v3/`、`paper/`、`kernelfoundry/`、`KernelBench/`、`external_files/`),否则会扫到它。
