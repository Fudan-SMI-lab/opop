# C2 方法改进：核心污染与修复后探索性闭环的最终分析

更新：2026-09-18。T8 科学分析完成；最终核验回执单独交付。本次分析没有新增实验。

**首先区分两个事实：原始 H−G0 数值门槛通过 3/4 cell，但核心实验的预期源码保真核验 FAIL。** harness 的 staging 覆盖使实际验证/调优源码偏离 parameterizer 输出，故不能将这些结果作为预期方法有效性或 intent handoff 因果收益的证据。B0 的 key mismatch 已定位为 harness 源码复制冲突，**不是已证实的模型坏输出，也不是已证实的环境故障**。原始机会、失败和测量全部保留。

修复后的 A 是原预算内剩余八机会的 **EXPLORATORY（探索性）跟进**；原始 3/4 不构成未受污染的预注册晋级链。A 源码/测量独立核验 **APPROVE**：共同截点两任务数值通过，但同卡终点 **task43 的 H 慢 7.830148%，task21 的 H 快 41.559002%**。这是有限且混合的观测结果，不是普遍有效或独立 intent 收益的证明。

## 1. 计划、公开版本与实现范围

- 获批计划：[方法改进与快速实验计划](plan-c2-method-improvements-and-fast-experiments.md)。在本轮产品修改前，计划已单独归档并推送 GitHub：[3fa0789](https://github.com/Fudan-SMI-lab/opop/commit/3fa0789f8f2c2daf49e6b2c96eb1813c8745b290)。归档先于 T2–T4 产品实现，不是看到结果后补写门槛。
- 已公开的核心实现及本次核心执行版本：[1090b673](https://github.com/Fudan-SMI-lab/opop/commit/1090b673d6a7332d20e3f43dc04a9b7b0424dda1)，[固定版本源码树](https://github.com/Fudan-SMI-lab/opop/tree/1090b673d6a7332d20e3f43dc04a9b7b0424dda1)。这也是含 staging 缺陷的实验版本，不能把后续修复归到该 SHA。
- staging 修复已公开于 [d61b305](https://github.com/Fudan-SMI-lab/opop/commit/d61b30528b3105aa1012a52c015e857b0dcca267)；A 执行、ready 比较和 terminal 组合已公开于 [3ea43cc](https://github.com/Fudan-SMI-lab/opop/commit/3ea43cc1cca2454dc26a19c4697dd65c769df7ff)。两台既有服务器部署该精确 A 版本；A 于 2026-09-18 00:50:29 UTC 完成最后终测，operator 00:52:54–55 检查无相关进程、服务端口关闭、GPU 空闲。

实现按职责分类，不把文件清单当作效果证据：

| 类别 | 已实现内容及代表位置 | 证据边界 |
|---|---|---|
| 提示语义与输入接口 | `agents/method_prompts.py`、`agents/modules.py`：真实编译 wall 与 conditional responses 分开；whole-task/legacy-local 对照；可选 advisory rewrite intent | 接口与路由实现，不等于收益成立 |
| 原生续调与产物传播 | `control/accepted_candidate.py`、`control/orchestrator.py`、实验 retune/closed-loop 适配：复用 witness、native tuning、候选局部历史与实际 selected artifact | 选中产物对应真实测量，但上游 staging 保真曾失效 |
| 核心实验编排 | `c2_method_*`：固定四槽、三条生成链、P/H 共享 rewrite、机会状态、deadline、轮转 finals 与四 cell 门槛 | 当时只交付核心与分支决策，未交付已执行的 A/B/C |
| staging 修复与回归 | `c2_retune.py`；新增 staging 回归、强化真实布局 protocol fixture | 修复数据传递，不改域、B40、质量标准、种子或选择算法 |
| A 有界闭环与比较 | `c2_method_a.py`、`c2_method_a_compare.py`、`c2_method_a_terminal.py` 及既有 loop/helper 接线：原始 P1、两轮 seed、ready 截点、隔离 helper、终点转移和同卡测量 | 源码保真通过的探索性执行；原核心污染标签保留 |

既有发布回执记录核心组合测试 102 passed；原测试布局未覆盖真实的 `parameterized.py` 与 `source.py` 同目录冲突。修复后记录 106 passed、最终核心/staging **20 passed**、15 份实际归档检查 **1 passed / 3 deselected**；A 实现回执171 passed，父任务23个 focused tests passed。集合重叠，不相加为独立实验次数。本次分析另跑既有 ready-cutoff CPU 回归 **2 passed / 3 deselected（10.23s）**，并对真实结果调用 production `compare_ready`、独立 NumPy 算术复核；没有 GPU/model/SSH 或实现修改。

## 2. 核心设计与实际执行账目

核心为 task43、task21 各两次 fresh rep，均从各任务相同的原始 P1 出发；不是接续历史 winner。G0 为全任务 prompt、无 fresh responses、有自身 intent；L 为修正事实标签后的 legacy-local、有 responses、无 intent；P 为全任务、有 responses、无 intent；H 与 P 使用**同一份参数化前 rewrite**、有 intent。四 cell 的 P/H proposal 与两份 parameterizer 输入拷贝均已核验字节一致。

每 cell 的 L/P/H 共用一批 12 acquisitions；原生每个空间 B40，witness 保留在预算内。核心 sampler/evaluation/validator seed=0，expansion/S7/scanner OFF。同槽计时串行；模型调用可与无依赖的工作重叠，不扩大 proposal 数。

下表单元格为 `状态 / asked / selected`；C=complete，F=failed，I=invalid。

| Cell／物理槽 | task／rep；调优顺序 | acquisition complete/fail | G0 | L | P | H |
|---|---|---:|---|---|---|---|
| A0／A GPU0 | 43/0；G0,L,P,H | 11/1 | C/40/child | C/40/child | C/40/child | C/40/child |
| A1／A GPU1 | 21/0；L,H,G0,P | 12/0 | C/40/child | F/0/parent | C/40/child | C/40/child |
| B0／B GPU0 | 21/1；P,G0,H,L | 12/0 | I/0/parent | I/0/parent | I/0/parent | I/0/parent |
| B1／B GPU1 | 43/1；H,P,L,G0 | 11/1 | C/40/child | C/40/child | C/40/child | C/40/child |

- **48 acquisitions 已解析（resolved）=46 complete+2 fail**，不是 operator 初稿所称的 48 个成功 trial。A0/B1 第二个 endpoint 均有 shared-memory runtime error（需求 115200 bytes，硬件上限 101376）；这是已解析负证据，不是 censor。
- **16 个固定机会=11 个完整 B40+1 failed+4 invalid**；没有补抽、替换或删行。**440 asked=355 complete（有效记录）+85 fail**，包含 witness 复用，不是 440 次全新 full evaluation。失败种类为 72 infeasible_shared_memory、9 correctness_mismatch、4 runtime_error。
- 计划 12 条生成链、16 次参数化机会；实际 **12 analyst+11 rewriter+15 parameterizer=38 started，37 finished、1 failed**。A1/L analyst 两次尝试后仍无可解析 JSON，因此未进入 rewriter/parameterizer；保留 parent。不能把计划的 16 次参数化写成实际完成数。
- 所有 cell 的 parent 与四臂 incumbent 均完成各 3 个 full blocks：**60 blocks、6000 性能样本、300/300 correctness trials**。每块 100 个性能样本、5 次 correctness；全部 final 状态 complete，无 censor。native selection 在 finals 前冻结，finals 不重选赢家。

## 3. 同卡终测：全部 60 个 block median

单位 ms；下列 20 行每行三块，完整覆盖 60 块，显示至小数点后 9 位。`M` 是三块 median 的中位数，不是最快块或合并样本的 median。每个 cell 的全部比较在该 cell 同一 GPU 上完成；**不跨卡直接比较 raw latency**。A1/L 与 B0 全部臂是本次新测的 retained parent，不使用历史父成绩。

| Cell | 产物 | block 1 | block 2 | block 3 | M |
|---|---|---:|---:|---:|---:|
| A0 | parent | 3.492863894 | 3.492863894 | 3.491839886 | 3.492863894 |
| A0 | G0 | 3.117055893 | 3.120127916 | 3.122143984 | 3.120127916 |
| A0 | L | 2.988032103 | 2.987008095 | 2.990080118 | 2.988032103 |
| A0 | P | 2.680320024 | 2.682879925 | 2.676192045 | 2.680320024 |
| A0 | H | 2.935807943 | 2.934783936 | 2.935311913 | 2.935311913 |
| A1 | parent | 8.360960007 | 8.364543915 | 8.360447884 | 8.360960007 |
| A1 | G0 | 5.472256184 | 5.472256184 | 5.477344036 | 5.472256184 |
| A1 | L（保留父） | 8.360464096 | 8.361984253 | 8.358912468 | 8.360464096 |
| A1 | P | 5.648832083 | 5.645311832 | 5.637119770 | 5.645311832 |
| A1 | H | 5.209023952 | 5.208064079 | 5.212671995 | 5.209023952 |
| B0 | parent | 8.361984253 | 8.374272346 | 8.371199608 | 8.371199608 |
| B0 | G0（保留父） | 8.363519669 | 8.363007545 | 8.376319885 | 8.363519669 |
| B0 | L（保留父） | 8.369664192 | 8.367103577 | 8.373248100 | 8.369664192 |
| B0 | P（保留父） | 8.365104198 | 8.368127823 | 8.362495899 | 8.365104198 |
| B0 | H（保留父） | 8.368127823 | 8.364031792 | 8.363007545 | 8.364031792 |
| B1 | parent | 3.482624054 | 3.485696077 | 3.487232089 | 3.485696077 |
| B1 | G0 | 3.390464067 | 3.391488075 | 3.394560099 | 3.391488075 |
| B1 | L | 3.012608051 | 3.013632059 | 3.014656067 | 3.013632059 |
| B1 | P | 3.001343966 | 3.002367973 | 3.001343966 | 3.001343966 |
| B1 | H | 2.987008095 | 2.993151903 | 2.994175911 | 2.993151903 |

终测顺序为“parent+该槽臂顺序”、反序、循环移位，已按事件核验。既有独立分析将 60 份 worker 输出与 trial 对应，并由导出样本重算 median；样本保留五位小数，最大差 **0.00000425292968664337 ms**，低于半个末位精度容差 **0.000005000001 ms**，不改变门槛结果。报告使用 worker 未舍入 median；JSON 保存完整精度。

质量口径为原有 relaxed/FP64 policy，不是严格数学等价。raw worker 的 FP64 rescue 总次数为 **A0=75、A1=45、B0=0、B1=75**：A0/B1 每块 5 次；A1 G0/P/H 每块 5 次，parent/L 为 0；B0 全部为 0。normalized trial 中 rescue 字段为 null，不能据此宣称零 rescue。

## 4. 原始门槛：全四 cell，不掩盖无效机会

每个比较需同时满足 `1−M_treatment/M_baseline ≥ 2%` 和 `max(treatment blocks)<min(baseline blocks)`；主门槛要求至少 3/4 且覆盖两个任务。范围只是描述性三块范围，**不是置信区间或统计显著性**。生产 `summarize_core` 与既有独立算术核验一致。

| 比较 | A0：改善%；严格范围分离；结果 | A1 | B0 | B1 | 通过数 |
|---|---|---|---|---|---:|
| H−G0 | +5.923347%；是；PASS | +4.810305%；是；PASS | invalid H/G0；否；FAIL | +11.745174%；是；PASS | **3/4** |
| P−L | +10.298152%；是；PASS | +32.476095%；是；PASS* | invalid P/L；否；FAIL | +0.407750%；是；FAIL（<2%） | **2/4** |
| H−P | −9.513487%；否；FAIL | +7.728322%；是；PASS | invalid H/P；否；FAIL | +0.272946%；是；FAIL（<2%） | **1/4** |

*A1 的 P−L 对照是失败 L 的新测 retained parent，不是成功 L rewrite。L 的已解析失败本身不取消主门槛；所有 finals 已解析，故原始计算并非 inconclusive。若核心或测量有 censor，按规则全局 inconclusive、不能触发 A。

B0 五个产物都在约 8.37 ms，是同一 retained parent 的重复噪声，不是五个有效新候选；invalid H 仍计入分母并判 FAIL，改善值记 N/A/null，不删 cell。**A0 H 相对 P 的改善为 −9.5%（即更慢），H−P 仅 1/4，不支持“一贯的 intent 收益”。** 即便其他原始数值通过，也不能越过下节源码保真失败作方法效力或因果推断。

## 5. 关键 staging 缺陷、影响范围与已验证修复

### 原因与不能据此推断的事

旧路径先把 `candidate/parameterized.py` 写到 `retune/inputs/source.py`，随后把同目录 `candidate/source.py` 当 helper 复制到同一目标，**覆盖主源码**。validator 随后从被覆盖的源码提取 PARAMS，却与 parameterizer 的 domain 对照。

- 15/15 实际 staging 源码都等于该 helper；其中 **13/15 与预期 parameterized 文件字节不同，10/15 在移除 PARAMS 后仍有 AST body 差异**，不是仅格式或默认值区别。
- body 差异涉及 A0 L/P、A1 H、B0 四臂、B1 G0/L/P，不能把污染局限于 B0。
- B0 G0 的预期文件确含 `STATS_STAGES`，L/P/H 确含 `DW_STAGES`；四份预期文件的 PARAMS key set 与 domain 匹配。被覆盖源码缺这些 key，导致记录中的 `key_mismatch`、asked=0。因而不能归咎“模型没生成 key”。
- 实际 selected source/params 与终测材料彼此一致，证明测量属于**实际运行产物**；但它不证明运行的是预期参数化产物。B0 预期源码是否能编译、正确或更快没有 GPU 证据，不能因修复可复制就追认成功。

因此核心账目/原始算术 PASS，**预期源码保真 FAIL、核心受污染**。16 个机会、失败状态及 raw 文件均不回写、不补抽。类似旧调用方可能经过同一 retune 路径，但本任务没有独立审计所有历史运行，**不据此全面重判旧结果或改写历史报告**。

### 修复实现事实（不是旧实验补跑）

共享 `c2_retune.py` staging 现在按下表复制字节；不是事后修改 proposal，也不删除可能真正需要的 import helper。

| 角色 | 修复后目标 |
|---|---|
| primary source | `<run>/inputs/<原始文件名>`，如 `inputs/parameterized.py` |
| helper | `<run>/inputs/<相对 primary.parent 的路径>`，保留 `source.py`、`reference.py` 和嵌套包 |
| 实际任务 reference | `<run>/reference.py`，位于 helper 命名空间之外 |

`tune_existing` 读取保留原名的 primary；manifest 增加 `staged_source`/`staged_reference`；worker/final helper import root 仍为 `inputs/`，selected measured artifact 拷贝方式不变。不同文件撞到 primary 时在 GPU 工作前拒绝；显式重复的 primary helper 跳过。

**15 份真实归档输入全部完成 CPU-only 验证**：只向临时目录复制，停止于 validator 入口；新 staged primary 与预期原文件字节相等，入口 PARAMS 与未改 domain 相符。真实布局测试亦覆盖 reserved-name helper、嵌套 import、CRLF 及 selected artifact 对应，最终相关回归20 passed。修复后核验 **3253 个导出 hash 条目、3235 条唯一 raw 路径未变**。这证明复制路径修复，不证明旧预期候选的 GPU 有效性或性能；修复已随 d61b305 发布，A 的新执行另行核验，不追认旧核心。

## 6. 核心机制案例：proposal 声称、参数化修改与选择不能混为一谈

以下仅摘录已有 proposal 的机制描述，**不是已测因果归因**；proposal 中自述的 profiler/历史 timing 也不充当本报告新结果。

| Cell | G0 的声称机制 | P/H 共享 rewrite 的声称机制 | 实际原生选择的部分差异 |
|---|---|---|---|
| A0 task43 | attention 输出直接以 compute dtype 物化，减少独立 cast | projection GEMM 与物化边界重写 | P：G_BK/G_WARPS/G_STAGES=16/4/4；H：64/16/3；attention stages=2/1 |
| A1 task21 | GEMM tiling/grid 与存储布局 | 缩小中间 buffer 存储精度 | P COMPUTE_DTYPE=tf32，H=fp16；两者 STORE_DTYPE=fp16 |
| B0 task21 | expand GEMM 输入复用及存储布局 | INTER_DTYPE 中间物化 | staging-induced invalid，无成功新候选选择 |
| B1 task43 | 减少 head dimension 的无效 MMA lanes | 输出物化与 head-dimension 布局 | P BLOCK_M/NUM_STAGES=128/2；H=64/4 |

需区分四层：rewriter 给出的结构输入；parameterizer 对 **body** 的改写；parameterizer 声明的 **参数域/choices/defaults**；native TPE 在该空间内实际选中的 **配置与 measured artifact**。相同 P/H 初始 rewrite 只控制第一层，不保证后三层相同。本次 staging 丢失部分 parameterizer body，进一步破坏预期处理对比。H−P 同时涉及参数化随机性、域与选点变化，不能把一项 latency 差归给 intent 本身。

## 7. 核心时间与成本：观测窗口不是 GPU 工时

| 项目 | 实际证据 | 解释 |
|---|---|---|
| 核心共享窗口 | 2026-09-17 17:28:09Z 至 20:04:31Z，**9382s=2h36m22s** | 四槽并行 launch-to-finish；落在预估 2–3.5h 内，不等于全部方法成功 |
| 调度上限 / drain | 核心上限 4h；各槽记录 drain=0 | 无 deadline censor；不是通用无等待保证 |
| 各槽程序 elapsed | A0 8898.595729s；A1 7846.971362s；B0 2545.295169s；B1 9380.735314s | 保留原独立时长，不相加冒充共享窗口或 GPU busy |
| 实现、staging、修复、分析 | 与核心窗口分开；现有已读证据不足以给出完整人工工时 | 4–8h 实现、0.5–1h 分析是原预测，不是实测；测试运行秒数不等于工程工时 |
| Provider / worker 成本 | 完整 provider cost/tokens、完整 worker wall 均 unknown | 有部分事件字段，不可补零或补造全量重试次数 |

共享实验复用每 cell 的 acquisition 与 P/H rewrite，且模型/调优可重叠；**不能把共享窗口按臂平分，或把 P/H 共用 rewrite 算作独立部署 H 的零生成成本**。当前证据不足以给出可靠的 G0/L/P/H 独立运行总时长或账单估计，暂记 unknown；不由 elapsed 推导 GPU busy，不主张严格四卡加速比。导出、传输、分析及修复发生在核心窗口之外，不回填 ready time。

## 8. A：八个剩余机会的探索性执行与独立核验

修复后从两任务原始 P1 重新生成 A 自身轨迹，**不是重画核心、挑核心最佳 H 作父或有效核心 gate 的干净晋级**。固定槽 A0=task43/G0、A1=task21/H、B0=task21/G0、B1=task43/H；各两轮，sampler1/2、evaluation/validator0，expansion/S7/scanner OFF。没有第三轮、替代 proposal、B/C 或延长到12h。

独立核验结论 **APPROVE（A 源码、续接、测量与算术）**：8/8 intended primary 与 staged 原名文件完全相等，8/8 reference 相等；7 个 selected.py 与实测 trial 字节及 PARAMS 一致；4 条第二轮 Shared/source/helper 续接与第一轮选中父一致，四个起点等于原 P1。通过实际 `TerminalArm.verify_export` 校验并核对转移/终测拷贝。旧核心3253条、新A2347条 manifest SHA-256 全部匹配；原始文件未改。

| 槽／轮 | task/arm | 状态 / asked / selected | elapsed_ready_s | own-initial 归一化 R |
|---|---|---|---:|---:|
| A0/1 | 43/G0 | valid/40/child | 2640.644254 | 0.893686690 |
| A0/2 | 43/G0 | valid/40/child | 6396.777382 | 0.809016556 |
| A1/1 | 21/H | valid/40/child | 3079.048669 | 0.537733243 |
| A1/2 | 21/H | valid/40/child | 6386.642037 | 0.423995932 |
| B0/1 | 21/G0 | valid/40/child | 2982.164936 | 0.739676156 |
| B0/2 | 21/G0 | failed/0/parent | 5179.744485 | 0.739737422 |
| B1/1 | 43/H | valid/40/child | 2654.781335 | 0.858129979 |
| B1/2 | 43/H | valid/40/child | 4823.528896 | 0.873860629 |

**8 个机会=7 个完整 B40+1 个 default witness 拒绝；280 asked=223 complete+57 fail**（49 infeasible_shared_memory、7 correctness_mismatch、1 runtime_error）。generation 事件确认8 analyst+8 rewriter+8 parameterizer，共24 started/24 finished，无额外 terminal 模型调用。H48 endpoints resolved=44 complete+4 fail（A1两轮12/0、11/1；B1两轮11/1、10/2），G0不采。

B0第二轮 default witness 在 trial2 correctness mismatch：ieee cosine0.98833693低于0.99985，FP64相对 RMSE 比217.152；asked=0，保留第一轮父，并记录一个 failed-hypothesis 条目。**不是 staging 缺陷，不是 censor，也不是成功 child**。没有第三轮，因此仅证实失败反馈被正确记录，不能声称它改善了后续性能。该 default witness 不计入280 asked。该轮无3个 child finals，合法缺失，不补造。

### A 全部57个 full-block median（ms）

轨迹45块=12+12+9+12；terminal12块，总57块、5700性能样本、285/285 correctness。下表19个非空数据行每行3块，B0/r2 child单列拒绝。每块仍100样本/5 correctness；所有实际 full blocks 有效。样本重算最大 median 误差0.000004107055664004378ms，在五位小数导出容差0.000005000001ms内；worker与记录数组/median一一对应，重算不改变门槛。Raw rescue：A0=60、A1=45、B0=0、B1=60、terminal43=30、terminal21=15；非严格数学等价。

| 槽/轮 | 产物 | block 1 | block 2 | block 3 |
|---|---|---:|---:|---:|
| A0/1 | parent（初始基线） | 3.487696052 | 3.490816116 | 3.492863894 |
| A0/1 | child | 3.117055893 | 3.120127916 | 3.119695902 |
| A0/2 | parent | 3.117055893 | 3.118655920 | 3.117055893 |
| A0/2 | child | 2.820096016 | 2.824128032 | 2.824192047 |
| A1/1 | parent（初始基线） | 8.367615700 | 8.363552094 | 8.362495899 |
| A1/1 | child | 4.497407913 | 4.495359898 | 4.497359991 |
| A1/2 | parent | 4.495872021 | 4.497407913 | 4.497407913 |
| A1/2 | child | 3.542016029 | 3.550208092 | 3.546112061 |
| B0/1 | parent（初始基线） | 8.355839729 | 8.365024090 | 8.358944416 |
| B0/1 | child | 6.189568043 | 6.176255941 | 6.182911873 |
| B0/2 | parent（保留） | 6.181375980 | 6.183423996 | 6.197247982 |
| B0/2 | child（default witness拒绝） | — | — | — |
| B1/1 | parent（初始基线） | 3.481600046 | 3.482624054 | 3.485696077 |
| B1/1 | child | 2.988544106 | 2.985984087 | 2.989056110 |
| B1/2 | parent | 2.984960079 | 2.987008095 | 2.988064051 |
| B1/2 | child | 3.043328047 | 3.042304039 | 3.044352055 |
| terminal43/A0 | G0 | 2.823168039 | 2.826143980 | 2.825183988 |
| terminal43/A0 | H | 3.046400070 | 3.041280031 | 3.057663918 |
| terminal21/B0 | G0 | 6.164992094 | 6.180351973 | 6.177279949 |
| terminal21/B0 | H | 3.610064030 | 3.606528044 | 3.618815899 |

## 9. Ready 与 terminal：不同问题，不相互替换

共同截点 `H_time=min(total_elapsed_G0,total_elapsed_H)`。取截点前最新完整状态，`R=median(该状态三块)/median(该臂自己GPU初始父三块)`，保守范围为 `[min(child)/max(initial),max(child)/min(initial)]`。两任务都须相对R改善≥2%且H上界严格低于G0下界；是描述性投资门槛，不是置信区间。实际 production `compare_ready`、独立 NumPy 算术与 raw terminal 中的 ready 记录完全一致。

| task | 截点s | 采用G0/H状态 | R_G0 [范围] | R_H [范围] | H改善 | 判定 |
|---|---:|---|---|---|---:|---|
| 43 | 4823.52977315709 | r1 / r2 | 0.893686690 [0.892406915,0.894610043] | 0.873860629 [0.872796702,0.874411769] | +2.218458% | PASS |
| 21 | 5179.7466883175075 | r2保留父 / r1 | 0.739737422 [0.738954953,0.741666689] | 0.537733243 [0.537233073,0.537806890] | +27.307552% | PASS |

task43 G0/r2在6396.777382s才ready；task21 H/r2在6386.642037s才ready，均晚于各自共同截点，**不得拿后来更好的终点回填**。B0/r2虽然机会失败，父状态完整resolved，按规则仍可ready；不把它伪装成新成功候选。

终点转移冻结产物，task43两臂到A0、task21两臂到B0，按G0/H、H/G0、G0/H测三块；使用比较卡本次初始父为共同归一化分母，不跨卡直接比raw。终测没有新study或模型调用。

| task/比较卡 | M_G0 ms | M_H ms | terminal R_G0 / R_H | H改善 | 严格有利范围分离 | 终点判定 |
|---|---:|---:|---|---:|---|---|
| 43/A0 | 2.825183988 | 3.046400070 | 0.809319051 / 0.872689929 | **−7.830148%** | 否（H全范围反而更慢） | FAIL |
| 21/B0 | 6.177279949 | 3.610064030 | 0.739002396 / 0.431880373 | **+41.559002%** | 是 | PASS |

task43“ready通过而终点失败”不是矛盾：截点比较G0/r1与H/r2，终点比较两臂最终r2。G0第二轮更晚才提供2.82ms产物，H的更早完成只影响截点可用性，不保证终点最好。task21 H的r2进一步改善，但它只进入terminal，ready结论仍使用r1。

**另一层限制是 native quick selection 与 fresh full performance 不同。** B1/H第二轮声明的父baseline来自第一轮native20样本成绩3.043776035ms；child native20样本3.026960015ms，低约0.552472%，符合实现中的`child_ms < parent_baseline`晋级规则（不是外部门槛2%）。但第二轮fresh full父median2.987008095ms、child3.043328047ms；第一轮child full亦约2.988544106ms。finals前选择冻结，finals不rerank，所以保留了全量测量更慢的r2 child。已确认选中的是正确实测artifact；这是有限样本/选择目标与独立full测量间的限制，**不新增“代码挑错产物”的结论、不事后换回r1美化终点**。

## 10. A 的实际方法动作与可支持的经验

以下从实际 selected source 的函数与PARAMS确认；proposal的流量估算、内部profiler/timing和因果解释仅为生成者声称，不当作独立测量。源码位于各`A/<slot>/trajectory/round-N/opportunity/retune/report/selected.py`；B0/r2没有selected child，使用其保留父与生成源码区分讨论。

| 任务/臂/轮 | 源码中实际动作 | 选中区域与代价 | 可支持的解释边界 |
|---|---|---|---|
| 21 H/A1 r1 | `_gemm_fullk_atomic`、`_dw_atomic`输出raw1/raw2为fp16，fp32累计stats，消费者先转回fp32；single-K projection融合stats | STORE_DTYPE=fp16，M/N/K=128/64/16，SPLIT_M=32，SPLIT_K=1 | 缩小物化宽度但引入中间舍入；raw3此轮仍fp32，TRAIN batch-stat依赖仍在 |
| 21 H/A1 r2 | expansion的A load置于`SUB_N`内循环外，显式跨N子块复用；projection/DW重排grid；single-K时raw3也存fp16 | SUB_N=4，M/N/K=128/32/16，GEMM_STAGES=1；实际SPLIT_M=32，不是proposal建议的≥64 | 复用是真实代码动作；grid顺序不保证cache命中，更大复用与grid规模/流水隐藏有权衡；最终BN仍单独pass |
| 21 G0/B0 r1 | `_proj_fused`无条件替代split-K projection，融合输出统计，去掉该段累加/独立统计路径 | M/N/K=64/64/64；中间存储仍fp32；移除SPLIT_K调参项 | 更窄的结构简化，不等于完全移除物化或证明资源瓶颈 |
| 21 G0/B0 r2 | 拟fp16 raw1/raw2、N-fastest GEMM与channel-batched DW；default正确性拒绝，**实际保留r1父** | 尝试BUF_DTYPE=fp16、DW_CH=1、DW_STAGES=3；无新selected region | 参数化body还改变projection mask；不可单独归咎fp16存储 |
| 43 G0/A0 r1 | attention y直接用compute dtype存储，使后续同dtype cast冗余 | bf16；flash M/N=128/32，4 warps/2 stages | 移动物化/舍入边界；两次F.linear及其他casts仍在 |
| 43 G0/A0 r2 | `_cached_cast`缓存3个参数cast；输入projection仍cuBLAS；输出改`_proj_gemm`，fp32 bias、直接写任务输出dtype | GEMM M/N/K=128/128/32，8 warps/3 stages | 消除输出bias cast而非“缓存4个”；cache依赖指针/version/dtype和稳态复用，输入activation cast仍在 |
| 43 H/B1 r1 | 同样降低y物化宽度，另将head拆64+32，QK部分dot相加、PV分开累加 | flash64/32，4 warps/4 stages | 减少padding同时增加分段运算/改变归约结构，不能由lane数或寄存器直接推出收益 |
| 43 H/B1 r2 | `_gemm_bias`替代两次F.linear，native输入在tile内转型，fp32 bias；保留split-head attention | 两个不同shape共用GEMM256/64/32、group8、4 warps/3 stages；flash64/16、2 warps/3 stages | 独立cast减少但转换未消失，自定义GEMM与vendor GEMM有吞吐权衡；实际终点未赢G0 |

**Fresh evidence 的覆盖也是有界的。** 例如A1/r2 response记录BLOCK_M两端7.284224/4.605952ms，BLOCK_N一端invalid，SPLIT_K两端4.606976/7.030256ms；超过12-endpoint预算的其他轴明确`probe_budget/unknown`。B1/r2的COMPUTE_DTYPE与BLOCK_N含invalid endpoint。这些是当轮父的观测，不是所有域、所有资源因果关系或全局结构最优性的证据；B40也未证明穷尽空间。

**参数化不只是填choices。** B0/r2 `generation/sandboxes/parameterizer-965bfda9/candidate/`的`source.py`到`parameterized.py`，`_proj_fused`权重load mask从`mask_n[None,:] & mask_k[:,None]`改为`mask_n[None,:] & mask_k[None,:]`，并增加DW_STAGES接线。该可执行mask轴变化与尾块正确性有关，是实际body混杂；未做纠正ablation，不能断言它是拒绝的唯一原因，更不能说“fp16中间存储已被证明不可行”。新staging正确保存输出，与输出本身质量合格是两回事。

整体经验：whole-task prompt、intent、真实response与candidate-local反馈已接入并运行；task21 H确走出了不同的显式复用/存储路径，task43 H更激进地替代两次projection却未赢终点。**A比较的是整个G0/H生成与续调package及随机轨迹，不能隔离某个接口或意图的效应**；核心P/L、H/P消融因污染不能补上这项因果识别。B0最后的失败反馈没有后续轮消费，因此不宣称“反馈带来收益”。

## 11. 全程序成本与总账

| 阶段 | 实际共享执行窗口 | 预算/口径 |
|---|---|---|
| 核心 | Sep17 17:28:09–20:04:31 UTC，2h36m22s | 预测2–3.5h、调度cap4h；源码污染仍须保留 |
| 两窗之间 | Sep17 20:04:31–22:59:12，**2h54m41s** | 独立于实验窗口，含分析、修复、A实现、发布部署等；不能拆造人工工时 |
| A含最后terminal | Sep17 22:59:12–Sep18 00:50:29 UTC，**1h51m17s** | cap2.5h；最后terminal仍在原deadline内，未重置 |
| 两个不连续实验窗口之和 | **4h27m39s（16059s）** | 不是连续墙钟、累计GPU busy或账单；无12h延长 |

A各槽loop总时长A0=6396.778509s、A1=6386.643379s、B0=5179.746688s、B1=4823.529773s。可由已有事件归集的部分阶段wall如下（秒，按两轮求和；retune含validation/compile/tuning/child finals）：

| 槽/臂 | generation agent wall | acquisition worker wall | parent full worker wall | native retune wall |
|---|---:|---:|---:|---:|
| A0/G0 | 2396.118 | 0（确实不采） | 193.831 | 3797.614 |
| A1/H | 2130.583 | 547.406 | 172.794 | 3526.456 |
| B0/G0 | 3742.430 | 0（确实不采） | 161.043 | 1264.460 |
| B1/H | 1423.026 | 493.652 | 124.121 | 2771.049 |

G0/task43的r2 agent wall1633.669s、retune1996.029s使其较晚ready；H/task43的r2相应689.603s、1238.228s。这解释观测关键路径的组成，不证明某种prompt必然省时。两项terminal transfer/wait为task43=137.147730s、task21=148.623075s；终测142.029922s、112.817124s，单列保留，不回填ready。

阶段计数器不覆盖所有等待/调度开销，不把它们冒充完整成本分解；partial provider计费/token字段存在但总账仍unknown，不能用缺失作零账单。四卡并行与模型流水化压缩了观测共享窗口，但没有串行counterfactual或隔离等待实验，**不宣称4×加速**。两臂不同模型时长、无效机会和共同截点选择都属于该次实际运行，不是通用独立部署估计。

**合计：24机会、18完整B40（核心11+A7），720 native记录=578 complete+142 fail；117 full blocks=11700样本/585 correctness；96 acquisitions resolved=90 complete+6 fail。** 含核心污染、A拒绝及witness复用，不能写成24个成功study或720个新full evaluation。全程记录drain=0，B/C未运行。

## 12. 最终分层结论与交付边界

1. **软件接线完成。** 提示输入分离、advisory intent、native continuation、实际artifact/helper传播、失败反馈、deadline、ready/terminal比较均已实现；新A8份源码/reference与实际7个native选中trial可追溯。
2. **核心不是干净方法证据。** 13/15字节差异、10/15 body差异影响多个臂；保留原raw3/4与P−L2/4、H−P1/4只为审计，不能据此归因或清洗晋级链。
3. **修复后观察到混合结果。** 两任务ready数值通过；task21 H同卡终点改善41.559002%，task43 H终点反而慢7.830148%。没有统一终点赢家；native quick选择也不保证fresh full改善。
4. **没有isolated intent、prompt或反馈收益结论。** 两任务各一条新G0/H轨迹、每臂两轮，生成随机性、body/domain/配置及部署卡分配均有限制，不能证明泛化、全局结构最优或统计显著性。expanded/conditional路径全程OFF，未在本运行得到实证验证。
5. **停止本预算内实验，不自动追加投入。** 不重画核心、不补无效候选、不追加B/C；T8分析完成，最终核验单独交付，未授权或建议自动延长实验。

本报告独立列出核心60+A57块与全部native选中配置，不依赖读者访问raw才能理解主要结果。以下链接是工作区内证据，**不是公开数据集**；`results/`受gitignore保护，`.omo`回执也是本地材料。远端根`/root/autodl-tmp/c2-method-improvements`经export manifests映射到本地`D:/Pyhon_projects/opop/v5/results/c2-method-improvements`。

- [核心分析JSON](../results/c2-method-improvements/core-analysis.json)、[A分析JSON](../results/c2-method-improvements/A-analysis.json)：完整精度、计数、门槛与轨迹。
- [核心独立核验](../.omo/evidence/v5-c2-method-improvements/T7-core-gate.md)、[A独立核验](../.omo/evidence/v5-c2-method-improvements/T7-gates-branch.md)：哈希、源码、测量、算术与限制。
- [修复回执](../.omo/evidence/v5-c2-method-improvements/T6-staging-fix.md)、[核心operator](../.omo/evidence/v5-c2-method-improvements/T6-core.md)、[A operator](../.omo/evidence/v5-c2-method-improvements/T7-A-execution.md)：执行与归档；核心旧回执的错误由后续证据明确纠正，不覆盖原文。
- [核心发布](../.omo/evidence/v5-c2-method-improvements/T5-publish.md)、[修复/A发布](../.omo/evidence/v5-c2-method-improvements/T7-A-publish.md)：GitHub发布证据，不代表raw已公开。

| 工作项 | 状态 | 限度/后续 |
|---|---|---|
| T2–T5核心实现 | 完成，1090b673已公开 | 实现完成不等于效力验证 |
| T6核心执行 | 完成归档，**源码保真FAIL** | 原raw不变，不重判全部历史运行 |
| staging修复 | 完成，d61b305已公开 | CPU15份归档证明复制正确，不补测旧候选 |
| T7 A | 执行完成，3ea43cc；独立源码/测量APPROVE | 探索性；ready2/2，terminal1/2 |
| T7 B/C | 未运行 | 不追加 |
| T8科学分析 | 科学分析完成 | 本文件随报告提交归档；最终核验单独交付 |

## 附录：11+7个native选中空间、trial与完整配置

以下是**实际测量产物**，非全局最优声明。核心根为`core/<slot>/<arm>/retune/`；A根为`A/<slot>/trajectory/round-N/opportunity/retune/`。每行`report/selected.py`对应`candidates/<candidate>/trials/<trial>.py`；空间完整choices/约束在该run的`SPACE_PUBLISHED`事件，selected配置如下。核心保真问题不因该表而消失。五个核心保留父机会及A B0/r2拒绝见前表；后者继续B0/r1配置，没有第八个新selected trial。

| 标识 | candidate / space / trial | PARAMS |
|---|---|---|
| core/A0/G0 | cand-4337516d / sp-812d4674 / tr-0c61c07a | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| core/A0/L | cand-39f33d74 / sp-8699907f / tr-b1c82258 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":3}` |
| core/A0/P | cand-572e9ff3 / sp-706b2423 / tr-65db6e0f | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2,"G_BM":128,"G_BN":128,"G_BK":16,"G_GROUP_M":4,"G_WARPS":4,"G_STAGES":4}` |
| core/A0/H | cand-404fc9e3 / sp-c425ffd8 / tr-7c0c96a3 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":1,"G_BM":128,"G_BN":128,"G_BK":64,"G_GROUP_M":4,"G_WARPS":16,"G_STAGES":3}` |
| core/A1/G0 | cand-ec49067b / sp-029a026d / tr-7369a8eb | `{"EX_BLOCK_M":32,"EX_BLOCK_N":32,"EX_BLOCK_K":16,"EX_SPLIT_M":32,"EX_WARPS":1,"EX_STAGES":1,"PROJ_BLOCK_M":128,"PROJ_BLOCK_N":64,"PROJ_BLOCK_K":32,"PROJ_SPLIT_K":1,"PROJ_WARPS":2,"PROJ_STAGES":4,"STATS_BLOCK_M":32,"STATS_BLOCK_C":16,"STATS_WARPS":1,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"DW_WARPS":8,"EW_BLOCK":128,"EW_WARPS":2,"FIN_BLOCK_C":8,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","STORE1_DTYPE":"fp16","STORE2_DTYPE":"fp16"}` |
| core/A1/P | cand-3391fc40 / sp-4014f594 / tr-29842df0 | `{"BLOCK_M":32,"BLOCK_N":32,"BLOCK_K":64,"SPLIT_M":16,"SPLIT_K":1,"STATS_BLOCK_M":128,"STATS_BLOCK_C":16,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":512,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"tf32","STORE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":2,"GEMM_STAGES":2,"STATS_WARPS":4,"DW_WARPS":8,"EW_WARPS":8}` |
| core/A1/H | cand-87a35077 / sp-ec574bdf / tr-a4de43e2 | `{"BLOCK_M":64,"BLOCK_N":64,"BLOCK_K":32,"SPLIT_M":16,"SPLIT_K":2,"STATS_BLOCK_M":128,"STATS_BLOCK_C":16,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":256,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","STORE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":2,"STATS_WARPS":4,"DW_WARPS":8,"EW_WARPS":8}` |
| core/B1/G0 | cand-9335b6bb / sp-3c7ee843 / tr-2cb5b600 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":64,"NUM_WARPS":4,"NUM_STAGES":2}` |
| core/B1/L | cand-0dd2ded1 / sp-45cabf63 / tr-51052472 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| core/B1/P | cand-c38d8500 / sp-3dddb179 / tr-937bfc96 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"BLOCK_D":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| core/B1/H | cand-9ee75ee0 / sp-2d478a33 / tr-34518aa7 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":32,"BLOCK_D":32,"NUM_WARPS":4,"NUM_STAGES":4}` |
| A/A0/1 | cand-6614c588 / sp-1624fba2 / tr-01d3f836 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2}` |
| A/A0/2 | cand-57350b8f / sp-57ebacb2 / tr-1ea4cccc | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":128,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":2,"GEMM_BM":128,"GEMM_BN":128,"GEMM_BK":32,"GEMM_WARPS":8,"GEMM_STAGES":3}` |
| A/A1/1 | cand-0930aa24 / sp-ce86a26a / tr-c0dbf642 | `{"BLOCK_M":128,"BLOCK_N":64,"BLOCK_K":16,"SPLIT_M":32,"SPLIT_K":1,"STATS_BLOCK_M":64,"STATS_BLOCK_C":32,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":256,"FIN_BLOCK_C":32,"COMPUTE_DTYPE":"fp16","STORE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":8,"GEMM_STAGES":2,"STATS_WARPS":2,"DW_WARPS":8,"EW_WARPS":4}` |
| A/A1/2 | cand-fe33920c / sp-f4fe5efe / tr-0475bef0 | `{"BLOCK_M":128,"BLOCK_N":32,"BLOCK_K":16,"SPLIT_M":32,"SPLIT_K":1,"SUB_N":4,"STATS_BLOCK_M":64,"STATS_BLOCK_C":16,"DW_BLOCK_H":8,"DW_BLOCK_W":64,"EW_BLOCK":512,"FIN_BLOCK_C":64,"COMPUTE_DTYPE":"fp16","STORE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":1,"STATS_WARPS":2,"DW_WARPS":8,"EW_WARPS":4}` |
| A/B0/1 | cand-242aaad3 / sp-acd00341 / tr-f854c006 | `{"BLOCK_M":64,"BLOCK_N":64,"BLOCK_K":64,"SPLIT_M":16,"DW_BLOCK_H":32,"DW_BLOCK_W":64,"EW_BLOCK":1024,"FIN_BLOCK_C":16,"COMPUTE_DTYPE":"fp16","DOT_MODE":"plain","GEMM_WARPS":4,"GEMM_STAGES":3,"DW_WARPS":8,"EW_WARPS":8}` |
| A/B1/1 | cand-99051b34 / sp-4ab19156 / tr-8794e0f1 | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":32,"NUM_WARPS":4,"NUM_STAGES":4}` |
| A/B1/2 | cand-fdbb4b2a / sp-5ea65edb / tr-69ccf3ac | `{"COMPUTE_DTYPE":"bf16","DOT_MODE":"plain","BLOCK_M":64,"BLOCK_N":16,"NUM_WARPS":2,"NUM_STAGES":3,"GEMM_BLOCK_M":256,"GEMM_BLOCK_N":64,"GEMM_BLOCK_K":32,"GEMM_GROUP_M":8,"GEMM_NUM_WARPS":4,"GEMM_NUM_STAGES":3}` |
