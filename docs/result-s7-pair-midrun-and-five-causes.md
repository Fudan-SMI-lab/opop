# S7(斜率引导采样)对照跑:中途状态与五条已定量的零投递成因

**日期** 2026-09-13 · **box4,同机双卡 RTX 4090** · 处理臂 GPU0 `s7-treatment`、控制臂 GPU1 `s7-control`,
两臂同一 run id `run-l3-43-20260913-202332`,均自 `39f01ea` 启动 · 任务 L3:43

> **这是中途快照,不是结果。** 两臂均未写 `RUN_FINISHED`,任何数字在收尾前都不得引用。
> 收尾须重跑三件工具(命令见文末)。
>
> **哪些数字会漂移,哪些不会** —— 本文已因手抄漂移量而返工三次,故明确区分:
> * **会漂移(引用必须带时点,本文统一为 3.50 h 处)**:trial 数、空间数、重算次数、速率比、
>   尾部占比、某配置类占 GPU 时间的百分比(分子分母同时在长,方向不定)、总搜索量差。
> * **不漂移(结构性事实,可直接引用)**:配置只差一个键;每空间预算 40;控制臂**任何**空间都未声明
>   `NUM_WARPS=1` ⇒ 该类恒 0;两臂**无共享候选** ⇒ 种子不配对;同机共用一份 `calibration.json`;
>   `walls_worthless` **结构上不含**归因失败;`RESOURCE_SOFT_WALL` **没有** `space_id`。
> * 结论一律建立在第二类之上;第一类只用于说明"当前进度",不作论据。

## 0. 本文能与不能回答什么

**能**:S7 在真实语料上的**投递率**,以及零投递的**逐门归因**。
**不能**:"S7 能否找到更好的配置"。**16 次重算 0 投递**(3.50 h 处;该计数随墙钟增长,
本文内一律引用此时点)⇒ 采样上两臂无差别,
latency 差只反映两次独立搜索的方差。读数器 `analyze_s7_pair.py` 会在 `not p1` 分支
**拒绝作结论**并打印下表的成因 —— 这是**有诊断的 null**,不是空白的 null。

## 1. 有效性:五项全部关闭

| 项 | 结论 | 依据 |
|---|---|---|
| 自变量唯一 | ✅ **119 键,恰好 4 个不同** —— `v3.slope_guide.enabled` 加三条强制隔离路径 | `audit_arm_comparability.py` → `COMPARABLE` |
| **每空间**预算相同 | ✅ 两臂均 **40 trial/空间**(按**已关闭**空间作答) | `check_arm_search_parity.py` |
| **总**搜索量相同 | ⚠️ **否 —— 235 vs 160 trial(6 vs 4 空间),差 1.9 个空间**(3.50 h 处) | 同上,新增的第三问 |
| 速率差 | ✅ 判为 **TAIL**,不随墙钟累积(实测正在**收窄**:1.67x → 1.46x);归因到**候选**而非开关 | 中位间隔 32.9 vs 34.7 s(**1.06x**,在容差内) |
| 分母 | ✅ **同机双卡、共用一份 `calibration.json`**(schema 5)⇒ 字面相同 | 跨机才需 `compare_calibrations.py` |
| **两臂不同卡** | ✅ **处理臂只在物理 GPU0(`CVD=0`)、控制臂只在 GPU1(`CVD=1`),无重叠** | `gpu_pinning_check.py`,连续采样 120 s |
| 实现无缺陷 | ✅ 出厂 `SlopeGuide` 重放与运行日志 **8/8 逐字段一致**(含每个跳过计数器) | `s7_replay_vs_log.py`;负对照会失败(错 `use_soft_wall` ⇒ 3/8 MISMATCH) |

**"两臂不同卡"是本次新增的第六项,此前无任何工具会查。** 若两臂挤在同一张卡,时间片共享
SM/L2/带宽 ⇒ **两臂每个延迟都被邻居污染**,那是**必须停下重启**而非记一条注意事项。
`CUDA_VISIBLE_DEVICES` 在**启动命令**里而不在 config 文件 ⇒ `audit_arm_comparability.py` 比不到它。
**单次 `nvidia-smi` 不能作答**:GPU 作业是一次性子进程,臂在作业间隙显示 0% / 0 MiB
—— 我第一次的快照就只见 GPU1 有进程,连续采样后才确认分离。

**第三项是本次新增的检查,原先根本没被问过。** `trials_per_space` 是**按空间**收费的,而一次 **K 扩展会为同一候选发布第二个空间** ⇒ 该臂**再领一份 40 trial 预算**。检查器此前只问"每空间预算是否相等"(相等,40)与"速率是否相等"(中位数相等),把 `expansions 2` 与 `1` 当装饰打印,然后照旧宣布"latency 差可归因于开关"。**不可归因** —— 一臂就是多搜了 1.7 个空间。

修法与其自身的验证:阈值不拍百分比,单位取**检查器自己测出的每空间预算**,并以"**空间当量**"(`trial/单位`)比较,因为被墙钟截断的空间只买到 1/40 个空间的搜索量而不是 1 个 —— 我的第一版就是直接比原始 trial 差,把"少了整整一个空间"读成 39 < 40 而放过。`revert_check_total_search.py` 造 8 个变体,**8/8 全被抓**;其中变体 2/4/6 第一次**没被抓住**,病根在我的 fixture 太整齐(空间数、trial 数、模态预算同步变动),已补三个把这些量拆开的测试。中途只**记录**不判决(落后的臂还会追),结语行同步改成"PARITY OK ON WHAT IS ANSWERABLE NOW",避免它被单独引用。

**处理臂现在也有 1 次 K 扩展**(第 4 空间已开),所以"扩展只发生在控制臂"是我在 2.8 h 快照上的过度断言,实为 **2 vs 1**;方向不变。

**这个不对称不可能是开关的下游**:S7 唯一作用路径是 `study.enqueue_trial`,而 **16 次重算 0 投递** ⇒ 两臂在同一空间同一种子下的抽样序列逐位相同。既然 S7 从未改变任何一次抽样,它就无法改变 `at_boundary`、从而无法触发或抑制扩展。**必然是候选侧性质。**

**期限已复验**(两臂一致):`screen_timeout_s` = **120 s**(D9 修复生效,非旧的 1200 s);
真 trial 期限 = **1800 s**(build 1200 + eval 600),由 `worker_client.py:301`
`proc.communicate(timeout=)` 执行、310 行 kill,每 job 独立进程组(288 行)
⇒ 一个 job 超时不会连带杀掉共享通道里健康的另一个。

## 2. 零投递的五条成因(全部定量,均**非**可中途修复的缺陷)

| # | 成因 | 实测率 | 能否靠采样解决 |
|---|---|---|---|
| 1 | **墙被后来的成功 trial 抹掉** —— 判据要求被拒值在**已测范围之外**(`wall_attribution.py:265`) | `NUM_WARPS=16`:trial 6 被拒、**trial 14 跑通** ⇒ 墙从 prefix 20 起消失 | 否(越跑越少) |
| 2 | **软墙适用门要求最优 trial 自己 spill** | 79 个空间只 **30 个(38.0%)** | 否 |
| 3 | **`monotone` 单独否掉的墙比它保留的还多** | 32 道墙:留 8、增益≤0 丢 15、**仅 monotone 丢 9(28.1%)**;但 **4/6 末段回升 +27%~+120%** 远超 16% 噪声 ⇒ 多数是**真拐头** | 否(放宽会引入变差的墙,上限仅约 2 道) |
| 4 | **软墙恒指高侧,而最优点常已在域顶** | 24 道软墙 **11 道(45.8%)** | **否 —— 只有扩域或改写** |
| **5** | **墙可以通过斜率过滤器却在归因阶段失败** —— 归因在过滤器**之后**跑,`walls_worthless` **结构上不含**它的失败 | **16 次归因事件只有 1 道墙走完两门 = 0.06/事件** | 否(与采样无关) |

**成因 5 是最紧的一条,而且它让前四条的计价全部偏高。** 三道被探测的墙全部 `not_attributed`,
原因完全一致 —— **θ* 处根本没超共享内存上限**:

| run | knob | 拒绝值 | tail_gain | `max_shared`/limit | over_ratio |
|---|---|---|---|---|---|
| arm3 | `BLOCK_N_G` | 256 | **60.69%** | 73728 / 101376 | 0.727 |
| arm3 | `NUM_WARPS` | 32 | 25.56% | 73728 / 101376 | 0.727 |
| **s7-control** | `NUM_WARPS` | 16 | 21.88% | 98304 / 101376 | **0.97** |

那些 trial 的 `failure_kind` 确实是 `infeasible_shared_memory`,但从 θ* 出发按该 knob 消融重测时
**不超限** ⇒ 归因**正确地**判 `not_attributed`,没有任何资源上限可供改写去解开。
**控制臂 `wallsfound=1` 因此从未产出任何可用输出。**

⇒ **任何"有多少道墙可用"的判据是 `verdict == "attributed"`,不是 `walls_found - walls_worthless`。**
按后者算,seed 来源是 0.25 道/事件;按前者算是 **0.06**,**差 4 倍**。

**成因 4 的现场证据**(处理臂第二候选,`spills_at_best=286.0` ⇒ 适用门**通过**):

```
n_told=20  soft_wall BM_ATT   gain 13.0%  choices [32,64,128] incumbent=128 -> None
n_told=40  soft_wall BM_PROJ  gain 16.6%  choices [32,64,128] incumbent=128 -> None
RESOURCE_SOFT_WALL: applicable=True n_walls=1 param=BM_PROJ gain=16.6%
```

这两次记的是 `n_skipped_no_value_toward_wall`,**不是** `n_skipped_no_wall` ——
该计数器在墙**已经通过** `find_walls` 与 `select_for_probing` 之后才递增。
**读零投递必须分开看这两个计数器。**

**预算报价随之修正**:"hard+soft ≈ 18 点/run" 要再乘 38%(适用率)再乘约 54%(可投递率)。

## 3. 两个**不可**归因于开关的差异(记录以防被误引用)

**(a) latency 差 5.90%,超出噪声底** —— 控制臂 3.1775 ms vs 处理臂 3.3649 ms。
逐候选拆开后,领先**全部**来自 `cand-13ead9f3`(控制臂 generator 自己写的种子),
而该空间里 S7 本来也没投递任何点。两臂种子**按构造不配对**:

```
处理臂  cand-6f5cdc80 3.3649 <== 臂最佳   cand-5e033365 3.5108   cand-61b42d30 4.3904
控制臂  cand-13ead9f3 3.1775 <== 臂最佳   cand-d70a3f18 3.4232
两臂无任何共享候选 ⇒ 未配对种子差异,不得报成效应
```
**噪声底只能排除测量噪声,对"某臂拿到了哪些候选"没有约束力。**

**逐空间拆开后,这条更强了**(`space_accounting.py`,含 K 扩展的 v1→v2 两版空间):

| 空间 | best (ms) | 扩展带来的增益 |
|---|---|---|
| 控制 `cand-13ead9f3` v1 | 3.2082 | — |
| 控制 `cand-13ead9f3` **v2**(扩展后)| **3.1775** ← 该臂最佳 | **0.96%** ⇒ **噪声底内** |
| 控制 `cand-d70a3f18` v1 | 3.8410 | — |
| 控制 `cand-d70a3f18` v2(扩展后)| 3.4232 | **10.88%** ⇒ 真增益 |
| 处理 `cand-6f5cdc80` v1 | 3.3649 ← 该臂最佳 | — |

`cand-13ead9f3` 的 **v1 就已比处理臂最佳快 4.66%**,扩展只再贡献 0.96%(噪声内)⇒ **5.90% 的差仍是种子差,不是扩展带来的**。
同时这是"**2/2 次扩展都变好、但只有 1/2 可归因**"的第二个独立实例,与
[[k-expansion-causal-evidence-l3-43]] 记录的比例一致。

**(b) trial 速率差** —— 最贵的配置类(`NUM_WARPS=1` + 大 tile + `ieee`)在
**控制臂每一个空间里都未声明** ⇒ 该臂**抽不到**(0 trial / 0 s),
而处理臂 8 个 trial 吃掉该臂 **54.2%** 的 GPU 时间(占比随墙钟漂移,引用须带时点)。两个最贵的:

| trial | 墙钟(`job_wall_s`) | 记录的 kind | 真实原因 |
|---|---|---|---|
| `tr-4b6d6cdf` | 1081.4 s | `runtime_error` | 共享内存 `Required 131072 / 限 101376` |
| `tr-0ed5597e` | **1733.8 s** | `runtime_error` | **同一原因,同样被误标** |

⇒ **不再是"一个离群点"**:同一形状在同一臂**复发**,累计占该臂 GPU 时间的多数。

**期限已实测未被突破**:两臂 255 个带 `job_wall_s` 的 trial 中 **0 个超过 1800 s**,`job_timed_out` **恒为 false**
⇒ 那条杀进程路径至今仍是**未被执行的分支**([[an-unreachable-branch-is-not-a-safeguard]]),
其唯一证据是 `tests/test_timeout_kill_reaches_the_tree.py` 在 box4 上实测 `killpg` 会连带杀掉孙进程(ptxas 的位置)。

> ⚠️ **不要把检查器打印的 `slowest: 1854 s` 当成 trial 墙钟**。那是**事件间隔**,
> 因 `max_shared_jobs=2` 把编译与计时重叠,间隔会跨越并发通道里发生的其他事
> —— 检查器自己那行括号提示就是这个意思。真值取 trial 记录的 `job_wall_s`。

`check_arm_search_parity.py` 判 **`PARITY OK ON WHAT IS ANSWERABLE NOW`**(3.50 h 处):
中位间隔 32.9 vs 34.7 s = **1.06x**,判据本身没问题 —— 中位数确实说明**分布主体一致**、
这笔成本不随墙钟按比例累积。**速率差正在收窄而非扩大**:1.36x → 1.67x → **1.46x**,
尾部占间隔总和从 57% 降到 **48%**,与"TAIL 不随墙钟累积"的判定一致。
收尾时若该比值反向扩大,应在报告里**同时列出**"中位数对等"与速率比,
不要只引用前者(这正是 `PARITY NOT OK` 的措辞所要求的:把混淆与数字并列,而不是收紧容差)。

**（c）两臂第 4 个候选的命运不同,同样与开关无关。** 处理臂的 `cand-13daa2b8` 在见证门被拒
(`witness_default_failed`)并进入 loop A 修复,而控制臂的 `cand-5add4f88` 直接通过、已开始调参。
这会**进一步扩大总搜索量差**。

拒绝是**候选自己的问题,门判得对**(与 [[gate-not-at-fault]] 一致,不是 fp16 见证门弱点):

| 判据 | 读数 |
|---|---|
| 非有限输出 | **29,581,824 / 50,331,648 = 58.8%**(NaN 29,529,843 + ±Inf 51,981)|
| `cosine` / `frac_within_tol` | 0.1817 / 0.1453 |
| `p99_rel_err` | **4.951e+04** |
| ieee 参考 vs tf32 参考 | 两者**读数几乎相同** ⇒ 与 dtype 无关 |
| fp64 相对门 | 也失败,`candidate_rmse_vs_fp64 = nan` |
| **参考自身** ieee-vs-tf32 噪声底 | `cosine 0.99999993`、`median_rel_err 3.8e-04` ⇒ 门未被噪声推动 |

NaN 占非有限值的 **99.8%** ⇒ 候选自己算出了 NaN(典型为 softmax 中 `exp` 上溢或除零),不是容差问题。

**loop A 修好了它,而且诊断可核对**(本对首次跑通 loop A,修复调用 341 s,远低于配置注释记录的 0.99 h):

> `fixed.py` 与 `broken.py` **逐字节相同**,只在 `_softmax_kernel` 里加一段:在原 store 循环之后,
> 用 `for start_n in range(((hi + BN - 1) // BN) * BN, T, BN)` 对**从未被计算过**的 key 位置写 `tl.zeros`。

这与见证门读数**完全吻合**:因果掩码下超出 `hi` 的位置从未被写 ⇒ 输出缓冲区留着未初始化内存 ⇒ 读出 NaN。
"58.8% 非有限、其中 99.8% 是 NaN"正是"过半位置没被写"的形状。修复只加零填充、其余字节不动,
也满足单一 `PARAMS` 契约。随后 `parameterizer` 对修好的源码重新参数化(loop A 的第 2 次尝试,
上界 `repair_attempts = 2` ⇒ 最多 3 次参数化 / 2 次修复)。

**agent 时间已排除为 trial 差的原因**(`agent_time_by_module.py`,3.56 h 处):

| 臂 | agent 总时长 | 占墙钟 | 分模块(总秒)|
|---|---|---|---|
| 控制 | **0.97 h** | 27% | analyst 1496、parameterizer 1154、generator 846 |
| 处理 | **1.06 h** | 30% | parameterizer 1258、analyst 1257、generator 970、**repair 341** |

两臂只差 **0.09 h**,远不足以解释 80 个 trial 的差距 ⇒ trial 差仍归因于**空间预算**(扩展 2 vs 1)
与**昂贵配置类**,与 agent 开销无关。`generator` 两臂各只 1 次调用却占各自 agent 时间的 24% / 25%,
是一次性成本,不影响对比。

**那两个误标 trial 同时是 §4 第一条缺陷的现场代价**:它们之所以能跑到 1081 s / 1734 s,
是因为 launch 前的 screen 在 8 MB 级 PTX 上超了 120 s 的 cap(设计如此:screen 失败绝不构成判决),
于是真 trial 照跑、直到 Triton 在 launch 时才拒绝。**修标签不会省下这些秒**(钱花在 `ptxas` 里、
发生在分类之前),但会让这些点进入 `PRUNED` 从而**减少重复撞墙**——
下一轮按"减少重复碰撞次数"计价。

## 3b. "等 K 扩展就会出新墙"—— 实测 5/48,且 5 次全被更硬的门挡住

K 扩展是唯一能解除成因 4 的事件(它用**更宽的域**重调**同一候选**,于是"最优点已在域顶"不再成立)。
实测 box4 全部 48 次域加宽(`expansion_direction_vs_soft_wall.py`):

| 判据 | 结果 |
|---|---|
| 加宽方向 | **43/48 加在高端**,与软墙恒指高侧**同向** ⇒ 方向不是障碍 |
| 加宽的 knob 与同空间某道墙点名的 knob 重合 | **5/48** |
| 其中方向也一致 | 3 ALIGNED / 2 OPPOSITE |
| **最终可用** | **0** —— 3 道 `verdict=not_attributed`,2 道增益低于噪声底 |

```
arm3       cand-dc87a93a NUM_WARPS  ALIGNED  但 verdict=not_attributed
arm3       cand-948343ba GEMM_BN    ALIGNED  但无 verdict 字段(旧 run)
arm3       cand-948343ba NUM_STAGES_APPLY  OPPOSITE  且 gain −23.78%
s7-control cand-d70a3f18 NUM_WARPS  ALIGNED  但 verdict=not_attributed
s7-control cand-13ead9f3 BLOCK_M    OPPOSITE  且 gain 0.45% < 噪声底 2.35%
```

**处理臂本轮的首次扩展加宽的是 `COMBINE_NUM_STAGES`,而该臂软墙点名的是 `BM_PROJ`** ⇒ 不重合,
所以那 16 个 trial **不是** S7 的测试窗口,不得把它们读成对 S7 的检验。

**更根本的一层:更宽的域本身从未产出更优点**(`expansion_paid_off.py`,5 次加宽全查):

| 臂 | 候选 | 加宽的 knob | 新增值 | 新值处最好 vs 父最优 |
|---|---|---|---|---|
| 处理 | `cand-5e033365` | `COMBINE_NUM_STAGES` | 4, 5 | **−1.44%** |
| 控制 | `cand-d70a3f18` | `BLOCK_N` | 256 | **−78.18%** |
| 控制 | `cand-d70a3f18` | `NUM_WARPS` | 32 | **−34.48%** |
| 控制 | `cand-13ead9f3` | `BLOCK_M` | 8, 16 | **−8.60%** |
| 控制 | `cand-5add4f88` | `BLOCK_M` | 256 | **从未被抽到** |

**0 / 5 让空间自己的最优点落在新增值上。** 所以 `cand-d70a3f18` 那 **10.88%** 的改善
**不来自新值**(新值处最好只有 5.1656 / 6.8439 ms,而 v2 最优是 3.4232 ms),
而来自扩展额外领到的 **40 个 trial 预算** ⇒ 这是**预算效应,不是域效应**,
也是「2/2 变好但只有 1/2 可归因」的机制层解释。

**对成因 4 的重新解读**:软墙恒指高侧,而往高侧扩出来的值实测都更差(−78% / −34%)
⇒ 成因 4 未必是"缺少可投的值",而可能是"**高侧本来就没有更好的点**"。
这也削弱了"提高 C2 覆盖率应扩宽初始域"这条建议。

> ⚠️ **v1 与 v2 的 best 可能逐位相同,那不是缺陷。** 扩展会把父空间已测点作为 **anchor 重新入队**
> ⇒ **同一次测量被两个空间共享**。既非巧合打平,也非"未重测"
> —— 我曾把它误读成后者,并差点据此报出"这 40 个 trial 白花了"。

> ⚠️ **两次读法错误,记录以防重犯。** `RESOURCE_SOFT_WALL` 的 payload **只有 `candidate_id`,没有 `space_id`**
> (空间要从紧邻其前的 `RESOURCE_WALL_ATTRIBUTED` 取)。
> 第一版探针把软墙**按整个 arm 池化**,打印 `{'BLOCK_M': 1}`,我据此与另一个空间的 `BLOCK_M` 扩展匹配
> ⇒ **伪造出一个匹配**;第二版按不存在的 `space_id` 分组 ⇒ 全落进 `"?"` 桶、**伪造出 0/48 无匹配**。
> 两者是同一个错:**读了一个不存在的键**。且第二版只看软墙,漏掉硬墙,而 5/48 里 4 次是硬墙。

## 4. 顺带发现的两个缺陷(**本轮不改**,已记录方案)
- **共享内存拒绝被误标 `runtime_error`**(`worker_main.py:35-39` 只识别 `out of memory`)。
  实测 **15/755 = 2.0%** 的拒绝被误标,3/7 个 run 至少一次。影响三个消费者:
  `find_walls` 的**全部输入**、`tpe.py:218` 的 PRUNED/FAIL、`deweight.py` 的候选级汇总。
  按**墙的计数**看是 2.0%(噪声底内);按**机时**看,处理臂两次误标即占该臂 GPU 时间的多数(见 §3b)。
  不中途改的理由与改法见 `next-round-changes.md` §4.1 / §4.1b。
- **`audit_arm_comparability.py` 五处被引用却从未提交** —— 已补写并加 9 个测试;
  它自己第一次运行也是错的(`sys.path.insert` 导入了 box4 上另一个早于 S7 的 checkout,
  致 96 键/全 `None`、误报"两臂配置相同"),修法与两个测试见该文件。

## 5. 收尾必须按序重跑

```bash
# 1) 搜索是否对等(传 run 目录)
python scripts/check_arm_search_parity.py <control_run> <treatment_run>
# 2) config 是否只差一处(传 config 文件)
python scripts/audit_arm_comparability.py \
    configs/experiments_s7_control_box4gpu1.yaml \
    configs/experiments_s7_treatment_box4gpu0.yaml \
    --expect v3.slope_guide.enabled
# 3) P1–P5 判读(传 run 目录,处理臂在前)
python scripts/analyze_s7_pair.py <treatment_run> <control_run>
```
同机对**不需要** `compare_calibrations.py`(那是给跨机对定分母的)。
在 box4 上运行 (2) 与 (3) 时必须 `PYTHONPATH=/root/autodl-tmp/work/opop/src`,
否则会导入另一个 checkout —— 见 §4 第二条。

**三件工具已在本对的真实数据上试跑通过(3.9 h 处),收尾不会再有工具意外**:

| 工具 | 试跑结果 |
|---|---|
| `check_arm_search_parity.py` | 跑通;报 `PARITY OK ON WHAT IS ANSWERABLE NOW` + 1 条 PROVISIONAL |
| `audit_arm_comparability.py` | 跑通;**`COMPARABLE`**,119 键 4 处不同;并印出所加载的 config 模块路径(防导入错 checkout 的守卫) |
| `analyze_s7_pair.py` | 跑通;试跑中**查出并修好两处**(见下) |

试跑 `analyze_s7_pair.py` 查出的两处(均已修):
1. **对控制臂那道墙无话可说** —— 它印 `walls_found 1 worthless 0` 而 `no_wall_cause` 留空,
   读起来像"有一道可用的墙却没人用"。现会数 `walls_attributed_total`、点名
   `not_attributed (NUM_WARPS, over_ratio 0.970)`,并说明 `walls_worthless` 为何看不到。
2. **in-flight 警告在顶部、结论在 90 行之后** —— 任何 `tail` 读法都会拿到脱离警告的
   `THE MECHANISM DID NOT FIRE`(**我自己就这么读了一次**)。现在警告在结论块内就地重复。

另加 `gpu_pinning_check.py` 作为**第 0 步**(见 §1 第六项):它查的是别的工具都查不到的东西。
