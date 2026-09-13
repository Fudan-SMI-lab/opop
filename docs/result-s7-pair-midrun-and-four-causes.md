# S7(斜率引导采样)对照跑:中途状态与五条已定量的零投递成因

**日期** 2026-09-13 · **box4,同机双卡 RTX 4090** · 处理臂 GPU0 `s7-treatment`、控制臂 GPU1 `s7-control`,
两臂同一 run id `run-l3-43-20260913-202332`,均自 `39f01ea` 启动 · 任务 L3:43

> **这是中途快照,不是结果。** 两臂均未写 `RUN_FINISHED`,任何数字在收尾前都不得引用。
> 收尾须重跑三件工具(命令见文末)。

## 0. 本文能与不能回答什么

**能**:S7 在真实语料上的**投递率**,以及零投递的**逐门归因**。
**不能**:"S7 能否找到更好的配置"。9 次重算 **0 投递** ⇒ 采样上两臂无差别,
latency 差只反映两次独立搜索的方差。读数器 `analyze_s7_pair.py` 会在 `not p1` 分支
**拒绝作结论**并打印下表的成因 —— 这是**有诊断的 null**,不是空白的 null。

## 1. 有效性:五项全部关闭

| 项 | 结论 | 依据 |
|---|---|---|
| 自变量唯一 | ✅ **119 键,恰好 4 个不同** —— `v3.slope_guide.enabled` 加三条强制隔离路径 | `audit_arm_comparability.py` → `COMPARABLE` |
| **每空间**预算相同 | ✅ 两臂均 **40 trial/空间**(按**已关闭**空间作答) | `check_arm_search_parity.py` |
| **总**搜索量相同 | ⚠️ **否 —— 200 vs 131 trial(5 vs 4 空间),差 1.7 个空间** | 同上,新增的第三问 |
| 速率差 | ✅ 判为 **TAIL**,不随墙钟累积;归因到**候选**而非开关 | 中位间隔 32.8 vs 34.8 s(**1.06x**,在容差内) |
| 分母 | ✅ **同机双卡、共用一份 `calibration.json`**(schema 5)⇒ 字面相同 | 跨机才需 `compare_calibrations.py` |
| 实现无缺陷 | ✅ 出厂 `SlopeGuide` 重放与运行日志 **8/8 逐字段一致**(含每个跳过计数器) | `s7_replay_vs_log.py`;负对照会失败(错 `use_soft_wall` ⇒ 3/8 MISMATCH) |

**第三项是本次新增的检查,原先根本没被问过。** `trials_per_space` 是**按空间**收费的,而一次 **K 扩展会为同一候选发布第二个空间** ⇒ 该臂**再领一份 40 trial 预算**。检查器此前只问"每空间预算是否相等"(相等,40)与"速率是否相等"(中位数相等),把 `expansions 2` 与 `1` 当装饰打印,然后照旧宣布"latency 差可归因于开关"。**不可归因** —— 一臂就是多搜了 1.7 个空间。

修法与其自身的验证:阈值不拍百分比,单位取**检查器自己测出的每空间预算**,并以"**空间当量**"(`trial/单位`)比较,因为被墙钟截断的空间只买到 1/40 个空间的搜索量而不是 1 个 —— 我的第一版就是直接比原始 trial 差,把"少了整整一个空间"读成 39 < 40 而放过。`revert_check_total_search.py` 造 8 个变体,**8/8 全被抓**;其中变体 2/4/6 第一次**没被抓住**,病根在我的 fixture 太整齐(空间数、trial 数、模态预算同步变动),已补三个把这些量拆开的测试。中途只**记录**不判决(落后的臂还会追),结语行同步改成"PARITY OK ON WHAT IS ANSWERABLE NOW",避免它被单独引用。

**处理臂现在也有 1 次 K 扩展**(第 4 空间已开),所以"扩展只发生在控制臂"是我在 2.8 h 快照上的过度断言,实为 **2 vs 1**;方向不变。

**这个不对称不可能是开关的下游**:S7 唯一作用路径是 `study.enqueue_trial`,而 13 次重算 **0 投递** ⇒ 两臂在同一空间同一种子下的抽样序列逐位相同。既然 S7 从未改变任何一次抽样,它就无法改变 `at_boundary`、从而无法触发或抑制扩展。**必然是候选侧性质。**

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
**控制臂四个空间里全部未声明** ⇒ 该臂**抽不到**(0 trial / 0 s),
而处理臂 6 个 trial 吃掉该臂 **61.8%** 的 GPU 时间。两个最贵的:

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
`check_arm_search_parity.py` 仍判 `PARITY OK`(中位间隔 33.1 vs 35.2 s = **1.06x**,比早先更紧),
判据本身没问题 —— 中位数确实说明**分布主体一致**、这笔成本不随墙钟按比例累积。
**但必须如实记下该判据吃紧之处**:速率差已从 1.36x 扩大到 **1.67x**,尾部占间隔总和的 **57%**。
收尾时若该比值继续扩大,应在报告里**同时列出**"中位数对等"与"速率 1.67x",
不要只引用前者(这正是 `PARITY NOT OK` 的措辞所要求的:把混淆与数字并列,而不是收紧容差)。

**这两个 trial 同时是 §4 第一条缺陷的现场代价**:它们之所以能跑到 1081 s / 1734 s,
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
