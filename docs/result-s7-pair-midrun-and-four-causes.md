# S7(斜率引导采样)对照跑:中途状态与四条已定量的零投递成因

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
| 预算相同 | ✅ 两臂均 **40 trial/空间**(按**已关闭**空间作答) | `check_arm_search_parity.py` |
| 速率差 | ✅ 判为 **TAIL**,不随墙钟累积;归因到**候选**而非开关 | 中位间隔 20.9 vs 23.2 s(**1.11x**,在容差内) |
| 分母 | ✅ **同机双卡、共用一份 `calibration.json`**(schema 5)⇒ 字面相同 | 跨机才需 `compare_calibrations.py` |
| 实现无缺陷 | ✅ 出厂 `SlopeGuide` 重放与运行日志 **8/8 逐字段一致**(含每个跳过计数器) | `s7_replay_vs_log.py`;负对照会失败(错 `use_soft_wall` ⇒ 3/8 MISMATCH) |

**期限已复验**(两臂一致):`screen_timeout_s` = **120 s**(D9 修复生效,非旧的 1200 s);
真 trial 期限 = **1800 s**(build 1200 + eval 600),由 `worker_client.py:301`
`proc.communicate(timeout=)` 执行、310 行 kill,每 job 独立进程组(288 行)
⇒ 一个 job 超时不会连带杀掉共享通道里健康的另一个。

## 2. 零投递的四条成因(全部定量,均**非**可中途修复的缺陷)

| # | 成因 | 实测率 | 能否靠采样解决 |
|---|---|---|---|
| 1 | **墙被后来的成功 trial 抹掉** —— 判据要求被拒值在**已测范围之外**(`wall_attribution.py:265`) | `NUM_WARPS=16`:trial 6 被拒、**trial 14 跑通** ⇒ 墙从 prefix 20 起消失 | 否(越跑越少) |
| 2 | **软墙适用门要求最优 trial 自己 spill** | 79 个空间只 **30 个(38.0%)** | 否 |
| 3 | **`monotone` 单独否掉的墙比它保留的还多** | 32 道墙:留 8、增益≤0 丢 15、**仅 monotone 丢 9(28.1%)**;但 **4/6 末段回升 +27%~+120%** 远超 16% 噪声 ⇒ 多数是**真拐头** | 否(放宽会引入变差的墙,上限仅约 2 道) |
| 4 | **软墙恒指高侧,而最优点常已在域顶** | 24 道软墙 **11 道(45.8%)** | **否 —— 只有扩域或改写** |

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

**(b) trial 速率差** —— 最贵的配置类(`NUM_WARPS=1` + 大 tile + `ieee`)在
**控制臂四个空间里全部未声明** ⇒ 该臂**抽不到**(0 trial / 0 s),
而处理臂 4 个 trial 吃掉该臂 **39.3%** 的 GPU 时间(单个最贵 1081 s = 该臂中位数的 **47 倍**)。
去掉那一个 trial,处理臂速率从 34.8 回到 41.1(控制臂 45.2),余下差距是 agent 延迟(39% vs 33% 墙钟)。

## 4. 顺带发现的两个缺陷(**本轮不改**,已记录方案)

- **共享内存拒绝被误标 `runtime_error`**(`worker_main.py:35-39` 只识别 `out of memory`)。
  实测 **14/725 = 1.9%** 的拒绝被误标,3/7 个 run 至少一次。影响三个消费者:
  `find_walls` 的**全部输入**、`tpe.py:218` 的 PRUNED/FAIL、`deweight.py` 的候选级汇总。
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
