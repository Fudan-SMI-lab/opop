# 外部候选真实复测方案(2026-09-23 拟定,待执行)

> **起因**:论文里外部框架的门槛目前全部来自**它们自己声明的数字**;我对
> K-Search L3:43 那个 1000 μs 的排除也是**纯理论推算(FLOP / 峰值),没有跑过一行代码**。
> 用户提出的口径:**把所有外部代码真实跑一遍,每格取"声明值"与"实测值"里较差的那个。**
> 本文记录方案、已有资产、必须先修的问题,以及为什么现在还不能跑。

---

## 1. 为什么这件事值得做(它解决的是一个真实的弱点)

| 现状 | 问题 |
|---|---|
| 外部门槛 = 它们 README/result.json 自报的 candidate_us | 各家 adapter、基线、计时方法都不同(它们的 README 自己声明),**不是一个协议** |
| 我方数字 = 我们自己 harness 的 `final independent re-eval` | 与外部不同门、不同计时路径 |
| K-Search L3:43 的排除 | **只有理论论证**:两个 fp32 投影 309.24 GFLOP,TF32 也需 1.872 ms > 1.000 ms。**未实测** |

⇒ 复测后,每个门槛都变成"**我们亲手在同卡、同门、同计时下量到的数**",
而"取较差值"使这个口径**在任何方向上都不利于我们自己** —— 这是审稿时最强的姿态。

---

## 2. 已有资产:脚本早就写好了,只是从未执行

`external_files/4090_52_experiments_results_and_best_sources/` 下:

| 文件 | 作用 | 状态 |
|---|---|---|
| `_verify_external_official_gate.py`(150 行) | 把候选送进 KernelBench **官方** `eval_kernel_against_ref` 复测 | **已写好,没跑过**(无结果文件) |
| `_verify_manifest.json` | **45 个候选**的清单,含 `claim_speedup` / `claim_cand_us` / `claim_ref_us` | 齐全 |
| `_gate_compare.py`(63 行) | 逐对比较 cosine / RMSE,用于精度口径对照 | 已写好 |

**清单覆盖度**(45 个候选 = 6 任务 × 4 框架,部分任务含 3 个基座模型):

| 任务 | Foundry | K-Search | KernelBand | ShinkaEvolve |
|---|---:|---:|---:|---:|
| L2:16 | 3 | 3 | 3 | 3 |
| L2:59 | 1 | 1 | 1 | 1 |
| L2:61 | 3 | 1 | 1 | 1 |
| L3:21 | 3 | 3 | 3 | 3 |
| L3:43 | 3 | 1 | 1 | 1 |
| L3:48 | 3 | 1 | — | 1 |

脚本的关键设计(都已实现,可直接用):
- `eval_kernel_against_ref(..., num_correct_trials=5, num_perf_trials=100,
  timing_method="cuda_event", precision=torch.float32, check_for_excessive_speedup=True)`
- **同卡 eager 基线由 KernelBench 自己算**(`ref_runtime`),所以
  `measured_speedup_vs_eager = ref_runtime / runtime` 是一个协议内的自洽比值
- `--shard i/k` 分片、`--only <task>` 过滤、断点续跑(读回 `--out` 里 `done` 的记录)、每条增量存盘
- PATH 里自动补 `nvcc` / `ninja`(踩过的坑已内建)

**环境已就绪**:box1 `/root/KernelBench`、box7 `/root/autodl-tmp/kb-4090/KernelBench`、
box9 `/root/autodl-tmp/kb-5090/KernelBench`、box6 `/root/autodl-tmp/work/KernelBench`,
**四处 commit 全部是 `423217d`**(与所有外部包一致)。

---

## 3. 必须先修的一个问题:`precision=torch.float32` 会把低精度候选全判错

脚本硬编码 `precision=torch.float32`,对应官方 **atol=rtol=1e-4** 绝对门。
但**六任务中五个的 tf32 底噪就低于 0.99**,而且我们已确证多个外部候选用了低精度
(K-Search L3:43 = fp16 hi/lo 喂张量核;Foundry L3:48 = 纯 fp32 但 Shinka L2:16 = cuDNN NHWC)。

⇒ 在 1e-4 绝对门下,**低精度候选会被判 WRONG,于是拿不到 latency**,
"取较差值"就退化成"取声明值",这件事白做。

**修法(二选一,建议 A)**:

- **A. 双门都跑,分别记录。** 每个候选跑两次:
  (a) **官方门** `precision=fp32` / 1e-4 → 得到 `official_pass` 与该门下的 latency;
  (b) **我方门** `dual_witness_relaxed` + `fp64_relative_gate` → 得到 `our_gate_pass` 与 latency。
  论文表里用 **(b)** 做性能对比(与我方数字同门,公平),用 **(a)** 单独报一列
  "在官方 1e-4 门下是否通得过" —— 这一列本身就是有价值的发现。
- B. 只跑我方门。省一半机时,但失去"官方门通过率"这个独立信息。

另外两处要改:
1. **latency 必须在"判错"时也保留**(我们自己的 `--record-failed-timing` 思路):
   否则 WRONG 的候选无法参与"取较差值"。
2. **`check_for_excessive_speedup=True` 要记下它的触发**:它正是用来抓
   K-Search L3:43 那种不合理加速的,触发记录本身就是证据。

---

## 4. "取较差值"的判定规则(写死,避免事后挑数)

对每个 (任务, 框架) 格子:

```
bar = max( claim_cand_us , measured_cand_us )        # 耗时取大者 = 对它更差
```

配套规则:
1. **若实测 WRONG(我方门也没过)**:该候选**不作为门槛**,但必须在论文里列出
   "声明 X μs,实测在两种门下均未通过正确性"——这比静默排除更有力。
2. **若实测 COMPILE_FAIL**:同上,单独列。
3. **若实测比声明快**:仍取声明值(较差者),并注明"实测更快,我们仍采用其自报值"。
4. **每格的门槛 = 该任务全部框架里"取较差值后"最快的那个**(仍是最严门槛)。
5. **cross-check**:KernelBench 自算的 `ref_runtime` 应与我们自测的同卡 eager 相符;
   偏离 >5% 则该格作废重测(说明机器被占或形状不一致)。

---

## 5. 为什么现在不能跑

**15 个 run 正在占用全部 18 张卡**(A800 2、4090 8、5090 5 + box7/box9 余卡)。
复测要做 100 次 perf trial 的 CUDA event 计时,**与正在跑的实验互相污染**:
被测候选抢 SM/带宽会让在跑 run 的 `tuned_ms` 偏大,反过来在跑的 run 也让复测结果偏大。
`_gate_compare.py` 与我方 harness 都要求"GPU 空闲"才计时。

**可执行的时间窗**:
- 最早:A800 两路 12h 墙钟到期(约 2026-09-24 03:45,软,可能晚 1–2 h)→ 腾出 2 张 A800
- 4090 八路同期到期 → 腾出 8 张 4090(**box1 有现成的 `/root/KernelBench`**)
- 5090 第二轮 L2:59 到期更晚(17:00 启动 + 12h ≈ 次日 05:00)

**机时估算**:45 个候选 × (5 correctness + 100 perf) × 两种门 ≈ 90 次评测。
按每次 1–3 min(含编译)算,单卡约 2–4.5 h;`--shard 0/4`..`3/4` 四卡并行约 **40–70 min**。
这是很小的代价。

---

## 6. 执行清单(按序)

1. A800 / 4090 的 run 到期后,先出 `final independent re-eval` 报告(那是主表数据)
2. 确认卡真空闲:`nvidia-smi --query-compute-apps` 为空 + `tmux ls` 无残留
3. 改脚本:加我方门、加 `--record-failed-timing`、记录 `excessive_speedup` 触发
4. 把包 + 脚本传到 box1(它有 `/root/KernelBench` @ `423217d`)
5. 先单跑 **L3:43 / K-Search** 一格验证链路(这是争议最大的一格,也是最快的验证)
6. 四卡分片跑全部 45 个
7. 用 §4 规则重算全部门槛,更新
   `docs/paper-contribution-evidence-map-2026-09-23.md` 的三张主结果表
8. K-Search L3:43 的结论按实测改写:
   - 若实测 ≈ 1.0 ms 且正确 → **我的理论论证错了,必须公开更正并接受该门槛**
   - 若实测远慢或判错 → 理论论证得到实证支持,排除理由从"推算"升级为"实测"

---

## 7. 这件事对论文的意义(一句话)

现在三张主结果表的外部列是"**它们说的**";复测后是"**我们量的,且每格都取对自己更不利的那个**"。
后者能挡掉审稿人最自然的一击:"你们凭什么相信/不相信对手的数字"。

⚠ **安全**:复测只读外部候选源码与 KernelBench 参考,不触碰任何 run 目录,
不涉及 provider 配置,无凭据外泄风险。
