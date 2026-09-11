# 待修复问题清单 + 后续实施/实验规划

**日期** 2026-09-11 · **checkout** `D:\Pyhon_projects\opop\v3` @ `03ab1a3`(= `origin/v3`,已推送)·
A800 `/root/autodl-tmp/work/opop` @ `03ab1a3`,全套 **834 passed, 1 skipped**

三个 run 全部结束。本文档把「还没修的」和「接下来干什么」放在一处,每条都带**证据强度**与**为什么
现在才动**。凡是状态断言都带 host:path@commit —— memory 库被多会话共用。

---

## 第一部分:本轮已完成(供对照,不再重复)

| commit | 内容 | 守卫 |
|---|---|---|
| `68aa1b3` | 8.68% 的分离不得写成「在噪声底内」 | 3 变体 + 1 测试 |
| `13937b8` | 重导脚本把每个族的 parent 混池(`cand-3760b4d7` 1/7 ↔ 真值 7/1) | 复现 3/2/3 与 7/1/0 |
| `80a4c14` | 复用测量不写 `.py` ⇒ `launch_bound` 对 ~40% 候选不可达 | 4 变体(A800) |
| `067e469` | 两份结果文档(配对臂 + A800 L3:48) | 落盘复核 |
| `59d5a71` | **prescreen 独立短超时**(原借用 `build_timeout_s`,box3 两批各 1200 s 零答案) | 6 变体 |
| `da1c39b` | **S2d(c) 保留槽位**(默认关) | 9 变体 |
| `03ab1a3` | **测得的 0 被记成「未测量」**(`n_spills` 11/128) | 3 变体 |

---

## 第二部分:未修复问题,按「是否阻塞论文」分层

### A 层 —— 阻塞论文结论,必须在下一轮实验前修

#### A1. ~~`excessive_speedup` 的 10x 常量对 L3:48 系统性偏低~~ —— **已修复(`c399640`)**

**实施结果与原方案不同,而且更彻底**:不只是「阈值从物理上界导出」,还**删掉了两处死代码并去掉了
每 trial 的参考重测**。

落盘审计(25 run / 6894 trial / 三台机器)给出的三条决定性事实:
1. **拒绝次数 0**,抬旗 3 次(全部 L3:48,全部 5/5 正确);
2. **拒绝分支在 relaxed 路径上不可达** —— `ref_latency_ms` 只在 `if correct and num_perf > 0`
   下赋值,而 `result["ok"] = correct` 已在上游拒掉不正确的 trial ⇒ 守卫里的 `if not correct:`
   永不执行。**一个不能执行的分支不是保险**;
3. **每 trial 重测参考的成本是实的** —— 每个计时 trial 多 ~13 次参考执行,仅 box1 就 30511 次,
   换来 0 次拦截。

**现在的形态**:阈值 = `max(compulsory_bytes/dram_tbs, flop_count/peak_tflops)` 导出的物理下界,
`reference_ms / floor_ms` 即**任何正确实现都不可能超过**的上界,再乘 1.5 安全边际。
参考延迟改用 run 自己的 100 采样 baseline(不再每 trial 重测)。三个拒答情形(无标定 / 工作集
装进 L2 / 两项都测不到)一律**不给数**并写明原因 —— 一个错的界比没有界更糟。

**已在三台机器的实跑中验证(2026-09-11 23:0x,落盘 `PLAUSIBILITY_BOUND`)**:

| box | 任务 | floor | 绑定项 | 物理上界 | 抬旗门 | 说明 |
|---|---|---|---|---|---|---|
| 1 | L3:43 | 2.3487 ms | compute | **9.20x** | 13.79x | 现有最好 3.647x,远低于门 |
| 2 | L3:43 | 2.3487 ms | compute | **9.15x** | 13.73x | 同上 |
| 3 | L3:48 | 0.8011 ms | dram | **17.48x** | **26.21x** | **旧 10x 抬旗三次的 14.29x 现在合法通过** |

顺带删除:`suspicious_speedup: 2.0`(**全代码库无任何消费者**,却出现在每个 config 里 ——
比缺失更糟,因为读者会以为有一道 2x 筛)。

守卫:14+12 个测试,`scripts/revert_check_plausibility.py` **13 变体双向覆盖**(界不再约束 /
界误伤诚实 kernel),A800 全 CAUGHT。

#### A2. ~~`rescue_from_sandbox` 丢掉 S2d 的 `expectations`~~ —— **已修复(`c399640`)**

按推荐方案实施:rewriter 额外把声明写进 `rewrites/expectations.json`,救援时读回。
救回的每一条仍走 `ResourceExpectation`(`extra="forbid"`)与同一份封闭词表,
且**按条丢弃而非按文件丢弃** —— 一个错名不该让整轮报废。

**构建期被 revert-check 抓到一个死分支**:原先写了 `sb.exists()` 预检,其变体**不改变任何行为**
(下面的 `except Exception` 已覆盖 FileNotFoundError)⇒ **删掉而不是留着**,与
`families.py` 的 `max_families_active < 2` 同一处理。

守卫:10 个测试,`scripts/revert_check_s2d_a2_rescue.py` **9 变体双向覆盖**(声明再次丢失 /
未校验内容被放进账本),A800 全 CAUGHT。

**风险仍然成立**:改了 prompt 契约 ⇒ **改后的 run 与改前不可比**。所以它落在 E1 之前,
E1 两臂同时带上它,是对等的。

### B 层 —— 影响预算效率或证据完整性,不影响已有结论

#### B1. D1:永不成功的 categorical 取值被反复抽到

**证据**:960 个 trial 里 **104 个(10.8%)** 花在「抽了 N 次、0 次成功」的取值上。根因已定位到
`tpe.py:119`(`correctness_mismatch` 报 `TrialState.FAIL`,Optuna 把 FAIL 完全排除在 TPE 模型之外)。

**修法**:per-(candidate, knob, value) 累计 N 次全败后从 FAIL 升级为 PRUNED,**绝不跨候选合并**
(L3:48 上 `tf32` 对两个候选无望,而 L3:21 上同一个 `tf32` 16/17 成功)。且**不动前 N 次抽样**。
N=3 是审计用的起点,不是实测最优。

**注意**:这**不是** dtype 禁令(已被明确排除),它不从空间里删掉任何取值。

**风险**:中 —— 改变 TPE 访问的点 ⇒ **改后的 run 与改前不可比**。

#### B2. D5:prescreen 在多 kernel 候选上定价过高(item 1 & 2)

**`59d5a71` 只解决了「跑飞」的封顶**,没解决**定价**。D5 实测 L3:21 上 12 次 prescreen 花 27.0 min、
只避开 56 个配置(17.4 min 的 trial 时间)⇒ **净 −9.6 min**。剩下两个修法方向:
1. **采样量随候选的 kernel 数缩放**(现在恒为 `trials_per_space`=40,不管一个变体编 1 个还是 4 个 kernel);
2. **只筛 shared-affecting 子网格**(只差 `NUM_WARPS` 或 cache hint 的两个变体产生相同的 `metadata.shared`,
   编两次是纯浪费)。

item 3(跨空间扩展复用)已在 2026-09-09 **证伪**,无可回收。

**风险**:低-中。不改变哪些配置**能**被测量(超时仍只留空缓存),但改变**采样哪些**去筛 ⇒ 严格来说
仍是 run 间不可比的改动。

#### B3. `correctness_mismatch` 约 21%(box3 上 30.6%)

**证据**:语料里 537 trial / 4.7 h。已确认**不是门的问题**(memory `gate-not-at-fault`:537/537 是候选
自己的问题),真杠杆在**候选生成侧**。

**这是独立课题,不是一个 fix。** 需要先做归因(哪几类结构性错误占大头),再决定是 prompt 改动还是
parameterizer 改动。**建议排在所有实验之后。**

### C 层 —— 记录在案,判据尚不充分

| 项 | 证据 | 为什么不动 |
|---|---|---|
| D3 rewriter 的 backend 切换在生产中未验证 | 35/35 全 Triton 是 prompt 要求的结果,不是发现 | CUTLASS 从未进 prompt 也没装;要验证得先装 |
| D4 实测天花板对 agent 行为的影响没有干净实验 | — | 与下面的「第四臂」是同一个实验设计问题 |
| #4 parameterizer 静默回退 repair | `finding-parameterizer-reverts-the-repair.md`;`REPAIR_REVERTED` 未实现 | 早先决定不实现;若要做需先量化发生率 |
| Loop D 在 19 个 run 中零执行 | 族数 ≥ `max_families_total`,且 D 排在调度最末 | 墙钟总是先到 ⇒ 要么抬预算要么改调度顺序,两者都改变可比性 |

---

## 第三部分:实验规划

### 前提:三个已完成 run 的定位

| run | 结论强度 | 可用于论文的部分 |
|---|---|---|
| box1 对照 `label` 3.11 ms | — | 配对实验的对照臂 |
| box2 处理 `vector+ledger` 2.84 ms | **−8.68%,超噪声底 3.7 倍** | **主结果**(S2 的 J2-5) |
| box3 L3:48 A800 0.9728 ms | 8.2263x 同精度,达物理上界 82% | 跨卡可移植性 + 绝对性能 |

**已经拿到的**:S2(vector vs label)的 J2-5 单边判决、G27 的首份配对证据、S2d(a)/(b) 的生产证据。
**没拿到的**:S2d(c)(两臂均未施测)、账本准确率的显著性(p=0.1585)、「结构 vs 体积」的区分。

### E1(建议优先):S2d(c) 的配对实验 —— 12 h × 2

**自变量**:`reserve_round_for_reconciled` False vs True(两臂都开 `vector` + `expectation_ledger`)。
⇒ 恰好是**「账本是否到达下一轮 prompt」**。

- **为什么现在能做**:`da1c39b` 已实现且默认关闭,9 个 revert 变体全 CAUGHT。
- **为什么值得做**:S2d 是三个子命题里唯一**完全没有证据**的一个,而它是「反馈闭环」这个卖点的核心。
- **前置**:建议先做 **A2**(救援丢声明),否则一次 transport 超时就让某一轮的账本为空 —— box3 已经
  发生过一次。
- **注意**:开关改变搜索顺序,所以**新的一对不能与现有三个 run 合并比较**,只能自成一对。
- **成本**:两台机器各 12 h。box1 + box2(同为 4090,标定身份一致)。

### E2:第四臂 —— 结构化判断 **+** 距上界真实数值

**这是用户此前明确表示最感兴趣的方向**:「同时给 vector 臂语言判断,又有类似 label 臂的精确数值
(或处理后能反映距上界的真实数值)」。

- **为什么必须新写渲染函数**:不能绕过 `assert_no_raw_vector`(它是 `for_prompt()` 唯一入口的护栏),
  所以要一个**受 config 门控的第三种 mode**,而不是放宽护栏。
- **必须排除的量**:`pct_of_dram_peak` / `pct_of_compute_peak` —— 它们是 **1/gpu_ms 的改写**
  (memory `dram-and-compute-pressure-are-latency-restated`),给进去等于给延迟本身。
- **可以给的量**:`n_regs`/`shared_bytes`/`occupancy` 的实测值 + **到 FLOOR 的距离**(现在 vector 臂
  只给后者的文字版)+ `candidate_aten_bytes` 相对 task floor 的比值。
- **为什么需要更大 n**:现有配对是 3 族/臂,而这一臂与 vector 臂的预期差异**小于** vector 与 label 的
  差异(G9 第三次 run:`verdict` 与 `rich` 差 4.19% 而臂内跨度 4.72%,未分离)。**需要 ≥4–6 族/臂**
  ⇒ 又一对 12 h,或把 `wall_clock_hours` 抬到 18–20 h。
- **风险**:这是**三臂比较**(label / vector / vector+numbers),要么跑三臂(3 × 12 h),要么用 vector
  臂的现有数据当第二臂(但那是**不同 run**,只能作为参考而非对照)。**建议跑完整三臂。**

### E3:L3:48 在 A800 上的复现 —— 12 h × 1

box3 那一次有 **1 次 resume + 2 次 agent 超时(0.83 h)**,事件跨度 13.67 h 而自报 12.383 h。
**0.9728 ms 这个数字本身可信**(5/5 正确、独立复测),但**「A800 上 12 h 能做到什么」这个说法不可信**。
`59d5a71` 的 prescreen 封顶预计能回收 ~0.9 h,所以一次干净复现值得做 —— 且它顺便验证 prescreen 修复。

**优先级低于 E1/E2**,因为它不产生新结论,只是收紧一个已有数字。

### 建议顺序

```
1. A1  excessive_speedup 阈值从物理上界导出        ✅ 已完成 (c399640)
2. A2  救援保留 S2d 声明                          ✅ 已完成 (c399640)
3. E1  S2d(c) 配对实验                            🟢 已启动 (12 h × 2, box1+box2)
   ↕ 并行: E3 L3:48 A800 干净复现                 🟢 已启动 (12 h × 1, box3)
4. 分析 E1/E3 → 决定 E2 的臂数与预算
5. E2  三臂实验(label / vector / vector+numbers)  (12 h × 3 或 18 h × 3)
6. B1 + B2  预算效率修复                          (实验之间做, 不在实验中途)
7. B3  correctness_mismatch 归因                  (独立课题, 最后)
```

**关键约束**:B1/B2 都改变 TPE 访问的点或采样,**必须在两次实验之间**做,不能在一对实验的中途 —— 否则
两臂不可比。A1/A2 同理,但它们在 E1 之前,所以 E1 的两臂会同时带上它们,是对等的。

### 本轮启动状态(2026-09-11 23:0x,三台机器均 `842e2a6`,全套 896 passed / 1 skipped)

| box | 实验 | config | run id | 自变量 |
|---|---|---|---|---|
| 1 | **E1 对照** | `experiments_e1_box1_control.yaml` | `run-l3-43-20260911-230217` | `reserve_round_for_reconciled: false` |
| 2 | **E1 处理** | `experiments_e1_box2_reserved.yaml` | `run-l3-43-20260911-230736` | `reserve_round_for_reconciled: true` |
| 3 | **E3 复现** | `experiments_l3_glm_a800_s2.yaml` | `run-l3-48-20260911-231217` | 无(单臂) |

E1 两臂的**解析后 config 差异只有两项**:那个开关 + `wsl.venv`(机器路径)。
`device` / `budgets` / `evaluation` / `agents` 四个块**逐字段相等**,由
`tests/test_config_strictness.py` 的 5 个新测试守住。两臂**都开** `mode: vector` +
`expectation_ledger: true` —— 否则开关会因 `wiring.py` 的合取而失效,实验测不到任何东西。

**标定前置条件已复核**:box1/box2 的 `runs-v3/calibration.json` **md5 完全相同**
(`e0c4baaa…`,含相同 `measured_at`)—— 这是**刻意**的:box2 自己的测量被保留为
`calibration.box2-own.json.bak`(dram 0.9098 vs 0.9095,bf16 167.42 vs 164.29,**差异 ≤1.9%**),
安装 box1 的文件是为了让两臂**除以完全相同的天花板**,使延迟差异不含分母差异。
box3 用自己的 A800 标定(`l2_bytes` = 40 MiB,L3:48 的 1.35 GB 远超之 ⇒ DRAM 项适用)。

### 顺带上线:资源→性能转化速率(`842e2a6`,**只落盘、不消费**)

见 `docs/plan-resource-to-latency-conversion-efficiency.md`。`CONVERSION_RATES` 事件每候选一条,
**代码里没有任何消费者**(由一个 grep 源码的测试守住)⇒ **携带它的 run 与不携带的仍然可比**,
所以它可以现在上线,让 E1/E3 把语料从 99 个 series 扩到约 150 个,再决定是否进 prompt(那属于 E2)。

---

## 第四部分:交付前必做(用户侧)

1. **轮换明文 API key** —— `.opencode/opencode.jsonc`、`opencode_backup.jsonc`、`kimi-provider.yaml`、
   全局 `~/.config/opencode/opencode.jsonc`,以及**三台机器上** GLM run 沙箱里的 `opencode.json`。
   `sandbox_config_path` 会把整个 provider 块(含 key)复制进 `runs_dir` 下**每一个** agent 沙箱 ⇒
   一次 `tar`/`scp` `runs_dir` 就会把 key 带出机器。
2. **轮换 GitHub PAT**。
