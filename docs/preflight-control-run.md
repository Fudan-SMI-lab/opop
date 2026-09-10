# 对照 run 的 pre-flight:参数、场地选择、以及一个在启动前抓到的高危缺陷

**日期** 2026-09-11 · **分界 commit** `51b2d21` · **机器** box 3 / A800 80GB PCIe · **A800(权威)** 668 passed / 0 failed · **五个反向验证** 60 个错误实现全部判别

---

## 一、pre-flight 抓到的东西:处理臂差点是第二个对照臂

**这是本次 pre-flight 唯一的代码改动,也是它存在的理由。**

`load_config` 只读一个文件、**没有底层**,省略的键回落到 pydantic 字段默认 —— 而 v3 三个开关的默认值**就是 v2 的行为**。同时 pydantic 的**默认行为是忽略未知键**。两者相乘:

| 处理臂 YAML 写成 | 修复前的结果 |
|---|---|
| `v3: {diagnosis: {modes: vector}}` | **校验通过**,`mode='label'` |
| `v3: {diagnosic: {mode: vector}}` | **校验通过**,`mode='label'` |
| `... {mode: vector, expectation_ledgers: true}` | **校验通过**,`ledger=False` |

**而本项目此前没有任何 config 带过 `v3:` 块** —— 这条 YAML→pydantic 通路**零验证**,处理臂正要用第一个这样的块启动。

**拼错一个字母,那个 run 就是第二个对照臂**:两个 12h run 会一致,记录下来的结论是「**向量形态没有影响**」,**全程零报错**。24h GPU 换一个反的结论。

**与 107.8% 那次同形**:不崩溃、不为空,而是一个**可信的错误结论**。详见 G45。

**修法**:`StrictConfig`(`extra="forbid"`)全块继承。**在模型层而不是键名清单** —— `_apply_override` 用 `setdefault` 建路径,自己无法拒绝拼错,一处修改同时封住 YAML 与 `-o` 两条路。

**泛化测试抓到我第一版修得太窄**:`DeviceLimits` 在 `models/core.py`,只改 `config.py` 会漏掉后果最严重的那个块(丢掉 `max_shared_bytes_optin` ⇒ 每个 agent 被告知这台 A800 只有 101376B 而非 166912B,**静默禁掉这台机器存在的意义**)。

---

## 二、场地:L3:43,并且这是量出来的

**G9 有一次场地选错的记录**(起点已是平台期冠军 ⇒ 三臂 best 落在 1.84% 内,判据未被验证)。所以场地按**改写轮产量**选,而不是按印象:

| 任务 | run 数 | `FAMILY_ROUND_RECORDED` 合计 | 判断 |
|---|---|---|---|
| **L3:43** | 7 | **38** | ✅ **选中** |
| L3:21 | 8 | 24 | 次选,但带 train-mode BN 信息缺口 |
| L3:48 | 4 | **7** | ❌ **已确认平台期** |

**为什么改写轮数是决定性的**:S2d 的账本**第二轮起才有内容**,而 S2 的每维判决是逐候选的。5 个跑完的 run 一共只用了 9 个改写轮 ⇒ **任何 per-round 机制每 run 只有个位数样本**。选轮数最多的任务是唯一能拉高这个样本的手段。

**为什么排除 L3:48**:三个 run 都落在 1.55ms 附近、8/10 候选跨 4 个族全在 2% 内、DRAM 均达 93.7–95.5% 实测屋顶 ⇒ **平台期是物理不是噪声**。一个没有余量的任务**无法显示诊断形态的差别** —— 这正是 G9 犯过的错。

**L3:43 的正面理由**:最高改写轮产量(单 run 最高 12 轮);仍有余量(3.01ms / 3.647x,是**主动终止而非收敛**);且它有记录在案的 dtype / 张量核故事,正对 S2 的每维判决。

**已知风险,写下来而不是假设无害**:L3:43 forward 里有 `nn.Dropout` 且跑 train 模式,**看起来**不确定。实际不是(该任务自己的 `attn_pdrop`/`resid_pdrop` 都是 0.0,p=0 不抽 RNG)。但见证循环只 seed 一次就调参考两次 ⇒ **任何活跃 RNG 都会让「精度噪声底」其实是 RNG 噪声**。这台机器上已实测该任务噪声底 **0.9765**(与 box 1 的 0.9767 差 0.02%),所以这条在本次 run 上不成立;记下来是因为它对 p>0 的任务会成立。

---

## 三、两臂:恰好差两个键,并且有测试守着

```
对照臂  configs/experiments_l3_glm_a800.yaml      (无 v3: 块 = v2 行为)
处理臂  configs/experiments_l3_glm_a800_s2.yaml   (+ mode: vector, expectation_ledger: true)
```

**对照臂是「没有 `v3:` 块的那个文件」,不是「把默认值写出来的文件」**:`load_config` 无底层,把默认值明写会多出一处与代码漂移的地方。

**解析后的实测差异(不是文件 diff,是模型 diff)**:

```
v3.diagnosis.expectation_ledger: False -> True
v3.diagnosis.mode:              'label' -> 'vector'
```

**两个测试守这一条**,都对着**磁盘上真实的两个文件**:

1. `test_the_shipped_a800_arm_pair_differs_in_exactly_the_two_switches` —— 差异在**整棵解析后模型**上算,所以手抄 100 行时任何一行带进的改动都会被抓到。**这是唯一守住这件事的检查**:两个文件都能加载、两个 run 都能跑完,而一个漂移的 budget 会让 J2-5 无法归因。
2. `test_the_treatment_arm_actually_turns_both_switches_on` —— 「差两个键」也会被「**两臂互换**」满足,那会让 run 带着**反的符号**被记录下来。

**反向验证点名了这两个测试**(`revert_check_config.py` 的最后两个变体:处理臂少一小时墙钟 / 处理臂开关反向),各自 1/1 判别。

---

## 四、启动命令与顺序

**串行,绝不并发** —— 两个 run 同占一张卡会同时计时,静默污染彼此的测量:

```bash
cd /root/autodl-tmp/work/opop
# 臂 1(对照)
kernel-opt --config configs/experiments_l3_glm_a800.yaml    run --task level3:43
# 臂 2(处理),等臂 1 完全结束后
kernel-opt --config configs/experiments_l3_glm_a800_s2.yaml run --task level3:43
```

**先跑对照臂**:如果只有时间跑一个,一个对照臂加上五个既有语料仍是可比的;而一个孤立的处理臂什么都不能比。

---

## 五、这台机器上已复验的前置条件

| 项 | 状态 |
|---|---|
| checkout `51b2d21`,工作区干净 | ✅ `/root/autodl-tmp/work/opop`,branch v3 |
| Linux 变体不被主线覆盖 | ✅ 3 个(`cli` / `runtime` / `worker_client`),本次未 scp 任何文件,走 git pull |
| GPU 空闲 | ✅ 0% / 0 MiB,无 orchestrator 进程 |
| torch / triton / optuna | ✅ 2.8.0+cu128 / 3.4.0 / 5.0.0,`cuda.is_available()` True,读出 "NVIDIA A800 80GB PCIe" |
| venv 尺寸只有 273M(疑点) | ✅ **已查明**:`--system-site-packages`,torch 在 venv 外。**按 import 复验而不是按 du 判断** |
| 磁盘 | ✅ 49.1G 可用 |
| 标定 schema | ✅ **5**,与 `CALIBRATION_SCHEMA_VERSION` 一致;含 fp16/bf16/tf32/fp32 四个 Triton 可达分数 ⇒ **107.8% 的前提已被覆盖** |
| 任务噪声底 | ✅ L3:43 = **0.9765**(本机实测) |
| 两个开关真的被 orchestrator 读 | ✅ `orchestrator.py:1461`(prompt 分支)与 `:2272`(账本回灌) |

---

## 六、收尾必须显式检查的三件事

**写在启动之前,因为「以为修好了」的成本已经付过一次(G21 修复当天复发)。**

### 1. G27 的首次生产验证 —— **这条最容易被漏掉**

`conversion` 至今**零生产证据**:9 个 `FAMILY_ROUND_RECORDED` 中 **0 个**带 `conversion`、**0 个**带 `resource_deltas`(五个语料都早于修复 `677d956`)。

**收尾时必须查**:本次 run 的 `FAMILY_ROUND_RECORDED` 有多少个带 `conversion` 字段。**若仍是 0,那是一个新缺陷,不是「没有东西转化」** —— 而 `report.py` 的那一节会点名 G27 来区分这两者。

### 2. J2-5 / J2d-9:终局不变差

判据 `final_reeval_ms(处理) <= final_reeval_ms(对照) × (1 + 噪声底)`。

**必须用 `final_reeval_ms` 而不是 `tuned_ms`** —— 后者系统性乐观 1.5~6.7%。

**先验(G9)**:结构化分析有效(逐候选 7/7),但**信息量的边际收益饱和**。所以这条的风险是真实的:更多信息到达 agent 不被假定为改善(外部对照:few-shot 优化范例把 KernelBench L1 的 fast_1 从 10% 压到 6%)。

### 3. S3 / S4′ 在真实数据上的表现

- `ceiling_provenance` 是否带上了精度,`precision_mismatch()` 是否在任何候选上触发;
- 互补松弛检查报的是「无法运行」/「没有任何维被判 slack(**这不是通过**)」/「WEAK pass」/ 点名违规 —— **四种状态哪一种**;
- 低于地板的读数(融合良好的候选会读出低于 `compulsory_bytes`)是否**被报告而不是被 clamp**。

---

## 七、明确不折进这次启动的两件事

1. **低精度 prompt/契约的具体改法**(未补偿的 `dot` / `PREC` 门控整个算法)。**你说过「我需要先确认」** ⇒ 未落地。且按现行纪律它改 prompt 与候选契约 ⇒ **影响所有 agent 行为,必须落在实验之间,不可在 run 中途**。
2. **明文 API key 与 GitHub PAT 的轮换**。交付时的**你侧**任务:key 在三台机器上(box 1 / box 3 / 本机),且 `sandbox_config_path` 会把整个 provider 块(含 key)按每次 agent 调用复制进 `runs_dir` 下的沙箱 —— **tar/scp 一次 `runs_dir` 就会把 key 带出机器**。
