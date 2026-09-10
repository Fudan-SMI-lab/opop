# S2d 实施完成:方向预期 + 逐维对账账本

**日期** 2026-09-11 · **commit** `35fb1a9` + `14492fb` · **A800(权威)** 592 passed / 0 failed · **反向验证** 13 个错误实现**全部**在 A800 上被判别(本机 11/13,2 个需 optuna)

---

## 一、修的是什么:一个**结构上无法被证伪**的字段

| | 现状 | 后果 |
|---|---|---|
| `Hypothesis.expected_effect` | **自由文本** | 「should reduce memory pressure」**永远不可能是错的** |
| `failed_hypotheses` 条目 | 只有 `{id, change, round}` | rewriter 只看到「H1 试过、没帮上」,**看不到「H1 说 shared 会降,实际升了」** |

## 二、为什么这次要预测,而 D-2 撤掉过一个预测要求

**两处根本区别**,不是程度差异:

| | 被撤掉的那个(D-2) | S2d |
|---|---|---|
| 要什么 | **延迟增益的数值** | **资源变化的符号** |
| 可验证性 | 只能跑完才知道 | **编译期就能验证** |
| 推断难度 | 需要知道量 | 「加大 tile → shared 变多」**不需要知道多少字节** |
| 用途 | **分配预算** | **被检查的对象**,不分配任何东西 |

**用一个关于延迟幅度的失败去否决一个关于资源符号的声明,是把两件事混为一谈。**

## 三、三个部分

### (a) schema:`ResourceExpectation{dimension, expect, why}`

**只有方向,没有幅度字段** —— 三条实测说明变化率事先不可导出(shared 无闭式 0/96、代价图不可分离 0/10、连符号在宽扫上都会反转 13 条切片)。**给一个数字的字段就是给一个编造的字段。**

**`extra="forbid"` 是承重的,不是整洁**:pydantic 默认会**静默丢弃**未知字段,所以 `{"dimension": "n_regs", "expect": "down", "expected_pct": 40}` 会**校验通过而那个 40 消失** —— agent 就据一个从未被检查的量在推理,**正是这个 schema 要防的事**。我为此写的测试第一次是失败的,这个缺口就是那样暴露的。

**J2d-8 的硬边界**(grep + 测试):`expectation` 在 `families.py` / `convergence.py` / `tpe.py` / `stats.py` / `correctness.py` **零出现**。它只有两个出口:账本,和下一轮 prompt。

### (b) `evaluation/reconcile.py`:四种结局

| 结局 | 含义 |
|---|---|
| `hit` | 声明的方向与实测一致 |
| `miss` | 方向相反,或声明 `unchanged` 而它动了 |
| `vacuous` | 声明 `unknown` —— **计数,绝不静默跳过** |
| `unpredicted` | **动了而 agent 没提** |
| `unmeasured` | 声明了但一侧无读数 |

**`miss` 与 `unpredicted` 不许合并**:一个是「想错了」,一个是「没想到」,**对下一轮要说的话不同**。

**极性刻意不参与。** `up`/`down` 说的是**数字**,不是好坏 —— occupancy 上升就是 `up`,尽管更高更好。实测方向取 `sign(delta)`,**绝不取 `conversion` 的 `direction`**(那个是极性感知的):从 `direction` 推导会**把每一个「越低越好」的维度都反过来**,而那是绝大多数维度。

**实质阈值从 `conversion.py` import,不重述** —— 账本两段**不许在「有没有动过」上产生分歧**。

### (c) 回灌:账本进 prompt

**每条账本必带重新调参 caveat,而这是代码事实不是顾虑**:`conversion_verdict` 比的是父/子各自的 θ_best,而**改写后会重新调参** ⇒ 一个 `miss` 可能是 tuner 走到了别处。

**没有这条 caveat,账本就会为 tuner 的选择去责怪 agent** —— 而据错误反馈调整的 agent 比没有反馈的更糟,这正是 KernelPro 那条原始计数器臂的实测形状(1.77× vs 无反馈的 3.35×,p=0.0007)。

**渲染成散文,每轮一段、每维一行** —— JSON dump 正是旧的 `failed_hypotheses.json` 已经是的形态,而 G9 实测信息量边际收益饱和(4.19% 差距 vs 臂内 4.72% 散布,代价 2.1×)。

## 四、编排器侧的两个时序要求

| 要求 | 为什么 |
|---|---|
| **声明在 `REWRITE_PRODUCED` 里、在任何测量存在之前落盘** | 事后记录的声明**可能被结果塑形过,而日志里看不出来**。这个顺序才让后面的核对是**预测检验**而不是描述 |
| **对账时 `pop` 而非 `get`** | 否则第 N+1 轮会被拿第 N 轮的预测去打分 —— **一份看起来完全合理、而彻底错误的账本**,每个 hit/miss 都挂在错的轮次上 |

**记录无条件,只有 prompt 被开关控制**(与 S2 的向量同一不对称,同一理由):对照臂必须能从**自己的日志**被分析,否则 J2d-9 得多跑一个 run。

**resume 会恢复账本**(从 `EXPECTATIONS_RECONCILED`),与 `failed_hypotheses` 同一通道 —— 丢了账本的 rewriter 会重提一个本 run 已经检验过的想法,而**改写轮是最稀缺的预算**(5 个跑完的 L3 run 一共只用了 9 轮)。

## 五、反向验证抓到两个「不是证据」的测试 —— 而且**同一个根因**

**根因:fixture 与断言不是生产形状的。**

| 测试 | 为什么它不是证据 | 修法 |
|---|---|---|
| **occupancy 极性** | 我的 `deltas()` 辅助函数**硬编码 `direction: "changed"`**。而那个错误实现读的正是 `direction` —— 有个常数在那里,它**永远不走自己那条分支**,于是测试在一个坏的 reconciler 上「通过」。**一个不是生产形状的 fixture 什么都无法证伪** | `direction` 改为按 `_DIMENSIONS[name]["lower_is_better"]` 推导,与 `conversion.py` 完全一致 |
| **caveat 进渲染** | 只断言那些字**出现在文本某处**。而 `render_ledger` 把模块级 `CAVEAT` 作为收尾段落追加 ⇒ **「把 caveat 字段清空」和「整体 dump 成 JSON」两个变体都通过了它** | 改为断言**条目自己的字段** + 渲染是散文(`'"caveat"' not in text`) |
| **词表共享** | `"occupancy" in VOCAB` 和 `hits == len(VOCAB)` **都被一个三元素硬编码子集满足** | 改为断言集合相等 + 计数相等 |

## 六、A800 上一个既有测试失败了 —— 而它是对的

`test_rewrite_rounds_record_their_conversion_verdict` 用**源码文本切片**从 `store.append("FAMILY_ROUND_RECORDED"` 到下一个 `else:`,**假定判决是在 append 调用内部算的**。S2d 把 `conversion_verdict(...)` 提成了局部变量(为了让账本复用同一份 `resource_deltas` —— **账本两段不许在「什么动了」上分歧**),调用就移出了那个切片。

**行为没变,是测试的取景错了。** 两条断言改为从 `if evaluated:` 起切 —— 那才是真正决定一轮记录什么的块;并**单独断言判决进了 payload**,那才是真有风险的部分(算了却丢掉,正是这个测试要防的 `launch_bound` 形状)。

**但源码文本断言看不出判决是否还**到得了** payload**,而本仓库有一条实测记录:一个源码文本断言**在坏代码上通过、在修好后失败**。所以补了行为侧的一半:驱动 `conversion_verdict` + `reconcile` 走同一份输入,断言两段对「什么动了」一致 —— 包括**profile 缺失时账本侧必须给 `unmeasured` 而不是编一个 flat 读数**。

## 七、判据落地

| 判据 | 状态 |
|---|---|
| **J2d-1** 预期结构化可校验 | ✅ 词表在 `check_output` 校验,**拒绝信息列出全部合法名**(不说合法值的报错把校验变成猜谜,本项目曾为一条把编码问题说成内容问题的信息烧掉 3 次 repair) |
| **J2d-2** 四种情形分类正确 | ✅ 含极性双向对照(occupancy `up` 与 n_regs `down` 必须**同时**成立) |
| **J2d-3** 两段独立 | ✅ 四种组合在四轮上作为**集合**断言 —— 「独立」是关于联合分布的断言,不是单轮的 |
| **J2d-4** 账本进真实 prompt | ✅ 驱动 `seed_sandbox`,并**全沙箱扫**原始向量标记 |
| **J2d-5** 账本不膨胀 | ✅ 断言**增长速率恒定**而非绝对长度(绝对上限是个会被人调大的数字,超线性增长才是真故障) |
| **J2d-6** `unknown` 被计数 | ✅ |
| **J2d-7** caveat 在 | ✅ |
| **J2d-8** 预期不进选择 | ✅ 五个选择模块 grep 零出现 |
| **J2d-9** 终局不变差 | ⏳ **对照 run** |
| **N3** 读数不是常数 | ✅ 既有 `check_collection` |

## 八、待办

- [ ] S3(每维下界 + `ceiling_provenance`)
- [ ] S4′(conversion 的消费者)
- [ ] 一次对照 run:J2-5 / J2d-9 / **首次验证 G27**
