# 复合探索方向该用什么算法:文献调研 + 与本项目实测的对账

**日期** 2026-09-13 · **checkout** `D:\Pyhon_projects\opop\v3` · 检索工具 `scripts/fetch_web_text.ps1`
(本机 WebSearch 不可用、WebFetch 拒 arxiv.org、curl 到 export.arxiv.org 超时;
Invoke-WebRequest + TLS1.2 + 浏览器 UA 能通 —— 这是本机唯一可用通道)
抓到的原文存 `external_files/lit_c2_sampler/`(已 gitignore,不入库;下文给 arXiv id 可复取)

---

## 0. 要回答的问题(你提出的)

> 调参不是找到墙就结束,而是**优先探索斜率高的参数空间里的墙**,但还有其他因素影响探索路线:
> 比如**优先探索未知空间**,或**优先探索更有概率获得高性能的空间**。最终方向应该是这几方面的综合,
> 而不是单一指标。类似 UCB,但比 UCB 复杂。**TPE 能不能解决?有没有更好的算法?**

拆成三个可分别回答的问题:

1. 这个"复合"在文献里叫什么、有没有成熟形式?
2. TPE 能表达其中哪几项、哪几项表达不了?
3. 有没有现成更合适的算法,代价是什么?

---

## 1. 你描述的东西在文献里的名字:**约束边界 / 水平集主动学习 + 优化**

你要的不是"多目标",也不是普通的 explore-exploit。它的特征是:
**最优解就在可行/不可行的边界上,而边界位置未知,必须一边找边界一边优化。**
这在文献里是一个**有名字、有专门方法**的问题,不是我们自创的框架。

### 1.1 最贴的一篇:BE-CBO(ICML 2024,arXiv:2402.07692)

**标题** Boundary Exploration for Bayesian Optimization With Unknown Physical Constraints

**它的核心观察,逐字**(原文 §1):

> "infeasible regions are typically discovered when **pushing the limits of what is physically
> possible**, and **the optimal solution lies on the boundary of feasible regions**."
> (不可行区域通常是在**逼近物理极限**时才被发现的,而**最优解就在可行域的边界上**。)

以及它对全部真实基准的统计:

> "while analyzing many real-world benchmark problems, we observe that **all these problems have the
> optimal solution on the boundary** between feasible and infeasible designs."

**这与我们的处境是同一个结构,而且我们已经独立实测到了同一件事**:
- 我们的不可行 = 共享内存超 101376 B 被硬拒(`infeasible_shared_memory`);
- 记忆 `a-shared-memory-wall-is-attributable-but-only-at-the-optimum` 实测:
  **从最优点 θ\* 出发 6/6 个墙都能归因到单个 knob,从默认配置出发只有 1/6**
  —— 即"墙"这个现象只在最优点附近才可见/可归因;
- 记忆 `boundary-flag-anchored-on-winning-trial` 也把边界判据锚在获胜 trial 上。

⇒ **BE-CBO 的前提在我们的数据里成立**。这既是好消息(有成熟方法可借)也是**新颖性风险**
(我们不能宣称"发现最优在边界上"这个观察,它 2024 年已发表)。

**它的机制,三件事:**

1. **可行性用分类器建模,不用 GP。** 训练一组独立 MLP(Deep Ensembles)输出"可行概率" `C(x)`,
   取均值作预测、取方差作不确定度。原文明说选 DE 而非 GP 的理由是边界复杂时 GP 精度不够,
   并声明"我们是第一个把 Deep Ensembles 用于此目的的"。
2. **把边界写成 acquisition 的约束,而不是加权项。** 解
   `max EI(x) s.t. l(x) ≤ C(x) ≤ u(x)`。
   `u(x)=1`(可行域内部**全部允许**探索,因为可行样本能喂目标模型);
   `l(x)` 决定**能往不可行侧探多深** —— 允许打进不可行侧一点点,才能把边界往外推。
3. **`l(x)` 随不确定度动态放宽(受 UCB 启发)。** 模型早期不准就把边界带放宽,准了再收紧。
   原文点名:朴素做法是固定 `l(x)=0.4`(留 10% 余量),但早期模型很不准,所以要动态。

**对我们的直接映射:**

| BE-CBO | 我们的对应物 | 现状 |
|---|---|---|
| 可行性分类器 `C(x)` | 共享内存是否超限 | **我们不需要学它** —— 有解析/可编译判定 |
| "往不可行侧探一点" | 我们把不可行报 PRUNED | 已有,但实测教育作用弱(见 §2.3) |
| 边界带随不确定度放宽 | 无 | 未做 |
| 最优在边界上 | 三臂获胜者都坐在实测资源墙上 | **已独立实测到** |

**最要紧的一条不对称**:BE-CBO 花大力气**学**边界,是因为它的约束是黑箱的
("a sample is considered infeasible if it fails before being able to measure its performance")。
**我们的约束不是黑箱** —— 共享内存用量可以在**编译时**便宜地拿到
(记忆 `resource-point-costs-one-thirtieth-of-a-trial`:4090 上一个资源点中位 614.6ms,
约为一个 trial 的 1/30)。所以 BE-CBO 里最贵、最需要论文来解决的那一半,在我们这儿**几乎免费**。
这是我们的**优势**,也说明**不该照搬 BE-CBO 的 Deep Ensembles**。

### 1.2 更贴我们"先找到区域再在其中优化"的:BALLET(arXiv:2307.13371)

**标题** Learning Regions of Interest for Bayesian Optimization with Adaptive Level-Set Estimation

**做法**:两个模型 —— **粗 GP** 把空间的一个**超水平集**筛成"感兴趣区域(ROI)",
**局部 GP** 只在 ROI 内做优化。有理论结果:能收缩搜索空间、regret 界比不筛更紧。

**为什么值得注意**:它的形状正是"**先用一个便宜信号定位区域,再把预算集中过去**",
和我们想做的"用斜率/墙定位区域 → 把 trial 投过去"同构。

**但有一条硬冲突,必须先说清**:BALLET 是**收缩搜索空间**。
你已经明确否决过按硬编码缩小探索空间(记忆 `never-narrow-the-search-space-to-control-cost`),
而且 `a-prescreen-timeout-is-a-cost-cap-not-a-space-restriction` 立下的区分在这儿同样适用:
**改变采样顺序/密度 = 成本分配(允许);排除取值 = 空间限制(不允许)**。
⇒ **BALLET 的思想可以借,它的"筛掉 ROI 之外"这一步不能照做**。可借的是"用便宜模型定位、
把预算倾斜过去",实现上必须是**加候选点(enqueue)而不是删取值**。

### 1.3 这条线的根:straddle 与 TruVaR

- **Bryan et al., NIPS 2005**(`external_files/lit_c2_sampler/_pdf_bryan2005.pdf`):
  最早把"找函数阈值边界"作为主动学习目标,提出基于熵/误分类率/方差及其组合的选点准则
  (后世称 **straddle**)。**这是"专门为找边界而选点"的起点**。
- **TruVaR**(NIPS 2016,arXiv:1610.07379):**把 BO 和水平集估计统一在一个算法里** ——
  贪心地收缩"潜在最大点集(BO)"或"未分类点集(LSE)"上的截断方差之和。
  原文明确说它能处理**逐点代价(pointwise costs)与异方差噪声**,并给了统一的理论保证。

**TruVaR 对我们特别相关的两点**:
1. 它就是"**优化**"与"**找边界**"的统一体 —— 正是你说的"综合而非单一指标",而且不是加权和,
   是同一个方差缩减目标下的两种点集;
2. 它原生支持**逐点代价** —— 我们的 trial 代价是**非常不均匀**的
   (记忆 `trial-cost-scales-with-kernel-latency`、`a-mean-over-a-heavy-tail-names-the-tail`:
   均值 13.5min vs 中位 1.1min),所以"每个点代价相同"的算法在我们这里本身就失配。
   **这是 TruVaR 相对其他方法的一个具体、可用的优势。**

---

## 2. TPE 能表达你说的三项中的哪几项:逐项对账

你的三项是:(A) 斜率大的参数优先、(B) 未知空间优先、(C) 高性能概率大的空间优先。

### 2.1 (C) 高性能优先 —— **TPE 能,而且这正是它唯一在做的事**

TPE 建的是密度比 `l(x)/g(x)`(好组 / 坏组),选点就是让这个比值大 ⇒ 纯粹的"往好的地方去"。
这一项不缺。

### 2.2 (B) 未知空间优先 —— **TPE 结构上做不到"显式"的**

TPE **没有后验方差**,只有密度比,所以**没有一个可以挂探索奖励的项**
(UCB 的 `+β·σ(x)` 在 TPE 里无处可挂)。它的探索性来自
`n_startup_trials` 的随机阶段 + 密度的平滑,是**副产品而非可控项**。

**这一项在我们的数据里已经量化过**(`scripts/probes/do_the_three_signals_disagree.py`,
4 run / 48 候选 / 479 个(候选,参数)行):

| | 覆盖度中位数 | 数量 |
|---|---|---|
| 有墙的参数 | **0.78** | 14 |
| 其余参数 | **1.00** | 465 |

且**覆盖度 < 1.0 的参数只有 24 个,其中 14 个(58%)有墙**。
⇒ "先测没测过的"这条**朴素规则本身就指向墙**,不需要斜率模型。
这**削弱**了"必须用斜率引导"的论证,但**不削弱**斜率作为**排序**信号的价值:
14 个有墙参数的斜率从 −359% 到 +58%,**覆盖度告诉你去哪看,只有斜率告诉你看了值不值**。

### 2.3 (A) 斜率优先 —— **TPE 完全不表达,且有两个实测限制**

**限制一:它对"取值有序"是盲的。** 我们的参数有序(BLOCK_N 16/32/64/128)但用
`suggest_categorical` 声明,TPE 默认按**无序**处理。
实测(`scripts/probes/can_tpe_express_this.py`):7 个有墙参数的**最高取值**被采到的
平均归一化位置 **0.52**(个别 0.39–0.74)—— 均匀散布,**没有"向外走"的迹象**。
⇒ 采样器不是定向逼近墙,是**撞**上去的。

**这一项有一个具体可用的修法,而且前提已经验证过。**
Optuna 4.9 的 `categorical_distance_func` 能给类别一个距离。但**任何**"给采样器序"的修法
(距离函数、改成整型/序数参数、GP 加索引核)都**预设了目标沿这个序平滑变化** ——
如果延迟在取值索引上是锯齿状的(16 快、32 慢、64 快),"64 好所以试 128"就是迷信。
本项目已经两次因为**不检查前提**而被坑(speed-of-light 头寸与 1/latency 的 ρ=+1.000;
边界饱和度 ρ 只有 −0.11…+0.24),所以这个前提也测了
(`scripts/probes/is_latency_smooth_along_the_order.py`):

**决定性指标 = 平均|相邻取值延迟差| / 平均|远距取值延迟差|**,p50 = **0.735**
⇒ **相邻取值确实比远距取值更相似,序是携带信息的**。
⇒ **"给采样器序"是一个有地基的修法**,不是把噪声当信号。这是本节唯一一条明确的正面结论。

**限制二:把不可行报成 PRUNED,对 TPE 的教育作用很弱。**
我们故意把硬性共享内存拒绝报 `PRUNED` 而非 `FAIL`,因为 Optuna 把 pruned 留在 TPE 模型里、
把 failed 丢掉(实测:12 个 FAIL 只剩 1 个可见,同样 12 个 PRUNED 剩 13 个)。
实测每个 pass 前半 vs 后半的不可行比例,26 个 pass:**12 个下降、5 个持平、9 个上升**。
⇒ 这个通道**没有**明显把采样器推离不可行区。
⇒ **一个显式的约束模型是真实增益,不是对已有机制的微调。**

**还有一个必须先解决的冲突**:Optuna 4.9 内建的 `constraints_func`
(TPESampler / GPSampler / NSGAIISampler 都支持,我们一个都没用)文档明确说
**不会对 pruned 或 failed 的 trial 调用**。
⇒ 「用内建 constraints_func」与「我们把不可行报成 PRUNED」**互斥**,不能同时是机制。
选了前者就必须改回 FAIL 或改成"可行但记违约量"的报法。

### 2.4 小结表

| 你的要求 | TPE 现状 | 判断 |
|---|---|---|
| (C) 高性能优先 | 密度比 `l/g`,**这就是它** | ✅ 够用 |
| (B) 未知空间优先 | 无方差项,靠随机启动的副产品 | ⚠️ 不可控,但朴素覆盖度规则已能找到 58% 的墙 |
| (A) 斜率优先 | **完全不表达** | ❌ 缺;`categorical_distance_func` 是有地基的修法(序携带信息 0.735) |
| 硬约束(资源墙) | 报 PRUNED,实测未推离 | ❌ 通道弱;内建 `constraints_func` 与 PRUNED 报法互斥 |

**回答"TPE 能不能解决"**:**不能完整解决**。它做好了(C),(B) 不可控,(A) 与显式约束**结构性缺失**。
但**不建议直接换掉 TPE** —— 理由见 §3.4。

---

## 3. 有没有更好的算法:候选、代价、以及为什么我不推荐大改

### 3.1 四个候选

| 方案 | 能补上 | 代价 / 风险 |
|---|---|---|
| **① TPE + `categorical_distance_func`** | (A) 的一半:给采样器"序",让它能向外走 | **最小**。前提已验证(0.735)。不改预算、不删取值、不换算法 |
| **② TPE + 斜率引导的 enqueue**(= 已写的 S7) | (A) | 中。改变 trial 分配 ⇒ **run 间不可比**;需对照臂(又一对 12h) |
| **③ 显式约束模型**(GPSampler + `constraints_func`) | 硬约束 | 中高。**与 PRUNED 报法互斥**,须改报法;GP 在类别空间上弱 |
| **④ TruVaR 式统一 BO+LSE** | (A)(B)(C) + **逐点代价** | **最高**。要自己实现;我们的空间是类别型而它是 GP 框架 |

### 3.2 一个不该忽视的负面证据

`boundary-saturation-does-not-predict-headroom` 已经实测过:**调研推荐的最强"哪里有空位"信号**,
五种读法 ρ 全在 −0.11…+0.24,低于 incumbent 的 0.43/0.52,且与 K 扩展**负相关**。
⇒ **不能假设"斜率是好的分配先验"**。这正是 S7 的 P4 要测的东西,也是我不推荐从 ① 直接跳到 ④ 的原因。

### 3.3 新颖性:我们还能宣称什么

**不能宣称的**:
- "最优解在可行/不可行边界上" —— BE-CBO(ICML 2024)已发表,且是它的核心观察;
- "用水平集/边界主动学习定位区域再优化" —— Bryan 2005 / TruVaR 2016 / BALLET 2023 已覆盖。

**检索未找到先例的**(=可能可守):
- **把实测的"参数斜率被资源上限截断"这件事,用来引导算子的结构改写**(而不是引导采样)。
  检索"敏感度引导 acquisition"零结果;最近的"重要性引导 BO"是 MetaSHAP(arXiv:2512.19246,
  ICTAI 2025),但它的重要性来自**跨 900 万条历史 pipeline 的 meta-learning**,
  **不是本次调参的实测斜率**,且它引导的是**超参搜索**,不是**改写被优化对象本身的结构**。
- ⇒ **C2 真正独特的地方不是"引导采样",而是"引导结构改写"** —— 前者文献密集,后者没找到先例。
  这对论文定位有直接影响:**把 C2 写成"采样器改进"会撞进一个拥挤且成熟的领域;
  写成"用调参反馈驱动结构搜索"才是我们的地盘**(也正是 paper 标题本来的说法)。

### 3.4 建议的次序(基于上面的证据,不基于偏好)

1. **先做零风险的一条:把墙文本发给所有族的改写者。**
   现状覆盖 25%(4 族 / 2 有墙 / 1 被投递),3/4 个族改写时零墙信息。
   不改变"测什么",只改变"谁看到" ⇒ 可比性风险等同于原 2e,且能把 A3 的样本从 n=1 扩大。
2. **再做 ①(`categorical_distance_func`)。** 它是唯一"前提已实测成立(0.735)、
   不动预算、不删取值、不需对照臂"的改动。**它也顺带修了限制一**。
3. **② / ③ 需要你批准** —— 都会破坏 run 间可比性(② 改分配、③ 改不可行报法),各需一对 12h 对照。
4. **④ 不建议本轮做** —— 实现成本高,且 §3.2 的负面证据说明"斜率是好分配先验"这个前提
   **尚未被证实**;应该先用 ② 的 P4 测出这个前提再谈。
5. **无论走哪条,报告都必须保留当前负结果** —— "2e 只进 prompt 时无端到端收益"是已测事实;
   改进版若成功,它就是对照。

---

## 4. 检索的边界(哪些没查到,以免把"没查"当成"没有")

- 本机 **WebSearch 不可用**(模型不支持该工具),**WebFetch 拒 arxiv.org**,
  **curl 到 export.arxiv.org 超时** ⇒ 只用了 arXiv 网页检索端点 + NeurIPS proceedings。
- arXiv 网页检索是**全字段 AND 精确短语**匹配,**召回偏低**:
  "autotuning GPU kernel Bayesian optimization" 只返回 1 条,而这个领域实际有
  Kernel Tuner / GPTune / ytopt / OpenTuner 等一批工作 ⇒ **GPU 自动调优这一支没有查全**。
- **未查**:OpenTuner 的 AUC-Bandit 多技术集成、SMAC3 的随机森林+EI 在类别空间的表现、
  BoTorch 的 `qNEI` + outcome constraints 实现细节。这三个都是**具体可查**的,只是本轮没做。

## 5. 本文引用的自有数字的出处与复跑方式(重要)

**BE-CBO / BALLET / TruVaR / Bryan 的每一句引文都已逐字核对**过抓到的原文
(`external_files/lit_c2_sampler/`:`l(x)=0.4`、"10% margin"、`u(x)=1`、
"we are the first to propose the use of Deep Ensembles" 均在 `_c_becbo.txt` 中定位到)。

**但本文引用的"我们自己的数字"(0.735、0.52、26 pass 的 12/5/9、479 个知识行、
14/24 覆盖度、25% 族覆盖)都产自上一段会话在三臂 `events.jsonl` 上的实跑,
而那三份日志不在本机** —— 本机 `.tmp-arms/` 下最大的两份是 9/8–9/9 的早期 run
(`run-l3-48-20260909-115701`、`run-l3-21-20260908-232211`),**不是** 9/13 的三臂。
⇒ 这些数字**本轮未在本机复核**。要复核,须先取回日志再跑:

```
python scripts/probes/is_latency_smooth_along_the_order.py  src arm1=<run_dir> arm2=... arm3=...
python scripts/probes/can_tpe_express_this.py               src arm1=<run_dir> ...
python scripts/probes/do_the_three_signals_disagree.py      src arm1=<run_dir> ...
```

(三份日志在 box1 / box4 的 `runs_dir` 与远端备份里;**scp 前先在那边 `git status --porcelain`**,
并注意 `runs_dir` 下每个 sandbox 的 `opencode.json` 含明文 API key,打包时必须排除。)

