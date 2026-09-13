# Paper 修改规划:main_zh.tex 在当前 contribution 下如何调整

**日期** 2026-09-14 · **paper** `D:\Pyhon_projects\opop\paper\main_zh.tex`(820 行)· **checkout** `v3` @ `80baee0`

**本文件是规划,不改 paper。** 目标:据此可以完成一版"figure 与部分实验数据为 placeholder"的初版 paper。

**必须先读**:`memory/contribution-framework-current-three.md`(当前三条 contribution 的定义与证据状态)。
本文件的一切结论都以那份为准 —— paper 正文里现在写的三条 contribution **已经与它不一致**,这是最大的改动量所在。

---

## 0. 一句话结论:paper 现在的三条 contribution 与实测证据错位

paper 第 48–54 行现有的三条(下称 P1/P2/P3)与 memory 里的 C1/C2/C3 **不是同一套**:

| paper 现有 | 实测状态 | 处理 |
|---|---|---|
| **P1** 用完整调参过程判断受什么限制、解除能有多少收益 | 这是 **C2** 的一半。但 **C2 的"新能力"主张已被否证**:控制臂(2e 关闭、0 道已归因墙)的 analyst 在 24 个假设里有 4 个(17%)自己做出了"上限截断 + 斜率"推理,且这是 prompt 设计使然(`agents/modules.py:1059-1088`,1079 行明确要求它判断某方向是否被硬件限制阻止) | **改写为"更早/更可靠/定量更准"三个可分别测量的主张**,不能写成"让框架能做到以前做不到的事" |
| **P2** 低成本验证筛选结构修改 | 机制存在(见证门 + quick test),但**没有独立实验**。且本轮实测发现:28 个可判定改写里 **3 个(11%)在固定 knob 下六个资源维度全恒等**、3 个源码贡献为负 ⇒ **21% 无正贡献**,而框架全部接受了它们 | **保留但降级**,并**必须**报告 21% 这个负结果 —— 否则"低成本验证有效"与实测矛盾 |
| **P3** 小规模动态结构集合替代 MAP-Elites 固定网格 | 机制存在(族 + K 上限 + loop D)。**但缺一个对照**:没有跑过真正的 MAP-Elites 基线 | **保留,标注对照缺失**;若不补 MAP-Elites 基线,措辞须从"优于固定网格"降为"不需要固定网格" |
| — | **C1 资源维度对齐 + 到达物理上限** | **新增为第一条 contribution**。证据最强:两次配对同向(−8.68%、−12.0%),三臂赢家 `n_binding` 3/2/1 且无 unreachable ceiling。**但"在多个已达上限者中取最优"已被两条实测否证**,不得写入 |
| — | **C3 信息获取效率** | **无机制、无实验** ⇒ **本轮不作为 contribution 写入**;若要保留,只能是 future work 一句话 |

**⇒ 最小改动方案(建议采用)**:contribution 列表改成
**C1(资源维度对齐并到达上限)+ C2(调参斜率引导结构改写,主张更早/更可靠/更准)+ P3(动态结构集合)**,
C3 撤下,P2 并入 C2 作为其中的"低成本验证"步骤而不单列。

---

## 1. Introduction(第 37–62 行)

### 1.1 必改

1. **第 48–54 行的 itemize 整段重写**,按 §0 的三条。
2. **第 46 行**"参数调优能否不仅告诉我们一个结构最终有多快,还能告诉我们下一步最值得改变什么"
   —— 这个设问**要加限定**。基线 analyst 已经能回答它(17%)。改成
   "能否**更早、更可靠、更定量地**告诉我们",并在该句后直接给出三者是不同主张。
3. **第 37 行 abstract 的 placeholder** 与 **第 62 行**的主要结果 placeholder:
   按 §3 的四组实验填,顺序与 §3 一致。

### 1.2 新增一段(放在第 46 行之后)

**C1 需要一段独立动机**,现在 introduction 里完全没有"资源维度对齐 / 到达物理上限"这条线。
建议内容:算子性能的上界由硬件各资源维度共同决定;一个算子只有在**某个维度真正绑定**时才可能接近上限;
框架显式跟踪八个资源维度的绑定状态(`DIMENSION_STATE.binding`),使 agent 能利用**尚未绑定的空闲维度**
换取已绑定维度的松弛。**不要写**"在多个已达上限的算子中取最优" —— 已否证。

---

## 2. Background & Problem Setting(第 64–260 行)

现状:这部分**是全篇最完整的**,主体结构可保留。但有三处必须动。

### 2.1 必改

1. **第 195/213 行的两个诊断实验 placeholder(单路线 / MAP-Elites)**:
   这两个诊断实验**我们没有做过**,而且做它们需要实现 MAP-Elites 与单路线两个基线。
   ⇒ **两个选择**,须尽早定:
   - (a) 补做 —— 成本是两套基线实现 + 各自的 run;
   - (b) 改为**引用式论证**(引 KernelFoundry / K-Search 等已有报告),把诊断实验降为附录里的小规模示例。
   **建议 (b)**,理由:这两条是"现有方法的局限",不是我们的 contribution,用自己的机时去证别人的缺点性价比低。
2. **第 221 行**已标注"这段部分内容放到 method 里也可以" —— 该段(what parameter tuning reveals)
   **应当拆开**:前半(调参记录 $\mathcal{D}_S$ 的定义,式 6)留在 background;
   后半(同一限制有多种解除方式 → 需要筛选)移到 method 的 C2 小节,因为它现在与 method 重复。
3. **第 245 行的诊断实验 placeholder**(比较四种信息源对真实收益的预测能力)
   —— 这一条**保留且提升为主实验**,它正是 C2"定量更准"唯一有支持证据的主张
   ([[triton-flash-attn-shared-memory-formula]]:agent 手写共享内存约束中位只有真值 32% 且从不拒绝)。
   见 §3.3。

### 2.2 新增

**problem statement 里要加入"资源维度"这一层**,否则 C1 在 method 里会突然出现。
建议在 §2.1(algorithm structures, parameter spaces, objective)之后插一小节:
把 $\mathbf{r}_i$(现在只在式 6 里一笔带过)提升为一个显式对象 —— 八个维度、各自的硬上限、
"绑定"的定义(哪一维在最优点处触及上限)。这同时为 C1 与 C2 提供共同的记号基础。

---

## 3. Experiments(第 405–780 行):按用户给的四组重写

现状:第 412 行自己标注"绝大部分是 AI 生成的,暂时没有什么参考意义"。⇒ **整节按下述四组重建**。

### 3.0 全节通用的三条硬约束(写进 Experimental Setup)

1. **报告 `final_reeval_ms`,不报调参过程中的 best**。
   再评测差异实测 ±2–4% 且**符号不固定**([[reeval-gap-is-the-real-number]])。
2. **一切延迟比较用 median,不用 mean**。median 排名正确率 93.2% vs mean 64.8%;
   `LatencyStats.robust_ms` 是 median-else-mean。**mean 会命名尾部**(13.5 min vs 中位 1.1 min)。
3. **臂间对等必须按"总搜索量"而不是"每空间预算"对齐**。K 扩展会让一个臂多领整份预算:
   实测 8 空间 vs 6 空间 = 1.6 个空间的差距,而所有既有对等检查都通过
   ([[equal-per-space-budget-is-not-equal-total-search]])。工具:`scripts/check_arm_search_parity.py`。

### 3.1 实验一:KernelBench 综合结果(三维:level × 方法 × base model)

**设计**

| 维度 | 取值 | 备注 |
|---|---|---|
| level | 1 / 2 / 3 | **level 4 单列为 §3.4**,它是 21 个 HuggingFace 整模型任务(如 `1_EleutherAI-gpt-neo-2p7B_bs32_seq256.py`),性质与 1–3 不同 |
| 方法 | ours / KernelFoundry / K-Search / KernelBand / KernelBlaster | 见下"基线可得性" |
| base model | glm-5.3 / gpt-5.6-sol / deepseek-v4-pro | |
| 采样任务数 | 20(待定) | 见下"采样与预算" |

**⚠ 基线可得性必须先核实,这是本组实验最大的未知**
- `paper/references.bib` 里有 `wiedemann2026kernelfoundry`、`cao2026ksearch`、`gai2026kernelpro`;
  **没有 KernelBand 与 KernelBlaster 的条目** ⇒ 需要先补引用,并确认这两个是否有**可运行的公开实现**。
- 本地 `D:\Pyhon_projects\opop\kernelfoundry` 有完整代码树 ⇒ KernelFoundry 可实跑。
- 其余三个若**只能引论文数字**,则:(a) 硬件/任务/预算几乎一定不同 ⇒ **不能放进同一张表直接比**;
  (b) 处理方式是**分两张表**:一张"同硬件同预算实跑"(ours + KernelFoundry + 任何可跑的),
  一张"文献报告值"并**逐列标注其硬件与预算**。混在一张表里是最容易被 reviewer 抓的点。

**采样与预算(20 个任务的代价必须先算)**
- 已知:**8/8 跑完的 run 都由墙钟结束**,单 run 12 h 量级;`trials_per_space=40`。
- ⇒ 20 任务 × 5 方法 × 3 模型 = **300 个 run**。按 12 h/run 是 3600 GPU·h。
  **当前实际可用 GPU 只有 3 张**:box1 ×1、box4 ×2(两张 4090);
  box2 已归其他租户、A800(box3)已关机需重开。
  ⇒ 3 卡满载也需 **约 50 天**,**不可行**。
- **必须做的取舍(建议在文档评审时定)**:
  (a) 主表只做 **1 个 base model(glm-5.3)× 5 方法 × 20 任务 = 100 run**,
      base model 维度降为**单 level 上的子实验**(如 level 2,5 方法 × 3 模型 × 8 任务 = 120 run);
  (b) 或缩短单 run 墙钟(会改变所有既有结果的可比性,**不建议**);
  (c) 或减任务数到 10–12。
  **不要"按卡型收紧预算"** —— 用户已否决([[never-narrow-the-search-space-to-control-cost]])。
- **任务采样必须记录并公开采样种子与被排除任务**。**本 checkout 实测的池大小**:
  `KernelBench/KernelBench/level1` **100** 个、`level2` **100** 个、`level3` **50** 个 `.py`。
  ⇒ **level3 采样 20 个 = 40% 的池**,这个比例要在正文说明(而不是让读者以为是小样本抽检)。
  已知排除规则:forward 内含 `randn` 的任务不可用([[l3-task-selection-constraints]])
  —— **该规则命中的是 level3,不是 level1**(level1 的 100 个文件里 `randn` 匹配数为 0);
  24 GiB 卡上 **34/78** 个 level1 不可评测([[l1-oom-ceiling-is-not-the-candidates-fault]])
  ⇒ **level1 的有效池明显小于 100,采样前须先过滤并报告过滤后的池大小与过滤理由。**

**报什么**
- 主指标:**相对 PyTorch eager 的 speedup(median-based,final_reeval)**,按 level 分别给几何平均;
  外加**成功率**(通过正确性门的任务比例)—— 后者不能省,因为方法间失败率差异会让平均值失真。
- **不要报单一平均延迟**:不同任务延迟量级跨几个数量级,平均会被最慢任务支配。

**figure/table placeholder**
- `tab:kb_main`:行 = 方法 × base model,列 = level1/2/3 的几何平均 speedup + 成功率。
- `fig:kb_profile`:每个 level 一个 box plot(per-task speedup 分布),显示方差而不只是均值。

### 3.2 实验二:典型 kernel × 多显卡(C1 的主战场)

**设计**
- kernel 类型:**GEMM 类、attention 类、RNN/序列类、reduction/normalization 类、逐元素类**。
  选择判据应写明:**每类的绑定维度不同**(GEMM 偏 compute/shared;attention 偏 shared + 寄存器;
  reduction/逐元素偏带宽)。已有支持:L3:48 已达 DRAM 屋顶的 **90.4%**、911 GB/s
  ([[l3-48-is-bandwidth-bound-at-90-percent]]);L3:43 是共享内存受限的 attention。
- 硬件:**RTX 4090(box4 ×2)+ A800(box3,需重新开机)**。已知两者共享内存上限不同
  (4090 **101376 B**、A800 **166912 B**)⇒ 这正是"同一算子在不同硬件上找到各自上限"的天然对照。
- **⚠ 跨卡 trial 数不可直接比**:box1 与 box4 的 CPU 不同(8358P vs 8352V),
  compile_s +32%、job_wall +26% ⇒ 840 vs 760 trial 是硬件差异
  ([[box1-and-box4-have-different-cpus-so-trial-counts-are-not-comparable]])。
  ⇒ **跨卡比较只能比"是否到达该卡的上限"与"到达上限所用的墙钟比例",不能比 trial 数。**

**这一组要证明的正是 C1,所以指标必须是"上限"而非"速度"**
- 主指标:**达到该硬件 roofline / 已测后端天花板的百分比**。
  天花板定义须写清:**已测后端的最大值**([[a-ceiling-is-the-max-over-measured-backends]]);
  **cuBLAS 在 fp32 上会高估 19%**,所以不能只用 cuBLAS。
- 第二指标:**`n_binding`(最优点处触及上限的维度数)与是否存在 unreachable ceiling**。
  已有证据:三臂赢家 `n_binding` 分别 3/2/1 且无 unreachable ceiling。
- 第三指标:**到达上限的墙钟比例**。arm2 在 50% 处就到达最终值(3.335→2.488→2.488),
  且它在慢 24% 的 CPU 上 ⇒ 混淆方向对它不利,这是 C1 前半句最强的一条证据。

**⚠ 不得写入的主张**:"在多个已达上限的算子中取性能最优者"。
两条实测否证:[[fixed-denominator-comparison-explains-nothing]](寄存器同为 ≥250/255 的赢家延迟仍差 12.8x/2.1x)、
[[leaner-resource-use-means-less-headroom-not-more]](4/4 个 leaner 剩余增益都更少)。

**figure placeholder**
- `fig:ceiling_by_hw`:横轴 kernel 类型,纵轴"达到该卡天花板的百分比",每卡一组柱 + 各方法一色。
- `fig:binding_dims`:堆叠柱,显示每个赢家在哪些维度绑定(说明"不同硬件绑定不同维度")。

### 3.3 实验三:消融 + 逐 contribution 结果

**这一组是最需要诚实处理的,因为几条已有负结果必须出现在正文。**

#### (a) C1 消融:关闭资源维度对齐

- 开关:`DIMENSION_STATE` 相关信息是否进入 agent prompt。
- 已有配对结果:两次同向 **−8.68%(干净对比)与 −12.0%(本轮)**。
- **可直接复用已有 run**,不需要新机时。

#### (b) C2 消融:2e 开/关(已跑,且结果对我们不利,必须如实报)

- **已跑完三臂**:arm2(**无 2e**)**2.49 ms / 4.4177x** 最好、arm3(有 2e)2.69 ms、arm1 2.83 ms
  ([[step3-three-arm-result-2e-arm-came-second]])。
- **正结果只有机制层面**:被投递的 `BLOCK_N=128` 墙被改写解开,且在**该取值处**共享内存降到上限内
  (77824 / 45056..98304 vs 122880);反事实 arm2 无 2e 时 7 个墙 0 解开
  ([[a3-the-steered-rewrite-freed-the-wall-arm2-freed-none]])。
- **⇒ C2 在正文里的写法**:机制结果 + 端到端负结果并列。**不要**只报机制。
- **本轮 S7(斜率进采样器)结果**:32 次触发、**0 次投递**。
  **两臂的成因不对称,报告时必须逐臂说明**(6.4 h 时点实测):
  - **控制臂**:68 个共享内存拒绝 → 4 道墙 → 2 道过斜率 → **0 道过归因**
    (over_ratio 0.485 / 0.970)。根因是**归因射程**:被拒点与 θ\* 相差 3–10 个 knob,
    而归因做的是从 θ\* 出发的单 knob 消融(详见 `docs/next-round-changes.md` §4.5)。
  - **处理臂**:35 个拒绝、**`walls_found = 0`** —— **每个被拒值都落在该 knob 的已测范围之内**,
    按定义不构成墙(同一取值换搭档跑通后墙会消失)。**它连候选墙都没有,与 §4.5 无关。**
  ⇒ **不能把 S7 的零投递合并归因为"归因失败"**;正文应写成"机制的两个前置条件各自都会失效",
  并说明只修归因射程**不会**让处理臂这类 run 产出墙。

#### (c) C2"更早/更可靠/更准"三个主张分别测(这是 C2 能否守住的关键)

| 主张 | 怎么测 | 现状 |
|---|---|---|
| **更早** | 比较"首次可得的 trial 序号":2e 在调参结束即产出,analyst 要等完整统计 | **待测**。S7 本就是把它提前到调参内部 |
| **更可靠** | 同一处截断,analyst 命中比例 vs 2e 命中比例 | **待测**,需先有可归因的墙(本对 0 道)⇒ **依赖 §4.5 的修复** |
| **定量更准** | 墙给出精确 `over_ratio`;agent 是估算 | **唯一有证据**:agent 手写共享内存约束中位只有真值 **32%** 且从不拒绝 |

⇒ **"定量更准"应作为 C2 的主打**,它把 C2 从"新能力"降为"精度替代 agent 估算" —— 更小但站得住。
对应 §2.1 第 3 点里那个诊断实验(四种信息源的预测能力),**提升为主实验**。

#### (d) P2(低成本验证)消融:必须报 21% 负结果

- 28 个可判定改写:**3 个(11%)在固定 knob 下六个资源维度全恒等**、3 个源码贡献为负
  ⇒ **6 个(21%)无正贡献,而全部被框架接受**。
- 成因是机制级的:**接受判据是族最优的改善,而族最优由 TPE 在新空间另找的点提供**
  ⇒ 衡量改写的量里混着调参([[the-framework-only-collects-confirmations]])。
- ⇒ 正文写法:低成本验证减少了多少次完整调参(正面数字,**待测**)
  + 它**没有**过滤掉这 21%(负面,已测)。
- 引用须带:**单任务 L3:43**、**数字曾因父代认定修正而变过一次**(24%→11%)。

#### (e) P3(动态结构集合)消融

- 已有:loop C 4/4 族靠改写改善(单臂观测)、e3 loop C 增益 41%、3/4 族 45.5/41.0/1.4%。
- **缺 MAP-Elites 对照** ⇒ 若不补,措辞降为"不需要固定网格",不能说"优于固定网格"。

**figure/table placeholder**
- `tab:ablation`:行 = 完整方法 / −C1 / −C2 / −低成本验证 / −多路线,列 = 各 level 的几何平均 speedup。
- `fig:c2_claims`:三格子图,分别对应"更早/更可靠/更准"的度量。

### 3.4 实验四:多卡环境 case study

**⚠ 一处事实修正**:用户提到"kernelbench 中 level 4 的 task"。
实测 `KernelBench/KernelBench/level4/` 是 **21 个 HuggingFace 整模型任务**
(`1_EleutherAI-gpt-neo-2p7B_bs32_seq256.py`、`13_google-reformer-enwik8_bs32_seq256.py` 等),
**它们是"多 kernel 的整模型",不是"多卡"任务** —— 多卡这一层需要我们自己引入。

**⇒ 两种设计,建议二选一**
- (a) **多 kernel 单卡 case study(低风险)**:直接用 level4 的一个模型任务,
  展示框架在一个含数十个算子的真实模型上如何分配预算、哪些算子被优化、端到端加速多少。
  这与"多个 kernel"对得上,**不涉及多卡**。
- (b) **多卡(高风险,需新增能力)**:当前 harness 是**单 GPU 串行 + 文件锁**,
  计时**必须独占**(MIG 在消费卡不可用、MPS 在 WSL2 不受支持,时间片下 latency 必然互相污染)。
  ⇒ 多卡意味着**新增张量并行/流水并行维度**,这是一个新的结构空间,不是配置改动。
  **本轮不建议**,除非用户明确要这条。

**若选 (a),报什么**
- 端到端模型延迟(eager / torch.compile / ours),外加**逐算子表**:哪些算子被改写、各自 speedup、
  以及**未被优化的算子占比**(诚实点:框架不会覆盖全部算子)。
- **task 级融合头寸**已实测是最大杠杆(L3:43 达 **69.09x**,[[task-cost-fusion-headroom]])
  ⇒ 整模型任务上这一条会最显眼,值得作为 case study 的主线。

---

## 4. Method(第 260–405 行)

现状:第 260 行自标"整个 method 部分都还是 placeholder"。**结构可用,内容需按 C1/C2 重排。**

### 4.1 必改

1. **加入 C1 的一小节**(现在完全没有)。位置:`Overview` 之后、`Finding a Promising Parameter Target` 之前。
   内容:八个资源维度、绑定判定、"用空闲维度换已绑定维度"的动作。
   **注意极性**:occupancy **低**才算绑定,且极性必须显式声明
   ([[occupancy-is-nested-and-a-flat-read-fakes-unmeasured]])。
2. **C2 小节要写清两道门**,现在正文只写了一道。
   墙必须过:(1) 斜率过滤器(`monotone and tail_gain_pct > 0`);(2) **归因**(在过滤器之后跑)。
   实测 33 个归因事件 → 8 道墙 → 4 道过斜率 → **1 道被归因**。
   ⇒ 正文若只写"找到截断就引导改写",会与实验数据矛盾。
3. **删掉或重写"寄存器超限"这个贯穿全文的例子**(第 37、46、266 行等多处)。
   实测:**八个维度里只有 `shared_bytes` 有拒绝判据、能产生硬墙**;
   Triton 上**寄存器会被上限截断而不是 spill**,`n_spills` 是**软墙**需第二种判据
   ([[only-shared-memory-produces-a-wall-of-the-eight-dimensions]]、
   [[triton-caps-regs-instead-of-spilling]])。
   ⇒ 例子应换成**共享内存**,否则方法描述与实现不符。
4. **式 6 里的 $\mathbf{r}_i$ 要与 §2.2 新增的资源维度小节统一记号**。

### 4.2 建议新增(可选,取决于是否补 §4.5 的修复)

若下一轮实现了"多 knob 联合消融"(`docs/next-round-changes.md` §4.5),
method 里应写成:归因不是从 θ\* 单 knob 消融,而是**从被拒点向 θ\* 逐维回退求最小维度集合**。
已量代价:**每层一批合并所有墙 = 123 秒**(每墙一批则 1.21 h,不批量 45.1 h)。
**若本轮不实现,method 就按现状(单 knob)写,并在 limitation 里点明射程不足。**

---

## 5. 需要用户决定的事项(按紧急度)

1. **base model × 方法 × level 的全交叉不可行(300 run / 3600 GPU·h)。** 采用 §3.1 的哪个缩减方案?
2. **KernelBand / KernelBlaster 是否有可运行实现?** 若只能引数字,是否接受"实跑表 + 文献表"分两张?
3. **多卡 case study**:选 §3.4 的 (a) 多 kernel 单卡,还是 (b) 真多卡(需新增并行维度,本轮不建议)?
4. **§2.1 的两个诊断实验(单路线 / MAP-Elites)**:补做,还是改为引用式论证?
5. **是否补 MAP-Elites 基线?** 不补则 P3 措辞须降级。
6. **C3 是否彻底撤下?**(当前无机制、无实验)

---

## 6. 写初版 paper 时可立即使用的既有数据(不需新机时)

| 用处 | 数据 | 来源 |
|---|---|---|
| C1 消融 | −8.68% / −12.0% 两次同向配对;`n_binding` 3/2/1 | memory C1 行 |
| C2 端到端负结果 | arm2 2.49 ms/4.4177x > arm3 2.69 ms > arm1 2.83 ms | [[step3-three-arm-result-2e-arm-came-second]] |
| C2 机制正结果 | `BLOCK_N=128` 墙被解开,77824 vs 122880;反事实 7 墙 0 解开 | [[a3-the-steered-rewrite-freed-the-wall-arm2-freed-none]] |
| C2"更准" | agent 手写共享内存约束中位仅真值 32%,且从不拒绝 | [[triton-flash-attn-shared-memory-formula]] |
| C2 两道门的漏斗 | 33 事件 → 8 墙 → 4 过斜率 → 1 归因 | `scripts/probes/wall_event_vs_verdict.py` |
| 改写 21% 无正贡献 | 28 可判定:3 恒等 + 3 负 | `scripts/probes/rewrite_source_vs_knob.py` |
| 单卡上限 | L3:48 达 DRAM 屋顶 90.4%(911 GB/s);同精度 9.29x | [[l3-48-is-bandwidth-bound-at-90-percent]] |
| 三任务最终结果 | L3:21 6.92 ms/2.18x;L3:43 3.0126 ms/3.647x;L3:48 1.41 ms | 三条 result memory |
| 外部对比 | 三任务我们都赢(L3:48 平手);手写 CUDA 同算法 0.874x | [[external-cuda-vs-triton-verdict]] |
| loop C 有效性 | 4/4 族靠改写改善;e3 增益 41%,3/4 族 45.5/41.0/1.4% | 两条 memory |
| 噪声底 | 再评测 ±2–4% 且符号不固定;每 trial std 16% | 两条 memory |

---

## 7. 一条贯穿全文的写作纪律

**每一处引用数字都要带"在什么条件下测的"。** 本项目里已经有多次因省略条件而失真的先例:
- "改写改善 X%"若不说明是**固定 knob 匹配点**还是**两侧各自最优点**,方向可能相反;
- trial 数跨箱不可比(CPU 不同);
- 事件计数不等于结果计数(`RESOURCE_WALL_ATTRIBUTED` 是**步骤名**,数它会把原材料夸大 33 倍);
- 墙钟漂移的比例须带时点。

⇒ 建议在附录加一张**"每个数字的测量条件"表**,而不是在正文各处零散加脚注。
