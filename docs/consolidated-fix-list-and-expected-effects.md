# 修复列表(重新梳理)与预期效果

来源:三个外部纯 CUDA 算子的同卡对照(`finding-external-cuda-vs-our-triton.md`)+ 为方案补做的
六组实测 + 两轮代码审计(`p5-detail-and-hard-constraint-audit.md`)。本文是**执行清单**,
诊断细节在那两份文档里。

## 排序依据(以及一个实测出来的事实)

排序 = **风险 × 影响面 ÷ 成本**,其中"风险"特指**排除或伪造掉合法优化手段的可能性**。

我核实了一个影响排序的事实:`_contract_doc()` / `_triton_pitfalls_doc()`
(`agents/modules.py:33-46`)**没有缓存**,每次 agent 调用都 `read_text` 重读;两台机器都是
editable install(`import kernel_optimizer` 指向 `src/`)。所以**契约与 prompt 的改动对正在运行的
实验立即生效**,代码侧的 driver 改动则必须重启(见 `opop-v2-worker-vs-driver-fix-propagation`)。

当前两台机器 GPU 都空闲、无实验在跑,所以这个区分**这次不构成约束** —— 10 项都可以在下一轮开跑前
做完。它只在"下一轮跑起来之后又想改"时才重要。

---

## 第一梯队:契约与代码实际规则不一致(纯措辞/配置,极低成本)

这四项的共同点:**agent 正在一个与真实规则不符的世界里工作**。修法都不是"放宽"或"收紧",
而是让文字与代码强制的东西对齐。

### F1 — 契约写明 `try` / `except` / `pass` / `threading` 被禁

**问题**:`worker_main.py:1470` 不传 `forbidden`,KernelBench 的 STRICT `code_bypass` 生效,纯正则
禁掉任何 `try:`/`except`/`pass`/threading;`_strip_comments` 不剥字符串,所以**字符串里的 "pass"
一词就足以拒掉整个候选**。契约通读全文零处提及。

**改动**:契约 "Correctness and honesty" 段加四行,连**原因**和**字符串陷阱**一起写。

**预期效果**:
- 消除一类**指控与事实无关**的失败 —— 今天一个写了 `try` 的候选收到的错误是 "inheritance bypass"。
- 省下发现它所需的整个 repair 轮次(实测每轮 agent 调用 ~2-30 min)。
- **注意这是预防性的**:实测 20 个真实候选 0/20 命中,这道门至今没开火过。所以短期收益是
  **消除一个尾部风险**,不是回收已知损失。诚实地说,它排第一是因为成本几乎为零而后果很难看,
  不是因为它正在流血。

### F2 — 放宽"torch 算子只允许 layout/reshaping"

**问题**:`candidate_contract.md:145` 把 torch 算子的合法用途窄化为布局操作,等于禁止调 cuBLAS 做
计算。而代码侧 `torch_computation_ops` 在 KernelBench 里是 **WARNING**,从不拒绝 ——
**只有我们的文字在禁**。

**改动**:改成权衡说明 + 保留硬底线(必须有自己的 kernel、必须承担声称优化的部分),并要求在
`approach_summary` 里说明选了哪边。**不能**反向变成"GEMM 一律用 cuBLAS"。

**预期效果**:
- 打开一条实测有价值的路:L3:43 strict ieee 下 cuBLAS 比我们手写 Triton GEMM 快 **1.33x**
  (7.551 vs 10.063 ms),该任务 85.7% 的 FLOP 在这两个 projection 上。
- **上限有限,要说清楚**:tf32/fp16/bf16 上我们的 Triton 已是 cuBLAS 的 1.04–1.05x、达实测屋顶
  87–93%,那里几乎没有可捡的。所以这一项的收益**集中在 fp32 类任务**,不是普遍加速。
- 风险:agent 可能偷懒把计算全交给 torch。由 F5(warning 进报告)提供可见性,硬底线提供下限。

### F3 — `Prefer triton` / `prefer tf32` 改成带依据的选择

**问题**:两处无条件祈使句(`:141` 和 `:117-119`)。已记录 35/35 候选全 Triton 是
"compliance, not a preference"。本次实测给了具体反例:strict ieee 下手写 CUDA 的 attention 达
fp32 屋顶 **55–74%**,我们的 Triton 只有 **18%**,且**全部 36 点 tile sweep 都无法弥补**
—— `tl.dot(input_precision="ieee")` 在这张卡没有快路径。

**改动**:说明默认从 Triton 开始的**理由**(自带 profile 元数据、迭代快),以及何时该选 CUDA
(需要 strict IEEE 的 dot、需要 Triton 表达不了的 warp 原语)。第二处改成"低精度通常是最大杠杆,
但由调参器用实测决定,不要在源码里预先定死"。

**预期效果**:后端选择第一次成为一个**有证据的决定**而非默认值。诚实的预期:大多数任务仍会选
Triton(那通常是对的),这一项的价值是让"选 CUDA"在正确的场合成为可能,并让论文里的
"35/35 全 Triton" 不再是一个由 prompt 造成的假发现。

### F4 — `configs/default.yaml` 显式写出 `correctness_mode`

**问题**:字段默认是 `strict`(`config.py:223`,1e-4 全张量 allclose),而契约自己说 tf32 是最大
杠杆 —— 三个 L3 任务的两精度噪声底(0.9554/0.9767/0.9778)全部低于这个门。5 个实验 yaml 都
显式覆盖了,**但 `default.yaml` 完全没这个键**(实测 grep 计数 0),而 `load_config` 只读一个
yaml、无 base 层,省略即静默回落。

**改动**:在 `default.yaml` 里显式写出这个键。**不改字段默认值本身** —— strict 对 L1 那类纯
elementwise 任务是正确的门,一刀切改掉是另一个错误。

**预期效果**:消除一个静默陷阱。正式实验不受影响(都已覆盖),受益的是下一个基于
`default.yaml` 起新配置的人。

---

## 第二梯队:P5 — 让 shared-memory 可行性由编译器判定(代码,中等成本)

**问题**(实测):agent 手写的 dtype-aware shared-memory 约束在 L3:43 上**放行全部 36 个配置、
一个都没拒**,与真值一致率 53% = "全部放行"基线。手算值中位数只有真值的 32%(最差 12%)。
根因是结构性的:`metadata.shared` 含编译器自己决定的多缓冲/对齐/中间张量,**且不单调**,
从源码算不出来。而 `agents/modules.py:645-687` 仍在教 agent 算。

**修法**(上一版"前移进 `guard_ok`"已被实测否决 —— 单次 probe 16.7s 对上它要替代的 18.6s 浪费
trial,几乎不省,还会让一次 `ask()` 最坏卡 17.8 分钟):

- **F5a** job 接口从"一个配置"改成"一批"(`jobs.py:62-70` + `worker_main.py:497` 包 for 循环)。
  实测批量化把边际成本从 16.7s 降到 **7 ms(73x)**。
- **F5b** `guard_ok` 两级:先 `check_config`(微秒),再查 shared-feasibility 缓存;miss 时批量
  预取 ~16 个。
- **F5c** 保留 `orchestrator.py:1085` 现有 screen 作兜底(anchors/witness/缓存复用不经过 `ask()`)。
- **F5d** 撤下手算教学:删掉"字节宽随精度变"和那个约束模板,保留"一 kernel 一约束"和"从 body
  推导"(对寄存器/线程数仍有效),换成"shared memory 由 harness 在编译期判定,不要为它写约束"。

**为什么不能穷举预筛**:实测真实空间里影响 shared 的子网格最大 **60 万点**(7ms × 600k = 70 min)。
但 `trials_per_space` 只有 40,搜索只碰网格极小一角 —— 所以是**按需批量**。

**预期效果**:
- 回收 trial 预算:历史 **180/1004 trial(18%,0.93h)**死在 shared memory。P1 之后这些不再死在
  启动时,但仍花掉 materialize + 探测;F5 把它们移到 ask 阶段的缓存命中(微秒级)。
  L3:43 那 36 个配置里 **17 个(47%)**应在采样阶段被排除。
- **更重要的是消除认知污染**:一条恒真的约束让人以为这一维被保护着,这正是"一套 tile 通吃所有
  精度"被埋很久的原因。修完后 `infeasible_shared_memory` 的计数成为该维度的**真实体温计**。
- **验证判据(必须能失败)**:重放那 36 个配置应拒 17 放行 19,与 `confusion.py` 真值列逐行一致;
  **反向对照**是把设备上限设成 1e9 应 0 拒绝 —— 没有它,"永远拒绝"和"永远放行"两种坏实现都会
  看起来通过。

---

## 第三梯队:让已有信息浮现出来(纯读侧,低成本)

这三项**不改变搜索行为**,只让已经在 events.jsonl 里的事实进入报告。它们的价值是让下一轮的判断
建立在可见证据上。

### F6 — 报告按精度分层

**问题**:`report.py:717` 只有一行全局 total/complete/failed;精度只在 best candidate 层面出现
(`:544-545`)。所以"θ_best 有两个自己声明的精度根本跑不起来"完全不可见 —— 它表现为一堆混在
一起的 failed trial。

**改动**:(a) 渲染 per-precision 的 trial 表(数据已全在 events.jsonl,不需改 worker);
(b) 接上 `STATS_DONE` 渲染 —— `stats.py:170-196` 的 failure cluster **本来就能**发现
"某精度下全失败",只是 report.py 不读它。

**预期效果**:重放 L3:43 的 run,报告里应出现 "tf32 complete=0" 这样一行。这是**可用旧 run 立即
验证**的一项,不必等新实验。

### F7 — fp64 rescue 计数进报告

**问题**:`fp64_rescued_trials` 进 events.jsonl(`orchestrator.py:1110`)但**不进 report.md**、
**不影响排名**(写入后全项目无读者)。

**证据**:我们 L3:48 的 1.411 ms 靠 fp64 相对门 **5/5 全部救回**;外部 CUDA 1.477 ms **0 次救回**、
直接过主门。同速下他们更准 —— 而这个差别我们自己的报告里一个字都没有。

我实测了"换 bf16 能否既保速又免救回",结论是**不能**(bf16 直接 0/5 失败,`USE_EXP2` 与 bf16 尾数
交互不良)。**所以 fp16 在那里是必要的,救回是真实代价而非缺陷 —— 这恰好说明它必须被报告,
而不是被调掉。**

**改动**:best-candidate 段落加一行 rescue 计数,`rescued == correctness_trials` 时明确标注
"完全依赖 fp64 相对门通过"。**不改自动接受逻辑**(正确性决定接受,这条不动)。

**预期效果**:一个需要人知道的权衡变可见。避免下一次拿 1.411 ms 去和别人的 1.477 ms 比较时
**漏掉数值代价**这一半。

### F8 — `torch_computation_ops` warning 进报告

**问题**:`static_warnings` 被记录(`correctness.py:167, 212`)但**全项目零读者**。

**改动**:让这个 warning 显示在候选行。**只显示,不拒绝** —— 升级成 strict 就回到 F2 的错误。

**预期效果**:F2 的配套安全网 —— 放宽 cuBLAS 措辞后,"这个候选把多少计算交给了 torch 算子"
成为可见事实,人可以判断它是好权衡还是偷懒。**F2 和 F8 应该一起上**。

---

## 第四梯队:清理不一致(低风险,极低成本)

### F9 — 删掉 `tilelang`/`cute` 死分支

`Backend = Literal["triton", "cuda"]`(`core.py:18`)而 `worker_main.py:1877` 分支在
`("triton", "tilelang", "cute")` 上 —— **类型系统不允许后两个值到达那里,是死代码**。

**改动**:删死分支,契约里明说"本 harness 目前只支持 triton 和 cuda"。**不建议现在扩后端** ——
CUTLASS 依赖根本没装进 worker venv,扩类型只会把编译失败伪装成候选缺陷。

**预期效果**:消除一处"看起来支持其实不支持"的假象。这也是把 D3 从待决项**关掉**(选择不支持
并说明),而不是继续悬着。

### F10 — 显式传 `forbidden` 清单 + 注释

`worker_main.py:1470` 静默接受 KernelBench 的默认 `STRICT_CHECKS`。传入我们要的清单(即使就是
当前默认),注释说明为什么这四项 strict、那四项 warning。

**预期效果**:防一次**静默的口径漂移** —— 上游改动 `STRICT_CHECKS` 时我们的接受标准不会跟着变
而无人知晓。与 F1 同源,建议一起做。

### F11 — accumulator 语气降级

`candidate_contract.md:134-139` 用 REQUIRED/MUST 语气,但代码只有**不阻断的 warning**
(`triton_lint.py:337-339` 明说 "WARNING only (never blocks)")。物理上这条几乎总对,
所以风险低;但语气与代码不符,且理论上存在合法反例(短归约 + bf16 累加)。

**改动**:降为强建议 + 给判据("除非归约很短且 diff-test 证明够准")。

---

## 明确不做

| 项 | 理由 |
|---|---|
| **`torch.compile` 禁令** | 论证最充分的一条硬约束:注释自己承认是 blanket 且会误伤"真 kernel 旁的 compile glue",然后给出理由 —— 160 个候选只有 2 个命中(都在废弃 run),270 个 KernelBench 参考 0 个使用,**假阳性类为空**;而真阳性类不空(L3:21 那个把 no-op copy kernel 焊在 compiled graph 上、立刻成为 run 最优)。**不要动。** |
| **把 `code_bypass` 从 strict 降级** | 它防的两种作弊真实存在。要修的是"契约没说"(F1)和字符串误报,不是这道门。 |
| **"必须有自定义 kernel"** | 这正是 F2 里那条硬底线,且 KernelBench 的评测目标就是写 kernel。 |
| **把 cuBLAS 设成 GEMM 默认** | tf32/fp16/bf16 上我们已在屋顶 87–93%,强制换库会丢掉融合机会 —— L3:21 赢的那版正是靠把 BN 统计融进 producer kernel。 |
| **因 L3:48 需要 rescue 就调门限** | 实测 bf16 在那里直接失败,fp16 是必要的;门在正常工作。 |
| **继续加强 shared-memory 的 prompt 教学** | 已证伪:24/24 低估,真实约束 36/36 恒真。再写只会让 agent 更自信地算错。 |
| **改 `correctness_mode` 的字段默认值** | strict 对 L1 纯 elementwise 任务是正确的门;F4 只补显式声明。 |

---

## 执行顺序与验证判据

| # | 项 | 类型 | 成本 | 验证判据(必须能失败) |
|---|---|---|---|---|
| 1 | **F1** 契约写明 try/except/pass/threading | 措辞 | 极低 | 用 `test_bypass_check.py` 的四个反例;契约文本含"字符串"提醒 |
| 2 | **F10** 显式传 `forbidden` + 注释 | 代码 | 极低 | 传入当前默认时行为不变(0/20 候选判定翻转) |
| 3 | **F2** 放宽厂商库措辞 | 措辞 | 低 | 下轮 L3:43 出现调 `F.linear` 的候选;**且**无 kernel 的候选仍被拒 |
| 4 | **F8** `torch_computation_ops` 进报告 | 读侧 | 低 | 造一个用 `F.linear` 的候选,报告里应显示该 warning |
| 5 | **F3** 后端/精度措辞改成带依据 | 措辞 | 低 | 下轮是否出现非 Triton 候选(不强求,但要有可能) |
| 6 | **F4** `default.yaml` 显式 `correctness_mode` | 配置 | 极低 | grep 该键计数从 0 变 1 |
| 7 | **F6** 报告按精度分层 | 读侧 | 低 | **重放 L3:43 旧 run**,出现 "tf32 complete=0" |
| 8 | **F7** rescue 计数进报告 | 读侧 | 低 | **重放 L3:48 旧 run**,best 段落出现 rescued=5 |
| 9 | **F5a-d** 批量 compile screen + 撤教学 | worker+driver | 中 | 36 配置拒 17 放行 19,逐行对齐 `confusion.py`;**反向对照**:上限设 1e9 应 0 拒绝 |
| 10 | **F9** 删死分支 + 契约说明范围 | 代码 | 低 | `grep tilelang src/` 零命中 |
| 11 | **F11** accumulator 语气降级 | 措辞 | 极低 | — |

F6、F7 可用**旧 run 重放**独立验证,不必等新实验 —— 建议先做这两项,因为它们**给后面的判断提供
可见性**。1、2、6、10、11 是机械改动。9 有 worker+driver 两侧,需重启才全生效。

---

## 全部修完后的预期效果(分三类,诚实分级)

### 会直接变快的(有实测支撑的量)

1. **fp32 类任务上的 GEMM**:F2 打开 cuBLAS 这条路,L3:43 strict ieee 下 projection 部分
   **1.33x**(节省约 2.5 ms / 单次 forward)。**只在 fp32 类任务上**;tf32/fp16/bf16 上收益接近零。
2. **trial 预算**:F5 把 18% 的 trial(L3:43 实测 0.93h)从"花钱才知道不行"变成"采样时就知道"。
   同样的 12h 墙钟能测更多真实配置。这**不直接变快**,是**同预算下多探索**。

### 会让"我们知道自己在哪"的(可见性,可能是更大的杠杆)

3. **精度维度不再是盲区**(F5d + F6):今天一个候选可以声明支持 4 个精度、实际只有 1 个能跑,
   而报告完全看不出来。修完后这件事在报告里是一行零。
4. **数值代价可见**(F7):低精度候选的 fp64 救回次数进报告,速度与精度的权衡不再只报一半。
5. **后端选择成为有证据的决定**(F3 + F9):"35/35 全 Triton"不再是 prompt 造成的假发现;
   同时诚实声明我们只支持 triton/cuda。

### 会消除的尾部风险(预防性,今天没在流血)

6. **F1 + F10**:一个手滑的裸 `pass`、一个 host 侧回退的 `try`、或字符串里的 "pass" 一词,
   今天会让候选带着一个与事实无关的指控被拒。实测 0/20 命中,所以这是**尾部风险**,
   排第一是因为成本几乎为零。
7. **F4**:`default.yaml` 起的新配置不会静默拿到一个与契约自相矛盾的正确性门。

### 明确不承诺的

- **不承诺 L3:43 会因此追上外部 CUDA 的 strict ieee 成绩。** 那里有一条真实的 Triton 短板
  (ieee `tl.dot` 无快路径,attention 只到屋顶 18% vs 他们 55–74%,36 点 tile sweep 无法弥补)。
  F2 能拿回 projection 那 1.33x,但 attention 那 3–4x 需要写 CUDA(F3 让它成为可能,不保证发生)。
- **不承诺整体加速比提升。** 我们的三个任务最优成绩(4.414 / 3.013 / 1.411 ms)已经都赢外部算子;
  这批修复主要买的是**同预算下更充分的探索**和**不再自欺的报告**,不是一个更大的数字。
- **F5 的 trial 回收不等于成绩提升。** 多探索 18% 的预算是否换来更好的 θ_best,要下一轮实测才知道。
  L3:43 重跑(对照 `docs/result-l3-43-final.md` 的 3.0126 ms)是这批修复的归因实验。

---

## 复现本文档引用的实测

```sh
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python confusion.py'    # 约束 36/36 恒真
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python probe_cost.py'   # 单次探测 16.7s
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python probe_batch.py'  # 批量 7ms,73x
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python l3_48_bf16.py'   # bf16 在 L3:48 上失败
ssh autodl  'python /root/grid_sizes.py'                             # 子网格最大 60 万点
ssh autodl  'python /root/test_bypass_check.py'                      # try/except/pass 实测被拒
ssh autodl  'python /root/why_pass_ok.py'                            # 为何 20 个候选未触发
```
