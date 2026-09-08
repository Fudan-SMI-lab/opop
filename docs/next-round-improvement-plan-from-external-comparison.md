# 下一轮改进方案(基于 L3:21 / L3:43 / L3:48 三个外部对照案例)

三个外部纯 CUDA 算子的对照测量(`finding-external-cuda-vs-our-triton.md`)暴露的问题,加上为写这份
方案而补做的四组实测。**每一条都带本机实测判据,并且都是泛化改动 —— 没有针对单个 task/candidate 的
特判或硬编码。**

## 先说结论:最大的问题不是我们以为的那个

我原本准备写的第一条建议是"教 agent 更好地计算 shared memory 约束"。查了代码之后这条作废:
`agents/modules.py:645-687` 里**已经有**极其详尽的指导 —— 一个 kernel 一条约束、字节宽随精度 knob
变化、从 kernel body 推导驻留量、以及正是 L3:43 该用的那个析取语法模板。agent 也确实照做了:
18 个 SPACE_PUBLISHED 里绝大多数都写了 dtype-aware 的 shared-memory 约束。

问题是**这些约束算出来的数字全是错的**。用 guard 自己的求值器跑 L3:43 `cand-969997e3`
**真实声明**的那条约束(逐字取自 events.jsonl),在 36 个 tile×精度配置上与编译期真值对照:

| | 真实能装下 | 真实会死 |
|---|---|---|
| 约束放行 | 19 | **17** |
| 约束拒绝 | 0 | **0** |

**该约束放行了全部 36 个配置,从未拒绝任何东西 —— 它等于不存在。** 与真值的一致率 53%,恰好等于
"全部放行"的基线。手算值中位数只有真值的 **32%**(最差 12%);换另一个 agent 在 L3:48 上写的同型
约束交叉验证,16 个点里 10 个低估,中位数 0.82、最低 0.28。

原因是结构性的,不是 agent 不够仔细:Triton 的 `metadata.shared` 包含多缓冲、对齐、softmax 中间张量
等一切编译器自己决定的分配。**从源码手算这个数字在原理上就做不到。** L3:43 的
`ATTN_BLOCK_M=128, BLOCK_N=64, stages=2` 真值 98304,而任何"按 tile 元素数乘字节宽"的公式都给不出
这个数 —— 它甚至不随 BLOCK_M 单调(128/32/1 和 128/64/1 都是 65536)。

这直接改变了方案的重心:**不要继续教 agent 算,要把这件事从 agent 手里拿走。**

---

## P5(最高优先):用编译期真值取代手写 shared-memory 约束

> **本节的"怎么修"已被 `p5-detail-and-hard-constraint-audit.md` 取代 —— 先读那一份。**
> 实测否决了下面"直接前移进 `guard_ok`"的做法:一次 compile probe 要 **16.7s**(按 harness 真实
> 的一 job 一进程方式),而它要替代的浪费 trial 是 18.6s —— 几乎不省时间,还会让一次 `ask()` 最坏
> 卡 **17.8 分钟**(64 次拒绝上限)。正确做法是**批量化**:48 个配置一个进程共 11.02s,
> **边际 7 ms,便宜 73 倍**。诊断(手写约束不可用)不变,手段变了。

**现状**:`guard` 在 `ask()` 内部拒绝(`tpe.py:69-71`),不产生 TrialRecord;而 P1 的 compile screen
在 materialize 之后、quick_test 之前(`orchestrator.py:1085`),拒绝时产生一个
`infeasible_shared_memory` 的 record 并被 tell 成 PRUNED。所以今天的实际行为是:约束形同虚设 →
trial 被 ask 出来 → materialize → compile screen 拦下 → 记 PRUNED。P1 已经止住了"死在启动时"的血,
但每个这样的 trial 仍然花掉了一次 materialize + 一次编译探测。

**改动**:把 compile screen 的结论**前移进 `guard_ok`**,即 `ask()` 的拒绝回路里。
`orchestrator.py:1010` 现在传的是
`guard_ok=lambda p: check_config(space, p, self.cfg.device) is None`;改成先跑 `check_config`
(便宜、纯算术),通过后再查一个**按 materialized source 缓存**的编译探测(`correctness.py` 的
`_screen_cache` 已经是这个形状,`compile_screen` 也已经只在"编译器自己报的数字超限"时才拒绝)。

**为什么这是泛化改动**:它不引入任何关于 tile 或 dtype 的知识,只是把"编译器说装不下"这个事实用在
采样阶段而不是评测阶段。对任何 kernel、任何后端、任何精度都成立。

**收益量化**:L3:43 那 36 个点里 17 个(47%)本可以在 ask 阶段就被排除。历史数据是 180/1004 个 trial
(18%,0.93h)死在 shared memory 上。

**风险与守护**:编译探测比算术贵。所以 (a) 必须缓存(同一 materialized source 只探测一次 —— space
expansion 后的 re-tune 会反复 ask 同一配置);(b) `ask()` 的 `max_guard_rejects_per_ask=64`
上限要按探测成本重新定,否则一次 ask 最坏情况会跑 64 次编译;(c) 探测不可用时必须放行
(`compile_screen` 现在就是这个语义,"never be the thing that rejects a candidate"),否则一台没有
探测能力的机器上整个搜索会静默停摆。

**同时要做的事**:既然手写约束在这一维上不可信,`agents/modules.py:645-687` 那一大段
shared-memory 约束教学应当**删掉或降级**。它现在的净效果是让 agent 花 token 写出一条恒真的约束,
并让读报告的人(包括我)误以为这一维被保护着。保留 (a)"一个 kernel 一条约束"和
(c)"从 kernel body 推导"这两条对**寄存器/线程数**约束仍然有效的部分,把 shared memory 明确交给
harness:告诉 agent "shared memory 的可行性由 harness 在编译期判定,不要为它写约束"。

---

## P6:报告与搜索都必须按精度分层

**现状**(子代理核查):`report.py` 的 trials.csv 有 `params` 列(JSON 原文,COMPUTE_DTYPE 在里面),
但 trial 统计只有一行全局聚合(`report.py:717`:total/complete/failed)。精度只在 best candidate
层面出现(`report.py:544-545`)。**没有任何按精度分组的统计。**

这就是为什么 L3:43 的 θ_best "有两个自己声明的精度根本跑不起来"这件事,在整个 run 的报告里
完全看不见 —— 它表现为一堆 failed trial,和别的失败混在一起。

**改动 A(便宜,先做)**:report.py 渲染一个 per-precision 的 trial 表:每个 COMPUTE_DTYPE 取值的
trial 数 / complete 数 / 该精度下的最优 ms。这需要的数据**已经全在 events.jsonl 里**,不需要改
worker 或 orchestrator。判据很直接:如果某个精度取值的 complete 数是 0,报告里就会有一行零,
而今天它是不可见的。

**改动 B**:`tuning/stats.py` 里已经有 `failure_rate_by_value`(stats.py:86-93)和
`_failure_clusters`(stats.py:170-196),后者的触发条件正是"某个 (param, value) 的失败率 ≥
max(2×总体, 0.5)" —— 它**本来就能**发现"COMPUTE_DTYPE=tf32 下全失败"。这些数据经 `STATS_DONE`
写进事件日志,但 **report.py 不读 STATS_DONE、不渲染**。接上这一条渲染,是纯读侧改动。

**改动 C(需要设计)**:报告 per-precision 最优,而不是单一 θ_best。一个候选在 4 个精度上声明支持、
实际只有 1 个能跑,今天的 `final_reeval` 只测 θ_best(`benchmark.py:106-110`),这个事实不会浮现。

---

## P7:契约要允许"该用厂商库的地方就用厂商库"

**现状**:`candidate_contract.md:145` 写的是

> torch operations are allowed around the custom kernel(s) (**layout, reshaping**),
> but the core computation you claim to optimize must run in your kernel.

括号里那两个词把 cuBLAS 排除在**计算**之外。grep 整个 prompts/ 目录,`cuBLAS`/`cublas`/`vendor`
**零命中**。

**证据**:外部 CUDA 算子在 L3:43 上的两个 projection 就是 `at::linear`(即 cuBLAS),我们是手写
Triton GEMM。这占该任务 85.7% 的 FLOP。在 strict ieee 下这个选择本身让我们多花约 2.5 ms
(我们 10.063 ms vs cuBLAS 7.551 ms,1.33x)。

**注意不要过度推广**:在 tf32/fp16/bf16 上我们的 Triton GEMM 是 cuBLAS 的 1.04–1.05x,而且已达
实测屋顶的 87%/93%/93% —— 那里几乎没有可捡的。所以措辞不能是"GEMM 一律用 cuBLAS",而应该是
让 agent 有权做这个权衡,并说明权衡依据:

> 当某个子算子是一个大而规整的 GEMM/conv,厂商库(cuBLAS/cuDNN,经 `torch.nn.functional.linear`
> / `conv2d`)通常已经在硬件极限附近,把它留给厂商库、把 kernel 写在它**周围**(融合前后的
> elementwise/归约/layout)往往比重写它更快。**前提是你的候选仍然要有自己的 kernel** ——
> 一个只调用 torch 算子的文件仍然按现有规则拒绝。

**为什么这不会变成"agent 什么都不写"**:现有的两条硬门都还在 —— 无 kernel 的文件被拒
(`candidate_contract.md:155-156`),`torch.compile` 被静态检查拒。这条只是把"计算必须全部在你的
kernel 里"放松成"你必须有 kernel 且承担你声称优化的那部分"。

---

## P8:低精度不是免费的,fp64 rescue 计数必须进报告

**现状**(子代理核查):`fp64_rescued_trials` 进 events.jsonl(`orchestrator.py:1110`),
**不进 report.md**(report.py 全文无引用),**不影响接受或排名**(只有 orchestrator.py:1110 一处写,
之后再无读者;best 只按 `robust_ms`)。

**证据**:我们的 L3:48 候选 1.411 ms 靠 fp64 相对门 **5/5 全部救回**;外部 CUDA 1.477 ms
**0 次救回**、直接过主门。同速下他们更准 —— 而这个差别在我们自己的报告里一个字都看不到。

我为这份方案实测了"换 bf16 能否既保速又免救回",结论是**不能**:

```
fp16 (shipped)        1.409 ms  5/5  rescued=5
bf16 compute+cache      FAIL   0/5  correctness_mismatch
bf16 compute only       FAIL   0/5  correctness_mismatch
fp32 cache only       1.581 ms  5/5  rescued=5
```

所以 L3:48 上 fp16 是必要的(`USE_EXP2` 与 bf16 的尾数交互不良),救回是这个任务的真实代价,
不是可以调掉的缺陷。**这恰好说明为什么它必须被报告出来**:它是一个需要人知道的权衡,不是一个 bug。

**改动**:(a) report.md 的 best-candidate 段落加一行 rescue 计数,`rescued == correctness_trials`
时明确标注"该候选完全依赖 fp64 相对门通过";(b) 两个候选延迟在噪声内(比如 5% 以内)时,报告
应指出 rescue 次数少的那个在数值上更可信 —— 不改变自动接受逻辑(正确性决定接受,这条不动),
只是让报告不再隐藏这个维度。

---

## P9:标定要覆盖 ieee attention 这类"后端真实短板"

这一条是唯一确认的**真实 Triton 限制**,值得单独记账而不是混在调参问题里。

strict IEEE fp32 下我们的 attention kernel 只到 fp32 屋顶的 18%(9.9 TFLOP/s),外部 CUDA 约
55–74%。全部 36 点 tile sweep 的上界就是 5.225 ms —— **调参无法弥补**。原因是
`tl.dot(input_precision="ieee")` 在这张卡上没有快路径。

但同一后端的 GEMM 在 ieee 下仍到 56% 屋顶 —— 所以这不是"Triton 在 ieee 上普遍不行",而是
**特定于 attention 这种带 softmax 的 shape**。

**改动**:标定阶段增加一条 ieee `tl.dot` 的 yardstick(现在的标定只测 `torch.matmul`,
`worker_main.py:1070-1081`,那是 cuBLAS 不是 Triton)。有了它,bottleneck 报告才能对 ieee 候选说
"你已经在这个后端这个精度的实际上限附近了",而不是拿 cuBLAS 的 fp32 屋顶去比、报出一个虚低的百分比
并让 agent 徒劳地去"优化"。这与 P3 修 fp16/bf16 屋顶是同一类错误的另一个面。

**优先级说明**:实际影响小(这些任务都不要求 strict ieee),但成本也低,且它防止的是一类**误导**
而不是一类失败。

---

## 不建议做的事

- **不要把 cuBLAS 设成 GEMM 的默认**。tf32/fp16/bf16 上我们的 Triton 已在屋顶 87–93%,强制换库
  会丢掉融合机会(L3:21 赢的那一版正是靠把 BN 统计融进 producer kernel)。
- **不要因为 L3:48 需要 rescue 就调门限**。实测 bf16 在那里直接失败,fp16 是必要的;门在正常工作。
- **不要继续加强 shared-memory 约束的 prompt 教学**。已经证明这条路走不通(24/24 低估,真实约束
  36/36 恒真),再写只会让 agent 更自信地算错。

---

## 建议的实施顺序

| | 项 | 类型 | 成本 | 判据 |
|---|---|---|---|---|
| 1 | **P5** compile screen 前移进 `guard_ok` + 撤下 shared-memory prompt 教学 | worker+driver | 中 | L3:43 那 36 点里 17 个应在 ask 阶段被拒;`infeasible_shared_memory` 的 record 数应大幅下降 |
| 2 | **P6-A/B** report 的 per-precision trial 表 + 接上 STATS_DONE 渲染 | 纯读侧 | 低 | 重放旧 run 就能验证:L3:43 报告里应出现"tf32 complete=0" |
| 3 | **P8** rescue 计数进报告 | 纯读侧 | 低 | 重放 L3:48 run,best 段落出现 rescued=5 |
| 4 | **P7** 契约放宽厂商库 | prompt | 低 | 下一轮 L3:43 是否出现调用 `F.linear` 的候选;不应出现无 kernel 的候选 |
| 5 | **P9** ieee `tl.dot` yardstick | worker | 低 | 标定输出多一个数;ieee 候选的 ceiling 百分比不再虚低 |
| 6 | **P6-C** per-precision 最优上报 | driver | 高 | 需要设计,放最后 |

2、3 是纯读侧改动,可以在实验运行期间做并用重放验证;1 有 driver 侧成分,需要重启才生效
(见 `opop-v2-worker-vs-driver-fix-propagation`)。

## 复现本方案的实测

```sh
# agent 真实约束 vs 编译期真值的混淆矩阵(P5 的判据)
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python confusion.py'
# 手算公式的系统性低估,两个任务交叉验证
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python constraint_vs_truth.py'
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python cg2.py'
# L3:48 换 bf16 能否免除 fp64 救回(P8 的反面证据)
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python l3_48_bf16.py'
```
