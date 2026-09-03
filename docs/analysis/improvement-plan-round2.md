# v2 Harness 改进实施方案 —— 第二轮(train/eval 语义 · 空间扩展 · dtype knob 一致性)

> 日期:2026-09-04(方案)。依据:改进后 L3 重跑的磁盘取证
> —— `run-l3-43-20260903-145357`(attention,best 19.4ms,RUN_FINISHED)与
> `run-l3-21-20260903-210650`(MBConv,best=None)。
> 本方案是 `improvement-implementation-plan.md`(第一轮 A–F + H1/H2/H3)的**续篇**,
> 针对第一轮改进落地后**新暴露**的三个问题。每项给出改动点(文件:行)、代码草图、
> **通用场景下的负面作用评估**、风险、验收。
>
> **前置结论(已磁盘核实,勿凭记忆复述)**:
> - L3:43 成功:H1/H2/H3 让候选自主走 tf32,best 从上轮 29.3ms(ieee 天花板)降到 19.4ms
>   (vs eager 2.02× / eager_tf32 1.40× / compile 1.72× / **compile_tf32 0.874×** 未超但逼近)。
> - L3:21 失败(best=None):4 seed 中 2 个真实正确性失败(train-mode BN)、2 个被网络降级杀死。
>   网络已用真实 agent-smoke 调用确认恢复。
>
> **原则(延续第一轮 §0)**:只加泛化增益,不写死 case 补丁或"押某方向"的性能策略;
> 改动可开关、可回退、默认行为在验收前不变;不破坏事件溯源/resume 与现有 96 测试。

---

## 0. 三个问题的定性(先分清性质)

| 编号 | 问题 | 性质 | 磁盘证据 |
|---|---|---|---|
| **I** | L3:21 候选用 running stats 做 BatchNorm,但参考跑 train 模式(用 batch stats)→ 固定误差 5.655 | **信息缺口**(agent 不知道参考的运行模式),非流程 bug、非容差问题 | 6 次 `witness_*_failed`,diff 恒 5.655153;`worker_main.py:371` 建参考无 `.eval()` |
| **II** | boundary 检测到"某 knob 在已测范围上界且单调向边缘",但只喂给 rewriter(重结构改写),缺"轻量扩展该 knob 范围再调一轮"的闭环 | **闭环缺失**(有检测无针对性修正动作) | L3:43 winner BLOCK_M 曲线 32→20.9/64→20.6/128→19.4,单调撞上界;`stats.py:_param_stat` 已能标 `at_boundary` |
| **III** | rewriter 把 dtype cast 硬编码进 kernel body(`.to(fp16)`),parameterizer 却另加一个与 body 不一致的 `DOT_PRECISION` knob → 纯 fp16 从未被独立评测(tf32 vs ieee 只差 1%) | **knob 语义不一致**(契约"一切可调项走 PARAMS"对 body 内 dtype 约束不足) | `opop-v2-fp16-knob-gap`;winner best-per-value:tf32=19.4 / ieee=19.6 |

三者都指向同一个更高层的观察:**框架的"检测/生成"能力已到位(boundary 检测、精度维度识别、dtype 改写),但"把检测结果闭环回参数空间本身"的能力不足**——要么信息没传给 agent(I),要么检测结果只有重动作出口(II),要么可调维度被写死在 body 逃过 knob 化(III)。

---

## 改进 J:参考运行模式(train/eval)探测并作为任务事实告知 agent

**动机**(问题 I):L3:21 反复 best=None 的根因是候选用 `bn.running_mean`/`bn.running_var`
(未训练模型 = 0/1),而 KernelBench 评测时参考处于 **train 模式**,BatchNorm 用当前 batch 的
mean/var。这是 KernelBench 的评测约定,**不是通用真理**。

**关键设计决策(回应"通用场景负面作用"顾虑)**:
> ❌ **不要**在契约里写死"BN 必须用 batch 统计"——那会在框架面向通用 kernel 优化(推理场景,
> 参考通常 eval 模式、BN 用 running stats)时**主动教 agent 写错**。
>
> ✅ **正确做法**:让 harness **探测参考的实际运行模式(以及各 norm 子模块的 training flag)**,
> 作为**任务事实**传给 agent;契约改成"**复现参考在其被评测的运行模式下的语义**"。
> 前提从"假设"变为"探测到的事实",对 KernelBench(train)和未来通用场景(eval)都正确。

**改动点**:

1. **worker 探测并回传运行模式**(`gpu/worker_main.py`,新增一个轻量 job `probe_model_semantics`
   或并入现有 `env_probe`):建好参考模型后,读取
   ```python
   ref_model = Model(*init_inputs)          # worker_main.py:371,现状无 .eval()
   # 核心 flag:直接读活对象的运行时状态(不是解析源码文本、不是正则匹配 .eval())。
   # 参考前向就在这个状态下跑,所以该 flag 定义上正确,不依赖代码写法。
   semantics = {
       "training": bool(ref_model.training),
       "norm_layers": [
           # 泛化性关键:用"能力检测"而非"类型列表 isinstance"——只要该层带 running
           # stats 缓冲区就识别,能 catch 自定义 BN-like 层,不受类型硬编码限制。
           {"type": type(m).__name__, "training": bool(m.training),
            "has_running_stats": (hasattr(m, "running_mean")
                                  or hasattr(m, "running_var")),
            "track_running_stats": getattr(m, "track_running_stats", None),
            "momentum": getattr(m, "momentum", None)}
           for m in ref_model.modules()
           if hasattr(m, "running_mean") or hasattr(m, "running_var")
           or hasattr(m, "track_running_stats")
       ],
   }
   ```
   仅回传**标量/短列表**(每层 ~5 个字段),不回传权重/张量。

   **泛化性/失败模式评估**(回应"是否硬编码检查代码、是否检测失败"):
   - 核心 `ref_model.training` 是**读活对象的通用属性**,不是静态代码检查,**不存在"代码写得刁钻检测不到"的问题**——因为我们根本不看代码文本,只读那个即将产出参考输出的对象的真实状态。
   - 唯一"参考不是 torch Module"时读不到 `.training`——但那样整个正确性比对路径(`Model(*init_inputs)` + `model(*inputs)`)本就跑不起来,不是 J 新引入的失败。
   - per-layer 细节改用**能力检测(`hasattr(running_mean)`)** 而非类型列表,泛化到自定义 norm;即便某自定义层仍漏判,也只影响"提示完整性",不影响正确性门。
   - **最终安全网**:双精度见证门是裁判——agent 即使拿到 flag 仍猜错语义,门会安全判 fail(→ repair/丢弃),**绝不误接受错误候选**。启发式只"引导",执行"裁决",降级安全。


2. **把 semantics 存进任务 spec / run manifest**,并注入 generator/parameterizer/repair 的沙箱
   (`agents/modules.py` 的 `seed_sandbox`),写成 `task/eval_semantics.json`。

3. **契约新增段**(`agents/prompts/candidate_contract.md`,"Correctness and honesty" 后):
   > ## 参考的运行模式(train vs eval)
   > harness 在 `task/eval_semantics.json` 里告诉你参考模型被评测时的运行模式。你的 kernel
   > **必须复现该模式下的语义**,而不是想当然地按推理写。特别是对 BatchNorm/InstanceNorm 这类
   > 对 train/eval 敏感的层:
   > - **参考处于 train 模式(`training: true`)时**:BatchNorm 用**当前 batch 的 mean/var**
   >   归一化(不是 running_mean/running_var——对未训练模型它们是 0/1,会导致大偏差)。
   > - **参考处于 eval 模式(`training: false`)时**:用 running_mean/running_var。
   > 逐层核对 `eval_semantics.json.norm_layers[*].training`,不要跨层假设一致。

4. **repair 分级提示补一条**(见改进 L,与 K 联动):当误差为"大而恒定的系统性偏移"时,
   优先提示核对 train/eval 语义。

**通用场景负面作用评估**:**无**。因为规则不写死具体做法,只要求"匹配 harness 告知的模式"。
KernelBench 场景 → 告知 train → 写 batch-stat;通用推理场景 → 告知 eval → 写 running-stat。
`eval_semantics.json` 缺失时(老 run / 探测失败),契约段降级为"若不确定,默认按参考源码的
显式 `.eval()`/`.train()` 调用推断",不强加约束。

**风险**:探测 job 增加一次极短的 WSL 调用(可并入 baseline 阶段,几乎零成本)。norm 层枚举
若遇到自定义 norm 可能漏判 → 只影响提示完整性,不影响正确性门(门仍以真实前向为准)。

**验收**:
- 单测:构造含 BN 的假模型,`probe_model_semantics` 返回 `training: true` 且列出 BN 层。
- L3:21 重跑:seed 的 witness diff 从 5.655 量级降到可过门(或 repair 一次内修正)。

---

## 改进 K:boundary + 空闲资源触发"轻量参数空间扩展"闭环

**动机**(问题 II):L3:43 winner 的 BLOCK_M 在已测范围上界(128)单调向边缘、且资源有余量
(shared 仅用 49152/101376 = 48%),说明"再大可能更快",但当前框架把 `at_boundary` 只喂给
**rewriter**(重结构改写,易引 bug、耗预算),缺一条**轻量出口**:保持 kernel 结构不变,
只把该 knob 的 choices 向边缘扩展一档,重调一轮。

**先确认的真实机制(已核实,纠正之前"硬编码"的错误表述)**:
- 参数空间**由 parameterizer agent 动态生成**(`modules.py:227` prompt:每个 knob 提议 2–8 个
  choices,不同候选可有不同 knob 与范围),**非框架硬编码**。
- harness 做**确定性校验 + GPU 双见证**(`validation.py`:key 一致性 / ≥2 choices / 默认值在集内 /
  约束 ≥25% 可行 / default+minimal 两见证过门),是 **agent 提议 + harness 硬校验** 的结合。
- boundary 检测是**确定性的**(`stats.py:_param_stat:59-92`):best 落在已测范围边缘 + 末 3 单调
  向边缘 + 边缘外的值不存在或必然失败 → `at_boundary=True, boundary_direction`。

**改动点**:

1. **新增一个"空间扩展"动作**(`control/orchestrator.py`,在 `_analyze` 之后、进入 rewrite 之前):
   当 stats 报告存在 `at_boundary=True` 且 `boundary_direction=="max"`(或 min)且**资源快照显示
   有余量**(如 `shared_frac_of_limit < 0.8` 或 `regs_frac_of_limit < 1.0`,来自
   `stats.py:_resource_snapshot`)的 knob,调用 parameterizer 的一个**受限模式**:
   ```
   # 伪代码:轻量扩展,不改结构
   if space_expandable(stats) and expansions_used[cand] < MAX_SPACE_EXPANSIONS:
       new_space = parameterizer.expand_boundary_knobs(
           source=best_source, stats=stats,
           # 只允许扩展被标 at_boundary 的 knob,向 boundary_direction 追加 1-2 档
       )
       # 复用 validation.validate_and_publish(受 guard 硬约束守卫)
       # 通过则用扩展空间重调一轮(loop B),仍是同一候选、同一结构
   ```

2. **parameterizer 新增受限提示**(`agents/modules.py`,一个 `expand=True` 分支或新方法):
   > 以下 knob 在调参中触到了所提供范围的边缘且延迟仍在向边缘改善,同时硬件资源仍有余量:
   > `{blocked_knobs}`。请**只**为这些 knob 向该方向追加 1–2 个更大/更小的合法档位
   > (遵守硬件约束),**不要改动 kernel 结构、不要改其他 knob**。

3. **约束与有界**:
   - 扩展值必须过 `guard.check_config`(硬件约束求值),避免提议必然 OOM 的值。
   - 每候选每 knob 最多扩展 `MAX_SPACE_EXPANSIONS`(建议 1–2)次,防无限扩张。
   - 扩展后若最优仍落在新边缘但**不再改善**(effect_pct < min_improvement_pct)→ 停止扩展,
     转正常 rewrite 流程。
   - config flag `budgets.space_expansion.enabled`(默认关,验收后对 L3 开)。

**通用场景负面作用评估**:低。扩展**只针对确定性检测到的 boundary+空闲资源信号**,受 guard 硬约束
和次数上限双重保护,且保持结构不变(不引入新 kernel 逻辑)。最坏情况是多花几轮调参预算,
不会产生错误候选(仍过双见证门)。

**风险**:parameterizer 可能提议出结构上不支持的更大 tile(如超出 tl.dot 维度约束)→ 由现有
`validation` + `triton_lint` + 见证门拦截,退回不扩展。

**验收**:
- 单测:构造一个 `at_boundary=True, direction=max, shared_frac=0.4` 的 stats,`space_expandable`
  返回 True 并生成只含该 knob 的扩展请求;资源已满时返回 False。
- L3:43 重跑:BLOCK_M 触界后触发一次扩展(如加 256),重调一轮;报告记录扩展前后延迟对比。

---

## 改进 L:dtype 作为贯穿 cast+dot 的单一 knob(修 fp16 knob 不一致)

**动机**(问题 III):rewriter 写的 fp16 kernel 把 `.to(tl.float16)` 硬编码在 body,parameterizer
另加了个 `DOT_PRECISION`(tf32/ieee)knob——两者不是一回事,导致纯 fp16 从未被独立评测
(tf32 vs ieee 只差 1%,说明 winner 的速度几乎不来自 dot 精度)。要诚实回答"fp16 到底能到多少",
必须让**同一个 knob 同时驱动 cast 和 dot 精度**。

**方案 A(主,推荐):parameterizer 把 body 内 dtype 用法抽成单一 `COMPUTE_DTYPE` knob**
- parameterizer 识别 kernel 里的精度控制点:`.to(tl.float16/bfloat16/float32)` 的算子输入 cast、
  `tl.dot(..., input_precision=...)`,把它们统一绑定到**一个** PARAMS knob:
  ```python
  PARAMS = { ..., "COMPUTE_DTYPE": "fp16" }   # choices: ["fp16","bf16","tf32","ieee"]
  ```
  该 knob 值同时决定:输入 cast 的 dtype(fp16/bf16 → tl.float16/bfloat16;tf32/ieee → 保持
  float32)与 `tl.dot` 的 `input_precision`(fp16/bf16 → 走对应 tensor-core;tf32/ieee → tf32/ieee)。
  **累加器恒 fp32**(契约已有此硬约束)。
- 这样调参器能在**真 fp16 / 真 bf16 / tf32 / ieee** 之间做真实对比,fp16 才被干净评测。

**方案 B(兜底,契约层强制)**:`candidate_contract.md` 的"Precision and the tensor-core path"段
补一条硬要求:
> 若你的 kernel 通过 dtype cast(如 `.to(tl.float16)`)使用低精度 tensor core,**该 dtype
> 必须作为 PARAMS knob**(不得硬编码在 body),使 harness 能对不同精度做真实对比。

**改动点**:
- `agents/prompts/candidate_contract.md`:加方案 B 的硬约束 + 说明 `COMPUTE_DTYPE` 的推荐形态。
- `agents/modules.py`:parameterizer prompt(§步骤 1)从"若有 tl.dot 暴露 DOT_PRECISION"升级为
  "把**所有**精度控制点(cast + dot)统一为一个 `COMPUTE_DTYPE` knob";rewriter prompt 在产出
  dtype 类改写时要求精度走该 knob。
- 无需改 materializer/guard(仍是 PARAMS 字面量替换),但 parameterizer 生成的 kernel body 需
  用 `PARAMS["COMPUTE_DTYPE"]` 驱动 cast——这对 agent 的 AST 改写能力有要求(见风险)。

**权衡分析(回应"如何改进")**:
- 方案 A **更彻底**(真正让 fp16 可评测),但**实现更难**:parameterizer 要把散落 body 多处的
  dtype 用法可靠地绑成一个 knob,容易出 materialize/一致性 bug。建议配 lint 检查(见下)。
- 方案 B **更轻**(只加契约约束),但**依赖 agent 自觉**,不如 A 强制。
- **采纳:A 为主 + B 为契约兜底 + 一个轻量 lint**——`paramspace/triton_lint.py` 加一条
  **warning**(非 hard error):若 body 出现硬编码的 `.to(tl.float16/bfloat16)` 且 PARAMS 无
  dtype 类 knob,提示"dtype 疑似硬编码、应提为 knob"。warning 不阻断,只回传给 agent 作反馈。

**lint warning 的不稳定性评估(回应"硬编码监测的不稳定性")**:
- **关键安全属性:warning 而非 hard error** —— 现有 `triton_lint.py` 分两层:hard error(阻断候选)
  和 warning(不阻断、只回传 agent)。L 的 dtype 检测属**后者**。
- **误报代价 = 至多浪费一点 agent 注意力,永不误拒候选**(不阻断)。可能误报:body 里那个
  `.to(fp16)` 是输出存储 dtype 而非 dot 输入、或候选确实需要一个与可调 dot 无关的固定 fp16 cast
  —— 这些情况 agent 可忽略 warning,候选照常进见证门。
- **漏报也不致命**:body 用 AST 不认得的方式构造精度(`tl.cast`、dtype 作 kernel 参数传入)→
  不响 warning;但它只是"提醒把 dtype 提为 knob"的助推,**真正判断"fp16 有没有被干净评测"的是
  调参面 + report 的实测数据**(tf32 vs fp16 真实延迟),不是 lint。
- **实现要点(避免脆弱)**:检测"是否已有 dtype knob"时**不匹配固定 key 名**(不能只找
  `DOT_PRECISION`,否则 agent 用别名 `COMPUTE_DTYPE` 就误报),改用 **name-agnostic 值域检测**:
  PARAMS 里是否存在某 str 值 ∈ `{fp16,bf16,tf32,ieee}` 的 knob。
- **统一原则(与 J 共通)**:所有硬编码/启发式部分都是**建议性/信息性,从不充当正确性或接受性的
  裁判**。裁判永远是——正确性用运行时双精度见证门,性能用实测调参数据。启发式失手 = 优雅降级。

**一个前提(诚实预期)**:即使 fp16 被干净评测,也**未必**过双精度见证门——bf16 已因 diff 0.0014
被拒。fp16(10-bit 尾数)比 bf16(7-bit)更可能过,但 attention 长 reduction 仍有风险。
**方案 L 的价值是"让框架能诚实回答 fp16 行不行",不是"保证 fp16 更快"。**

**通用场景负面作用评估**:无。dtype-as-knob 是纯泛化的可调维度提升,对任何含 matmul/conv 的
kernel 都适用;knob 有 ieee 档兜底,需要全精度的候选仍可表达。

**风险**:parameterizer 把多处 dtype 绑成一个 knob 时可能遗漏某处 → body 内 dtype 不一致 →
见证门会因数值错误拦截(安全失败),lint warning 也会提示。

**验收**:
- 单测:给一个 body 硬编码 `.to(tl.float16)` 且无 dtype knob 的源码,lint 返回该 warning;
  给一个已用 `COMPUTE_DTYPE` knob 的源码,无 warning。
- L3:43 重跑:winner 候选的空间含 `COMPUTE_DTYPE` 且 choices 覆盖 fp16;报告能看到 fp16 与 tf32
  的真实延迟对比(而非 tf32/ieee 仅差 1% 的假象)。

---

## 实施顺序与依赖

```
J(train/eval 探测+契约)   ── 独立,最高优先(直接救 L3:21 通过率)
   └─ L(dtype knob 一致性) ── 独立于 J,但同改 contract/parameterizer,建议紧随
K(空间扩展闭环)           ── 依赖 stats(已有)+ parameterizer 受限模式;最后做
```

- **J 先做**:它是 L3:21 从 best=None 翻身的关键,且改动集中(worker 探测 + 契约段 + 沙箱注入)。
- **L 次之**:与 J 同属"契约 + parameterizer prompt"改动区,一起改减少反复。
- **K 最后**:涉及 orchestrator 新动作 + parameterizer 受限模式 + 新 config flag,面最大,
  且是"锦上添花"(L3:43 已成功,K 是想再压 BLOCK_M)。

三项均以 config flag 默认关 / 契约段在 `eval_semantics.json` 缺失时降级,保证默认行为与 resume 兼容。

---

## 每项改动的测试与回归要求

1. **不破坏现有 96 passed / 1 skipped**;新增字段(eval_semantics、COMPUTE_DTYPE、扩展计数)
   向后兼容,replay 老 run 不报错(缺字段走默认分支)。
2. **J**:worker 探测单测(假 BN 模型)+ 契约段渲染进沙箱的断言。
3. **K**:`space_expandable` 判定单测(boundary+空闲→True,资源满→False)+ 扩展有界性单测。
4. **L**:lint warning 单测(硬编码 dtype 无 knob → warn)+ parameterizer 输出含 COMPUTE_DTYPE 的
   schema 校验。
5. **端到端**:先 L1 relaxed smoke(便宜)验证三项不破坏流水,再 L3:21(验 J)+ L3:43(验 K/L)。

---

## 与第一轮的关系 & 诚实口径

- 第一轮(A–F + H1/H2/H3)修好了"通过率 / 多轮 Loop C / 真收敛 / 诚实基线 / tf32 探索"。
- 本轮(J/K/L)修的是第一轮**落地后新暴露**的三个更细的问题:train/eval 语义缺口、
  boundary 闭环缺失、dtype knob 不一致。
- **预期**:J 大概率让 L3:21 至少产出能过门的候选(即使最终仍未超 tf32 torch.compile——MBConv 的
  conv 是 cuDNN 强项,LLM Triton 难超);K/L 有机会把 L3:43 从 0.874× 再往 1.0× 推。
  任何结果都以 events.jsonl 磁盘为准、以诚实同精度判定汇报。
