# P5 详解 + 硬性约束全面审计

两部分:(1) P5 的问题/方法/为什么有效,逐点配实测和例子;(2) 类似"只允许 layout/reshaping"这种
硬性约束的系统审计,含风险分级和具体修法。

**重要更正**:为写这份文档补做的两组实测**推翻了我上一版 P5 的具体做法**。上一版说"把 compile
screen 前移进 `guard_ok`",实测证明那样做几乎不省时间还会让一次 `ask()` 最坏停 17.8 分钟。正确的
做法是**批量化**,下面第 1.4 节给出。结论(不要让 agent 手算 shared memory)不变,手段变了。

---

# 第一部分:P5 详解

## 1.1 当前的问题是什么

### 现象:一条从不拒绝任何东西的约束

参数空间里每个候选都可以声明约束(`Constraint.expr`),由 `guard.py` 的受限求值器判定。对 tile 尺寸
与精度的交互,`agents/modules.py:645-687` 教 agent 写 dtype-aware 的 shared-memory 约束。agent 照做了
—— L3:48 的 18 个 `SPACE_PUBLISHED` 里绝大多数都有这类约束。

我把 L3:43 `cand-969997e3` **逐字取自 events.jsonl** 的那条真实约束

```
(COMPUTE_DTYPE == "fp16" or COMPUTE_DTYPE == "bf16") and
  NUM_STAGES * 2 * BLOCK_N * 64 * 2 <= MAX_SHARED_BYTES_OPTIN
 or (COMPUTE_DTYPE == "tf32" or COMPUTE_DTYPE == "ieee") and
  NUM_STAGES * 2 * BLOCK_N * 64 * 4 <= MAX_SHARED_BYTES_OPTIN
```

用 **guard 自己的求值器**(`eval_constraint`,不是我另写的模拟)在 36 个 tile×精度配置上跑,与编译期
真值(`metadata.shared`)对照:

| | 真实能装下 | 真实会死 |
|---|---|---|
| 约束放行 | 19 | **17** |
| 约束拒绝 | **0** | **0** |

**它放行了全部 36 个配置,一个都没拒。** 与真值一致率 53%,恰好等于"全部放行"这个平凡基线 ——
换句话说,删掉这条约束,行为完全不变。

### 具体例子:它错在哪

以 `COMPUTE_DTYPE=ieee, BLOCK_N=64, NUM_STAGES=2` 为例:

```
约束算出:  2 * 2 * 64 * 64 * 4 = 65536  <= 101376   -> 放行
编译器说:  metadata.shared     = 164352  > 101376   -> 启动必死
```

低估 2.5 倍。全部 24 个测点低估率中位数 **0.32**(最差 0.12);换 L3:48 上另一个 agent 写的同型约束
交叉验证,16 点里 10 个低估,中位 0.82、最低 0.28、最高 2.36 —— **两个方向都会错**。

### 根因:这个数原理上算不出来

不是 agent 不够仔细。`metadata.shared` 包含 Triton 编译器自己决定的一切:多缓冲(`num_stages` 的
实际实现)、对齐填充、softmax 的中间张量、swizzle 布局。而且它**不单调**:

```
fp16 BM=128 BN=32 stages=1 -> 65536
fp16 BM=128 BN=64 stages=1 -> 65536     <- BLOCK_N 翻倍,一个字节没多
fp16 BM=128 BN=64 stages=3 -> 163840
```

任何"tile 元素数 × 字节宽"形式的公式都产生不出这样的序列。`worker_main.py:512-516` 里其实已经写了
这个结论("审计 16 个 L3:43 候选,`BLOCK_M*BLOCK_N*stages` 的失败最小值低于通过最大值 15/15,
任何此类乘积都无法分离两个集合,让 parameterizer 写一条等于让它猜"),**但 prompt 仍在让 agent 猜**。

### 为什么必须修:两层代价

**第一层 —— 浪费的 trial。** 历史数据 180/1004 个 trial(18%,0.93h)死在 shared memory。P1 之后
这些被 compile screen 在启动前拦下,但**仍然花掉了** materialize + 一次探测。

**第二层,更严重 —— 静默的认知污染。** 一条恒真的约束让读报告的人(包括我)以为这一维被保护着。
这直接导致了 L3:43 那个"一套 tile 通吃所有精度"的问题被埋了很久:θ_best 在 tf32/ieee 下
**连启动都做不到**,而报告里只有一堆混在一起的 failed trial。

---

## 1.2 上一版的修法为什么不对(实测否决)

上一版 P5 说:把 compile screen 前移进 `ask()` 的 `guard_ok` 回路。理由是那样就能在采样阶段排除,
不必等到 materialize。

**我实测了一次 compile probe 的真实成本**(`scripts/external_comparison/probe_cost.py`,
按 harness 真实调用方式 —— 每次一个独立 worker 进程):

```
fp16  BM= 64 BN=32 stg=1   11.90s
fp16  BM=128 BN=64 stg=3   15.89s
tf32  BM=128 BN=32 stg=2   16.69s
ieee  BM=128 BN=64 stg=2   19.41s
tf32  BM= 64 BN=64 stg=3   25.62s
中位数 16.69s
```

对照:L3:43 那 180 个浪费的 trial 是 0.93h,即**每个 18.6s**。

**所以前移进 guard 几乎不省任何时间**(16.7s 换 18.6s),而且引入一个新的坏结果:`ask()` 的
`max_guard_rejects_per_ask=64`(`tpe.py:30`)意味着一次 ask 最坏要连拒 64 次 ——
**64 × 16.7s = 17.8 分钟**,期间整个调参循环卡住。这个方案不能要。

成本的来源在 `worker_client.py:159`:`subprocess.run`,**一 job 一进程**。16.7s 里绝大部分是进程
启动 + torch import + CUDA context + KernelBench import,真正编译一个 kernel 只占很小一块。

---

## 1.3 关键实测:批量化把成本降 73 倍

既然固定成本是主项,那就让一个进程筛很多配置。实测(`probe_batch.py`,48 个配置一个进程):

```
48 configs screened in 11.02s total
marginal per config: median 7 ms   min 6 ms   max 1336 ms
19/48 exceed the 101376 limit -> would be refused

48 个独立 worker 进程 @16.7s = 13.4 min
一个批量进程                  = 0.18 min   -> 73x 更便宜
```

**边际成本 7 ms。** 这个数字改变了一切:7 ms 的筛查放在采样回路里完全无感(相比之下一个真实 trial
是 18.6s,差 2600 倍)。

### 为什么可以批量:探测只依赖启动配置

`metadata.shared` 只由 launch 配置(tile/warps/stages/dtype constexpr)决定,与输入数据无关 ——
这也是 `run_compile_probe` 能用 `warmup=True` 只编译不启动的原因(`worker_main.py:518-521`)。
所以在**一个已经建好 CUDA context 的进程里**,换一组 PARAMS 重新编译,就是一次纯编译,7 ms。

### 但不能穷举:网格太大

我数了真实空间里"影响 shared 的 knob"子网格规模(`grid_sizes.py`,按名字含
BLOCK/STAGE/WARP/DTYPE/CHUNK/TILE/SPLIT 筛选):

```
run-l3-43: 中位 36,864   最大   139,968
run-l3-48: 中位  5,120   最大   600,000   <- 7ms x 600k = 70 分钟
```

**所以"开跑前把整个网格筛一遍"不可行。** 但这不要紧:`trials_per_space` 只有 40,搜索永远只会碰到
网格的极小一角。方案因此是**按需批量**,不是穷举。

---

## 1.4 P5 的正确修法

### 改动 A:job 接口从"一个配置"改成"一批配置"

现状 `make_compile_probe_job(ref, kernel_src_path, backend)` 只接一个 materialized 文件
(`jobs.py:62-70`)。改成可接一批:

```
make_compile_probe_job(ref, kernel_src_paths=[...], backend=...)
   -> {"ok": True, "results": {<path or key>: {"max_shared": N, "kernels": [...]}, ...}}
```

worker 侧 `run_compile_probe`(`worker_main.py:497`)已经在一个进程里做完了"加载候选 + monkeypatch
`JITFunction.run` 强制 warmup + 收集 metadata",把它包在一个 for 循环里即可。**语义完全不变**,
只是把固定成本摊薄。

`ok: False` 的语义必须逐条保留:某个配置探测不了,就它自己 fall through,不影响同批其他配置。

### 改动 B:在 `ask()` 里用"预取一批"而不是"每次探一个"

`orchestrator.py:1010` 现在是

```python
guard_ok=lambda p: check_config(space, p, self.cfg.device) is None
```

改成两级:

1. **先跑 `check_config`**(纯算术,微秒级)—— 域/类型/choices 校验和 agent 写的其余约束仍然有用
   (寄存器、线程数、纯逻辑关系),只是不再指望它管 shared memory。
2. **再查一个 shared-feasibility 缓存**。缓存 miss 时,不是探一个,而是**让 tuner 多 ask 几个候选
   配置(比如 16 个)一次批量探测**,把结果全部写进缓存。之后的 miss 大概率变成命中。

缓存键就是现在的 materialized source hash(`correctness.py:96`,`_screen_cache` 已是这个形状)。

### 改动 C:保留 P1 现有的 screen 作为兜底,不要删

`orchestrator.py:1085` 的 screen 仍然留着。理由:批量预筛只覆盖被 ask 到的配置,而 anchors、
witness、缓存复用路径不经过 `ask()`。两层都在,成本可忽略(命中缓存),而删掉任何一层都会开一个口子。

### 改动 D:撤下 prompt 里让 agent 手算 shared memory 的教学

`agents/modules.py:645-687` 那一大段。**这一条和上面三条一样重要,不是附带**:它现在的净效果是让
agent 花 token 写恒真约束,并制造"这一维被保护着"的错觉。

具体保留/删除:
- **保留** (a)"一个 kernel 一条约束"和 (c)"从 kernel body 推导" —— 这两条对**寄存器数、线程数、
  纯逻辑关系**的约束仍然有效,那些确实是知识可及的。
- **删除** (b)"字节宽随精度 knob 变"以及那个 shared-memory 约束模板。
- **替换成一句明确的话**:"shared memory 的可行性由 harness 在编译期判定(编译器自己的
  `metadata.shared`),**不要为它写约束** —— 从源码估算这个数并不可行,一条估算错的约束比没有约束
  更糟。" 并说明这不影响它该做的事:tile 该多大仍由调参器用真实测量决定。

### 为什么这样修能解决问题

| 问题 | 这一改动为什么解决它 |
|---|---|
| 约束恒真、形同虚设 | 判定不再来自 agent 的估算,而来自编译器自己的数字 —— 该数字与运行期 `Required` 已 6/6 逐字节验证 |
| 无法用闭式约束表达 | 不再尝试表达。`metadata.shared` 不单调、含编译器内部决策,批量探测直接问它 |
| 前移太贵(16.7s) | 批量化把边际成本降到 7 ms(73x),这才让"在采样阶段判定"变得可行 |
| 一次 ask 最坏卡 17.8 min | 一批 16 个配置一次探测(约 11s),而不是 64 次独立探测 |
| 认知污染 | prompt 不再声称这一维有保护;报告里 `infeasible_shared_memory` 的数量成为该维度的真实体温计 |

### 风险与守护(逐条)

1. **探测不可用时必须放行。** `compile_screen` 现在就是这个语义
   (`correctness.py:86-88`:"This screen must never be the thing that rejects a candidate")。
   批量化后要保证是**逐配置**放行,不是整批放行 —— 否则一个坏配置会让同批 15 个好配置全被跳过。
2. **`max_guard_rejects_per_ask` 要按新成本重定。** 命中缓存是微秒级,但每 16 次 miss 触发一次
   ~11s 探测。64 这个上限在最坏情况下是 4 批 ≈ 44s,可接受;但要把这个算术写进注释,否则下一个人
   改批大小时会重新踩进 17.8 分钟。
3. **不要把批量探测放进独占计时通道。** 它是编译,走 shared 通道(现在 `compile_screen` 传的就是
   `lock_mode="shared"`,`correctness.py:101`)。
4. **验证判据(必须能失败)**:重放 L3:43 那 36 个配置,批量 screen 应拒掉 17 个、放行 19 个,
   与 `confusion.py` 的真值列完全一致;并且**反向对照** —— 故意把设备上限设成 1e9,应该 0 拒绝。
   没有这个反向对照,一个"永远拒绝"和一个"永远放行"的实现都会看起来通过。

---

# 第二部分:硬性约束审计

按"排除掉合法优化手段的风险"从高到低。每条标注**措辞 vs 代码强制**,这个区别决定修法。

## 关键背景:KernelBench 自己把这些划成了 warning

`worker_main.py:1470-1472` 调 `validate_kernel_static(code, backend, precision)`,不传 `forbidden`,
所以用默认的 `STRICT_CHECKS`。查 KernelBench 源码:

```python
STRICT_CHECKS = ["code_bypass", "timing_event_patch", "thread_injection", "lazy_eval"]

WARNING_CHECKS = ["pytorch_wrap",            # nn.Linear / nn.Conv2d 等计算层
                  "torch_computation_ops",   # torch.matmul / F.linear / F.conv2d 等
                  "stream_injection",
                  "precision_downgrade"]
```

**`torch.matmul` / `F.linear` / `F.conv2d` 只是 warning,从来不会拒绝候选。** 也就是说
**调用 cuBLAS 在代码层面一直是允许的,只有我们的契约文字禁止它。** 这让 P7 变成纯文本改动,
零代码风险。

顺带一个缺口:`static_warnings` 被记录(`correctness.py:167, 212`)但**全项目无人读取**
(grep `.get("warnings")` 零命中)。所以放宽措辞后,我们也看不到谁真的调了 cuBLAS —— 修法见 H3。

---

## H1(高风险,措辞)"torch 算子只允许 layout/reshaping"

**位置**:`candidate_contract.md:145`

> torch operations are allowed around the custom kernel(s) (**layout, reshaping**),
> but the core computation you claim to optimize must run in your kernel.

**代码强制?** 否 —— 见上,`torch_computation_ops` 是 warning。

**过度限制:是。** 括号里那两个词把 cuBLAS 排除在**计算**之外。实测代价:L3:43 的两个 projection
占 85.7% FLOP,外部 CUDA 用 `at::linear`(cuBLAS),我们手写 Triton GEMM,strict ieee 下多花
约 2.5 ms(10.063 vs 7.551,1.33x)。

**修法**(注意**不能**反向变成"GEMM 一律用 cuBLAS" —— tf32/fp16/bf16 上我们的 Triton 已在屋顶
87–93%,强制换库会丢掉融合机会,L3:21 赢的那版正是靠把 BN 统计融进 producer kernel):把规定改成
**权衡说明 + 一条硬底线**:

> torch 算子可以出现在你的 kernel 周围。特别地:当某个子算子是一个大而规整的 GEMM/卷积,
> 厂商库(cuBLAS/cuDNN,经 `F.linear`/`F.conv2d`)通常已在硬件极限附近,把它留给厂商库、
> 把 kernel 写在它**周围**(融合前后的 elementwise/归约/layout)往往比重写它更快 ——
> 也往往更慢,如果重写能带来融合。这是**你要做的权衡**,请在 `approach_summary` 里说明你选了哪边、
> 为什么。
> **硬底线不变**:文件必须有自己的 kernel,且你声称优化的那部分必须由它承担;
> 一个只调 torch 算子的文件仍然被拒。

## H2(高风险,措辞)"Prefer triton" + "prefer it for matmul/conv-bound work"

**位置**:`candidate_contract.md:141`(`Prefer triton`)、`:117-119`
(`prefer it for matmul/conv-bound work`)

**代码强制?** 部分。后端本身不强制,但**声明 `cuda` 会触发 KernelBench 的
`BACKEND_IMPL_CHECK["cuda"] = "cuda_impl"`**,那是 strict —— 即声明 cuda 就必须真有 CUDA 实现。
这一条是合理的(声明与实现一致)。

**过度限制:是,但是"事实性"过度。** 已记录 35/35 候选全 Triton 是 prompt 要求的结果而非发现。
本次实测给了它一个具体的反例:strict ieee 下手写 CUDA 的 attention 达 fp32 屋顶 55–74%,
我们的 Triton 只有 18%,而全部 36 点 tile sweep 都无法弥补 —— 那是 `tl.dot(input_precision="ieee")`
在这张卡没有快路径。**所以"prefer triton"在某些精度/shape 上是错的建议。**

**修法**:把无条件的 `Prefer triton` 改成带依据的选择说明 —— Triton 自带 profile 元数据和更短的
迭代周期,所以默认从它开始;但如果任务需要 strict IEEE fp32 的 dot,或需要 Triton 表达不了的
warp 级原语,手写 CUDA 是正当选择。同时把第二处 "prefer it (tf32) for matmul/conv-bound work"
改成"tf32/低精度通常是最大杠杆,但由调参器用真实测量决定,不要在源码里预先定死"
—— 后半句正是 L3:43 那个"一套 tile 通吃"问题的预防。

## H3(中风险,代码缺口)静态检查的 warnings 无人读取

**位置**:`correctness.py:167, 212` 写入 `static_warnings`;全项目零消费者。

**过度限制:否 —— 这是相反的问题(过于宽松)。** 但它与 H1 直接相关:放宽 cuBLAS 措辞之后,
我们需要看得见"这个候选把计算交给了 torch 算子多少",否则无法判断 H1 的修改是否被滥用。

**修法**:让 `torch_computation_ops` 这个 warning 进入报告的候选行(不拒绝,只显示)。这样
"某候选实际把 GEMM 交给了 cuBLAS"成为可见事实,人可以判断它是好权衡还是偷懒。**注意不要把它
升级成 strict** —— 那正好回到 H1 的错误。

## H4(中风险,措辞+代码不一致)后端白名单与 dead branch

**位置**:`models/core.py:18`:`Backend = Literal["triton", "cuda"]`;
而 `worker_main.py:1877`:`if backend.lower() in ("triton", "tilelang", "cute")`。

**过度限制:是,而且是自相矛盾的。** worker 为 `tilelang`/`cute` 准备了加载分支,但类型系统根本不
允许这两个值到达那里 —— 那是**死代码**。KernelBench 本身支持
`cute`/`cutlass`/`tilelang`/`thunderkittens`/`hip`(见 `BACKEND_IMPL_CHECK`)。

**修法**:这是之前记为 D3 的项。两条路选一条,不要停在现状:
- 若不打算支持,**删掉 worker 里的死分支**,并在契约里明说"本 harness 目前只支持 triton 和 cuda";
- 若打算支持,扩 `Backend` Literal 并**先在 worker venv 里装上依赖** —— 现在连 CUTLASS 都没装,
  扩类型只会把编译失败伪装成候选缺陷。
建议先做第一条(诚实且便宜),把第二条留给单独一轮。

## H5(低风险,措辞)"accumulator 必须 fp32"

**位置**:`candidate_contract.md:130-135`(`keep the accumulator in fp32`,标了 REQUIRED 语气)

**代码强制?** 否。`triton_lint.py` 只对"硬编码低精度 cast 且无 dtype knob"发**警告**
(`triton_lint.py:336-347`,注释明确"WARNING only (never blocks)")。

**过度限制:不确定,倾向否。** 这条在物理上几乎总是对的(长归约的低精度累加会掉出容差),而且它是
**建议**不是门,所以坏情况有限。但严格说,存在合法反例:短归约 + bf16 累加可能既够准又更快。

**修法**:把 REQUIRED 语气降为强建议并给出判据("除非你的归约长度很短且 diff-test 证明够准")。
优先级低。

## H6(低风险,已论证充分)`torch.compile` 一律拒绝

**位置**:`triton_lint.py:225-246` + `candidate_contract.md:146-153`。**代码强制:是。**

**过度限制:否 —— 这条应当保留。** 它的注释已经诚实地承认这是 blanket 规则、会误伤"真 kernel 旁边
有个 compile 的 glue path",并给了保留它的理由:160 个候选里只有 2 个命中(都在已废弃的 run),
270 个 KernelBench 参考里 0 个使用 —— **假阳性类为空**;而真阳性类不空(L3:21 那个把 no-op copy
kernel 焊在 compiled graph 上、立刻成为 run 最优的例子)。这是本项目里论证得最好的一条硬约束,
**不要动它**。

## H7(不是限制,但同类错误)`STRICT_CHECKS` 我们从未审视过

**位置**:`worker_main.py:1470-1472` 不传 `forbidden`,静默接受 KernelBench 的默认。

`STRICT_CHECKS` 的四项(`code_bypass`/`timing_event_patch`/`thread_injection`/`lazy_eval`)都是
反作弊,保留正确。**但我们从未在代码或文档里记录过"我们选择了默认值"这个决定** —— 一次
KernelBench 升级改动这个列表,我们的接受口径会静默改变。

**修法**:显式传入我们要的清单(即使就是当前默认),并加一行注释说明为什么这四项 strict、
那四项 warning。**成本极低,防的是一次静默的口径漂移。**

---

## 建议一起修的集合与顺序

| | 项 | 类型 | 风险 | 成本 |
|---|---|---|---|---|
| 1 | **H1** 契约放宽厂商库(带权衡说明+硬底线) | 措辞 | 高 | 低 |
| 2 | **H2** `Prefer triton` / `prefer tf32` 改成带依据的选择 | 措辞 | 高 | 低 |
| 3 | **P5 A-D** 批量 compile screen + 撤下手算教学 | 代码+措辞 | 高 | 中 |
| 4 | **H3** `torch_computation_ops` warning 进报告 | 代码(读侧) | 中 | 低 |
| 5 | **H4** 删掉 tilelang/cute 死分支 + 契约说明支持范围 | 代码 | 中 | 低 |
| 6 | **H7** 显式传 `forbidden` 清单 + 注释 | 代码 | 低 | 极低 |
| 7 | **H5** accumulator 语气降级 | 措辞 | 低 | 极低 |

**明确不动**:H6(`torch.compile` 禁令,论证充分且假阳性类为空)、`STRICT_CHECKS` 的四项反作弊。

1、2、4 可在实验运行期做(措辞和读侧);3 有 worker+driver 两侧,需重启才全生效
(见 `opop-v2-worker-vs-driver-fix-propagation`)。

## 复现本文档的实测

```sh
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python confusion.py'      # 约束 36/36 恒真
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python probe_cost.py'     # 单次探测 16.7s
ssh autodl2 'cd /root/autodl-tmp/ext-eval && python probe_batch.py'    # 批量 7ms,73x
ssh autodl  'python /root/grid_sizes.py'                               # 子网格规模,穷举不可行
```
