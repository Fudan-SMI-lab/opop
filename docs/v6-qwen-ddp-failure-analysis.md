# Qwen 与 DDP 失败原因核验（2026-10-21）

本轮只做失败分析和开发者诊断，没有修改框架、接受规则或原运行的候选，没有生成或发布新的优化结果。
原始自主流程见 [Stage 3 记录](evidence/v6-stage3-20261021/final/README.md)；本轮证据见
[诊断归档](evidence/v6-failure-analysis-20261021/README.md)。

## 范围与方法

- box4-2 双 RTX 4090，实际 worker Python；Torch 2.13.0+cu129、Transformers 5.16.1。
- 新诊断固定使用 `187617ce316403ea627c0d33392cc020bca2cb63` 框架、原 adapter、原输入和原候选字节。
- 四个 Qwen 失败候选分别在新进程复现；同模型先执行两次原路径建立重复性对照，再执行两次候选完整生成。
  基线重复和候选卸载后的原路径均为零误差。所有比较保持原 `atol=rtol=0.02` 和完整 greedy token 要求。
- 八个 Qwen 诊断变体用于分离影响因素；它们是开发者修改的分析副本，不计为自主生成、性能接受或可部署修复。
- DDP 在原连续两步全量参考下比较原路径、绑定透传和原候选；两次使用相反执行顺序，共 14 个独立 worker 端点。
  每端点先检查两 rank 的完整梯度、参数和 loss，再进行原定预热与重复测量。
- 另做五组双 rank CPU profile，并只读分解旧通信目标回放的 48 个原始 trace。
  profiler 时间只用于诊断，不替代干净性能样本。

本轮未增加 Agent prompt 或启动新一轮搜索；分析对象来自此前的真实 Agent 调用与 GPU 运行。

## Qwen：四轮数值失败均复现，但原候选的因果解释有误

四轮 greedy tokens 在本次复现中仍完全一致，但完整 logits 超过原要求，拒绝有效。
不能因 token 一致而降低原质量要求。

| 原候选 | 原始／本次复现的完整生成最大绝对误差 | 最早输出分歧 | 已证实的原因 |
|---|---:|---|---|
| TTFT 第一轮 `trial-9a65eeeca9` | 2.25 | prefill logits | 合并 GEMM 与 Triton 融合路径分别引入数值差异，不能归因于 CUDA Graph 本身或只归因于 Triton |
| TTFT 第二轮 `trial-af78dd3283` | 5.0625 | prefill logits | 捕获原 forward 时，Transformers 的 mask 分支发生变化，实际 attention 路径已不同于 eager 原路径 |
| 吞吐第一轮 `trial-3c2ed50501` | 1.65625 | 第一个 eager decode | 补齐的 KV 窗口和显式 mask 改变 SDPA 路径；误差在 Graph 捕获前已出现 |
| 吞吐第二轮 `trial-87dde5b43d` | 1.0 | 第一个 eager decode | StaticCache 的补齐窗口／显式 mask 改变 attention 路径；另有缓存类型导致的绑定回退问题 |

这里的“最早输出分歧”来自逐生成步比较。对后两类 attention 路径还进行了第一层中间值比较，
不再把评测遍历首先失败的 `output.logits` 当作执行中的最早分歧。

### TTFT 第一轮：两个可以独立产生误差的改动

原方案同时做 Q/K/V、gate/up 合并 GEMM、Triton 融合和 CUDA Graph。
保持相同模型、输入和完整输出，诊断结果如下；每个变体的两次完整生成结果一致。

| 诊断路径 | 完整生成最大绝对误差 | 原质量检查 |
|---|---:|---|
| 原方案：合并 GEMM + Triton + Graph | 2.25 | 失败 |
| 合并 GEMM + Triton，关闭 Graph | 2.25 | 失败 |
| 合并 GEMM + PyTorch 计算，关闭 Graph | 2.953125 | 失败 |
| 恢复独立 GEMM + Triton，关闭 Graph | 4.078125 | 失败 |
| 恢复独立 GEMM + PyTorch 计算，关闭 Graph | 0 | 通过 |

因此，Graph 不是该轮误差的必要条件；合并矩阵乘法的数值路径与 Triton 融合路径都不能直接视为
原 BF16 实现的数值等价替换。候选重新绑定权重后，卸载候选的原 forward 仍零误差，排除了权重值
被改变作为这次失败的解释。本轮未继续将 Triton 差异拆到某个 kernel 的某条舍入指令，也没有宣称
两类误差可以线性相加。

### TTFT 第二轮：调用同一个 Python forward，不等于捕获相同的计算路径

真实安装版本中：

1. `transformers.utils.is_tracing()` 将 CUDA stream capture 视为 tracing。
2. `_ignore_causal_mask_sdpa()` 在 tracing 时返回 False，prefill 因而生成显式 causal mask。
3. `use_gqa_in_sdpa()` 遇到非空 mask 不使用原来的 GQA 分支，改为展开 K/V heads。

实际观测的第一层 SDPA 输入从 `K=[1,8,256,128] / mask=None / enable_gqa=True / is_causal=True`
变为 `K=[1,32,256,128] / bool mask / enable_gqa=False / is_causal=False`。
此前 embedding、输入 norm、Q/K/V projection、Q/K norm 和 query 一致；第一层 attention 输出最大
差异约 0.0009765625，之后经过模型放大，prefill logits 最大差异为 0.3125，完整生成为 5.0625。

诊断中，仅对这个已验证无 padding 的完整 prefill 固定原 eager 的 mask 省略决定，保留 Graph，
两次完整生成都恢复零误差且实际执行 Graph 路径。这确立了本轮失败与 capture 引发的 mask／attention
分支变化的因果联系。这个限定负载的干预不是通用关闭 mask 的实现建议。

### 吞吐第一轮：误差早于 Graph 捕获

原候选前三个 decode 调用使用 eager composite，之后才捕获 Graph。逐步复现表明第一个 eager decode
就已产生 logits 差异；完全关闭 Graph 后最大误差仍为 1.65625。

在关闭 Graph 的诊断副本中，将 attention 从“容量 296 的补齐 KV + 显式 mask”改为
“当前有效 KV 前缀 + 原来的无 mask GQA 调用”，其余计算不变，两次完整生成恢复零误差。
因此候选注释中“首次 Graph 产生的步骤发生分歧”的解释不成立。

### 吞吐第二轮：数值路径与接口不兼容是两个独立问题

从完全相同的 prefill cache 和 teacher token 开始，只执行原 forward 的第一个 decode：

- DynamicCache：K 长度 257、8 heads、无 mask、启用 GQA。
- StaticCache：K 长度 296、展开到 32 heads、bool mask、未启用 GQA。

第一层 attention 输出已经不同（最大差异约 0.00048828125），该步 logits 最大差异为 0.1875。
在诊断副本中恢复有效前缀和原 attention 调用，完整生成两次均零误差，缓存类型仍为 StaticCache。
这说明数值错误不能简单解释为缓存内容损坏。

与此同时，绑定契约只观测到 `DynamicCache` 输入，原候选却返回 `StaticCache`；其后 decode 调用
被签名守卫挡住。原运行和复现均为两次完整生成合计 2 次候选调用、60 次框架回退，Graph 路径没有
获得执行机会。恢复 attention 数值路径不会自动修复这个接口问题；诊断变体仍保留相同回退。

## DDP：主要退化来自绑定路径，P2P 第一轮实际未生效

### 完整训练步第一轮的 P2P 方案没有执行

原候选先调用 `_reduce_op_is_sum()`。实际 dispatcher schema 传入
`__torch__.torch.classes.c10d.ReduceOp`；其 `op()` 返回整数 `0`。生成的检查只接受特定
ReduceOp／enum 类型，不接受该 boxed 对象取出的整数；其额外导入的顶层 `RedOpType` 在这个
Torch 版本中也不存在。

原始正式配对记录与本轮复现均显示 13 次 `parent:guard_reduce_op`、13 次原实现调用，没有
`p2p_oneshot`。该候选在 `_execute_exchange()` 之前就回退了。
所以原先“P2P 方案变慢”的措辞只能描述提交方案及其完整项目结果，不能作为该 P2P 算法性能证据。

### 即使只透传，functional 绑定也已经超过整步约束

下表为每端点五次干净测量的 rank 最大值；同一对照使用相同 adapter、质量参考和候选站点。

| 任务／路径 | 正序端点（ms） | 反序端点（ms） |
|---|---:|---:|
| 通信任务：原路径 | 134.787 | 137.304 |
| 通信任务：绑定透传，省略候选 build／计算 | 228.652 | 235.029 |
| 通信任务：第一轮，每个 bucket 做 CPU rendezvous | 249.962 | 248.127 |
| 通信任务：第二轮，仅首个 bucket 做 CPU rendezvous | 237.479 | 251.751 |
| 整步任务：原路径 | 145.352 | 135.209 |
| 整步任务：绑定透传 | 239.583 | 239.546 |
| 整步任务：原 P2P 候选，实际全部回退 | 238.957 | 230.525 |

全部端点的两 rank、连续两步检查通过，每步比较 248,879,621 个元素，最大绝对误差为 0。
原参数筛查中的固定实例被直接复用：P2P `chunks=4`；两个 rendezvous 候选
`sync_wait_budget_ms=2000`。这不是重新寻找最佳参数，也不混用其他参数实例的性能数字。

通信任务中，透传相对原路径已经增加约 94–98 ms，单独就超过原 151.864 ms 约束。
候选只替换 13 个 collective，但 legacy functional 绑定在整个项目调用中启用 TorchDispatchMode，
并维护 module 上下文。独立 cProfile 实际记录每步 7,564 次 `__torch_dispatch__` 调用和 604 次
module pre-hook。插桩 profile 自身有明显扰动，其累计毫秒不能直接当成干净差值的精确分摊。

CPU rendezvous 也有额外代价，但现有结果不支持把约 100 ms 的主要退化全部归因于它。
第二轮确实已降到每步一次等待，仍无法消除上述绑定路径开销。

### 通信指标下降与稳定通信改善不同

两个通信候选继续调用原 allreduce，只调整 host 到达同步。它们可以把部分等待移出 NCCL kernel
计时，既没有减少传输量，也没有因此满足完整训练步约束。Agent 将隔离校准值称为“纯传输下界”
以及从 `RING_LL` kernel 名推断实际 LL 协议的解释没有成立；先前真实 tuning 日志已确认默认使用
SIMPLE，不能以这些解释证明链路存在预期收益。

对计算候选跨通信目标的首次通过及追加失败，重新分解了两批原始 AB/BA 的 48 个 trace：

- 每个 trace 都是 13 个 NCCL kernel，重新计算的总时长与原记录逐项一致。
- 首个 kernel 时长约 3.273–36.100 ms；其余 12 个 kernel 合计约 59.033–60.904 ms。
- 首次比较中出现很大的配对收益，追加比较中没有稳定复现；波动主要集中于首个 kernel。

这与包含 rank 到达等待、调度及 profiler 扰动的诊断指标相符。没有证据说明链路传输变快。
本轮没有将首个 kernel 的时长进一步唯一拆成等待和实际传输，所以也不将这些范围称为纯传输下界。

## 当前可以据此作出的判断

Qwen 当前的主要失败是生成候选改变了数值执行路径，另有一个明确的 cache 接口不兼容；
DDP 则同时存在框架 functional 绑定开销、候选 ReduceOp 接口错误，以及通信指标解释问题。
它们不能统称为 Agent 缺少更多 profiling 信息，也不能仅通过放宽质量检查解决。

本轮已经将这些原因从生成候选的假设推进到原样复现和受控干预证据。下一步实现范围应基于这些结论
另行确定；未在诊断中修改原搜索策略、失败路由、验收规则或已接受版本。
