# v6 组合退化与通信时序复核

本轮按用户确认的前两项优先级实施。基线为 `bfe2f39`；没有增加按失败类型分流、限制信息获取、历史检索或搜索策略状态机。开发者定义并启动诊断协议，框架执行原始 Agent 候选的完整质量、独立 AB/BA 和采集；本轮诊断不冒充新的无人干预自主优化。

此前实验的自动化范围与运行中补充信息见 [复核文档](../../v6-autonomy-audit-and-followup.md)。真实性能测量成立，并不自动证明从原始项目输入到发布的全过程无人介入。

## 相同父版本上的新增绑定对照

`run_binding_ablation.py` 保留既有父候选，将新 bundle 的实际调用站点接入绑定，但令其全部透传父实现，跳过新候选 build 和计算。质量标准、输入、连续步骤和计时范围不变。以下均为各比较两次端点中位数的中位数，单位 ms；原始样本及每 rank 完整检查见相邻 JSON。

| 项目 | 父版本 → 新站点透传 | 独立另一组配对：父版本 → 实际组合 |
|---|---:|---:|
| Qwen B8 | 689.657 → 1930.637 | 694.780 → 2419.743 |
| GPT-2 DDP，取最大 rank | 146.141 → 227.745 | 132.985 → 235.326 |
| ResNet | 1.489 → 3.157 | 1.434 → 2.954 |

三者质量均通过，但新增绑定路径本身已造成明显退化。直接执行与框架空载对照的方向在反向配对中变化，未证明稳定空载退化。两列有各自的父测量，不可跨列相减来分配精确贡献。

配对汇总见 [binding-summary.json](binding-summary.json)。DDP 父候选实际安装 module 替换，新增两个 functional 通信站点后启用覆盖整个 step 的 `TorchDispatchMode`，其他算子也会经过该分派。这解释了为什么只替换少数通信调用仍可能引入全项目开销；尚未建立每项开销的精确因果分解。

ResNet 此处使用组合中实际继承的 `trial-paired-repair-a` 作为父版本，另一个 bundle 为 `trial-14435a0cc3`。此前 `0.797→3.693 ms` 的整版推广以独立 Graph 版本作为主分支比较对象，不能将其当成同一父版本的单项增量归因。本轮专门消除了这一混淆。

Qwen 透传两步最大绝对误差均为 0.125，仍在原阈值内；父版本为 0。不能假设“透传”等于完全相同的原生执行。独立 native trace 中两端均为 66,064 次 kernel，但有一次 BF16 CUTLASS GEMM 换成 GEMV，kernel 名称种类由 33 变为 34。这是观测到的路径差异，不足以唯一解释误差，也不能把全部时差因果归于 Python。

同一次 native 诊断中，kernel 活动区间并集约为 540.36→552.63 ms，kernel 之间的区间约为 814.74→2549.92 ms；该间隙还可能包含复制等工作，不称为纯 GPU 空闲或可恢复的时间。恢复既有绑定实现后的独立 cProfile 记录到 85,954 次 `__torch_dispatch__` 和 167,183 次 `tensor_description`。这些事实支持优先调查主机分派/提交路径；profiler 与 cProfile 自身有扰动，不将其时差当成干净项目的精确成本分解。

## 一次未采纳的热路径微优化

`f8c9fe0` 试验了算子对象索引、模块/phase 身份缓存和预编译签名守卫；256 组输入描述对照及活跃 tensor 元数据修改检查通过。完整 Qwen 重放仍通过原质量，但透传端点约 2001–2016 ms，实际组合约 2343–2359 ms，未证明整体稳定改善；不同源码批次也不是严格交错的跨版本显著性实验。

因此 `15326f6` 撤回该热路径修改，保留既有绑定语义与新诊断能力。原代码、测试、失败/无改善数据仍在 Git 和证据中。CPU call profile 单独记录采集扰动；累计 CPU 时间、GPU 时间和干净项目时间不能相加。

## 通信采集补齐及测量协议修正

新增 `execution_timeline` 和 `python_runtime` 可选采集器，经现有 registry 调用；没有新增下一轮行为路由。

- 每次 DDP capture 都观察到 13 次 collective，原生关联得到 CPU 提交及 CUDA launch；逻辑输入为每 rank 497,759,232 bytes，不是物理链路流量。
- 每 rank 观察到 592 次 CPU post-accumulate 回调，覆盖 13 个 reducer bucket 的参数集合。
- 可选 GPU stream marker 同样每 rank 592 个，记录回调当前 stream 上的 CUDA event；没有逐参数 synchronize。这不等于已经证明 bucket GPU 输入就绪或消费完成。
- CPU 跨 rank 比较要求相同 host/boot/monotonic 时钟域，并由 capture 前后锚点给出映射区间。它不是 GPU 到达偏差。
- 通信/计算区间按设备分别计算；同步、复制/填充另列。kernel 区间的空隙和末尾通信区间不是可直接节省的关键路径。

首次时间线先同步再初始化各 rank profiler。其启动差异会进入 workload 到达偏差：父版本两 rank 的诊断时长曾为 195.34/216.24 ms。改为 profiler 初始化后、workload 窗口前完成准备，新的两对父采样为 189.26/189.51、187.41/187.59 ms；新增候选为 324.58/324.55、328.36/328.65 ms。这是采集准备方式的改进，不是模型性能加速。

这些末端时长接近不代表逐 collective 的 CPU 提交偏差消失。修正后的 [时间线汇总](communication-startup-after.json) 保留各提交区间；CPU 到达、GPU 到达和最终完成须分别解释。

加入梯度回调后，父版本诊断时长约 202.5–208.9 ms，候选约 346.7–358.8 ms。回调有可见扰动；不同采集条件的数据保持分开，不能用于干净目标接受。

同样的问题已在通信目标 `torch_communication` 中修正，协议标记为 `profiler_start_then_rank_preparation_v2`。旧历史结果不回写、不与新协议混合。初次实测暴露 Kineto 为同名范围生成 CPU 和 GPU 两个 annotation；初版严格窗口检查拒绝了该样本。`d9e0929` 明确选择 CPU annotation，随后双卡真实评价通过：每 rank 两个连续步骤各比较 248,879,623 个元素、误差为 0，干净 step 约 139.63/139.74 ms；三次独立通信采样按 rank 记录，不据此宣称通信收益。

所有时间线保留原始文件。本目录的派生汇总使用修正后的 parser 重读这些不可变 trace，并显式记录 parser 源码和原文件哈希；不将重解析伪装成重跑实验。GPU 就绪、唯一消费关系和等待/传输分离仍有 unknown，原计划这些维度不能一概声称已完整解决。

## 真实 Agent 与验证范围

实际 provider 分别读取绑定资料和通信资料，生成诊断分析；它们没有编辑候选、改变目标或执行 GPU 测试。分析保留原文和会话用量，仍需逐项核实。首个绑定分析使用部分进度快照，其中将 decode 次数按 32 计算（实际 eager 解码为 31 次），以及将 parent/passthrough 视作相同绑定集合等说法不作为测量事实。实际协议和源码优先。

[通信分析原文](communication-review.json) 中，“反向配对仍不能识别方向”“kernel 数相同即 GPU 工作不变”“kernel 间隙就是纯 GPU 空闲”等判断没有被采纳：实测支持本协议下重复出现退化，但计数和区间本身不足以证明语义相同或精确归因。提供给 Agent 的精简 GPU marker 输入未明确标注 arm/revision，且当时 parser 未识别出唯一 workload 窗口；这一输入缺陷保留在原会话中，不能把后来重解析的字段说成 Agent 当时已看到的信息。

两次真实 `zhipuai/glm-5.3` 会话共 74 条 assistant 消息，provider 报告费用合计 $1.8689852322，仅统计、不限制使用；不包含外层 Codex 或此前实验。详见 [用量回执](agent-usage.json)。

最新相关检查为 24 passed，另有真实三项目质量/配对、双卡通信目标和 native trace 回执。不重复要求完整历史回归。所有诊断均未提交候选，原接受版本指针保持不变；本轮没有新的通信加速或通用绑定加速结论。

最终 [24 项检查回执](followup-metric-protocol-fixed-tests.json)、[双卡完整评价](metric-protocol-fixed.json) 和 [六个通信采样窗口](metric-protocol-observations.json) 均保留。此前窗口歧义导致的拒绝也见 [失败回执](metric-protocol-failed.json)，没有回写为成功。

`raw-evidence-manifest.json` 记录 box4-2 原始文件与 SHA-256；完整 trace、pstats、参考 tensors 和模型资产留在服务器。新的采集器可提供给后续 Agent 更多事实，不根据失败类别硬编码下一步动作。
