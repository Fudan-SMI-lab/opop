# P5 中间检查点：三项目适配、通信归属与采集降级

本目录是开发过程的真实证据，P5 的自动分布式候选和 P6 矩阵仍在运行，不能据此宣称全部计划完成。配置继续使用真实 `zhipuai/glm-5.3`，费用仅统计。

## 自动适配与独立质量

Qwen3-4B 的自动生成适配器在 B=1、prompt=256、new_tokens=32 下，通过两个连续步骤的完整 logits/token 检查。与原始就绪脚本保存的输出比较，每步 4,861,984 元素完全一致，误差为 0。第一轮真实 Agent 联合打包 gate/up 投影，72 个 phase/module 站点实际执行，但完整 logits 有约 27.2% 元素超容差（错误消息报告最大差 1.96875），因此被拒绝；token 相同不能代替 logits 检查。历史失败回执中的 `max_abs_error: 0` 只统计失败前已通过的叶节点；该反馈缺陷已修复，原始回执未改写。

GPT-2 124M DDP 的自动适配器保留 2 rank、microbatch=1、sequence=128、K=4 的正确 no_sync 累积和完整 SGD update。每步有效 batch=8、1024 tokens。独立校验调用原项目代码，按同一全局 batch 串行累积两步，对两个 rank 的每个梯度和更新参数逐元素比较。梯度最大误差为 6.799e-8，参数为 3.726e-9，均通过原容差；没有增加第二个参考模型到性能测量路径。

DDP 首次 Agent 把已知的合成数据、rank 绑定及采集器职责列为缺失信息。根据源码和已有 worker ABI 澄清后，真实请求修正了输出。随后发现 `torch.isfinite` 的内部算子带有 `inf` 标量参数；原序列化器错误地将合法签名当成非有限指标。现在签名用类型化文本保存 inf/nan，测量指标仍要求有限值。

## 通信与资源证据

- 原始 clean step 的 rank 中位数为 153.311 / 153.735 ms。单独的 plain Torch 通信诊断中位数为 63.708 / 69.645 ms。
- 跨 rank 匹配 13 对 allreduce，无未匹配项；每 rank 逻辑输入为 497,759,232 bytes。这是 collective 逻辑 payload，不是 PCIe 实际流量。
- NCCL 初始化日志记录实际 `SHM/direct/direct` transport；peer access 为 false。未推断 NVLink/RDMA 能力。
- 使用实际 c10d 输入存储与存活梯度的字节区间关系，再经 CPU annotation → CUDA launch correlation → NCCL sequence 关联，13/13 collective 在两个 rank 上均匹配到准确 bucket 及参数所属调用组。共享的 embedding/lm_head 参数保留多个名称和所属调用。一个 bucket 对多个组的成本仍为共享观测，不分摊成虚构的独立收益。
- Torch 采集每 rank 4004 个 kernel，全部关联；2677 个属于 backward，2456 个 autograd range 有 forward sequence 关联。208 个歧义 sequence 显式保留，不假定唯一来源。
- 配置中的 Nsys 经真实 collection-planner 选择后执行；每 rank 同样发现 4004 个 kernel 和 13 个 NCCL kernel。原始 SQLite/trace 留在服务器，文件路径见 JSON。
- 独立启用 Nsys GPU counters 得到 `ERR_NVGPUCTRPERM`。框架记录 `permission_denied`，随后关闭可选 counters 重新采集基础 trace，成功保留 81 个 ResNet graph kernel。

采集扰动不能忽略：完整发现带有 Python hooks，DDP wall 约 1.275–1.292 s；plain Torch 诊断 wall 约 206–215 ms，仍高于 clean step。它们分别用于发现/资源诊断，不能互换为 clean 项目时间。通信目标明确是独立诊断的 NCCL kernel duration，包含等待；不称为 wire time 或实测关键路径。collection-planner 的原始 rationale 是模型假设，其中对等待/传输可分离性的推断未经采集证明；框架没有输出这类分离指标，后续提示已明确该边界。

## 代码与验证

本检查点增加按 rank 合并的调用/依赖、各 rank 覆盖验证、同模块多 phase 安装、语义导航索引、逐调用 launch 配置、采集器注册接口及配置扩展、等待设备空闲的作业执行，以及冻结适配器的目标/保留负载复用入口。参数筛选同时检查原始基线约束，避免最快但不可行的参数遮蔽可行选项。旧默认入口保持原路由。

`tests.log` 为 33 个相关检查通过；重点是实际 functional/phase 分发、事务与 rank 覆盖、配对测量、约束和既有 TaskSearch 评价行为。没有将完整历史测试套件作为门槛。

`artifact-manifest.json` 记录公开产物原始哈希；新增 README/tests 不在该导出清单中。原始运行分别在 box4-2 的 `p4-qwen-ttft-a` 与 `p5-ddp-communication-a`，全部从真实调用生成，未使用预制目标组。
