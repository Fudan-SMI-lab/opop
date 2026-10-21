# v6 真实项目验证任务包

这是三套可执行的模型项目与独立基线工具，用于 C3 开发和验收。它们不实现自动选组，也不预设待优化算子组。模型执行函数是项目理解的实际输入；已有基线工具用于交叉检查 Agent 生成的 evaluator。自动工作流由 `kernel_optimizer.project` 实现，入口与扩展方式见 [项目工作流](../../docs/v6-project-workflow.md)，真实实施进度见 [执行记录](../../docs/v6-execution.md)。本页下方的就绪检查不替代自动优化验收。

| 项目入口 | 模型／数据 | 目标 |
|---|---|---|
| `qwen_inference.py` | 本机完整 Qwen3-4B 预训练权重，BF16 SDPA，固定文本派生 token 输入 | 首 token 延迟；同一交互负载的解码吞吐；batch=8 总生成吞吐 |
| `resnet_inference.py` | 固定 revision 的预训练 ResNet-18，三张真实公开示例图片，FP32 | batch=1 延迟；batch=32 吞吐；同一 batch=32 下带吞吐下限的显存目标 |
| `gpt2_ddp_training.py` | GPT-2 124M 架构，确定性随机初始化和合成 token，真实双卡 NCCL 训练 | 训练步时；带整步时间约束的 NCCL kernel 时间目标 |

DDP 是模型系统训练负载，不是预训练模型的语言质量或收敛实验。三张图片也不足以声称 ImageNet 准确率。Qwen 固定生成长度且禁用 EOS 提前结束，输入文本按 token 重复／截断；它验证实际推理链路和参考一致性，不衡量回答质量。

## 部署与运行

GPU 环境使用已有 `/root/autodl-tmp/dev-venv/bin/python`，无需安装新 GPU 依赖。ResNet 使用已安装的 Transformers 实现，避免修改当前 PyTorch／TorchVision 环境。主机路径和 GPU UUID 只出现在部署文件；换环境时复制并修改 `deployment.box42.example.json`。

ResNet 资产可在能访问 Hugging Face 的机器上准备，要求 numpy、Pillow：

```powershell
python examples/v6_validation/prepare_resnet_assets.py --output <asset-dir>/resnet18
```

本次用本地现有 Anaconda 准备后上传 box4-2。模型 revision 固定为 `65a5785d9156231087c481e0c7dd33a5ff6f7e3e`；图片 revision、预处理定义和所有 SHA256 保存在 `asset-manifest.json`。图像解码和预处理在资产准备阶段完成。资产不提交 Git。

box4-2 上从 v6 worktree 运行：

```bash
cd /root/autodl-tmp/dev/opop-v6
/root/autodl-tmp/dev-venv/bin/python -u examples/v6_validation/run_suite.py \
  --deployment examples/v6_validation/deployment.box42.example.json \
  --output /root/autodl-tmp/dev/v6-work/validation-runs/<new-run-id> \
  --warmup 2 --repeats 5 --profile
```

可用 `--cases resnet_inference` 等选择子集。输出目录必须是新目录。套件逐个运行任务，并对每个任务的独立进程组设置超时、终止与强制回收；不应在另有 GPU／PCIe 密集作业运行时启动。脚本不实现跨作业的全局设备使用权，正式调度仍需 P0 的设备管理。

每个项目也支持单独运行，参数为 `--assets`（推理项目）、`--output`、`--workloads`、`--warmup`、`--repeats`、`--profile`。DDP 使用 `python -m torch.distributed.run --standalone --nproc-per-node=2 .../gpt2_ddp_training.py`。套件使用 GPU UUID 设置子进程环境，绕过旧 OpenCode server 剥离 CUDA 环境的问题；这不代表已修复框架的 Agent worker 绑定。

## 计量与正确性

- 所有主计时都排除模型加载、tokenization／图片预处理、主机输入传输、正确性检查和 profiling，测量固定 GPU 输入上的完整模型执行。它们不是含排队、网络和请求解析的服务 SLA。
- Qwen 首 token 边界是 prefill＋argmax 完成后同步；随后单独计时 `new_tokens-1` 个 cached decode。总生成吞吐计入第一个 token。参考 logits 使用固定上下文 teacher forcing，另检查逐步 greedy token 完全相同。保留完整 reference tensor 文件，数值阈值由 `workloads.json` 固定。
- ResNet 检查完整 1000 类 logits 和 top-1 一致。指标是每批调用延迟及该固定 batch 的 images/s；不同 batch 的比较不能冒充同一工作负载下的目标切换。
- DDP 使用 FP32、SGD、有效 batch=8、每步 1024 token、前 3 个 microbatch `no_sync()`。基线已经避免多余梯度同步。全量梯度对照逐个处理同一全局 batch 的单卡参考，另检查所有 rank 梯度完全相同及更新后参数。
- 每次训练测量前恢复初始权重；CUDA 同步和控制组 barrier 在测量边界明确执行。主分数取同一步各 rank 本地耗时的最大值，不相加不同 rank 时间；Gloo 控制 barrier 排除在步时外。
- `peak_allocated_bytes` 是该进程的 PyTorch CUDA allocated 峰值，不等于整卡显存、reserved、NCCL／驱动分配。跨 rank 取最大值，不求和冒充单卡显存。
- 所有任务包含 identity 重放和人工损坏输出的校验器负控制。负控制只证明当前参考校验能够拒绝明显错误，不证明未来 bundle 回装和自动 evaluator 已经正确；这些仍需完整候选注入验收。
- `--profile` 的 CUDA trace 与 clean timing 分离。DDP 单独采集 3 次各 rank 的 NCCL kernel 时间。通信目标定义为“每次各 rank 的 NCCL kernel duration 总和的最大值，再对重复取中位数”，包含设备等待，不能解释为线速时间、暴露通信或关键路径。此目标必须同时满足 clean step ≤ 原基线的 1.05 倍，并报告采集扰动。
- DDP 的 bucket 字节／次数在额外一次标准 allreduce hook 诊断中记录，不污染上述默认 reducer 的 clean timing 或 trace。字节数是每 rank 的逻辑 tensor payload，不能直接当成 PCIe 或实际传输总量。

五次 clean 样本和三次通信 trace 是就绪检查，不足以报告正式优化收益、稳定 p95 或尾延迟 SLA。正式实验先标定波动，再固定配对 AB/BA、重复数、停止条件和独立复核规则。

## 用来验证什么

`workloads.json` 声明 workload、目标、约束和保留形状。它不包含目标算子组、人工瓶颈排序或固定替换站点。第一轮以 ResNet 接通通用流程；再接入 Qwen；最后启用 DDP 分布式协议。进入后两个项目时冻结核心版本，新增任务不得依赖核心中的模型名分支。

验收时只给 Agent 项目执行代码、允许的运行入口、当前 workload／目标及资产定位。独立基线结果、reference tensors、保留 workload 和人工分析保留在评测侧。当前文件列出的保留 workload 是预留验证配置，不是已经做到访问隔离的盲测集；正式验收需要生成受限输入包并审计 Agent 文件访问。

必须记录各目标的自动发现结果、选组依据、真实生成和回装、模型指标、通信观察、接受／拒绝，以及反馈传入下一轮 Agent 请求的证据。同一交互负载的 Qwen TTFT／decode，以及同一离线 ResNet 的吞吐／显存约束，用于区分目标切换与输入变化。DDP 的 step 与通信目标也固定相同有效 batch 和精度。

目标切换不强制要求组选项或赢家一定不同；应检验框架是否依据实际影响选择、是否能解释相同选择，并通过候选跨目标交叉评测验证，而不是硬编码目标到组的映射。不得把固定 batch 下互为倒数的 step time 和 tokens/s 当成两个独立目标。

完整项目验收还要求：同一测试版本中至少两个组的自动候选同时生效、组合质量与项目指标复测、联合边界改写、执行计划变更、动态分组及原子回退。三个项目的任务包用于提供独立参考，实际优化范围仍由框架自动确定。候选可能只在联合修改后合法或有收益，不要求每个子项先被单独接受；不能运行不闭合的子编辑来伪造单项对照。

新增状态检查覆盖 Qwen 连续请求／KV cache reset 和 DDP 连续训练步骤；正式实现时生成独立专项证据。本目录当前脚本仍是已经执行的就绪基线工具，尚未实现上述候选搜索、事务及连续状态专项验收。详细要求见[验证方案第3.3节](../../docs/v6-validation-projects.md#33-多组与整个计算流程的验收)。

正式验证流程和当前运行结果见 [验证方案](../../docs/v6-validation-projects.md)。
