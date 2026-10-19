# v6 开发环境与通信能力预检

日期：2026-10-20。对象：AutoDL box4-2。代码基线：`8de73b4f673a82f277dac539f04aa7da4ae25f70`。正式实施计划见[通用 C3 与通信资源计划](plan-v6-general-c3-and-communication.md)。

box4-2 可以访问，双卡 NCCL、PyTorch profiler 和 Nsight Systems 基础时间线可执行。硬件计数器因权限不足不可用，PCIe 仅验证了设备级粗粒度速率。已准备独立 v6 工作树和私有运行目录。此次运行证明开发环境与若干执行组件可用，不代表通用项目工作流已经完成。

## 1 连接和开发位置

```sh
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no \
  -p 30726 root@connect.bjb1.seetacloud.com
```

| 用途 | 路径或状态 |
|---|---|
| 本地 v6 | `D:\Pyhon_projects\opop\v6`，分支 `v6` |
| 远端 v6 | `/root/autodl-tmp/dev/opop-v6`，分支 `v6` |
| 原开发树 | `/root/autodl-tmp/dev/opop`，`v5-C3@8de73b4`，核查时干净 |
| Python | `/root/autodl-tmp/dev-venv/bin/python` |
| v6 工作目录 | `/root/autodl-tmp/dev/v6-work` |
| 启动器 | `/root/autodl-tmp/dev/v6-work/kernel-opt-v6` |
| 配置 | `/root/autodl-tmp/dev/v6-work/configs/box42-v6.yaml` |
| 私有 provider | `v6-work/.opencode/opencode.jsonc`，服务器内复制，`0600` |
| 私有调用数据 | `v6-work/private-runs` 与 `v6-work/private-runtime` |
| 本次远端原始证据 | `v6-work/preflight-20261020` |
| 本地公开回执 | [evidence/v6-preflight-20261020](evidence/v6-preflight-20261020/) |

两个 v6 分支均从同一基线创建，采用独立 worktree。文档和预检脚本是本轮新增内容；未实现 v6 项目工作流。本轮没有推送 GitHub。

旧 venv 的 editable 目标未修改。v6 启动器和本次驱动显式设置 `PYTHONPATH`，实测 `kernel_optimizer.__file__` 指向 `dev/opop-v6/src`。后续依赖升级应使用独立环境并记录 lock，不能影响旧结果的复现。

单卡入口示例：

```sh
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/dev/v6-work/kernel-opt-v6 doctor
```

当前这个启动器仍运行 v5-C3 基线 CLI，不包含尚未实现的 `optimize-project`。后续工作从正式计划 P0 开始，不直接启动旧 C3 campaign 作为通用流程。

现有 OpenCode server 默认剥离父进程的 CUDA 环境变量，因此上面的 `CUDA_VISIBLE_DEVICES` 约束主进程，不足以证明 Agent 任意启动的 GPU 子进程也已隔离。本次预检独占两卡；后续并发实验前，P0 必须在受控 helper/worker 入口重新绑定设备，并核对实际 PID→GPU UUID。不能以启动命令中的 CVD 字段代替这个验证。

## 2 硬件与软件

- hostname：`autodl-container-e8e84ea25b-cffd5c81`。
- GPU：两张 NVIDIA GeForce RTX 4090，分别为 `GPU-4bead555-f80c-df43-5a46-daa7a0ec80b2` 和 `GPU-69c59246-e383-f1f1-1f46-5e2dfebb858d`，各 24564 MiB。
- driver 580.105.08；nvcc 12.8.93；torch 使用的 CUDA runtime 为 12.9。这三个值不是同一个版本字段。
- Python 3.12.4；torch 2.13.0+cu129；NCCL 2.29.7；Triton 3.7.1；Transformers 5.16.1；Optuna 4.9.0；Pydantic 2.13.5。
- OpenCode 1.18.29；Nsys 2025.3.1；NCU 存在于 `/usr/local/cuda/bin/ncu`，不是默认 PATH 中的工具。
- 宿主机可见 128 逻辑 CPU，但 `cpu.max=3200000 100000`，相当于 32 核 quota；`memory.max=257698037760`，即 240 GiB；`memory.high=255550554112`，即 238 GiB。
- 核查时数据盘 `/root/autodl-tmp` 总计约 122 GiB、空闲约 73 GiB，系统盘空闲约 28 GiB。

`doctor` 通过 GPU、worker、OpenCode、KernelBench、torch/Triton 和 CUDA backend 检查。本次新工作目录没有硬件资源校准缓存；正式 C1/C2 运行前需要在隔离目录执行校准，不能把 doctor 通过当作已完成带宽和算力校准。

## 3 文件盘点与历史资产

已只读检查六棵代码树，详情见[inventory.json](evidence/v6-preflight-20261020/inventory.json)。`kb-4090/opop` 和 `opop-workspace/opop` 也存在历史未提交／未跟踪内容，不能以“旧树只读”误解为“旧树干净”。没有修改这些树或旧 venv。

完整 Qwen3-4B 资产的实际位置是：

```text
/root/autodl-tmp/c3-qwen3-4b-operator-goals/assets/Qwen3-4B/
  1cfa9a7208912126459214e8b04321603b3df60c/
```

其中三个正规 safetensors 分片分别为 3,957,900,840、3,987,450,520、99,630,640 字节，合计 **8,044,982,000 字节**。本次检查了文件位置和大小，未重新加载整个模型验证质量。原交接中把 `range-resume` 描述为权重入口的说法应修正：该目录主要是下载恢复记录，部分以 `.safetensors` 结尾的路径实际是目录。精确清单见[model-assets.json](evidence/v6-preflight-20261020/model-assets.json)。

六组通信实验目录均存在。五组 SHA256SUMS 共核验 **207** 个文件，最初通信探测的 manifest 另核验 **174** 个文件，合计 **381** 项无缺失、无哈希不匹配。见[historical-integrity.json](evidence/v6-preflight-20261020/historical-integrity.json)。

本地交接 ZIP 为 34,836,996 字节，402 个成员，清单覆盖 401 项；CRC 和每项 SHA256 均通过。ZIP SHA256 为 `d48b49c4616bceba1adb1d2eecf634cee4581b53d02595562804fb26a16d2741`。见[archive-verification.json](evidence/v6-preflight-20261020/archive-verification.json)。没有重跑旧 DDP 或 Qwen 模型优化实验，也没有改写旧报告。

## 4 新执行的真实通信检查

所有短检查使用新的输出目录，复用历史 probe 的只读副本。来源和代码哈希在[preparation.json](evidence/v6-preflight-20261020/preparation.json)，命令、退出码和耗时在[status.jsonl](evidence/v6-preflight-20261020/status.jsonl)。

| 检查 | 本次结果 | 能支持的结论 |
|---|---|---|
| 拓扑和访问 | `NODE`；P2P read/write 为 `CNS`；torch 双向 peer=false | 当前双卡不提供直接 CUDA peer access |
| NCCL all-reduce | 三档 payload，逐次全元素结果为 3，作业成功 | 当前 NCCL 双 rank 通信和数值检查可运行；不是完整模型性能结论 |
| PyTorch profiler | 2 ranks × 21 contexts，**42/42** raw 窗口均有 CUDA 与 NCCL kernel | 独立 profiler 的重复生命周期通过；尚不能替代 Agent/helper 路径验收 |
| Nsys CUDA/NVTX | 6 个标记窗口，6 个 NCCL kernel、96 个计算 kernel；两个 PID 和设备均匹配 | 基础双卡时间线可采，观察到通信与计算区间重叠 |
| 实际 transport | 新日志 channel route 为 `SHM/direct/direct` | 本次路径是共享内存通道，不是由名称推断的 P2P 或 RDMA |
| NCU counter | exit 1，`ERR_NVGPUCTRPERM` | 工具存在，但硬件计数器无权限 |
| Nsys GPU Metrics | exit 1，两卡均 Insufficient privilege | 不应按成功采集或零利用率处理 |
| PCIe dmon | 负载期间有非零 RX/TX 样本 | 秒级设备流量可观察；不是 per-operator 字节 |

Nsys 三个窗口合计的通信区间并集为 rank0 **5816.470 µs**、rank1 **6402.254 µs**；计算并集分别为 **358.157 / 847.689 µs**；重叠分别为 **307.732 / 796.818 µs**。这些是单次诊断数据，不用于宣称通信占比、纯传输时间、瓶颈归因或优化收益。

Nsys 的完整 report 含初始化和预热，因此全文件的 kernel 数量大于上表标记窗口的计数。聚合脚本按 PID/device 与标记窗口选取区间，不能把这一最小探针分析称为已完成通用算子关联。正式实现还要加入 launch correlation、collective 身份和覆盖校验。

当前 Nsys 的完整可执行路径：

```text
/root/autodl-tmp/tools/nsys-2025.3.1/opt/nvidia/
nsight-systems-cli/2025.3.1/target-linux-x64/nsys
```

它的 `profile --help` 未列出新版高级 NCCL trace。PATH 中也没有 dcgmi 和 all_reduce_perf。本次没有安装工具、更换驱动或修改计数器权限。

## 5 真实 provider 调用与验收边界

复用服务器现有配置，通过 v6 工作树运行真实 `agent-smoke --module generator --task level1:19`，使用 `zhipuai/glm-5.3`。调用最终 **exit 0**，耗时 **795.910 秒**，一次框架 attempt 返回两份候选，通过 generator 的结构检查。会话中有 48 个工具调用，5 个工具输出包含 CUDA OutOfMemoryError；这些失败没有从回执中删除。见[调用回执](evidence/v6-preflight-20261020/provider-receipt.json)。未额外执行候选完整正确性与性能验收。

费用记录还暴露了口径差异：框架终端报告约 **$0.0256**，OpenCode 数据库中各 assistant 消息记录费用之和为 **$0.72339384**。源码当前取最终响应的 `info.cost`，因此不能把终端数值当整个多轮会话的总费用；两者均未与 provider 账单独立核对。v6 的预算和回执需要按完整会话归集，并防止重试重复计数。

结束后两卡均为 0 MiB／0% 使用率，没有 GPU compute 进程，OpenCode 的 15360 端口已关闭。

这个发现不改变已有单算子框架的默认行为，但要求 v6 新模式为评测和自测设置明确的资源预算、阶段状态和失败分类。尤其需要区分候选错误、fixture／比较代码自身的峰值内存不足，以及 profiler 失败。上述短通信 probe 并未覆盖完整 Agent→helper→分布式 worker 生命周期，该路径列为正式计划的真实验收项。

provider 配置、OpenCode 数据库和 sandbox 中可能含凭据，留在服务器私有目录；本地只保存白名单字段和非敏感回执。预检未修改旧框架算法、prompt 或质量阈值。

## 6 复核和后续起点

本地预检脚本位于[evidence 目录](evidence/v6-preflight-20261020/)，是本次特定机器审计的来源记录。`prepare_remote.py` 要求输出目录尚不存在，不能覆盖已有回执。重复预检须换新的输出目录并记录新身份；`run_remote_checks.py` 会执行真实 GPU 和付费 provider 请求，不能当作纯离线解析脚本运行。

后续首先完成正式计划 P0：固定完整源码快照、校准缓存、旧行为回归与跨平台测试数据定位，然后实现公共契约和运行接口。此前交接的 2743 passed／11 failed 是历史整套测试结果；本次未重新运行整套 pytest，也未将历史测试记录冒充 v6 验收。
