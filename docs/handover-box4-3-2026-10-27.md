# box4-3 交接与环境耦合审查（2026-10-27）

结论：box4-3 满足当前 v6 开发、真实 Agent 调用、Qwen/ResNet 推理与双卡 DDP 通信实验的续做要求。就已核验的项目资料和运行能力，后续不依赖 box4-2 在线；可以退役旧实例。本次没有执行关闭或删除旧实例。

旧实例无法启动，不能做两机整盘逐文件对比。此结论依据克隆机实跑、先前保存的实验哈希和维护清单，不表示核验了所有无关历史文件。完整性校验的范围见下文。

## 连接、环境与工作位置

```bash
ssh -p 51018 root@connect.bjb1.seetacloud.com
```

本机现有 SSH key 登录成功。凭据沿用私有部署文件，不进入本交接文档。

| 项目 | 实测情况 |
|---|---|
| 主仓库 | `/root/autodl-tmp/dev/opop-v6`，`v6`；克隆初始 HEAD 为 `42be92e` |
| Python | `/root/autodl-tmp/dev-venv/bin/python` |
| 工作目录 | `/root/autodl-tmp/dev/v6-work` |
| 新部署配置 | `configs/box43-v6.yaml`，OpenCode 端口 19043 |
| 入口 | `/root/autodl-tmp/dev/v6-work/kernel-opt-v6`，已切换新配置 |
| Provider | 原私有配置，`zhipuai/glm-5.3`；真实调用通过 |
| GPU | 2 × RTX 4090，单卡 24,564 MiB，driver 580.105.08 |
| 容器资源 | 内存上限 240 GiB，CPU 配额 32 核，`/dev/shm` 120 GiB |
| 包 | Torch 2.13.0+cu129、Triton 3.7.1、Transformers 5.16.1、Optuna 4.9.0 |
| 磁盘 | 数据盘 150 GiB，初始空闲约 43.84 GiB；本轮验证后约 39.9 GiB。系统盘空闲约 3.75 GiB |

`nvcc`/`nsys` 不在默认 PATH，但实际安装可用；现有配置保留显式路径。CUDA 扩展已从数据盘新目录编译并加载，未使用 `/dev/shm` 存放共享库或永久证据。

新机器的卡号/UUID 仅作本次观测记录：

- 0：`GPU-6c11763a-1e00-9af3-e17d-274f5d7f78ba`
- 1：`GPU-cc3c3439-2f34-c8f6-34cb-c3e8feb69ae4`

框架没有固定这些值；可以换卡、换主机。清单：[初始环境](evidence/v6-box43-20261027/inventory.json)、[配置元数据](evidence/v6-box43-20261027/config-audit.json)、[入口检查](evidence/v6-box43-20261027/launcher-check.json)、[最终磁盘](evidence/v6-box43-20261027/space-final.json)。

## 真实验证

### 资料与会话

- 原 OpenCode 数据库 `quick_check=ok`，验证前有 477 个 session、12,342 条 message。
- 最新自主实验批次和 targeted-isolation 批次的已登记 exports/大文件，按先前 SHA-256 核验。
- 3,081 份压缩 worker 记录同时验证压缩件与解压后的原始字节；621 份去重文件验证内容哈希。
- 合计 **9,972 项检查通过，0 缺失、0 不一致**；四个实验根目录未发现断链。
- 这不是 9,972 个独立文件的全盘扫描；压缩件/原始内容是分别计数的检查。

见[校验摘要](evidence/v6-box43-20261027/integrity-summary.json)及[完整清单 gzip](evidence/v6-box43-20261027/integrity.json.gz)。

### 三项目迁移

输入来自 `v6-work/evidence-followup-20261021/fresh/`。通过正式 `optimize-project --reuse-evaluator --devices ... --prepare-only` 入口复用原 adapter，并在新 GPU 上生成独立 controls、参考和基线。随后额外以旧参考张量运行完整 evaluator 比较。原请求、旧控制结果和候选内容没有改写。

| 案例 | 选择设备 | 新 controls | 旧参考比较（每个连续步骤） |
|---|---|---|---|
| Qwen3-4B TTFT | `auto`，保持单卡 | 通过 | 4,862,240 个元素；两步均通过，最大绝对误差 0 |
| GPT-2 DDP 通信 | `auto`，保持双 rank | 两 rank 通过 | 每 rank 248,879,621 个元素；两步均通过，最大绝对误差 0 |
| ResNet-18 batch 32 | `1` | 通过 | 32,032 个元素；两步均通过，最大绝对误差 0 |

检查保留了原质量阈值和完整输出/梯度/参数范围。DDP 的通信目标采集也真实执行。这是迁移验证，没有生成新优化候选，也不把新旧机器计时差异算作优化收益。见[完整结果](evidence/v6-box43-20261027/models-result.json)与[执行脚本](evidence/v6-box43-20261027/models-controller.py)。

### Agent、执行器与 profiling

- 真实 Agent 生成 GPU helper 并实际执行；独立验证两卡计算、NCCL collective、注入 rank 失败后的回收与恢复。
- 设备索引 `1 0` 的真实双 rank 执行成功，证明卡号解析和顺序生效；receipt 分别保存输入选择器和实际绑定 UUID。
- 新编译 CUDA 扩展加载并计算 4,096 个元素，结果正确。
- Torch profiler CUDA/NCCL trace 可采集；Nsight Systems 实际捕获 kernel，并成功导出 SQLite。
- 新增真实 Agent 文档读取验证：`webfetch` 两次成功调用 PyTorch 官方文档。记录实际工具状态，而非只接受 Agent 的文字声明。
- 相关 CPU 测试 **13 项通过**。没有把整套旧回归作为开发前置条件。

两次真实 Agent 验证分别记录 provider 报价 0.06055337898 和 0.02134794792；只统计，不据此限制请求。金额是 provider 报告值，不是账单核算。

见 [P0 receipt](evidence/v6-box43-20261027/p0/public-receipt.json)、[工具实测](evidence/v6-box43-20261027/tools-result.json)、[真实 webfetch](evidence/v6-box43-20261027/agent-webfetch/public-receipt.json)、[相关测试](evidence/v6-box43-20261027/final-tests.json)。P0 在原 HEAD 运行；三项目和工具在设备选择修正后运行；文档访问和最终测试在 v6 权限默认值修正后运行。各 receipt 记录对应源码哈希。

代码修正已提交 `1339824` 并推送 GitHub；box4-3 同步到该提交，工作区干净。开发期间上传的部分文件带 CRLF，Git 部署后统一为 LF，所以逐字节源码哈希不同；同步前已确认变更文件按换行标准化后与提交内容一致。部署后相关 **13 项测试再次通过**，见[同步记录](evidence/v6-box43-20261027/github-sync.json)与[部署后测试](evidence/v6-box43-20261027/post-sync-tests.json)。

## 通信能力与当前环境限制

实际拓扑为同 NUMA 的 `NODE`；双向 `torch.cuda.can_device_access_peer` 均为 false。NCCL 日志记录 `SHM/direct/direct`。NVML PCIe tx/rx 速率、GPU/内存利用率、显存使用可读取。

速率采样不是单次 collective 的精确链路字节数；没有 NVLink 流量或跨 rank 时钟对齐的测量结论。P2P/NVLink 不可用没有阻断 DDP 执行。

NCU 实跑返回 **ERR_NVGPUCTRPERM**：驱动环境没有开放性能计数器。此项与[旧机器交接](handover-box4-2-2026-10-20.md)中的限制一致；基础 trace、实际计时和通信诊断可继续使用。没有把计数器权限作为框架启动条件，也没有假定所有 4090 永远不支持某项采集。见 [NCU receipt](evidence/v6-box43-20261027/ncu-result.json) 和 [实际错误](evidence/v6-box43-20261027/ncu-log.txt)。

## 不必要限制的检查与修正

| 检查对象 | 发现与处理 |
|---|---|
| 执行器只接受 UUID | 已支持当前机器的 `nvidia-smi` 卡号、`cuda:N`、UUID、`auto`/CPU；执行时解析并记录实际 GPU |
| 入口部署配置只接受 UUID | 已共用相同选择器；新项目默认探测当前设备，无须手工填 UUID |
| 冻结 evaluator 不允许换 GPU/锁目录 | 已允许同一任务卡数下修改 `devices`/`lease_dir`；自动在新环境运行 controls 与基线，无须重写 adapter |
| v6 继承 `webfetch: deny` | 已将 v6 默认设为 allow，并保留 read/edit/bash/external_directory 的既有可用性；真实工具调用通过。旧单算子默认值不变 |
| 固定主机名、box 地址、4090 型号准入 | 在 v6 project/runtime 主执行路径未发现此类检查；历史实验脚本中的具体路径属于实验记录 |
| P2P、NVLink、NCU 权限、PID 观测缺失 | 通常作为能力状态记录；默认不要求存在。若优化目标本身需要该指标，则必须能测出，不能用缺失值冒充结果 |
| Agent 请求/token/费用配额 | v6 使用 observe_only；部署 request timeout 为 None。provider 重试协调响应真实过载/恢复时间，不是总用量额度 |
| GPU 进程互扰 | 当前仍会等待其他 CUDA 作业结束，并拒绝混跑期间的性能结果；用于保证计时有效，不绑定旧硬件身份。未扩大限制 |
| 源码/adapter/候选内容哈希、完整质量比较 | 检查实际执行与待评估内容是否相同，避免把改代码或减任务当优化；无同主机要求 |
| Agent 快照 A/B 重放的源码/模型/工具配置检查 | 属于控制变量的专用实验入口。源码变化已有显式参数；普通新任务不要求用旧源码/模型。精确重放旧快照时仍须注意工具配置一致性，不能把变更后的权限默认为同一实验条件 |
| 旧产物绝对路径 | 仍有可移植性缺口：不同目录结构需重定位 source/assets/reference/store 路径；本次同目录克隆不受影响。没有新增路径白名单或强制原主机要求 |

UUID 用于本次作业的绑定、互斥与来源记录，不是硬件准入凭证。显式填写一个不存在的卡号/UUID 仍会报“设备不存在”，同时给出换机运行方法；不会偷偷把明确指定的卡换成另一张。

原 `--resume` 表示继续同一冻结运行，仍保留原请求与基线。换硬件时使用下述路径创建有新基线的运行，这是区分测量条件，不要求恢复旧 GPU。设备数量变化可能改变 batch/rank/状态语义，不冒充单纯换卡；需要相应任务/adapter。

```bash
/root/autodl-tmp/dev/v6-work/kernel-opt-v6 optimize-project \
  --request OLD_RUN/request.json \
  --reuse-evaluator OLD_RUN \
  --devices auto \
  --output NEW_RUN \
  --prepare-only
```

也可显式写 `--devices 1` 或 `--devices 1 0`。去掉 `--prepare-only` 会继续发现与优化流程；已有候选可通过既有 replay 接口在新运行上重测。

## 后续落点

本轮全部原始验证目录为 `/root/autodl-tmp/dev/box43-preflight-20261027`。三项目的新运行位于其 `models/`，已准备好新 controls。正式后续实验继续在数据盘工作，不把大产物转移到系统盘。

此前磁盘治理已减少 worker 元数据重复写入；原始 trace 和完整参考仍会增长。本轮新增约 4 GiB，主要来自 DDP 新参考与通信 trace。继续沿用[存储维护记录](v6-storage-audit-20261026.md)中的按清单归档策略；当前未引入自动删除或 API 使用限制。
