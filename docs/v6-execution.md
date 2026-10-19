# v6 执行记录

更新：2026-10-20。正式设计见 [实施方案](plan-v6-general-c3-and-communication.md)，真实项目见 [验证矩阵](v6-validation-projects.md)。当前分支为 `v6`，开发与 GPU 验证环境为 box4-2。本文件是持续更新的执行清单，不是新一轮等待批准的方案。

## 执行约定

- 按 P0–P6 连续实施，使用真实运行反馈调整接口、实现和验证；不在每阶段结束时重新请求开始下一阶段的许可。
- 开发 Agent/API 的费用、请求数和 token 只统计，不设用量额度或总时长截止；不根据费用自动停止、降级模型或减少验证。复用 box4-2 的实际 provider 配置。
- 旧输入继续走旧默认路径；必要回归按受影响行为选择。完整历史套件不作为门槛，不修复无关、已退役的历史实验才能开始开发。
- 保留数值、有效 batch、状态、候选实际执行、回装／恢复和测量完成条件的有效检查。它们决定优化结果是否成立。
- 每阶段将代码、相关验证摘要、有效配置的公开部分、产物哈希和已知限制提交并推送 GitHub。原始 trace、模型和私有调用数据留在服务器，通过清单引用。
- 正式对照实验预先固定搜索／GPU 测量协议以保证比较公平；其结束条件不成为开发 Agent/API 限额。

## 当前状态

准备阶段已提交并推送：[`104e44a`](https://github.com/Fudan-SMI-lab/opop/commit/104e44a)。这是环境、设计和三项目就绪基线的存档，不是 P0–P6 完成提交。

| 项目 | 状态与证据 |
|---|---|
| box4-2 与独立 v6 工作树 | 已核验；[预检记录](v6-box42-preflight-20261020.md) |
| 三个真实任务 | Qwen3-4B、ResNet-18、GPT-2 DDP 均已完成就绪运行；不是自动优化验收 |
| GitHub 存档 | v6 分支已建立，SSH 推送成功；commit author 为 ZihangZ |
| 开发请求截止时间 | 新增 `request_timeout_s: null`，保留旧默认 3600 秒；box4-2 开发配置同时关闭派生 idle 截止 |
| 本次真实调用 | `zhipuai/glm-5.3` 成功读取 ResNet 源码并生成分析，51.01 秒；4 条 assistant 消息，完整会话报告成本 $0.0269474；[回执](evidence/v6-development-20261020/provider-receipt.json) |
| 必要回归 | 203 passed；包括旧入口、资源反馈、worker/store、配置和本次变更 |
| 完整会话用量 | P0 已实现按 session/message 去重的只统计账本；[P0 证据](evidence/v6-p0-20261020/README.md) |
| 通用项目工作流 | P1 事务、P2 ResNet 自动评测及 P3 自动发现／联合候选通道已验证；P4–P6 继续实施 |

本次测试删除与取舍：

1. 删除 `test_c3_quality_arithmetic.py` 中 7 个依赖固定 `host-A/native-raw/.../oracle.json` 的参数化历史回放用例。保留不依赖历史文件的数值边界测试；诊断回放工具仍可按旧环境单独使用。
2. 删除 `test_c3_manual_task.py` 中 `torch not in sys.modules` 的全局断言。该断言受其他测试是否已导入 torch 影响，不能证明当前入口的行为；保留入口加载和无运行状态时返回无效结果的检查。
3. 首次扩大选择时，29 个旧 manual-task 用例因未随仓库提供的历史 `contract.json` 在 fixture 准备阶段失败。保留失败日志；本次只运行受修改的入口用例。没有把这些失败标成通过，也没有删除仍有语义价值的其余检查。
4. 未发现独立、需要整体保留的“安全测试套件”。后续按现行要求删除过时检查，不以名称筛选删除配置正确性或优化质量测试。

真实请求的新反馈：Agent 在仅收到源码时提出“增加 batch”作为优化假设。对已固定 workload 的比较，该改动不合法；精度、计时边界也不能由候选自行放宽。P2 的请求必须包含冻结 contract 与 workload，P3/P4 的候选安装与评测须独立检查。该回复是未经性能验证的分析产物，不作为候选被接受；详见[本次结果说明](evidence/v6-development-20261020/README.md)。

## 可执行阶段清单

| 阶段 | 按顺序实施的工作 | 阶段完成证据与存档条件 |
|---|---|---|
| P0，完成 | 会话用量只统计；受控 GPU helper 设备绑定、PID→UUID 与清理；必要旧行为检查 | [真实 provider、双卡执行、故障恢复证据](evidence/v6-p0-20261020/README.md)；提交 `bd9192a` |
| P1，完成 | 版本化四种 scope、执行计划、组注册表、完整候选与事务；rank 生命周期 | [真实 Agent 联合编辑、质量负控制、提交及恢复](evidence/v6-p1-20261020/README.md)；提交 `bc7a61e`；属于基础设施控制实验 |
| P2，ResNet 路径完成 | 实现 `optimize-project` 入口及真实项目理解／评测生成；输入、状态和测量协议 | [完整 logits 与独立原项目基线一致；identity、负控制及 reset](evidence/v6-p2-20261020/README.md)；提交 `7ff369b`；其他模型随后验证 |
| P3，最小通道完成 | module／functional 自动发现、依赖与包含闭包、真实选组与联合候选、schema 自动修复、AB／BA 评价及逐组反馈 | [真实联合候选](evidence/v6-p3-20261020/README.md)：20 个站点执行并通过质量；性能不稳定被拒绝；核心初始提交 `589144f` |
| P4 | 旧单算子内循环通过适配器接入；实现多组共存、区域／执行计划改写、原子回装和项目测量；反馈路由与重选 | 同一项目至少两个组候选生效；联合与执行计划候选实测；至少两轮真实请求消费前轮反馈；失败可恢复 |
| P5 | 在公共执行路径加入通信能力发现、PyTorch/Nsys/transport/PCIe 采集和 collective 关联；跨 rank 改写、完成条件、重叠及故障回收 | GPT-2 DDP 的真实候选通过梯度／参数和项目目标验证；缺测降级、单卡跳过、rank 故障整体退出有证据 |
| P6 | 冻结核心后接入其他项目／保留输入；运行目标切换、跨目标交叉评测、组合退化、联合策略及反馈对照；完善复现入口 | 三项目与整流程矩阵有结果，未预注册任务无需修改通用核心；分别报告收益、失败、无改善和尚不可测项 |

P1 的分布式执行设施与 P5 的通信采集共用数据模型；可提前做真实双 rank 探针，但不能把探针成功当作自动优化闭环成功。先贯通 ResNet，再接入 Qwen；DDP 为首个通信主案例。Qwen TP2 对单卡质量不一致仍是扩展案例的待解决问题，不阻塞前述三项目主线。

每个阶段使用相同的执行循环：实现最小可运行路径 → 必要局部检查 → 真实 Agent／GPU 运行 → 读取原始失败或效果证据 → 调整实现 → 完成验收 → 提交并推送 → 更新本表。阶段标准不因已经花费的 API 用量而降低。

## 命令与运行位置

本地仓库：`D:\Pyhon_projects\opop\v6`。服务器仓库：`/root/autodl-tmp/dev/opop-v6`。服务器工作目录：`/root/autodl-tmp/dev/v6-work`。Python：`/root/autodl-tmp/dev-venv/bin/python`。

运行前显式设置源代码位置，避免旧 editable 安装导入 v5-C3：

```sh
cd /root/autodl-tmp/dev/opop-v6
export PYTHONPATH=/root/autodl-tmp/dev/opop-v6/src:/root/autodl-tmp/dev/opop-v6
/root/autodl-tmp/dev-venv/bin/python -B -c 'import kernel_optimizer; print(kernel_optimizer.__file__)'
```

box4-2 的 `configs/box42-v6.yaml` 使用以下开发覆盖；该文件含部署配置，留在服务器：

```yaml
opencode:
  request_timeout_s: null
  idle_abort_frac: 0
```

本次验证的精确测试选择与运行结果保存在 [回执](evidence/v6-development-20261020/test-receipt.json)。后续按变更调整选择，不把当前测试数量设成新的固定验收目标。真实请求探针见 [probe_no_deadline.py](evidence/v6-development-20261020/probe_no_deadline.py)；它只验证请求配置和真实源代码检查，不验证自动分组或候选性能。

GitHub 推送优先使用现有 SSH，不修改其他 worktree 共用的远程 URL：

```sh
git -c remote.origin.pushurl=git@github.com:Fudan-SMI-lab/opop.git push origin v6
```

实现提交说明应写明实际问题、行为变化、相关验证及未完成边界。阶段完成后无需等待新的计划模式或逐阶段批准；后续工作继续按本表推进。
