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
| 通用项目工作流 | 三项目自动评测已通过；跨目标／保留集、整版回传、可行探索分支、真实限流恢复已接入，P3–P6 的剩余矩阵继续运行 |

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
| P2，三项目适配通过 | `optimize-project` 入口及真实项目理解／评测生成；输入、状态和测量协议 | [ResNet 独立输出一致](evidence/v6-p2-20261020/README.md)；[Qwen 全 logits/token 与 DDP 两步完整梯度/参数独立校验](evidence/v6-p5-20261020/README.md)；不等于三项目优化全部完成 |
| P3，自动探索验证中 | module／functional 自动发现、依赖与包含闭包、真实选组与联合候选、schema 自动修复、AB／BA 评价及逐组反馈；新增可行非贪心前沿和四轮真实试验 | [真实联合候选](evidence/v6-p3-20261020/README.md)；[分支与整版评价](evidence/v6-p6-branches-20261020/README.md)；核心初始提交 `589144f`，分支提交 `7936d90` |
| P4，组合、整版回传与资源响应通过 | 旧 TaskSearch/TPE 项目适配；多候选完整回装、质量／覆盖分离、整版独立比较与 CAS；提交后重分组 | ResNet 两个 bundle 同时执行但组合变慢，拒绝；另一个 Graph 版本整版复测后提交；外层 Graph 绕过旧 bundle 的案例输出正确但覆盖失败，已单独归因；[组合证据](evidence/v6-p6-branches-20261020/README.md)与[资源反馈后接受版本](evidence/v6-feedback-closure-20261020/README.md) |
| P5，采集与归属通过，优化继续 | 通信能力发现、Torch/Nsys、实际 transport；13/13 collective 的跨 rank 匹配及梯度存储/bucket/组多对多关联；硬件计数器权限不足时实测降级 | [真实中间证据](evidence/v6-p5-20261020/README.md)；DDP 自动候选仍在运行，不能以采集成功替代完整优化闭环 |
| P6，进行中 | 八目标均已进入真实优化；新纯函数入口和保留 shape/seed 已验证；跨目标、组合、策略与反馈对照继续 | [目标切换及保留集的中间结果](evidence/v6-p6-branches-20261020/README.md)；正式策略对照 B 使用独立冻结代码，尚未全部结束 |

P1 的分布式执行设施与 P5 的通信采集共用数据模型；可提前做真实双 rank 探针，但不能把探针成功当作自动优化闭环成功。先贯通 ResNet，再接入 Qwen；DDP 为首个通信主案例。Qwen TP2 对单卡质量不一致仍是扩展案例的待解决问题，不阻塞前述三项目主线。

每个阶段使用相同的执行循环：实现最小可运行路径 → 必要局部检查 → 真实 Agent／GPU 运行 → 读取原始失败或效果证据 → 调整实现 → 完成验收 → 提交并推送 → 更新本表。阶段标准不因已经花费的 API 用量而降低。

## 最新实际运行与调整

资源采集检查点 `3c0ea60` 已推送 GitHub 并同步 box4-2。使用方式、模块接口、恢复入口及实验策略见 [项目工作流说明](v6-project-workflow.md)。整版回传阶段 28 项、后续资源阶段 36 项相关检查通过，统计不与之前重复用例相加。

正式比较固定在 `/root/autodl-tmp/dev/opop-v6-freezes/7936d902b88b`，源码哈希 `cae6cc6e6591fba0d0262db4bfe03575167fed086547483768348a9e9d3ed8e4`。工作路径继续在 `v6`；冻结目录为该提交的 detached 实验副本，不随开发修改。对照包含 ResNet 的 whole/leaf/score_only/local_time、Qwen 与 DDP 的 whole/leaf/score_only，以及 DDP 关闭额外通信观测的两轮协议。各路径共享原始参考和采样规则，实际 API／GPU 消耗单独记录，不声称消耗完全相同。此前 A 组中的 HTTP 429、参数表达式错误及源码变化均保留，不冒充完成的正式对照。

真实 provider 429 已触发共享排队、冷却和逐步恢复并发；不设置开发用量或重试次数上限。原 session 中已恢复 DDP 通信、DDP 整步、Qwen TTFT 和信号项目角色。DDP 通信第二候选修复 Work 接口后，第一步正确、第二连续步梯度失败，继续拒绝；框架没有放宽阈值。真实 `implies` 参数表达式失败触发生成阶段语法校验与原会话修复，修复后的候选能完成项目评价，但明显变慢，继续拒绝。

Qwen 一个 decode 候选虽然质量通过，其 62 次调用全部经过父实现，诊断内核数量也与原始相同。该结果仅证明安装与回退兼容，不作为融合优化生效证据。新增父实现调用和候选自报路径记录；自报路径必须由 trace 佐证，不能替代独立执行覆盖。

通信采集新增实际可读取的 NVML PCIe TX/RX 速率、设备活动比例和设备显存读数；双 rank DDP 已完成三个独立诊断窗口。实际 Collector Agent 选择了 `torch/nsys/device_telemetry`，NVLink API 返回不支持并明确保留。PCIe 原生采样窗口和每次读取区间分别保存，不把速率再次差分或换算成逐算子 wire bytes。匹配两种实际 payload 的独立 SUM allreduce 参考校准已在双 rank 通过；它不表示模型关键路径或性能下界。详见[新增资源与通信证据](evidence/v6-resources-20261020/README.md)。

TaskSearch 的参数响应现在带有实际替换内核的启动数、诊断时长、寄存器／线程和共享内存，目标 J／吞吐／workload 常量不再自动标成资源。真实三档 `num_warps` 实验得到非零资源差分，完整质量通过但配对收益仍未超过噪声，继续拒绝。下一轮真实 Selection Agent 已消费这些字段并改变启动策略；最终候选仍须测量。本阶段 36 项相关检查通过。

该下一轮候选随后完成配对验收：纯函数项目的融合核 Graph replay 将父版本 0.0792845 ms 降至 0.0498955 ms，已提交；保留形状走普通 Graph 回退且未获可靠收益。Qwen 的真实 decoder-layer 候选在独立固定实现下通过复测，TTFT 27.384616→22.3693795 ms，完成提交及重分组，不补写为中断的 A 组正式比较。详见[反馈闭环与复核](evidence/v6-feedback-closure-20261020/README.md)。

大图重分组新增调用索引：同一真实 Qwen trace 的 15,059 个组与旧算法全部一致，索引及全部查询约 0.313 秒。失败候选源码另作为独立历史附件提供给下一轮，避免只回传 artifact ID；新附件的实际消费回执继续验证。正式 B 组继续使用原冻结实现，不混入这些协调器调整。

剩余交付重点：完成自动探索前沿和各项目多候选共存的真实验证，汇总稳定协议的对照／消融，补齐最终逐项验收与可复现证据。未完成项不会因部分案例已经加速而自动标为完成。

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
