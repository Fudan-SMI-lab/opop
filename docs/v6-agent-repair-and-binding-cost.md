# Agent 修复与绑定成本采集（2026-10-21）

## 责任边界核验

DDP 的约 100 ms 增量主要来自框架安装的 functional 绑定，不是 adapter Agent 生成的训练代码。
实际链路为：

1. adapter Agent 生成 `prepare/reset/run/observe/metrics/models`，复现原项目。
2. selection Agent 选择 group 与受影响调用闭包。
3. candidate Agent 通过 proposal 与 `build(context)` 声明实际替换站点。
4. `evaluation._worker_entry` 创建 `BindingSession`；后者为 functional 替换启用 `TorchDispatchMode`。
5. evaluator 在完整 `adapter.run()` 周围激活该 mode。

原 DDP 通信候选替换两个 `c10d.allreduce_.default` 站点，每步分别调用 1 次、12 次。
两者的 `site_parent` 都是 `project.invoke|run` 的 ID `5d7d6bb13691b02f533f62b0`。
它不是可解析的 module forward 边界，现有 scoped 后端仍会回退到 legacy。

因此改进责任分为：框架公开实际绑定机制与可测成本；selection/candidate Agent 根据完整项目目标选择
合适边界并生成实现。不能通过让 adapter 优化原基线来消除这项成本，也不能把所有退化归因于候选算法。
本轮先验证现有 module/project 边界能否解决问题，尚未新增绑定后端或改写搜索调度。

## 已实现的通用改动

- 新增可选 `binding_cost` 采集器，接入已有 collector registry 和采集规划 Agent。
  对保留父版本的新增 bundle，执行 parent/passthrough、passthrough/parent 四个独立端点。
  每个端点保留原连续质量检查、预热、重复测量与设备租约。透传省略新增候选的 build 和计算。
- 无候选或替换了父 bundle 时返回 `not_applicable`，不推断不存在的成本；执行或质量失败时保留端点证据，
  不给出耗时对照结论。成本属于完整调用及其组合，不分摊到算子组，不从验收得分中扣除。
- 修正诊断透传：只安装实际声明的 rank-local replacement calls，不把整个 affected closure 都安装成替换。
  这避免 module/project 候选的子算子闭包人为引入全局 functional dispatch。
- binding contract 解释 adapter、选择、候选和执行层的责任，以及 scope 名称与实际绑定站点的区别。
- 可选 `profiling.feedback_agent_plan=true` 复用现有采集规划 Agent，在候选实际结果出来后重新选择
  采集项。参数筛查全部未入选时，向 Agent 提供每个实际参数版本的小型结果摘要，由它明确选择一个
  revision 和采集项；不预设失败分类或下一步优化动作。默认关闭，旧输入保持原采集调用方式。
- 新增 `scripts/v6_diagnostic_feedback_experiment.py`：冻结原失败候选、父版本、常规反馈和任务，
  使用独立真实会话比较单项 `supplemental_evidence` 增补，再交给原参数搜索与配对 GPU 验证。
  结果仅发布到隔离实验分支。新版本按已有机制将完整 collector 记录放到索引附件中。

## 当前验证记录

第一批 collector、snapshot、observation 和 binding 相关测试在 box4-2 通过 14 项。
修正透传站点后，binding_cost、scoped、functional 相关测试通过 9 项。
本机完整 snapshot 测试曾因缺少 optuna 无法通过，已使用实际 worker 环境完成验证。

新采集器真实 DDP 探针的两个 AB/BA 对照，rank 最大完整步中位数分别为：

| 对照 | parent（ms） | binding passthrough（ms） |
|---|---:|---:|
| AB | 143.812453 | 261.473880 |
| BA | 155.667821 | 245.851864 |

四个双 rank 端点完整质量检查通过。它们证明存在显著绑定成本；端点变化包含测量波动，
不把某个差值当作固定开销常数。探针未提交优化版本。

## 真实 Agent 对照

首批实验固定框架 `95280f2`，仅新增独立实验入口，保持原框架执行行为。
每个任务从原失败现场建立 control/treatment 新会话，首次各一份，不声称统计显著性。

| 实验 | 角色 | treatment 的唯一额外事实 |
|---|---|---|
| Qwen TTFT | candidate | capture 改变 mask/GQA 分支的实测因果证据 |
| Qwen throughput | candidate | 有效 KV 窗口数值差异与缓存类型引发回退是独立问题 |
| DDP step | candidate | 实际 boxed ReduceOp ABI 及 P2P 从未执行的证据 |
| DDP communication | selection → candidate | 候选 build/计算完全省略后仍存在的大额绑定成本 |

这些是开发者提供诊断的受控修复实验。新的自主端到端运行需要另外从原项目入口发起，不能混称。
原质量阈值、通信目标和完整步约束保持不变，API 使用仅统计。

运行目录：`/root/autodl-tmp/dev/v6-work/agent-repair-20261021`。
此记录会在实验完成后补入实际决定、执行路径证据、独立复测和未解决问题。

## 中途环境故障与恢复边界

本轮遇到 box4-2 的 `/root/autodl-tmp` 122 GB 数据盘耗尽，四组修复实验和第一批新入口验证中断。
保留中断日志与已经生成的候选，不把缺少完整终态记录的运行当作成功实验。
恢复的 control 使用相同候选字节重新执行完整 GPU 验证；新 treatment 从冻结输入重新发起真实会话。
新入口自主验证另行重跑，明确区别于开发者恢复的修复实验。

空间处理先对已结束历史目录中 819 个内容完全相同的文件逐项校验 SHA-256，以硬链接保留原路径和字节，
共释放约 6.16 GiB（跨系统盘和数据盘）。活动实验、运行数据库和 provider 配置不参与去重。
一个传输 ZIP 在确认本机副本 SHA-256 相同后删除，原诊断文件仍保留。
早期 `stage2-autonomy-0e4c751` 批次的完整冷归档已转存本机，4,445,620,276 字节，
SHA-256 `0f24eae78103c62d2f8e1908d4e769a3f180fbea1f86357c5b0ef2e8fb204d2c`。
本机与服务器归档哈希一致后，移除服务器该批次的展开副本，原路径保留 `ARCHIVED.json` 恢复说明。
数据盘恢复至约 14.6 GiB 可用；四组修复会话及采集探针继续运行。

## 采集规划输入的最小修正

真实的候选后采集规划已经自主选择 `binding_cost`，说明新增采集器可以通过现有 Agent 使用。
但首次规划把 intrusive discovery 的通信核时间约 284.7 ms 与干净候选测量直接比较，
而当时输入没有给出原始干净通信基线 94.49 ms。为此仅在 `evaluation.json` 增加
`original_reference`：原始 revision、逐 rank 的干净指标及来源解释；不附加原始大日志。

冻结同一生成前快照、同一 prompt 和模型，仅增加这项输入，在新真实会话中重放：
Agent 改为使用 94.49 → 71.35 ms 的干净测量，保留完整步 227.04 ms 超过 151.86 ms
约束的判断，选择相同候选 revision，仍请求绑定成本诊断；未再用 284.7 ms 宣称优化收益。
这是单次规划对照，证明该信息修正了本次基准混用，不代表优化成功或统计稳定性。
